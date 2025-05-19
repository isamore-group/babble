use crate::{
  DiscriminantEq, Printable,
  analysis::SimpleAnalysis,
  ast_node::{Arity, AstNode},
  bb_query::BBInfo,
  extract::beam_pareto::{ClassMatch, ISAXAnalysis, TypeInfo},
  learn::LearnedLibraryBuilder,
  rewrites::TypeMatch,
  runner::{
    AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost, OperationInfo,
    ParetoConfig,
  },
  schedule::Schedulable,
  teachable::{ShieldingOp, Teachable},
};
use bitvec::vec;
use egg::{
  CostFunction, EGraph, Id, Language, Pattern, RecExpr, Rewrite, Runner,
  Searcher, Subst, Var, rewrite,
};
use lexpr::print;
use serde::Deserialize;
use std::{
  collections::{BTreeMap, HashMap, HashSet, VecDeque},
  fmt::{Debug, Display},
  hash::{Hash, Hasher},
  ops::Deref,
  path::PathBuf,
  str::FromStr,
  sync::{Arc, mpsc},
  time::Duration,
};

/// 用于进行向量化的config
#[derive(Debug, Clone, Deserialize)]
pub struct VectorConfig {
  /// 是否进行向量化
  pub vectorize: bool,
  /// 是否启用gather节点(gather节点搜索数目较少， 对整体速度的影响不是很大，
  /// 默认为true)
  pub enable_gather: bool,
  /// 是否启用shuffle节点(shuffle节点搜索数目很大， 对整体速度的影响较大，
  /// 默认为false， 但是加入shuffle节点之后， 向量化的性能会很高)
  pub enable_shuffle: bool,
  /// 是否启用post-check(在预处理阶段，
  /// 对于store和load节点的深嵌套进行了解耦，但是可能导致错误的结果)
  pub enable_post_check: bool,
  /// 最大允许的向量化组大小、
  pub max_vec_len: usize,
  /// lift-rules的文件名
  pub lift_rules: Option<String>,
  /// transfrom-rules的文件名
  pub transform_rules: Option<String>,
}

impl Default for VectorConfig {
  fn default() -> Self {
    Self {
      vectorize: false,
      enable_gather: true,
      enable_shuffle: false,
      enable_post_check: false,
      max_vec_len: 8,
      lift_rules: None,
      transform_rules: None,
    }
  }
}

impl VectorConfig {
  /// Create a new VectorConfig with the given parameters
  pub fn new(
    vectorize: bool,
    enable_gather: bool,
    enable_shuffle: bool,
    enable_post_check: bool,
    max_vec_len: usize,
    lift_rules: Option<String>,
    transform_rules: Option<String>,
  ) -> Self {
    Self {
      vectorize,
      enable_gather,
      enable_shuffle,
      enable_post_check,
      max_vec_len,
      lift_rules,
      transform_rules,
    }
  }
}

// 核心数据结构
struct VecGroupFinder<Op, T> {
  /// 记录每个eclass的向量化组ID (使用并查集)
  union_find: HashMap<Id, Id>,
  /// 缓存每个eclass的结构签名
  signature_cache: HashMap<Id, u64>,
  sig_to_group: HashMap<u64, Id>,
  /// 存储最终发现的向量化组
  groups: HashMap<Id, Vec<Id>>,
  phantom_op: std::marker::PhantomData<Op>,
  phantom_ty: std::marker::PhantomData<T>,
}

impl<Op, T> VecGroupFinder<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + Debug
    + OperationInfo
    + BBInfo
    + 'static
    + Default,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr,
  AstNode<Op>: TypeInfo<T>,
{
  pub fn new(initial_pairs: &[(Id, Id)]) -> Self {
    let mut union_find = HashMap::new();

    // 初始化并查集
    for &(a, b) in initial_pairs {
      union_find.entry(a).or_insert(a);
      union_find.entry(b).or_insert(b);
      Self::union(&mut union_find, a, b);
    }

    Self {
      union_find,
      signature_cache: HashMap::new(),
      sig_to_group: HashMap::new(),
      groups: HashMap::new(),
      phantom_op: std::marker::PhantomData,
      phantom_ty: std::marker::PhantomData,
    }
  }

  /// 主入口：递归发现向量化组
  pub fn find_groups(
    &mut self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
    roots: &[Id],
  ) -> Vec<Vec<Id>> {
    // 初始化并查集和签名缓存
    self.union_find.clear();
    self.signature_cache.clear();

    // 初始化并查集为每个eclass的ID指向自己（自环）
    for class in egraph.classes() {
      self.union_find.insert(class.id, class.id);
    }
    {
      // 自顶向下递归处理所有根节点
      for (id, &root) in roots.iter().enumerate() {
        self.process_class(root, egraph);
      }

      // 生成最终分组
      self.build_final_groups(egraph)
    }
  }

  /// 递归处理单个eclass
  fn process_class(
    &mut self,
    class_id: Id,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> Id {
    let class_id = egraph.find(class_id);

    // 如果已经处理过，直接返回组代表
    // if let Some(_sig) = self.signature_cache.get(&class_id) {
    //   return self.union_find[&class_id];
    // }
    // println!("Processing class {}", class_id);
    // 获取所有可能的节点结构
    let mut signatures = Vec::new();
    for node in &egraph[class_id].nodes {
      // println!("Processing node {:?}", node);
      // 递归处理子节点并获取其组代表
      let child_groups: Vec<u64> = node
        .children()
        .iter()
        .map(|&child_id| {
          let child_class = egraph.find(child_id);
          self.process_class(child_class, egraph);
          // 获取子节点的签名哈希值
          self
            .signature_cache
            .get(&child_class)
            .cloned()
            .unwrap_or_else(|| {
              panic!("No signature found for child class {}", child_class)
            })
        })
        .collect();
      signatures.push(
        // 计算当前节点的签名哈希值
        self.sig_hash(
          node.operation().to_shielding_op(),
          node.operation().get_bbs_info(),
          &child_groups,
        ),
      );
    }
    // 计算当前类的代表签名（取最小哈希）
    let min_sig =
      signatures.iter().cloned().min().unwrap_or_else(|| {
        panic!("No signatures found for class {}", class_id)
      });
    // println!("Class {} has signature hash {}", class_id, min_sig);
    // 缓存签名并尝试合并组
    self.signature_cache.insert(class_id, min_sig.clone());

    // 查找可合并的候选组
    let group_rep = match self.sig_to_group.get(&min_sig) {
      Some(&existing) => Self::union(&mut self.union_find, class_id, existing),
      None => {
        self.sig_to_group.insert(min_sig.clone(), class_id);
        class_id
      }
    };
    group_rep
  }

  /// 构建最终分组结果
  fn build_final_groups(
    &self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> Vec<Vec<Id>> {
    let mut temp_groups: HashMap<Id, Vec<Id>> = HashMap::new();

    // 收集所有eclass到其代表组
    for class in egraph.classes() {
      let rep = self.union_find[&class.id];
      temp_groups.entry(rep).or_default().push(class.id);
    }
    // 过滤和整理
    temp_groups
      .into_iter()
      .filter(|(_, v)| v.len() > 1)
      .map(|(_, mut group)| {
        group.sort_unstable();
        group.dedup();
        group
      })
      .collect()
  }

  /// 并查集辅助方法
  fn find(uf: &mut HashMap<Id, Id>, mut x: Id) -> Id {
    if uf[&x] != x {
      // First find the root and then store it
      let root = Self::find(uf, uf[&x]);
      // Now insert the root into the HashMap
      uf.insert(x, root);
    }
    uf[&x]
  }

  fn union(uf: &mut HashMap<Id, Id>, x: Id, y: Id) -> Id {
    let x_root = Self::find(uf, x);
    let y_root = Self::find(uf, y);

    if x_root != y_root {
      uf.insert(x_root, y_root);
    }
    y_root
  }

  /// 签名哈希函数
  fn sig_hash(
    &self,
    op: ShieldingOp,
    bbs: Vec<String>,
    child_hashes: &[u64],
  ) -> u64 {
    // 对child_hashes进行排序
    let mut sorted_hashes = child_hashes.to_vec();
    sorted_hashes.sort_unstable();
    // 对bbs进行排序
    let mut sorted_bbs = bbs.clone();
    sorted_bbs.sort_unstable();
    // 计算哈希值
    let mut hasher = seahash::SeaHasher::default();
    op.hash(&mut hasher);
    for &hash in &sorted_hashes {
      hasher.write_u64(hash);
    }
    for bb in &sorted_bbs {
      hasher.write(bb.as_bytes());
    }
    hasher.finish()
  }
}

pub struct VectorCF;

impl<Op> CostFunction<AstNode<Op>> for VectorCF
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + Debug
    + OperationInfo,
{
  type Cost = f64;

  fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
  where
    C: FnMut(Id) -> Self::Cost,
  {
    let op = enode.operation();
    let mut base_cost = op.get_simple_cost();

    // ——计算“原向量奖励”——
    let vector_bonus = if op.is_vector_op() {
      if op.is_vec() {
        let vec_len = enode.args().len();
        10.0 + (vec_len * vec_len) as f64
      } else {
        let vec_len = op.get_vec_len();
        base_cost = base_cost / vec_len as f64;
        base_cost
      }
    } else {
      0.0
    };

    // 子节点成本
    let children_cost: f64 = enode.fold(0.01, |sum, id| sum + costs(id));
    // println!("cost: {}", children_cost);
    // 最终成本：基准 + 子节点 + 惩罚
    (base_cost + children_cost - vector_bonus).max(children_cost)
  }
}

pub fn find_vec_containments_and_gathers<Op, T>(
  egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
) -> (HashMap<Id, HashSet<Id>>, HashMap<(Id, Id), Vec<usize>>)
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  // 收集所有 vec 元素列表，按长度分组
  let mut len_groups: BTreeMap<usize, Vec<(Id, Vec<Id>, HashSet<Id>)>> =
    BTreeMap::new();

  for eclass in egraph.classes() {
    if let Some(enode) = eclass.nodes.iter().find(|n| n.operation().is_vec()) {
      let elements_vec: Vec<Id> = enode.args().to_vec();
      let elements_set: HashSet<Id> = elements_vec.iter().cloned().collect();
      let len = elements_vec.len();
      len_groups.entry(len).or_default().push((
        eclass.id,
        elements_vec,
        elements_set,
      ));
    }
  }

  let mut direct_map: HashMap<Id, HashSet<Id>> = HashMap::new();
  let mut gather_map: HashMap<(Id, Id), Vec<usize>> = HashMap::new();

  for (current_len, current_vecs) in &len_groups {
    for (_higher_len, higher_vecs) in len_groups.range(current_len + 1..) {
      for (a_id, a_vec, a_set) in current_vecs {
        for (b_id, b_vec, b_set) in higher_vecs {
          if a_set.is_subset(b_set) {
            // 更新包含关系
            direct_map.entry(*a_id).or_default().insert(*b_id);

            // 计算 gather 索引
            let mut indices = Vec::new();
            for a_elem in a_vec {
              if let Some(pos) =
                b_vec.iter().position(|b_elem| b_elem == a_elem)
              {
                indices.push(pos);
              } else {
                panic!("Element from A not found in B, inconsistent state");
              }
            }
            gather_map.insert((*a_id, *b_id), indices);
          }
        }
      }
    }
  }

  (direct_map, gather_map)
}

pub fn find_vec_shuffles_and_indices<Op, T>(
  egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
) -> (
  HashMap<Id, HashSet<(Id, Id)>>, /* For each C, which (A,B) pairs can
                                   * shuffle into C */
  HashMap<(Id, Id, Id), Vec<usize>>, /* For each (C, A, B), the index‐vector
                                      * to gather from [A; B] to C */
)
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  // 1) 收集所有 vec 表达式，按长度分组
  let mut len_groups: BTreeMap<usize, Vec<(Id, Vec<Id>, HashSet<Id>)>> =
    BTreeMap::new();

  for eclass in egraph.classes() {
    if let Some(enode) = eclass.nodes.iter().find(|n| n.operation().is_vec()) {
      let elems: Vec<Id> = enode.args().to_vec();
      let set: HashSet<Id> = elems.iter().cloned().collect();
      len_groups
        .entry(elems.len())
        .or_default()
        .push((eclass.id, elems, set));
    }
  }

  // 2) 准备输出容器
  let mut shuffle_pairs: HashMap<Id, HashSet<(Id, Id)>> = HashMap::new();
  let mut shuffle_map: HashMap<(Id, Id, Id), Vec<usize>> = HashMap::new();

  // 3) 对每一对 (A, B) 组合尝试所有可能的 C 假设 m + n = L，即 A.len() +
  //    B.len() == C.len()
  let lengths: Vec<usize> = len_groups.keys().cloned().collect();
  for &len_a in &lengths {
    for &len_b in &lengths {
      let total_len = len_a + len_b;
      // 对每个具体的 A、B vec
      for (a_id, a_vec, a_set) in &len_groups[&len_a] {
        for (b_id, b_vec, b_set) in &len_groups[&len_b] {
          if a_id > b_id {
            continue; // 避免重复处理 (A, B) 和 (B, A)
          }
          // 预先构造 [A; B] 的向量和并集
          let mut ab_vec = Vec::with_capacity(total_len);
          ab_vec.extend(a_vec.iter().cloned());
          ab_vec.extend(b_vec.iter().cloned());
          let union_set: HashSet<Id> = a_set.union(b_set).cloned().collect();

          // 只考虑 C 长度为 total_len 的候选
          for (&len_c, c_group) in &len_groups {
            if len_c != total_len {
              continue; // Skip if C's length doesn't match A+B length
            }
            for (c_id, c_vec, c_set) in c_group {
              // 若 C 的元素集合是 A∪B 的子集，则可能是 shuffle
              if !c_set.is_subset(&union_set) {
                continue;
              }
              // 计算从 [A; B] gather 出 C 所需的索引
              let mut indices = Vec::with_capacity(c_vec.len());
              for &elem in c_vec {
                if let Some(pos) = ab_vec.iter().position(|&x| x == elem) {
                  indices.push(pos);
                } else {
                  // 逻辑上不应发生：C_set ⊆ union_set 已保证
                  panic!(
                    "Internal error: element {:?} of C={:?} \
                                       not found in concatenated A={:?},B={:?}",
                    elem, c_id, a_id, b_id
                  );
                }
              }
              // 记录这一对 (A, B) 能生成 C
              shuffle_pairs
                .entry(*c_id)
                .or_default()
                .insert((*a_id, *b_id));
              shuffle_map.insert((*c_id, *a_id, *b_id), indices);
            }
          }
        }
      }
    }
  }

  (shuffle_pairs, shuffle_map)
}

/// 目前向量化直接使用liblearn中的au-search进行向量化
pub fn vectorize<Op, T, LA, LD>(
  egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  roots: &[Id],
  lift_dsrs: &Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  transfrom_dsrs: &Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  config: ParetoConfig<LA, LD>,
) -> EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  LA: Debug + Clone + Default,
  LD: Debug + Clone + Default,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  println!("there are {} lift dsrs", lift_dsrs.len());
  println!("there are {} transfrom dsrs", transfrom_dsrs.len());
  let timeout = Duration::from_secs(60 * 100_000);
  let mut egraph = egraph.clone();
  println!(
    "before vectorize, eclass size: {}, egraph size: {}",
    egraph.classes().len(),
    egraph.total_size()
  );
  let vetorize_lib_config = LiblearnConfig::new(
    LiblearnCost::Size,
    AUMergeMod::Greedy,
    EnumMode::PruningGold,
  );
  // let mut vec_finder = VecGroupFinder::<Op, T>::new(&[]);

  // let vec_groups = vec_finder.find_groups(&egraph, roots);
  // println!("vec_groups size: {}", vec_groups.len());
  // for group in vec_groups {
  //   println!("vec size: {}", group.len());
  //   // 向量包为空
  //   if group.is_empty() {
  //     continue;
  //   }
  //   // 向量包超过8时，机器无法支持这种过高的并行度，不会进行深度的向量化
  //   if group.len() > 8 {
  //     continue;
  //   }
  //   println!("vec_node: {:?}", egraph[group[0]].nodes[0]);
  //   let mut tys = Vec::new();
  //   let mut vec_ids = Vec::new();
  //   for id in group.iter() {
  //     let eclass = egraph.find(*id);
  //     let ty = egraph[eclass].data.get_type()[0].clone();
  //     tys.push(ty.clone());
  //     vec_ids.push(eclass);
  //   }
  //   // 首先加入Vec节点
  //   let vec_node = AstNode::new(Op::make_vec(tys.clone()), vec_ids.clone());
  //   let list_id = egraph.add(vec_node);
  //   // 针对每个id，构造一个get节点
  //   for i in 0..vec_ids.len() {
  //     let get_node =
  //       AstNode::new(Op::make_get(i, vec![tys[i].clone()]), vec![list_id]);
  //     let get_id = egraph.add(get_node);
  //     // 进行替换
  //     egraph.union(vec_ids[i], get_id);
  //   }
  //   // 重建egraph
  //   egraph.rebuild();
  // }

  // // 进行向量化组发现

  // 进行库学习
  let learned_lib = LearnedLibraryBuilder::default()
    .learn_trivial(true)
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    // .with_co_occurs(co_occurs)
    .with_last_lib_id(0)
    .with_liblearn_config(vetorize_lib_config)
    .with_clock_period(config.clock_period)
    .build(&egraph);
  let lib_rewrites: Vec<(
    Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>,
    Pattern<_>,
  )> = learned_lib.rewrites().collect::<Vec<_>>();
  println!("learned {} libs", lib_rewrites.len());
  let mut id_set = HashSet::new();
  for (id, rewrite) in lib_rewrites.clone().into_iter().enumerate() {
    // let rewrite = rewrite.0.clone();
    let (tx, rx) = mpsc::channel();
    let egraph_clone = egraph.clone();
    let searcher = rewrite.1.clone();
    std::thread::spawn(move || {
      let results: Vec<egg::SearchMatches<'_, AstNode<Op>>> =
        searcher.search(&egraph_clone);
      let classes = results.iter().map(|x| x.eclass).collect::<Vec<_>>();
      let _ = tx.send(classes);
    });

    match rx.recv_timeout(Duration::from_secs(5)) {
      Ok(results) => {
        // 向量包超过8时，机器无法支持这种过高的并行度，不会进行深度的向量化
        if results.len() > config.vectorize_config.max_vec_len
          || results.len() < 2
        {
          continue;
        }
        // println!("results.len(): {}", results.len());
        // 检查当前的包是不是已经存在
        let mut id_pack = results.clone().into_iter().collect::<Vec<_>>();
        id_pack.sort();
        id_pack.dedup();
        if id_set.contains(&id_pack) {
          continue;
        }
        id_set.insert(id_pack);
        // println!("found {} matches for {:?}", results.len(), rewrite);
        let mut tys = Vec::new();
        let matched_eclass_id = results
          .iter()
          .map(|x| {
            let eclass = x.clone();
            let ty = egraph[eclass].data.get_type()[0].clone();
            tys.push(ty.clone());
            eclass
          })
          .collect::<Vec<_>>();
        // 首先加入Vec节点
        let vec_node =
          AstNode::new(Op::make_vec(tys.clone()), matched_eclass_id.clone());
        let list_id = egraph.add(vec_node);
        // 针对每个id，构造一个get节点
        for i in 0..matched_eclass_id.len() {
          let get_node =
            AstNode::new(Op::make_get(i, vec![tys[i].clone()]), vec![list_id]);
          // println!("tys[i]: {:?}", tys[i]);
          let get_id = egraph.add(get_node);
          // println!("get_eclass: {:?}", egraph[get_id]);
          // 进行替换
          egraph.union(matched_eclass_id[i], get_id);
          // println!("success union");
        }
        // 重建egraph
        egraph.rebuild();
      }
      Err(_) => {
        eprintln!("rewrite {} timed out, skip", id);
        // 直接继续下一个，不用等子线程
        continue;
      }
    }
  }
  // egraph.dot().to_png("target/foo2.png").unwrap();
  println!(
    "after add list, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );

  //使用dsrs进行重写
  let runner = Runner::<_, _, ()>::new(ISAXAnalysis::empty())
    .with_egraph(egraph)
    .with_time_limit(timeout)
    .with_iter_limit(10)
    .run(lift_dsrs);

  // 目前已经实现了各类vec的构建，现在需要去寻找egraph中含有的vec
  // enode，然后加入gather节点
  let mut egraph = runner.egraph.clone();
  if config.vectorize_config.enable_gather {
    let (vec_containments, vec_gathers) =
      find_vec_containments_and_gathers(&egraph);
    println!("size of vec_containments: {}", vec_containments.len());
    // 根据包含关系加入gather节点
    for (i, (id, fathers)) in vec_containments.into_iter().enumerate() {
      // println!("i:{}, id: {}, fathers: {}", i, id, fathers.len());
      for father in fathers {
        let gather_op = Op::make_gather(&vec_gathers[&(id, father)]);
        let cani_father = egraph.find(father);
        let gather_node = AstNode::new(gather_op.clone(), vec![cani_father]);
        let gather_id = egraph.add(gather_node);
        // 进行替换
        let cani_id = egraph.find(id);
        egraph.union(cani_id, gather_id);
        egraph.rebuild();
      }
    }
  }
  if config.vectorize_config.enable_shuffle {
    // 除此之外，还需要加入shuffle节点
    let (shuffle_pairs, shuffle_map) = find_vec_shuffles_and_indices(&egraph);
    println!("size of shuffle_pairs: {}", shuffle_pairs.len());
    // egraph.dot().to_png("target/foo1.png").unwrap();
    // 根据 shuffle 关系加入 shuffle 节点
    for (c_id, pairs) in shuffle_pairs.into_iter() {
      // println!("c_id: {}, pairs: {}", c_id, pairs.len());
      for (a_id, b_id) in pairs {
        // 从 shuffle_map 中获取 (C, A, B) 对应的 gather 索引
        let indices = &shuffle_map[&(c_id, a_id, b_id)];

        // 查找 A 和 B 的父节点
        let a_parent = egraph.find(a_id);
        let b_parent = egraph.find(b_id);

        // 创建 shuffle 操作节点
        let shuffle_op = Op::make_shuffle(indices); // 这里假设 `make_shuffle`
        // 是定义的函数
        let shuffle_node =
          AstNode::new(shuffle_op.clone(), vec![a_parent, b_parent]);

        // 添加 shuffle 节点到 egraph
        let shuffle_id = egraph.add(shuffle_node);

        // 进行替换
        let cani_id = egraph.find(c_id);
        egraph.union(cani_id, shuffle_id);

        // 重新构建 egraph
        egraph.rebuild();
      }
    }
  }

  // egraph.dot().to_png("target/foo.png").unwrap();
  println!(
    "after vectorization, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );

  // 使用transform_dsrs进行重写
  // let runner = Runner::<_, _, ()>::new(ISAXAnalysis::empty())
  //   .with_egraph(egraph)
  //   .with_time_limit(timeout)
  //   .with_iter_limit(10)
  //   .run(transfrom_dsrs);
  // egraph = runner.egraph.clone();
  // 恢复类型信息，将每个eclass的类型信息传给里面的enode

  let cloned_egraph = egraph.clone();
  for class in egraph.classes_mut() {
    for node in class.nodes.iter_mut() {
      let tys = cloned_egraph[class.id].data.get_type();
      node.operation_mut().set_result_type(tys.clone());
    }
  }

  egraph
}
