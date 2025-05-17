use std::{
  collections::{HashMap, HashSet},
  fmt::{Debug, Display, format},
  hash::Hash,
  str::FromStr,
  sync::Arc,
  time::Duration,
};

/// 用于扩展生成的库，如(x * y) + 1 和 (m / n) + 1, 会生成(?a f ?b) + 1 f =
/// select(idx, (+, *))
use crate::{
  Arity, AstNode, DiscriminantEq, Expr, ParetoConfig, PartialExpr, Printable,
  Teachable,
  extract::beam_pareto::{ISAXAnalysis, TypeInfo},
  learn::{self, LearnedLibraryBuilder},
  runner::{AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost, OperationInfo},
  schedule::Schedulable,
};
use egg::{
  EGraph, ENodeOrVar, Id, Pattern, RecExpr, Rewrite, Runner, Searcher, Symbol,
  Var,
};
use lexpr::print;
use log::debug;
use nom::lib;

fn au2expr<Op>(pe: PartialExpr<Op, Var>, idx: usize) -> Expr<Op>
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
  match pe {
    PartialExpr::Node(astnode) => {
      let operation = astnode.operation().clone();
      let args = astnode.args();
      let mut new_args = Vec::with_capacity(args.len());
      for arg in args {
        let expr: Expr<Op> = au2expr(arg.clone(), idx);
        new_args.push(Arc::new(expr));
      }
      let node = AstNode::new(operation, new_args);
      node.into()
    }
    PartialExpr::Hole(var) => {
      // 首先，我们将PE转化为Expr只是为了转化成Recexpr进行delay计算，
      // 所以Hole可以不需要在意，将其作为一个叶节点处理就好，
      // 目前直接使用rulevar表示
      let s = format!("{}_{}", var.to_string(), idx).to_string();
      let node = AstNode::leaf(Op::make_rule_var(s));
      node.into()
    }
  }
}

/// 需要新加了mask节点，搜索meta_au的过程中需要自己手动写searcher的实现，
/// 目前仿照pattern的写法，只不过记录一下Opmask的特例
/// 实现一个searcher
/// 目前只允许使用一个Opmask节点，
struct MetaAUOpSearcher<Op> {
  pub searcher: Pattern<AstNode<Op>>,
  pub mask_results: HashSet<Id>,
  pub var_results: HashMap<Var, HashSet<Id>>,
}

impl<Op> MetaAUOpSearcher<Op>
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
    + Default
    + 'static,
{
  pub fn new(searcher: Pattern<AstNode<Op>>) -> Self {
    MetaAUOpSearcher {
      searcher,
      mask_results: HashSet::new(),
      var_results: HashMap::new(),
    }
  }

  pub fn search<T: Debug + Default + Clone + Ord + Hash>(
    &mut self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) where
    AstNode<Op>: TypeInfo<T>,
  {
    // 使用searcher进行搜索
    let searcher = self.searcher.clone();
    let search_results = searcher.search(egraph);
    let ast = searcher.ast.clone();
    // 记录有多少个Opmask节点，初始化results
    let mut opmask_count = 0;
    for node in ast.iter() {
      match node {
        ENodeOrVar::ENode(enode) => {
          if enode.operation().is_opmask() {
            opmask_count += 1;
          }
        }
        ENodeOrVar::Var(_) => {}
      }
    }
    if opmask_count > 1 {
      // 目前只支持表达式里有一个Opmask节点
      return;
    }
    // 如果表达式太短，直接返回
    if ast.len() < 2 {
      return;
    }
    for eclass in egraph.classes() {
      let id = eclass.id;
      let enode = ast.last().unwrap().clone();
      match enode {
        ENodeOrVar::ENode(enode) => {
          self.found_matched_eclasses(enode, id, egraph);
        }
        ENodeOrVar::Var(_) => {
          // 变量节点不需要处理
        }
      }
    }
    // 除此之外还需要拿到Var节点的Id
    let vars = self.searcher.vars();
    for var in vars {
      let var_id = var.clone();
      let mut var_set = HashSet::new();
      for result in &search_results {
        let substs = result.substs.clone();
        for subst in substs {
          if subst.get(var.clone()).is_some() {
            let var_id = subst.get(var.clone()).unwrap();
            var_set.insert(*var_id);
          }
        }
      }
      self.var_results.insert(var_id, var_set);
    }
  }

  fn found_matched_eclasses<T: Debug + Default + Clone + Ord + Hash>(
    &mut self,
    enode: AstNode<Op>,
    id: Id,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> bool
  where
    AstNode<Op>: TypeInfo<T>,
  {
    let mut found = false;
    for node in egraph[id].nodes.iter() {
      if node.operation() == enode.operation()
        || (enode.operation().is_opmask()
          && node.args().len() == enode.args().len())
      {
        // 相同之后可以继续深度匹配
        // 首先拿出ast中的下一个enode
        for i in 0..enode.args().len() {
          let ast_arg = enode.args()[i];
          match self.searcher.ast[ast_arg].clone() {
            ENodeOrVar::ENode(en) => {
              // 拿到node的子节点

              let egraph_arg = node.args()[i];
              // 递归调用
              if self.found_matched_eclasses(en.clone(), egraph_arg, egraph) {
                found = true;
              }
            }
            ENodeOrVar::Var(_) => {
              // 变量节点不需要处理
              found = true;
            }
          }
        }
      }
    }
    if found {
      // 如果找到匹配的节点，就将其加入mask_results中
      if enode.operation().is_opmask() {
        self.mask_results.insert(id);
      }
    }
    found
  }
}

// expand函数的输入是一个EGraph，之后在这个EGraph上进行库学习拿到AU，
// 然后使用AU变化成的rewrite进行搜索，将搜索到的子节点打包处理，
// 在EGraph中加入新的节点
pub fn expand<Op, T, LA, LD>(
  egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  config: ParetoConfig<LA, LD>,
) -> EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>
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
  println!(
    "before expand, eclass size: {}, egraph size: {}",
    egraph.classes().len(),
    egraph.total_size()
  );
  let mut new_egraph = egraph.clone();
  // 使用learned_library进行库学习，因为和vetorize的目标一致，
  // 所以直接套用vectorize的learn部分
  let expansion_lib_config = LiblearnConfig::new(
    LiblearnCost::Size,
    AUMergeMod::Greedy,
    EnumMode::PruningGold,
  );
  let learned_lib = LearnedLibraryBuilder::default()
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    // .with_co_occurs(co_occurs)
    .with_last_lib_id(0)
    .with_liblearn_config(expansion_lib_config)
    .with_clock_period(config.clock_period)
    .vectorize()
    .build(&egraph);
  println!("expand::learned {} libs", learned_lib.size());

  for lib in learned_lib.libs() {
    println!("lib: {}", lib);
  }

  let learned_aus: Vec<_> = learned_lib.anti_unifications().collect();
  let mut meta_egraph = EGraph::new(ISAXAnalysis::new(
    config.final_beams,
    config.inter_beams,
    config.lps,
    config.strategy,
  ));

  // 将au生成的Recexpr加入到meta_egraph中
  for i in 0..learned_aus.len() {
    let au = learned_aus[i].clone();
    // 在转化的过程中，需要考虑每一个Var如何转化，目前只是转化成了一个RuleVar，
    // 但是不同au中的RuleVar理应是不一样的
    let expr = au2expr(au.clone(), i);
    let recexpr = RecExpr::from(expr);
    meta_egraph.add_expr(&recexpr);
  }

  meta_egraph.rebuild();

  // meta_egraph.dot().to_png("target/expand.png").unwrap();

  let meta_au_lib_config =
    LiblearnConfig::new(LiblearnCost::Size, AUMergeMod::Greedy, EnumMode::All);

  // 进行meta_au-search
  let learn_meta_lib = LearnedLibraryBuilder::default()
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    .with_liblearn_config(meta_au_lib_config)
    .with_clock_period(config.clock_period)
    .meta_au_search()
    .vectorize()
    .build(&meta_egraph);

  let lib_rewrites: Vec<_> =
    learn_meta_lib.rewrites::<ISAXAnalysis<_, _>>().collect();

  println!("expand::learned {} meta libs", learn_meta_lib.size());
  for lib in learn_meta_lib.libs() {
    println!("meta lib: {}", lib);
  }

  for rewrite in lib_rewrites {
    let mut searcher = MetaAUOpSearcher::new(rewrite.1);
    // 使用searcher进行搜索
    searcher.search(&egraph);
    // 记录搜索到的结果
    let mask_results: HashSet<Id> = searcher.mask_results.clone();
    let var_results: HashMap<Var, HashSet<Id>> = searcher.var_results.clone();
    // 接下来需要将mask_results中的节点进行打包处理
    // 1. 合并普通变量结果和 mask 结果到一个 Vec 用 Option<Var> 表示：Some(var)
    //    是普通变量，None 表示 mask。
    let mut entries: Vec<(Option<Var>, HashSet<Id>)> = Vec::new();

    // 普通变量
    for (var, ids) in var_results {
      entries.push((Some(var.clone()), ids.clone()));
    }
    // mask 结果
    entries.push((None, mask_results.clone()));

    // 2. 对每个 entry 做 pack + select
    for (opt_var, ids) in entries {
      // 首先我要做一步检查，如果ids对应的节点全部都是args ==
      // 0的节点，那么完全没有必要pack 直接跳过
      let mut all_args_zero = true;
      for id in &ids {
        let nodes = &egraph[*id].nodes;
        for node in nodes {
          if node.args().len() > 0 {
            all_args_zero = false;
            break;
          }
        }
      }
      if all_args_zero {
        continue; // 如果ids对应的节点全部都是args == 0的节点，那么完全没有必要pack
      }
      if ids.is_empty() {
        continue; // 如果某个 var 或 mask 没匹配到任何 id，就跳过
      }

      // 收集所有 operation 名称，去重排序
      let mut op_vec: Vec<String> = ids
        .iter()
        .flat_map(|&eid| egraph[eid].nodes.iter())
        .map(|n| n.operation().to_string())
        .collect();
      op_vec.sort_unstable();
      op_vec.dedup();

      // 建立 op -> index 映射
      let op_idx: HashMap<_, _> = op_vec
        .iter()
        .enumerate()
        .map(|(i, op)| (op.clone(), i))
        .collect();

      // 插入 pack 节点
      let pack_args: Vec<Id> = ids.iter().cloned().collect();
      let pack_op = Op::make_op_pack(op_vec.clone());
      let pack_id = new_egraph.add(AstNode::new(pack_op, pack_args));

      // 插入 select 节点，并 union 回原 eclass
      for &eid in &ids {
        for node in &egraph[eid].nodes {
          let idx = *op_idx.get(&node.operation().to_string()).unwrap();
          let mut args = vec![pack_id];
          args.extend_from_slice(node.args());
          let select_node = AstNode::new(Op::make_op_select(idx), args);
          let new_id = new_egraph.add(select_node);
          new_egraph.union(eid, new_id);
        }
      }

      // 日志：打印当前处理的是哪个变量或 mask
      match &opt_var {
        Some(var) => debug!("处理变量 {:?} 的 pack/select", var),
        None => debug!("处理 MASK 的 pack/select"),
      }

      // 重建 egraph
      new_egraph.rebuild();
    }
  }
  // new_egraph.dot().to_png("target/expand_meta.png").unwrap();
  new_egraph
}
