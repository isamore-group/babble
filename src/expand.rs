use bitvec::vec;
use rand::prelude::*;
use serde::{Deserialize, de};
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
  Arity, AstNode, BindingExpr, DiscriminantEq, Expr, ParetoConfig, PartialExpr,
  Printable, Teachable,
  bb_query::{self, BBQuery},
  extract::beam_pareto::{ISAXAnalysis, TypeInfo},
  learn::{self, LearnedLibraryBuilder},
  rewrites::TypeMatch,
  runner::{AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost, OperationInfo},
  schedule::{Schedulable, Scheduler},
};
use egg::{
  EGraph, ENodeOrVar, Id, Pattern, RecExpr, Rewrite, Runner, Searcher, Symbol,
  Var, rewrite,
};
use lexpr::print;
use log::debug;
use nom::lib;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct OpPackConfig {
  /// 是否进行expand
  pub pack_expand: bool,
  /// 是否进行eclass pair剪枝
  pub prune_eclass_pair: bool,
  /// 是否学习所有可能的AU(要不要过滤掉一些trivial的AU)
  pub learn_trivial: bool,
  /// 保留多少个含有mask的Meta_AU
  pub num_meta_au_mask: usize,
  /// 至多允许一个pack里面出现多少个operation
  pub max_operation: usize,
}

impl Default for OpPackConfig {
  fn default() -> Self {
    OpPackConfig {
      pack_expand: false,
      prune_eclass_pair: true,
      learn_trivial: true,
      num_meta_au_mask: 100,
      max_operation: 5,
    }
  }
}

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
      let s = var.to_string(); //format!("{}_{}", var.to_string(), idx).to_string();
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
  pub mask_results: HashSet<Op>,
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
    let mut matched = false;
    for node in egraph[id].nodes.iter() {
      if node.args().len() != enode.args().len() {
        // 如果args的个数不一样，就直接跳过
        continue;
      }
      if node.operation() == enode.operation() || enode.operation().is_opmask()
      {
        // 相同之后可以继续深度匹配
        // 首先拿出ast中的下一个enode
        let mut found = true;
        for i in 0..enode.args().len() {
          let ast_arg = enode.args()[i];
          match self.searcher.ast[ast_arg].clone() {
            ENodeOrVar::ENode(en) => {
              // 拿到node的子节点
              let egraph_arg = node.args()[i];
              // 递归调用
              if !self.found_matched_eclasses(en.clone(), egraph_arg, egraph) {
                found = false;
                break;
              }
            }
            ENodeOrVar::Var(_) => {
              // 变量节点不需要处理
            }
          }
        }
        if found {
          matched = true;
          // 如果当前是Opmask节点，就将其加入到mask_results中
          if enode.operation().is_opmask() {
            // 这里需要将Opmask节点的操作符加入到mask_results中
            let op = node.operation().clone();
            self.mask_results.insert(op);
          }
          break;
        }
      }
    }
    matched
  }
}

fn add_op_pack<Op>(
  initial_pe: PartialExpr<Op, Var>,
  pack_op: Op,
) -> PartialExpr<Op, Var>
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
  // 一个递归函数，找到opmask节点，将其替换成op_select(oppack, args)即可
  match initial_pe {
    PartialExpr::Node(astnode) => {
      let mut operation = astnode.operation().clone();
      let args = astnode.args();
      let mut new_args = Vec::with_capacity(args.len());
      // 如果当前节点是opmask节点，就将其替换成op_select
      if operation.is_opmask() {
        let pack_pe: PartialExpr<Op, Var> =
          PartialExpr::Node(AstNode::leaf(pack_op.clone()));
        operation = Op::make_op_select();
        new_args.push(pack_pe);
      }
      for arg in args {
        let expr: PartialExpr<Op, Var> =
          add_op_pack(arg.clone(), pack_op.clone());
        new_args.push(expr);
      }
      let node = AstNode::new(operation, new_args);
      node.into()
    }
    PartialExpr::Hole(_) => {
      // 直接返回就行
      initial_pe
    }
  }
}

fn fill_specific_op<Op>(
  initial_pe: PartialExpr<Op, Var>,
  op: Op,
) -> PartialExpr<Op, Var>
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
  match initial_pe {
    PartialExpr::Node(astnode) => {
      let mut operation = astnode.operation().clone();
      let args = astnode.args();
      let mut new_args = Vec::with_capacity(args.len());
      if operation.is_opmask() {
        // 如果当前节点是opmask节点，就将其替换成具体的操作符
        operation = op.clone();
      }
      for arg in args {
        let expr: PartialExpr<Op, Var> =
          fill_specific_op(arg.clone(), op.clone());
        new_args.push(expr);
      }
      let node = AstNode::new(operation, new_args);
      node.into()
    }
    PartialExpr::Hole(_) => {
      // 直接返回就行
      initial_pe
    }
  }
}

#[derive(Debug, Clone)]
pub struct ExpandMessage<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + OperationInfo
    + Debug
    + 'static,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  pub all_au_rewrites: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  pub conditions: HashMap<usize, TypeMatch<T>>,
  pub libs: HashMap<usize, (PartialExpr<Op, Var>, Pattern<AstNode<Op>>)>,
  pub normal_au_count: usize,
  pub meta_au_rewrites:
    HashMap<usize, Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>>,
}

// expand函数的输入是一个EGraph，之后在这个EGraph上进行库学习拿到AU，
// 然后使用AU变化成的rewrite进行搜索，将搜索到的子节点打包处理，
// 在EGraph中加入新的节点
pub fn expand<Op, T, LA, LD>(
  egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  config: ParetoConfig<LA, LD>,
  bb_query: bb_query::BBQuery,
  max_lib_id: usize,
) -> ExpandMessage<Op, T>
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
  // 使用learned_library进行库学习，因为和vetorize的目标一致，
  // 所以直接套用vectorize的learn部分
  let expansion_lib_config = LiblearnConfig::new(
    LiblearnCost::Size,
    AUMergeMod::Greedy,
    EnumMode::PruningGold,
    // 后面的配置直接使用config.liblearn中的配置
    config.liblearn_config.sample_num,
    config.liblearn_config.hamming_threshold,
    config.liblearn_config.jaccard_threshold,
    config.liblearn_config.max_libs,
    config.liblearn_config.max_lib_size,
  );
  let learned_lib = LearnedLibraryBuilder::default()
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    .with_last_lib_id(max_lib_id)
    .with_liblearn_config(expansion_lib_config)
    .with_clock_period(config.clock_period)
    .with_area_estimator(config.area_estimator.clone())
    .with_delay_estimator(config.delay_estimator.clone())
    .with_bb_query(bb_query.clone())
    .build(&egraph);
  println!("expand::learned {} libs", learned_lib.size());

  // for lib in learned_lib.libs() {
  //   println!("lib: {}", lib);
  // }

  let learned_aus: Vec<_> = learned_lib.anti_unifications().collect();
  let mut meta_egraph = EGraph::new(ISAXAnalysis::new(
    config.final_beams,
    config.inter_beams,
    config.lps,
    config.strategy,
    BBQuery::default(),
  ));

  // // 将au生成的Recexpr加入到meta_egraph中
  let learned_messages: Vec<_> = learned_lib.messages();
  for i in (0..learned_aus.len()).rev() {
    // 这里从大到小检查，能将一些meta_egraph中已经存在的AU去掉
    let msg = learned_messages[i].clone();
    // 非常重要，添加之前先看这条规则能不能search到匹配，如果已经有匹配，
    // 就直接跳过
    let searcher: Pattern<_> = msg.searcher_pe.into();
    if searcher.search(&meta_egraph).len() > 0 {
      continue;
    }
    let au = learned_aus[i].clone();
    // 在转化的过程中，需要考虑每一个Var如何转化，
    // 目前只是转化成了一个RuleVar，   // 但是不同au中的RuleVar理应是不一样的
    let expr = au2expr(au.clone(), i);
    let recexpr = RecExpr::from(expr);
    meta_egraph.add_expr(&recexpr);

    meta_egraph.rebuild();
  }

  // meta_egraph.dot().to_png("target/expand.png").unwrap();
  println!(
    "For meta_egraph, eclass size: {}, egraph size: {}",
    meta_egraph.classes().len(),
    meta_egraph.total_size()
  );

  let meta_au_lib_config = LiblearnConfig::new(
    LiblearnCost::Size,
    AUMergeMod::Greedy,
    EnumMode::PruningGold,
    // 后面的配置直接使用config.liblearn中的配置
    config.liblearn_config.sample_num,
    config.liblearn_config.hamming_threshold,
    config.liblearn_config.jaccard_threshold,
    config.liblearn_config.max_libs,
    config.liblearn_config.max_lib_size,
  );

  // 计算一个新的max_lib_id
  let max_lib_id = max_lib_id + learned_lib.size();

  // 进行meta_au-search
  let learn_meta_lib = LearnedLibraryBuilder::default()
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    .with_liblearn_config(meta_au_lib_config)
    .with_clock_period(config.clock_period)
    .with_op_pack_config(config.op_pack_config)
    .with_last_lib_id(max_lib_id)
    .with_area_estimator(config.area_estimator.clone())
    .with_delay_estimator(config.delay_estimator.clone())
    .with_bb_query(bb_query.clone())
    .build(&meta_egraph);

  let meta_messages: Vec<_> = learn_meta_lib.messages();

  println!("expand::learned {} meta libs", learn_meta_lib.size());
  // for lib in learn_meta_lib.libs() {
  //   println!("meta lib: {}", lib);
  // }

  let mut all_au_rewrites = Vec::new();
  let mut conditions = HashMap::new();
  let mut libs: HashMap<usize, (PartialExpr<Op, Var>, Pattern<_>)> =
    HashMap::new();
  let mut meta_au_rewrites = HashMap::new();

  for msg in learned_messages {
    all_au_rewrites.push(msg.rewrite.clone());
    conditions.insert(msg.lib_id.clone(), msg.condition.clone());
    libs.insert(
      msg.lib_id.clone(),
      (msg.searcher_pe.clone(), msg.applier_pe.clone().into()),
    );
  }

  for msg in meta_messages {
    let mut rewrites = Vec::new();
    let mut searcher = MetaAUOpSearcher::new(msg.searcher_pe.clone().into());
    // 使用searcher进行搜索
    searcher.search(&egraph);
    // 记录搜索到的结果
    let mask_results: HashSet<Op> = searcher.mask_results.clone();
    // TODO: 目前只对mask_results进行处理，后续可能需要对var_results进行处理
    // 第一步，拿到OpPack具体需要Pack的操作符
    if mask_results.len() == 0 {
      continue; // 如果没有mask结果，就跳过
    }
    if mask_results.len() > config.op_pack_config.max_operation {
      continue; // 如果操作符的个数超过了限制，就直接跳过
    }

    // 否则新建一个OpPack节点和一个OpSelect节点
    let op_pack =
      Op::make_op_pack(mask_results.iter().map(|x| x.to_string()).collect());
    // 将新的partial_expr转化成applier
    let mut new_applier: Pattern<_> =
      add_op_pack(msg.applier_pe.clone(), op_pack.clone()).into();
    // Calculate the gain and cost of the new applier
    let ast = &new_applier.ast;
    let new_expr = ast
      .iter()
      .map(|node| match node {
        egg::ENodeOrVar::ENode(ast_node) => ast_node.clone(),
        egg::ENodeOrVar::Var(_) => Op::var(0),
      })
      .collect::<Vec<AstNode<Op>>>();
    let rec_expr: RecExpr<AstNode<Op>> = new_expr.into();
    let scheduler = Scheduler::new(
      config.clock_period,
      config.area_estimator.clone(),
      config.delay_estimator.clone(),
      bb_query.clone(),
    );
    let (latency_gain, area) = scheduler.asap_schedule(&rec_expr);
    for node in new_applier.ast.iter_mut() {
      match node {
        egg::ENodeOrVar::ENode(ast_node) => {
          if let Some(BindingExpr::Lib(id, _, _, _, _)) =
            ast_node.as_binding_expr()
          {
            let op = ast_node.operation_mut();
            *op = Op::make_lib(id.into(), latency_gain, area);
          }
        }
        egg::ENodeOrVar::Var(_) => {}
      };
    }
    for (id, op) in mask_results.iter().enumerate() {
      let searcher: Pattern<_> =
        fill_specific_op(msg.searcher_pe.clone(), op.clone()).into();
      let rewrite: Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>> = Rewrite::new(
        format!("{}_{}", msg.rewrite.name, id),
        searcher,
        new_applier.clone(),
      )
      .unwrap_or_else(|_| unreachable!());
      rewrites.push(rewrite.clone());
      all_au_rewrites.push(rewrite.clone());
    }
    conditions.insert(msg.lib_id.clone(), msg.condition.clone());
    libs.insert(
      msg.lib_id.clone(),
      (msg.searcher_pe.clone(), msg.applier_pe.clone().into()),
    );
    meta_au_rewrites.insert(msg.lib_id.clone(), rewrites);
  }
  ExpandMessage {
    all_au_rewrites,
    conditions,
    libs,
    normal_au_count: max_lib_id + 1,
    meta_au_rewrites,
  }
  //   let var_results: HashMap<Var, HashSet<Id>> =
  // searcher.var_results.clone();   //
  // 接下来需要将mask_results中的节点进行打包处理   // 1. 合并普通变量结果和
  // mask 结果到一个 Vec 用 Option<Var> 表示：Some(var)   //
  // 是普通变量，None 表示 mask。   let mut entries: Vec<(Option<Var>,
  // HashSet<Id>)> = Vec::new();   let mut set_ids = HashSet::new();
  //   // 普通变量
  //   for (var, ids) in var_results {
  //     if ids.len() > config.op_pack_config.max_eclass_id {
  //       // 如果ids的个数超过了限制，就直接跳过
  //       continue;
  //     }
  //     let mut vec_ids = ids.clone().into_iter().collect::<Vec<_>>();
  //     vec_ids.sort_unstable();
  //     if set_ids.contains(&vec_ids) {
  //       continue; // 如果ids已经存在，就跳过
  //     }
  //     entries.push((Some(var.clone()), ids.clone()));
  //     set_ids.insert(vec_ids);
  //   }
  //   // mask 结果
  //   entries.push((None, mask_results.clone()));

  //   // 2. 对每个 entry 做 pack + select
  //   for (opt_var, ids) in entries {
  //     // 首先我要做一步检查，如果ids对应的节点全部都是args ==
  //     // 0的节点，那么完全没有必要pack 直接跳过
  //     let mut all_args_zero = true;
  //     for id in &ids {
  //       let nodes = &egraph[*id].nodes;
  //       for node in nodes {
  //         if node.args().len() > 0 {
  //           all_args_zero = false;
  //           break;
  //         }
  //       }
  //     }
  //     if all_args_zero {
  //       continue; // 如果ids对应的节点全部都是args ==
  // 0的节点，那么完全没有必要pack     }
  //     if ids.is_empty() {
  //       continue; // 如果某个 var 或 mask 没匹配到任何 id，就跳过
  //     }

  //     // 收集所有 operation 名称，去重排序
  //     let mut op_vec: Vec<String> = ids
  //       .iter()
  //       .flat_map(|&eid| egraph[eid].nodes.iter())
  //       .map(|n| n.operation().to_string())
  //       .collect();
  //     // 如果op_vec中操作符的个数超过5个，就直接跳过
  //     if op_vec.len() > config.op_pack_config.max_operation {
  //       continue;
  //     }
  //     op_vec.sort_unstable();
  //     op_vec.dedup();

  //     // 建立 op -> index 映射
  //     let op_idx: HashMap<_, _> = op_vec
  //       .iter()
  //       .enumerate()
  //       .map(|(i, op)| (op.clone(), i))
  //       .collect();

  //     // 插入 pack 节点
  //     let pack_args: Vec<Id> = ids.iter().cloned().collect();
  //     let pack_op = Op::make_op_pack(op_vec.clone());
  //     let pack_id = new_egraph.add(AstNode::new(pack_op, pack_args));

  //     // 插入 select 节点，并 union 回原 eclass
  //     for &eid in &ids {
  //       for node in &egraph[eid].nodes {
  //         let idx = *op_idx.get(&node.operation().to_string()).unwrap();
  //         let mut args = vec![pack_id];
  //         args.extend_from_slice(node.args());
  //         let select_node = AstNode::new(Op::make_op_select(idx), args);
  //         let new_id = new_egraph.add(select_node);
  //         new_egraph.union(eid, new_id);
  //       }
  //     }

  //     // 日志：打印当前处理的是哪个变量或 mask
  //     match &opt_var {
  //       Some(var) => debug!("处理变量 {:?} 的 pack/select", var),
  //       None => debug!("处理 MASK 的 pack/select"),
  //     }

  //     // 重建 egraph
  //     new_egraph.rebuild();
  //   }
  // }
  // // new_egraph.dot().to_png("target/expand_meta.png").unwrap();
}
