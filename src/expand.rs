use std::{
  collections::{HashMap, HashSet},
  fmt::{Debug, Display},
  hash::Hash,
  str::FromStr,
};

/// 用于扩展生成的库，如(x * y) + 1 和 (m / n) + 1, 会生成(?a f ?b) + 1 f =
/// select(idx, (+, *))
use crate::{
  Arity, AstNode, DiscriminantEq, ParetoConfig, Printable, Teachable,
  extract::beam_pareto::{ISAXAnalysis, TypeInfo},
  learn::LearnedLibraryBuilder,
  runner::{AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost, OperationInfo},
  schedule::Schedulable,
};
use egg::EGraph;
use log::debug;

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

  // 提取出获得的lib_rewrite
  let lib_rewrites = learned_lib.rewrites();
  // 每个rewrite进行搜索，找到子节点包
  for rewrite in lib_rewrites {
    debug!("rewrite: {:?}", rewrite);
    let vars = rewrite.searcher.vars();
    let results = rewrite.search(&egraph);
    let mut var_ids = vars
      .iter()
      .map(|var| (var.clone(), HashSet::new()))
      .collect::<HashMap<_, _>>();
    for result in results {
      let substs = result.substs;
      for subst in substs {
        for var in vars.clone() {
          if subst.get(var).is_some() {
            let var_id = subst.get(var).unwrap();
            let var_set = var_ids.get_mut(&var).unwrap();
            var_set.insert(*var_id);
          }
        }
      }
    }

    for (var, var_set) in var_ids.iter() {
      // 首先收集所有eclass中可能的操作符
      let mut op_set = HashSet::new();
      for var_id in var_set.iter() {
        let eclass = egraph[*var_id].nodes.clone();
        for node in eclass.iter() {
          op_set.insert(node.operation().to_string());
        }
      }

      let op_vec: Vec<String> = op_set.iter().map(|s| s.clone()).collect();
      let mut op_idx_map: HashMap<String, usize> = HashMap::new();
      for (idx, op) in op_vec.iter().enumerate() {
        op_idx_map.insert(op.clone(), idx);
      }
      // 加入ALU节点
      let alu_id =
        new_egraph.add(AstNode::new(Op::make_op_pack(op_vec), var_set.clone()));

      // 对于每一个enode，加入新的op_select节点
      for var_id in var_set.iter() {
        let eclass = egraph[*var_id].nodes.clone();
        for node in eclass.iter() {
          let op = node.operation().to_string();
          let op_idx = op_idx_map.get(&op).unwrap();
          let mut args = vec![alu_id];
          args.extend_from_slice(node.args());
          let new_node = AstNode::new(Op::make_op_select(*op_idx), args);
          let new_id = new_egraph.add(new_node);
          // 合并两个eclass
          new_egraph.union(*var_id, new_id);
        }
      }

      // rebuild
      new_egraph.rebuild();
    }
  }
  println!(
    "after expand, eclass size: {}, egraph size: {}",
    new_egraph.classes().len(),
    new_egraph.total_size()
  );
  new_egraph
}
