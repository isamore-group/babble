use egg::{EGraph, Id, Language};
use std::{
  cmp::Ord,
  collections::{HashMap, HashSet},
  default::Default,
  fmt::Debug,
  hash::Hash,
};

use crate::{
  Arity, AstNode, Teachable,
  bb_query::BBInfo,
  extract::beam_pareto::{ISAXAnalysis, TypeInfo},
  runner::OperationInfo,
};

pub fn perf_infer<Op, T>(
  egraph: &mut EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  roots: &[Id],
  bb_info: Vec<String>,
) where
  Op: Clone
    + Debug
    + Ord
    + Hash
    + Teachable
    + Arity
    + OperationInfo
    + Send
    + Sync
    + BBInfo
    + 'static,
  T: Debug + Default + Clone + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  let mut visited = HashSet::new();
  let mut bb_info_map: HashMap<Id, Vec<String>> = HashMap::new();
  let mut work_stack: Vec<(Id, Vec<String>)> =
    roots.iter().map(|&id| (id, bb_info.clone())).collect();

  while let Some((current_id, current_bb_info)) = work_stack.pop() {
    if !visited.insert(current_id) {
      continue;
    }

    // println!("perf_infer: {:?}", current_id);

    let mut ecls_bb_info = current_bb_info;
    let ecls = &mut egraph[current_id];

    for node in &mut ecls.nodes {
      let bb_info = node.operation_mut().get_mut_bbs_info();
      if bb_info.len() > 0 {
        ecls_bb_info = bb_info.clone();
        break;
      }
    }

    if ecls_bb_info.len() > 1 {
      ecls_bb_info = vec![ecls_bb_info[0].clone()];
    }
    bb_info_map.insert(current_id, ecls_bb_info.clone());

    for node in &mut ecls.nodes {
      for id in node.children() {
        if !visited.contains(id) {
          work_stack.push((*id, ecls_bb_info.clone()));
        }
      }
    }
  }

  for ecls in egraph.classes_mut() {
    let id = ecls.id;
    if let Some(ecls_bb_info) = bb_info_map.get(&id) {
      ecls.data.bb = ecls_bb_info.clone();
      for node in &mut ecls.nodes {
        let bb_info = node.operation_mut().get_mut_bbs_info();
        if bb_info.len() > 0 {
          *bb_info = ecls_bb_info.clone();
          break;
        }
      }
    }
  }

  egraph.rebuild();
}
