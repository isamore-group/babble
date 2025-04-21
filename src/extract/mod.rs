//! Extracting expressions with learned libs out of egraphs

pub mod beam_pareto;

use egg::{Analysis, EGraph, Id, RecExpr, Rewrite, Runner};

use crate::{
  ast_node::{Arity, AstNode},
  teachable::Teachable,
};

/// Given an `egraph` that contains the original expression at `roots`,
/// and a set of library `rewrites`, extract the programs rewritten using the
/// library with pareto optimization for area and delay.
pub fn apply_libs_pareto<Op, A>(
  egraph: EGraph<AstNode<Op>, A>,
  roots: &[Id],
  rewrites: &[Rewrite<AstNode<Op>, A>],
  strategy: f32,
) -> (RecExpr<AstNode<Op>>, f32)
where
  Op: Clone
    + Teachable
    + Ord
    + std::fmt::Debug
    + std::fmt::Display
    + std::hash::Hash
    + Arity
    + Send
    + Sync,
  A: Analysis<AstNode<Op>> + Default + Clone,
{
  let mut fin = Runner::<_, _, ()>::new(Default::default())
    .with_egraph(egraph)
    .run(rewrites.iter())
    .egraph;
  let root = fin.add(AstNode::new(Op::list(), roots.iter().copied()));

  let mut extractor = beam_pareto::LibExtractor::new(&fin, strategy);
  let best = extractor.best(root);
  let cost = extractor.cost(&best);
  (best, cost)
}
