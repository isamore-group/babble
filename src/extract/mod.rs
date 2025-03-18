//! Extracting expressions with learned libs out of egraphs

pub mod beam;
pub mod beam_pareto;
pub mod beam_knapsack;
pub mod cost;

use std::collections::HashMap;

use egg::{Analysis, EGraph, Id, Language, RecExpr, Rewrite, Runner};

use crate::{
  ast_node::{Arity, AstNode},
  learn::LibId,
  teachable::{BindingExpr, Teachable},
};

/// Given an `egraph` that contains the original expression at `roots`,
/// and a set of library `rewrites`, extract the programs rewritten using the library.
pub fn apply_libs<Op, A>(
  egraph: EGraph<AstNode<Op>, A>,
  roots: &[Id],
  rewrites: &[Rewrite<AstNode<Op>, A>],
) -> RecExpr<AstNode<Op>>
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

  let mut extractor = beam::LibExtractor::new(&fin);
  let best = extractor.best(root);
  lift_libs(&best)
}

/// Given an `egraph` that contains the original expression at `roots`,
/// and a set of library `rewrites`, extract the programs rewritten using the library
/// with knapsack optimization for area and delay.
pub fn apply_libs_knapsack<Op, A, LA, LD>(
  egraph: EGraph<AstNode<Op>, A>,
  roots: &[Id],
  rewrites: &[Rewrite<AstNode<Op>, A>],
  lang_cost: LA,
  lang_gain: LD,
) -> RecExpr<AstNode<Op>>
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
  LA: cost::LangCost<Op> + Clone + Default,
  LD: cost::LangGain<Op> + Clone + Default,
{
  let mut fin = Runner::<_, _, ()>::new(Default::default())
    .with_egraph(egraph)
    .run(rewrites.iter())
    .egraph;
  let root = fin.add(AstNode::new(Op::list(), roots.iter().copied()));

  let mut extractor = beam_knapsack::LibExtractor::new(&fin, lang_cost.clone(), lang_gain.clone());
  extractor.best(root)
}

/// Given an `egraph` that contains the original expression at `roots`,
/// and a set of library `rewrites`, extract the programs rewritten using the library
/// with Pareto optimization for area and delay.
pub fn apply_libs_pareto<Op, A>(
  egraph: EGraph<AstNode<Op>, A>,
  roots: &[Id],
  rewrites: &[Rewrite<AstNode<Op>, A>],
  strategy: beam::OptimizationStrategy,
) -> RecExpr<AstNode<Op>>
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

  let mut extractor = beam::LibExtractor::with_strategy(&fin, strategy);
  let best = extractor.best(root);
  lift_libs(&best)
}

/// Given an `egraph` that contains the original expression at `roots`,
/// and a set of library `rewrites`, extract the programs rewritten using the library
/// with Pareto optimization for area and delay.
pub fn apply_libs_area_delay<Op, A, LA, LD>(
  egraph: EGraph<AstNode<Op>, A>,
  roots: &[Id],
  rewrites: &[Rewrite<AstNode<Op>, A>],
  area_cost: LA,
  delay_cost: LD,
  strategy: beam::OptimizationStrategy,
) -> RecExpr<AstNode<Op>>
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
  LA: cost::LangCost<Op> + Clone + Send + Sync + 'static,
  LD: cost::LangGain<Op> + Clone + Send + Sync + 'static,
{
  // First, run the rewrites on the original egraph
  let mut fin = Runner::<_, _, ()>::new(Default::default())
    .with_egraph(egraph)
    .run(rewrites.iter())
    .egraph;
  
  // Add a root node that combines all roots
  let root = fin.add(AstNode::new(Op::list(), roots.iter().copied()));
  
  // Create a LibExtractor with the strategy
  let mut extractor = beam::LibExtractor::with_strategy(&fin, strategy);
  let best = extractor.best(root);
  
  // Lift the libraries to the top
  lift_libs(&best)
}

/// Given an expression and a set of library rewrites, extract the programs
/// rewritten using the library with Pareto-optimal beam search for area and delay.
/// This function creates a new egraph with BeamAreaDelay analysis.
pub fn extract_with_beam_area_delay<Op, LA, LD>(
  expr: &RecExpr<AstNode<Op>>,
  rewrites: &[Rewrite<AstNode<Op>, beam_pareto::BeamAreaDelay>],
  area_cost: LA,
  delay_cost: LD,
  strategy: beam::OptimizationStrategy,
  beam_size: usize,
  inter_beam: usize,
  lps: usize,
) -> RecExpr<AstNode<Op>>
where
  Op: Clone
    + Teachable
    + Ord
    + std::fmt::Debug
    + std::fmt::Display
    + std::hash::Hash
    + Arity
    + Send
    + Sync
    + 'static,
  LA: cost::LangCost<Op> + Clone + Send + Sync + 'static,
  LD: cost::LangGain<Op> + Clone + Send + Sync + 'static,
{
  // Create a new egraph with BeamAreaDelay analysis
  let mut egraph = EGraph::new(beam_pareto::BeamAreaDelay::new(
    beam_size,
    inter_beam,
    lps,
    strategy,
  ));
  
  // Add the expression to the egraph
  let root = egraph.add_expr(expr);
  
  // Run the rewrites
  let runner = Runner::<_, _, ()>::new(Default::default())
    .with_egraph(egraph)
    .run(rewrites);
  
  // Use the LibExtractor to get the best expression
  let mut extractor = beam::LibExtractor::with_strategy(&runner.egraph, strategy);
  let best = extractor.best(root);
  
  // Lift the libraries to the top
  lift_libs(&best)
}

fn build<Op: Clone + Teachable + std::fmt::Debug>(
  orig: &[AstNode<Op>],
  cur: Id,
  mut seen: impl FnMut(LibId, Id),
) -> AstNode<Op> {
  match orig[Into::<usize>::into(cur)].as_binding_expr() {
    Some(BindingExpr::Lib(id, lam, c)) => {
      seen(id, *lam);
      build(orig, *c, seen)
    }
    _ => orig[Into::<usize>::into(cur)].clone(),
  }
}

/// Given an expression `expr` containing library function definitions, move
/// those definitions to the top.
#[must_use]
pub fn lift_libs<Op>(expr: &RecExpr<AstNode<Op>>) -> RecExpr<AstNode<Op>>
where
  Op: Clone + Teachable + Ord + std::fmt::Debug + std::hash::Hash,
{
  let orig: Vec<AstNode<Op>> = expr.as_ref().to_vec();
  let mut seen = HashMap::new();

  let rest = orig[orig.len() - 1].build_recexpr(|id| {
    build(&orig, id, |k, v| {
      seen.insert(k, v);
    })
  });
  let mut res = rest.as_ref().to_vec();

  // Work queue for functions we still have to do
  let mut q: Vec<(LibId, Id)> = seen.iter().map(|(k, v)| (*k, *v)).collect();

  // TODO: order based on libs dependency w each other?
  while let Some((lib, expr)) = q.pop() {
    let body = res.len() - 1;
    let value: Vec<_> = orig[Into::<usize>::into(expr)]
      .build_recexpr(|id| {
        build(&orig, id, |k, v| {
          if seen.insert(k, v).is_none() {
            q.push((k, v));
          }
        })
      })
      .as_ref()
      .iter()
      .cloned()
      .map(|x| x.map_children(|x| (usize::from(x) + res.len()).into()))
      .collect();
    res.extend(value);
    res.push(Teachable::lib(lib, Id::from(res.len() - 1), Id::from(body)));
  }

  res.into()
}
