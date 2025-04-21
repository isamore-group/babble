//! Runner module for executing library learning experiments
//!
//! This module provides functionality for running library learning experiments
//! using either regular beam search or Pareto-optimal beam search.

use std::{
  collections::HashMap,
  fmt::{Debug, Display},
  hash::Hash,
  sync::Arc,
  time::{Duration, Instant},
};

use egg::{EGraph, Id, RecExpr, Rewrite, Runner as EggRunner};
use log::{debug, info};

use crate::{
  Arity, AstNode, DiscriminantEq, Expr, LearnedLibraryBuilder, Pretty,
  Printable, Teachable,
  extract::{
    apply_libs_pareto,
    beam_pareto::{ISAXAnalysis, TypeInfo},
  },
  schedule::Schedulable,
};

use crate::USE_RULES;

/// 定义一个trait名为LibOperation,其中有两个函数，分别为get_libid和change_libid
pub trait LibOperation {
  fn is_lib(&self) -> bool;
  /// Get the library ID of the operation
  fn get_libid(&self) -> usize;
}

/// A trait for running library learning experiments with Pareto optimization
pub trait BabbleParetoRunner<Op, T>
where
  Op: Arity
    + Teachable
    + Printable
    + Debug
    + Display
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + Schedulable
    + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  /// Run the experiment on a single expression
  fn run(&self, expr: Expr<Op>) -> ParetoResult<Op, T>;

  /// Run the experiment on multiple expressions
  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> ParetoResult<Op, T>;

  /// Run the experiment on groups of equivalent expressions
  fn run_equiv_groups(
    &self,
    expr_groups: Vec<Vec<Expr<Op>>>,
  ) -> ParetoResult<Op, T>;
}

/// Result of running a BabbleRunner experiment
#[derive(Clone)]
pub struct ParetoResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + Schedulable
    + 'static,
  T: Debug,
{
  /// The final expression after library learning and application
  pub final_expr: Expr<Op>,
  /// The number of libraries learned
  pub num_libs: usize,
  /// The rewrites representing the learned libraries
  pub rewrites: HashMap<usize, Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// The final cost of the expression
  pub final_cost: f32,
  /// The time taken to run the experiment
  pub run_time: Duration,
}

impl<Op, T> std::fmt::Debug for ParetoResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + Schedulable
    + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ParetoResult")
      .field("num_libs", &self.num_libs)
      .field("final_cost", &self.final_cost)
      .field("run_time", &self.run_time)
      .finish()
  }
}

/// Configuration for beam search
#[derive(Debug, Clone, Copy)]
pub struct ParetoConfig {
  /// The final beam size to use
  pub final_beams: usize,
  /// The inter beam size to use
  pub inter_beams: usize,
  /// The number of times to apply library rewrites
  pub lib_iter_limit: usize,
  /// The number of libs to learn at a time
  pub lps: usize,
  /// Whether to learn "library functions" with no arguments
  pub learn_constants: bool,
  /// Maximum arity of a library function
  pub max_arity: Option<usize>,
  /// Strategy of balancing area and delay
  pub strategy: f32,
  /// clock period used for scheduling
  pub clock_period: usize,
}

impl Default for ParetoConfig {
  fn default() -> Self {
    Self {
      final_beams: 10,
      inter_beams: 5,
      lib_iter_limit: 3,
      lps: 1,
      learn_constants: false,
      max_arity: None,
      strategy: 0.8,
      clock_period: 100,
    }
  }
}

/// A Pareto Runner that uses Pareto optimization with beam search
pub struct ParetoRunner<Op, T>
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
    + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  /// The domain-specific rewrites to apply
  dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// lib rewrites
  lib_rewrites: HashMap<usize, Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// Configuration for the beam search
  config: ParetoConfig,
}

impl<Op, T> ParetoRunner<Op, T>
where
  Op: Arity
    + LibOperation
    + Teachable
    + Printable
    + Debug
    + Default
    + Display
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + Schedulable
    + 'static,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display,
  AstNode<Op>: TypeInfo<T>,
{
  /// Create a new BeamRunner with the given domain-specific rewrites and
  /// configuration
  pub fn new<I>(
    dsrs: I,
    lib_rewrites: HashMap<usize, Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
    config: ParetoConfig,
  ) -> Self
  where
    I: IntoIterator<Item = Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  {
    // 如果USE_RULES为false，将dsrs清空
    let dsrs = if !USE_RULES {
      Vec::new()
    } else {
      dsrs.into_iter().collect()
    };
    Self {
      dsrs,
      lib_rewrites,
      config,
    }
  }

  /// Run the e-graph and library learning process
  fn run_egraph(
    &self,
    roots: &[Id],
    egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> ParetoResult<Op, T> {
    let start_time = Instant::now();
    let timeout = Duration::from_secs(60 * 100_000);

    println!(
      "Initial egraph size: {}, eclasses: {}",
      egraph.total_size(),
      egraph.classes().len()
    );
    println!("Running {} DSRs... ", self.dsrs.len());

    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::empty())
      .with_egraph(egraph)
      .with_time_limit(timeout)
      .with_iter_limit(3)
      .run(&self.dsrs);

    let aeg = runner.egraph;

    println!(
      "Final egraph size: {}, eclasses: {}",
      aeg.total_size(),
      aeg.classes().len()
    );

    info!(
      "Finished in {}ms; final egraph size: {}",
      start_time.elapsed().as_millis(),
      aeg.total_size()
    );

    // println!("Running co-occurrence analysis... ");
    // let co_time = Instant::now();
    // let co_ext = COBuilder::new(&aeg, roots);
    // let co_occurs = co_ext.run();
    // println!("Finished in {}ms", co_time.elapsed().as_millis());

    println!("Running anti-unification... ");
    let au_time = Instant::now();
    // 在进行learn之前，先提取出LibId最大的Lib
    let mut max_lib_id = 0;
    for eclass in aeg.classes() {
      for node in eclass.iter() {
        let op = node.operation();
        if op.is_lib() {
          max_lib_id = std::cmp::max(max_lib_id, op.get_libid());
        }
      }
    }
    max_lib_id += 1;
    let mut learned_lib = LearnedLibraryBuilder::default()
      .learn_constants(self.config.learn_constants)
      .max_arity(self.config.max_arity)
      // .with_co_occurs(co_occurs)
      .with_last_lib_id(max_lib_id)
      .with_clock_period(self.config.clock_period)
      .build(&aeg);
    println!(
      "Found {} patterns in {}ms",
      learned_lib.size(),
      au_time.elapsed().as_millis()
    );

    info!("Deduplicating patterns... ");
    let dedup_time = Instant::now();
    learned_lib.deduplicate(&aeg);
    let lib_rewrites: Vec<_> = learned_lib.rewrites().collect();
    // let first_rewrite = lib_rewrites[0].clone();
    // println!(
    //   "first rewrite: {:?}",
    //   first_rewrite
    // );
    // let search_result = first_rewrite.search(&aeg);
    // for matches in search_result {
    //   println!("{:?}", matches);
    // }
    info!(
      "Reduced to {} patterns in {}ms",
      learned_lib.size(),
      dedup_time.elapsed().as_millis()
    );

    println!("learned {} libs", learned_lib.size());
    // for lib in &learned_lib.libs().collect::<Vec<_>>() {
    //   println!("{}", lib);
    // }

    info!("Adding libs and running beam search... ");
    let extract_time = Instant::now();
    let lib_rewrite_time = Instant::now();
    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.config.strategy,
    ))
    .with_egraph(aeg.clone())
    .with_iter_limit(self.config.lib_iter_limit)
    .with_time_limit(timeout)
    .with_node_limit(1_000_000)
    .run(lib_rewrites.iter());

    let mut egraph = runner.egraph;
    let root = egraph.add(AstNode::new(Op::list(), roots.iter().copied()));
    let mut isax_cost = egraph[egraph.find(root)].data.clone();
    // println!("root: {:#?}", egraph[egraph.find(root)]);
    // println!("cs: {:#?}", cs);
    // println!("cs: {:#?}", isax_cost.cs.clone());
    isax_cost
      .cs
      .set
      .sort_unstable_by_key(|elem| elem.full_cost as usize);

    info!("Finished in {}ms", lib_rewrite_time.elapsed().as_millis());
    info!("Stop reason: {:?}", runner.stop_reason.unwrap());
    info!("Number of nodes: {}", egraph.total_size());

    // egraph.dot().to_png("target/foo.png").unwrap();

    // println!("egraph: {:#?}", egraph);

    println!("learned libs");
    // let all_libs: Vec<_> = learned_lib.libs().collect();
    // println!("cs: {:#?}", isax_cost.cs);
    let mut chosen_rewrites = Vec::new();
    let mut rewrites_map = HashMap::new();
    for lib in &isax_cost.cs.set[0].libs {
      if lib.0.0 < max_lib_id {
        // 从self.lib_rewrites中取出
        // 打印self.lib_rewrites
        // println!("{}: {:?}", lib.0.0, self.lib_rewrites);
        chosen_rewrites.push(self.lib_rewrites.get(&lib.0.0).unwrap().clone());
        rewrites_map
          .insert(lib.0.0, self.lib_rewrites.get(&lib.0.0).unwrap().clone());
      } else {
        let new_lib = lib.0.0 - max_lib_id;
        chosen_rewrites.push(lib_rewrites[new_lib].clone());
        rewrites_map.insert(lib.0.0, lib_rewrites[new_lib].clone());
      }
    }

    debug!(
      "upper bound ('full') cost: {}",
      isax_cost.cs.set[0].full_cost
    );

    let ex_time = Instant::now();
    info!("Extracting... ");
    let (best, final_cost) = apply_libs_pareto(
      aeg.clone(),
      roots,
      &chosen_rewrites,
      self.config.strategy,
    );

    println!("extracting using {}s", extract_time.elapsed().as_secs());
    // Lifting the lib will result in incorrect cost
    // let lifted = extract::lift_libs(&best);

    info!("Finished in {}ms", ex_time.elapsed().as_millis());
    debug!("final cost: {}", final_cost);
    debug!("{}", Pretty::new(Arc::new(Expr::from(best.clone()))));
    info!("round time: {}ms", start_time.elapsed().as_millis());

    ParetoResult {
      final_expr: best.into(),
      num_libs: chosen_rewrites.len(),
      rewrites: rewrites_map,
      final_cost: final_cost,
      run_time: start_time.elapsed(),
    }
  }
}

impl<Op, T> BabbleParetoRunner<Op, T> for ParetoRunner<Op, T>
where
  Op: Arity
    + LibOperation
    + Teachable
    + Printable
    + Debug
    + Default
    + Display
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + Schedulable
    + 'static,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display,
  AstNode<Op>: TypeInfo<T>,
{
  fn run(&self, expr: Expr<Op>) -> ParetoResult<Op, T> {
    self.run_multi(vec![expr])
  }

  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> ParetoResult<Op, T> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexprs: Vec<RecExpr<AstNode<Op>>> =
      exprs.into_iter().map(RecExpr::from).collect();
    let mut egraph = EGraph::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.config.strategy,
    ));
    let roots = recexprs
      .iter()
      .map(|x| egraph.add_expr(x))
      .collect::<Vec<_>>();
    egraph.rebuild();

    self.run_egraph(&roots, egraph)
  }

  fn run_equiv_groups(
    &self,
    expr_groups: Vec<Vec<Expr<Op>>>,
  ) -> ParetoResult<Op, T> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexpr_groups: Vec<Vec<_>> = expr_groups
      .into_iter()
      .map(|group| group.into_iter().map(RecExpr::from).collect())
      .collect();

    let mut egraph = EGraph::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.config.strategy,
    ));

    let roots: Vec<_> = recexpr_groups
      .into_iter()
      .map(|mut group| {
        let first_expr = group.pop().unwrap();
        let root = egraph.add_expr(&first_expr);
        for expr in group {
          let class = egraph.add_expr(&expr);
          egraph.union(root, class);
        }
        root
      })
      .collect();

    egraph.rebuild();

    self.run_egraph(&roots, egraph)
  }
}

// Implement Debug that doesn't depend on Op being Debug
impl<Op, T> std::fmt::Debug for ParetoRunner<Op, T>
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
    + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BeamRunner")
      .field("dsrs_count", &self.dsrs.len())
      .field("config", &self.config)
      .finish()
  }
}
