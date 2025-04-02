//! Runner module for executing library learning experiments
//!
//! This module provides functionality for running library learning experiments
//! using either regular beam search or Pareto-optimal beam search.

use std::{
  collections::HashMap,
  fmt::{Debug, Display},
  hash::Hash,
  marker::PhantomData,
  sync::Arc,
  time::{Duration, Instant},
};

use egg::{
  AstSize, CostFunction, EGraph, Id, RecExpr, Rewrite, Runner as EggRunner,
};
use log::{debug, info};

use crate::{
  Arity, AstNode, COBuilder, DiscriminantEq, Expr, LearnedLibraryBuilder,
  Pretty, Printable, Teachable,
  extract::{
    apply_libs, apply_libs_pareto,
    beam::PartialLibCost,
    beam_pareto::{ISAXAnalysis, TypeInfo},
    cost::{AreaCost, DelayCost, LangCost, LangGain},
  },
};

use crate::USE_RULES;

/// 定义一个trait名为LibOperation,其中有两个函数，分别为get_libid和change_libid
pub trait LibOperation {
  fn is_lib(&self) -> bool;
  /// Get the library ID of the operation
  fn get_libid(&self) -> usize;
}

/// Result of running a BabbleRunner experiment

#[derive(Clone)]
pub struct BabbleResult<Op>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + 'static,
{
  /// The final expression after library learning and application
  pub final_expr: Expr<Op>,
  /// The number of libraries learned
  pub num_libs: usize,
  /// The rewrites representing the learned libraries
  pub rewrites: Vec<Rewrite<AstNode<Op>, PartialLibCost>>,
  /// The initial cost of the expression(s)
  pub initial_cost: usize,
  /// The final cost of the expression
  pub final_cost: usize,
  /// The time taken to run the experiment
  pub run_time: Duration,
}

impl<Op> BabbleResult<Op>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + 'static,
{
  /// Calculate the compression ratio achieved
  pub fn compression_ratio(&self) -> f64 {
    if self.initial_cost == 0 {
      1.0
    } else {
      1.0 - (self.final_cost as f64 / self.initial_cost as f64)
    }
  }
}

impl<Op> std::fmt::Debug for BabbleResult<Op>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + 'static,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BabbleResult")
      .field("num_libs", &self.num_libs)
      .field("initial_cost", &self.initial_cost)
      .field("final_cost", &self.final_cost)
      .field("compression_ratio", &self.compression_ratio())
      .field("run_time", &self.run_time)
      .finish()
  }
}

/// A trait for running library learning experiments
pub trait BabbleRunner<Op>
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
    + 'static,
{
  /// Run the experiment on a single expression
  fn run(&self, expr: Expr<Op>) -> BabbleResult<Op>;

  /// Run the experiment on multiple expressions
  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> BabbleResult<Op>;

  /// Run the experiment on groups of equivalent expressions
  fn run_equiv_groups(
    &self,
    expr_groups: Vec<Vec<Expr<Op>>>,
  ) -> BabbleResult<Op>;
}

/// Configuration for beam search
#[derive(Debug, Clone, Copy)]
pub struct BeamConfig {
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
}

impl Default for BeamConfig {
  fn default() -> Self {
    Self {
      final_beams: 10,
      inter_beams: 5,
      lib_iter_limit: 3,
      lps: 5,
      learn_constants: false,
      max_arity: None,
    }
  }
}

/// A BabbleRunner that uses regular beam search
pub struct BeamRunner<Op, LD>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + Send + Sync + 'static,
  LD: LangGain<Op> + Clone + Send + Sync + 'static,
{
  /// The domain-specific rewrites to apply
  dsrs: Vec<Rewrite<AstNode<Op>, PartialLibCost>>,
  /// Configuration for the beam search
  config: BeamConfig,

  lang_gain: LD,
}

impl<Op, LD> BeamRunner<Op, LD>
where
  Op: Arity
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
    + 'static,
  LD: LangGain<Op> + Clone + Default + Send + Sync + 'static,
{
  /// Create a new BeamRunner with the given domain-specific rewrites and
  /// configuration
  pub fn new<I>(dsrs: I, config: BeamConfig, lang_gain: LD) -> Self
  where
    I: IntoIterator<Item = Rewrite<AstNode<Op>, PartialLibCost>>,
  {
    // 如果USE_RULES为false，将dsrs清空
    let dsrs = if !USE_RULES {
      Vec::new()
    } else {
      dsrs.into_iter().collect()
    };
    Self {
      dsrs,
      config,
      lang_gain,
    }
  }

  /// Run the e-graph and library learning process
  fn run_egraph(
    &self,
    roots: &[Id],
    egraph: EGraph<AstNode<Op>, PartialLibCost>,
  ) -> BabbleResult<Op> {
    let start_time = Instant::now();
    let timeout = Duration::from_secs(60 * 100_000);

    info!("Initial egraph size: {}", egraph.total_size());
    info!("Running {} DSRs... ", self.dsrs.len());

    let runner = EggRunner::<_, _, ()>::new(PartialLibCost::empty())
      .with_egraph(egraph)
      .with_time_limit(timeout)
      .with_iter_limit(3)
      .run(&self.dsrs);

    let aeg = runner.egraph;

    info!(
      "Finished in {}ms; final egraph size: {}",
      start_time.elapsed().as_millis(),
      aeg.total_size()
    );

    info!("Running co-occurrence analysis... ");
    let co_time = Instant::now();
    let co_ext = COBuilder::new(&aeg, roots);
    let co_occurs = co_ext.run();
    info!("Finished in {}ms", co_time.elapsed().as_millis());

    info!("Running anti-unification... ");
    let au_time = Instant::now();
    let mut learned_lib = LearnedLibraryBuilder::default()
      .learn_constants(self.config.learn_constants)
      .max_arity(self.config.max_arity)
      .with_co_occurs(co_occurs)
      .build(&aeg, self.lang_gain.clone());
    info!(
      "Found {} patterns in {}ms",
      learned_lib.size(),
      au_time.elapsed().as_millis()
    );

    info!("Deduplicating patterns... ");
    let dedup_time = Instant::now();
    learned_lib.deduplicate(&aeg);
    let lib_rewrites: Vec<_> = learned_lib.rewrites().collect();
    info!(
      "Reduced to {} patterns in {}ms",
      learned_lib.size(),
      dedup_time.elapsed().as_millis()
    );

    // println!("learned {} libs", learned_lib.size());
    // for lib in &learned_lib.libs().collect::<Vec<_>>() {
    //     println!("{}", lib);
    // }

    info!("Adding libs and running beam search... ");
    let lib_rewrite_time = Instant::now();
    let runner = EggRunner::<_, _, ()>::new(PartialLibCost::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
    ))
    .with_egraph(aeg.clone())
    .with_iter_limit(self.config.lib_iter_limit)
    .with_time_limit(timeout)
    .with_node_limit(1_000_000)
    .run(lib_rewrites.iter());

    let mut egraph = runner.egraph;
    let root = egraph.add(AstNode::new(Op::list(), roots.iter().copied()));
    let mut cs = egraph[egraph.find(root)].data.clone();
    // println!("root: {:#?}", egraph[egraph.find(root)]);
    // println!("cs: {:#?}", cs);
    cs.set.sort_unstable_by_key(|elem| elem.full_cost);

    info!("Finished in {}ms", lib_rewrite_time.elapsed().as_millis());
    info!("Stop reason: {:?}", runner.stop_reason.unwrap());
    info!("Number of nodes: {}", egraph.total_size());

    println!("learned libs");
    // let all_libs: Vec<_> = learned_lib.libs().collect();
    let mut chosen_rewrites = Vec::new();
    for lib in &cs.set[0].libs {
      // println!("{}: {}", lib.0, &all_libs[lib.0.0]);
      chosen_rewrites.push(lib_rewrites[lib.0.0].clone());
    }

    debug!("upper bound ('full') cost: {}", cs.set[0].full_cost);
    println!("Size of cs.set: {}", cs.set.len());

    let ex_time = Instant::now();
    info!("Extracting... ");
    let lifted = apply_libs(aeg.clone(), roots, &chosen_rewrites);
    let final_cost = AstSize.cost_rec(&lifted);

    info!("Finished in {}ms", ex_time.elapsed().as_millis());
    info!("final cost: {}", final_cost);
    debug!("{}", Pretty::new(Arc::new(Expr::from(lifted.clone()))));
    info!("round time: {}ms", start_time.elapsed().as_millis());

    // Calculate initial cost
    let initial_cost = {
      let s: usize = roots
        .iter()
        .map(|id| {
          let extractor = egg::Extractor::new(&egraph, AstSize);
          let (_, expr) = extractor.find_best(*id);
          AstSize.cost_rec(&expr)
        })
        .sum();
      s + 1 // Add one to account for root node
    };

    BabbleResult {
      final_expr: lifted.into(),
      num_libs: chosen_rewrites.len(),
      rewrites: chosen_rewrites,
      initial_cost,
      final_cost,
      run_time: start_time.elapsed(),
    }
  }
}

impl<Op, LD> BabbleRunner<Op> for BeamRunner<Op, LD>
where
  Op: Arity
    + Teachable
    + Printable
    + Debug
    + Display
    + Default
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + 'static,
  LD: LangGain<Op> + Clone + Default + Send + Sync + 'static,
{
  fn run(&self, expr: Expr<Op>) -> BabbleResult<Op> {
    self.run_multi(vec![expr])
  }

  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> BabbleResult<Op> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexprs: Vec<RecExpr<AstNode<Op>>> =
      exprs.into_iter().map(RecExpr::from).collect();

    let mut egraph = EGraph::new(PartialLibCost::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
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
  ) -> BabbleResult<Op> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexpr_groups: Vec<Vec<_>> = expr_groups
      .into_iter()
      .map(|group| group.into_iter().map(RecExpr::from).collect())
      .collect();

    let mut egraph = EGraph::new(PartialLibCost::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
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
impl<Op, LD> std::fmt::Debug for BeamRunner<Op, LD>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + Send + Sync + 'static,
  LD: LangGain<Op> + Clone + Default + Send + Sync + 'static,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BeamRunner")
      .field("dsrs_count", &self.dsrs.len())
      .field("config", &self.config)
      .field("lang_gain", &PhantomData::<LD>)
      .finish()
  }
}

/// A trait for running library learning experiments with Pareto optimization
pub trait BabbleParetoRunner<Op, T, LA, LD>
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
    + 'static,
  LA: LangCost<Op> + Clone + Default,
  LD: LangGain<Op> + Clone + Default,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  /// Run the experiment on a single expression
  fn run(&self, expr: Expr<Op>) -> ParetoResult<Op, T, LA, LD>;

  /// Run the experiment on multiple expressions
  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> ParetoResult<Op, T, LA, LD>;

  /// Run the experiment on groups of equivalent expressions
  fn run_equiv_groups(
    &self,
    expr_groups: Vec<Vec<Expr<Op>>>,
  ) -> ParetoResult<Op, T, LA, LD>;
}

/// Result of running a BabbleRunner experiment
#[derive(Clone)]
pub struct ParetoResult<Op, T, LA, LD>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + Debug + 'static,
  LA: LangCost<Op> + Clone + Default,
  LD: LangGain<Op> + Clone + Default,
  T: Debug,
{
  /// The final expression after library learning and application
  pub final_expr: Expr<Op>,
  /// The number of libraries learned
  pub num_libs: usize,
  /// The rewrites representing the learned libraries
  pub rewrites:
    HashMap<usize, Rewrite<AstNode<Op>, ISAXAnalysis<Op, T, LA, LD>>>,
  /// The initial cost of the expression
  pub initial_cost: (usize, usize, f32),
  /// The final cost of the expression (area_cost, delay_cost, balanced_cost)
  pub final_cost: (usize, usize, f32),
  /// The time taken to run the experiment
  pub run_time: Duration,
}

impl<Op, T, LA, LD> ParetoResult<Op, T, LA, LD>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + Debug + 'static,
  LA: LangCost<Op> + Clone + Default,
  LD: LangGain<Op> + Clone + Default,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  /// Calculate the compression ratio achieved
  pub fn compression_ratio(&self) -> f32 {
    if self.initial_cost.2 == 0.0 {
      1.0
    } else {
      1.0 - (self.final_cost.2 / self.initial_cost.2)
    }
  }
}

impl<Op, T, LA, LD> std::fmt::Debug for ParetoResult<Op, T, LA, LD>
where
  Op: Display + Hash + Clone + Ord + Teachable + Arity + Debug + 'static,
  LA: LangCost<Op> + Clone + Default,
  LD: LangGain<Op> + Clone + Default,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BabbleResult")
      .field("num_libs", &self.num_libs)
      .field("initial_cost", &self.initial_cost.2)
      .field("final_cost", &self.final_cost.2)
      .field("compression_ratio", &self.compression_ratio())
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
  /// strategy of balancing area and delay
  pub strategy: f32,
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
    }
  }
}

/// A BabbleRunner that uses regular beam search
pub struct ParetoRunner<Op, T, LA, LD>
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
  LA: LangCost<Op> + Clone + Default,
  LD: LangGain<Op> + Clone + Default,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  /// The domain-specific rewrites to apply
  dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T, LA, LD>>>,
  /// lib rewrites
  lib_rewrites:
    HashMap<usize, Rewrite<AstNode<Op>, ISAXAnalysis<Op, T, LA, LD>>>,
  /// Configuration for the beam search
  config: ParetoConfig,
  lang_cost: LA,
  lang_gain: LD,
}

impl<Op, T, LA, LD> ParetoRunner<Op, T, LA, LD>
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
    + 'static,
  LA: LangCost<Op> + Clone + Default + 'static,
  LD: LangGain<Op> + Clone + Default + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  /// Create a new BeamRunner with the given domain-specific rewrites and
  /// configuration
  pub fn new<I>(
    dsrs: I,
    lib_rewrites: HashMap<
      usize,
      Rewrite<AstNode<Op>, ISAXAnalysis<Op, T, LA, LD>>,
    >,
    config: ParetoConfig,
    lang_cost: LA,
    lang_gain: LD,
  ) -> Self
  where
    I: IntoIterator<Item = Rewrite<AstNode<Op>, ISAXAnalysis<Op, T, LA, LD>>>,
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
      lang_cost,
      lang_gain,
    }
  }

  /// Run the e-graph and library learning process
  fn run_egraph(
    &self,
    roots: &[Id],
    egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T, LA, LD>>,
  ) -> ParetoResult<Op, T, LA, LD> {
    let start_time = Instant::now();
    let timeout = Duration::from_secs(60 * 100_000);

    let mut init_delay: usize = match roots.len() {
      0 | 1 => 0,
      _ => 1,
    };
    let mut init_area: usize = 0;
    for root in roots {
      let delay_extractor =
        egg::Extractor::new(&egraph, DelayCost::new(self.lang_gain.clone()));
      let (_, expr) = delay_extractor.find_best(*root);
      let delay_cost = DelayCost::new(self.lang_gain.clone()).cost_rec(&expr);
      let selected_delay_cost = match delay_cost.2 {
        true => delay_cost.0,
        false => delay_cost.1,
      };
      init_delay += selected_delay_cost;

      let area_extractor =
        egg::Extractor::new(&egraph, AreaCost::new(self.lang_cost.clone()));
      let (_, expr) = area_extractor.find_best(*root);
      let area_cost = AreaCost::new(self.lang_cost.clone()).cost_rec(&expr);
      let selected_area_cost = match area_cost.2 {
        true => area_cost.0,
        false => area_cost.1,
      };
      init_area += selected_area_cost;
    }
    let init_cost = self.config.strategy * (init_delay as f32)
      + (1.0 - self.config.strategy) * (init_area as f32);

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

    info!("Running co-occurrence analysis... ");
    let co_time = Instant::now();
    let co_ext = COBuilder::new(&aeg, roots);
    let co_occurs = co_ext.run();
    info!("Finished in {}ms", co_time.elapsed().as_millis());

    info!("Running anti-unification... ");
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
      .with_co_occurs(co_occurs)
      .with_last_lib_id(max_lib_id)
      .build(&aeg, self.lang_gain.clone());
    info!(
      "Found {} patterns in {}ms",
      learned_lib.size(),
      au_time.elapsed().as_millis()
    );

    info!("Deduplicating patterns... ");
    let dedup_time = Instant::now();
    learned_lib.deduplicate(&aeg);
    let lib_rewrites: Vec<_> = learned_lib.rewrites().collect();
    info!(
      "Reduced to {} patterns in {}ms",
      learned_lib.size(),
      dedup_time.elapsed().as_millis()
    );

    println!("learned {} libs", learned_lib.size());
    for lib in &learned_lib.libs().collect::<Vec<_>>() {
      println!("{}", lib);
    }

    info!("Adding libs and running beam search... ");
    let lib_rewrite_time = Instant::now();
    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.lang_cost.clone(),
      self.lang_gain.clone(),
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
    println!("root: {:#?}", egraph[egraph.find(root)]);
    // println!("cs: {:#?}", cs);
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
    let best = apply_libs_pareto(
      aeg.clone(),
      roots,
      &chosen_rewrites,
      self.lang_cost.clone(),
      self.lang_gain.clone(),
      self.config.strategy,
    );

    let delay_cost = DelayCost::new(self.lang_gain.clone()).cost_rec(&best);
    let selected_delay_cost = match delay_cost.2 {
      true => delay_cost.0,
      false => delay_cost.1,
    };

    let area_cost = AreaCost::new(self.lang_cost.clone()).cost_rec(&best);
    let selected_area_cost = match area_cost.2 {
      true => area_cost.0,
      false => area_cost.1,
    };
    let fin_cost = self.config.strategy * (selected_delay_cost as f32)
      + (1.0 - self.config.strategy) * (selected_area_cost as f32);

    // Lifting the lib will result in incorrect cost
    // let lifted = extract::lift_libs(&best);

    info!("Finished in {}ms", ex_time.elapsed().as_millis());
    debug!("final cost: {}", fin_cost);
    debug!("{}", Pretty::new(Arc::new(Expr::from(best.clone()))));
    info!("round time: {}ms", start_time.elapsed().as_millis());

    ParetoResult {
      final_expr: best.into(),
      num_libs: chosen_rewrites.len(),
      rewrites: rewrites_map,
      initial_cost: (init_area, init_delay, init_cost),
      final_cost: (selected_area_cost, selected_delay_cost, fin_cost),
      run_time: start_time.elapsed(),
    }
  }
}

impl<Op, T, LA, LD> BabbleParetoRunner<Op, T, LA, LD>
  for ParetoRunner<Op, T, LA, LD>
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
    + 'static,
  LA: LangCost<Op> + Clone + Default + 'static,
  LD: LangGain<Op> + Clone + Default + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  fn run(&self, expr: Expr<Op>) -> ParetoResult<Op, T, LA, LD> {
    self.run_multi(vec![expr])
  }

  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> ParetoResult<Op, T, LA, LD> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexprs: Vec<RecExpr<AstNode<Op>>> =
      exprs.into_iter().map(RecExpr::from).collect();
    let mut egraph = EGraph::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.lang_cost.clone(),
      self.lang_gain.clone(),
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
  ) -> ParetoResult<Op, T, LA, LD> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexpr_groups: Vec<Vec<_>> = expr_groups
      .into_iter()
      .map(|group| group.into_iter().map(RecExpr::from).collect())
      .collect();

    let mut egraph = EGraph::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.lang_cost.clone(),
      self.lang_gain.clone(),
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
impl<Op, T, LA, LD> std::fmt::Debug for ParetoRunner<Op, T, LA, LD>
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
  LA: LangCost<Op> + Clone + Default,
  LD: LangGain<Op> + Clone + Default,
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
