//! Runner module for executing library learning experiments
//! 
//! This module provides functionality for running library learning experiments
//! using either regular beam search or Pareto-optimal beam search.

use std::{
    fmt::{self, Debug, Display, Formatter},
    hash::Hash,
    time::{Duration, Instant},
    marker::PhantomData,
};

use egg::{AstSize, CostFunction, EGraph, Id, RecExpr, Rewrite, Runner as EggRunner};
use log::{debug, info};

use crate::{
    extract::{
        apply_libs, 
        apply_libs_pareto,
        apply_libs_area_delay,
        beam::PartialLibCost,
        beam::OptimizationStrategy,
        // beam_pareto::{BeamAreaDelay, LibSelAreaDelay, CostSetAreaDelay},
        cost::{LangCost, LangGain},
    },
    Arity, AstNode, COBuilder, DiscriminantEq, Expr, LearnedLibraryBuilder,
    Pretty, Printable, Teachable,
};

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
    fn run_equiv_groups(&self, expr_groups: Vec<Vec<Expr<Op>>>) -> BabbleResult<Op>;
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
pub struct BeamRunner<Op>
where
    Op: Display + Hash + Clone + Ord + Teachable + Arity + Send + Sync + 'static,
{
    /// The domain-specific rewrites to apply
    dsrs: Vec<Rewrite<AstNode<Op>, PartialLibCost>>,
    /// Configuration for the beam search
    config: BeamConfig,
}

impl<Op> BeamRunner<Op>
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
    /// Create a new BeamRunner with the given domain-specific rewrites and configuration
    pub fn new<I>(
        dsrs: I,
        config: BeamConfig,
    ) -> Self
    where
        I: IntoIterator<Item = Rewrite<AstNode<Op>, PartialLibCost>>,
    {
        Self {
            dsrs: dsrs.into_iter().collect(),
            config,
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
        let mut learned_lib = LearnedLibraryBuilder::make_with_egraph(aeg.clone())
            .learn_constants(self.config.learn_constants)
            .max_arity(self.config.max_arity)
            .with_co_occurs(co_occurs)
            .build(&aeg);
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
        cs.set.sort_unstable_by_key(|elem| elem.full_cost);

        info!("Finished in {}ms", lib_rewrite_time.elapsed().as_millis());
        info!("Stop reason: {:?}", runner.stop_reason.unwrap());
        info!("Number of nodes: {}", egraph.total_size());

        debug!("learned libs");
        let all_libs: Vec<_> = learned_lib.libs().collect();
        let mut chosen_rewrites = Vec::new();
        for lib in &cs.set[0].libs {
            debug!("{}: {}", lib.0, &all_libs[lib.0.0]);
            chosen_rewrites.push(lib_rewrites[lib.0.0].clone());
        }

        debug!("upper bound ('full') cost: {}", cs.set[0].full_cost);

        let ex_time = Instant::now();
        info!("Extracting... ");
        let lifted = apply_libs(aeg.clone(), roots, &chosen_rewrites);
        let final_cost = AstSize.cost_rec(&lifted);

        info!("Finished in {}ms", ex_time.elapsed().as_millis());
        info!("final cost: {}", final_cost);
        debug!("{}", Pretty(&Expr::from(lifted.clone())));
        info!("round time: {}ms", start_time.elapsed().as_millis());

        // Calculate initial cost
        let initial_cost = {
            let s: usize = roots.iter().map(|id| {
                let extractor = egg::Extractor::new(&egraph, AstSize);
                let (_, expr) = extractor.find_best(*id);
                AstSize.cost_rec(&expr)
            }).sum();
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

impl<Op> BabbleRunner<Op> for BeamRunner<Op>
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
    fn run(&self, expr: Expr<Op>) -> BabbleResult<Op> {
        self.run_multi(vec![expr])
    }

    fn run_multi(&self, exprs: Vec<Expr<Op>>) -> BabbleResult<Op> {
        // First, let's turn our list of exprs into a list of recexprs
        let recexprs: Vec<RecExpr<AstNode<Op>>> = exprs.into_iter().map(RecExpr::from).collect();

        let mut egraph = EGraph::new(PartialLibCost::new(
            self.config.final_beams,
            self.config.inter_beams,
            self.config.lps,
        ));
        let roots = recexprs.iter().map(|x| egraph.add_expr(x)).collect::<Vec<_>>();
        egraph.rebuild();

        self.run_egraph(&roots, egraph)
    }

    fn run_equiv_groups(&self, expr_groups: Vec<Vec<Expr<Op>>>) -> BabbleResult<Op> {
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
impl<Op> std::fmt::Debug for BeamRunner<Op>
where
    Op: Display + Hash + Clone + Ord + Teachable + Arity + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeamRunner")
            .field("dsrs_count", &self.dsrs.len())
            .field("config", &self.config)
            .finish()
    }
}

// /// Configuration for Pareto-optimal beam search
// #[derive(Debug, Clone, Copy)]
// pub struct ParetoConfig {
//     /// The final beam size to use
//     pub final_beams: usize,
//     /// The inter beam size to use
//     pub inter_beams: usize,
//     /// The number of times to apply library rewrites
//     pub lib_iter_limit: usize,
//     /// The number of libs to learn at a time
//     pub lps: usize,
//     /// Whether to learn "library functions" with no arguments
//     pub learn_constants: bool,
//     /// Maximum arity of a library function
//     pub max_arity: Option<usize>,
//     /// The optimization strategy to use
//     pub strategy: OptimizationStrategy,
// }

// impl Default for ParetoConfig {
//     fn default() -> Self {
//         Self {
//             final_beams: 10,
//             inter_beams: 5,
//             lib_iter_limit: 3,
//             lps: 5,
//             learn_constants: false,
//             max_arity: None,
//             strategy: OptimizationStrategy::Balanced(0.5),
//         }
//     }
// }
