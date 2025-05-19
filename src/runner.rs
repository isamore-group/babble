//! Runner module for executing library learning experiments
//!
//! This module provides functionality for running library learning experiments
//! using either regular beam search or Pareto-optimal beam search.

use serde_json::json;
use std::{
  collections::{HashMap, HashSet},
  fmt::{Debug, Display},
  fs::File,
  hash::Hash,
  io::{self, Write},
  path::{Path, PathBuf},
  str::FromStr,
  sync::Arc,
  time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;

use crate::{
  Arity, AstNode, DiscriminantEq, Expr, LearnedLibraryBuilder, Pretty,
  Printable, Teachable,
  bb_query::{BBInfo, BBQuery},
  expand::{OpPackConfig, expand},
  extract::{
    self,
    beam_pareto::{ClassMatch, ISAXAnalysis, LibExtractor, TypeInfo, TypeSet},
  },
  rewrites::TypeMatch,
  schedule::Schedulable,
  vectorize::{VectorCF, VectorConfig, vectorize},
};
use egg::{
  AstSize, EGraph, Extractor, Id, Language, RecExpr, Rewrite,
  Runner as EggRunner,
};
use log::{debug, info};

/// 定义一个trait名为OperationInfo,其中有两个函数，分别为get_libid,
/// change_libid, get_const
pub trait OperationInfo {
  fn is_lib(&self) -> bool;
  /// Get the library ID of the operation
  fn get_libid(&self) -> usize;
  /// make a list op
  fn make_vec(result_tys: Vec<String>) -> Self;
  /// make a get op
  fn make_get(id: usize, result_tys: Vec<String>) -> Self;
  /// Get the constant value of the operation
  fn get_const(&self) -> Option<(i64, u32)>;
  /// Make an AST node for type-awared Const
  fn make_const(const_value: (i64, u32)) -> Self;
  /// judge whether a node is dummy
  fn is_dummy(&self) -> bool;
  /// judge whether a node is a vector op
  fn is_vector_op(&self) -> bool;
  /// get_simple_cost
  fn get_simple_cost(&self) -> f64;
  /// For List and Vec, though args may have different order , but they are the
  /// sam, this function is used to judge
  fn is_vec(&self) -> bool {
    false
  }
  /// make gather node
  fn make_gather(indices: &Vec<usize>) -> Self;
  /// make shuffle node
  fn make_shuffle(indices: &Vec<usize>) -> Self;
  /// get_bbs_info: will be used in vectorization
  fn get_bbs_info(&self) -> Vec<String> {
    vec![]
  }
  /// 获得当前操作符的类型
  fn get_result_type(&self) -> Vec<String> {
    vec![]
  }
  /// 为每个Operation设置返回类型
  fn set_result_type(&mut self, result_ty: Vec<String>) {
    // do nothing
  }
  /// get_vec_len
  fn get_vec_len(&self) -> usize {
    1
  }
  /// 加入Op_pack节点
  fn make_op_pack(ops: Vec<String>) -> Self;
  /// 加入Op_select节点
  fn make_op_select(idx: usize) -> Self;
  /// 加入rule_var节点
  fn make_rule_var(name: String) -> Self;
  /// 加入Opmask节点
  fn make_opmask() -> Self;
  /// 是不是Opmask节点
  fn is_opmask(&self) -> bool {
    false
  }
  /// 是不是RuleVar节点
  fn is_rule_var(&self) -> bool {
    false
  }
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
    + 'static
    + OperationInfo,
  T: Debug,
{
  /// The final expression after library learning and application
  pub final_expr: Expr<Op>,
  /// The number of libraries learned
  pub num_libs: usize,
  /// The rewrites representing the learned libraries
  pub rewrites_with_conditon:
    HashMap<usize, (Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, TypeMatch<T>)>,
  /// The final cost of the expression
  pub final_cost: f32,
  /// The time taken to run the experiment
  pub run_time: Duration,
  /// the learned lib
  pub learned_lib: Vec<(usize, egg::Pattern<AstNode<Op>>)>,
  /// message map to record the message
  pub message: HashMap<String, String>,
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
    + 'static
    + OperationInfo,
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
#[derive(Debug, Clone)]
pub struct ParetoConfig<LA, LD>
where
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
{
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
  /// area estimator
  pub area_estimator: LA,
  /// delay estimator
  pub delay_estimator: LD,
  /// whether to add all types
  pub add_all_types: bool,
  /// liblearn config
  pub liblearn_config: LiblearnConfig,
  /// vectorize config
  pub vectorize_config: VectorConfig,
  /// op_pack config
  pub op_pack_config: OpPackConfig,
}

impl<LA, LD> Default for ParetoConfig<LA, LD>
where
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
{
  fn default() -> Self {
    Self {
      final_beams: 10,
      inter_beams: 5,
      lib_iter_limit: 3,
      lps: 1,
      learn_constants: false,
      max_arity: None,
      strategy: 0.8,
      clock_period: 1000,
      area_estimator: LA::default(),
      delay_estimator: LD::default(),
      add_all_types: false,
      liblearn_config: LiblearnConfig::default(),
      vectorize_config: VectorConfig::default(),
      op_pack_config: OpPackConfig::default(),
    }
  }
}

/// Config for Learning Library
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LiblearnCost {
  /// type of cost : "delay", "Match", "size"
  #[serde(rename = "match")]
  Match,
  Delay,
  Size,
}
impl Default for LiblearnCost {
  fn default() -> Self {
    Self::Delay
  }
}
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AUMergeMod {
  /// type of AU merge : "random", "kd", "greedy", "cartesian"
  Random,
  Kd,
  Greedy,
  #[serde(rename = "cartesian")]
  Cartesian,
}
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnumMode {
  /// enumerate mode: "all", "pruning vanilla", "pruning gold", "cluster test"
  All,
  PruningVanilla,
  PruningGold,
  ClusterTest,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LiblearnConfig {
  /// type of cost : "delay", "Match", "size"
  pub cost: LiblearnCost,
  /// type of AU merge : "random", "kd", "greedy"
  pub au_merge_mod: AUMergeMod,
  /// enumerate mode: "all", "pruning vanilla", "pruning gold"
  pub enum_mode: EnumMode,
}

impl Default for LiblearnConfig {
  fn default() -> Self {
    Self {
      cost: LiblearnCost::Delay,
      au_merge_mod: AUMergeMod::Greedy,
      enum_mode: EnumMode::All,
    }
  }
}

impl LiblearnConfig {
  /// Create a new LiblearnConfig with the given parameters
  pub fn new(
    cost: LiblearnCost,
    au_merge_mod: AUMergeMod,
    enum_mode: EnumMode,
  ) -> Self {
    Self {
      cost,
      au_merge_mod,
      enum_mode,
    }
  }
}

/// A Pareto Runner that uses Pareto optimization with beam search
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
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  TypeSet<T>: ClassMatch,
  AstNode<Op>: TypeInfo<T>,
{
  /// The domain-specific rewrites to apply
  dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// BB query
  bb_query: BBQuery,
  /// lib rewrites
  lib_rewrites_with_condition:
    HashMap<usize, (Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, TypeMatch<T>)>,
  /// Configuration for the beam search
  config: ParetoConfig<LA, LD>,
  /// lift_dsrs
  lift_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// transform_dsrs
  transform_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
}

impl<Op, T, LA, LD> ParetoRunner<Op, T, LA, LD>
where
  Op: Arity
    + OperationInfo
    + BBInfo
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
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display
    + FromStr,
  TypeSet<T>: ClassMatch,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  /// Create a new BeamRunner with the given domain-specific rewrites and
  /// configuration
  pub fn new<I>(
    dsrs: I,
    bb_query: BBQuery,
    lib_rewrites_with_condition: HashMap<
      usize,
      (Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, TypeMatch<T>),
    >,
    config: ParetoConfig<LA, LD>,
  ) -> Self
  where
    I: IntoIterator<Item = Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  {
    // 如果USE_RULES为false，将dsrs清空
    let dsrs = dsrs.into_iter().collect();
    Self {
      dsrs,
      bb_query,
      lib_rewrites_with_condition,
      config,
      lift_dsrs: vec![],
      transform_dsrs: vec![],
    }
  }

  pub fn with_vectorize_dsrs(
    &mut self,
    lift_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
    transform_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  ) {
    self.lift_dsrs = lift_dsrs;
    self.transform_dsrs = transform_dsrs;
  }

  /// Run the e-graph and library learning process
  fn run_egraph(
    &self,
    roots: &[Id],
    egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> ParetoResult<Op, T> {
    let mut message = HashMap::new();
    let timeout = Duration::from_secs(60 * 100_000);
    // let mut egraph = egraph.clone();
    // let root = egraph.add(AstNode::new(Op::list(), roots.iter().copied()));

    println!(
      "Initial egraph size: {}, eclasses: {}",
      egraph.total_size(),
      egraph.classes().len()
    );
    message.insert(
      "initial_eclass_size".to_string(),
      format!("{}", egraph.classes().len()).to_string(),
    );
    println!("Running {} DSRs... ", self.dsrs.len());
    let start_time = Instant::now();
    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::empty())
      .with_egraph(egraph)
      .with_time_limit(timeout)
      .with_iter_limit(3)
      .run(&self.dsrs);

    let mut aeg = runner.egraph;
    crate::perf_infer::perf_infer(&mut aeg, roots, vec![]);
    // for ecls in aeg.classes() {
    //   println!("ecls_bb: {:?}", ecls.data.bb);
    // }

    let mut root_vec = roots.to_vec();

    // aeg.dot().to_png("target/initial_egraph.png").unwrap();
    if self.config.op_pack_config.pack_expand {
      // 进行expand
      let select_expand_time = Instant::now();
      aeg = expand(aeg.clone(), self.config.clone());
      println!(
        "Expand finished in {}ms",
        select_expand_time.elapsed().as_millis()
      );

      message.insert(
        "select_expand_time".to_string(),
        format!("{}ms", select_expand_time.elapsed().as_millis()).to_string(),
      );
    } else if self.config.vectorize_config.vectorize {
      let vectorize_time = Instant::now();
      let mut vectorized_egraph = vectorize(
        aeg.clone(),
        roots,
        &self.lift_dsrs,
        &self.transform_dsrs,
        self.config.clone(),
      );

      let new_root =
        vectorized_egraph.add(AstNode::new(Op::list(), roots.iter().copied()));

      let (cost, expr) =
        Extractor::new(&vectorized_egraph, VectorCF).find_best(new_root);
      debug!("cost: {:?}", cost);

      // 新建一个EGraph，将expr加入
      let mut new_egraph = EGraph::new(ISAXAnalysis::new(
        self.config.final_beams,
        self.config.inter_beams,
        self.config.lps,
        self.config.strategy,
        self.bb_query.clone(),
      ));

      root_vec = vec![new_egraph.add_expr(&expr)];

      // 使用部分transform_dsr进行重写
      let mut new_runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::empty())
        .with_egraph(new_egraph)
        .with_time_limit(timeout)
        .with_iter_limit(3)
        .run(&self.transform_dsrs);

      aeg = new_runner.egraph;
      // let pretty_expr = Pretty::new(Arc::new(Expr::from(expr.clone())));
      // debug!("pretty expr: {}", pretty_expr);
      // println!("Expression: ");
      let mut vecop_cnt = 0;
      for (id, node) in expr.iter().enumerate() {
        debug!("{}: {:?}", id, node);
        if node.operation().is_vector_op() {
          vecop_cnt += 1;
        }
      }

      println!("Vectorized Nodes: {}", vecop_cnt);
      println!(
        "Vectorized egraph size: {}, eclasses: {}",
        vectorized_egraph.total_size(),
        vectorized_egraph.classes().len()
      );
      message.insert(
        "vectorized_eclass_size".to_string(),
        format!("{}", vectorized_egraph.classes().len()).to_string(),
      );
      message.insert(
        "vectorized_egraph_size".to_string(),
        format!("{}", vectorized_egraph.total_size()).to_string(),
      );
      message.insert(
        "vectorized_time".to_string(),
        format!("{}ms", vectorize_time.elapsed().as_millis()).to_string(),
      );
    }

    println!(
      "Final egraph size: {}, eclasses: {}",
      aeg.total_size(),
      aeg.classes().len()
    );

    message.insert(
      "final_eclass_size".to_string(),
      format!("{}", aeg.classes().len()).to_string(),
    );

    message.insert(
      "running_dsr_time".to_string(),
      format!("{}ms", start_time.elapsed().as_millis()).to_string(),
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
    max_lib_id = if max_lib_id == 0 { 0 } else { max_lib_id + 1 };
    let learned_lib = LearnedLibraryBuilder::default()
      .learn_constants(self.config.learn_constants)
      .max_arity(self.config.max_arity)
      // .with_co_occurs(co_occurs)
      .with_last_lib_id(max_lib_id)
      .with_liblearn_config(self.config.liblearn_config.clone())
      .with_clock_period(self.config.clock_period)
      .with_area_estimator(self.config.area_estimator.clone())
      .with_delay_estimator(self.config.delay_estimator.clone())
      .with_bb_query(self.bb_query.clone())
      .build(&aeg);
    println!(
      "Found {} patterns in {}ms",
      learned_lib.size(),
      au_time.elapsed().as_millis()
    );

    message.insert(
      "au_time".to_string(),
      format!("{}ms", au_time.elapsed().as_millis()).to_string(),
    );

    // info!("Deduplicating patterns... ");
    let dedup_time = Instant::now();
    // learned_lib.deduplicate(&aeg);
    let lib_rewrites: Vec<_> = learned_lib.rewrites().map(|(r, _)| r).collect();
    let rewrite_conditions: Vec<_> = learned_lib.conditions().collect();
    info!(
      "Reduced to {} patterns in {}ms",
      learned_lib.size(),
      dedup_time.elapsed().as_millis()
    );

    message.insert(
      "dedup_time".to_string(),
      format!("{}ms", dedup_time.elapsed().as_millis()).to_string(),
    );

    println!("learned {} libs", learned_lib.size());

    message.insert(
      "learned_libs".to_string(),
      format!("{}", learned_lib.size()).to_string(),
    );

    println!("Adding libs and running beam search... ");
    let extract_time = Instant::now();
    let lib_rewrite_time = Instant::now();
    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.config.strategy,
      self.bb_query.clone(),
    ))
    .with_egraph(aeg.clone())
    .with_iter_limit(self.config.lib_iter_limit)
    .with_time_limit(timeout)
    .with_node_limit(1_000_000)
    .run(lib_rewrites.iter());

    let mut egraph = runner.egraph;
    // egraph.dot().to_png("target/final_egraph.png").unwrap();
    let root = egraph.add(AstNode::new(Op::list(), root_vec.iter().copied()));
    let mut isax_cost = egraph[egraph.find(root)].data.clone();
    // println!("root: {:#?}", egraph[egraph.find(root)]);
    // println!("cs: {:#?}", cs);
    // println!("cs: {:#?}", isax_cost.cs.set[0]);
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
    println!("cs: {:#?}", isax_cost.cs);
    let mut chosen_rewrites = Vec::new();
    let mut learned_libs = Vec::new();
    let mut rewrites_map = HashMap::new();
    for lib in &isax_cost.cs.set[0].libs {
      if lib.0.0 < max_lib_id {
        // 从self.lib_rewrites中取出
        // 打印self.lib_rewrites
        // println!("{}: {:?}", lib.0.0, self.lib_rewrites_with_condition);
        chosen_rewrites.push(
          self
            .lib_rewrites_with_condition
            .get(&lib.0.0)
            .unwrap()
            .0
            .clone(),
        );
        rewrites_map.insert(
          lib.0.0,
          self
            .lib_rewrites_with_condition
            .get(&lib.0.0)
            .unwrap()
            .clone(),
        );
      } else {
        let new_lib = lib.0.0 - max_lib_id;
        chosen_rewrites.push(lib_rewrites[new_lib].clone());
        learned_libs.push((
          lib.0.0,
          learned_lib.libs().collect::<Vec<_>>()[new_lib].clone(),
        ));
        rewrites_map.insert(
          lib.0.0,
          (
            lib_rewrites[new_lib].clone(),
            rewrite_conditions[new_lib].clone(),
          ),
        );
      }
    }

    debug!(
      "upper bound ('full') cost: {}",
      isax_cost.cs.set[0].full_cost
    );

    // for (id, rewrite) in lib_rewrites.clone().iter().enumerate() {
    //   println!("lib{}", id);
    //   let mut egraph = EggRunner::<_, _, ()>::new(ISAXAnalysis::default())
    //     .with_egraph(aeg.clone())
    //     .with_iter_limit(1)
    //     .with_time_limit(timeout)
    //     .with_node_limit(1_000_000)
    //     .run(&[rewrite.clone()])
    //     .egraph;
    //   let root = egraph.add(AstNode::new(Op::list(), roots.iter().copied()));
    //   let mut extractor =
    //     LibExtractor::new(&egraph, self.config.strategy);
    //   let best = extractor.best(root);
    //   let final_cost = extractor.cost(&best);
    //   println!("final cost: {}", final_cost);
    // }

    let ex_time = Instant::now();
    println!("Rewriting");
    let mut egraph = EggRunner::<_, _, ()>::new(ISAXAnalysis::default())
      .with_egraph(aeg)
      .with_iter_limit(1)
      .with_time_limit(timeout)
      .with_node_limit(1_000_000)
      .run(chosen_rewrites.iter())
      .egraph;
    println!("Rewriting done");
    let root = egraph.add(AstNode::new(Op::list(), root_vec.iter().copied()));

    crate::perf_infer::perf_infer(&mut egraph, roots, vec![]);

    let mut extractor =
      LibExtractor::new(&egraph, self.config.strategy, self.bb_query.clone());
    println!("Extracting");
    let best = extractor.best(root);
    let final_cost = extractor.cost(&best);
    println!("Extracting done");

    println!("extracting using {}s", extract_time.elapsed().as_secs());
    // Lifting the lib will result in incorrect cost
    // let lifted = extract::lift_libs(&best);

    info!("Finished in {}ms", ex_time.elapsed().as_millis());
    debug!("final cost: {}", final_cost);
    debug!("{}", Pretty::new(Arc::new(Expr::from(best.clone()))));
    info!("round time: {}ms", start_time.elapsed().as_millis());

    message.insert(
      "final_cost".to_string(),
      format!("{}", final_cost).to_string(),
    );

    message.insert(
      "extract_time".to_string(),
      format!("{}ms", ex_time.elapsed().as_millis()).to_string(),
    );

    ParetoResult {
      final_expr: best.into(),
      num_libs: chosen_rewrites.len(),
      rewrites_with_conditon: rewrites_map,
      final_cost: final_cost,
      run_time: start_time.elapsed(),
      learned_lib: learned_libs,
      message,
    }
  }
}

impl<Op, T, LA, LD> BabbleParetoRunner<Op, T> for ParetoRunner<Op, T, LA, LD>
where
  Op: Arity
    + OperationInfo
    + BBInfo
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
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display
    + FromStr,
  TypeSet<T>: ClassMatch,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
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
      self.bb_query.clone(),
    ));
    let roots = recexprs
      .iter()
      .map(|x| egraph.add_expr(x))
      .collect::<Vec<_>>();
    egraph.rebuild();

    if self.config.add_all_types {
      // 加入类型重写，相当于是给egraph中添加node并做合并
      let widths = vec![8 as u32, 16 as u32, 32 as u32, 64 as u32];
      let mut int_width: HashMap<i64, HashSet<u32>> = HashMap::new();
      let mut ecls_ids: HashMap<i64, HashSet<Id>> = HashMap::new();
      // 收集已经存在的位宽信息和eclass信息
      for ecls in egraph.classes() {
        for node in ecls.nodes.clone() {
          if let Some((a, aw)) = node.operation().get_const() {
            if !int_width.contains_key(&a) {
              int_width.insert(a, HashSet::new());
              ecls_ids.insert(a, HashSet::new());
            }
            let int_node = int_width.get_mut(&a).unwrap();
            int_node.insert(aw);
            let ecls_id = ecls_ids.get_mut(&a).unwrap();
            ecls_id.insert(ecls.id);
          }
        }
      }
      // 遍历width，如果已经存在的位宽不包含当前的位宽，则添加
      for width in widths.iter() {
        for (a, aw) in int_width.clone().iter() {
          if !aw.contains(width) {
            let new_node = AstNode::leaf(Op::make_const((*a, *width)));
            let new_ecls_id = egraph.add_uncanonical(new_node);
            // 将当前的eclass添加到ecls_ids中
            let ecls_id = ecls_ids.get_mut(a).unwrap();
            ecls_id.insert(new_ecls_id);
          }
        }
      }
      // 遍历ecls_ids，进行union
      for (_, ecls_id) in ecls_ids.iter() {
        let mut ids = Vec::new();
        for id in ecls_id.iter() {
          ids.push(*id);
        }
        if ids.len() > 1 {
          let first = ids[0];
          for i in 1..ids.len() {
            egraph.union(first, ids[i]);
          }
        }
      }
      egraph.rebuild();
    }

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
      self.bb_query.clone(),
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
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  TypeSet<T>: ClassMatch,
  AstNode<Op>: TypeInfo<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BeamRunner")
      .field("dsrs_count", &self.dsrs.len())
      .field("config", &self.config)
      .finish()
  }
}
