//! The primary interface for library learning through antiunification.
//!
//! The antiunification algorithm is as follows: For each pair of eclasses (a,
//! b), we store a set of partial expressions AU(a, b). Partial expressions are
//! expressions where some of the sub-expressions may be replaced by another
//! data type. For example, a pattern is (equivalent to) a partial expression
//! over variables. In this algorithm, we use partial expressions over pairs of
//! enodes.
//!
//! To compute the set AU(a, b): For each pair of enodes with matching operators
//! and arities op(x1, ..., xn) \in a and op(y1, ..., yn) \in b, we recursively
//! compute AU(x1, y1), ..., AU(xn, yn). Then, for every n-tuple of partial
//! expressions (z1, ..., zn) with z1 \in AU(x1, y1), ..., zn \in AU(xn, yn), we
//! add the partial expression op(z1, ..., zn) to the set AU(a, b).
//! If the set AU(a, b) is empty, we add to it the partial expression (a, b).
// 使用随机数
use crate::runner::{AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost};
use crate::{
  COBuilder,
  ast_node::{Arity, AstNode, Expr, PartialExpr},
  co_occurrence::CoOccurrences,
  dfta::Dfta,
  extract::beam_pareto::{ClassMatch, EmptyAnalysis},
  schedule::{Schedulable, Scheduler},
  teachable::{BindingExpr, Teachable},
};
use bitvec::prelude::*;
use egg::{
  Analysis, EGraph, Id, Language, Pattern, RecExpr, Rewrite, Searcher, Var,
};
use itertools::Itertools;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::{
  collections::{BTreeMap, BTreeSet, HashMap},
  fmt::{Debug, Display},
  num::ParseIntError,
  str::FromStr,
};
use std::{hash::Hash, time::Instant, vec};
use thiserror::Error;

use crate::au_search::{get_random_aus, greedy_aus, kd_random_aus};

use rayon::prelude::*;

/// A library function's name.
#[derive(
  Debug,
  Clone,
  Copy,
  PartialEq,
  Eq,
  PartialOrd,
  Ord,
  Hash,
  Serialize,
  Deserialize,
)]
pub struct LibId(pub usize);

impl Display for LibId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "l{}", self.0)
  }
}

/// An error when parsing a [`LibId`].
#[derive(Clone, Debug, Error)]
pub enum ParseLibIdError {
  /// The string did not start with "$"
  #[error("expected de Bruijn index to start with 'l")]
  NoLeadingL,
  /// The index is not a valid unsigned integer
  #[error(transparent)]
  InvalidIndex(ParseIntError),
}

impl FromStr for LibId {
  type Err = ParseLibIdError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if let Some(n) = s.strip_prefix('l') {
      let n = n.parse().map_err(ParseLibIdError::InvalidIndex)?;
      Ok(LibId(n))
    } else {
      Err(ParseLibIdError::NoLeadingL)
    }
  }
}

/// Signature of a pattern match in an e-graph.
/// Used to deduplicate equivalent patterns.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub struct Match {
  /// The e-class that the match is found in.
  class: Id,
  /// The range of the match's substitution
  /// (a multiset of class ids, stores as a sorted vector).
  actuals: Vec<Id>,
}

impl Match {
  pub fn new(class: Id, mut actuals: Vec<Id>) -> Self {
    actuals.sort();
    Self { class, actuals }
  }
}
// 用来保存AU和其对应的Match
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct AU<Op, T> {
  /// The anti-unification
  expr: PartialExpr<Op, T>,
  /// The matches
  matches: Vec<Match>,
  /// delay
  delay: usize,
  /// strategy for sort
  liblearn_cost: LiblearnCost,
}

impl<Op, T> AU<Op, T> {
  pub fn new(
    expr: PartialExpr<Op, T>,
    matches: Vec<Match>,
    delay: usize,
    liblearn_cost: LiblearnCost,
  ) -> Self {
    Self {
      expr,
      matches,
      delay,
      liblearn_cost,
    }
  }
  pub fn new_with_expr(
    expr: PartialExpr<Op, T>,
    egraph: &EGraph<AstNode<Op>, EmptyAnalysis<Op>>,
    liblearn_cost: LiblearnCost,
  ) -> Self
  where
    Op: Clone
      + Arity
      + Debug
      + Default
      + Display
      + Ord
      + Send
      + Sync
      + Teachable
      + 'static
      + Hash
      + Schedulable,
    AstNode<Op>: Language,
    T: Clone + Debug + Hash + Ord + Default,
  {
    let matches = expr.clone().get_match(egraph);
    let delay = expr.get_delay();
    Self {
      expr,
      matches,
      delay,
      liblearn_cost,
    }
  }

  pub fn expr(&self) -> &PartialExpr<Op, T> {
    &self.expr
  }

  pub fn matches(&self) -> &Vec<Match> {
    &self.matches
  }

  pub fn delay(&self) -> usize {
    self.delay
  }

  pub fn liblearn_cost(&self) -> LiblearnCost {
    self.liblearn_cost
  }
}
// 为AU实现Ord，只对比matches的大小
impl<Op: Eq, T: Eq> PartialOrd for AU<Op, T> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    // Some(self.matches.cmp(&other.matches))
    // Some(self.expr.size().cmp(&other.expr.size()))
    // Some(self.delay.cmp(&other.delay))
    match self.liblearn_cost {
      LiblearnCost::Match => Some(self.matches.len().cmp(&other.matches.len())),
      LiblearnCost::Size => Some(self.expr.size().cmp(&other.expr.size())),
      LiblearnCost::Delay => Some(self.delay.cmp(&other.delay)),
    }
  }
}

impl<Op: Eq, T: Eq> Ord for AU<Op, T> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // self.matches.cmp(&other.matches)
    // self.expr.size().cmp(&other.expr.size())
    // self.delay.cmp(&other.delay)
    match self.liblearn_cost {
      LiblearnCost::Match => self.matches.len().cmp(&other.matches.len()),
      LiblearnCost::Size => self.expr.size().cmp(&other.expr.size()),
      LiblearnCost::Delay => self.delay.cmp(&other.delay),
    }
  }
}

#[derive(Clone, Debug)]
pub struct LearnedLibraryBuilder<Op>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + Schedulable,
{
  egraph: EGraph<AstNode<Op>, EmptyAnalysis<Op>>,
  learn_trivial: bool,
  learn_constants: bool,
  max_arity: Option<usize>,
  banned_ops: Vec<Op>,
  roots: Vec<Id>,
  co_occurences: Option<CoOccurrences>,
  dfta: bool,
  last_lib_id: usize,
  clock_period: usize,
  liblearn_config: LiblearnConfig,
}

impl<Op> Default for LearnedLibraryBuilder<Op>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + Schedulable,
{
  fn default() -> Self {
    Self {
      egraph: EGraph::default(),
      learn_trivial: false,
      learn_constants: false,
      max_arity: None,
      banned_ops: vec![],
      roots: vec![],
      co_occurences: None,
      dfta: true,
      last_lib_id: 0,
      clock_period: 3,
      liblearn_config: LiblearnConfig::default(),
    }
  }
}

// impl<Op: Ord+ Debug + Clone + Hash, A: egg::Analysis<AstNode<Op>>> Default
// for LearnedLibraryBuilder<Op, A> {   fn default() -> Self {
//     Self {
//       egraph: EGraph::new(PartialLibCost::default()),
//       learn_trivial: false,
//       learn_constants: false,
//       max_arity: None,
//       banned_ops: vec![],
//       roots: vec![],
//       co_occurences: None,
//       dfta: true,
//     }
//   }
// }

// 为 LearnedLibraryBuilder 实现自定义的构造函数make_with_egraph
impl<Op> LearnedLibraryBuilder<Op>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + std::hash::Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + Schedulable,
  AstNode<Op>: Language,
{
  pub fn make_with_egraph_ld(
    egraph: EGraph<AstNode<Op>, EmptyAnalysis<Op>>,
  ) -> Self {
    Self {
      egraph,
      learn_trivial: false,
      learn_constants: false,
      max_arity: None,
      banned_ops: vec![],
      roots: vec![],
      co_occurences: None,
      dfta: true,
      last_lib_id: 0,
      clock_period: 3,
      liblearn_config: LiblearnConfig::default(),
    }
  }
}

impl<Op> LearnedLibraryBuilder<Op>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + Sync
    + Send
    + Display
    + std::hash::Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + Schedulable,
  AstNode<Op>: Language,
{
  #[must_use]
  pub fn learn_trivial(mut self, trivial: bool) -> Self {
    self.learn_trivial = trivial;
    self
  }

  #[must_use]
  pub fn learn_constants(mut self, constants: bool) -> Self {
    self.learn_constants = constants;
    self
  }

  #[must_use]
  pub fn max_arity(mut self, arity: Option<usize>) -> Self {
    self.max_arity = arity;
    self
  }

  #[must_use]
  pub fn ban_op(mut self, op: Op) -> Self {
    self.banned_ops.push(op);
    self
  }

  #[must_use]
  pub fn ban_ops(mut self, iter: impl IntoIterator<Item = Op>) -> Self {
    self.banned_ops.extend(iter);
    self
  }

  #[must_use]
  pub fn with_roots(mut self, roots: Vec<Id>) -> Self {
    self.roots = roots;
    self
  }

  #[must_use]
  pub fn with_co_occurs(mut self, co_occurrences: CoOccurrences) -> Self {
    self.co_occurences = Some(co_occurrences);
    self
  }

  #[must_use]
  pub fn with_dfta(mut self, dfta: bool) -> Self {
    self.dfta = dfta;
    self
  }

  #[must_use]
  pub fn with_last_lib_id(mut self, last_lib_id: usize) -> Self {
    self.last_lib_id = last_lib_id;
    self
  }

  #[must_use]
  pub fn with_clock_period(mut self, clock_period: usize) -> Self {
    self.clock_period = clock_period;
    self
  }

  #[must_use]
  pub fn with_liblearn_config(
    mut self,
    liblearn_config: LiblearnConfig,
  ) -> Self {
    self.liblearn_config = liblearn_config;
    self
  }

  pub fn build<A>(
    self,
    egraph: &EGraph<AstNode<Op>, A>,
  ) -> LearnedLibrary<Op, (Id, Id)>
  where
    A: Analysis<AstNode<Op>> + Clone + Sync + Send + 'static,
    <A as Analysis<AstNode<Op>>>::Data: ClassMatch + Sync + Send,
    AstNode<Op>: Language,
    <AstNode<Op> as Language>::Discriminant: Sync + Send,
    Op: crate::ast_node::Printable,
  {
    let roots = &self.roots;
    debug!("Computing co-occurences");
    let co_occurs = self.co_occurences.unwrap_or_else(|| {
      let co_ext = COBuilder::new(egraph, roots);
      co_ext.run()
    });

    debug!("Constructing learned libraries");
    LearnedLibrary::new(
      egraph,
      self.egraph,
      self.learn_trivial,
      self.learn_constants,
      self.max_arity,
      self.banned_ops,
      co_occurs,
      self.dfta,
      self.last_lib_id,
      self.clock_period,
      self.liblearn_config,
    )
  }
}

pub trait DiscriminantEq {
  fn discriminant_eq(&self, other: &Self) -> bool;
}

/// A `LearnedLibrary<Op>` is a collection of functions learned from an
/// [`EGraph<AstNode<Op>, _>`] by antiunifying pairs of enodes to find their
/// common structure.
///
/// You can create a `LearnedLibrary` using
/// [`LearnedLibrary::from(&your_egraph)`].
#[derive(Debug, Clone)]
pub struct LearnedLibrary<Op, T>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + DiscriminantEq
    + Hash
    + 'static
    + Teachable,
{
  egraph: EGraph<AstNode<Op>, EmptyAnalysis<Op>>,
  /// A map from DFTA states (i.e. pairs of enodes) to their antiunifications.
  aus_by_state: BTreeMap<T, BTreeSet<AU<Op, T>>>,
  /// A set of all the antiunifications discovered.
  aus: BTreeSet<AU<Op, Var>>,
  /// Whether to learn "trivial" anti-unifications.
  learn_trivial: bool,
  /// Whether to also learn "library functions" which take no arguments.
  learn_constants: bool,
  /// Maximum arity of functions to learn.
  max_arity: Option<usize>,
  /// Operations that must never appear in learned abstractions.
  banned_ops: Vec<Op>,
  /// Data about which e-classes can co-occur.
  co_occurrences: CoOccurrences,
  // /// 存储deduplicate_from_candidates中cache已经存储过的值
  // pattern_cache: HashMap<PartialExpr<Op, (Id, Id)>, Vec<Match>>,
  last_lib_id: usize,
  /// clock period used in scheduling
  clock_period: usize,
  /// config for liblearn
  liblearn_config: LiblearnConfig,
}

#[allow(unused)]
fn mem_usage_of<F, X, L>(label: L, f: F) -> (X, usize)
where
  F: FnOnce() -> X,
  L: std::fmt::Display,
{
  let before_mem = memory_stats::memory_stats().unwrap().physical_mem;
  let x = f();
  let after_mem = memory_stats::memory_stats().unwrap().physical_mem;
  // println!("{label} {}mb", ((after_mem - before_mem) as f64) / 1e6);
  (x, after_mem - before_mem)
}

impl<'a, Op> LearnedLibrary<Op, (Id, Id)>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + Sync
    + Send
    + Display
    + DiscriminantEq
    + std::hash::Hash
    + Teachable
    + Schedulable
    + 'static,
  AstNode<Op>: Language,
{
  /// Constructs a [`LearnedLibrary`] from an [`EGraph`] by antiunifying pairs
  /// of enodes to find their common structure.
  fn new<A: Analysis<AstNode<Op>> + Clone>(
    egraph: &'a EGraph<AstNode<Op>, A>,
    my_egraph: EGraph<AstNode<Op>, EmptyAnalysis<Op>>,
    learn_trivial: bool,
    learn_constants: bool,
    max_arity: Option<usize>,
    banned_ops: Vec<Op>,
    co_occurrences: CoOccurrences,
    dfta: bool,
    last_lib_id: usize,
    clock_period: usize,
    liblearn_config: LiblearnConfig,
  ) -> Self
  where
    <A as Analysis<AstNode<Op>>>::Data: ClassMatch + Sync + Send,
    Op: crate::ast_node::Printable,
    A: Sync + Send + 'static,
    <AstNode<Op> as Language>::Discriminant: Sync + Send,
  {
    let mut learned_lib = Self {
      egraph: my_egraph,
      aus_by_state: BTreeMap::new(),
      aus: BTreeSet::new(),
      learn_trivial,
      learn_constants,
      max_arity,
      banned_ops,
      co_occurrences,
      last_lib_id,
      clock_period,
      liblearn_config: liblearn_config.clone(),
    };

    if !dfta {
      let dfta = Dfta::from(egraph);
      let dfta = dfta.cross_over();
      debug!("crossed over dfta");
      // println!("there are {} output states",
      // dfta.output_states().cloned().count());

      for &state in dfta.output_states() {
        learned_lib.enumerate_over_dfta(&dfta, state);
      }
    } else {
      let classes: Vec<_> = egraph.classes().map(|cls| cls.id).collect();

      fn hamming_distance(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
      }

      fn jaccard_similarity(a: &u64, b: &u64) -> f64 {
        let intersection = (a.clone() & b.clone()).count_ones() as f64;
        let union = (a.clone() | b.clone()).count_ones() as f64;
        if union == 0.0 {
          1.0 // 完全相同
        } else {
          intersection / union
        }
      }

      fn type_match(a: &String, b: &String) -> bool {
        // FIXME: 判断类型的时候，如果有一方是unknown，那么按理来说应该也能匹配
        // if a == "unknown" || b == "unknown" {
        //   return true;
        // }
        a == b
      }

      fn level_match(a: &(u64, u64), b: &(u64, u64)) -> bool {
        let hash_similar = hamming_distance(a.0, b.0) < 36;
        let subtree_similar = jaccard_similarity(&a.1, &b.1) > 0.67;
        hash_similar && subtree_similar
      }

      fn group_by_type_ranges(
        class_data: &[(&Id, String, u64, u64)],
      ) -> Vec<(usize, usize)> {
        let mut ranges = vec![];
        let mut i = 0;
        while i < class_data.len() {
          let current_ty = &class_data[i].1;
          let ty_end = class_data[i..]
            .iter()
            .position(|(_, ty, _, _)| ty != current_ty)
            .map(|pos| i + pos)
            .unwrap_or(class_data.len());
          ranges.push((i, ty_end));
          i = ty_end;
        }
        ranges
      }
      // 初始化AU计算日志
      let file = match liblearn_config.init_log() {
        Ok(file) => file,
        Err(_) => {
          // 创建必要的目录
          if let Some(parent) =
            std::path::Path::new(&liblearn_config.log_file).parent()
          {
            if !parent.exists() {
              if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Unable to create log directory: {}", e);
              }
            }
          }

          match std::fs::File::create(&liblearn_config.log_file) {
            Ok(file) => file,
            Err(e) => {
              eprintln!("Unable to create log file {}", e);
              panic!("Failed to create log file");
            }
          }
        }
      };
      let mut file = file;
      match liblearn_config.enum_mode {
        EnumMode::All => {
          // 计算所有pair
          let eclass_pairs = classes
            .iter()
            .cartesian_product(classes.iter())
            .map(|(ecls1, ecls2)| (egraph.find(*ecls1), egraph.find(*ecls2)))
            .collect::<Vec<_>>();
          let start = Instant::now();
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph(egraph, pair);
          }
          let elapsed = start.elapsed();
          let _ = liblearn_config.write_log(
            &mut file,
            "Not need to calculate pruned pairs, so use 0s",
          );
          let _ = liblearn_config.write_log(
            &mut file,
            format!("Enumerate over dfta takes {:?}", elapsed).as_str(),
          );
        }
        EnumMode::PruningVanilla => {
          // 无剪枝优化+并行处理+模块分组，仅有针对pair的筛选
          let mut eclass_pairs = vec![];
          let mut class_data: Vec<_> = classes
            .iter()
            .map(|cls| {
              let ty = egraph[*cls].data.get_type();
              let cls_hash = egraph[*cls].data.get_cls_hash();
              let subtree_levels = egraph[*cls].data.get_subtree_levels();
              (cls, ty, cls_hash, subtree_levels)
            })
            .collect();
          class_data
            .sort_unstable_by_key(|(ecls, _, _, _)| usize::from(**ecls));
          // 用一个二维数组来存储pairs的配对情况
          for i in 0..class_data.len() {
            for j in i..class_data.len() {
              let (ecls1, ty1, cls_hash1, subtree_levels1) =
                class_data[i].clone();
              let (ecls2, ty2, cls_hash2, subtree_levels2) =
                class_data[j].clone();
              if !level_match(
                &(cls_hash1, subtree_levels1),
                &(cls_hash2, subtree_levels2),
              ) {
                continue;
              }
              if !learned_lib.co_occurrences.may_co_occur(*ecls1, *ecls2) {
                continue;
              }
              if !type_match(&ty1, &ty2) {
                continue;
              }
              eclass_pairs.push((*ecls1, *ecls2));
            }
          }
        }
        EnumMode::PruningGold => {
          // 剪枝优化+并行处理
          let mut class_data: Vec<_> = classes
            .iter()
            .map(|cls| {
              let ty = egraph[*cls].data.get_type();
              let cls_hash = egraph[*cls].data.get_cls_hash();
              let subtree_levels = egraph[*cls].data.get_subtree_levels();
              (cls, ty, cls_hash, subtree_levels)
            })
            .collect();
          class_data.sort_unstable_by_key(|(ecls, ty, _, _)| {
            (ty.clone(), usize::from(**ecls))
          });
          // for data in class_data.clone() {
          //   println!("Id: {}, Type: {}", data.0, data.1);
          // }
          let ranges = group_by_type_ranges(&class_data);
          // 用一个二维数组来存储pairs的配对情况
          let enum_start = Instant::now();
          let all_pairs: Vec<(Id, Id)> = ranges
            .into_par_iter()
            .flat_map(|(start, end)| {
              let mut local_pairs = vec![];

              for i in start..end {
                let (ecls1, _, cls_hash1, subtree_levels1) = &class_data[i];
                let popcount1 = cls_hash1.count_ones();
                let subtree_cnt1 = subtree_levels1.count_ones();
                for j in i..end {
                  let (ecls2, _, cls_hash2, subtree_levels2) = &class_data[j];
                  let popcount2 = cls_hash2.count_ones();
                  if (popcount1 as i32 - popcount2 as i32).abs() >= 36 {
                    continue; // 汉明距离不可能<36
                  }

                  let subtree_cnt2 = subtree_levels2.count_ones();
                  let all_one = (subtree_levels1.clone()
                    | subtree_levels2.clone())
                  .count_ones();
                  if (subtree_cnt1.max(subtree_cnt2) as f64)
                    < (0.67 * all_one as f64)
                  {
                    continue; // Jaccard相似度不可能>0.67
                  }
                  if !level_match(
                    &(*cls_hash1, subtree_levels1.clone()),
                    &(*cls_hash2, subtree_levels2.clone()),
                  ) {
                    continue;
                  }

                  if !learned_lib.co_occurrences.may_co_occur(**ecls1, **ecls2)
                  {
                    continue;
                  }

                  local_pairs.push((**ecls1, **ecls2));
                }
              }

              local_pairs
            })
            .collect();
          let eclass_pairs = all_pairs.clone();
          let _ = liblearn_config.write_log(
            &mut file,
            format!("use {:?} to collect pairs", enum_start.elapsed()).as_str(),
          );
          let start = Instant::now();
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph(egraph, pair);
          }
          let elapsed = start.elapsed();
          let _ = liblearn_config.write_log(
            &mut file,
            format!("enumerate over dfta takes {:?}", elapsed).as_str(),
          );
        }
        EnumMode::ClusterTest => {
          let mut eclass_pairs = vec![];
          let prune_start = Instant::now();
          let mut class_data: Vec<_> = classes
            .iter()
            .map(|cls| {
              let ty = egraph[*cls].data.get_type();
              let cls_hash = egraph[*cls].data.get_cls_hash();
              let subtree_levels = egraph[*cls].data.get_subtree_levels();
              (cls, ty, cls_hash, subtree_levels)
            })
            .collect();
          class_data
            .sort_unstable_by_key(|(ecls, _, _, _)| usize::from(**ecls));
          let mut matched_patterns: HashMap<BitVec<u64>, Vec<(usize, usize)>> =
            HashMap::new();
          let mut patterns = vec![];
          for i in 0..class_data.len() {
            for j in i..class_data.len() {
              let (ecls1, ty1, cls_hash1, subtree_levels1) =
                class_data[i].clone();
              let (ecls2, ty2, cls_hash2, subtree_levels2) =
                class_data[j].clone();
              if !type_match(&ty1, &ty2) {
                continue;
              }
              if !level_match(
                &(cls_hash1, subtree_levels1),
                &(cls_hash2, subtree_levels2),
              ) {
                continue;
              }
              if !learned_lib.co_occurrences.may_co_occur(*ecls1, *ecls2) {
                continue;
              }
              let pattern =
                egraph[*ecls1].data.get_pattern(&egraph[*ecls2].data);
              if pattern.count_ones() == 0 {
                continue;
              }
              if matched_patterns.contains_key(&pattern) {
                matched_patterns.get_mut(&pattern).unwrap().push((i, j));
              } else {
                matched_patterns.insert(pattern.clone(), vec![(i, j)]);
                patterns.push(pattern.clone());
              }
            }
          }

          patterns.sort_by(|a, b| b.count_ones().cmp(&a.count_ones()));
          let m = 30;
          for i in 0..patterns.len() {
            if i > m {
              break;
            }
            let pattern = &patterns[i];
            let mut pairs = matched_patterns.get(pattern).unwrap().clone();
            pairs.sort_by(|a, b| b.cmp(a));
            // 如果小于k个pair就直接插入，否则取前k个
            let k = 10;
            let id_pair = pairs
              .clone()
              .into_iter()
              .map(|(i, j)| (classes[i], classes[j]))
              .collect::<Vec<_>>();
            if pairs.len() < k {
              eclass_pairs.extend(id_pair);
            } else {
              // 取前两个
              for i in 0..k {
                eclass_pairs.push(id_pair[i]);
              }
            }

            // println!("pattern: {:?}", pattern);
            // println!("matched patterns: {:?}",
            // matched_patterns.get(pattern));
          }
          let _ = liblearn_config.write_log(
            &mut file,
            format!("use {:?} to collect pairs", prune_start.elapsed())
              .as_str(),
          );
          let enum_start = Instant::now();
          for (ecls1, ecls2) in eclass_pairs {
            learned_lib.enumerate_over_egraph(egraph, (ecls1, ecls2));
          }
          let _ = liblearn_config.write_log(
            &mut file,
            format!("enumerate over dfta takes {:?}", enum_start.elapsed())
              .as_str(),
          );
        }
      };
      println!(
        "we all need to calculate {} pairs of eclasses",
        learned_lib.aus_by_state.len()
      );
    }

    // 如果learned_lib中的aus数量大于500，就从排序结果中随机选取500个
    // if learned_lib.aus.len() > 500 {
    //   let mut aus = learned_lib.aus.iter().cloned().collect::<Vec<_>>();
    //   aus.shuffle(&mut rand::thread_rng());
    //   let aus = aus.into_iter().take(500).collect::<BTreeSet<_>>();
    //   learned_lib.aus = aus;
    // }

    if learned_lib.aus.len() > 500 {
      let aus = learned_lib.aus.iter().collect::<Vec<_>>();
      let mut sampled_aus = BTreeSet::new();
      let step = aus.len() / 500;
      for i in (0..aus.len()).step_by(step) {
        sampled_aus.insert(aus[i].clone());
      }
      learned_lib.aus = sampled_aus;
    }

    learned_lib
  }
}

impl<Op, T> LearnedLibrary<Op, T>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Display
    + Ord
    + Send
    + Sync
    + Teachable
    + DiscriminantEq
    + 'static
    + Hash
    + Schedulable,
  AstNode<Op>: Language,
{
  /// Returns an iterator over rewrite rules that replace expressions with
  /// equivalent calls to a learned library function.
  ///
  /// For example, the expression
  ///
  /// ```text
  /// (* (+ 1 2) 5)
  /// ```
  ///
  /// might be rewritten to
  ///
  /// ```text
  /// (lib f (lambda (* (+ 1 $0) 5))
  ///  (apply f 2))
  /// ```
  ///
  /// by a rewrite rule
  ///
  /// ```text
  /// (* (+ 1 ?x) 5) => (lib f (lambda (* (+ 1 $0) 5)) (apply f ?x))
  /// ```
  pub fn rewrites<A: Analysis<AstNode<Op>>>(
    &self,
  ) -> impl Iterator<Item = Rewrite<AstNode<Op>, A>> + '_ {
    self.aus.iter().enumerate().map(|(i, au)| {
      let new_i = i + self.last_lib_id;
      let searcher: Pattern<_> = au.expr.clone().into();
      let applier: Pattern<_> =
        reify(LibId(new_i), au.expr.clone(), self.clock_period).into();
      let name = format!("anti-unify {i}");
      debug!("Found rewrite \"{name}\":\n{searcher} => {applier}");

      // Both patterns contain the same variables, so this can never fail.
      Rewrite::new(name, searcher, applier).unwrap_or_else(|_| unreachable!())
    })
  }

  /// Right-hand sides of library rewrites.
  pub fn libs(&self) -> impl Iterator<Item = Pattern<AstNode<Op>>> + '_ {
    self.aus.iter().enumerate().map(|(i, au)| {
      let new_i = i + self.last_lib_id;
      let applier: Pattern<_> =
        reify(LibId(new_i), au.expr.clone(), self.clock_period).into();
      applier
    })
  }

  pub fn for_each_anti_unification<F>(&mut self, f: F)
  where
    F: Fn(&PartialExpr<Op, Var>) -> PartialExpr<Op, Var>,
  {
    let mut new_aus = BTreeSet::new();
    for au in &self.aus {
      let new_au = f(&au.expr);
      new_aus.insert(AU::new(
        new_au,
        au.matches.clone(),
        au.delay,
        self.liblearn_config.cost.clone(),
      ));
    }
    self.aus = new_aus;
  }

  /// The raw anti-unifications that we have collected
  pub fn anti_unifications(
    &self,
  ) -> impl Iterator<Item = &PartialExpr<Op, Var>> {
    self.aus.iter().map(|au| &au.expr)
  }

  /// Extend the set of anti-unifications externally
  pub fn extend(&mut self, aus: impl IntoIterator<Item = AU<Op, Var>>) {
    self.aus.extend(aus);
  }

  /// Number of patterns learned.
  #[must_use]
  pub fn size(&self) -> usize {
    self.aus.len()
  }

  /// If two candidate patterns (stored in `nontrivial_aus`) have the same set
  /// of matches, only preserve the smaller one of them.
  /// Here a match is a pair of the e-class where the match was found
  /// and the range of its substitution
  /// (as a multiset of e-classes; which variables matched which e-classes is
  /// irrelevant). The reason two such patterns are equivalent is because
  /// their corresponding library functions can be used in exactly the same
  /// places and will have the same multiset (and hence size) of actual
  /// arguments.
  ///
  /// For example, after running a DSR (+ ?x ?y) => (+ ?y ?x),
  /// for any learned pattern containing (+ ?x0 ?x1), there will be an
  /// equivalent pattern containing (+ ?x1 ?x0), which will be eliminated
  /// here.
  pub fn deduplicate<A: Analysis<AstNode<Op>>>(
    &mut self,
    egraph: &EGraph<AstNode<Op>, A>,
  ) {
    // println!("before deduplicating: ");
    // for lib in self.libs().collect::<Vec<_>>() {
    //   println!("{}", lib);
    // }
    // The algorithm is simply to iterate over all patterns,
    // and save their matches in a dictionary indexed by the match set.
    let mut cache: BTreeMap<Vec<Match>, PartialExpr<Op, Var>> = BTreeMap::new();
    let default_delay = 0;
    for au in &self.aus {
      let au = au.expr.clone();
      let pattern: Pattern<_> = au.clone().into();
      // A key in `cache` is a set of matches
      // represented as a sorted vector.
      let mut key = vec![];

      for m in pattern.search(egraph) {
        for sub in m.substs {
          let actuals: Vec<_> =
            pattern.vars().iter().map(|v| sub[*v]).collect();
          let match_signature = Match::new(m.eclass, actuals);
          key.push(match_signature);
        }
      }

      key.sort();
      match cache.get(&key) {
        Some(cached) if cached.size() <= au.clone().size() => {
          debug!(
            "Pruning pattern {}\n as a duplicate of {}",
            pattern,
            Pattern::from(cached.clone())
          );
        }
        _ => {
          cache.insert(key, au.clone());
        }
      }
    }
    self.aus = cache
      .into_iter()
      .map(|(matches, expr)| {
        AU::new(
          expr,
          matches,
          default_delay,
          self.liblearn_config.cost.clone(),
        )
      })
      .collect();
  }

  pub fn deduplicate_from_candidates<A: Analysis<AstNode<Op>>>(
    &mut self,
    candidates: impl IntoIterator<Item = PartialExpr<Op, (Id, Id)>>, /* 修改为 (Id, Id) */
  ) -> Vec<AU<Op, (Id, Id)>> {
    // 修改返回类型
    // 创建一个缓存，用于保存已经遇到的匹配集合与对应的最小模式
    let mut cache: BTreeMap<Vec<Match>, (PartialExpr<Op, (Id, Id)>, usize)> =
      BTreeMap::new();
    // info!("cache.size: {}", self.pattern_cache.len());
    // 遍历所有候选的模式
    for au in candidates {
      if au.size() > 400 {
        continue;
      }
      let pattern: Pattern<_> = normalize(au.clone()).0.into();
      let mut key = vec![];
      let matches = pattern.search(&self.egraph);
      // 计算每个au的delay
      let delay = au.get_delay();
      for m in matches {
        for sub in m.substs {
          let actuals: Vec<_> =
            pattern.vars().iter().map(|v| sub[*v]).collect();
          let match_signature = Match::new(m.eclass, actuals);
          key.push(match_signature);
        }
      }
      // 如果大于100ms，就打印出来

      key.sort();
      // 直接将key作为matches

      // 如果缓存中已经有相同的匹配集合，则只保留较小的那个
      match cache.get(&key) {
        Some((cached, _delay)) => {
          // 计算cached和au的delay
          let cached_delay = cached.get_delay();
          //   if cached.size() <= au.size(){
          // }
          if cached.size() >= au.size() || cached_delay >= delay {
            cache.insert(key, (au.clone(), delay));
          }
        }
        None => {
          cache.insert(key, (au.clone(), delay));
        }
      }
    }
    // 将cache中的模式转换为AU
    let result = cache
      .into_iter()
      .map(|(matches, (expr, delay))| {
        AU::new(expr, matches, delay, self.liblearn_config.cost.clone())
      })
      .collect();
    result
  }
}

impl<Op> LearnedLibrary<Op, (Id, Id)>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + DiscriminantEq
    + Hash
    + Sync
    + Send
    + Display
    + 'static
    + Teachable
    + Schedulable,
{
  /// Computes the antiunifications of `state` in the DFTA `dfta`.
  fn enumerate_over_dfta(
    &mut self,
    dfta: &Dfta<(Op, Op), (Id, Id)>,
    state: (Id, Id),
  ) {
    if self.aus_by_state.contains_key(&state) {
      // We've already enumerated this state, so there's nothing to do.
      return;
    }
    info!("Enumerating over state {:?}", state);
    // We're going to recursively compute the antiunifications of the inputs
    // of all of the rules leading to this state. Before we do, we need to
    // mark this state as in progress so that a loop in the rules doesn't
    // cause infinite recursion.
    //
    // By initially setting the antiunifications of this state to empty, we
    // exclude any antiunifications that would come from looping sequences
    // of rules.
    self.aus_by_state.insert(state, BTreeSet::new());

    if !self.co_occurrences.may_co_occur(state.0, state.1) {
      // a在b的co-occurrence中或者b在a的co-occurrence中
      return;
    }

    let mut aus: BTreeSet<AU<Op, (Id, Id)>> = BTreeSet::new();

    let mut same = false;
    let mut different = false;

    // if there is a rule that produces this state
    if let Some(rules) = dfta.get_by_output(&state) {
      // 获得可以产生当前状态的rule(输入状态和op)
      for ((op1, op2), inputs) in rules {
        info!(
          "op1: {:?}, op2: {:?}, input size: {}",
          op1,
          op2,
          inputs.len()
        );
        if op1 == op2 {
          same = true;
          if inputs.is_empty() {
            let new_au = PartialExpr::from(AstNode::leaf(op1.clone()));
            aus.insert(AU::new_with_expr(
              new_au,
              &self.egraph,
              self.liblearn_config.cost.clone(),
            ));
          } else {
            // Recursively enumerate the inputs to this rule.
            for &input in inputs {
              self.enumerate_over_dfta(dfta, input);
            }

            // For a rule `op(s1, ..., sn) -> state`, we add an
            // antiunification of the form `(op a1 ... an)` for every
            // combination `a1, ..., an` of antiunifications of the
            // input states `s1, ..., sn`, i.e., for every `(a1, ..., an)`
            // in the cartesian product
            // `antiunifications_by_state[s1] × ... ×
            // antiunifications_by_state[sn]`

            let smax_arity = self.max_arity;

            let new_aus = {
              let au_range = inputs.iter().map(|input| {
                let data: Vec<_> =
                  self.aus_by_state[input].iter().cloned().collect();
                data
              });
              let au_range: Vec<Vec<AU<Op, (Id, Id)>>> =
                au_range.collect::<Vec<_>>();
              let new_au = match self.liblearn_config.au_merge_mod {
                AUMergeMod::Random => get_random_aus(au_range, 1000),
                AUMergeMod::Kd => kd_random_aus(au_range, 1000),
                AUMergeMod::Greedy => greedy_aus(au_range),
                AUMergeMod::Catesian => greedy_aus(au_range),
              };
              // get_random_aus(au_range, 10)
              //beam_search_aus(au_range, 1000, 2000)
              // genetic_algorithm_aus(au_range, 1000, 500, 50)
              // au_range.into_iter().multi_cartesian_product()
              new_au
                .into_iter()
                .map(|inputs| {
                  // println!("inputs length is {}", inputs.len());
                  // let inputs = inputs.into_iter().map(|au| au.expr.clone());
                  let result =
                    PartialExpr::from(AstNode::new(op1.clone(), inputs));
                  result
                })
                .filter(|au| {
                  let result = smax_arity.map_or(true, |max_arity| {
                    au.unique_holes().len() <= max_arity
                  });
                  result
                })
            };
            // info!("length: {}",new_aus.clone().count());
            // let mut new_aus = new_aus.collect::<Vec<_>>();
            // info!("new_aus is done");
            // // 如果超过1000个，就随机抽取1000个
            // if new_aus.len() > 100 {
            //   let mut new_aus_vec: Vec<PartialExpr<Op, (Id, Id)>> =
            // new_aus.clone();   let mut rng = rand::thread_rng();
            //   new_aus_vec.shuffle(&mut rng);
            //   new_aus = new_aus_vec.into_iter().take(100).collect();
            // }
            // // 使用deduplicate_from_candidates去重
            info!("aus_state.len() is {}", self.aus_by_state.len());
            info!("deduplicating new_aus last: {} ", new_aus.clone().count());
            let new_aus_dedu = self
              .deduplicate_from_candidates::<EmptyAnalysis<Op>>(
                new_aus.clone(),
              );
            info!("now  is {}", new_aus_dedu.len());
            let start_total = Instant::now(); // 总的执行时间
            aus.extend(new_aus_dedu);
            info!(
              "Total processing time: {:?}, lenth of new_aus and aus is {}",
              start_total.elapsed(),
              aus.len()
            ); // 总的处理时间
          }
        } else {
          different = true;
        }
      }
    }

    if same && different {
      let new_expr = PartialExpr::Hole(state);
      aus.insert(AU::new_with_expr(
        new_expr,
        &self.egraph,
        self.liblearn_config.cost.clone(),
      ));
    }
    self.filter_aus(aus, state);
  }

  /// Computes the antiunifications of `state` in the `egraph`.
  /// This avoids computing all of the cross products ahead of time, which is
  /// what the `Dfta` implementation does. So it should be faster and more
  /// memory-efficient. However, I'm not totally sure it's equivalent to the
  /// `Dfta` version.
  fn enumerate_over_egraph<A: Analysis<AstNode<Op>> + Clone>(
    &mut self,
    egraph: &EGraph<AstNode<Op>, A>,
    state: (Id, Id),
  ) {
    if self.aus_by_state.contains_key(&state) {
      // we have already enumerated this state
      return;
    }

    self.aus_by_state.insert(state, BTreeSet::new());

    if !self.co_occurrences.may_co_occur(state.0, state.1) {
      return;
    }

    let mut aus: BTreeSet<AU<Op, (Id, Id)>> = BTreeSet::new();

    let mut same = false;
    let mut different = false;

    let ops1 = egraph[state.0].nodes.iter().map(AstNode::as_parts);
    let ops2 = egraph[state.1].nodes.iter().map(AstNode::as_parts);

    for (op1, args1) in ops1 {
      for (op2, args2) in ops2.clone() {
        if op1 == op2 {
          same = true;
          if args1.is_empty() && args2.is_empty() {
            // FIXME: is that right?
            let new_expr = AstNode::leaf(op1.clone()).into();
            aus.insert(AU::new_with_expr(
              new_expr,
              &self.egraph,
              self.liblearn_config.cost.clone(),
            ));
          } else {
            info!("Processing op1 {:?} and op2 {:?}", op1, op2);
            // recursively enumerate the inputs to this rule.
            let inputs: Vec<_> =
              args1.iter().copied().zip(args2.iter().copied()).collect();

            for next_state in &inputs {
              self.enumerate_over_egraph(egraph, *next_state);
            }

            let smax_arity = self.max_arity;
            // let au_range: Vec<Vec<PartialExpr<Op, (Id, Id)>>> =
            // au_range.collect::<Vec<_>>();
            let new_aus = inputs.iter().map(|input| {
              let aus =
                self.aus_by_state[input].iter().cloned().collect::<Vec<_>>();
              aus
            });
            info!("get_range_aus");
            let au_range = new_aus.collect::<Vec<_>>();
            let new_aus = match self.liblearn_config.au_merge_mod {
              AUMergeMod::Random => get_random_aus(au_range, 1000),
              AUMergeMod::Kd => kd_random_aus(au_range, 1000),
              AUMergeMod::Greedy => greedy_aus(au_range),
              AUMergeMod::Catesian => {
                // 笛卡尔积
                // 首先取出所有的expr
                let new_exprs = au_range.into_iter().map(|inputs| {
                  let inputs = inputs.into_iter().map(|au| au.expr.clone());
                  inputs.collect::<Vec<_>>()
                });
                let new_expr = new_exprs.into_iter().multi_cartesian_product();
                new_expr.collect::<Vec<_>>()
              }
            };
            info!("filtering according to arity");
            let new_aus = new_aus
              .into_iter() //kd_random_aus(au_range, 1000).into_iter()
              .map(|inputs| {
                PartialExpr::from(AstNode::new(op1.clone(), inputs))
              })
              .filter(|au| {
                smax_arity.map_or(true, |max_arity| {
                  au.unique_holes().len() <= max_arity
                })
              });

            info!("aus_state.len() is {}", self.aus_by_state.len());
            info!("deduplicating new_aus last: {} ", new_aus.clone().count());

            let new_aus_dedu = self
              .deduplicate_from_candidates::<EmptyAnalysis<Op>>(
                new_aus.clone(),
              );
            // info!("now  is {}", new_aus_dedu.len());
            let start_total = Instant::now(); // 总的执行时间
            // 为每一个新的模式生成一个AU
            // aus.extend(new_aus.map(|au| AU::new_cal_matches(au,
            // &self.egraph)));
            // println!("aus_size: {}", aus.len());
            // println!("{:?}", new_aus_dedu);
            aus.extend(new_aus_dedu);
            info!(
              "Total processing time: {:?}, lenth of new_aus and aus is {}",
              start_total.elapsed(),
              aus.len()
            ); // 总的处理时间
          }
        } else {
          different = true;
        }
      }
    }

    if same && different {
      let new_expr = PartialExpr::Hole(state);
      aus.insert(AU::new_with_expr(
        new_expr,
        &self.egraph,
        self.liblearn_config.cost.clone(),
      ));
    }

    self.filter_aus(aus, state);
  }

  fn filter_aus(
    &mut self,
    mut aus: BTreeSet<AU<Op, (Id, Id)>>,
    state: (Id, Id),
  ) {
    if aus.is_empty() {
      let new_expr = PartialExpr::Hole(state);
      aus.insert(AU::new_with_expr(
        new_expr,
        &self.egraph,
        self.liblearn_config.cost.clone(),
      ));
    } else {
      // If the two e-classes cannot co-occur in the same program, do not
      // produce an AU for them! We filter out the anti-unifications which
      // are just concrete expressions with no variables, and then convert
      // the contained states to pattern variables. The conversion takes
      // alpha-equivalent anti-unifications to the same value, effectively
      // discarding redundant anti-unifications.

      let learn_constants = self.learn_constants;
      let learn_trivial = self.learn_trivial;
      let banned_ops = &self.banned_ops;

      let nontrivial_aus = aus
        .iter()
        .filter(|au| learn_constants || au.expr.has_holes())
        .cloned()
        .map(|au| {
          (
            normalize(au.expr().clone()),
            au.matches().clone(),
            au.delay().clone(),
          )
        })
        .filter_map(|((au, num_vars), matches, delay)| {
          // Here we filter out rewrites that don't actually simplify
          // anything. We say that an AU rewrite simplifies an
          // expression if it replaces that expression with a function
          // call that is strictly smaller than the original
          // expression.
          //
          // The size of a function call `f e_1 ... e_n` is size(e1) +
          // ... + size(e_n) + n + 1, as there are n applications and
          // the function's identifier `f`.
          //
          // The size of an expression e containing n subexpressions
          // e_1, ..., e_n is k_1 * size(e_1) + ... + k_n * size(e_n)
          // + size(e[x_1/e_1, ..., x_n/e_n]) - (k_1 + ... + k_n),
          // where k_i is the number of times e_i appears in e and
          // x_1, ..., x_n are variables.
          //
          // Because size(e_i) can be arbitrarily large, if any
          // variable k_i is greater than 1, the difference in size
          // between the function call and the original expression can
          // also be arbitrarily large. Otherwise, if k_1 = ... = k_n
          // = 1, the rewrite can simplify an expression if and only
          // if size(e[x_1/e_1, ..., x_n/e_n]) > 2n + 1. This
          // corresponds to an anti-unification containing at least n
          // + 1 nodes.
          if learn_trivial
            || num_vars < au.num_holes() // 原来是num_vars < au.num_holes()
            || au.num_nodes() > 1 + num_vars
          // FIXME:num_vars + 1, 这里改为2*num_vars是为了减少重复的模式
          {
            // println!("learn_trivial: {}, num_vars < au.num_holes(): {},
            // au.num_nodes() > num_vars: {}", learn_trivial, num_vars <
            // au.num_holes(), au.num_nodes() > num_vars);
            Some(AU::new(
              au,
              matches,
              delay,
              self.liblearn_config.cost.clone(),
            ))
          } else {
            None
          }
        })
        .filter(|au| match au.expr.clone() {
          PartialExpr::Node(ast_node) => !banned_ops
            .iter()
            .any(|op| ast_node.operation().discriminant_eq(op)),
          PartialExpr::Hole(_) => true,
        });
      info!(
        "length of nontrivial_aus is {}",
        nontrivial_aus.clone().count()
      );
      self.aus.extend(nontrivial_aus);
    }
    *self.aus_by_state.get_mut(&state).unwrap() = aus;
  }
}

/// Replaces the metavariables in an anti-unification with pattern variables.
/// Normalizing alpha-equivalent anti-unifications produces identical
/// anti-unifications. Returns a pair of the anti-unification and the number of
/// unique variables it contains.
#[must_use]
pub fn normalize<Op, T: Eq>(
  au: PartialExpr<Op, T>,
) -> (PartialExpr<Op, Var>, usize) {
  let mut metavars = Vec::new();
  let to_var = |metavar| {
    let index = metavars
      .iter()
      .position(|other| other == &metavar)
      .unwrap_or_else(|| {
        metavars.push(metavar);
        metavars.len() - 1
      });

    let var = format!("?x{index}")
      .parse()
      .unwrap_or_else(|_| unreachable!());
    PartialExpr::Hole(var)
  };
  let normalized = au.fill(to_var);
  (normalized, metavars.len())
}

#[allow(dead_code)]
fn patternize<Op>(au: &PartialExpr<Op, (Id, Id)>) -> Pattern<AstNode<Op>>
where
  Op: Arity + Clone + Display + Ord + Send + Sync + 'static,
  AstNode<Op>: Language,
{
  let au = au.clone();
  au.fill(|(s1, s2)| {
    let var = format!("?s_{s1}_{s2}")
      .parse()
      .unwrap_or_else(|_| unreachable!());
    PartialExpr::Hole(var)
  })
  .into()
}

/// Converts an anti-unification into a partial expression which defines a new
/// named function and applies it to the metavariables in the anti-unification.
/// The new function reifies the anti-unification, replacing metavariables by
/// lambda arguments.
///
/// For example, the anti-unification
///
/// ```text
/// (* ?x (+ ?y 1))
/// ```
///
/// would be converted to the partial expression
///
/// ```text
/// (lib l_i (lambda (lambda (* $0 (+ $1 1))))
///  (apply (apply l_i ?y) ?x))
/// ```
///
/// assuming `name` is "foo".
#[must_use]
fn reify<Op, T>(
  ix: LibId,
  au: PartialExpr<Op, T>,
  clock_period: usize,
) -> PartialExpr<Op, T>
where
  Op: Clone
    + Default
    + Arity
    + Debug
    + Display
    + Ord
    + Send
    + Sync
    + Teachable
    + 'static
    + Hash
    + Schedulable,
  AstNode<Op>: Language,
  T: Eq + Clone + Hash + Debug,
{
  let mut metavars = Vec::new();

  // Replace every metavariable in this antiunification with a de
  // Bruijn-indexed variable.
  // Metavariables might be located inside lambdas. To deal with this,
  // the de Brujin index that we return is equal to the index of the
  // metavar, added to however many lambdas wrap the metavar at that
  // point.
  let mut fun = au.clone().fill_with_binders(|metavar, num_binders| {
    let index = metavars
      .iter()
      .position(|other: &(T, usize)| other.0 == metavar)
      .unwrap_or_else(|| {
        metavars.push((metavar, num_binders));
        metavars.len() - 1
      });
    let index = index + num_binders;

    let mut res = PartialExpr::Hole(index);

    for i in (0..num_binders).rev() {
      res = Op::apply(res, Op::var(i).into()).into();
    }

    res
  });

  // foo (\. \. $0 $2 ?hole) => foo (\. \. $0 $2 ?$2)
  //                                          ^ binders = 2

  // All the function variables
  let offset = metavars.len();

  let mut max_locals = 0;

  fun =
    fun.map_leaves_with_binders(|node, binders| match node.as_binding_expr() {
      Some(BindingExpr::Var(index)) if index.0 >= binders => {
        max_locals = std::cmp::max(max_locals, index.0 - binders + 1);
        Op::var(index.0 + offset).into()
      }
      _ => node.into(),
    });

  // foo (\. \. $0 $2 ?$2) => foo (\. \. $0 $3 ?$2)

  let mut fun = fun.fill(|index| Op::var(index).into());

  // Wrap that in a lambda-abstraction, one for each variable we introduced.
  for _ in 0..(metavars.len() + max_locals) {
    fun = Op::lambda(fun).into();
  }

  // Now apply the new function to the metavariables in reverse order so they
  // match the correct de Bruijn indexed variable.
  let mut body = Op::lib_var(ix).into();
  while let Some((metavar, binders)) = metavars.pop() {
    let mut fn_arg = PartialExpr::Hole(metavar);
    for _i in 0..binders {
      fn_arg = Op::lambda(fn_arg).into();
    }
    body = Op::apply(body, fn_arg).into();
  }

  for index in 0..max_locals {
    body = Op::apply(body, Op::var(index).into()).into();
  }

  // Calculate the gain and the cost of the au
  let expr: Expr<Op> = au.clone().try_into().unwrap();
  let rec_expr: RecExpr<AstNode<Op>> = expr.into();
  // println!("rec_expr: {:?}", rec_expr);
  let (gain, cost) = Scheduler::new(clock_period).asap_schedule(&rec_expr);
  // println!("gain: {}, cost: {}", gain, cost);

  PartialExpr::Node(BindingExpr::Lib(ix, fun, body, gain, cost).into())
}
