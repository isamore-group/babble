//! `extract::partial` implements a non-ILP-based extractor based on partial
//! orderings of learned library sets.
use crate::{
  ast_node::{Arity, AstNode},
  learn::LibId,
  schedule::rec_cost,
  teachable::{BindingExpr, Teachable},
};
use bitvec::prelude::*;
use egg::{
  Analysis, AstSize, DidMerge, EGraph, Extractor, Id, Language, RecExpr, Runner,
};
use log::debug;
use rustc_hash::FxHashMap;
use std::{
  cmp::Ordering,
  collections::{BinaryHeap, HashMap, hash_map::Entry},
  fmt::Debug,
  hash::{DefaultHasher, Hash, Hasher},
};

/// 实现一个非常简单的没有实际意义的Analysis
/// 这个Analysis只用于LibExtractor
#[derive(Debug, Clone)]
pub struct EmptyAnalysis<Op> {
  _phantom: PhantomData<Op>,
}
impl<Op> Default for EmptyAnalysis<Op> {
  fn default() -> Self {
    EmptyAnalysis {
      _phantom: PhantomData,
    }
  }
}
impl<Op> Analysis<AstNode<Op>> for EmptyAnalysis<Op>
where
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display,
{
  type Data = ();
  fn merge(&mut self, _: &mut Self::Data, _: Self::Data) -> DidMerge {
    DidMerge(false, false)
  }
  fn make(_: &mut EGraph<AstNode<Op>, Self>, _: &AstNode<Op>) -> Self::Data {
    ()
  }
}

/// A `CostSet` is a set of pairs; each pair contains a set of library
/// functions paired with the cost of the current expression/eclass
/// without the lib fns, and the cost of the lib fns themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostSet {
  /// The set of library selections and their associated costs.
  /// Invariant: sorted in ascending order of `expr_cost`, except during
  /// pruning, when it's sorted in order of `full_cost`.
  pub set: Vec<LibSel>,
}

impl CostSet {
  /// Creates a `CostSet` corresponding to introducing a nullary operation.
  #[must_use]
  pub fn new() -> CostSet {
    let mut set = Vec::with_capacity(10);
    set.push(LibSel::new());
    CostSet { set }
  }

  /// Crosses over two `CostSet`s.
  /// This is essentially a Cartesian product between two `CostSet`s (e.g. if
  /// each `CostSet` corresponds to an argument of a node) such that paired
  /// `LibSel`s have their libraries combined and costs added.
  #[must_use]
  pub fn cross(&self, other: &CostSet, lps: usize, strategy: f32) -> CostSet {
    let mut set = Vec::new();

    for ls1 in &self.set {
      for ls2 in &other.set {
        match ls1.combine(ls2, lps, strategy) {
          None => continue,
          Some(ls) => {
            if let Err(pos) = set.binary_search(&ls) {
              set.insert(pos, ls);
            }
          }
        }
      }
    }

    CostSet { set }
  }

  /// Combines two `CostSets` by unioning them together.
  /// Used for e.g. different `ENodes` of an `EClass`.
  pub fn combine(&mut self, other: CostSet) {
    // println!("combine");
    let mut cix = 0;

    for elem in other.set {
      while cix < self.set.len() && elem >= self.set[cix] {
        cix += 1;
      } // Nadia: can't this be done with a binary search?

      self.set.insert(cix, elem);
      cix += 1;
    }
  }

  /// Performs trivial partial order reduction: if `CostSet` A contains a
  /// superset of the libs of another `CostSet` B, and A has a higher
  /// `expr_cost` than B, remove A.
  pub fn unify(&mut self) {
    // println!("unify");
    let mut i = 0;

    while i < self.set.len() {
      let mut j = i + 1;

      while j < self.set.len() {
        let ls1 = &self.set[i];
        let ls2 = &self.set[j];

        if ls1.is_subset(ls2) {
          self.set.remove(j);
        } else {
          j += 1;
        }
      }
      i += 1;
    }
  }

  #[must_use]
  pub fn add_lib(
    &self,
    lib: LibId,
    gain: usize,
    cost: usize,
    nested_libs: &CostSet,
    lps: usize,
    strategy: f32,
  ) -> CostSet {
    // println!("add_lib");
    // To add a lib, we do a modified cross.
    let mut set = Vec::new();

    for ls1 in &nested_libs.set {
      for ls2 in &self.set {
        match ls2.add_lib(lib, gain, cost, ls1, lps, strategy) {
          None => continue,
          Some(ls) => {
            if let Err(pos) = set.binary_search(&ls) {
              set.insert(pos, ls);
            }
          }
        }
      }
    }

    CostSet { set }
  }

  /// prune takes care of two different tasks to reduce the number
  /// of functions in a `LibSel`:
  ///
  /// - If we have an lps limit, we remove anything that has more fns than we
  ///   allow in a `LibSel`
  /// - Then, if we still have too many `LibSel`s, we prune based on beam size.
  ///
  /// Our pruning strategy preserves n `LibSel`s per # of libs, where
  /// n is the beam size. In other words, we preserve n `LibSel`s with
  /// 0 libs, n `LibSel`s with 1 lib, etc.
  pub fn prune(&mut self, n: usize, lps: usize) {
    use std::cmp::Reverse;

    let old_set = std::mem::take(&mut self.set);

    // First, we create a table from # of libs to a list of LibSels
    let mut table: HashMap<usize, BinaryHeap<Reverse<LibSelFC>>> =
      HashMap::new();

    // We then iterate over all of the LibSels in this set
    for ls in old_set {
      let num_libs = ls.libs.len();

      // We don't need to do this anymore, because this is happening in cross:
      // If lps is set, if num_libs > lps, give up immediately
      // if num_libs > lps {
      //     panic!("LibSels that are too large should have been filtered out by
      // cross!"); }

      let h = table.entry(num_libs).or_default();

      h.push(Reverse(LibSelFC(ls)));
    }

    // From our table, recombine into a sorted vector
    let mut set = Vec::new();
    let beams_per_size = std::cmp::max(1, n / lps);

    for (_sz, mut h) in table {
      // Take the first n items from the heap
      let mut i = 0;
      while i < beams_per_size {
        if let Some(ls) = h.pop() {
          let ls = ls.0.0;

          if let Err(pos) = set.binary_search(&ls) {
            set.insert(pos, ls);
          }

          i += 1;
        } else {
          break;
        }
      }
    }

    self.set = set;
  }

  pub fn unify2(&mut self) {
    // println!("unify");
    let mut i = 0;

    while i < self.set.len() {
      let mut j = i + 1;

      while j < self.set.len() {
        let ls1 = &self.set[i];
        let ls2 = &self.set[j];

        if ls2.is_subset(ls1) {
          self.set.remove(j);
        } else {
          j += 1;
        }
      }
      i += 1;
    }
  }
}

/// A `LibSel` is a selection of library functions, paired with two
/// corresponding cost values: the cost of the expression without the library
/// functions, and the cost of the library functions themselves
#[derive(Debug, Clone)]

pub struct LibSel {
  /// The full cost of the expression, balancing the latency gain and area
  /// cost.
  pub full_cost: f32,
  /// The libraries used in this expression. Each library is binded with its
  /// gain, cost and its times being used.
  pub libs: HashMap<LibId, (usize, usize, usize)>,
}

impl Eq for LibSel {}

impl PartialEq for LibSel {
  fn eq(&self, other: &Self) -> bool {
    self.full_cost == other.full_cost && self.libs == other.libs
  }
}

impl PartialOrd for LibSel {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    let r = self.full_cost.partial_cmp(&other.full_cost);
    match r {
      Some(ord) => Some(ord),
      None => None,
    }
  }
}

impl Ord for LibSel {
  fn cmp(&self, other: &Self) -> Ordering {
    let r = self.full_cost.partial_cmp(&other.full_cost);
    match r {
      Some(ord) => ord,
      None => {
        // If the full cost is NaN, we want to sort it to the end.
        if self.full_cost.is_nan() {
          Ordering::Greater
        } else {
          Ordering::Less
        }
      }
    }
  }
}

impl LibSel {
  #[must_use]
  pub fn new() -> LibSel {
    LibSel {
      full_cost: 0.0,
      libs: HashMap::new(),
    }
  }

  /// Combines two `LibSel`s. Unions the lib sets, adds
  /// the expr
  #[must_use]
  pub fn combine(
    &self,
    other: &LibSel,
    lps: usize,
    strategy: f32,
  ) -> Option<LibSel> {
    let mut res = self.clone();

    for (lib_id, lib_info) in &other.libs {
      match res.libs.entry(*lib_id) {
        Entry::Occupied(mut entry) => {
          let (_, _, times) = entry.get_mut();
          *times += lib_info.2;
          res.full_cost -= lib_info.2 as f32 * lib_info.0 as f32 * strategy;
        }
        Entry::Vacant(entry) => {
          entry.insert(*lib_info);
          res.full_cost += lib_info.1 as f32 * (1.0 - strategy)
            - lib_info.2 as f32 * lib_info.0 as f32 * strategy;
          if res.libs.len() > lps {
            return None;
          }
        }
      }
    }

    Some(res)
  }

  #[must_use]
  pub fn add_lib(
    &self,
    lib: LibId,
    gain: usize,
    cost: usize,
    nested_libs: &LibSel,
    lps: usize,
    strategy: f32,
  ) -> Option<LibSel> {
    let mut res = self.clone();

    // Add all nested libs that the lib uses, then add the lib itself.
    for (nested_lib, lib_info) in &nested_libs.libs {
      match res.libs.entry(*nested_lib) {
        Entry::Occupied(mut entry) => {
          let (_, _, times) = entry.get_mut();
          *times += lib_info.2;
          res.full_cost -= lib_info.2 as f32 * lib_info.0 as f32 * strategy;
        }
        Entry::Vacant(entry) => {
          entry.insert(*lib_info);
          res.full_cost += lib_info.1 as f32 * (1.0 - strategy)
            - lib_info.2 as f32 * lib_info.0 as f32 * strategy;
          if res.libs.len() > lps {
            return None;
          }
        }
      }
    }

    match res.libs.entry(lib) {
      Entry::Occupied(mut entry) => {
        let (_, _, times) = entry.get_mut();
        *times += 1;
        res.full_cost -= gain as f32 * strategy;
      }
      Entry::Vacant(entry) => {
        entry.insert((gain, cost, 1));
        res.full_cost +=
          cost as f32 * (1.0 - strategy) - gain as f32 * strategy;
        if res.libs.len() > lps {
          return None;
        }
      }
    }

    Some(res)
  }

  /// O(n) subset check
  #[must_use]
  pub fn is_subset(&self, other: &LibSel) -> bool {
    // For every element in this LibSel, we want to see
    // if it exists in other.
    for (lib_id, _) in &self.libs {
      match other.libs.get(lib_id) {
        Some(_) => {}
        None => {
          return false;
        }
      }
    }
    true
  }
}

/// A wrapper around `LibSel`s that orders based on their full cost.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LibSelFC(pub(crate) LibSel);

impl PartialOrd for LibSelFC {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.0.full_cost.partial_cmp(&other.0.full_cost)
  }
}

impl Ord for LibSelFC {
  fn cmp(&self, other: &Self) -> Ordering {
    self.partial_cmp(other).unwrap()
  }
}

// --------------------------------
// --- The actual Analysis part ---
// --------------------------------
use std::marker::PhantomData;
#[derive(Debug, Clone)]
pub struct ISAXAnalysis<Op, T>
where
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable + Arity,
{
  /// The number of `LibSel`s to keep per `EClass`.
  beam_size: usize,
  inter_beam: usize,
  /// The maximum number of libs per lib selection. Any lib selections with a
  /// larger amount will be pruned.
  lps: usize,
  strategy: f32,
  /// use a TypeMap to get result map
  type_info_map: HashMap<(String, Vec<T>), T>,
  /// Marker to indicate that this struct uses the Op type parameter
  op_phantom: PhantomData<Op>,
  ty_phantom: PhantomData<T>,
}

impl<Op, T> ISAXAnalysis<Op, T>
where
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable + Arity,
{
  #[must_use]
  pub fn new(
    beam_size: usize,
    inter_beam: usize,
    lps: usize,
    strategy: f32,
    type_info_map: HashMap<(String, Vec<T>), T>,
  ) -> ISAXAnalysis<Op, T> {
    ISAXAnalysis {
      beam_size,
      inter_beam,
      lps,
      strategy,
      type_info_map,
      op_phantom: PhantomData,
      ty_phantom: PhantomData,
    }
  }

  #[must_use]
  pub fn empty() -> ISAXAnalysis<Op, T> {
    ISAXAnalysis {
      beam_size: 0,
      inter_beam: 0,
      lps: 1,
      strategy: 1.0,
      type_info_map: HashMap::new(),
      op_phantom: PhantomData,
      ty_phantom: PhantomData,
    }
  }
}

impl<Op, T> Default for ISAXAnalysis<Op, T>
where
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable + Arity,
{
  fn default() -> Self {
    ISAXAnalysis::empty()
  }
}

// 定义StructuralHash ,结构化哈希包含cls_hash和subtree_hash,均为u64
#[derive(Debug, Clone)]
pub struct StructuralHash {
  pub cls_hash: u64,
  pub subtree_levels: BitVec<u64, Lsb0>,
  pub level_conflict: LevelConflictState,
}

impl Default for StructuralHash {
  fn default() -> Self {
    Self {
      cls_hash: 0,
      subtree_levels: bitvec![u64, Lsb0; 0; 64], // 64位初始化为0
      level_conflict: LevelConflictState::new(),
    }
  }
}

impl StructuralHash {
  pub fn merge(&mut self, other: &Self) {
    // 首先合并cls_hash
    // 为了和enode进行区分，会使用加盐值和rotate的方法打散哈希
    let mut state = seahash::SeaHasher::new();
    let cls_hashes = [self.cls_hash, other.cls_hash];
    for &h in &cls_hashes {
      // 加salt混合 or shift
      (h.rotate_left((h % 13) as u32)).hash(&mut state);
    }
    let merged_cls_hash = state
      .finish()
      .rotate_left((cls_hashes.len() as u32 * 7) % 64);
    // 之后合并subtree_levels
    let mut merged_subtree_levels = bitvec!(u64, Lsb0; 0; 64);
    for i in 0..64 {
      if self.subtree_levels[i] & other.subtree_levels[i] {
        self.level_conflict.update(i, true);
        merged_subtree_levels.set(i, true);
      } else if self.subtree_levels[i] ^ other.subtree_levels[i] {
        self.level_conflict.update(i, false);
        let prob = self.level_conflict.get(i);
        if rand::random::<f32>() > prob {
          merged_subtree_levels.set(i, true);
        }
      }
    }
    self.cls_hash = merged_cls_hash;
    self.subtree_levels = merged_subtree_levels;
  }
}

#[derive(Debug, Clone)]
pub struct LevelConflictState {
  // 本结构体用于在不同的enode进行合并时使用，记录不同层级的冲突状态
  states: [f32; 64],
  // 学习率
  alpha: f32,
}
impl LevelConflictState {
  pub fn new() -> Self {
    Self {
      states: [0.5; 64], // 初始化0.5的冲突概率
      alpha: 0.1,
    }
  }

  pub fn update(&mut self, level: usize, is_real_conflict: bool) {
    self.states[level] = (1.0 - self.alpha) * self.states[level]
      + self.alpha * is_real_conflict as u8 as f32 * 0.5;
  }

  pub fn get(&self, level: usize) -> f32 {
    self.states[level]
  }
}

#[derive(Debug, Clone)]
pub struct ISAXCost<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  pub cs: CostSet,
  pub ty: T,
  pub hash: StructuralHash,
}

impl<T> PartialEq for ISAXCost<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  fn eq(&self, other: &Self) -> bool {
    self.cs == other.cs && self.ty == other.ty
  }
}

impl<T: PartialEq> PartialEq<T> for ISAXCost<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  fn eq(&self, other: &T) -> bool {
    self.ty == *other
  }
}

impl<T: Debug + Default + Clone + PartialEq + Ord + Hash> ISAXCost<T> {
  #[must_use]
  pub fn new(cs: CostSet, ty: T, hash: StructuralHash) -> Self {
    ISAXCost { cs, ty, hash }
  }
}

// 计算层级哈希映射
fn compute_hash_level<Op: Hash>(
  enode: &AstNode<Op>,
  child_hashes: &[u64],
) -> usize
where
  Op: Teachable,
{
  let mut hasher = seahash::SeaHasher::new();
  let op = enode.operation().to_shielding_op();
  op.hash(&mut hasher);
  for &h in child_hashes {
    h.hash(&mut hasher);
  }
  (hasher.finish() % 64) as usize // 取模映射到固定范围
}
/// 计算不考虑层级的精确哈希
fn compute_full_hash<Op: Hash>(enode: &AstNode<Op>, child_hashes: &[u64]) -> u64
where
  Op: Teachable,
{
  let mut hasher = seahash::SeaHasher::new();
  let op = enode.operation().to_shielding_op();
  op.hash(&mut hasher);
  for &h in child_hashes {
    hasher.write_u64(h);
  }
  hasher.finish()
}

impl<Op, T> Analysis<AstNode<Op>> for ISAXAnalysis<Op, T>
where
  Op: Ord
    + std::hash::Hash
    + Debug
    + Teachable
    + Arity
    + Eq
    + Clone
    + Send
    + Sync
    + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  type Data = ISAXCost<T>;

  fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
    // println!("merge");
    // println!("{:?}", to);
    // println!("{:?}", &from);
    let a0 = to.clone();

    // Merging consists of combination, followed by unification and beam
    // pruning.
    to.cs.combine(from.cs.clone());
    to.cs.unify();
    to.cs.prune(self.beam_size, self.lps);
    // we also need to merge the type information
    (*to).ty = AstNode::merge_types(&to.ty, &from.ty);
    // 合并哈希
    // 合并ecls哈希，目前取最大值
    // to.hash.cls_hash = max(to.hash.cls_hash, from.hash.cls_hash);
    // to.hash.subtree_levels |= from.clone().hash.subtree_levels;
    to.hash.merge(&from.hash);
    // println!("{:?}", to);
    // println!("{} {}", &a0 != to, to != &from);
    // TODO: be more efficient with how we do this
    DidMerge(&a0 != to, to != &from)
    // DidMerge(false, false)
  }

  fn make(
    egraph: &mut EGraph<AstNode<Op>, Self>,
    enode: &AstNode<Op>,
  ) -> Self::Data {
    // calculate the type
    // println!("enode: {:?}", enode);
    let child_types: Vec<T> = enode
      .children()
      .iter()
      .map(|&child| egraph[child].data.ty.clone())
      .collect();
    let ty = enode.get_rtype(&egraph.analysis.type_info_map, &child_types);
    // 计算子节点哈希
    let mut child_hashes = enode
      .children()
      .iter()
      .map(|&child| egraph[child].data.hash.cls_hash.clone())
      .collect::<Vec<_>>();
    // 排序子节点哈希
    child_hashes.sort();
    // 计算当前节点哈希
    let current_level = compute_hash_level(enode, &child_hashes);
    // 合并子树层级
    let mut subtree_levels = bitvec![u64, Lsb0; 0; 64];
    subtree_levels.set(current_level, true);
    for child in enode.children() {
      let child_level = egraph[*child].data.hash.subtree_levels.clone();
      subtree_levels |= child_level;
    }
    // 计算完整哈希
    let extract_hash = compute_full_hash(enode, &child_hashes);
    let hash = StructuralHash {
      cls_hash: extract_hash,
      subtree_levels,
      level_conflict: LevelConflictState::new(),
    };
    let x = |i: &Id| &egraph[*i].data.cs;

    let self_ref = &egraph.analysis;

    match Teachable::as_binding_expr(enode) {
      Some(BindingExpr::Lib(id, f, b, gain, cost)) => {
        // This is a lib binding!
        // cross e1, e2 and introduce a lib!
        // println!("before adding lib: {:#?}", x(b));
        let mut e =
          x(b).add_lib(id, gain, cost, x(f), self_ref.lps, self_ref.strategy);
        // println!("new cost set: {:#?}", e);
        e.unify();
        e.prune(self_ref.beam_size, self_ref.lps);
        ISAXCost::new(e, ty, hash)
      }
      Some(_) | None => {
        // This is some other operation of some kind.
        // We test the arity of the function
        if enode.is_empty() {
          // 0 args. Return new.
          ISAXCost::new(CostSet::new(), ty, hash)
        } else if enode.args().len() == 1 {
          // 1 arg. Get child cost set, inc, and return.
          let e = x(&enode.args()[0]).clone();
          ISAXCost::new(e, ty, hash)
        } else {
          // 2+ args. Cross/unify time!
          let mut e = x(&enode.args()[0]).clone();

          for cs in &enode.args()[1..] {
            e = e.cross(x(cs), self_ref.lps, self_ref.strategy);
            // Intermediate prune.
            e.unify();
            e.prune(self_ref.inter_beam, self_ref.lps);
          }

          e.unify();
          e.prune(self_ref.beam_size, self_ref.lps);
          ISAXCost::new(e, ty, hash)
        }
      }
    }
  }
}

/// Library context is a set of library function names.
/// It is used in the extractor to represent the fact that we are extracting
/// inside (nested) library definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LibContext {
  set: Vec<LibId>,
  hash: u32,
}

impl LibContext {
  fn new() -> Self {
    let mut ctx = Self {
      set: Vec::new(),
      hash: 0,
    };
    ctx.cal_hash();
    ctx
  }

  /// Add a new lib to the context if not yet present,
  /// keeping it sorted.
  fn add(&mut self, lib_id: LibId) {
    if let Err(pos) = self.set.binary_search(&lib_id) {
      self.set.insert(pos, lib_id);
      self.cal_hash();
    }
  }

  /// Does the context contain the given lib?
  fn contains(&self, lib_id: LibId) -> bool {
    self.set.binary_search(&lib_id).is_ok()
  }

  /// Calculate hash
  fn cal_hash(&mut self) {
    let mut hasher = DefaultHasher::new();
    self.set.hash(&mut hasher);
    let full_hash = hasher.finish();
    self.hash = (full_hash ^ (full_hash >> 32)) as u32;
  }
}

impl Hash for LibContext {
  fn hash<H: Hasher>(&self, state: &mut H) {
    // 如果设计正确，这里应该用预计算的hash
    // 但需要确保dirty时为false！
    state.write_u32(self.hash);
  }
}

type MaybeExpr<Op> = Option<RecExpr<AstNode<Op>>>;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
  id: u32,   // 假设Id是u32或类似
  hash: u32, // 如果可能，减小哈希位数
}

/// Extractor that minimizes AST size but ignores the cost of library
/// definitions (which will be later lifted to the top).
/// The main difference between this and a standard extractor is that
/// instead of finding the best expression *per eclass*,
/// we need to find the best expression *per eclass and lib context*.
/// This is because when extracting inside library definitions,
/// we are not allowed to use those libraries;
/// so the best expression is different depending on which library defs we are
/// currently inside.
#[derive(Debug)]
pub struct LibExtractor<
  'a,
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display,
  N: Analysis<AstNode<Op>>,
> {
  /// Remembers the best expression so far for each pair of class id and lib
  /// context; if an entry is absent, we haven't visited this class in this
  /// context yet; if an entry is `None`, it's currently under processing,
  /// but we have no results for it yet; if an entry is `Some(_)`, we have
  /// found an expression for it (but it might still be improved).
  memo: FxHashMap<CacheKey, usize>,
  all_exprs: Vec<MaybeExpr<Op>>,
  all_costs: Vec<Option<f32>>,
  /// Current lib context:
  /// contains all lib ids inside whose definitions we are currently
  /// extracting.
  lib_context: LibContext,
  /// The egraph to extract from.
  egraph: &'a EGraph<AstNode<Op>, N>,
  /// This is here for pretty debug messages.
  indent: usize,
  /// The relative weight of area cost and delay cost, from 0.0 (all areaa) to
  /// 1.0 (all delay)
  strategy: f32,
}

impl<'a, Op, N> LibExtractor<'a, Op, N>
where
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + Arity,
  N: Analysis<AstNode<Op>> + Clone,
{
  /// Create a lib extractor for the given egraph
  pub fn new(egraph: &'a EGraph<AstNode<Op>, N>, strategy: f32) -> Self {
    Self {
      memo: FxHashMap::default(),
      all_exprs: Vec::with_capacity(10000),
      all_costs: Vec::with_capacity(10000),
      lib_context: LibContext::new(),
      egraph,
      indent: 0,
      strategy,
    }
  }

  /// Get best best expression for `id` in the current lib context.
  fn get_from_memo(&self, id: Id) -> Option<&usize> {
    // let start = Instant::now();
    let index = self.memo.get(&CacheKey {
      id: usize::from(id) as u32,
      hash: self.lib_context.hash,
    });
    // println!("get_from_memo: {}ms", start.elapsed().as_millis());
    index
  }

  /// Set best best expression for `id` in the current lib context.
  fn insert_into_memo(
    &mut self,
    id: Id,
    val: MaybeExpr<Op>,
    cost: Option<f32>,
  ) {
    // let start = Instant::now();
    // 使用prune进行修剪
    let new_val = self.prune(val.clone());
    self.all_exprs.push(new_val);
    self.all_costs.push(cost);
    self.memo.insert(
      CacheKey {
        id: usize::from(id) as u32,
        hash: self.lib_context.hash,
      },
      self.all_exprs.len() - 1,
    );
    // println!("inserted into memo: {}ms", start.elapsed().as_millis());
  }

  /// prune using egraph
  fn prune(&self, val: MaybeExpr<Op>) -> MaybeExpr<Op> {
    // 如果val为None，直接返回
    if val.is_none() {
      return val;
    }
    // 如果val不为None,取出RecExpr
    let timeout = std::time::Duration::from_millis(60);
    let expr = val.clone().unwrap();
    let runner = Runner::<_, _, ()>::new(EmptyAnalysis::default())
      .with_expr(&expr)
      .with_time_limit(timeout)
      .with_iter_limit(4)
      .run(vec![]);
    // 提取出最优的表达式
    let egraph = runner.egraph;
    let extractor = Extractor::new(&egraph, AstSize);
    Some(extractor.find_best(runner.roots[0]).1)
  }

  /// Extract the smallest expression for the eclass `id`.
  /// # Panics
  /// Panics if extraction fails
  /// (this should never happen because the e-graph must contain a non-cyclic
  /// expression)
  pub fn best(&mut self, id: Id) -> RecExpr<AstNode<Op>> {
    // Populate the memo:
    //println!("extracting eclass {id}");
    self.extract(id);
    // println!("id: {:#?}", id);
    // println!("{:#?}", self.egraph[id]);
    // Get the best expression from the memo:
    let index = self.get_from_memo(id).unwrap().clone();
    self.all_exprs[index]
      .clone()
      .expect("Failed to extract expression")
  }

  /// Expression gain used by this extractor
  pub fn cost(&self, expr: &RecExpr<AstNode<Op>>) -> f32 {
    let (latency_gain, area) = rec_cost(expr);
    // println!("used {}ms to get the cost", start.elapsed().as_millis());
    (1.0 - self.strategy) * (area as f32)
      - self.strategy * (latency_gain as f32)
  }

  /// Extract the expression with the largest gain from the eclass id and its
  /// descendants in the current context, storing results in the memo
  fn extract(&mut self, id: Id) {
    // let extract_start = Instant::now();
    // println!("---------------memo.size(): {}", self.memo.len());
    self.debug_indented(&format!("extracting eclass {id}"));
    if self.get_from_memo(id) == None {
      // Initialize memo with None to prevent infinite recursion in case of
      // cycles in the egraph
      self.insert_into_memo(id, None, None);
      // Extract a candidate expression from each node
      // println!("Extracting eclass {:#?}", self.egraph[id]);
      // let mut cnt = 0;
      for node in self.egraph[id].iter() {
        // println!("extracting node {}/{}", cnt, self.egraph[id].len());
        // cnt += 1;
        match self.extract_node(node) {
          None => (), // Extraction for this node failed (must be a cycle)
          Some(cand) => {
            // Extraction succeeded: check if cand is better than what we have
            // so far print!("eclss: {}, ", id);
            // print!("node: {}, ", node.operation());
            // print!("cand: {}, ", cand.pretty(100));
            // println!("cand gain: {}", Self::gain(&self, &cand));
            let mut flag = true;
            let mut cand_cost = None;
            let mut prev_msg = (0, None);
            let mut renew_flag = false;
            // 首先，如果self.get_from_memo(id) 有值，并且all_exprs中有值
            if let Some(index) = self.get_from_memo(id) {
              // println!("index: {}", index);
              if let Some(prev) = self.all_exprs[*index].clone() {
                // 存在表达式，首先计算prev的cost，
                // 如果all_costs中有值就不用计算，直接取出
                let prev_cost = if let Some(cost) = self.all_costs[*index] {
                  cost
                } else {
                  let cost = Self::cost(&self, &prev);
                  prev_msg = (index.clone(), Some(cost));
                  renew_flag = true;
                  cost
                };
                // 接下来计算cand的cost，并赋给cand_cost
                let c_cost = Self::cost(&self, &cand);
                // 如果prev_cost < c_cost，说明cand的cost更大,
                // 将flag置为false,否则就将c_cost赋给cand_cost
                if prev_cost < c_cost {
                  flag = false;
                } else {
                  cand_cost = Some(c_cost);
                }
              }
            }
            if renew_flag {
              self.all_costs[prev_msg.0] = prev_msg.1;
            }
            // 如果flag为true，说明cand的cost更小,需要进行替换
            if flag {
              self.insert_into_memo(id, Some(cand.clone()), cand_cost);
            }
            // match self.get_from_memo(id) {
            //   // If we already had an expression and it was better, do
            // nothing   Some(index){
            //     // 首先检查all_cost中是否有值
            //     if let Some(cost) = self.all_costs[*index] {
            //       if cost < Self::cost(&self, &cand) {
            //         flag = false;
            //       }
            //     } else {
            //       // 如果没有值，检查
            //     }

            //   }
            // }
            //   // Otherwise, update the memo;
            //   // note that updating the memo after each better candidate is
            //   // found instead of at the end is slightly
            //   // suboptimal (because it might cause us to go around some
            // cycles   // once), but the code is simpler and it
            // doesn't   // matter too much.
            //   _ => {
            //     self.debug_indented(&format!(
            //       "new best for {id}: {} (gain {})",
            //       cand.pretty(100),
            //       Self::cost(&self, &cand)
            //     ));
            //     let start = Instant::now();
            //     self.insert_into_memo(id, Some(cand));
            //     println!(
            //       "using {}ms to insert into memo",
            //       start.elapsed().as_millis()
            //     );
            //   }
          }
        }
      }
      // println!(
      //   "using {}ms to extract eclass {id}",
      //   extract_start.elapsed().as_millis()
      // );
    }
  }

  /// Extract the smallest expression from `node`.
  fn extract_node(&mut self, node: &AstNode<Op>) -> MaybeExpr<Op> {
    self.debug_indented(&format!("extracting node {node:?}"));
    if let Some(BindingExpr::Lib(lid, _, _, _, _)) = node.as_binding_expr() {
      // println!("checking lib {lid}");
      if self.lib_context.contains(lid) {
        // This node is a definition of one of the libs, whose definition we are
        // currently extracting: do not go down this road since it leads
        // to lib definitions using themselves
        self.debug_indented(&format!("encountered banned lib: {lid}"));
        return None;
      }
    }
    // Otherwise: extract all children
    let mut child_indexes = vec![];
    // println!("extracting children of {node:?}");
    self.extract_children(node, 0, vec![], &mut child_indexes)
  }

  /// Process the children of `node` starting from index `current`
  /// and accumulate results in `partial expr`;
  /// `child_indexes` stores the indexes of already processed children within
  /// `partial_expr`, so that we can use them in the `AstNode` at the end.
  fn extract_children(
    &mut self,
    node: &AstNode<Op>,
    current: usize,
    mut partial_expr: Vec<AstNode<Op>>,
    child_indexes: &mut Vec<usize>,
  ) -> MaybeExpr<Op> {
    // let child_start = Instant::now();
    // println!("begin extracting children");
    if current == node.children().len() {
      // println!("current == children.len()");
      // Done with children: add ourselves to the partial expression and return
      let child_ids: Vec<Id> =
        child_indexes.iter().map(|x| (*x).into()).collect();
      let root = AstNode::new(node.operation().clone(), child_ids);
      partial_expr.push(root);
      // println!(
      //   "using {}ms to extract children",
      //   child_start.elapsed().as_millis()
      // );
      Some(partial_expr.into())
    } else {
      // If this is the first child of a lib node (i.e. lib definition) add this
      // lib to the context:
      let old_lib_context = self.lib_context.clone();
      if let Some(BindingExpr::Lib(lid, _, _, _, _)) = node.as_binding_expr() {
        if current == 0 {
          self.debug_indented(&format!(
            "processing first child of {node:?}, adding {lid} to context"
          ));
          self.lib_context.add(lid);
        }
      }

      // Process the current child
      let child = &node.children()[current];
      self.indent += 1;
      // println!(">>>extracting child {child:?}");
      self.extract(*child);
      self.indent -= 1;
      // We need to get the result before restoring the context
      // println!("begin getting child {child:?}");
      // let start = Instant::now();
      let child_res = self.get_from_memo(*child).clone();
      let child_res = if let Some(index) = child_res {
        self.all_exprs[*index].clone()
      } else {
        None
      };
      // println!("using {}ms to get the expr", start.elapsed().as_millis());
      // Restore lib context
      self.lib_context = old_lib_context;
      // println!("begin matching");
      match child_res {
        None => None, /* Failed to extract a child, so the extraction of */
        // this node fails
        Some(expr) => {
          // We need to clone the expr because we're going to mutate it (offset
          // child indexes), and we don't want it to affect the memo
          // result for child.
          let mut new_expr = expr.as_ref().to_vec();
          // println!("begin offsetting");
          for n in &mut new_expr {
            // Increment all indexes inside `n` by the current expression
            // length; this is needed to make a well-formed
            // `RecExpr`
            Self::offset_children(n, partial_expr.len());
          }
          // println!(">>>>> new expr");
          partial_expr.extend(new_expr);
          child_indexes.push(partial_expr.len() - 1);
          let exp = self.extract_children(
            node,
            current + 1,
            partial_expr,
            child_indexes,
          );
          // println!(">>>>>>> returning from child {child:?}");
          // println!(
          //   "using {}ms to extract children",
          //   child_start.elapsed().as_millis()
          // );
          exp
        }
      }
    }
  }

  /// Add `offset` to all children of `node`
  fn offset_children(node: &mut AstNode<Op>, offset: usize) {
    for child in node.children_mut() {
      let child_index: usize = (*child).into();
      *child = (child_index + offset).into();
    }
  }

  /// Print a debug message with the current indentation
  /// TODO: this should be a macro
  fn debug_indented(&self, msg: &str) {
    debug!("{:indent$}{msg}", "", indent = 2 * self.indent);
  }
}

// 定义trait TypeInfo
pub trait TypeInfo<T> {
  fn get_rtype(
    &self,
    type_info_map: &HashMap<(String, Vec<T>), T>,
    child_types: &Vec<T>,
  ) -> T;
  fn merge_types(a: &T, b: &T) -> T;
}

// 定义GetType trait
pub trait ClassMatch {
  fn get_type(&self) -> String {
    "".to_string()
  }
  fn get_cls_hash(&self) -> u64 {
    0
  }
  fn get_subtree_levels(&self) -> u64 {
    0
  }
  fn get_pattern(&self, _other: &Self) -> BitVec<u64, Lsb0> {
    bitvec!(u64, Lsb0; 0; 64)
  }
}

// 为ISAXCost实现ClassMatch trait
impl<T: PartialEq + Debug + Default + Clone + Ord + Hash + ToString> ClassMatch
  for ISAXCost<T>
{
  // fn type_match(&self, other: &Self) -> bool {
  //   self.ty == other.ty
  // }
  // fn level_match(&self, other: &Self) -> bool {
  //   let hash_similar =
  //     hamming_distance(self.hash.cls_hash, other.hash.cls_hash) < 36;
  //   let subtree_similar =
  //     jaccard_similarity(&self.hash.subtree_levels,
  // &other.hash.subtree_levels)
  //       > 0.67;
  //   hash_similar && subtree_similar
  // }
  fn get_type(&self) -> String {
    self.ty.to_string()
  }
  fn get_cls_hash(&self) -> u64 {
    self.hash.cls_hash
  }
  fn get_subtree_levels(&self) -> u64 {
    self.hash.subtree_levels.load()
  }
  fn get_pattern(&self, other: &Self) -> BitVec<u64, LocalBits> {
    self.hash.subtree_levels.clone() & other.hash.subtree_levels.clone()
  }
}
