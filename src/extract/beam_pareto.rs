//! `extract::beam_pareto` implements a Pareto-optimal beam search extractor
//! that considers both area and delay costs.

use egg::{Analysis, CostFunction, DidMerge, EGraph, Id, Language, RecExpr};
use log::debug;
use std::{
  cmp::Ordering,
  collections::{BinaryHeap, HashMap},
  fmt::Debug,
};

use crate::{
  ast_node::{Arity, AstNode},
  learn::LibId,
  teachable::{BindingExpr, Teachable},
  extract::beam::OptimizationStrategy,
};

/// A `LibSelAreaDelay` is a selection of library functions, paired with
/// area and delay cost values for the expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibSelAreaDelay {
  /// Area cost of the expression without library functions
  pub area_cost: usize,
  /// Delay cost of the expression without library functions
  pub delay_cost: usize,
  /// Memoized area_cost + sum of lib area costs
  pub full_area_cost: usize,
  /// Memoized delay_cost (taking into account critical path)
  pub full_delay_cost: usize,
  /// The selected libraries and their costs (area, delay)
  pub libs: Vec<(LibId, usize, usize)>,
}

impl LibSelAreaDelay {
  #[must_use]
  pub fn intro_op() -> LibSelAreaDelay {
    LibSelAreaDelay {
      libs: Vec::new(),
      area_cost: 1,
      delay_cost: 1,
      full_area_cost: 1,
      full_delay_cost: 1,
    }
  }

  /// Combines two `LibSelAreaDelay`s. Unions the lib sets, adds
  /// the area costs, and takes max of delay costs.
  #[must_use]
  pub fn combine(&self, other: &LibSelAreaDelay, lps: usize) -> Option<LibSelAreaDelay> {
    let mut res = self.clone();

    for (k, area_v, delay_v) in &other.libs {
      match res.libs.binary_search_by_key(k, |(id, _, _)| *id) {
        Ok(ix) => {
          // If we already have this lib, use the better costs
          if area_v < &res.libs[ix].1 {
            res.full_area_cost -= res.libs[ix].1 - *area_v;
            res.libs[ix].1 = *area_v;
          }
          if delay_v < &res.libs[ix].2 {
            // Only update delay if it's on the critical path
            if res.libs[ix].2 == res.full_delay_cost {
              // Recalculate full delay cost
              res.full_delay_cost = res.delay_cost.max(
                res.libs.iter()
                  .map(|(_, _, d)| *d)
                  .max()
                  .unwrap_or(0)
              );
            }
            res.libs[ix].2 = *delay_v;
          }
        }
        Err(ix) => {
          res.full_area_cost += *area_v;
          res.libs.insert(ix, (*k, *area_v, *delay_v));
          // Update delay if this new lib is on the critical path
          res.full_delay_cost = res.full_delay_cost.max(*delay_v);
          if res.libs.len() > lps {
            return None;
          }
        }
      }
    }

    // Update area cost
    res.area_cost = self.area_cost + other.area_cost;
    res.full_area_cost += other.area_cost;

    // Update delay cost (critical path)
    res.delay_cost = self.delay_cost.max(other.delay_cost);
    res.full_delay_cost = res.full_delay_cost.max(other.full_delay_cost);

    Some(res)
  }

  #[must_use]
  pub fn add_lib(
    &self,
    lib: LibId,
    cost: &LibSelAreaDelay,
    lps: usize,
  ) -> Option<LibSelAreaDelay> {
    let mut res = self.clone();
    let area_v = cost.area_cost + 1; // +1 for the lib node area
    let delay_v = cost.delay_cost + 1; // +1 for the lib node delay
    let mut full_area_cost = res.full_area_cost;
    let mut full_delay_cost = res.full_delay_cost;

    // Add all nested libs that the lib uses, then add the lib itself.
    for (nested_lib, nested_area, nested_delay) in &cost.libs {
      let nested_lib = *nested_lib;
      let nested_area = *nested_area;
      let nested_delay = *nested_delay;

      match res.libs.binary_search_by_key(&nested_lib, |(id, _, _)| *id) {
        Ok(ix) => {
          if nested_area < res.libs[ix].1 {
            full_area_cost -= res.libs[ix].1 - nested_area;
            res.libs[ix].1 = nested_area;
          }
          if nested_delay < res.libs[ix].2 {
            // Only update full delay if this lib was on the critical path
            if res.libs[ix].2 == full_delay_cost {
              // Recalculate full delay
              full_delay_cost = res.delay_cost.max(
                res.libs.iter()
                  .filter(|(id, _, _)| *id != nested_lib)
                  .map(|(_, _, d)| *d)
                  .max()
                  .unwrap_or(0)
                  .max(nested_delay)
              );
            }
            res.libs[ix].2 = nested_delay;
          }
        }
        Err(ix) => {
          full_area_cost += nested_area;
          res.libs.insert(ix, (nested_lib, nested_area, nested_delay));
          // Update delay if this new lib is on the critical path
          full_delay_cost = full_delay_cost.max(nested_delay);
          if res.libs.len() > lps {
            return None;
          }
        }
      }
    }

    match res.libs.binary_search_by_key(&lib, |(id, _, _)| *id) {
      Ok(ix) => {
        if area_v < res.libs[ix].1 {
          full_area_cost -= res.libs[ix].1 - area_v;
          res.libs[ix].1 = area_v;
        }
        if delay_v < res.libs[ix].2 {
          // Only update full delay if this lib was on the critical path
          if res.libs[ix].2 == full_delay_cost {
            // Recalculate full delay
            full_delay_cost = res.delay_cost.max(
              res.libs.iter()
                .filter(|(id, _, _)| *id != lib)
                .map(|(_, _, d)| *d)
                .max()
                .unwrap_or(0)
                .max(delay_v)
            );
          }
          res.libs[ix].2 = delay_v;
        }
      }
      Err(ix) => {
        full_area_cost += area_v;
        res.libs.insert(ix, (lib, area_v, delay_v));
        // Update delay if this new lib is on the critical path
        full_delay_cost = full_delay_cost.max(delay_v);
        if res.libs.len() > lps {
          return None;
        }
      }
    }

    res.full_area_cost = full_area_cost;
    res.full_delay_cost = full_delay_cost;
    Some(res)
  }

  pub fn inc_cost(&mut self) {
    self.area_cost += 1;
    self.full_area_cost += 1;
    self.delay_cost += 1;
    self.full_delay_cost += 1;
  }

  /// O(n) subset check
  #[must_use]
  pub fn is_subset(&self, other: &LibSelAreaDelay) -> bool {
    let mut oix = 0;

    // For every element in this LibSel, we want to see
    // if it exists in other.
    'outer: for (lib, _, _) in &self.libs {
      loop {
        // If oix is beyond the length of other, return false.
        if oix >= other.libs.len() {
          return false;
        }

        match &other.libs[oix].0.cmp(lib) {
          Ordering::Less => {
            // Increment oix by default
            oix += 1;
          }
          Ordering::Equal => {
            // If other[oix] is equal to lib, continue in the outer loop and increment oix
            oix += 1;
            continue 'outer;
          }
          Ordering::Greater => {
            // Otherwise if it's larger, there was no element equal. Not subset, ret false.
            return false;
          }
        }
      }
    }

    // We made it! ret true
    true
  }

  /// Calculate the combined cost based on the optimization strategy
  pub fn combined_cost(&self, strategy: &OptimizationStrategy) -> f64 {
    match strategy {
      OptimizationStrategy::MinArea => self.full_area_cost as f64,
      OptimizationStrategy::MinDelay => self.full_delay_cost as f64,
      OptimizationStrategy::Balanced(weight) => {
        // Normalize area and delay to be between 0 and 1
        // This is a simple approach - in practice you might want to use
        // more sophisticated normalization based on the range of values
        let area_norm = self.full_area_cost as f64;
        let delay_norm = self.full_delay_cost as f64;
        
        // Weighted sum
        (1.0 - weight) * area_norm + *weight * delay_norm
      }
    }
  }

  /// Dominates check for Pareto optimality
  /// Returns true if self dominates other (self is better in at least one dimension and not worse in any)
  pub fn dominates(&self, other: &LibSelAreaDelay) -> bool {
    // Check if self is better or equal in all dimensions
    let area_better_eq = self.full_area_cost <= other.full_area_cost;
    let delay_better_eq = self.full_delay_cost <= other.full_delay_cost;
    
    // And strictly better in at least one dimension
    let strictly_better = self.full_area_cost < other.full_area_cost || 
                          self.full_delay_cost < other.full_delay_cost;
    
    area_better_eq && delay_better_eq && strictly_better
  }
}

/// A wrapper around `LibSelAreaDelay`s that orders based on combined cost.
#[derive(Debug, Clone)]
struct LibSelCombined<'a>(pub(crate) LibSelAreaDelay, &'a OptimizationStrategy);

impl<'a> PartialEq for LibSelCombined<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<'a> Eq for LibSelCombined<'a> {}

impl<'a> PartialOrd for LibSelCombined<'a> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.0.combined_cost(self.1).partial_cmp(&other.0.combined_cost(self.1))
  }
}

impl<'a> Ord for LibSelCombined<'a> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.partial_cmp(other).unwrap_or(Ordering::Equal)
  }
}

/// A `CostSetAreaDelay` is a set of pairs; each pair contains a set of library
/// functions paired with the area and delay costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostSetAreaDelay {
  /// The set of library selections and their associated costs.
  pub set: Vec<LibSelAreaDelay>,
}

impl CostSetAreaDelay {
  /// Creates a `CostSetAreaDelay` corresponding to introducing a nullary operation.
  #[must_use]
  pub fn intro_op() -> CostSetAreaDelay {
    let mut set = Vec::with_capacity(10);
    set.push(LibSelAreaDelay::intro_op());
    CostSetAreaDelay { set }
  }

  /// Crosses over two `CostSetAreaDelay`s.
  #[must_use]
  pub fn cross(&self, other: &CostSetAreaDelay, lps: usize) -> CostSetAreaDelay {
    let mut set: Vec<LibSelAreaDelay> = Vec::new();

    for ls1 in &self.set {
      for ls2 in &other.set {
        match ls1.combine(ls2, lps) {
          None => continue,
          Some(ls) => {
            // Check if this is dominated by any existing solution
            let mut is_dominated = false;
            for existing in &set {
              if existing.dominates(&ls) {
                is_dominated = true;
                break;
              }
            }
            
            if !is_dominated {
              // Remove any solutions that this dominates
              set.retain(|existing| !ls.dominates(existing));
              set.push(ls);
            }
          }
        }
      }
    }

    CostSetAreaDelay { set }
  }

  /// Combines two `CostSetAreaDelay`s by unioning them together.
  pub fn combine(&mut self, other: CostSetAreaDelay) {
    for elem in other.set {
      // Check if this is dominated by any existing solution
      let mut is_dominated = false;
      for existing in &self.set {
        if existing.dominates(&elem) {
          is_dominated = true;
          break;
        }
      }
      
      if !is_dominated {
        // Remove any solutions that this dominates
        self.set.retain(|existing| !elem.dominates(existing));
        self.set.push(elem);
      }
    }
  }

  /// Performs Pareto-optimal unification
  pub fn unify(&mut self) {
    let mut i = 0;
    
    while i < self.set.len() {
      let mut j = i + 1;
      
      while j < self.set.len() {
        let ls1 = &self.set[i];
        let ls2 = &self.set[j];
        
        if ls1.dominates(ls2) {
          self.set.remove(j);
        } else if ls2.dominates(ls1) {
          self.set.remove(i);
          j = i + 1; // Reset j since i was removed
        } else {
          j += 1;
        }
      }
      i += 1;
    }
  }

  /// Increments the area and delay cost of every `LibSelAreaDelay` in this `CostSetAreaDelay`.
  pub fn inc_cost(&mut self) {
    for ls in &mut self.set {
      ls.inc_cost();
    }
  }

  #[must_use]
  pub fn add_lib(&self, lib: LibId, cost: &CostSetAreaDelay, lps: usize) -> CostSetAreaDelay {
    let mut set: Vec<LibSelAreaDelay> = Vec::new();

    for ls1 in &cost.set {
      if ls1.libs.iter().any(|(l, _, _)| l == &lib) {
        // If this libsel contains the lib we are defining,
        // we can't use it in the definition.
        continue;
      }
      for ls2 in &self.set {
        match ls2.add_lib(lib, ls1, lps) {
          None => continue,
          Some(ls) => {
            // Check if this is dominated by any existing solution
            let mut is_dominated = false;
            for existing in &set {
              if existing.dominates(&ls) {
                is_dominated = true;
                break;
              }
            }
            
            if !is_dominated {
              // Remove any solutions that this dominates
              set.retain(|existing| !ls.dominates(existing));
              set.push(ls);
            }
          }
        }
      }
    }

    CostSetAreaDelay { set }
  }

  /// Prune the Pareto front to keep a manageable number of solutions
  pub fn prune(&mut self, n: usize, lps: usize, strategy: &OptimizationStrategy) {
    use std::cmp::Reverse;

    if self.set.len() <= n {
      return; // No need to prune
    }

    let old_set = std::mem::take(&mut self.set);

    // First, we create a table from # of libs to a list of LibSels
    let mut table: HashMap<usize, BinaryHeap<Reverse<LibSelCombined>>> =
      HashMap::new();

    // We then iterate over all of the LibSels in this set
    for ls in old_set {
      let num_libs = ls.libs.len();
      let h = table.entry(num_libs).or_default();
      h.push(Reverse(LibSelCombined(ls, strategy)));
    }

    // From our table, recombine into a vector
    let mut set = Vec::new();
    let beams_per_size = std::cmp::max(1, n / lps);

    for (_sz, mut h) in table {
      // Take the first n items from the heap
      let mut i = 0;
      while i < beams_per_size {
        if let Some(ls) = h.pop() {
          set.push(ls.0.0);
          i += 1;
        } else {
          break;
        }
      }
    }

    self.set = set;
  }
}

/// Analysis that uses Pareto-optimal beam search with area and delay costs
#[derive(Debug, Clone, Copy)]
pub struct BeamAreaDelay {
  /// The number of `LibSelAreaDelay`s to keep per `EClass`.
  beam_size: usize,
  inter_beam: usize,
  /// The maximum number of libs per lib selection.
  lps: usize,
  /// The optimization strategy to use
  strategy: OptimizationStrategy,
}

impl BeamAreaDelay {
  #[must_use]
  pub fn new(
    beam_size: usize,
    inter_beam: usize,
    lps: usize,
    strategy: OptimizationStrategy,
  ) -> BeamAreaDelay {
    BeamAreaDelay { beam_size, inter_beam, lps, strategy }
  }

  #[must_use]
  pub fn empty() -> BeamAreaDelay {
    BeamAreaDelay { 
      beam_size: 0, 
      inter_beam: 0, 
      lps: 1,
      strategy: OptimizationStrategy::Balanced(0.5),
    }
  }
}

impl Default for BeamAreaDelay {
  fn default() -> Self {
    BeamAreaDelay::empty()
  }
}

impl<Op> Analysis<AstNode<Op>> for BeamAreaDelay
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
{
  type Data = CostSetAreaDelay;

  fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
    let a0 = to.clone();

    // Merging consists of combination, followed by unification and beam pruning.
    to.combine(from.clone());
    to.unify();
    to.prune(self.beam_size, self.lps, &self.strategy);

    DidMerge(&a0 != to, to != &from)
  }

  fn make(
    egraph: &mut EGraph<AstNode<Op>, Self>,
    enode: &AstNode<Op>,
  ) -> Self::Data {
    let x = |i: &Id| &egraph[*i].data;
    let self_ref = egraph.analysis;

    match Teachable::as_binding_expr(enode) {
      Some(BindingExpr::Lib(id, f, b)) => {
        // This is a lib binding!
        let mut e = x(b).add_lib(id, x(f), self_ref.lps);
        e.unify();
        e.prune(self_ref.beam_size, self_ref.lps, &self_ref.strategy);
        e
      }
      Some(_) | None => {
        // This is some other operation of some kind.
        if enode.is_empty() {
          // 0 args. Return intro.
          CostSetAreaDelay::intro_op()
        } else if enode.args().len() == 1 {
          // 1 arg. Get child cost set, inc, and return.
          let mut e = x(&enode.args()[0]).clone();
          e.inc_cost();
          e
        } else {
          // 2+ args. Cross/unify time!
          let mut e = x(&enode.args()[0]).clone();

          for cs in &enode.args()[1..] {
            e = e.cross(x(cs), self_ref.lps);
            // Intermediate prune.
            e.unify();
            e.prune(self_ref.inter_beam, self_ref.lps, &self_ref.strategy);
          }

          e.unify();
          e.prune(self_ref.beam_size, self_ref.lps, &self_ref.strategy);
          e.inc_cost();
          e
        }
      }
    }
  }
}

/// Cost function that considers both area and delay for extraction
#[derive(Debug, Clone, Copy)]
pub struct AreaDelayCost<LA, LD> {
  pub area_cost: LA,
  pub delay_cost: LD,
  pub strategy: OptimizationStrategy,
}

impl<LA, LD, Op> CostFunction<AstNode<Op>> for AreaDelayCost<LA, LD>
where
  LA: crate::extract::cost::LangCost<Op> + Clone,
  LD: crate::extract::cost::LangGain<Op> + Clone,
  Op: Ord + std::hash::Hash + Debug + Teachable + Clone,
{
  type Cost = usize;

  fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
  where
    C: FnMut(Id) -> Self::Cost,
  {
    match enode.as_binding_expr() {
      Some(BindingExpr::Lib(_, _, body)) => costs(*body),
      _ => {
        // Get costs of children
        let arg_costs: Vec<usize> = enode.args().iter().map(|&id| costs(id)).collect();
        
        // Calculate area cost
        let area = arg_costs.iter().sum::<usize>() + 
                  self.area_cost.op_cost(enode.operation(), &arg_costs);
        
        // Calculate delay cost
        let max_child_cost = arg_costs.iter().max().copied().unwrap_or(0);
        let delay = max_child_cost + 
                   self.delay_cost.op_gain(enode.operation(), &arg_costs);
        
        // Combine costs based on strategy
        match self.strategy {
          OptimizationStrategy::MinArea => area,
          OptimizationStrategy::MinDelay => delay,
          OptimizationStrategy::Balanced(weight) => {
            // Simple weighted combination
            ((1.0 - weight) * area as f64 + weight * delay as f64) as usize
          }
        }
      }
    }
  }
} 