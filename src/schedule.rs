//! A simple HLS scheduler used for estimating the area and latency of the
//! learned libraries.

use std::{cmp, collections::HashSet, fmt::Debug, hash::Hash};

use crate::{
  BindingExpr, LibId, Teachable, ast_node::AstNode, bb_query::BBQuery,
};
use egg::RecExpr;

/// A trait for languages that support HLS scheduling.
pub trait Schedulable<LA, LD>
where
  Self: Sized,
{
  /// Returns the delay of the operation.
  #[must_use]
  fn op_delay(&self, ld: &LD, args: Vec<Self>) -> usize;

  /// Returns the area of the operation.
  #[must_use]
  fn op_area(&self, la: &LA, args: Vec<Self>) -> usize;

  /// Returns the latency of the operation.
  #[must_use]
  fn op_latency(&self) -> usize;

  /// Returns the latency of the operation in the case of running on cpu
  #[must_use]
  fn op_latency_cpu(&self, bb_query: &BBQuery) -> usize;

  /// Returns the execution count of the operation.
  #[must_use]
  fn op_execution_count(&self, bb_query: &BBQuery) -> usize;
}

impl<Op> AstNode<Op>
where
  Op: Clone,
{
  pub fn get_op_args(&self, expr: &RecExpr<Self>) -> Vec<Self> {
    self
      .args()
      .iter()
      .map(|id| {
        let idx: usize = (*id).into();
        expr[idx].clone()
      })
      .collect::<Vec<Self>>()
  }
}

/// A struct representing a scheduler which used to estimate the area and
/// latency of the learned libraries.
#[derive(Clone)]
pub struct Scheduler<LA, LD> {
  /// The clock period of timing scheduling.
  clock_period: usize,
  /// The area estimator.
  area_estimator: LA,
  /// The delay estimator.
  delay_estimator: LD,
  /// BB query for the CPU latency.
  bb_query: BBQuery,
  // Maybe more parameters in the future.
}

/// The result type of scheduling operations, containing (gain, cost), where
/// gain = latency_cpu - latency_accelerator, cost = area.
pub type ScheduleResult = (usize, usize);

impl<LA, LD> Scheduler<LA, LD> {
  pub fn new(
    clock_period: usize,
    area_estimator: LA,
    delay_estimator: LD,
    bb_query: BBQuery,
  ) -> Self {
    Self {
      clock_period,
      area_estimator,
      delay_estimator,
      bb_query,
    }
  }

  pub fn asap_schedule<Op>(&self, expr: &RecExpr<AstNode<Op>>) -> ScheduleResult
  where
    Op: Eq + Hash + Clone + Teachable + Debug,
    AstNode<Op>: Schedulable<LA, LD>,
  {
    // println!("Scheduling...");
    // Start cycle of each AST node
    let mut sc: Vec<usize> = vec![0; expr.len()];
    // Start time in the cycle of each AST node
    let mut stic: Vec<usize> = vec![0; expr.len()];
    let mut scheduled: Vec<bool> = vec![false; expr.len()];
    let mut unscheduled_count = expr.len();

    // ASAP scheduling
    let mut cycle = 0;
    while unscheduled_count > 0 {
      for i in 0..expr.len() {
        if !scheduled[i] {
          let node = &expr[i];
          // Check if the operation is ready to be scheduled and calculate the
          // earliest time in the cycle the operation can start
          let mut ready = true;
          let mut earlist_stic = 0;
          for &child in node.args() {
            let idx: usize = child.into();
            if !scheduled[idx] {
              ready = false;
              break;
            } else {
              let dep = &expr[idx];
              let mut delay =
                dep.op_delay(&self.delay_estimator, dep.get_op_args(expr));
              let mut latency = dep.op_latency();
              latency += delay / self.clock_period;
              delay %= self.clock_period;
              let is_sequential = latency > 0;
              if is_sequential {
                // Check if the dependent operation is ready
                if sc[idx] + latency > cycle {
                  ready = false;
                  break;
                } else if sc[idx] + latency == cycle {
                  earlist_stic = cmp::max(earlist_stic, delay);
                }
              } else {
                if sc[idx] == cycle {
                  earlist_stic = cmp::max(earlist_stic, stic[idx] + delay);
                }
              }
            }
          }
          if ready {
            let mut delay =
              node.op_delay(&self.delay_estimator, node.get_op_args(expr));
            let mut latency = node.op_latency();
            latency += delay / self.clock_period;
            delay %= self.clock_period;
            let is_sequential = latency > 0;
            match is_sequential {
              true => {
                sc[i] = cycle;
                stic[i] = earlist_stic;
                scheduled[i] = true;
                unscheduled_count -= 1;
              }
              false => {
                if earlist_stic + delay <= self.clock_period {
                  sc[i] = cycle;
                  stic[i] = earlist_stic;
                  scheduled[i] = true;
                  unscheduled_count -= 1;
                }
              }
            }
          }
        }
      }
      cycle += 1;
    }
    // Calculate the gain
    // println!("Calculating gain...");
    // println!("Calculating gain...");
    let mut latency_accelerator = 0;
    for i in 0..expr.len() {
      let node = &expr[i];
      let delay = node.op_delay(&self.delay_estimator, node.get_op_args(expr));
      let mut latency = node.op_latency();
      latency += delay / self.clock_period;
      let is_sequential = latency > 0;
      if is_sequential {
        latency_accelerator = cmp::max(latency_accelerator, sc[i] + latency);
      } else {
        latency_accelerator = cmp::max(latency_accelerator, sc[i] + 1);
      }
    }
    let mut latency_cpu = 0;
    for node in expr {
      latency_cpu += node.op_latency_cpu(&self.bb_query);
    }
    // Calculate the area
    // println!("Calculating area...");
    let mut area = 0;
    for node in expr {
      area += node.op_area(&self.area_estimator, node.get_op_args(expr));
    }
    // Calculate the execution count
    let mut execution_count = 0;
    for node in expr {
      execution_count =
        cmp::max(execution_count, node.op_execution_count(&self.bb_query));
    }
    // println!("Area: {}", area);
    if latency_accelerator > latency_cpu {
      (0, area)
    } else {
      ((latency_cpu - latency_accelerator) * execution_count, area)
    }
  }
}

/// Calculate the cost of an expression.
pub fn rec_cost<Op: Teachable>(expr: &RecExpr<AstNode<Op>>) -> (usize, usize) {
  let mut used_lib: HashSet<LibId> = HashSet::new();
  let mut latency_gain: usize = 0;
  let mut area: usize = 0;
  for node in expr.iter() {
    if let Some(BindingExpr::Lib(lid, _, _, gain, cost)) =
      node.as_binding_expr()
    {
      latency_gain += gain;
      if used_lib.insert(lid) {
        area += cost;
      }
    }
  }
  (latency_gain, area)
}
