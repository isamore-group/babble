//! A simple HLS scheduler used for estimating the area and latency of the
//! learned libraries.

use std::{cmp, collections::HashMap, hash::Hash};

use crate::ast_node::AstNode;
use egg::RecExpr;

/// A trait for languages that support HLS scheduling.
pub trait Schedulable {
  /// Returns the delay of the operation.
  #[must_use]
  fn op_delay(&self) -> usize;

  /// Returns the area of the operation.
  #[must_use]
  fn op_area(&self) -> usize;

  /// Returns the latency of the operation.
  #[must_use]
  fn op_latency(&self) -> usize;

  /// Returns if the operation is sequential.
  #[must_use]
  fn is_sequential(&self) -> bool {
    self.op_latency() > 0
  }
}

impl<Op, T> AstNode<Op, T>
where
  Op: Schedulable,
{
  /// Returns the delay of the operation.
  #[must_use]
  pub fn op_delay(&self) -> usize {
    self.operation().op_delay()
  }

  /// Returns the area of the operation.
  #[must_use]
  pub fn op_area(&self) -> usize {
    self.operation().op_area()
  }

  /// Returns the latency of the operation.
  #[must_use]
  pub fn op_latency(&self) -> usize {
    self.operation().op_latency()
  }

  /// Returns if the operation is sequential.
  #[must_use]
  pub fn is_sequential(&self) -> bool {
    self.operation().is_sequential()
  }
}

/// A struct representing a scheduler which used to estimate the area and
/// latency of the learned libraries.
#[derive(Clone, Copy)]
pub struct Scheduler {
  /// The clock period of timing scheduling.
  clock_period: usize,
  // Maybe more parameters in the future.
}

/// The result type of scheduling operations, containing (latency, area).
pub type ScheduleResult = (usize, usize);

impl Scheduler {
  pub fn new(clock_period: usize) -> Self {
    Self { clock_period }
  }

  pub fn asap_schedule<Op: Schedulable + Eq + Hash + Clone>(
    &self,
    expr: &RecExpr<AstNode<Op>>,
  ) -> ScheduleResult {
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
              if dep.is_sequential() {
                // Check if the dependent operation is ready
                if sc[idx] + dep.op_latency() > cycle {
                  ready = false;
                  break;
                } else if sc[idx] + dep.op_latency() == cycle {
                  earlist_stic = cmp::max(earlist_stic, dep.op_delay());
                }
              } else {
                if sc[idx] == cycle {
                  earlist_stic =
                    cmp::max(earlist_stic, stic[idx] + dep.op_delay());
                }
              }
            }
          }
          if ready {
            match node.is_sequential() {
              true => {
                sc[i] = cycle;
                stic[i] = earlist_stic;
                scheduled[i] = true;
                unscheduled_count -= 1;
              }
              false => {
                if earlist_stic + node.op_delay() <= self.clock_period {
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
    // Calculate the latency
    let mut latency = 0;
    for i in 0..expr.len() {
      let node = &expr[i];
      if node.is_sequential() {
        latency = cmp::max(latency, sc[i] + node.op_latency());
      } else {
        latency = cmp::max(latency, sc[i] + 1);
      }
    }
    // Calculate the area
    // For each type of operation, we record the scheduled cycles of all the AST
    // nodes with this type. For example, if we have an operations scheduled in
    // cycles 1, 2, and 3 with a latency of 2, we will record a vector of [(1,
    // 2), (2, 3), (3, 4)] for the add operation.
    let mut op_schedule: HashMap<Op, Vec<(usize, usize)>> = HashMap::new();
    for i in 0..expr.len() {
      let node = &expr[i];
      let op = node.operation();
      let start: usize = sc[i];
      let end: usize = if node.is_sequential() {
        sc[i] + node.op_latency() - 1
      } else {
        sc[i]
      };
      let entry = op_schedule.entry(op.clone()).or_insert(vec![]);
      entry.push((start, end));
    }
    let mut area = 0;
    for (op, schedule) in op_schedule.iter() {
      let mut op_count = 0;
      for i in 0..latency {
        let mut cycle_count = 0;
        for (start, end) in schedule.iter() {
          if *start <= i && i <= *end {
            cycle_count += 1;
          }
        }
        op_count = cmp::max(op_count, cycle_count);
      }
      area += op_count * op.op_area();
    }
    (latency, area)
  }
}
