//! Cost models for extraction

use crate::{Teachable, ast_node::AstNode, teachable::BindingExpr};
use egg::CostFunction;
use std::fmt::Debug;

/// Trait for language-specific area cost
pub trait LangCost<Op>: Debug + Clone + Send + Sync {
  /// Calculate the area cost of an operation
  fn op_cost(&self, op: &Op, args: &[usize]) -> usize;
}

/// Trait for language-specific delay gain
pub trait LangGain<Op>: Debug + Clone + Send + Sync {
  /// Calculate the delay gain of an operation
  fn op_gain(&self, op: &Op, args: &[usize]) -> usize;
}

/// Cost function that uses LangCost to calculate area
#[derive(Debug, Clone)]
pub struct AreaCost<L, Op> {
  pub lang_cost: L,
  _phantom: std::marker::PhantomData<Op>,
}

impl<L, Op> AreaCost<L, Op>
where
  L: LangCost<Op>,
  Op: Clone + Debug,
{
  pub fn new(lang_cost: L) -> Self {
    Self {
      lang_cost,
      _phantom: std::marker::PhantomData,
    }
  }
}

impl<L, Op> CostFunction<AstNode<Op>> for AreaCost<L, Op>
where
  L: LangCost<Op>,
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable,
{
  type Cost = (usize, usize, bool);

  fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
  where
    C: FnMut(egg::Id) -> Self::Cost,
  {
    let arg_costs: Vec<Self::Cost> =
      enode.args().iter().map(|&id| costs(id)).collect();
    let arg_selected_costs: Vec<usize> = arg_costs
      .iter()
      .map(|&cost| if cost.2 { cost.0 } else { cost.1 })
      .collect();
    let op_cost = self
      .lang_cost
      .op_cost(enode.operation(), &arg_selected_costs);
    let in_lib = match enode.as_binding_expr() {
      Some(expr) => {
        match expr {
          BindingExpr::Lib(_, _, _)
          | BindingExpr::Apply(_, _)
          | BindingExpr::LibVar(_) => false,
          // Lambdas, Vars
          _ => true,
        }
      }
      None => {
        // If any of the children are in a Lib, we regard it as a in-library
        // operation
        arg_costs.iter().any(|&cost| cost.2)
      }
    };
    let in_lib_cost =
      arg_costs.iter().map(|&cost| cost.0).sum::<usize>() + op_cost;
    let out_of_lib_cost = arg_selected_costs.iter().sum::<usize>();
    // println!("op_type: {:?}, op_gain: {}", enode.operation(), op_gain);
    // println!("arg_costs: {:?}", arg_costs);
    // println!("in_lib_cost: {}, out_of_lib_cost: {}, in_lib: {}", in_lib_cost,
    // out_of_lib_cost, in_lib);
    (in_lib_cost, out_of_lib_cost, in_lib)
  }
}

/// Cost function that uses LangGain to calculate delay
#[derive(Debug, Clone)]
pub struct DelayCost<L, Op> {
  pub lang_gain: L,
  _phantom: std::marker::PhantomData<Op>,
}

impl<L, Op> DelayCost<L, Op>
where
  L: LangGain<Op>,
  Op: Clone + Debug,
{
  pub fn new(lang_gain: L) -> Self {
    Self {
      lang_gain,
      _phantom: std::marker::PhantomData,
    }
  }
}

impl<L, Op> CostFunction<AstNode<Op>> for DelayCost<L, Op>
where
  L: LangGain<Op>,
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable,
{
  /// The cost is a tuple of two usize values: the cost in the case that
  /// the operation is in a library, the cost in the case that the operation
  /// is not in a library, and a boolean indicating whether the operation is
  /// in a library (true) or not (false).
  type Cost = (usize, usize, bool);

  fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
  where
    C: FnMut(egg::Id) -> Self::Cost,
  {
    let arg_costs: Vec<Self::Cost> =
      enode.args().iter().map(|&id| costs(id)).collect();
    let arg_selected_costs: Vec<usize> = arg_costs
      .iter()
      .map(|&cost| if cost.2 { cost.0 } else { cost.1 })
      .collect();
    let op_gain = self
      .lang_gain
      .op_gain(enode.operation(), &arg_selected_costs);
    let in_lib = match enode.as_binding_expr() {
      Some(expr) => match expr {
        BindingExpr::Lib(_, _, _)
        | BindingExpr::Apply(_, _)
        | BindingExpr::LibVar(_) => false,
        // Lambdas, Vars
        _ => true,
      },
      None => {
        // If any of the children are in a Lib, we regard it as a in-library
        // operation
        arg_costs.iter().any(|&cost| cost.2)
      }
    };
    let in_lib_cost = match arg_costs.iter().map(|&cost| cost.0).max() {
      Some(cost) => cost + op_gain,
      None => op_gain,
    };
    let out_of_lib_cost = arg_selected_costs.iter().sum::<usize>() + op_gain;
    println!("op_type: {:?}, op_gain: {:#?}", enode.operation(), op_gain);
    // println!("arg_costs: {:?}", arg_costs);
    println!(
      "in_lib_cost: {}, out_of_lib_cost: {}, in_lib: {}",
      in_lib_cost, out_of_lib_cost, in_lib
    );
    (in_lib_cost, out_of_lib_cost, in_lib)
  }
}
