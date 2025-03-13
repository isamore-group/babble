//! Cost models for extraction

use std::fmt::Debug;
use egg::CostFunction;
use crate::{ast_node::AstNode, Teachable, teachable::BindingExpr};

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
    Op: Clone + Debug + Ord + std::hash::Hash,
{
    type Cost = usize;

    fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        // Get costs of children
        let arg_costs: Vec<usize> = enode.args().iter().map(|&id| costs(id)).collect();
        
        // Calculate the cost of this operation
        let op_cost = self.lang_cost.op_cost(enode.operation(), &arg_costs);
        
        // Sum up the costs
        arg_costs.iter().sum::<usize>() + op_cost
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
    type Cost = usize;

    fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        // Get costs of children
        let arg_costs: Vec<usize> = enode.args().iter().map(|&id| costs(id)).collect();
        
        // For delay, we take the maximum of child delays plus the operation delay
        let max_child_cost = arg_costs.iter().max().copied().unwrap_or(0);
        
        // Calculate the gain of this operation
        let op_gain = self.lang_gain.op_gain(enode.operation(), &arg_costs);
        
        match enode.as_binding_expr() {
            Some(expr) => {
                match expr {
                    BindingExpr::Lib(_, _, _) => {
                        max_child_cost / 2 + op_gain
                    }
                    _ => {
                        max_child_cost + op_gain
                    }
                }
            },
            None => {
                // Otherwise, just return the critical path delay
                max_child_cost + op_gain
            }
        }
    }
} 