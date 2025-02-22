//! Defines [`SimpleOp`], a simple lambda calculus which can be used with babble.

use std::{
  convert::Infallible,
  fmt::{self, Display, Formatter},
  str::FromStr,
};

use egg::Symbol;

use crate::{
  ast_node::{Arity, AstNode},
  learn::{LibId, ParseLibIdError},
  teachable::{BindingExpr, DeBruijnIndex, Teachable},
};

/// Simplest language to use with babble
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimpleOp {
  /// A function application
  Apply,
  /// A de Bruijn-indexed variable
  Var(DeBruijnIndex),
  /// A reference to a lib fn
  LibVar(LibId),
  /// An uninterpreted symbol
  Symbol(Symbol),
  /// An anonymous function
  Lambda,
  /// A library function binding
  Lib(LibId),
  /// A list of expressions
  List,
}

impl Arity for SimpleOp {
  fn min_arity(&self) -> usize {
    match self {
      Self::Var(_) | Self::Symbol(_) => 0,
      Self::Lambda | Self::LibVar(_) | Self::List => 1,
      Self::Apply | Self::Lib(_) => 2,
    }
  }
}

impl Display for SimpleOp {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    let s = match self {
      Self::Apply => "@",
      Self::Lambda => "λ",
      Self::Lib(libid) => {
        return write!(f, "lib {libid}");
      }
      Self::LibVar(libid) => {
        return write!(f, "l{libid}");
      }
      Self::Var(index) => {
        return write!(f, "${index}");
      }
      Self::Symbol(sym) => {
        return write!(f, "{sym}");
      }
      Self::List => "list",
    };
    f.write_str(s)
  }
}

impl FromStr for SimpleOp {
  type Err = Infallible;

  fn from_str(input: &str) -> Result<Self, Self::Err> {
    let op = match input {
      "apply" | "@" => Self::Apply,
      "lambda" | "λ" => Self::Lambda,
      "list" => Self::List,
      input => input
        .parse()
        .map(Self::Var)
        .or_else(|_| input.parse().map(Self::LibVar))
        .or_else(|_| {
          input
            .strip_prefix("lib ")
            .ok_or(ParseLibIdError::NoLeadingL)
            .and_then(|x| x.parse().map(Self::Lib))
        })
        .unwrap_or_else(|_| Self::Symbol(input.into())),
    };
    Ok(op)
  }
}

impl Teachable for SimpleOp {
  fn from_binding_expr<T>(binding_expr: BindingExpr<T>) -> AstNode<Self, T> {
    match binding_expr {
      BindingExpr::Lambda(body) => AstNode::new(Self::Lambda, [body]),
      BindingExpr::Apply(fun, arg) => AstNode::new(Self::Apply, [fun, arg]),
      BindingExpr::Var(index) => AstNode::leaf(Self::Var(index)),
      BindingExpr::Lib(ix, bound_value, body) => {
        AstNode::new(Self::Lib(ix), [bound_value, body])
      }
      BindingExpr::LibVar(ix) => AstNode::new(Self::LibVar(ix), []),
    }
  }

  fn as_binding_expr<T>(node: &AstNode<Self, T>) -> Option<BindingExpr<&T>> {
    let binding_expr = match node.as_parts() {
      (Self::Lambda, [body]) => BindingExpr::Lambda(body),
      (Self::Apply, [fun, arg]) => BindingExpr::Apply(fun, arg),
      (Self::Var(index), []) => BindingExpr::Var(*index),
      (Self::Lib(ix), [bound_value, body]) => {
        BindingExpr::Lib(*ix, bound_value, body)
      }
      (Self::LibVar(ix), []) => BindingExpr::LibVar(*ix),
      _ => return None,
    };
    Some(binding_expr)
  }

  fn list() -> Self {
    Self::List
  }
}
