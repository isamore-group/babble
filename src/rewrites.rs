//! Utilities for loading and parsing rewrites files.
//!
//! # Examples
//!
//! A rewrites file might look like this:
//!
//! ```text
//! one_plus_one: (+ 1 1) => 2
//! commutative: (+ ?x ?y) => (+ ?y ?x)
//! ```

use anyhow::anyhow;
use egg::{rewrite, Analysis, Condition, EGraph, FromOp, Id, Language, Pattern, Rewrite, Subst, Var, ConditionalApplier, DidMerge};
use std::{collections::HashMap, error::Error, fmt::Debug, fs, hash::Hash, io::ErrorKind, path::Path, str::FromStr, marker::PhantomData};

use crate::AstNode;
use crate::extract::beam_knapsack::TypeInfo;
/// Returns all the rewrites in the specified file.
///
/// # Errors
/// This function will return an error if the file doesn't exist or can't be opened.
///
/// It will also return an error if it could not parse the file.
pub fn from_file<L, A, P, T>(path: P) -> anyhow::Result<Vec<Rewrite<L, A>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
  P: AsRef<Path>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  let contents = fs::read_to_string(path)?;
  parse(&contents)
}

/// If the file specified by `path` exists, parse the file and return the
/// resulting rewrites. If the file does not exist, return `None`.
///
/// # Errors
/// This function will return an error if the file exists but can't be opened.
///
/// It will also return an error if it could not parse the file.
pub fn try_from_file<L, A, P, T>(
  path: P,
) -> anyhow::Result<Option<Vec<Rewrite<L, A>>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
  P: AsRef<Path>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  Ok(match fs::read_to_string(path) {
    Ok(contents) => Some(parse(&contents)?),
    Err(e) => match e.kind() {
      ErrorKind::NotFound => None,
      _ => Err(e)?,
    },
  })
}

/// Parse a rewrites file.
///
/// # Errors
/// This function will return an error if the rewrites file is invalid.
pub fn parse<L, A, T>(file: &str) -> anyhow::Result<Vec<Rewrite<L, A>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  let mut rewrites = Vec::new();
  for line in file
    .lines()
    .map(|line| {
      let line = line.split_once("//").map_or(line, |(line, _comment)| line);
      line.trim()
    })
    .filter(|line| !line.is_empty())
  {
    // 使用字符串"where"分割
    let (rewrite, condition) = line.split_once("where").unwrap_or((line, ""));
    let (lhs, rhs) =
      rewrite.split_once("==>").ok_or(anyhow!("missing arrow"))?;
    let name = line;
    let lhs = lhs.trim();
    let rhs = rhs.trim();
    let lhs: Pattern<L> = lhs.parse()?;
    let rhs: Pattern<L> = rhs.parse()?;
    if condition == "" {
      rewrites.push(Rewrite::new(name, lhs, rhs).map_err(|e| anyhow!("{}", e))?);
      continue;
    }else{
      let condition = {
      // 查询是否有all关键字，如果存在，就用:进行分割
      if condition.contains("all"){
        // 首先去掉"()",然后用:分割
        let condition = condition.strip_prefix("(").ok_or(anyhow!("missing '('"))?.strip_suffix(")").ok_or(anyhow!("missing ')')"))?;
        let (_, ty) = condition.split_once(":").ok_or(anyhow!("missing ':'"))?;
        // 去除可能的空格
        let ty = ty.trim();
        let ty = ty.parse::<T>()?;
        // 获取所有的变量
        let vars = lhs.vars();
        let mut type_map = HashMap::new();
        for var in vars {
          type_map.insert(var, ty.clone());
        }
        TypeMatch::new(type_map)
      }
      // 使用parse函数解析condition
      else {
        condition.parse::<TypeMatch<T>>()?
      }
    };
      // 针对使用
      let conditional_applier = ConditionalApplier{condition: condition, applier: rhs};
      rewrites.push(Rewrite::new(name, lhs, conditional_applier).map_err(|e| anyhow!("{}", e))?);
    }
  }
  Ok(rewrites)
}

// Condition
#[derive(Debug)]
pub struct TypeMatch<T>{
  pub type_map : HashMap<Var, T>,
}

impl <T> TypeMatch<T>{
  pub fn new(type_map: HashMap<Var, T>) -> Self{
    TypeMatch{
      type_map ,
    }
  }
}

impl  <T: FromStr + Send + Debug> FromStr for TypeMatch<T>
where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static
{
  type Err = anyhow::Error;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    // Remove "where(" and ")"
    let s = s.strip_prefix("(").ok_or(anyhow!("Missing '('"))?.strip_suffix(")").ok_or(anyhow!("Missing ')')"))?;
    
    // Split by commas
    let pairs = s.split(',').collect::<Vec<&str>>();
    let mut type_map = HashMap::new();

    for pair in pairs {
        let parts: Vec<&str> = pair.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid format"));
        }
        // Parse Var and  T
        let var = parts[0].trim().parse::<Var>()?;
        let egg_type = parts[1].trim().parse::<T>()?;
        type_map.insert(var, egg_type);
    }
    Ok(TypeMatch { type_map })
}
    
}

impl <L, N, T> Condition<L, N> for TypeMatch<T>
where L: Language + TypeInfo<T>,
      N: Analysis<L>,
      <N as Analysis<L>>::Data: PartialEq<T> + Debug,
      T: Debug,
{
  fn check(&self, egraph: &mut EGraph<L, N>, _eclass: Id, subst: &Subst) -> bool {
    for (var, egg_type) in &self.type_map {
        if let Some(ecls_id) = subst.get(var.clone()) {
            let ty = &egraph[*ecls_id].data;
            if ty != egg_type {
                return false;
            } 
        }
    }
    true
  }

  fn vars(&self) -> Vec<Var> {
      let vars = self.type_map.keys().cloned().collect::<Vec<_>>();
      vars
  }
}

