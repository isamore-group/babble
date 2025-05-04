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
use egg::{
  Analysis, Condition, ConditionalApplier, EGraph, FromOp, Id, Language,
  Pattern, Rewrite, Subst, Var,
};
use itertools::Itertools;
use std::{
  collections::HashMap, error::Error, fmt::Debug, fs, io::ErrorKind,
  path::Path, str::FromStr,
};

use crate::extract::beam_pareto::TypeInfo;
/// Returns all the rewrites in the specified file.
///
/// # Errors
/// This function will return an error if the file doesn't exist or can't be
/// opened.
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
    let ac_flag = rewrite.contains("<==>");
    let (lhs, rhs) = rewrite
      .split_once("==>")
      .or_else(|| rewrite.split_once("<==>"))
      .ok_or(anyhow!("missing '==>' or '<==>'"))?;
    let name = line;
    let lhs = lhs.trim();
    let rhs = rhs.trim();

    let lhs: Pattern<L> = lhs.parse()?;
    let rhs: Pattern<L> = rhs.parse()?;
    if condition == "" {
      rewrites.push(
        Rewrite::new(name, lhs.clone(), rhs.clone())
          .map_err(|e| anyhow!("{}", e))?,
      );
      if ac_flag {
        // 如果是ac_flag，就需要添加一个反向的Rewrite
        let lhs = rhs.clone();
        let rhs = lhs.clone();
        let name = format!("{}_reverse", line);
        rewrites
          .push(Rewrite::new(name, lhs, rhs).map_err(|e| anyhow!("{}", e))?);
      }
      continue;
    } else {
      let condition = {
        // 查询是否有all关键字，如果存在，就用:进行分割
        if condition.contains("all") {
          // 首先去掉"()",然后用:分割
          let condition = condition
            .strip_prefix("(")
            .ok_or(anyhow!("missing '('"))?
            .strip_suffix(")")
            .ok_or(anyhow!("missing ')')"))?;
          let (_, ty) =
            condition.split_once(":").ok_or(anyhow!("missing ':'"))?;
          // 去除可能的空格
          let ty = ty.trim();
          if ty.contains("[") {
            // 去除[]，并用,分割
            let ty = ty
              .strip_prefix("[")
              .ok_or(anyhow!("missing '['"))?
              .strip_suffix("]")
              .ok_or(anyhow!("missing ']'"))?;
            let ty = ty.split(",").map(|s| s.trim()).collect::<Vec<_>>();
            // 解析成Vec<T>
            let mut type_map = HashMap::new();
            for var in lhs.vars() {
              let mut egg_type = Vec::new();
              for t in &ty {
                egg_type.push(t.parse::<T>()?);
              }
              type_map.insert(var, egg_type);
            }
            TypeMatch::new(type_map)
          }
          // 解析成单个类型
          else {
            let egg_type = vec![ty.parse::<T>()?];
            let mut type_map = HashMap::new();
            for var in lhs.vars() {
              type_map.insert(var, egg_type.clone());
            }
            TypeMatch::new(type_map)
          }
        }
        // 使用parse函数解析condition
        else {
          condition.parse::<TypeMatch<T>>()?
        }
      };
      // 针对使用
      let conditional_applier = ConditionalApplier {
        condition: condition.clone(),
        applier: rhs.clone(),
      };
      rewrites.push(
        Rewrite::new(name, lhs.clone(), conditional_applier)
          .map_err(|e| anyhow!("{}", e))?,
      );
      if ac_flag {
        // 如果是ac_flag，就需要添加一个反向的Rewrite
        let lhs = rhs.clone();
        let rhs = lhs.clone();
        let name = format!("{}_reverse", line);
        // 针对使用
        let conditional_applier = ConditionalApplier {
          condition: condition,
          applier: rhs.clone(),
        };
        rewrites.push(
          Rewrite::new(name, lhs, conditional_applier)
            .map_err(|e| anyhow!("{}", e))?,
        );
      }
    }
  }
  Ok(rewrites)
}

// Condition
#[derive(Debug, Clone)]
pub struct TypeMatch<T> {
  pub type_map: HashMap<Var, Vec<T>>,
}

impl<T> TypeMatch<T> {
  pub fn new(type_map: HashMap<Var, Vec<T>>) -> Self {
    TypeMatch { type_map }
  }
}

impl<T: FromStr + Send + Debug> FromStr for TypeMatch<T>
where
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  type Err = anyhow::Error;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    // Remove "where(" and ")"
    let s = s
      .strip_prefix("(")
      .ok_or(anyhow!("Missing '('"))?
      .strip_suffix(")")
      .ok_or(anyhow!("Missing ')')"))?;

    // Split by commas
    let pairs = s.split(',').collect::<Vec<&str>>();
    let mut type_map = HashMap::new();

    for pair in pairs {
      let parts: Vec<&str> = pair.split(':').collect();
      if parts.len() != 2 {
        return Err(anyhow!("Invalid format"));
      }
      // 如果含有[], 那么说明有多个可能的类型
      let rparts = parts[1];
      if rparts.contains("[") {
        let rparts = rparts
          .strip_prefix("[")
          .ok_or(anyhow!("Missing '['"))?
          .strip_suffix("]")
          .ok_or(anyhow!("Missing ']'"))?;
        let rparts = rparts
          .split(',')
          .map(|s| s.trim().parse::<T>())
          .collect::<Result<Vec<T>, _>>()?;
        type_map.insert(parts[0].trim().parse::<Var>()?, rparts);
        continue;
      } else {
        // Parse Var and  T
        let var = parts[0].trim().parse::<Var>()?;
        let egg_type = vec![parts[1].trim().parse::<T>()?];
        type_map.insert(var, egg_type);
      }
    }
    Ok(TypeMatch { type_map })
  }
}

impl<L, N, T> Condition<L, N> for TypeMatch<T>
where
  L: Language + TypeInfo<T>,
  N: Analysis<L>,
  <N as Analysis<L>>::Data: PartialEq<T> + Debug,
  T: Debug,
{
  fn check(
    &self,
    egraph: &mut EGraph<L, N>,
    _eclass: Id,
    subst: &Subst,
  ) -> bool {
    for (var, egg_type) in &self.type_map {
      if let Some(ecls_id) = subst.get(var.clone()) {
        let ty = &egraph[*ecls_id].data;
        let mut found = false;
        for t in egg_type {
          if ty == t {
            found = true;
            break;
          }
        }
        if !found {
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
