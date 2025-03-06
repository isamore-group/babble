//! ycy：实现不使用笛卡尔积的au选择算法
use crate::learn::*;
use rand::seq::SliceRandom;

use crate::{
  ast_node::{Arity, AstNode, PartialExpr}, co_occurrence::CoOccurrences, dfta::Dfta, extract::beam::PartialLibCost, teachable::{BindingExpr, Teachable}, COBuilder, Pretty
};
use std::{hash::Hash, time::Instant};
use egg::{Analysis, EGraph, Id, Language, Pattern, Rewrite, Searcher, Var};
use itertools::Itertools;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::{
  collections::{BTreeMap, BTreeSet, HashSet},
  fmt::{Debug, Display},
  num::ParseIntError,
  str::FromStr,
};

use thiserror::Error;
use crate::learn::{Match, AU};


/// 定义Vec<PatialExpr<Op, Var>>的类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecPE<Op> {
    pub exprs: Vec<PartialExpr<Op, (Id, Id)>>,
    pub matches: Vec<Vec<Match>>,
}
/// 为VecPE实现new_with_egraph
impl <Op> VecPE<Op> {
    pub fn new(expr: Vec<PartialExpr<Op, (Id, Id)>>, matches: Vec<Vec<Match>>) -> Self {
        Self { exprs: expr, matches }
    }

}

/// 为VecPE实现PartialOrd
impl <Op: Eq> PartialOrd for VecPE<Op> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.matches.cmp(&other.matches))
        // 计算exprs中PE的size()的和
        // let self_size = self.exprs.iter().map(|x| x.size()).sum::<usize>();
        // let other_size = other.exprs.iter().map(|x| x.size()).sum::<usize>();
        // Some(self_size.cmp(&other_size))
    }
}

/// 为VecPE实现Ord
impl <Op: Eq> Ord for VecPE<Op> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.matches.cmp(&other.matches)
        // let self_size = self.exprs.iter().map(|x| x.size()).sum::<usize>();
        // let other_size = other.exprs.iter().map(|x| x.size()).sum::<usize>();
        // self_size.cmp(&other_size)
    }
}



// - 基于随机寻优的方法：
// - 从每个集合中随机抽样获得N个样本集
// - 使用cost对样本集进行排序，均匀选择M个解

/// 从指定的集合中随机抽取K个特征解

pub fn get_random_aus<Op>(
  aus: Vec<Vec<AU<Op, (Id, Id)>>>,
  m: usize,
) -> Vec<Vec<PartialExpr<Op, (Id, Id)>>>
where
  Op: Clone + Debug + Hash + Ord + Display + Arity + Teachable + Sync + Send + 'static,
{
    // 如果aus是空的，直接返回空
    if aus.is_empty() {
        return vec![];
    }

    info!("get_random_aus");
    let range = 50 * m;

    // 计算aus中每个Vec的size的乘积
    let cartesian_product_size:usize = aus.iter().map(|x| x.len()).product();

    let cal_cartesian_flag = cartesian_product_size < range;

    let cartesian_product = if cal_cartesian_flag {
        info!("cal_cartesian");
        // 计算aus中每个Vec的size的乘积
        let cartesian_product: Vec<Vec<AU<Op, (Id, Id)>>> = aus
            .iter()
            .multi_cartesian_product()
            .map(|x| x.into_iter().cloned().collect())
            .collect();
        cartesian_product
    } else {
        vec![]
    };

    // 如果cartesian_product的size小于m, 直接返回cartesian_product
    if cal_cartesian_flag && cartesian_product.len() < m {
        // 将cartesian_product中的每个Vec转换为PartialExpr<Op, (Id, Id)>的Vec
        return cartesian_product.iter().map(|x| x.iter().map(|y| y.expr().clone()).collect()).collect();
    }
    // 定义一个闭包，用于将Vec<AU<Op, (Id, Id)>>中的PE和matches分别对应收集起来，组成VecPE
    let au2pe = |vec: &Vec<AU<Op, (Id, Id)>>| {
        let mut exprs = Vec::new();
        let mut matches = Vec::new();
        for au in vec {
            exprs.push(au.expr().clone());
            matches.push(au.matches().clone());
        }
        VecPE::new(exprs, matches)
    };
    let candidates = if cal_cartesian_flag {
        info!("insert into BTreeSet");
        info!("cartesian_product.size: {}", cartesian_product.len());
        // 需要将VecPe插入进BTreeSet进行排序
        let start_time = Instant::now();
        let mut candidates = BTreeSet::new();
        for i in 0..cartesian_product.len() {
            // 使用au2pe将Vec<AU<Op, (Id, Id)>>转换为VecPE
            candidates.insert(au2pe(&cartesian_product[i]));
        }
        info!("insert into BTreeSet cost: {:?}", start_time.elapsed());
        candidates
    } else {
        // 如果cartesian_product的size大于range, 就需要自己另行构造candidates
        // 首先，由于aus中的每个vec都是从BTreeSet中收集的，所以已经根据cost排序了，首先需要取出上下界，上界取出最后一个，下界取出第一个
        let mut lower_bound = Vec::new();
        let mut upper_bound = Vec::new();
        for i in 0..aus.len() {
            lower_bound.push(aus[i][0].clone());
            upper_bound.push(aus[i][aus[i].len()-1].clone());
        }
        let mut selected_aus  = BTreeSet::new();
        // 将上下界加入candidates
        selected_aus.insert(au2pe(&lower_bound));
        selected_aus.insert(au2pe(&upper_bound));
        // 从每个集合中随机抽样获得range个样本集
        let mut rng = rand::thread_rng();
        for _ in 0..range {
            let sample_aus: Vec<AU<_, _>> = aus
                .iter()
                .filter_map(|vec| vec.choose(&mut rng).cloned()) // 从每个 `aus[i]` 里随机选一个
                .collect();
            selected_aus.insert(au2pe(&sample_aus));
        }
        selected_aus
    };
    let n = candidates.len();
    let step = n as f64 / m as f64; // 步长计算为浮点数
    let mut selected_elements = Vec::new();
    let selected_aus = candidates.iter().collect::<Vec<_>>();
    for i in 0..m {
        let index = (i as f64 * step).round() as usize;
        // 确保索引不会超出范围
        let index = index.min(n - 1); // 避免越界
        selected_elements.push(selected_aus[index].exprs.clone());
    }
    selected_elements
}


// 基于beam search的方法：
// 束搜索的思想是逐步构造完整解，在每个阶段只保留一部分“最有希望”的部分解，从而避免生成所有组合。

// （1）初始化
// 设定一个束宽度（beam width），例如一个比M稍大的数，记为B。
// 从第一个input开始，每个特征值形成一个初步的部分匹配模式。
// （2）迭代扩展
// 对于每个后续的input（共N个），对当前束内的每个部分匹配模式：
// 扩展：将当前部分模式与下一个input的每个特征值组合，生成新的部分匹配模式。
// 计算启发式成本：如果cost是要衡量整体匹配的“离散性”或者“差异”，可以用当前已选部分的cost（或结合一个下界估计）来评价这个部分解未来可能获得的成本值。
// 在扩展后，保留束宽度内的B个部分匹配模式（例如根据cost排序，选择cost指标较优的那B个）。
// （3）完成并选择
// 当所有N个input都处理完后，束内的每个元素都是一个完整的匹配模式。
// 从中挑选出满足你要求的M个匹配模式（比如按照cost排序，选择前M个）。



// pub fn beam_search_aus<Op>(
//     aus: Vec<Vec<PartialExpr<Op, (Id, Id)>>>,
//     m: usize,
//     b: usize, // 束宽
// ) -> Vec<Vec<PartialExpr<Op, (Id, Id)>>>
// where
//     Op: Clone + std::fmt::Debug + std::hash::Hash + Ord + std::fmt::Display,
// {
//     // 如果aus为空，返回空
//     if aus.is_empty() {
//         return vec![];
//     }

//     // 用来存储当前的解
//     let mut current_solutions: BTreeSet<VecPE<_>> = BTreeSet::new();
//     // 将aus[0]添加进current_solutions
//     for elem in &aus[0] {
//         current_solutions.insert(VecPE(vec![elem.clone()]));
//     }

//     // 对于后续的每一个集合，计算笛卡尔积
//     for i in 1..aus.len() {
//         info!("current index: {}", i);
//         // 如果aus[i]为空，直接continue
//         if aus[i].is_empty() {
//             continue;
//         }
//         let mut solutions: BTreeSet<VecPE<_>> = BTreeSet::new();
//         // 当前候选解
//         let mut candidate_vecs: BTreeSet<VecPE<_>> = BTreeSet::new();
//         for solution in &current_solutions {
//             for elem in &aus[i] {
//                 let mut new_solution = solution.clone();
//                 new_solution.0.push(elem.clone());
//                 candidate_vecs.insert(new_solution);
//             }
//         }
//         // 如果候选解的数量小于b, 直接将solutions设置为candidate_vecs
//         if candidate_vecs.len() < b {
//             current_solutions = candidate_vecs.iter().map(|x| x.clone()).collect();
//             info!("current_solutions.size: {}", current_solutions.len());
//             continue;
//         }
//         // 否则均匀选择b个解
//         let n = candidate_vecs.len();
//         let step = n as f64 / b as f64; // 步长计算为浮点数
//         let selected_aus = candidate_vecs.iter().collect::<Vec<_>>();
//         for i in 0..b {
//             let index = (i as f64 * step).round() as usize;
//             // 确保索引不会超出范围
//             let index = index.min(n - 1); // 避免越界
//             solutions.insert(selected_aus[index].clone());
//         }
//         current_solutions = solutions;
//     }
        
//     // 如果current_solutions的size小于m, 直接返回current_solutions
//     if current_solutions.len() < m {
//         info!("current_solutions.size: {}", current_solutions.len());
//         return current_solutions.iter().map(|x| x.0.clone()).collect::<Vec<_>>();
//     }

//     // 从current_solutions中均匀选择m个解
//     let n = current_solutions.len();
//     let step = n as f64 / m as f64; // 步长计算为浮点数
//     let mut selected_elements = Vec::new();
//     let selected_aus = current_solutions.iter().collect::<Vec<_>>();
//     for i in 0..m {
//         let index = (i as f64 * step).round() as usize;
        
//         // 确保索引不会超出范围
//         let index = index.min(n - 1); // 避免越界
//         selected_elements.push(selected_aus[index].clone().0);
//     }
//     info!("selected_elements.size: {}", selected_elements.len());
//     selected_elements
// }




