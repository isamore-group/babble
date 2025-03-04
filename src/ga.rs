
use rand::{Rng, thread_rng, seq::SliceRandom};
use rand::prelude::*;

use crate::learn::*;

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


type Cost = i32;



// 遗传算法个体
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Individual<Op, Ids> {
    genes: Vec<PartialExpr<Op, Ids>>,
    fitness: Cost,
}

impl <Op, Ids> Individual<Op, Ids> {
    fn get_cost(&self) -> Cost {
        self.fitness
    }
}


impl<Op: Clone + Debug + Hash + Ord, Ids: Clone + Debug + Hash + Ord> Individual<Op, Ids> {
    fn new(genes: Vec<PartialExpr<Op, Ids>>) -> Self {
        let fitness = calculate_fitness(&genes);
        Self { genes, fitness }
    }

    // 交叉操作
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> (Self, Self) {
        let mut child1 = self.genes.clone();
        let mut child2 = other.genes.clone();
        
        // 单点交叉
        let crossover_point = rng.gen_range(0..self.genes.len());
        for i in crossover_point..self.genes.len() {
            std::mem::swap(&mut child1[i], &mut child2[i]);
        }
        
        (Self::new(child1), Self::new(child2))
    }

    // 变异操作
    fn mutate(&mut self, aus: &[Vec<PartialExpr<Op, Ids>>], rng: &mut impl Rng) {
        let mutation_point = rng.gen_range(0..self.genes.len());
        self.genes[mutation_point] = aus[mutation_point]
            .choose(rng)
            .unwrap()
            .clone();
        self.fitness = calculate_fitness(&self.genes);
    }
}

// 遗传算法主体
pub fn genetic_algorithm_aus<Op>(
    aus: Vec<Vec<PartialExpr<Op, (Id, Id)>>>,
    m: usize,
    population_size: usize,
    generations: usize,
) -> Vec<Vec<PartialExpr<Op, (Id, Id)>>>
where
    Op: Clone + Debug + Hash + Ord + Display + Arity + Sync,
{
    // 如果aus是空的，或者里面的内容为空，直接返回空
    if aus.is_empty() || aus.iter().all(|x| x.is_empty()) {
        return vec![];
    }

    let mut rng = thread_rng();
    
    // 1. 初始化种群
    let mut population = initialize_population(&aus, population_size, &mut rng);
    
    // 2. 进化循环
    for generation in 0..generations {
        // 选择
        let selected = tournament_selection(&population, population_size / 2, &mut rng);
        
        // 交叉
        let mut offspring = Vec::new();
        for chunk in selected.chunks(2) {
            if let [p1, p2] = chunk {
                let (c1, c2) = p1.crossover(p2, &mut rng);
                offspring.push(c1);
                offspring.push(c2);
            }
        }
        
        // 变异
        for child in &mut offspring {
            if rng.gen_bool(0.1) { // 10% 变异概率
                child.mutate(&aus, &mut rng);
            }
        }
        
        // 替换
        population.extend(offspring);
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        population.truncate(population_size);
        // 每代记录统计信息
        println!("Generation {} | Best fitness: {:.4} | Avg fitness: {:.4}",
        generation,
        population[0].fitness,
        population.iter().map(|i| i.fitness).sum::<i32>() / population.len() as i32
        );

    }

    // 3. 选择最终结果
    select_diverse_solutions(population, m)
}

// 初始化种群
fn initialize_population<Op, Ids>(
    aus: &[Vec<PartialExpr<Op, Ids>>],
    size: usize,
    rng: &mut impl Rng,
) -> Vec<Individual<Op, Ids>> 
where 
    Op: Clone + Debug + Hash + Ord + Display + Arity + Sync,
    Ids: Clone + Debug + Hash + Ord,
{
    let mut population = HashSet::new();
    
    // 添加极值组合
    add_extreme_combinations(aus, &mut population);
    
    // 随机生成剩余个体
    while population.len() < size {
        let genes = aus.iter()
            .map(|options| options.choose(rng).unwrap().clone())
            .collect();
        population.insert(Individual::new(genes));
    }
    
    population.into_iter().collect()
}

// 计算适应度（方差）
fn calculate_fitness<Op, Ids>(genes: &[PartialExpr<Op, Ids>]) -> Cost {
    let sizes: Vec<f64> = genes.iter().map(|e| e.size() as f64).collect();
    let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
    let variance = sizes.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / sizes.len() as f64;
    let variance = variance * 10000 as f64;
    let fitness = variance as i32;
    fitness
}

// 锦标赛选择
fn tournament_selection<Op, Ids>(
    population: &[Individual<Op, Ids>],
    selection_size: usize,
    rng: &mut impl Rng,
) -> Vec<Individual<Op, Ids>> 
where 
    Op: Clone + Debug + Hash + Ord + Display + Arity + Sync,
    Ids: Clone + Debug + Hash + Ord,
    {
    let mut selected = Vec::new();
    for _ in 0..selection_size {
        let candidates: Vec<_> = population.choose_multiple(rng, 5).cloned().collect();
        let best = candidates.into_iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap();
        selected.push(best);
    }
    selected
}

// 添加极值组合
fn add_extreme_combinations<Op, Ids>(
    aus: &[Vec<PartialExpr<Op, Ids>>],
    population: &mut HashSet<Individual<Op, Ids>>,
) where 
    Op: Clone + Debug + Hash + Ord + Display + Arity + Sync,
    Ids: Clone + Debug + Hash + Ord,
{
    // 全最大组合
    let all_max = aus.iter()
        .map(|options| options.last().unwrap().clone())
        .collect();
    population.insert(Individual::new(all_max));

    // 全最小组合
    let all_min = aus.iter()
        .map(|options| options.first().unwrap().clone())
        .collect();
    population.insert(Individual::new(all_min));

    // 每个维度单独最大
    for i in 0..aus.len() {
        let genes = aus.iter().enumerate()
            .map(|(j, options)| if j == i {
                options.last().unwrap().clone()
            } else {
                options.first().unwrap().clone()
            })
            .collect();
        population.insert(Individual::new(genes));
    }
}

// 选择多样化解
fn select_diverse_solutions<Op, Ids>(
    mut population: Vec<Individual<Op, Ids>>,
    m: usize,
) -> Vec<Vec<PartialExpr<Op, Ids>>>
where
    Op: Clone + Debug + Hash + Ord + Display + Arity + Sync,
    Ids: Clone + Debug + Hash + Ord, 
{
    // 去重
    let mut seen = HashSet::new();
    population.retain(|indiv| seen.insert(indiv.genes.clone()));

    // 按cost排序（假设有get_cost方法）
    population.sort_by(|a, b| a.get_cost().partial_cmp(&b.get_cost()).unwrap());

    // 均匀采样
    let step = population.len() as f64 / m as f64;
    (0..m).map(|i| {
        let idx = (i as f64 * step).round() as usize;
        population[idx.min(population.len()-1)].genes.clone()
    }).collect()
}