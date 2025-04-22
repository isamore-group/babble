//! ycy：实现不使用笛卡尔积的au选择算法
use crate::learn::AU;
use crate::runner::LiblearnCost;
use crate::{
  ast_node::{Arity, PartialExpr},
  teachable::Teachable,
};
use egg::Id;
use itertools::Itertools;
use log::info;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{
  collections::BTreeSet,
  fmt::{Debug, Display},
};
use std::{hash::Hash, time::Instant};

/// 定义Vec<PatialExpr<Op, Var>>的类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecPE<Op> {
  aus: Vec<AU<Op, (Id, Id)>>,
  cost_config: LiblearnCost,
}
/// 为VecPE实现new_with_egraph
impl<Op> VecPE<Op> {
  pub fn new(aus: Vec<AU<Op, (Id, Id)>>) -> Self {
    let cost_config = if aus.len() > 0 {
      aus[0].liblearn_cost()
    } else {
      LiblearnCost::default()
    };
    Self { aus, cost_config }
  }
}

/// 为VecPE实现PartialOrd
impl<Op: Eq> PartialOrd for VecPE<Op> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    // Some(self.matches.cmp(&other.matches))
    // 计算exprs中每个au的expr()的size的和
    // let self_size = self.aus.iter().map(|x| x.expr().size()).sum::<usize>();
    // let other_size = other.aus.iter().map(|x|
    // x.expr().size()).sum::<usize>(); self_size.partial_cmp(&other_size)
    // 计算每个au.delay()的和
    // let self_delay = self.aus.iter().map(|x| x.delay()).sum::<usize>();
    // let other_delay = other.aus.iter().map(|x| x.delay()).sum::<usize>();
    // Some(self_delay.cmp(&other_delay))
    match self.cost_config {
      LiblearnCost::Match => {
        // 将matches收集起来
        let self_matched =
          self.aus.iter().map(|x| x.matches()).collect::<Vec<_>>();
        let other_matched =
          other.aus.iter().map(|x| x.matches()).collect::<Vec<_>>();
        Some(self_matched.cmp(&other_matched))
      }
      LiblearnCost::Size => {
        let self_delay = self.aus.iter().map(|x| x.delay()).sum::<usize>();
        let other_delay = other.aus.iter().map(|x| x.delay()).sum::<usize>();
        Some(self_delay.cmp(&other_delay))
      }
      LiblearnCost::Delay => {
        let self_size = self.aus.iter().map(|x| x.expr().size()).sum::<usize>();
        let other_size =
          other.aus.iter().map(|x| x.expr().size()).sum::<usize>();
        Some(self_size.cmp(&other_size))
      }
    }
  }
}

/// 为VecPE实现Ord
impl<Op: Eq> Ord for VecPE<Op> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // self.matches.cmp(&other.matches)
    // let self_size = self.aus.iter().map(|x| x.expr().size()).sum::<usize>();
    // let other_size = other.aus.iter().map(|x|
    // x.expr().size()).sum::<usize>(); self_size.cmp(&other_size)
    // let self_delay = self.aus.iter().map(|x| x.delay()).sum::<usize>();
    // let other_delay = other.aus.iter().map(|x| x.delay()).sum::<usize>();
    // self_delay.cmp(&other_delay)
    match self.cost_config {
      LiblearnCost::Match => {
        let self_matched =
          self.aus.iter().map(|x| x.matches()).collect::<Vec<_>>();
        let other_matched =
          other.aus.iter().map(|x| x.matches()).collect::<Vec<_>>();
        self_matched.cmp(&other_matched)
      }
      LiblearnCost::Size => {
        let self_size = self.aus.iter().map(|x| x.expr().size()).sum::<usize>();
        let other_size =
          other.aus.iter().map(|x| x.expr().size()).sum::<usize>();
        self_size.cmp(&other_size)
      }
      LiblearnCost::Delay => {
        let self_delay = self.aus.iter().map(|x| x.delay()).sum::<usize>();
        let other_delay = other.aus.iter().map(|x| x.delay()).sum::<usize>();
        self_delay.cmp(&other_delay)
      }
    }
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
  Op: Clone
    + Debug
    + Hash
    + Ord
    + Display
    + Arity
    + Teachable
    + Sync
    + Send
    + 'static,
{
  // 如果aus是空的或者aus中的元素是空的，直接返回空
  if aus.is_empty() || aus.iter().any(|x| x.is_empty()) {
    return vec![];
  }

  info!("get_random_aus");
  let range = 10 * m;

  // 计算aus中每个Vec的size的乘积
  let cartesian_product_size: usize = aus.iter().map(|x| x.len()).product();

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
    return cartesian_product
      .iter()
      .map(|x| x.iter().map(|y| y.expr().clone()).collect())
      .collect();
  }
  // 定义一个闭包，用于将Vec<AU<Op, (Id, Id)>>组成VecPE
  let au2pe = |vec: Vec<AU<Op, (Id, Id)>>| VecPE::new(vec);
  let mut lower_bound = Vec::new();
  let mut upper_bound = Vec::new();
  for i in 0..aus.len() {
    lower_bound.push(aus[i][0].clone());
    upper_bound.push(aus[i][aus[i].len() - 1].clone());
  }
  let candidates = if cal_cartesian_flag {
    info!("insert into BTreeSet");
    info!("cartesian_product.size: {}", cartesian_product.len());
    // 需要将VecPe插入进BTreeSet进行排序
    let start_time = Instant::now();
    let mut candidates = BTreeSet::new();
    for i in 0..cartesian_product.len() {
      // 使用au2pe将Vec<AU<Op, (Id, Id)>>转换为VecPE
      candidates.insert(au2pe(cartesian_product[i].clone()));
    }
    info!("insert into BTreeSet cost: {:?}", start_time.elapsed());
    candidates
  } else {
    // 如果cartesian_product的size大于range, 就需要自己另行构造candidates
    // 首先，由于aus中的每个vec都是从BTreeSet中收集的，所以已经根据cost排序了，
    // 首先需要取出上下界，上界取出最后一个，下界取出第一个
    let mut selected_aus = BTreeSet::new();
    // 将上下界加入candidates
    selected_aus.insert(au2pe(lower_bound.clone()));
    selected_aus.insert(au2pe(upper_bound.clone()));
    // 从每个集合中随机抽样获得range个样本集
    let mut rng = rand::thread_rng();
    for _ in 0..range {
      let sample_aus: Vec<AU<_, _>> = aus
        .iter()
        .filter_map(|vec| vec.choose(&mut rng).cloned()) // 从每个 `aus[i]` 里随机选一个
        .collect();
      selected_aus.insert(au2pe(sample_aus));
    }
    selected_aus
  };
  // // 将点集中size()>400的点全部删掉
  // let mut candidates = candidates.into_iter().filter(|x|
  // x.exprs.iter().map(|x| x.size()).sum::<usize>() <
  // 400).collect::<BTreeSet<_>>(); // 如果candidates的size小于m,
  // 直接返回candidates if candidates.len() < m {
  //     return candidates.iter().map(|x| x.exprs.clone()).collect::<Vec<_>>();
  // }
  let n = candidates.len();
  let step = n as f64 / m as f64; // 步长计算为浮点数
  let mut selected_elements = Vec::new();
  let selected_aus = candidates.iter().collect::<Vec<_>>();
  for i in 0..m {
    let index = (i as f64 * step).round() as usize;
    // 确保索引不会超出范围
    let index = index.min(n - 1); // 避免越界
    selected_elements.push(
      selected_aus[index]
        .aus
        .clone()
        .iter()
        .map(|x| x.expr().clone())
        .collect(),
    );
  }
  // 加入上界
  selected_elements
    .push(upper_bound.iter().map(|x| x.expr().clone()).collect());
  // 加入下界
  selected_elements
    .push(lower_bound.iter().map(|x| x.expr().clone()).collect());
  selected_elements
}

// 基于beam search的方法：
// 束搜索的思想是逐步构造完整解，在每个阶段只保留一部分“最有希望”的部分解，
// 从而避免生成所有组合。

// （1）初始化
// 设定一个束宽度（beam width），例如一个比M稍大的数，记为B。
// 从第一个input开始，每个特征值形成一个初步的部分匹配模式。
// （2）迭代扩展
// 对于每个后续的input（共N个），对当前束内的每个部分匹配模式：
// 扩展：将当前部分模式与下一个input的每个特征值组合，生成新的部分匹配模式。
// 计算启发式成本：如果cost是要衡量整体匹配的“离散性”或者“差异”，
// 可以用当前已选部分的cost（或结合一个下界估计）来评价这个部分解未来可能获得的成本值。
// 在扩展后，保留束宽度内的B个部分匹配模式（例如根据cost排序，
// 选择cost指标较优的那B个）。 （3）完成并选择
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
//             current_solutions = candidate_vecs.iter().map(|x|
// x.clone()).collect();             info!("current_solutions.size: {}",
// current_solutions.len());             continue;
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
//         return current_solutions.iter().map(|x|
// x.0.clone()).collect::<Vec<_>>();     }

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

fn binary_split(m: usize, n: usize) -> Vec<Vec<usize>> {
  // 初始区间：[0, m)
  let mut segments: Vec<Vec<usize>> = vec![(0..m).collect()];

  // 进行 n 次二分切分
  for _ in 0..n {
    let mut new_segments = Vec::new();
    for seg in segments.iter() {
      if seg.len() > 1 {
        // 取中间位置进行分割
        let mid = seg.len() / 2;
        new_segments.push(seg[..mid].to_vec());
        new_segments.push(seg[mid..].to_vec());
      } else {
        // 当区间只有一个元素时，复制该区间到两个子区间
        new_segments.push(seg.clone());
        new_segments.push(seg.clone());
      }
    }
    segments = new_segments;
  }

  segments
}

pub fn kd_random_aus<Op>(
  aus: Vec<Vec<AU<Op, (Id, Id)>>>,
  m: usize,
) -> Vec<Vec<PartialExpr<Op, (Id, Id)>>>
where
  Op: Clone
    + Debug
    + Hash
    + Ord
    + Display
    + Arity
    + Teachable
    + Sync
    + Send
    + 'static,
{
  // 如果aus是空的或者aus中的元素是空的，直接返回空
  if aus.is_empty() || aus.iter().any(|x| x.is_empty()) {
    return vec![];
  }

  // 计算aus中每个Vec的size的乘积
  let cartesian_product_size: usize = aus.iter().map(|x| x.len()).product();

  let cal_cartesian_flag = cartesian_product_size < m;

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
  if cal_cartesian_flag && cartesian_product.len() < 10 * m {
    // 将cartesian_product中的每个Vec转换为PartialExpr<Op, (Id, Id)>的Vec
    return cartesian_product
      .iter()
      .map(|x| x.iter().map(|y| y.expr().clone()).collect())
      .collect();
  }
  // 定义一个闭包，用于将Vec<AU<Op, (Id,
  // Id)>>中的PE和matches分别对应收集起来，组成VecPE
  let au2pe = |vec: Vec<AU<Op, (Id, Id)>>| VecPE::new(vec);
  let candidates = if cal_cartesian_flag {
    info!("insert into BTreeSet");
    info!("cartesian_product.size: {}", cartesian_product.len());
    // 需要将VecPe插入进BTreeSet进行排序
    let start_time = Instant::now();
    let mut candidates = BTreeSet::new();
    for i in 0..cartesian_product.len() {
      // 使用au2pe将Vec<AU<Op, (Id, Id)>>转换为VecPE
      candidates.insert(au2pe(cartesian_product[i].clone()));
    }
    info!("insert into BTreeSet cost: {:?}", start_time.elapsed());
    candidates
  } else {
    // 首先，对lgm求上界得到递归次数
    let depth = (m as f64).log2().ceil() as usize;
    // 接下来确定每个集合被切分的次数
    let n = aus.len();
    // 基础次数为depth/n
    let base = depth / n;
    // 创建切分次数数组，每个集合切分base次
    let mut split_times = vec![base; n];
    // depth%n次的切分加到前面的集合中
    for i in 0..depth % n {
      split_times[i] += 1;
    }
    // FIXME:为什么使用了优先队列之后反倒效果不是特别好？
    // // 维护一个优先队列，存储维度i和aus[i]切分后当前的长度，
    // 这个优先队列根据长度从大到小进行排序 let mut pq =
    // std::collections::BinaryHeap::new(); for i in 0..aus.len() {
    //     pq.push((aus[i].len(), i));
    // }
    // let n = aus.len();
    // // 进行n次切分，每次切分取出最大的集合进行切分，并更新相应split_times
    // let mut split_times = vec![0; n];
    // let mut cnt = 0;
    // for _ in 0..depth {
    //     let (len, i) = pq.pop().unwrap();
    //     // 如果len为1，直接跳出
    //     if len == 1 {
    //         break;
    //     }
    //     split_times[i] += 1;
    //     pq.push((len / 2, i));
    //     cnt += 1;
    // }
    // // 如果cnt < depth,
    // 那么说明有集合的长度为1，那么剩下的切分次数平摊到每个集合
    // if cnt < depth {
    //     let base = (depth - cnt) / n;
    //     for i in 0..n {
    //         split_times[i] += base;
    //     }
    //     for i in 0..(depth - cnt) % n {
    //         split_times[i] += 1;
    //     }
    // }
    // 初始化结果数组
    let mut results = BTreeSet::new();
    let mut segments = Vec::new();
    for i in 0..n {
      segments.push(binary_split(aus[i].len(), split_times[i]));
    }
    // 对segments进行笛卡尔积
    let mut cartesian_product = Vec::new();
    for seg in segments.iter().multi_cartesian_product() {
      let mut vec = Vec::new();
      for i in 0..n {
        vec.push(seg[i].clone());
      }
      cartesian_product.push(vec);
    }
    // cartesian_product[i]是抽取第i个解时，每个集合的限制条件，
    // 如果cartesian_product[i].size() =
    // k,那么就说明第i个解的前k个维度的索引值只能从cartesian_product[i][k]中选取，
    // 剩下的维度l的索引值可以从aus[l]中任意选取 遍历cartesian_product
    let rng = &mut thread_rng();
    for i in 0..cartesian_product.len() {
      // 每个方案抽取10个解
      for _ in 0..10 {
        let mut result = Vec::new();
        // 从cartesian_product[i]中的每个集合中随机选取一个元素作为index
        for j in 0..cartesian_product[i].len() {
          // 随机选取cartesian_product[i][j]中的一个元素
          let index = cartesian_product[i][j].choose(rng).unwrap().clone();
          result.push(aus[j][index].clone());
        }
        // 如果cartesian_product[i].size() < n,
        // 那么剩下的维度的索引值可以从aus中任意选取
        for j in cartesian_product[i].len()..n {
          // 在aus[j]中随机选取一个元素
          let au = aus[j].choose(rng).unwrap().clone();
          result.push(au.clone());
        }
        results.insert(au2pe(result.clone()));
      }
    }
    results
  };
  let n = candidates.len();
  let step = n as f64 / m as f64; // 步长计算为浮点数
  let mut selected_elements = Vec::new();
  let selected_aus = candidates.iter().collect::<Vec<_>>();
  for i in 0..m {
    let index = (i as f64 * step).round() as usize;
    // 确保索引不会超出范围
    let index = index.min(n - 1); // 避免越界
    selected_elements.push(
      selected_aus[index]
        .aus
        .clone()
        .iter()
        .map(|x| x.expr().clone())
        .collect(),
    );
  }
  // 加入上下界
  let mut lower_bound = Vec::new();
  let mut upper_bound = Vec::new();
  for i in 0..aus.len() {
    lower_bound.push(aus[i][0].clone());
    upper_bound.push(aus[i][aus[i].len() - 1].clone());
  }
  selected_elements
    .push(upper_bound.iter().map(|x| x.expr().clone()).collect());
  selected_elements
    .push(lower_bound.iter().map(|x| x.expr().clone()).collect());
  selected_elements
}

// 一个小测试，因为实验中发现好像采样方法改变好像对于结果影响不大，
// 所以此处写一个greedy_aus,对于给定的aus，只只取最大值和最小值
pub fn greedy_aus<Op>(
  aus: Vec<Vec<AU<Op, (Id, Id)>>>,
) -> Vec<Vec<PartialExpr<Op, (Id, Id)>>>
where
  Op: Clone
    + Debug
    + Hash
    + Ord
    + Display
    + Arity
    + Teachable
    + Sync
    + Send
    + 'static,
{
  // 如果aus是空的或者aus中的元素是空的，直接返回空
  if aus.is_empty() || aus.iter().any(|x| x.is_empty()) {
    return vec![];
  }
  // 将aus中的每个Vec的第一个和最后一个元素取出
  let mut result = Vec::new();
  // let lower_bound = aus.iter().map(|x| x[0].clone()).collect::<Vec<_>>();
  let upper_bound = aus
    .iter()
    .map(|x| x[x.len() - 1].clone())
    .collect::<Vec<_>>();
  // result.push(lower_bound);
  result.push(upper_bound);
  result
    .iter()
    .map(|x| x.iter().map(|y| y.expr().clone()).collect())
    .collect()
}
