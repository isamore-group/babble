use std::{collections::HashMap, path::Path};

#[derive(Debug, Clone)]
pub struct BBEntry {
  pub name: String,
  pub execution_count: usize,
  pub execution_count_normalized: f64,
  pub total_ticks: f64,
  pub average_ticks: f64,
  pub instr_count: usize,
  pub operation_count: usize,
  pub cpi: f64,
  pub cpo: f64,
}

impl BBEntry {
  pub fn new(
    name: String,
    execution_count: usize,
    execution_count_normalized: f64,
    total_ticks: f64,
    average_ticks: f64,
    instr_count: usize,
    operation_count: usize,
    cpi: f64,
    cpo: f64,
  ) -> Self {
    Self {
      name,
      execution_count,
      execution_count_normalized,
      total_ticks,
      average_ticks,
      instr_count,
      operation_count,
      cpi,
      cpo,
    }
  }
}

#[derive(Debug, Clone)]
pub struct BBQuery {
  pub map: HashMap<String, BBEntry>,
  pub factor: f64,
}

impl BBQuery {
  pub fn new<P: AsRef<Path>>(csr_path: P) -> Self {
    let mut map = HashMap::new();
    let mut rdr = csv::Reader::from_path(csr_path).unwrap();
    let mut max_execution_count: f64 = 0.0;
    for result in rdr.records() {
      let record = result.unwrap();
      let execution_count = record[1].parse::<f64>().unwrap();
      max_execution_count = max_execution_count.max(execution_count);
    }
    let factor = 10000.0 / max_execution_count;
    for result in rdr.records() {
      let record = result.unwrap();
      let name = record[0].to_string();
      let execution_count = record[1].parse::<usize>().unwrap();
      let execution_count_normalized =
        record[1].parse::<f64>().unwrap() * factor;
      let total_ticks = record[2].parse::<f64>().unwrap();
      let average_ticks = record[3].parse::<f64>().unwrap();
      let instr_count = record[4].parse::<usize>().unwrap();
      let operation_count = record[5].parse::<usize>().unwrap();
      let cpi = record[6].parse::<f64>().unwrap() / 1000.0;
      let cpo =
        total_ticks / execution_count as f64 / operation_count as f64 / 1000.0;
      // println!("bbs: {:?}, cpo: {}, cpi: {}", name, cpo, cpi);
      let cpo = if cpo / cpi > 20.0 { cpi * 2.0 } else { cpo };
      map.insert(
        name.clone(),
        BBEntry::new(
          name,
          execution_count,
          execution_count_normalized,
          total_ticks,
          average_ticks,
          instr_count,
          operation_count,
          cpi,
          cpo,
        ),
      );
    }
    Self { map, factor }
  }

  pub fn get(&self, name: &str) -> Option<&BBEntry> {
    self.map.get(name)
  }

  pub fn dump(&self) -> String {
    format!("{:#?}", self.map)
  }

  pub fn get_factor(&self) -> f64 {
    self.factor
  }
}

impl Default for BBQuery {
  fn default() -> Self {
    Self {
      map: HashMap::new(),
      factor: 1.0,
    }
  }
}

pub trait BBInfo {
  fn get_mut_bbs_info(&mut self) -> &mut Vec<String>;
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bb_query() {
    let bb_query = BBQuery::new(
      std::env::var("CARGO_MANIFEST_DIR").unwrap() + "/resource/bb_tracer.csv",
    );
    println!("{}", bb_query.dump());
  }
}
