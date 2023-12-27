use std::collections::BinaryHeap;

use anyhow::{bail, Result};
use lexicon::{FeatureOrLemma, Lexicon};

use parsing::beam::Beam;
use parsing::Rule;
use parsing::{expand_generate, expand_parse};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    Left,
    Right,
}

impl Direction {
    pub fn flip(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

pub struct ParsingConfig {
    pub min_log_prob: f64,
    pub merge_log_prob: f64,
    pub move_log_prob: f64,
    pub max_parses: usize,
}

pub fn parse<T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>(
    lexicon: &Lexicon<T, Category>,
    initial_category: Category,
    sentence: Vec<T>,
    config: &ParsingConfig,
) -> Result<Vec<Vec<Rule>>>
where
    FeatureOrLemma<T, Category>: std::fmt::Display,
{
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(Beam::new(lexicon, initial_category, &sentence)?);
    let mut parses = vec![];
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(
                expand_parse(
                    moment,
                    beam,
                    lexicon,
                    config.merge_log_prob,
                    config.move_log_prob,
                )
                .filter(|b| b.log_probability > config.min_log_prob),
            )
        } else if beam.good_parse() {
            parses.push(beam.rules);
            if parses.len() == config.max_parses {
                break;
            }
        }
    }
    if parses.is_empty() {
        bail!("No parses :(");
    }
    Ok(parses)
}

pub fn generate<T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>(
    lexicon: &Lexicon<T, Category>,
    initial_category: Category,
    config: &ParsingConfig,
) -> Vec<(f64, Vec<T>, Vec<Rule>)> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(Beam::new_empty(lexicon, initial_category).unwrap());
    let mut v = vec![];
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(
                expand_generate(
                    &moment,
                    &beam,
                    lexicon,
                    config.merge_log_prob,
                    config.move_log_prob,
                )
                .filter(|b| b.log_probability > config.min_log_prob),
            )
        } else if beam.queue.is_empty() {
            v.push((beam.log_probability, beam.sentence, beam.rules));
            if v.len() >= config.max_parses {
                break;
            }
        }
    }
    v
}

mod grammars;
pub mod lexicon;
mod parsing;
pub mod tree_building;

#[cfg(test)]
mod tests;
