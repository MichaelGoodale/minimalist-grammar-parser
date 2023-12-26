use std::collections::BinaryHeap;

use anyhow::{bail, Result};
use lexicon::Lexicon;

use parsing::beam::ParseBeam;
use parsing::expand;

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
) -> Result<()> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(ParseBeam::<T>::new(lexicon, initial_category, sentence)?);
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(
                expand(
                    &moment,
                    &beam,
                    lexicon,
                    false,
                    config.merge_log_prob,
                    config.move_log_prob,
                )
                .filter(|b| b.log_probability > config.min_log_prob),
            )
        } else {
            if beam.good_parse() {
                println!("Found parse");
                return Ok(());
            }
            break;
        }
    }
    bail!("No parse found :(")
}

pub fn generate<T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>(
    lexicon: &Lexicon<T, Category>,
    initial_category: Category,
    config: &ParsingConfig,
) -> Vec<(f64, Vec<T>)> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(ParseBeam::<T>::new(lexicon, initial_category, vec![]).unwrap());
    let mut v = vec![];
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(
                expand(
                    &moment,
                    &beam,
                    lexicon,
                    true,
                    config.merge_log_prob,
                    config.move_log_prob,
                )
                .filter(|b| b.log_probability > config.min_log_prob),
            )
        } else if beam.queue.is_empty() {
            v.push((beam.log_probability, beam.sentence));
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

#[cfg(test)]
mod tests;
