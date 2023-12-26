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

pub fn parse<T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>(
    lexicon: &Lexicon<T, Category>,
    initial_category: Category,
    sentence: Vec<T>,
    min_log_prob: f64,
) -> Result<()> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(ParseBeam::<T>::new(lexicon, initial_category, sentence)?);
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(
                expand(&moment, &beam, lexicon, false).filter(|b| b.log_probability > min_log_prob),
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
    min_log_prob: f64,
    max_parses: usize,
) -> Vec<(f64, Vec<T>)> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(ParseBeam::<T>::new(lexicon, initial_category, vec![]).unwrap());
    let mut v = vec![];
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(
                expand(&moment, &beam, lexicon, true).filter(|b| b.log_probability > min_log_prob),
            )
        } else if beam.queue.is_empty() {
            v.push((beam.log_probability, beam.sentence));
            if v.len() >= max_parses {
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
