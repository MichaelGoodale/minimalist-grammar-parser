use std::collections::BinaryHeap;

use anyhow::Result;
use lexicon::Lexicon;

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

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct ParsingConfig {
    pub min_log_prob: f64,
    pub merge_log_prob: f64,
    pub move_log_prob: f64,
    pub max_steps: usize,
}

pub struct Parser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    config: &'a ParsingConfig,
    parse_heap: BinaryHeap<Beam<&'a T>>,
}

impl<'a, T, Category> Parser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    pub fn new(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
        config: &'a ParsingConfig,
    ) -> Result<Parser<'a, T, Category>> {
        let mut parse_heap = BinaryHeap::new();
        let sentence: Vec<_> = sentence.iter().collect();
        parse_heap.push(Beam::new(lexicon, initial_category, sentence)?);
        Ok(Parser {
            lexicon,
            config,
            parse_heap,
        })
    }
}

impl<T, Category> Iterator for Parser<'_, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    type Item = (f64, Vec<Rule>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop() {
            if let Some(moment) = beam.pop() {
                self.parse_heap.extend(
                    expand_parse(
                        moment,
                        beam,
                        self.lexicon,
                        self.config.merge_log_prob,
                        self.config.move_log_prob,
                    )
                    .filter(|b| b.log_probability > self.config.min_log_prob)
                    .filter(|b| b.steps < self.config.max_steps),
                )
            } else if beam.good_parse() {
                return Some((beam.log_probability, beam.rules));
            }
        }
        None
    }
}

pub struct Generator<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    config: &'a ParsingConfig,
    parse_heap: BinaryHeap<Beam<T>>,
}

impl<'a, T, Category> Generator<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    pub fn new(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        config: &'a ParsingConfig,
    ) -> Result<Generator<'a, T, Category>> {
        let mut parse_heap = BinaryHeap::new();
        parse_heap.push(Beam::new_empty(lexicon, initial_category)?);
        Ok(Generator {
            lexicon,
            config,
            parse_heap,
        })
    }
}

impl<T, Category> Iterator for Generator<'_, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    type Item = (f64, Vec<T>, Vec<Rule>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop() {
            if let Some(moment) = beam.pop() {
                self.parse_heap.extend(
                    expand_generate(
                        &moment,
                        &beam,
                        self.lexicon,
                        self.config.merge_log_prob,
                        self.config.move_log_prob,
                    )
                    .filter(|b| b.log_probability > self.config.min_log_prob)
                    .filter(|b| b.steps < self.config.max_steps),
                )
            } else if beam.queue.is_empty() {
                return Some((beam.log_probability, beam.sentence, beam.rules));
            }
        }
        None
    }
}

mod grammars;
pub mod lexicon;
mod parsing;
pub mod tree_building;

#[cfg(test)]
mod tests;
