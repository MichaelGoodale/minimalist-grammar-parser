use anyhow::Result;
use lexicon::Lexicon;

use logprob::LogProb;
use min_max_heap::MinMaxHeap;
use parsing::beam::{Beam, GeneratorBeam, ParseBeam};
use parsing::expand;
use parsing::Rule;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Direction {
    #[default]
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
    pub min_log_prob: LogProb<f64>,
    pub move_prob: LogProb<f64>,
    pub max_steps: usize,
    pub max_beams: usize,
}

pub struct Parser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    config: &'a ParsingConfig,
    parse_heap: MinMaxHeap<ParseBeam<'a, T>>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(ParseBeam::new(lexicon, initial_category, sentence)?);
        Ok(Parser {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
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
    type Item = (LogProb<f64>, Vec<Rule>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop_max() {
            if let Some(moment) = beam.pop_moment() {
                self.parse_heap.extend(
                    expand(
                        moment,
                        beam,
                        self.lexicon,
                        self.move_log_prob,
                        self.merge_log_prob,
                    )
                    .filter(|b| b.log_probability > self.config.min_log_prob)
                    .filter(|b| b.steps < self.config.max_steps),
                );
                if self.parse_heap.len() > self.config.max_beams {
                    let n_to_remove = self.parse_heap.len() - self.config.max_beams;
                    for _ in 0..n_to_remove {
                        self.parse_heap.pop_min();
                    }
                };
            } else if beam.good_parse() {
                return Some((beam.log_probability, beam.rules));
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct Generator<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    config: &'a ParsingConfig,
    parse_heap: MinMaxHeap<GeneratorBeam<T>>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(GeneratorBeam::new(lexicon, initial_category)?);
        Ok(Generator {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
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
    type Item = (LogProb<f64>, Vec<T>, Vec<Rule>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop_max() {
            if let Some(moment) = beam.pop_moment() {
                self.parse_heap.extend(
                    expand(
                        moment,
                        beam,
                        self.lexicon,
                        self.move_log_prob,
                        self.merge_log_prob,
                    )
                    .filter(|b| b.log_probability > self.config.min_log_prob)
                    .filter(|b| b.steps < self.config.max_steps),
                );
                if self.parse_heap.len() > self.config.max_beams {
                    let n_to_remove = self.parse_heap.len() - self.config.max_beams;
                    for _ in 0..n_to_remove {
                        self.parse_heap.pop_min();
                    }
                };
            } else if beam.queue.is_empty() {
                return Some((beam.log_probability, beam.sentence, beam.rules));
            }
        }
        None
    }
}

pub mod grammars;
pub mod lexicon;
mod parsing;
pub mod tree_building;

#[cfg(test)]
mod tests;
