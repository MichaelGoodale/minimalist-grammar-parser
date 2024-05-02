use std::hash::Hash;

use anyhow::Result;
use bumpalo::Bump;
use lexicon::Lexicon;

use allocator_api2::alloc::{Allocator, Global};
use handlers::ParseHeap;
use logprob::LogProb;
use min_max_heap::MinMaxHeap;
use parsing::beam::{Beam, FuzzyBeam, GeneratorBeam, ParseBeam};
use parsing::expand;
pub use parsing::ParsingConfig;
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

impl From<Direction> for bool {
    fn from(value: Direction) -> Self {
        match value {
            Direction::Left => false,
            Direction::Right => true,
        }
    }
}

impl From<bool> for Direction {
    fn from(value: bool) -> Self {
        match value {
            false => Direction::Left,
            true => Direction::Right,
        }
    }
}

type ParserOutput<'a, T> = (LogProb<f64>, &'a [T], Vec<Rule>);
type GeneratorOutput<T> = (LogProb<f64>, Vec<T>, Vec<Rule>);

pub struct FuzzyParser<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<'a, T, FuzzyBeam<'a, T>>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
}

impl<'a, T, Category> FuzzyParser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
{
    pub fn new<U>(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
        config: &'a ParsingConfig,
    ) -> Result<Self>
    where
        U: AsRef<[T]>,
    {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(FuzzyBeam::new(lexicon, initial_category, sentences, true)?);
        Ok(FuzzyParser {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }

    pub fn new_skip_rules<U>(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
        config: &'a ParsingConfig,
    ) -> Result<Self>
    where
        U: AsRef<[T]>,
    {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(FuzzyBeam::new(lexicon, initial_category, sentences, false)?);
        Ok(FuzzyParser {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }
}
impl<'a, T, Category> Iterator for FuzzyParser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
{
    type Item = GeneratorOutput<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop() {
            if let Some(moment) = beam.pop_moment() {
                expand(
                    &mut self.parse_heap,
                    moment,
                    beam,
                    self.lexicon,
                    self.move_log_prob,
                    self.merge_log_prob,
                );
            } else if let Some(x) = beam.yield_good_parse() {
                return Some(x);
            }
        }

        None
    }
}

pub struct Parser<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<'a, T, ParseBeam<'a, T>>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
    buffer: Vec<ParserOutput<'a, T>>,
}

impl<'a, T, Category> Parser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
{
    pub fn new(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
        config: &'a ParsingConfig,
    ) -> Result<Parser<'a, T, Category>> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(ParseBeam::new_single(
            lexicon,
            initial_category,
            sentence,
            true,
        )?);
        Ok(Parser {
            lexicon,
            move_log_prob: config.move_prob,
            buffer: vec![],
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }

    pub fn new_skip_rules(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
        config: &'a ParsingConfig,
    ) -> Result<Parser<'a, T, Category>> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(ParseBeam::new_single(
            lexicon,
            initial_category,
            sentence,
            false,
        )?);
        Ok(Parser {
            lexicon,
            buffer: vec![],
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }

    pub fn new_multiple<U>(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
        config: &'a ParsingConfig,
    ) -> Result<Parser<'a, T, Category>>
    where
        U: AsRef<[T]>,
    {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(ParseBeam::new_multiple(
            lexicon,
            initial_category,
            sentences,
            true,
        )?);
        Ok(Parser {
            lexicon,
            buffer: vec![],
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }

    pub fn new_skip_rules_multiple<U>(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
        config: &'a ParsingConfig,
    ) -> Result<Parser<'a, T, Category>>
    where
        U: AsRef<[T]>,
    {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(ParseBeam::new_multiple(
            lexicon,
            initial_category,
            sentences,
            false,
        )?);
        Ok(Parser {
            lexicon,
            buffer: vec![],
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }
}

impl<'a, T, Category> Iterator for Parser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
{
    type Item = ParserOutput<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            while let Some(mut beam) = self.parse_heap.pop() {
                if let Some(moment) = beam.pop_moment() {
                    expand(
                        &mut self.parse_heap,
                        moment,
                        beam,
                        self.lexicon,
                        self.move_log_prob,
                        self.merge_log_prob,
                    );
                } else if let Some((mut good_parses, p, rules)) = beam.yield_good_parse() {
                    if let Some(next_sentence) = good_parses.next() {
                        self.buffer
                            .extend(good_parses.map(|x| (p, x, rules.clone())));
                        let next = Some((p, next_sentence, rules));
                        return next;
                    }
                }
            }
        } else {
            return self.buffer.pop();
        }

        None
    }
}

#[derive(Debug)]
pub struct Generator<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
    A: Allocator = Global,
> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<'a, T, GeneratorBeam<T>, A>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
}

impl<'a, T, Category> Generator<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
{
    pub fn new(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        config: &'a ParsingConfig,
    ) -> Result<Generator<'a, T, Category>> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(GeneratorBeam::new(lexicon, initial_category, true)?);
        Ok(Generator {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }

    pub fn new_skip_rules(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        config: &'a ParsingConfig,
    ) -> Result<Generator<'a, T, Category>> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(5));
        parse_heap.push(GeneratorBeam::new(lexicon, initial_category, false)?);
        Ok(Generator {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }

    pub fn new_skip_rules_bump<'b>(
        lexicon: &'a Lexicon<T, Category>,
        initial_category: Category,
        config: &'a ParsingConfig,
        bumpalo: &'b Bump,
    ) -> Result<Generator<'a, T, Category, &'b Bump>> {
        let mut parse_heap = MinMaxHeap::new_in(bumpalo);
        parse_heap.push(GeneratorBeam::new(lexicon, initial_category, false)?);
        Ok(Generator {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap::new(parse_heap, config),
        })
    }
}

impl<T, Category, A: Allocator> Iterator for Generator<'_, T, Category, A>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Hash + Eq + Clone + std::fmt::Debug,
{
    type Item = GeneratorOutput<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop() {
            if let Some(moment) = beam.pop_moment() {
                expand(
                    &mut self.parse_heap,
                    moment,
                    beam,
                    self.lexicon,
                    self.move_log_prob,
                    self.merge_log_prob,
                );
            } else if beam.queue.is_empty() {
                return Some((beam.log_probability, beam.sentence, beam.rules.to_vec()));
            }
        }
        None
    }
}

pub mod grammars;
mod handlers;
pub mod lexicon;
#[allow(clippy::single_range_in_vec_init)]
pub mod neural;
mod parsing;
pub mod tree_building;

#[cfg(test)]
mod tests;
