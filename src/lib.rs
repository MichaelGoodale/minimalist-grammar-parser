use std::borrow::Borrow;
use std::marker::PhantomData;

use anyhow::Result;
use lexicon::Lexicon;

use logprob::LogProb;
use min_max_heap::MinMaxHeap;
use parsing::beam::{FuzzyScan, GeneratorScan, ParseScan, Scanner};
use parsing::{BeamWrapper, PartialRulePool, expand};
use parsing::{RuleHolder, RulePool};
use petgraph::graph::NodeIndex;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ParsingConfig {
    min_log_prob: LogProb<f64>,
    move_prob: LogProb<f64>,
    dont_move_prob: LogProb<f64>,
    max_steps: usize,
    max_beams: usize,
}

impl ParsingConfig {
    pub fn new(
        min_log_prob: LogProb<f64>,
        move_prob: LogProb<f64>,
        max_steps: usize,
        max_beams: usize,
    ) -> ParsingConfig {
        let max_steps = usize::min(parsing::MAX_STEPS, max_steps);
        let merge_prob = move_prob.opposite_prob();
        ParsingConfig {
            min_log_prob,
            move_prob,
            dont_move_prob: merge_prob,
            max_steps,
            max_beams,
        }
    }
}

impl Default for ParsingConfig {
    fn default() -> Self {
        ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            256,
            256,
        )
    }
}

#[derive(Debug, Clone)]
struct ParseHeap<T, B: Scanner<T>> {
    parse_heap: MinMaxHeap<BeamWrapper<T, B>>,
    phantom: PhantomData<T>,
    config: ParsingConfig,
    rule_arena: Vec<RuleHolder>,
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> ParseHeap<T, B> {
    fn pop(&mut self) -> Option<BeamWrapper<T, B>> {
        self.parse_heap.pop_max()
    }

    fn push(&mut self, v: BeamWrapper<T, B>) {
        if v.log_prob() > self.config.min_log_prob && v.n_steps() < self.config.max_steps {
            if self.parse_heap.len() > self.config.max_beams {
                self.parse_heap.push_pop_min(v);
            } else {
                self.parse_heap.push(v);
            }
        }
    }

    fn new(start: BeamWrapper<T, B>, config: &ParsingConfig, cat: NodeIndex) -> Self {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(start);
        ParseHeap {
            parse_heap,
            phantom: PhantomData,
            config: *config,
            rule_arena: PartialRulePool::default_pool(cat),
        }
    }

    fn rules_mut(&mut self) -> &mut Vec<RuleHolder> {
        &mut self.rule_arena
    }
}

type ParserOutput<'a, T> = (LogProb<f64>, &'a [T], RulePool);
type GeneratorOutput<T> = (LogProb<f64>, Vec<T>, RulePool);

pub struct FuzzyParser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>
{
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<T, FuzzyScan<'a, T>>,
    config: &'a ParsingConfig,
}

impl<'a, T, Category> FuzzyParser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
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
        let cat = lexicon.find_category(&initial_category)?;
        let parse_heap = ParseHeap::new(FuzzyScan::new(cat, sentences)?, config, cat);
        Ok(FuzzyParser {
            lexicon,
            config,
            parse_heap,
        })
    }
}
impl<T, Category> Iterator for FuzzyParser<'_, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
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
                    self.config,
                );
            } else if let Some(x) = FuzzyScan::yield_good_parse(beam, &self.parse_heap.rule_arena) {
                return Some(x);
            }
        }

        None
    }
}

pub struct Parser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<T, ParseScan<'a, T>>,
    config: &'a ParsingConfig,
    buffer: Vec<ParserOutput<'a, T>>,
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
        let cat = lexicon.find_category(&initial_category)?;
        let parse_heap = ParseHeap::new(ParseScan::new_single(cat, sentence)?, config, cat);
        Ok(Parser {
            lexicon,
            config,
            buffer: vec![],
            parse_heap,
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
        let cat = lexicon.find_category(&initial_category)?;
        let parse_heap = ParseHeap::new(ParseScan::new_multiple(cat, sentences)?, config, cat);
        Ok(Parser {
            lexicon,
            buffer: vec![],
            config,
            parse_heap,
        })
    }
}

impl<'a, T, Category> Iterator for Parser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
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
                        self.config,
                    );
                } else if let Some((mut good_parses, p, rules)) =
                    ParseScan::yield_good_parse(beam, &self.parse_heap.rule_arena)
                {
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
pub struct Generator<L, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>
where
    L: Borrow<Lexicon<T, Category>>,
{
    lexicon: L,
    phantom: PhantomData<Category>,
    parse_heap: ParseHeap<T, GeneratorScan<T>>,
    config: ParsingConfig,
}

impl<L, T, Category> Generator<L, T, Category>
where
    L: Borrow<Lexicon<T, Category>>,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    pub fn new(
        lexicon: L,
        initial_category: Category,
        config: &ParsingConfig,
    ) -> Result<Generator<L, T, Category>> {
        let cat = lexicon.borrow().find_category(&initial_category)?;
        let parse_heap = ParseHeap::new(GeneratorScan::new(cat)?, config, cat);
        Ok(Generator {
            lexicon,
            config: *config,
            parse_heap,
            phantom: PhantomData,
        })
    }
}

impl<L, T, Category> Iterator for Generator<L, T, Category>
where
    L: Borrow<Lexicon<T, Category>>,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    type Item = GeneratorOutput<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop() {
            if let Some(moment) = beam.pop_moment() {
                expand(
                    &mut self.parse_heap,
                    moment,
                    beam,
                    self.lexicon.borrow(),
                    &self.config,
                );
            } else if let Some(x) =
                GeneratorScan::yield_good_parse(beam, &self.parse_heap.rule_arena)
            {
                return Some(x);
            }
        }
        None
    }
}

pub mod grammars;
pub mod lexicon;
mod parsing;

#[cfg(test)]
mod tests;
