use std::collections::BTreeSet;
use std::hash::Hash;
use std::marker::PhantomData;

use ahash::HashMap;
use anyhow::Result;
use bumpalo::Bump;
use burn::tensor::backend::Backend;
use lexicon::Lexicon;
use petgraph::graph::NodeIndex;

use allocator_api2::alloc::{Allocator, Global};
use logprob::LogProb;
use min_max_heap::MinMaxHeap;
use neural::neural_beam::{NeuralBeam, StringPath, StringProbHistory};
use neural::neural_lexicon::{NeuralLexicon, NeuralProbability, NeuralProbabilityRecord};
use parsing::beam::{Beam, FuzzyBeam, GeneratorBeam, ParseBeam};
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

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct ParsingConfig {
    min_log_prob: Option<LogProb<f64>>,
    move_prob: LogProb<f64>,
    max_steps: Option<usize>,
    max_beams: Option<usize>,
    global_steps: Option<usize>,
}

impl ParsingConfig {
    pub fn new(
        min_log_prob: LogProb<f64>,
        move_prob: LogProb<f64>,
        max_steps: usize,
        max_beams: usize,
    ) -> ParsingConfig {
        let max_steps = usize::min(parsing::MAX_STEPS, max_steps);
        ParsingConfig {
            min_log_prob: Some(min_log_prob),
            move_prob,
            max_steps: Some(max_steps),
            max_beams: Some(max_beams),
            global_steps: None,
        }
    }

    pub fn new_steps_only(move_prob: LogProb<f64>, max_steps: usize) -> ParsingConfig {
        let max_steps = usize::min(parsing::MAX_STEPS, max_steps);
        ParsingConfig {
            min_log_prob: None,
            move_prob,
            max_steps: Some(max_steps),
            max_beams: None,
            global_steps: None,
        }
    }

    pub fn new_with_global_steps(
        min_log_prob: LogProb<f64>,
        move_prob: LogProb<f64>,
        max_steps: usize,
        max_beams: usize,
        global_steps: usize,
    ) -> ParsingConfig {
        let max_steps = usize::min(parsing::MAX_STEPS, max_steps);
        ParsingConfig {
            min_log_prob: Some(min_log_prob),
            move_prob,
            max_steps: Some(max_steps),
            max_beams: Some(max_beams),
            global_steps: Some(global_steps),
        }
    }
}

#[derive(Debug, Clone)]
struct ParseHeap<'a, T, B: Beam<T>, A: Allocator = Global> {
    parse_heap: MinMaxHeap<B, A>,
    phantom: PhantomData<T>,
    global_steps: usize,
    config: &'a ParsingConfig,
}

impl<'a, T, B: Beam<T>, A: Allocator> ParseHeap<'a, T, B, A>
where
    B: Ord,
{
    fn pop(&mut self) -> Option<B> {
        self.parse_heap.pop_max()
    }

    fn push(&mut self, v: B) {
        self.global_steps += 1;
        let mut pushable = true;
        if let Some(max_steps) = self.config.global_steps {
            pushable = self.global_steps > max_steps;
        }

        if let Some(min_log_prob) = self.config.min_log_prob {
            if v.log_prob() < min_log_prob {
                pushable = false;
            }
        }

        if let Some(max_steps) = self.config.max_steps {
            if v.n_steps() > max_steps {
                pushable = false;
            }
        }

        if pushable && v.pushable(self.config) {
            if let Some(max_beams) = self.config.max_beams {
                if self.parse_heap.len() > max_beams {
                    self.parse_heap.push_pop_min(v);
                } else {
                    self.parse_heap.push(v);
                }
            } else {
                self.parse_heap.push(v);
            }
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
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

#[derive(Debug)]
pub struct NeuralGenerator<'a, B: Backend> {
    lexicon: &'a NeuralLexicon<B>,
    parse_heap: ParseHeap<'a, usize, NeuralBeam<'a>>,
    target_lens: Option<BTreeSet<usize>>,
    move_log_prob: NeuralProbability,
    merge_log_prob: NeuralProbability,
}

impl<'a, B: Backend> NeuralGenerator<'a, B> {
    pub fn new(
        lexicon: &'a NeuralLexicon<B>,
        targets: Option<&'a [Vec<usize>]>,
        lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
        weight_lookups: &'a HashMap<usize, LogProb<f64>>,
        alternatives: &'a HashMap<NodeIndex, Vec<NodeIndex>>,
        max_string_length: Option<usize>,
        config: &'a ParsingConfig,
    ) -> NeuralGenerator<'a, B> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(10000000));
        let target_lens = targets.map(|x| x.iter().map(|x| x.len()).collect());
        parse_heap.extend(
            NeuralBeam::new(
                lexicon,
                0,
                targets,
                lemma_lookups,
                weight_lookups,
                alternatives,
                max_string_length,
                false,
            )
            .unwrap(),
        );
        NeuralGenerator {
            lexicon,
            target_lens,
            move_log_prob: NeuralProbability(
                NeuralProbabilityRecord::MoveRuleProb,
                None,
                config.move_prob,
            ),
            merge_log_prob: NeuralProbability(
                NeuralProbabilityRecord::MergeRuleProb,
                None,
                config.move_prob.opposite_prob(),
            ),
            parse_heap: ParseHeap {
                global_steps: 0,
                parse_heap,
                config,
                phantom: PhantomData,
            },
        }
    }
}

impl<B: Backend> Iterator for NeuralGenerator<'_, B> {
    type Item = (StringPath, StringProbHistory);

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
            } else if let Some(sentence) = beam.yield_good_parse() {
                if self
                    .target_lens
                    .as_ref()
                    .map_or(true, |x| x.contains(&sentence.0.len()))
                {
                    return Some(sentence);
                }
            }
        }
        None
    }
}

pub mod grammars;
pub mod lexicon;
#[allow(clippy::single_range_in_vec_init)]
pub mod neural;
mod parsing;
pub mod tree_building;

#[cfg(test)]
mod tests;
