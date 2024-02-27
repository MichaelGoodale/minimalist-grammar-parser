use std::marker::PhantomData;

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use lexicon::Lexicon;

use logprob::LogProb;
use min_max_heap::MinMaxHeap;
use neural_lexicon::NeuralLexicon;
use parsing::beam::neural_beam::NeuralBeam;
use parsing::beam::{Beam, FuzzyBeam, GeneratorBeam, ParseBeam};
use parsing::expand;
use parsing::Rule;
use rand::Rng;

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
    min_log_prob: LogProb<f64>,
    move_prob: LogProb<f64>,
    max_steps: usize,
    max_beams: usize,
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
            min_log_prob,
            move_prob,
            max_steps,
            max_beams,
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
            min_log_prob,
            move_prob,
            max_steps,
            max_beams,
            global_steps: Some(global_steps),
        }
    }
}

#[derive(Debug, Clone)]
struct ParseHeap<'a, T, B: Beam<T>> {
    parse_heap: MinMaxHeap<B>,
    phantom: PhantomData<T>,
    global_steps: usize,
    done: bool,
    config: &'a ParsingConfig,
}

impl<'a, T, B: Beam<T>> ParseHeap<'a, T, B>
where
    B: Ord,
{
    fn pop(&mut self) -> Option<B> {
        self.parse_heap.pop_max()
    }

    fn push(&mut self, v: B) {
        self.global_steps += 1;
        if let Some(max_steps) = self.config.global_steps {
            self.done = self.global_steps > max_steps;
        }
        if !self.done && v.pushable(self.config) {
            if self.parse_heap.len() > self.config.max_beams {
                self.parse_heap.push_pop_min(v);
            } else {
                self.parse_heap.push(v);
            }
        }
    }
}

type ParserOutput<'a, T> = (LogProb<f64>, &'a [T], Vec<Rule>);
type GeneratorOutput<T> = (LogProb<f64>, Vec<T>, Vec<Rule>);

pub struct FuzzyParser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>
{
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<'a, T, FuzzyBeam<'a, T>>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(FuzzyBeam::new(lexicon, initial_category, sentences, true)?);
        Ok(FuzzyParser {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap {
                global_steps: 0,
                done: false,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(FuzzyBeam::new(lexicon, initial_category, sentences, false)?);
        Ok(FuzzyParser {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap {
                global_steps: 0,
                done: false,
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

pub struct Parser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<'a, T, ParseBeam<'a, T>>,
    move_log_prob: LogProb<f64>,
    merge_log_prob: LogProb<f64>,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
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
                done: false,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
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
                done: false,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
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
                done: false,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
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
                done: false,
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
pub struct Generator<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<'a, T, GeneratorBeam<T>>,
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
        parse_heap.push(GeneratorBeam::new(lexicon, initial_category, true)?);
        Ok(Generator {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap {
                global_steps: 0,
                done: false,
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
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(GeneratorBeam::new(lexicon, initial_category, false)?);
        Ok(Generator {
            lexicon,
            move_log_prob: config.move_prob,
            merge_log_prob: config.move_prob.opposite_prob(),
            parse_heap: ParseHeap {
                global_steps: 0,
                done: false,
                parse_heap,
                config,
                phantom: PhantomData,
            },
        })
    }
}

impl<T, Category> Iterator for Generator<'_, T, Category>
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

pub fn get_neural_outputs<B: Backend>(
    lemmas: Tensor<B, 3>,
    types: Tensor<B, 3>,
    categories: Tensor<B, 3>,
    n_grammars: usize,
    n_words: usize,
    n_padding: usize,
    config: &ParsingConfig,
    rng: &mut impl Rng,
) -> Tensor<B, 3>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem> + Into<f32>,
{
    let n_lemmas = lemmas.shape().dims[2];
    let mut tensor = Tensor::<B, 3>::zeros([n_words, n_padding, n_lemmas], &lemmas.device());
    for _ in 0..n_grammars {
        let lexicon =
            NeuralLexicon::new_random(types.clone(), lemmas.clone(), categories.clone(), rng);

        for (n, x) in NeuralGenerator::new(&lexicon, config)
            .filter(|x| x.shape().dims[0] < n_words)
            .take(n_words)
            .enumerate()
        {
            let length = x.shape().dims[0];
            let slice = [n..n + 1, 0..n_padding, 0..n_lemmas];
            let padding_prob = x
                .clone()
                .slice([length - 1..length, 0..n_lemmas])
                .repeat(0, n_padding - length);
            let x: Tensor<B, 3> = Tensor::cat(vec![x, padding_prob], 0).unsqueeze_dim(0);
            let x = tensor.clone().slice(slice.clone()) + x;
            tensor = tensor.slice_assign(slice, x);
        }
    }
    tensor
}

#[derive(Debug)]
pub struct NeuralGenerator<'a, B: Backend>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem>,
{
    lexicon: &'a NeuralLexicon<B>,
    parse_heap: ParseHeap<'a, (usize, usize), NeuralBeam<'a, B>>,
    move_log_prob: B::FloatElem,
    merge_log_prob: B::FloatElem,
}

impl<'a, B: Backend> NeuralGenerator<'a, B>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem>,
{
    pub fn new(lexicon: &'a NeuralLexicon<B>, config: &'a ParsingConfig) -> NeuralGenerator<'a, B> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams);
        parse_heap.push(NeuralBeam::new(lexicon, 0, false).unwrap());
        NeuralGenerator {
            lexicon,
            move_log_prob: Tensor::<B, 1>::from_floats(
                [config.move_prob.into_inner() as f32],
                lexicon.device(),
            )
            .into_scalar(),
            merge_log_prob: Tensor::<B, 1>::from_floats(
                [config.move_prob.opposite_prob().into_inner() as f32],
                lexicon.device(),
            )
            .into_scalar(),
            parse_heap: ParseHeap {
                global_steps: 0,
                done: false,
                parse_heap,
                config,
                phantom: PhantomData,
            },
        }
    }
}

impl<B: Backend> Iterator for NeuralGenerator<'_, B>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem>,
{
    type Item = Tensor<B, 2>;

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
                return Some(sentence);
            }
        }
        None
    }
}

pub mod grammars;
pub mod lexicon;
pub mod neural_lexicon;
mod parsing;
pub mod tree_building;

#[cfg(test)]
mod tests;
