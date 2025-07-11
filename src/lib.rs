//!This crate defines a number of structs and methods to parse and generate Minimalist Grammars
//!(MGs) from Stabler (1997). Specifically, it implements a variety of the MG algorithm adapted
//!from Stabler (2011) and Stabler (2013)
//!
//!
//!# Examples of usage
//!The following code generates 4 sentences from the $a^nb^n$ language.
//!```
//!use minimalist_grammar_parser::Lexicon;
//!use minimalist_grammar_parser::ParsingConfig;
//!# use minimalist_grammar_parser::lexicon::LexiconParsingError;
//!let lexicon = Lexicon::from_string("a::s= b= s\nb::b\n::s")?;
//!let v = lexicon
//!    .generate("s", &ParsingConfig::default())?
//!    .take(4)
//!    .map(|(_prob, s, _rules)| {
//!        s.into_iter()
//!            .map(|word| word.try_inner().unwrap())
//!            .collect::<Vec<_>>()
//!            .join("")
//!    })
//!    .collect::<Vec<_>>();
//!assert_eq!(v, vec!["", "ab", "aabb", "aaabbb"]);
//!# Ok::<(), anyhow::Error>(())
//!```
//!## References
//!
//! - Stabler, E. (1997). Derivational minimalism. In C. Retoré (Ed.), Logical Aspects of Computational Linguistics (pp. 68–95). Springer. <https://doi.org/10.1007/BFb0052152>
//! - Stabler, E. (2011). Top-Down Recognizers for MCFGs and MGs. In F. Keller & D. Reitter (Eds.), Proceedings of the 2nd Workshop on Cognitive Modeling and Computational Linguistics (pp. 39–48). Association for Computational Linguistics. <https://aclanthology.org/W11-0605>
//! - Stabler, E. (2013). Two Models of Minimalist, Incremental Syntactic Analysis. Topics in Cognitive Science, 5(3), 611–633. <https://doi.org/10.1111/tops.12031>
//!
//!
#![warn(missing_docs)]

use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

pub use lexicon::{Lexicon, ParsingError};

use logprob::LogProb;
use min_max_heap::MinMaxHeap;
use parsing::RuleHolder;

pub use parsing::RulePool;
use parsing::beam::{FuzzyScan, GeneratorScan, ParseScan, Scanner};
use parsing::{BeamWrapper, PartialRulePool, expand};
use petgraph::graph::NodeIndex;

use serde::{Deserialize, Serialize};
use thiserror::Error;

///The basic input type of the library used by generation and parsing a like
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PhonContent<T> {
    ///Normal words
    Normal(T),
    ///Words that are built out of combining heads with head movement
    Affixed(Vec<T>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Error)]
///Error caused by flattening a sentence that has affixes in it.
pub struct FlattenError {}
impl Display for FlattenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "This Input is not a Normal variant")
    }
}

impl<T> PhonContent<T> {
    ///Try to flatten the output assuming that it is [`PhonContent::Normal`]
    pub fn try_inner(self) -> Result<T, FlattenError> {
        match self {
            PhonContent::Normal(x) => Ok(x),
            PhonContent::Affixed(_) => Err(FlattenError {}),
        }
    }

    ///Create new input assuming there are no affixes.
    pub fn new(x: Vec<T>) -> Vec<PhonContent<T>> {
        x.into_iter().map(PhonContent::Normal).collect()
    }

    ///Create new input assuming there are no affixes
    pub fn from<const N: usize>(x: [T; N]) -> [PhonContent<T>; N] {
        x.map(PhonContent::Normal)
    }

    ///Try to flatten the output assuming that all members of the vector are
    ///[`PhonContent::Normal`]
    pub fn try_flatten(x: Vec<PhonContent<T>>) -> Result<Vec<T>, FlattenError> {
        x.into_iter().map(|x| x.try_inner()).collect()
    }
}
impl PhonContent<&str> {
    ///Try to flatten the output and join all affixes without spaces
    pub fn flatten(x: Vec<PhonContent<&str>>) -> Vec<String> {
        let mut v = vec![];
        for content in x.into_iter() {
            match content {
                PhonContent::Normal(val) => v.push(val.to_string()),
                PhonContent::Affixed(items) => v.push(items.join("")),
            }
        }
        v
    }
}

///Enum to record the direction of a merge/move operation (whether the phonological value goes to
///the right or left)
#[allow(missing_docs)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Direction {
    #[default]
    Left,
    Right,
}

impl Direction {
    ///Swaps direction so that left is right and vice-versa
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

///This struct defines the configuration used when parsing a [`Lexicon`].
///
///It has the following options:
/// - `min_log_prob`: The lowest probability string that the parser will consider. (Default = -256)
/// - `move_prob`: The probability of moving rather than merging when both are available (Default = log(0.5))
/// - `max_steps`: The maximum number of derivational steps before crashing (Default = 256)
/// - `max_beams`: The maximum number of competing parses available in the parse heap at a single time (Default = 256)
/// - `max_time`: The maximum amount of *time* before the parse crashes (not available on `wasm32). Disabled by default
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ParsingConfig {
    min_log_prob: Option<LogProb<f64>>,
    move_prob: LogProb<f64>,
    dont_move_prob: LogProb<f64>,
    max_steps: Option<usize>,
    max_beams: Option<usize>,

    #[cfg(not(target_arch = "wasm32"))]
    max_time: Option<Duration>,
}

impl ParsingConfig {
    ///Create a new [`ParsingConfig`] with no limits on parsing and default move probability. Be careful to ensure when parsing
    ///or generating with this config to avoid infinite loops (at the very least use
    ///[`ParsingConfig::with_max_time`]).
    pub fn empty() -> ParsingConfig {
        let move_prob = LogProb::from_raw_prob(0.5).unwrap();
        let dont_move_prob = move_prob.opposite_prob();

        ParsingConfig {
            min_log_prob: None,
            move_prob,
            dont_move_prob,
            max_steps: None,
            max_beams: None,

            #[cfg(not(target_arch = "wasm32"))]
            max_time: None,
        }
    }

    ///Create a new [`ParsingConfig`] with the following parameters
    pub fn new(
        min_log_prob: LogProb<f64>,
        move_prob: LogProb<f64>,
        max_steps: usize,
        max_beams: usize,
    ) -> ParsingConfig {
        let max_steps = usize::min(parsing::MAX_STEPS, max_steps);
        let merge_prob = move_prob.opposite_prob();
        ParsingConfig {
            min_log_prob: Some(min_log_prob),
            move_prob,
            dont_move_prob: merge_prob,
            max_steps: Some(max_steps),
            max_beams: Some(max_beams),
            #[cfg(not(target_arch = "wasm32"))]
            max_time: None,
        }
    }

    ///Set the maximum time before timing out a parse (not available on `wasm32`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_max_time(mut self, duration: Duration) -> Self {
        self.max_time = Some(duration);
        self
    }

    ///Set the minimum log probability for a parse.
    pub fn with_min_log_prob(mut self, min_log_prob: LogProb<f64>) -> Self {
        self.min_log_prob = Some(min_log_prob);
        self
    }

    ///Set the maximum number of derivational steps for a parse.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    ///Set the maximum number of competing parses at a single time.
    pub fn with_max_beams(mut self, max_beams: usize) -> Self {
        self.max_beams = Some(max_beams);
        self
    }

    ///Set the probability of moving as opposed to merging.
    pub fn with_move_prob(mut self, move_prob: LogProb<f64>) -> Self {
        self.move_prob = move_prob;
        self.dont_move_prob = self.move_prob.opposite_prob();
        self
    }
}

impl Default for ParsingConfig {
    fn default() -> Self {
        ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            128,
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

impl<T: Eq + std::fmt::Debug + Clone, B: Scanner<T> + Eq + Clone> ParseHeap<T, B> {
    fn pop(&mut self) -> Option<BeamWrapper<T, B>> {
        self.parse_heap.pop_max()
    }

    fn can_push(&self, v: &BeamWrapper<T, B>) -> bool {
        let is_probable_enough = self
            .config
            .min_log_prob
            .map(|p| v.log_prob() > p)
            .unwrap_or(true);
        let is_short_enough = self
            .config
            .max_steps
            .map(|max_steps| v.n_steps() < max_steps)
            .unwrap_or(true);
        is_short_enough && is_probable_enough
    }

    fn push(&mut self, v: BeamWrapper<T, B>) {
        if self.can_push(&v) {
            if let Some(max_beams) = self.config.max_beams
                && self.parse_heap.len() > max_beams
            {
                self.parse_heap.push_pop_min(v);
            } else {
                self.parse_heap.push(v);
            }
        }
    }

    fn new(start: BeamWrapper<T, B>, config: &ParsingConfig, cat: NodeIndex) -> Self {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(50));
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

type ParserOutput<'a, T> = (LogProb<f64>, &'a [PhonContent<T>], RulePool);
type GeneratorOutput<T> = (LogProb<f64>, Vec<PhonContent<T>>, RulePool);

///An iterator constructed by [`Lexicon::fuzzy_parse`]
pub struct FuzzyParser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>
{
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<T, FuzzyScan<'a, T>>,
    config: &'a ParsingConfig,
}

impl<T, Category> Iterator for FuzzyParser<'_, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug + Hash,
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

///An iterator constructed by [`Lexicon::parse`] and [`Lexicon::parse_multiple`]
pub struct Parser<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: &'a Lexicon<T, Category>,
    parse_heap: ParseHeap<T, ParseScan<'a, T>>,

    #[cfg(not(target_arch = "wasm32"))]
    start_time: Option<Instant>,
    config: &'a ParsingConfig,
    buffer: Vec<ParserOutput<'a, T>>,
}

impl<'a, T, Category> Iterator for Parser<'a, T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    type Item = ParserOutput<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        #[cfg(not(target_arch = "wasm32"))]
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        if self.buffer.is_empty() {
            while let Some(mut beam) = self.parse_heap.pop() {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(max_time) = self.config.max_time {
                    if max_time < self.start_time.unwrap().elapsed() {
                        return None;
                    }
                }

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

impl<T, Category> Lexicon<T, Category>
where
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug,
{
    ///Generates the strings in a grammar that are findable according to the parsing config.
    ///
    ///
    ///Returns an iterator, [`Parser`] which has items of type `(`[`LogProb<f64>`]`, Vec<T>,
    ///`[`RulePool`]`)` where the middle items are the generated strings and the [`RulePool`] has
    ///the structure of each genrerated string.
    pub fn generate(
        &self,
        category: Category,
        config: &ParsingConfig,
    ) -> Result<Generator<&Self, T, Category>, ParsingError<Category>> {
        let cat = self.find_category(&category)?;
        let beam = BeamWrapper::new(GeneratorScan { sentence: vec![] }, cat);
        let parse_heap = ParseHeap::new(beam, config, cat);
        Ok(Generator {
            lexicon: self,
            config: *config,
            parse_heap,
            phantom: PhantomData,
        })
    }

    ///Like [`Lexicon::generate`] but consumes the lexicon.
    pub fn into_generate(
        self,
        category: Category,
        config: &ParsingConfig,
    ) -> Result<Generator<Self, T, Category>, ParsingError<Category>> {
        let cat = self.find_category(&category)?;
        let beam = BeamWrapper::new(GeneratorScan { sentence: vec![] }, cat);
        let parse_heap = ParseHeap::new(beam, config, cat);
        Ok(Generator {
            lexicon: self,
            config: *config,
            parse_heap,
            phantom: PhantomData,
        })
    }

    ///Parses a sentence under the assumption it is has the category, `category`.
    ///
    ///Returns an iterator, [`Parser`] which has items of type `(`[`LogProb<f64>`]`, &'a [T],
    ///`[`RulePool`]`)`
    pub fn parse<'a, 'b: 'a>(
        &'a self,
        s: &'b [PhonContent<T>],
        category: Category,
        config: &'b ParsingConfig,
    ) -> Result<Parser<'a, T, Category>, ParsingError<Category>> {
        let cat = self.find_category(&category)?;

        let beam = BeamWrapper::new(
            ParseScan {
                sentence: vec![(s, 0)],
            },
            cat,
        );
        let parse_heap = ParseHeap::new(beam, config, cat);
        Ok(Parser {
            lexicon: self,
            config,
            #[cfg(not(target_arch = "wasm32"))]
            start_time: None,
            buffer: vec![],
            parse_heap,
        })
    }

    ///Functions like [`Lexicon::parse`] but parsing multiple grammars at once.
    pub fn parse_multiple<'a, 'b: 'a, U>(
        &'a self,
        sentences: &'b [U],
        category: Category,
        config: &'a ParsingConfig,
    ) -> Result<Parser<'a, T, Category>, ParsingError<Category>>
    where
        U: AsRef<[PhonContent<T>]>,
    {
        let cat = self.find_category(&category)?;
        let beams = BeamWrapper::new(
            ParseScan {
                sentence: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
            },
            cat,
        );
        let parse_heap = ParseHeap::new(beams, config, cat);
        Ok(Parser {
            lexicon: self,
            buffer: vec![],
            #[cfg(not(target_arch = "wasm32"))]
            start_time: None,
            config,
            parse_heap,
        })
    }

    ///Functions like [`Lexicon::parse`] or [`Lexicon::generate`] but will return parses that are
    ///close to the provided sentences, but are not necessarily the same.
    pub fn fuzzy_parse<'a, U>(
        &'a self,
        sentences: &'a [U],
        category: Category,
        config: &'a ParsingConfig,
    ) -> Result<FuzzyParser<'a, T, Category>, ParsingError<Category>>
    where
        U: AsRef<[PhonContent<T>]>,
    {
        let cat = self.find_category(&category)?;

        let beams = BeamWrapper::new(
            FuzzyScan {
                sentence_guides: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
                generated_sentences: vec![],
            },
            cat,
        );

        let parse_heap = ParseHeap::new(beams, config, cat);

        Ok(FuzzyParser {
            lexicon: self,
            config,
            parse_heap,
        })
    }
}

#[derive(Debug, Clone)]
///An iterator constructed by [`Lexicon::parse`] and [`Lexicon::parse_multiple`]
pub struct Generator<L, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug> {
    lexicon: L,
    phantom: PhantomData<Category>,
    parse_heap: ParseHeap<T, GeneratorScan<T>>,
    config: ParsingConfig,
}

impl<L, T, Category> Iterator for Generator<L, T, Category>
where
    L: Borrow<Lexicon<T, Category>>,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + Clone + std::fmt::Debug + Hash,
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
pub mod parsing;

#[cfg(test)]
mod tests;
