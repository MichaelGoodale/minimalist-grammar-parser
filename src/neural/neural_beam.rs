use allocator_api2::{alloc::Allocator, SliceExt};
use burn::tensor::backend::Backend;
use itertools::Itertools;
use logprob::LogProb;
use rand::{rngs::StdRng, Rng, SeedableRng};

use std::{
    cmp::Reverse,
    collections::{binary_heap::BinaryHeap, btree_map::Entry, BTreeMap, BTreeSet},
};

use crate::lexicon::Lexiconable;
use crate::parsing::{beam::Beam, FutureTree, GornIndex, ParseMoment, Rule};
use crate::{ParseHeap, ParsingConfig};
use anyhow::Result;
use petgraph::graph::NodeIndex;

use crate::neural::neural_lexicon::{NeuralLexicon, NeuralProbabilityRecord};

use thin_vec::{thin_vec, ThinVec};

use ahash::HashMap;

use super::neural_lexicon::{EdgeHistory, NeuralProbability};

#[derive(Debug, Clone, Default)]
pub struct StringPath(Vec<usize>);

impl StringPath {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }
}

impl IntoIterator for StringPath {
    type Item = usize;

    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone, Default)]
pub struct StringProbHistory(BTreeMap<NeuralProbabilityRecord, u32>, BTreeSet<NodeIndex>);

impl StringProbHistory {
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, NeuralProbabilityRecord, u32> {
        self.0.iter()
    }

    pub fn keys(&self) -> std::collections::btree_map::Keys<'_, NeuralProbabilityRecord, u32> {
        self.0.keys()
    }

    pub fn attested_nodes(&self) -> &BTreeSet<NodeIndex> {
        &self.1
    }
}

impl IntoIterator for StringProbHistory {
    type Item = (NeuralProbabilityRecord, u32);

    type IntoIter = std::collections::btree_map::IntoIter<NeuralProbabilityRecord, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct NeuralBeam<'a, B: Backend> {
    log_probability: NeuralProbability,
    lexicon: &'a NeuralLexicon<B>,
    max_log_prob: LogProb<f64>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    generated_sentence: StringPath,
    sentence_guides: Vec<(&'a [usize], usize, LogProb<f64>)>,
    lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
    weight_lookups: &'a HashMap<usize, LogProb<f64>>,
    alternatives: &'a HashMap<NodeIndex, Vec<NodeIndex>>,
    n_licensees: Vec<Option<usize>>,
    n_features: Vec<Option<usize>>,
    burnt: bool,
    rules: ThinVec<Rule>,
    probability_path: StringProbHistory,
    top_id: usize,
    steps: usize,
    record_rules: bool,
}

impl<'a, B: Backend> NeuralBeam<'a, B> {
    pub fn new<T>(
        lexicon: &'a NeuralLexicon<B>,
        initial_category: usize,
        sentences: Option<&'a [T]>,
        lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
        weight_lookups: &'a HashMap<usize, LogProb<f64>>,
        alternatives: &'a HashMap<NodeIndex, Vec<NodeIndex>>,
        record_rules: bool,
    ) -> Result<impl Iterator<Item = NeuralBeam<'a, B>> + 'a>
    where
        T: AsRef<[usize]>,
    {
        let category_indexes = lexicon.find_category(&initial_category)?;
        Ok(category_indexes.iter().map(move |(log_probability, node)| {
            let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
            queue.push(Reverse(ParseMoment::new(
                FutureTree {
                    node: *node,
                    index: GornIndex::default(),
                    id: 0,
                },
                thin_vec![],
            )));

            let record = log_probability.0;
            let mut history = StringProbHistory::default();
            history.0.insert(record, 1);
            let lex = if let NeuralProbabilityRecord::Lexeme(_, lex) = record {
                lex
            } else {
                panic!()
            };
            let n_licensees = (0..lexicon.n_lemmas())
                .map(|i| if i == lex { Some(0) } else { None })
                .collect();
            let n_features = std::iter::repeat(None).take(lexicon.n_lemmas()).collect();

            let log_one = log_probability.2;

            NeuralBeam {
                max_log_prob: log_probability.2,
                log_probability: *log_probability,
                n_licensees,
                n_features,
                queue,
                burnt: false,
                generated_sentence: StringPath(vec![]),
                sentence_guides: sentences
                    .map(|x| x.iter().map(|x| (x.as_ref(), 0, log_one)).collect())
                    .unwrap_or_default(),
                lemma_lookups,
                alternatives,
                weight_lookups,
                lexicon,
                rules: if record_rules {
                    thin_vec![Rule::Start(*node)]
                } else {
                    thin_vec![]
                },
                probability_path: history,
                top_id: 0,
                steps: 0,
                record_rules,
            }
        }))
    }

    pub fn yield_good_parse(self) -> Option<(StringPath, StringProbHistory)> {
        if self.queue.is_empty() && !self.generated_sentence.0.is_empty() {
            Some((self.generated_sentence, self.probability_path))
        } else {
            None
        }
    }
}

impl<B: Backend> PartialEq for NeuralBeam<'_, B> {
    fn eq(&self, other: &Self) -> bool {
        self.steps == other.steps
            && self.top_id == other.top_id
            && self.rules == other.rules
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<B: Backend> PartialOrd for NeuralBeam<'_, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: Backend> Eq for NeuralBeam<'_, B> {}

impl<B: Backend> Ord for NeuralBeam<'_, B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.max_log_prob.cmp(&other.max_log_prob)
    }
}

impl<B: Backend> Beam<usize> for NeuralBeam<'_, B> {
    type Probability = NeuralProbability;

    fn log_probability(&self) -> &Self::Probability {
        &self.log_probability
    }

    fn add_to_log_prob(&mut self, x: Self::Probability) {
        let NeuralProbability(record, edge_history, new_prob) = x;
        if let Some(edge_history) = edge_history {
            match edge_history {
                EdgeHistory::AtLeastNLicenseses {
                    lexeme_idx,
                    n_licensees,
                } => {
                    self.burnt &= if let Some(x) = self.n_licensees[lexeme_idx] {
                        x >= n_licensees
                    } else {
                        false
                    }
                }
                EdgeHistory::AtLeastNCategories {
                    lexeme_idx,
                    n_categories,
                } => {
                    self.burnt &= if let Some(x) = self.n_features[lexeme_idx] {
                        x >= n_categories
                    } else {
                        false
                    }
                }
                EdgeHistory::AtMostNLicenseses {
                    lexeme_idx,
                    n_licensees,
                } => {
                    self.burnt &= if let Some(x) = self.n_licensees[lexeme_idx] {
                        x == n_licensees
                    } else {
                        self.n_licensees[lexeme_idx] = Some(n_licensees);
                        false
                    }
                }
                EdgeHistory::AtMostNCategories {
                    lexeme_idx,
                    n_categories,
                } => {
                    self.burnt &= if let Some(x) = self.n_features[lexeme_idx] {
                        x == n_categories
                    } else {
                        self.n_features[lexeme_idx] = Some(n_categories);
                        false
                    }
                }
            }
        }
        match record {
            NeuralProbabilityRecord::Node(n) | NeuralProbabilityRecord::Lexeme(n, _) => {
                if !self.probability_path.1.contains(&n) {
                    self.log_probability.2 += new_prob;
                    self.max_log_prob += new_prob;
                }
                self.burnt &= self
                    .alternatives
                    .get(&n)
                    .unwrap()
                    .iter()
                    .any(|x| self.probability_path.1.contains(x));
                self.probability_path.1.insert(n);
            }
            _ => (),
        }
        match record {
            NeuralProbabilityRecord::Lexeme(_, _)
            | NeuralProbabilityRecord::OneProb
            | NeuralProbabilityRecord::MoveRuleProb
            | NeuralProbabilityRecord::MergeRuleProb => match self.probability_path.0.entry(record)
            {
                Entry::Occupied(entry) => {
                    match entry.key() {
                        NeuralProbabilityRecord::MergeRuleProb
                        | NeuralProbabilityRecord::MoveRuleProb => {
                            self.log_probability.2 += new_prob;
                            self.max_log_prob += new_prob;
                        }
                        NeuralProbabilityRecord::Lexeme(_, lexeme_idx) => {
                            let w = self.weight_lookups[&lexeme_idx];
                            self.log_probability.2 += w;
                            self.max_log_prob += w;
                        }
                        _ => (),
                    }
                    *entry.into_mut() += 1;
                }
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    self.log_probability.2 += new_prob;
                    self.max_log_prob += new_prob;
                }
            },
            NeuralProbabilityRecord::Node(_) => (),
        }
    }

    fn pop_moment(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    fn push_moment(&mut self, x: ParseMoment) {
        self.queue.push(Reverse(x))
    }

    fn push_rule(&mut self, x: Rule) {
        self.rules.push(x)
    }

    fn record_rules(&self) -> bool {
        self.record_rules
    }

    fn scan<A: Allocator>(
        v: &mut ParseHeap<usize, Self, A>,
        moment: &ParseMoment,
        mut beam: Self,
        s: &Option<usize>,
        child_node: NodeIndex,
        child_prob: Self::Probability,
    ) {
        beam.queue.shrink_to_fit();
        if let Some(x) = s {
            beam.generated_sentence.0.push(*x);
            beam.sentence_guides
                .iter_mut()
                .for_each(|(sentence, position, mut prob)| {
                    let lemma: usize = *sentence.get(*position).unwrap_or(&0);
                    prob += match lemma {
                        0 => LogProb::new(-1000.0).unwrap(),
                        _ => *beam.lemma_lookups.get(&(*x, lemma)).unwrap(),
                    }
                });

            beam.max_log_prob = beam.log_probability.2
                + beam
                    .sentence_guides
                    .iter()
                    .map(|(_, _, p)| *p)
                    .max()
                    .unwrap_or_else(|| LogProb::new(0.0).unwrap());
        }

        beam.add_to_log_prob(child_prob);

        if beam.record_rules() {
            beam.rules.push(Rule::Scan {
                node: child_node,
                parent: moment.tree.id,
            });
        }
        beam.steps += 1;
        v.push(beam);
    }

    fn inc(&mut self) {
        self.steps += 1;
    }

    fn n_steps(&self) -> usize {
        self.steps
    }

    fn top_id(&self) -> usize {
        self.top_id
    }

    fn top_id_mut(&mut self) -> &mut usize {
        &mut self.top_id
    }
    fn pushable(&self, config: &ParsingConfig) -> bool {
        !self.burnt
            && self.log_probability.2
                + self
                    .sentence_guides
                    .iter()
                    .map(|(_, _, p)| *p)
                    .max()
                    .unwrap_or_else(|| LogProb::new(0.0).unwrap())
                > config.min_log_prob
            && self.n_steps() < config.max_steps
            && if let Some(max_length) = config.max_length {
                self.generated_sentence.len() <= max_length
            } else {
                true
            }
    }
}
