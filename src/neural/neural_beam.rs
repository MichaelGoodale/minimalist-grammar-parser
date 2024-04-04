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

#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub enum NodeFeature {
    Node(NodeIndex),
    NLicensees {
        lexeme_idx: usize,
        n_licensees: usize,
    },
    NFeatures {
        lexeme_idx: usize,
        n_features: usize,
    },
}

#[derive(Debug, Clone, Default)]
pub struct StringProbHistory(
    BTreeMap<NeuralProbabilityRecord, u32>,
    BTreeSet<NodeFeature>,
);

impl StringProbHistory {
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, NeuralProbabilityRecord, u32> {
        self.0.iter()
    }

    pub fn keys(&self) -> std::collections::btree_map::Keys<'_, NeuralProbabilityRecord, u32> {
        self.0.keys()
    }

    pub fn attested_nodes(&self) -> &BTreeSet<NodeFeature> {
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

#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord)]
enum NFeatures {
    UnInitialized,
    Building(usize),
    Done(usize),
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
    n_licensees: Vec<NFeatures>,
    p_n_licensees: Vec<Option<LogProb<f64>>>,
    n_features: Vec<NFeatures>,
    p_n_features: Vec<Option<LogProb<f64>>>,
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
            history.1.insert(NodeFeature::NLicensees {
                lexeme_idx: lex,
                n_licensees: 0,
            });
            history.1.insert(NodeFeature::Node(*node));
            let n_licensees = (0..lexicon.n_lexemes())
                .map(|i| {
                    if i == lex {
                        NFeatures::Done(0)
                    } else {
                        NFeatures::UnInitialized
                    }
                })
                .collect();
            let p_n_licensees = (0..lexicon.n_lexemes())
                .map(|i| {
                    if i == lex {
                        Some(lexicon.prob_of_n_licensees(i, 0))
                    } else {
                        None
                    }
                })
                .collect();
            let n_features = std::iter::repeat(NFeatures::UnInitialized)
                .take(lexicon.n_lexemes())
                .collect();
            let p_n_features = std::iter::repeat(None).take(lexicon.n_lexemes()).collect();

            let log_one = log_probability.2;

            NeuralBeam {
                max_log_prob: log_probability.2,
                log_probability: *log_probability,
                n_licensees,
                p_n_licensees,
                n_features,
                p_n_features,
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
        let feature_p: LogProb<f64> = self
            .p_n_features
            .iter()
            .chain(self.p_n_licensees.iter())
            .filter_map(|x| *x)
            .fold(LogProb::new(0.0).unwrap(), |x, acc| x + acc);
        let other_feature_p: LogProb<f64> = other
            .p_n_features
            .iter()
            .chain(other.p_n_licensees.iter())
            .filter_map(|x| *x)
            .fold(LogProb::new(0.0).unwrap(), |x, acc| x + acc);
        (self.max_log_prob + feature_p).cmp(&(other.max_log_prob + other_feature_p))
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
                    self.burnt |= match self.n_licensees[lexeme_idx] {
                        NFeatures::UnInitialized | NFeatures::Building(_) => {
                            let p = self
                                .lexicon
                                .prob_of_at_least_n_licensees(lexeme_idx, n_licensees);
                            self.p_n_licensees[lexeme_idx] = Some(p);
                            if let NFeatures::Building(x) = self.n_licensees[lexeme_idx] {
                                if x < n_licensees {
                                    self.n_licensees[lexeme_idx] = NFeatures::Building(n_licensees);
                                }
                            } else {
                                self.n_licensees[lexeme_idx] = NFeatures::Building(n_licensees);
                            }
                            false
                        }
                        NFeatures::Done(x) => x <= n_licensees,
                    }
                }
                EdgeHistory::AtLeastNCategories {
                    lexeme_idx,
                    n_categories,
                } => {
                    self.burnt |= match self.n_features[lexeme_idx] {
                        NFeatures::UnInitialized | NFeatures::Building(_) => {
                            let p = self
                                .lexicon
                                .prob_of_at_least_n_features(lexeme_idx, n_categories);
                            self.p_n_features[lexeme_idx] = Some(p);
                            if let NFeatures::Building(x) = self.n_features[lexeme_idx] {
                                if x < n_categories {
                                    self.n_features[lexeme_idx] = NFeatures::Building(n_categories);
                                }
                            } else {
                                self.n_features[lexeme_idx] = NFeatures::Building(n_categories);
                            }
                            false
                        }
                        NFeatures::Done(x) => x <= n_categories,
                    }
                }
                EdgeHistory::AtMostNLicenseses {
                    lexeme_idx,
                    n_licensees,
                } => {
                    self.burnt |= match self.n_licensees[lexeme_idx] {
                        NFeatures::UnInitialized => {
                            self.n_licensees[lexeme_idx] = NFeatures::Done(n_licensees);
                            self.p_n_licensees[lexeme_idx] =
                                Some(self.lexicon.prob_of_n_licensees(lexeme_idx, n_licensees));
                            self.probability_path.1.insert(NodeFeature::NLicensees {
                                lexeme_idx,
                                n_licensees,
                            });
                            false
                        }
                        NFeatures::Building(x) => {
                            if x > n_licensees {
                                true
                            } else {
                                self.n_licensees[lexeme_idx] = NFeatures::Done(n_licensees);
                                self.p_n_licensees[lexeme_idx] =
                                    Some(self.lexicon.prob_of_n_licensees(lexeme_idx, n_licensees));
                                self.probability_path.1.insert(NodeFeature::NLicensees {
                                    lexeme_idx,
                                    n_licensees,
                                });
                                false
                            }
                        }
                        NFeatures::Done(x) => x != n_licensees,
                    }
                }
                EdgeHistory::AtMostNCategories {
                    lexeme_idx,
                    n_categories,
                } => {
                    self.burnt |= match self.n_features[lexeme_idx] {
                        NFeatures::UnInitialized => {
                            self.n_features[lexeme_idx] = NFeatures::Done(n_categories);
                            self.p_n_features[lexeme_idx] =
                                Some(self.lexicon.prob_of_n_features(lexeme_idx, n_categories));
                            self.probability_path.1.insert(NodeFeature::NFeatures {
                                lexeme_idx,
                                n_features: n_categories,
                            });
                            false
                        }
                        NFeatures::Building(x) => {
                            if x > n_categories {
                                true
                            } else {
                                self.n_features[lexeme_idx] = NFeatures::Done(n_categories);
                                self.p_n_features[lexeme_idx] =
                                    Some(self.lexicon.prob_of_n_features(lexeme_idx, n_categories));
                                self.probability_path.1.insert(NodeFeature::NFeatures {
                                    lexeme_idx,
                                    n_features: n_categories,
                                });
                                false
                            }
                        }
                        NFeatures::Done(x) => x != n_categories,
                    }
                }
            }
        };
        match record {
            NeuralProbabilityRecord::Node(n) | NeuralProbabilityRecord::Lexeme(n, _) => {
                let nf = NodeFeature::Node(n);
                if !self.probability_path.1.contains(&nf) {
                    self.log_probability.2 += new_prob;
                    self.max_log_prob += new_prob;
                }
                self.burnt |= self
                    .alternatives
                    .get(&n)
                    .unwrap()
                    .iter()
                    .any(|x| self.probability_path.1.contains(&NodeFeature::Node(*x)));
                self.probability_path.1.insert(nf);
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
