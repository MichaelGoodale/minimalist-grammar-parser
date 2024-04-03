use allocator_api2::alloc::Allocator;
use burn::tensor::backend::Backend;
use logprob::LogProb;

use std::{
    cmp::Reverse,
    collections::{binary_heap::BinaryHeap, btree_map::Entry, BTreeMap, BTreeSet},
};

use crate::lexicon::Lexiconable;
use crate::parsing::{beam::Beam, FutureTree, GornIndex, ParseMoment, Rule};
use crate::{ParseHeap, ParsingConfig};
use anyhow::Result;
use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::neural::neural_lexicon::{NeuralLexicon, NeuralProbabilityRecord};

use thin_vec::{thin_vec, ThinVec};

use ahash::HashMap;

use super::neural_lexicon::NeuralProbability;

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
pub struct StringProbHistory(BTreeMap<NeuralProbabilityRecord, u32>, BTreeSet<EdgeIndex>);

impl StringProbHistory {
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, NeuralProbabilityRecord, u32> {
        self.0.iter()
    }

    pub fn keys(&self) -> std::collections::btree_map::Keys<'_, NeuralProbabilityRecord, u32> {
        self.0.keys()
    }

    pub fn attested_nodes(&self) -> &BTreeSet<EdgeIndex> {
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
pub struct NeuralBeam<'a> {
    log_probability: NeuralProbability,
    max_log_prob: LogProb<f64>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    generated_sentence: StringPath,
    sentence_guides: Vec<(&'a [usize], usize, LogProb<f64>)>,
    lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
    weight_lookups: &'a HashMap<usize, LogProb<f64>>,
    alternatives: &'a HashMap<EdgeIndex, Vec<EdgeIndex>>,
    burnt: bool,
    rules: ThinVec<Rule>,
    probability_path: StringProbHistory,
    top_id: usize,
    steps: usize,
    record_rules: bool,
}

impl<'a> NeuralBeam<'a> {
    pub fn new<B: Backend, T>(
        lexicon: &'a NeuralLexicon<B>,
        initial_category: usize,
        sentences: Option<&'a [T]>,
        lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
        weight_lookups: &'a HashMap<usize, LogProb<f64>>,
        alternatives: &'a HashMap<EdgeIndex, Vec<EdgeIndex>>,
        record_rules: bool,
    ) -> Result<impl Iterator<Item = NeuralBeam<'a>> + 'a>
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

            let record = log_probability.0 .0;
            let mut history = StringProbHistory::default();
            history.0.insert(record, 1);
            let log_one = log_probability.0 .1;

            NeuralBeam {
                max_log_prob: log_probability.0 .1,
                log_probability: *log_probability,
                queue,
                burnt: false,
                generated_sentence: StringPath(vec![]),
                sentence_guides: sentences
                    .map(|x| x.iter().map(|x| (x.as_ref(), 0, log_one)).collect())
                    .unwrap_or_default(),
                lemma_lookups,
                alternatives,
                weight_lookups,
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

impl PartialEq for NeuralBeam<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.steps == other.steps
            && self.top_id == other.top_id
            && self.rules == other.rules
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl PartialOrd for NeuralBeam<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for NeuralBeam<'_> {}

impl Ord for NeuralBeam<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.max_log_prob.cmp(&other.max_log_prob)
    }
}

impl Beam<usize> for NeuralBeam<'_> {
    type Probability = NeuralProbability;

    fn log_probability(&self) -> &Self::Probability {
        &self.log_probability
    }

    fn add_to_log_prob(&mut self, x: Self::Probability) {
        let NeuralProbability((record, new_prob)) = x;
        match record {
            NeuralProbabilityRecord::Edge(e) | NeuralProbabilityRecord::Lexeme(e, _) => {
                if !self.probability_path.1.contains(&e) {
                    self.log_probability.0 .1 += new_prob;
                    self.max_log_prob += new_prob;
                }
                self.burnt = self
                    .alternatives
                    .get(&e)
                    .unwrap()
                    .iter()
                    .any(|x| self.probability_path.1.contains(x));
                self.probability_path.1.insert(e);
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
                            self.log_probability.0 .1 += new_prob;
                            self.max_log_prob += new_prob;
                        }
                        NeuralProbabilityRecord::Lexeme(_, lexeme_idx) => {
                            let w = self.weight_lookups[&lexeme_idx];
                            self.log_probability.0 .1 += w;
                            self.max_log_prob += w;
                        }
                        _ => (),
                    }
                    *entry.into_mut() += 1;
                }
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    self.log_probability.0 .1 += new_prob;
                    self.max_log_prob += new_prob;
                }
            },
            NeuralProbabilityRecord::Edge(_) => (),
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

            beam.max_log_prob = beam.log_probability.0 .1
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
            && self.log_probability.0 .1
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
