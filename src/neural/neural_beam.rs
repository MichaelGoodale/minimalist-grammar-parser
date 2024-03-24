use burn::tensor::backend::Backend;
use logprob::LogProb;

use std::{
    cmp::Reverse,
    collections::{binary_heap::BinaryHeap, hash_map::Entry},
};

use crate::lexicon::Lexiconable;
use crate::parsing::{beam::Beam, FutureTree, GornIndex, ParseMoment, Rule};
use crate::{ParseHeap, ParsingConfig};
use anyhow::Result;
use petgraph::graph::NodeIndex;

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

    pub fn into_iter(self) -> std::vec::IntoIter<usize> {
        self.0.into_iter()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }
}

#[derive(Debug, Clone, Default)]
pub struct StringProbHistory(HashMap<NeuralProbabilityRecord, u32>);

impl StringProbHistory {
    fn add_step(&mut self, x: NeuralProbabilityRecord) {
        match self.0.entry(x) {
            Entry::Occupied(entry) => {
                *entry.into_mut() += 1;
            }
            Entry::Vacant(entry) => {
                entry.insert(1);
            }
        };
    }
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, NeuralProbabilityRecord, u32> {
        self.0.iter()
    }

    pub fn into_iter(self) -> std::collections::hash_map::IntoIter<NeuralProbabilityRecord, u32> {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct NeuralBeam<'a> {
    log_probability: NeuralProbability,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    generated_sentence: StringPath,
    sentence_guides: Vec<(&'a [usize], usize, LogProb<f64>)>,
    lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
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
        sentences: &'a [T],
        lemma_lookups: &'a HashMap<(usize, usize), LogProb<f64>>,
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
            let log_one = LogProb::new(0.0).unwrap();

            NeuralBeam {
                log_probability: *log_probability,
                queue,
                generated_sentence: StringPath(vec![]),
                sentence_guides: sentences.iter().map(|x| (x.as_ref(), 0, log_one)).collect(),
                lemma_lookups,
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
        let a: LogProb<f64> = self.log_probability.0 .1
            + self
                .sentence_guides
                .iter()
                .map(|(_, _, p)| *p)
                .max()
                .unwrap_or_else(|| LogProb::new(0.0).unwrap());
        let b = other.log_probability.0 .1
            + other
                .sentence_guides
                .iter()
                .map(|(_, _, p)| *p)
                .max()
                .unwrap_or_else(|| LogProb::new(0.0).unwrap());
        a.cmp(&b)
    }
}

impl Beam<usize> for NeuralBeam<'_> {
    type Probability = NeuralProbability;

    fn log_probability(&self) -> &Self::Probability {
        &self.log_probability
    }

    fn add_to_log_prob(&mut self, x: Self::Probability) {
        let NeuralProbability((record, log_prob)) = x;
        self.probability_path.add_step(record);
        self.log_probability.0 .1 += log_prob;
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

    fn scan(
        v: &mut ParseHeap<usize, Self>,
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
                    prob += beam.lemma_lookups.get(&(*x, lemma)).unwrap();
                });
        }

        let NeuralProbability((record, log_prob)) = child_prob;
        beam.probability_path.add_step(record);
        beam.log_probability.0 .1 += log_prob;

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
        self.log_probability.0 .1 > config.min_log_prob && self.n_steps() < config.max_steps
    }
}
