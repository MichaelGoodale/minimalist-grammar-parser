use crate::parsing::ParseHolder;
use burn::tensor::backend::Backend;
use logprob::LogProb;

use std::{
    cmp::Reverse,
    collections::{binary_heap::BinaryHeap, btree_map::Entry, BTreeMap, BTreeSet},
};

use crate::lexicon::Lexiconable;
use crate::parsing::{beam::Beam, FutureTree, GornIndex, ParseMoment, Rule};
use crate::ParsingConfig;
use anyhow::Result;
use petgraph::graph::NodeIndex;

use crate::neural::neural_lexicon::{NeuralLexicon, NeuralProbabilityRecord};

use thin_vec::thin_vec;

use super::neural_lexicon::{EdgeHistory, NeuralProbability};
use super::parameterization::GrammarParameterization;

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
    NFeats {
        node: NodeIndex,
        lexeme_idx: usize,
        n_licensees: usize,
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

    pub(crate) fn add_node(&mut self, n: NodeFeature) {
        self.1.insert(n);
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
    max_log_prob: LogProb<f64>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    generated_sentence: StringPath,
    sentence_guides: Vec<(&'a [usize], LogProb<f64>)>,
    lexicon: &'a NeuralLexicon<B>,
    g: &'a GrammarParameterization<B>,
    n_features: Vec<Option<(usize, usize)>>,
    burnt: bool,
    probability_path: StringProbHistory,
    rules: Vec<Rule>,
    top_id: usize,
    steps: usize,
    max_string_len: usize,
    n_empties: usize,
}

impl<'a, B: Backend> NeuralBeam<'a, B> {
    pub fn new<T>(
        lexicon: &'a NeuralLexicon<B>,
        g: &'a GrammarParameterization<B>,
        initial_category: usize,
        sentences: Option<&'a [T]>,
        max_string_len: usize,
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
            let (lex, n_features) = if let NeuralProbabilityRecord::Lexeme {
                node: _,
                id: lex,
                n_features,
                n_licensees,
            } = record
            {
                history.1.insert(NodeFeature::NFeats {
                    node: *node,
                    lexeme_idx: lex,
                    n_licensees,
                    n_features,
                });
                (lex, n_features)
            } else {
                panic!()
            };
            history.1.insert(NodeFeature::Node(*node));
            let n_features = (0..lexicon.n_lexemes())
                .map(|i| {
                    if i == lex {
                        Some((n_features, 0))
                    } else {
                        None
                    }
                })
                .collect();

            let log_one = LogProb::new(0.0).unwrap();

            NeuralBeam {
                lexicon,
                max_log_prob: log_probability.2,
                log_probability: *log_probability,
                n_features,
                g,
                queue,
                rules: vec![Rule::Start(*node)],
                burnt: false,
                generated_sentence: StringPath(vec![]),
                sentence_guides: sentences
                    .map(|x| x.iter().map(|x| (x.as_ref(), log_one)).collect())
                    .unwrap_or_default(),
                probability_path: history,
                top_id: 0,
                steps: 0,
                max_string_len,
                n_empties: 0,
            }
        }))
    }

    pub fn into_completed_parse(self) -> (StringPath, StringProbHistory, bool) {
        (
            self.generated_sentence,
            self.probability_path,
            self.queue.is_empty(),
        )
    }

    pub fn burnt(&self) -> bool {
        self.burnt
    }

    pub fn yield_good_parse(self) -> Option<(StringPath, StringProbHistory)> {
        if self.queue.is_empty() {
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

impl<B: Backend> NeuralBeam<'_, B> {
    pub fn latest_rule(&self) -> &Rule {
        self.rules
            .last()
            .expect("A neural beam should never be made which doesn't have at least one rule")
    }
}

impl<B: Backend> Beam<usize> for NeuralBeam<'_, B> {
    type Probability = NeuralProbability;

    fn log_prob(&self) -> LogProb<f64> {
        self.log_probability.2
    }

    fn probability(&self) -> &Self::Probability {
        &self.log_probability
    }

    fn add_to_log_prob(&mut self, x: Self::Probability) {
        let NeuralProbability(record, edge_history, new_prob) = x;

        if let NeuralProbabilityRecord::Lexeme {
            node,
            id,
            n_features,
            n_licensees,
        } = record
        {
            match self.n_features[id] {
                None => {
                    self.n_features[id] = Some((n_features, n_licensees));
                    self.probability_path.1.insert(NodeFeature::NFeats {
                        node,
                        lexeme_idx: id,
                        n_licensees,
                        n_features,
                    });
                    self.probability_path.1.insert(NodeFeature::Node(node));
                }
                Some((x, y)) => {
                    self.burnt |= (x != n_features)
                        || (y != n_licensees)
                        || !self.probability_path.1.contains(&NodeFeature::Node(node));
                }
            }
        };
        match record {
            NeuralProbabilityRecord::Lexeme { .. }
            | NeuralProbabilityRecord::OneProb
            | NeuralProbabilityRecord::MoveRuleProb
            | NeuralProbabilityRecord::MergeRuleProb => match self.probability_path.0.entry(record)
            {
                Entry::Occupied(entry) => {
                    /* LogProb solely means the prob of grammar + strings given grammar, not the
                    * prob of string.
                    match entry.key() {
                        NeuralProbabilityRecord::MergeRuleProb
                        | NeuralProbabilityRecord::MoveRuleProb => {
                            self.log_probability.2 += new_prob;
                            self.max_log_prob += new_prob;
                        }
                        NeuralProbabilityRecord::Lexeme { .. } => {
                            let w = LogProb::new(-(self.lexicon.n_lexemes() as f64).ln()).unwrap();
                            self.log_probability.2 += w;
                            self.max_log_prob += w;
                        }
                        _ => (),
                    }*/
                    *entry.into_mut() += 1;
                }
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    self.log_probability.2 += new_prob;
                    self.max_log_prob += new_prob;
                }
            },
            NeuralProbabilityRecord::Node(n) => {
                let nf = NodeFeature::Node(n);
                if !self.probability_path.1.contains(&nf) {
                    self.log_probability.2 += new_prob;
                    self.max_log_prob += new_prob;
                }
                self.burnt |= self.lexicon.has_alternative(&n, &self.probability_path.1);
                self.probability_path.1.insert(nf);
            }
        }

        if let Some(edge_history) = edge_history {
            match edge_history {
                EdgeHistory::AtLeastNLicenseses {
                    lexeme_idx,
                    n_licensees,
                } => {
                    let max_n_licensee = self.n_features[lexeme_idx].unwrap().1;
                    self.burnt |= max_n_licensee < n_licensees;
                }
                EdgeHistory::AtLeastNCategories {
                    lexeme_idx,
                    n_categories,
                } => {
                    let max_n_features = self.n_features[lexeme_idx].unwrap().0;
                    self.burnt |= max_n_features < n_categories;
                }
                EdgeHistory::AtMostNLicenseses {
                    lexeme_idx,
                    n_licensees,
                } => {
                    let max_n_licensees = self.n_features[lexeme_idx].unwrap().1;
                    self.burnt |= max_n_licensees != n_licensees;
                }
                EdgeHistory::AtMostNCategories {
                    lexeme_idx,
                    n_categories,
                } => {
                    let max_n_features = self.n_features[lexeme_idx].unwrap().0;
                    self.burnt |= max_n_features != n_categories;
                }
            }
        };
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

    fn push_rule(&mut self, r: Rule) {
        self.rules.push(r);
    }

    fn record_rules(&self) -> bool {
        true
    }

    fn scan<H: ParseHolder<usize, Self>>(
        v: &mut H,
        moment: &ParseMoment,
        mut beam: Self,
        s: &Option<usize>,
        child_node: NodeIndex,
        child_prob: Self::Probability,
    ) {
        beam.queue.shrink_to_fit();
        if let Some(x) = s {
            let position = beam.generated_sentence.0.len();
            beam.sentence_guides
                .iter_mut()
                .for_each(|(sentence, mut prob)| {
                    let lemma: usize = *sentence.get(position).unwrap_or(&0);
                    prob += match lemma {
                        0 => LogProb::new(-1000.0).unwrap(),
                        _ => *beam.g.lemma_lookups().get(&(*x, lemma)).unwrap(),
                    }
                });
            beam.generated_sentence.0.push(*x);

            beam.max_log_prob = beam.log_probability.2
                + beam
                    .sentence_guides
                    .iter()
                    .map(|(_, p)| *p)
                    .max()
                    .unwrap_or_else(|| LogProb::new(0.0).unwrap());
        } else {
            beam.n_empties += 1;
        }

        beam.add_to_log_prob(child_prob);
        beam.steps += 1;
        beam.push_rule(Rule::Scan {
            node: child_node,
            parent: moment.tree.id,
        });
        v.add(beam);
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
    fn pushable(&self, _config: &ParsingConfig) -> bool {
        !self.burnt
            && self.generated_sentence.len() < self.max_string_len
            && self.n_empties < self.max_string_len * 2
    }
}
