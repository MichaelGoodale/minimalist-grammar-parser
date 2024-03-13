use burn::tensor::{backend::Backend, Tensor};

use std::collections::hash_map::Entry;

use crate::neural_lexicon::{NeuralLexicon, NeuralProbabilityRecord};

use super::*;

use ahash::HashMap;

#[derive(Debug, Clone, Default)]
pub struct StringPath(HashMap<NeuralProbabilityRecord, u32>);

impl StringPath {
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

    pub fn into_iter(self) -> std::collections::hash_map::IntoIter<NeuralProbabilityRecord, u32> {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct NeuralBeam<'a, B: Backend> {
    log_probability: (NeuralProbabilityRecord, LogProb<f64>),
    lexicon: &'a NeuralLexicon<B>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    generated_sentences: Vec<Tensor<B, 1>>,
    rules: ThinVec<Rule>,
    probability_path: StringPath,
    top_id: usize,
    steps: usize,
    record_rules: bool,
}

impl<'a, B: Backend> NeuralBeam<'a, B>
where
    NeuralLexicon<B>: Lexiconable<(usize, usize), usize>,
{
    pub fn new(
        lexicon: &'a NeuralLexicon<B>,
        initial_category: usize,
        record_rules: bool,
    ) -> Result<NeuralBeam<'a, B>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(&initial_category)?;

        queue.push(Reverse(ParseMoment::new(
            FutureTree {
                node: category_index,
                index: GornIndex::default(),
                id: 0,
            },
            thin_vec![],
        )));

        Ok(NeuralBeam {
            log_probability: (NeuralProbabilityRecord::OneProb, LogProb::new(0.0).unwrap()),
            queue,
            lexicon,
            generated_sentences: vec![],
            rules: if record_rules {
                thin_vec![Rule::Start(category_index)]
            } else {
                thin_vec![]
            },
            probability_path: StringPath::default(),
            top_id: 0,
            steps: 0,
            record_rules,
        })
    }

    pub fn yield_good_parse(self) -> Option<(Tensor<B, 2>, StringPath)> {
        if self.queue.is_empty() && !self.generated_sentences.is_empty() {
            let sentence = self.generated_sentences;
            Some((Tensor::stack(sentence, 0), self.probability_path))
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
        let a = self.log_probability.1;
        let b = other.log_probability.1;
        a.cmp(&b)
    }
}

impl<B: Backend> Beam<(usize, usize)> for NeuralBeam<'_, B> {
    type Probability = (NeuralProbabilityRecord, LogProb<f64>);

    fn log_probability(&self) -> &Self::Probability {
        &self.log_probability
    }

    fn add_to_log_prob(&mut self, x: Self::Probability) {
        let (record, log_prob) = x;
        self.probability_path.add_step(record);
        self.log_probability.1 += log_prob;
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
        v: &mut ParseHeap<(usize, usize), Self>,
        moment: &ParseMoment,
        mut beam: Self,
        s: &Option<(usize, usize)>,
        child_node: NodeIndex,
        child_prob: Self::Probability,
    ) {
        beam.queue.shrink_to_fit();
        if let Some((lex, pos)) = s {
            beam.generated_sentences
                .push(beam.lexicon.lemma_at_position(*lex, *pos));
        }

        let (record, log_prob) = child_prob;
        beam.probability_path.add_step(record);
        beam.log_probability.1 = beam.log_probability.1 + log_prob;

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
        self.n_steps() < config.max_steps
    }
}
