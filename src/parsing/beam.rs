use super::trees::{FutureTree, GornIndex, ParseMoment};
use super::Rule;
use crate::lexicon::Lexicon;
use anyhow::Result;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use thin_vec::thin_vec;

pub trait Beam<T>: Sized {
    fn log_probability(&self) -> &LogProb<f64>;

    fn log_probability_mut(&mut self) -> &mut LogProb<f64>;

    fn pop_moment(&mut self) -> Option<ParseMoment>;

    fn push_moment(&mut self, x: ParseMoment);

    fn push_rule(&mut self, x: Rule);

    fn scan(
        v: &mut Vec<Self>,
        moment: &ParseMoment,
        beam: Self,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    );

    fn inc(&mut self);

    fn top_id(&self) -> usize;

    fn top_id_mut(&mut self) -> &mut usize;
}

#[derive(Debug, Clone)]
pub struct ParseBeam<'a, T> {
    pub log_probability: LogProb<f64>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    pub sentence: &'a [T],
    pub sentence_position: usize,
    pub rules: Vec<Rule>,
    pub top_id: usize,
    pub steps: usize,
}

impl<T: Eq + std::fmt::Debug> PartialEq for ParseBeam<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.log_probability == other.log_probability
            && self.sentence == other.sentence
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug> PartialOrd for ParseBeam<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug> Eq for ParseBeam<'_, T> {}

impl<T: Eq + std::fmt::Debug> Ord for ParseBeam<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_probability
            .partial_cmp(&other.log_probability)
            .unwrap()
    }
}

impl<T> Beam<T> for ParseBeam<'_, T>
where
    T: std::cmp::PartialEq,
{
    fn log_probability(&self) -> &LogProb<f64> {
        &self.log_probability
    }

    fn log_probability_mut(&mut self) -> &mut LogProb<f64> {
        &mut self.log_probability
    }

    fn push_moment(&mut self, x: ParseMoment) {
        self.queue.push(Reverse(x))
    }

    fn push_rule(&mut self, x: Rule) {
        self.rules.push(x)
    }

    fn inc(&mut self) {
        self.steps += 1;
    }

    fn top_id(&self) -> usize {
        self.top_id
    }

    fn top_id_mut(&mut self) -> &mut usize {
        &mut self.top_id
    }

    fn scan(
        v: &mut Vec<Self>,
        moment: &ParseMoment,
        mut beam: Self,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        let should_push = match s {
            Some(s) if *s == beam.sentence[beam.sentence_position] => {
                beam.sentence_position += 1;
                true
            }
            None => true,
            _ => false,
        };
        if should_push {
            beam.log_probability += child_prob;
            beam.rules.push(Rule::Scan {
                node: child_node,
                parent: moment.tree.id,
            });
            beam.steps += 1;
            v.push(beam);
        };
    }

    fn pop_moment(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }
}

impl<'a, T: Eq + std::fmt::Debug + Clone> ParseBeam<'_, T> {
    pub fn new<Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
    ) -> Result<ParseBeam<'a, T>> {
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

        Ok(ParseBeam {
            log_probability: LogProb::new(0_f64).unwrap(),
            sentence_position: 0,
            sentence,
            queue,
            rules: vec![Rule::Start(category_index)],
            top_id: 0,
            steps: 0,
        })
    }

    pub fn good_parse(&self) -> bool {
        self.queue.is_empty() && self.sentence_position == self.sentence.len()
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorBeam<T> {
    pub log_probability: LogProb<f64>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    pub sentence: Vec<T>,
    pub rules: Vec<Rule>,
    pub top_id: usize,
    pub steps: usize,
}

impl<T: Eq + std::fmt::Debug> PartialEq for GeneratorBeam<T> {
    fn eq(&self, other: &Self) -> bool {
        self.log_probability == other.log_probability
            && self.sentence == other.sentence
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug> PartialOrd for GeneratorBeam<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug> Eq for GeneratorBeam<T> {}

impl<T: Eq + std::fmt::Debug> Ord for GeneratorBeam<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_probability
            .partial_cmp(&other.log_probability)
            .unwrap()
    }
}

impl<T: Clone> Beam<T> for GeneratorBeam<T> {
    fn log_probability(&self) -> &LogProb<f64> {
        &self.log_probability
    }

    fn log_probability_mut(&mut self) -> &mut LogProb<f64> {
        &mut self.log_probability
    }

    fn push_moment(&mut self, x: ParseMoment) {
        self.queue.push(Reverse(x))
    }

    fn push_rule(&mut self, x: Rule) {
        self.rules.push(x)
    }

    fn inc(&mut self) {
        self.steps += 1;
    }

    fn top_id(&self) -> usize {
        self.top_id
    }

    fn top_id_mut(&mut self) -> &mut usize {
        &mut self.top_id
    }

    fn scan(
        v: &mut Vec<Self>,
        moment: &ParseMoment,
        mut beam: Self,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        if let Some(s) = s {
            //If the word was None then adding it does nothing
            beam.sentence.push(s.clone());
        }
        beam.log_probability += child_prob;
        beam.rules.push(Rule::Scan {
            node: child_node,
            parent: moment.tree.id,
        });
        beam.steps += 1;
        v.push(beam);
    }

    fn pop_moment(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }
}

impl<T: Eq + std::fmt::Debug + Clone> GeneratorBeam<T> {
    pub fn new<Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
    ) -> Result<GeneratorBeam<T>> {
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

        Ok(GeneratorBeam {
            log_probability: LogProb::new(0_f64).unwrap(),
            sentence: vec![],
            queue,
            rules: vec![Rule::Start(category_index)],
            top_id: 0,
            steps: 0,
        })
    }
}
