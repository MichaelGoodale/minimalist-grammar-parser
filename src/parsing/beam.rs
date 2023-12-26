use super::trees::{FutureTree, GornIndex, ParseMoment};
use super::Rule;
use crate::lexicon::Lexicon;
use anyhow::Result;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct ParseBeam<'a, T> {
    pub log_probability: f64,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    pub sentence: Vec<&'a T>,
    pub rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
pub struct GenerationBeam<T> {
    pub log_probability: f64,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    pub sentence: Vec<T>,
    pub rules: Vec<Rule>,
}

pub trait Beam<T>: Sized {
    fn pop(&mut self) -> Option<ParseMoment>;
    fn new(
        log_probability: f64,
        queue: BinaryHeap<Reverse<ParseMoment>>,
        sentence: Vec<T>,
        rules: Vec<Rule>,
    ) -> Self;

    fn sentence(&self) -> &[T];
    fn log_probability(&self) -> f64;
    fn queue(&self) -> &BinaryHeap<Reverse<ParseMoment>>;
    fn rules(&self) -> &[Rule];
}

impl<T: Eq + std::fmt::Debug> PartialEq for ParseBeam<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.log_probability == other.log_probability
            && self.sentence == other.sentence
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug> PartialEq for GenerationBeam<T> {
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

impl<T: Eq + std::fmt::Debug> PartialOrd for GenerationBeam<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug> Eq for ParseBeam<'_, T> {}
impl<T: Eq + std::fmt::Debug> Eq for GenerationBeam<T> {}

impl<T: Eq + std::fmt::Debug> Ord for ParseBeam<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_probability
            .partial_cmp(&other.log_probability)
            .unwrap()
    }
}

impl<T: Eq + std::fmt::Debug> Ord for GenerationBeam<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_probability
            .partial_cmp(&other.log_probability)
            .unwrap()
    }
}

impl<'a, T: Eq + std::fmt::Debug> Beam<&'a T> for ParseBeam<'a, T> {
    fn pop(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    fn new(
        log_probability: f64,
        queue: BinaryHeap<Reverse<ParseMoment>>,
        sentence: Vec<&'a T>,
        rules: Vec<Rule>,
    ) -> Self {
        todo!()
    }

    fn sentence(&self) -> &[&'a T] {
        todo!()
    }

    fn log_probability(&self) -> f64 {
        todo!()
    }

    fn queue(&self) -> &BinaryHeap<Reverse<ParseMoment>> {
        todo!()
    }

    fn rules(&self) -> &[Rule] {
        todo!()
    }
}

impl<T: Eq + std::fmt::Debug> Beam<T> for GenerationBeam<T> {
    fn pop(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    fn new(
        log_probability: f64,
        queue: BinaryHeap<Reverse<ParseMoment>>,
        sentence: Vec<T>,
        rules: Vec<Rule>,
    ) -> Self {
        todo!()
    }

    fn sentence(&self) -> &[T] {
        todo!()
    }

    fn log_probability(&self) -> f64 {
        todo!()
    }

    fn queue(&self) -> &BinaryHeap<Reverse<ParseMoment>> {
        todo!()
    }

    fn rules(&self) -> &[Rule] {
        todo!()
    }
}
impl<'a, T: Eq + std::fmt::Debug> GenerationBeam<T> {
    pub fn new<Category: Eq + std::fmt::Debug>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
    ) -> Result<GenerationBeam<T>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(initial_category)?;

        queue.push(Reverse(ParseMoment {
            tree: FutureTree {
                node: category_index,
                index: GornIndex::default(),
            },
            movers: vec![],
        }));

        Ok(GenerationBeam {
            log_probability: 0_f64,
            sentence: vec![],
            queue,
            rules: vec![Rule::Start(category_index)],
        })
    }
}

impl<'a, T: Eq + std::fmt::Debug> ParseBeam<'a, T> {
    pub fn new<Category: Eq + std::fmt::Debug>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
    ) -> Result<ParseBeam<'a, T>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(initial_category)?;

        queue.push(Reverse(ParseMoment {
            tree: FutureTree {
                node: category_index,
                index: GornIndex::default(),
            },
            movers: vec![],
        }));

        Ok(ParseBeam {
            log_probability: 0_f64,
            sentence: sentence.iter().collect(),
            queue,
            rules: vec![Rule::Start(category_index)],
        })
    }

    pub fn good_parse(&self) -> bool {
        self.queue.is_empty() && self.sentence.is_empty()
    }
}
