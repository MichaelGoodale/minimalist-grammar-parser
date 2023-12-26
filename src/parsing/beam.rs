use super::trees::{FutureTree, GornIndex, ParseMoment};
use super::Rule;
use crate::lexicon::Lexicon;
use anyhow::Result;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct Beam<T> {
    pub log_probability: f64,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    pub sentence: Vec<T>,
    pub rules: Vec<Rule>,
}

impl<T: Eq + std::fmt::Debug> PartialEq for Beam<T> {
    fn eq(&self, other: &Self) -> bool {
        self.log_probability == other.log_probability
            && self.sentence == other.sentence
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug> PartialOrd for Beam<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug> Eq for Beam<T> {}

impl<T: Eq + std::fmt::Debug> Ord for Beam<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_probability
            .partial_cmp(&other.log_probability)
            .unwrap()
    }
}

impl<'a, T: Eq + std::fmt::Debug> Beam<T> {
    pub fn pop(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    pub fn new_empty<Category: Eq + std::fmt::Debug>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
    ) -> Result<Beam<T>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(initial_category)?;

        queue.push(Reverse(ParseMoment {
            tree: FutureTree {
                node: category_index,
                index: GornIndex::default(),
            },
            movers: vec![],
        }));

        Ok(Beam {
            log_probability: 0_f64,
            sentence: vec![],
            queue,
            rules: vec![Rule::Start(category_index)],
        })
    }
    pub fn new<Category: Eq + std::fmt::Debug>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
    ) -> Result<Beam<&'a T>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(initial_category)?;

        queue.push(Reverse(ParseMoment {
            tree: FutureTree {
                node: category_index,
                index: GornIndex::default(),
            },
            movers: vec![],
        }));

        Ok(Beam {
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
