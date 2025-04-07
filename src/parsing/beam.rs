use super::trees::{FutureTree, GornIndex, ParseMoment};
use super::{BeamWrapper, Rule};
use crate::lexicon::Lexicon;
use anyhow::Result;
use logprob::LogProb;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use thin_vec::thin_vec;

pub trait Scanner<T>: Sized {
    fn scan(&mut self, s: &Option<T>) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseScan<'a, T> {
    pub sentence: Vec<(&'a [T], usize)>,
}

impl<T> Scanner<T> for ParseScan<'_, T>
where
    T: std::cmp::Eq + std::fmt::Debug,
{
    fn scan(&mut self, s: &Option<T>) -> bool {
        self.sentence.retain_mut(|(sentence, position)| match s {
            Some(s) => {
                if let Some(string) = sentence.get(*position) {
                    if s == string {
                        *position += 1;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            None => true,
        });
        !self.sentence.is_empty()
    }
}

impl<'a, T: Eq + std::fmt::Debug + Clone> ParseScan<'a, T> {
    pub fn new_multiple<U, Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
    ) -> Result<BeamWrapper<T, ParseScan<'a, T>>>
    where
        U: AsRef<[T]>,
    {
        let category_index = lexicon.find_category(&initial_category)?;

        Ok(BeamWrapper::new(
            ParseScan {
                sentence: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
            },
            category_index,
        ))
    }

    pub fn new_single<Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
    ) -> Result<BeamWrapper<T, ParseScan<'a, T>>> {
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

        Ok(BeamWrapper::new(
            ParseScan {
                sentence: vec![(sentence, 0)],
            },
            category_index,
        ))
    }

    pub fn yield_good_parse(
        b: BeamWrapper<T, Self>,
    ) -> Option<(impl Iterator<Item = &'a [T]> + 'a, LogProb<f64>, Vec<Rule>)> {
        if b.is_empty() {
            Some((
                b.beam
                    .sentence
                    .into_iter()
                    .filter(|(s, pos)| s.len() == *pos)
                    .map(|(s, _)| s),
                b.log_prob,
                b.rules.into_iter().collect::<Option<Vec<_>>>().unwrap(),
            ))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyScan<'a, T> {
    generated_sentences: Vec<T>,
    sentence_guides: Vec<(&'a [T], usize)>,
}

impl<'a, T: Eq + std::fmt::Debug + Clone> FuzzyScan<'a, T> {
    pub fn new<U, Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
    ) -> Result<BeamWrapper<T, FuzzyScan<'a, T>>>
    where
        U: AsRef<[T]>,
    {
        let category_index = lexicon.find_category(&initial_category)?;
        Ok(BeamWrapper::new(
            FuzzyScan {
                sentence_guides: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
                generated_sentences: vec![],
                //           n_sentences: (sentences.len() + 1) as f64,
            },
            category_index,
        ))
    }

    pub fn yield_good_parse(b: BeamWrapper<T, Self>) -> Option<(LogProb<f64>, Vec<T>, Vec<Rule>)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.generated_sentences.to_vec(),
                b.rules.into_iter().collect::<Option<Vec<_>>>().unwrap(),
            ))
        } else {
            None
        }
    }
}

impl<T> Scanner<T> for FuzzyScan<'_, T>
where
    T: std::cmp::Eq + std::fmt::Debug + Clone,
{
    fn scan(&mut self, s: &Option<T>) -> bool {
        if let Some(s) = s {
            self.generated_sentences.push(s.clone());
        }
        self.sentence_guides
            .retain_mut(|(sentence, position)| match s {
                Some(s) => {
                    if let Some(string) = sentence.get(*position) {
                        if s == string {
                            *position += 1;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                None => true,
            });
        true
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GeneratorScan<T> {
    pub sentence: Vec<T>,
}

impl<T: Clone> Scanner<T> for GeneratorScan<T>
where
    T: std::cmp::Eq + std::fmt::Debug,
{
    fn scan(&mut self, s: &Option<T>) -> bool {
        if let Some(s) = s {
            //If the word was None then adding it does nothing
            self.sentence.push(s.clone());
        }
        true
    }
}

impl<T: Eq + std::fmt::Debug + Clone> GeneratorScan<T> {
    pub fn new<Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
    ) -> Result<BeamWrapper<T, GeneratorScan<T>>> {
        let category_index = lexicon.find_category(&initial_category)?;

        Ok(BeamWrapper::new(
            GeneratorScan { sentence: vec![] },
            category_index,
        ))
    }

    pub fn yield_good_parse(b: BeamWrapper<T, Self>) -> Option<(LogProb<f64>, Vec<T>, Vec<Rule>)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.sentence.to_vec(),
                b.rules.into_iter().collect::<Option<Vec<_>>>().unwrap(),
            ))
        } else {
            None
        }
    }
}
