use super::{BeamWrapper, RuleHolder, rules::RulePool};
use anyhow::Result;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use std::fmt::Debug;

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
    pub fn new_multiple<U>(
        category_index: NodeIndex,
        sentences: &'a [U],
    ) -> Result<BeamWrapper<T, ParseScan<'a, T>>>
    where
        U: AsRef<[T]>,
    {
        Ok(BeamWrapper::new(
            ParseScan {
                sentence: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
            },
            category_index,
        ))
    }

    pub fn new_single(
        category_index: NodeIndex,
        sentence: &'a [T],
    ) -> Result<BeamWrapper<T, ParseScan<'a, T>>> {
        Ok(BeamWrapper::new(
            ParseScan {
                sentence: vec![(sentence, 0)],
            },
            category_index,
        ))
    }

    pub fn yield_good_parse(
        b: BeamWrapper<T, Self>,
        rules: &[RuleHolder],
    ) -> Option<(impl Iterator<Item = &'a [T]> + 'a, LogProb<f64>, RulePool)> {
        if b.is_empty() {
            Some((
                b.beam
                    .sentence
                    .into_iter()
                    .filter(|(s, pos)| s.len() == *pos)
                    .map(|(s, _)| s),
                b.log_prob,
                b.rules.into_rule_pool(rules),
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
    pub fn new<U>(
        category_index: NodeIndex,
        sentences: &'a [U],
    ) -> Result<BeamWrapper<T, FuzzyScan<'a, T>>>
    where
        U: AsRef<[T]>,
    {
        Ok(BeamWrapper::new(
            FuzzyScan {
                sentence_guides: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
                generated_sentences: vec![],
                //           n_sentences: (sentences.len() + 1) as f64,
            },
            category_index,
        ))
    }

    pub fn yield_good_parse(
        b: BeamWrapper<T, Self>,
        rules: &[RuleHolder],
    ) -> Option<(LogProb<f64>, Vec<T>, RulePool)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.generated_sentences.to_vec(),
                b.rules.into_rule_pool(rules),
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
    pub fn new(category_index: NodeIndex) -> Result<BeamWrapper<T, GeneratorScan<T>>> {
        Ok(BeamWrapper::new(
            GeneratorScan { sentence: vec![] },
            category_index,
        ))
    }

    pub fn yield_good_parse(
        b: BeamWrapper<T, Self>,
        rules: &[RuleHolder],
    ) -> Option<(LogProb<f64>, Vec<T>, RulePool)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.sentence.to_vec(),
                b.rules.into_rule_pool(rules),
            ))
        } else {
            None
        }
    }
}
