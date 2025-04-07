use super::trees::{FutureTree, GornIndex, ParseMoment};
use super::{BeamWrapper, Rule};
use crate::lexicon::Lexicon;
use crate::ParseHeap;
use anyhow::Result;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::marker::PhantomData;
use thin_vec::thin_vec;

pub trait Beam<T>: Sized {
    fn scan(
        v: &mut ParseHeap<T, Self>,
        moment: &ParseMoment,
        beam: BeamWrapper<T, Self>,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBeam<'a, T> {
    pub sentence: Vec<(&'a [T], usize)>,
}

impl<T> Beam<T> for ParseBeam<'_, T>
where
    T: std::cmp::Eq + std::fmt::Debug,
{
    fn scan(
        v: &mut ParseHeap<T, Self>,
        moment: &ParseMoment,
        mut beam: BeamWrapper<T, Self>,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        beam.beam
            .sentence
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
        if !beam.beam.sentence.is_empty() {
            beam.log_prob += child_prob;
            if beam.record_rules() {
                beam.rules.push(Rule::Scan {
                    node: child_node,
                    parent: moment.tree.id,
                });
            }
            beam.n_steps += 1;
            v.push(beam);
        };
    }
}

impl<'a, T: Eq + std::fmt::Debug + Clone> ParseBeam<'a, T> {
    pub fn new_multiple<U, Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
        record_rules: bool,
    ) -> Result<BeamWrapper<T, ParseBeam<'a, T>>>
    where
        U: AsRef<[T]>,
    {
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

        Ok(BeamWrapper {
            beam: ParseBeam {
                sentence: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
            },
            queue,
            log_prob: LogProb::prob_of_one(),
            phantom: PhantomData,
            record_rules,
            rules: if record_rules {
                thin_vec![Rule::Start(category_index)]
            } else {
                thin_vec![]
            },
            top_id: 0,
            n_steps: 0,
        })
    }

    pub fn new_single<Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentence: &'a [T],
        record_rules: bool,
    ) -> Result<BeamWrapper<T, ParseBeam<'a, T>>> {
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

        Ok(BeamWrapper {
            beam: ParseBeam {
                sentence: vec![(sentence, 0)],
            },
            queue,
            log_prob: LogProb::prob_of_one(),
            phantom: PhantomData,
            record_rules,
            rules: if record_rules {
                thin_vec![Rule::Start(category_index)]
            } else {
                thin_vec![]
            },
            top_id: 0,
            n_steps: 0,
        })
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
                b.rules.to_vec(),
            ))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyBeam<'a, T> {
    generated_sentences: Vec<T>,
    sentence_guides: Vec<(&'a [T], usize)>,
}

impl<'a, T: Eq + std::fmt::Debug + Clone> FuzzyBeam<'a, T> {
    pub fn new<U, Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentences: &'a [U],
        record_rules: bool,
    ) -> Result<BeamWrapper<T, FuzzyBeam<'a, T>>>
    where
        U: AsRef<[T]>,
    {
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

        Ok(BeamWrapper {
            beam: FuzzyBeam {
                sentence_guides: sentences.iter().map(|x| (x.as_ref(), 0)).collect(),
                generated_sentences: vec![],
                //           n_sentences: (sentences.len() + 1) as f64,
            },
            record_rules,
            queue,
            log_prob: LogProb::prob_of_one(),
            phantom: PhantomData,
            rules: if record_rules {
                thin_vec![Rule::Start(category_index)]
            } else {
                thin_vec![]
            },
            top_id: 0,
            n_steps: 0,
        })
    }

    pub fn yield_good_parse(b: BeamWrapper<T, Self>) -> Option<(LogProb<f64>, Vec<T>, Vec<Rule>)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.generated_sentences.to_vec(),
                b.rules.to_vec(),
            ))
        } else {
            None
        }
    }
}

impl<T> Beam<T> for FuzzyBeam<'_, T>
where
    T: std::cmp::Eq + std::fmt::Debug + Clone,
{
    fn scan(
        v: &mut ParseHeap<T, Self>,
        moment: &ParseMoment,
        mut beam: BeamWrapper<T, Self>,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        if let Some(s) = s {
            beam.beam.generated_sentences.push(s.clone());
        }
        beam.beam
            .sentence_guides
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
        beam.log_prob += child_prob;
        if beam.record_rules() {
            beam.rules.push(Rule::Scan {
                node: child_node,
                parent: moment.tree.id,
            });
        }
        beam.n_steps += 1;
        v.push(beam);
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GeneratorBeam<T> {
    pub sentence: Vec<T>,
}

impl<T: Clone> Beam<T> for GeneratorBeam<T>
where
    T: std::cmp::Eq + std::fmt::Debug,
{
    fn scan(
        v: &mut ParseHeap<T, Self>,
        moment: &ParseMoment,
        mut beam: BeamWrapper<T, Self>,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        if let Some(s) = s {
            //If the word was None then adding it does nothing
            beam.beam.sentence.push(s.clone());
        }
        beam.log_prob += child_prob;
        if beam.record_rules {
            beam.rules.push(Rule::Scan {
                node: child_node,
                parent: moment.tree.id,
            });
        }
        beam.n_steps += 1;
        v.push(beam);
    }
}

impl<T: Eq + std::fmt::Debug + Clone> GeneratorBeam<T> {
    pub fn new<Category: Eq + std::fmt::Debug + Clone>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        record_rules: bool,
    ) -> Result<BeamWrapper<T, GeneratorBeam<T>>> {
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

        Ok(BeamWrapper {
            beam: GeneratorBeam { sentence: vec![] },
            queue,
            record_rules,
            rules: if record_rules {
                thin_vec![Rule::Start(category_index)]
            } else {
                thin_vec![]
            },
            log_prob: LogProb::prob_of_one(),
            phantom: PhantomData,
            top_id: 0,
            n_steps: 0,
        })
    }

    pub fn yield_good_parse(b: BeamWrapper<T, Self>) -> Option<(LogProb<f64>, Vec<T>, Vec<Rule>)> {
        if b.is_empty() {
            Some((b.log_prob, b.beam.sentence.to_vec(), b.rules.to_vec()))
        } else {
            None
        }
    }
}
