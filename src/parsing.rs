use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use crate::lexicon::{Feature, FeatureOrLemma, Lexicon};
use crate::{Direction, ParseHeap, ParsingConfig};
use anyhow::Result;
use beam::Scanner;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use rules::{PartialRulePool, RuleIndex};
use thin_vec::{thin_vec, ThinVec};
use trees::{FutureTree, GornIndex, ParseMoment};

pub use rules::{Rule, RulePool};

#[derive(Debug, Clone)]
pub struct BeamWrapper<T, B: Scanner<T>> {
    log_prob: LogProb<f64>,
    queue: BinaryHeap<Reverse<ParseMoment>>,
    rules: PartialRulePool,
    pub beam: B,
    phantom: PhantomData<T>,
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> PartialEq for BeamWrapper<T, B> {
    fn eq(&self, other: &Self) -> bool {
        self.beam == other.beam
            && self.log_prob == other.log_prob
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> Eq for BeamWrapper<T, B> {}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> PartialOrd for BeamWrapper<T, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> Ord for BeamWrapper<T, B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_prob.cmp(&other.log_prob)
    }
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> BeamWrapper<T, B> {
    fn scan(
        mut self,
        v: &mut ParseHeap<T, B>,
        moment: &ParseMoment,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        if self.beam.scan(s) {
            self.log_prob += child_prob;
            self.rules
                .push_rule(Rule::Scan { node: child_node }, moment.tree.id, None);
            v.push(self);
        }
    }

    fn new_future_tree(&mut self, node: NodeIndex, index: GornIndex) -> (FutureTree, RuleIndex) {
        let id = self.rules.fresh();
        (FutureTree { node, index, id }, id)
    }

    fn push_moment(
        &mut self,
        node: NodeIndex,
        index: GornIndex,
        movers: ThinVec<FutureTree>,
    ) -> RuleIndex {
        let (tree, id) = self.new_future_tree(node, index);
        self.queue.push(Reverse(ParseMoment { tree, movers }));
        id
    }

    fn new(beam: B, category_index: NodeIndex) -> Self {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        queue.push(Reverse(ParseMoment::new(
            FutureTree {
                node: category_index,
                index: GornIndex::default(),
                id: RuleIndex::one(),
            },
            thin_vec![],
        )));
        BeamWrapper {
            beam,
            queue,
            log_prob: LogProb::prob_of_one(),
            rules: PartialRulePool::start_from_category(category_index),
            phantom: PhantomData,
        }
    }

    pub fn log_prob(&self) -> LogProb<f64> {
        self.log_prob
    }

    pub fn n_steps(&self) -> usize {
        self.rules.n_steps()
    }

    pub fn pop_moment(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

fn clone_push<T: Clone>(v: &[T], x: T) -> ThinVec<T> {
    let mut v: ThinVec<T> = ThinVec::from(v);
    v.push(x);
    v
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmerge_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Clone + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    beam: &BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    config: &ParsingConfig,
) -> bool {
    let mut new_beam = false;
    for mover in moment.movers.iter() {
        for stored_child_node in lexicon.children_of(mover.node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Category(stored)) if stored == cat => {
                    let mut beam = beam.clone();
                    let child_id = beam.push_moment(child_node, moment.tree.index, thin_vec![]);
                    let stored_id = beam.push_moment(
                        stored_child_node,
                        mover.index,
                        moment
                            .movers
                            .iter()
                            .filter(|&v| v != mover)
                            .cloned()
                            .collect(),
                    );

                    beam.log_prob += stored_prob + child_prob + config.move_prob;

                    let trace_id = beam.rules.fresh_trace();
                    beam.rules.push_rule(
                        Rule::UnmergeFromMover {
                            child: child_node,
                            child_id,
                            stored_id,
                            trace_id,
                        },
                        moment.tree.id,
                        Some((trace_id, mover.id)),
                    );
                    v.push(beam);
                    new_beam = true;
                }
                _ => (),
            }
        }
    }
    new_beam
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmerge<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    rule_prob: LogProb<f64>,
) -> Result<()> {
    let complement = lexicon.find_category(cat)?;
    let complement_id = beam.push_moment(
        complement,
        moment.tree.index.clone_push(*dir),
        match dir {
            Direction::Right => moment.movers.clone(),
            Direction::Left => thin_vec![],
        },
    );
    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(dir.flip()),
        match dir {
            Direction::Right => thin_vec![],
            Direction::Left => moment.movers.clone(),
        },
    );

    beam.log_prob += child_prob + rule_prob;
    beam.rules.push_rule(
        Rule::Unmerge {
            child: child_node,
            child_id,
            complement_id,
        },
        moment.tree.id,
        None,
    );
    v.push(beam);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Clone + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    beam: &BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    config: &ParsingConfig,
) -> bool {
    let mut new_beam_found = false;
    for mover in moment.movers.iter() {
        for stored_child_node in lexicon.children_of(mover.node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Licensee(s)) if cat == s => {
                    let mut beam = beam.clone();
                    let (stored_tree, stored_id) =
                        beam.new_future_tree(stored_child_node, mover.index);

                    let movers = moment
                        .movers
                        .iter()
                        .filter(|&v| v != mover)
                        .cloned()
                        .chain(std::iter::once(stored_tree))
                        .collect();

                    let child_id = beam.push_moment(child_node, moment.tree.index, movers);
                    beam.log_prob += stored_prob + child_prob + config.move_prob;

                    let trace_id = beam.rules.fresh_trace();
                    beam.rules.push_rule(
                        Rule::UnmoveFromMover {
                            child_id,
                            stored_id,
                            trace_id,
                        },
                        moment.tree.id,
                        Some((trace_id, mover.id)),
                    );
                    v.push(beam);
                    new_beam_found = true;
                }
                _ => (),
            }
        }
    }
    new_beam_found
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    rule_prob: LogProb<f64>,
) -> Result<()> {
    let stored = lexicon.find_licensee(cat)?;
    let (stored, stored_id) =
        beam.new_future_tree(stored, moment.tree.index.clone_push(Direction::Left));

    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(Direction::Right),
        clone_push(&moment.movers, stored),
    );

    beam.log_prob += child_prob + rule_prob;
    beam.rules.push_rule(
        Rule::Unmove {
            child_id,
            stored_id,
        },
        moment.tree.id,
        None,
    );
    v.push(beam);
    Ok(())
}

pub fn expand<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone + 'a,
>(
    extender: &mut ParseHeap<'a, T, B>,
    moment: ParseMoment,
    beam: BeamWrapper<T, B>,
    lexicon: &'a Lexicon<T, Category>,
    config: &ParsingConfig,
) {
    let n_children = lexicon.n_children(moment.tree.node);
    let new_beams = itertools::repeat_n(beam, n_children);

    new_beams
        .zip(lexicon.children_of(moment.tree.node))
        .for_each(
            |(beam, child_node)| match lexicon.get(child_node).unwrap() {
                (FeatureOrLemma::Lemma(s), p) if moment.no_movers() => {
                    beam.scan(extender, &moment, s, child_node, p);
                }
                (FeatureOrLemma::Feature(Feature::Selector(cat, dir)), p) => {
                    let new_beam_found = unmerge_from_mover(
                        extender, lexicon, &moment, &beam, cat, child_node, p, config,
                    );
                    let _ = unmerge(
                        extender,
                        lexicon,
                        &moment,
                        beam,
                        cat,
                        dir,
                        child_node,
                        p,
                        if new_beam_found {
                            config.dont_move_prob
                        } else {
                            LogProb::prob_of_one()
                        },
                    );
                }
                (FeatureOrLemma::Feature(Feature::Licensor(cat)), p) => {
                    let new_beam_found = unmove_from_mover(
                        extender, lexicon, &moment, &beam, cat, child_node, p, config,
                    );
                    let _ = unmove(
                        extender,
                        lexicon,
                        &moment,
                        beam,
                        cat,
                        child_node,
                        p,
                        if new_beam_found {
                            config.dont_move_prob
                        } else {
                            LogProb::prob_of_one()
                        },
                    );
                }
                _ => (),
            },
        );
}

pub mod beam;
mod rules;
#[cfg(test)]
mod tests;
mod trees;
pub use trees::MAX_STEPS;
