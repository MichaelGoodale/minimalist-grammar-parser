use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use crate::lexicon::{Feature, FeatureOrLemma, Lexicon};
use crate::{Direction, ParseHeap, ParsingConfig};
use anyhow::Result;
use beam::Beam;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use thin_vec::{thin_vec, ThinVec};
use trees::{FutureTree, ParseMoment};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum Rule {
    Start(NodeIndex),
    Scan {
        node: NodeIndex,
        parent: usize,
    },
    Unmerge {
        child: NodeIndex,
        parent: usize,
        child_id: usize,
        complement_id: usize,
    },
    UnmergeFromMover {
        child: NodeIndex,
        child_id: usize,
        stored_id: usize,
        parent: usize,
        storage: usize,
    },
    Unmove {
        child_id: usize,
        stored_id: usize,
        parent: usize,
    },
    UnmoveFromMover {
        child_id: usize,
        stored_id: usize,
        parent: usize,
        storage: usize,
    },
}

#[derive(Debug, Clone)]
pub struct BeamWrapper<T, B: Beam<T>> {
    log_prob: LogProb<f64>,
    queue: BinaryHeap<Reverse<ParseMoment>>,
    rules: ThinVec<Rule>,
    record_rules: bool,
    pub beam: B,
    phantom: PhantomData<T>,
}

impl<T: Eq + std::fmt::Debug, B: Beam<T> + Eq> PartialEq for BeamWrapper<T, B> {
    fn eq(&self, other: &Self) -> bool {
        self.beam == other.beam
            && self.log_prob == other.log_prob
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug, B: Beam<T> + Eq> Eq for BeamWrapper<T, B> {}

impl<T: Eq + std::fmt::Debug, B: Beam<T> + Eq> PartialOrd for BeamWrapper<T, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug, B: Beam<T> + Eq> Ord for BeamWrapper<T, B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_prob.cmp(&other.log_prob)
    }
}

impl<T, B: Beam<T>> BeamWrapper<T, B> {
    fn push_moment(&mut self, moment: ParseMoment) {
        self.queue.push(Reverse(moment))
    }

    pub fn log_prob(&self) -> LogProb<f64> {
        self.log_prob
    }

    fn top_id(&self) -> usize {
        self.beam.top_id()
    }

    fn top_id_mut(&mut self) -> &mut usize {
        self.beam.top_id_mut()
    }

    fn push_rule(&mut self, r: Rule) {
        self.rules.push(r)
    }

    fn record_rules(&self) -> bool {
        self.record_rules
    }

    fn inc(&mut self) {
        self.beam.inc()
    }

    pub fn n_steps(&self) -> usize {
        self.beam.n_steps()
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

fn clone_push<T: Clone + Default>(v: &[T], x: T) -> ThinVec<T> {
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
    B: Beam<T> + Clone + Eq,
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
                    beam.push_moment(ParseMoment::new(
                        FutureTree {
                            node: child_node,
                            index: moment.tree.index,
                            id: beam.top_id() + 1,
                        },
                        thin_vec![],
                    ));
                    beam.push_moment(ParseMoment::new(
                        FutureTree {
                            node: stored_child_node,
                            index: mover.index,
                            id: beam.top_id() + 2,
                        },
                        moment
                            .movers
                            .iter()
                            .filter(|&v| v != mover)
                            .cloned()
                            .collect(),
                    ));

                    beam.log_prob += stored_prob + child_prob + config.move_prob;
                    if beam.record_rules() {
                        beam.push_rule(Rule::UnmergeFromMover {
                            child: child_node,
                            child_id: beam.top_id() + 1,
                            stored_id: beam.top_id() + 2,
                            parent: moment.tree.id,
                            storage: mover.id,
                        });
                    }
                    *beam.top_id_mut() += 2;
                    beam.inc();
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
    B: Beam<T> + Eq,
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
    beam.push_moment(ParseMoment::new(
        FutureTree {
            node: complement,
            index: moment.tree.index.clone_push(*dir),
            id: beam.top_id() + 1,
        },
        match dir {
            Direction::Right => moment.movers.clone(),
            Direction::Left => thin_vec![],
        },
    ));
    beam.push_moment(ParseMoment::new(
        FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(dir.flip()),
            id: beam.top_id() + 2,
        },
        match dir {
            Direction::Right => thin_vec![],
            Direction::Left => moment.movers.clone(),
        },
    ));

    beam.log_prob += child_prob + rule_prob;
    if beam.record_rules() {
        beam.push_rule(Rule::Unmerge {
            child: child_node,
            parent: moment.tree.id,
            child_id: beam.top_id() + 2,
            complement_id: beam.top_id() + 1,
        });
    }
    *beam.top_id_mut() += 2;
    beam.inc();
    v.push(beam);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Beam<T> + Clone + Eq,
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
                    beam.push_moment(ParseMoment::new(
                        FutureTree {
                            node: child_node,
                            index: moment.tree.index,
                            id: beam.top_id() + 1,
                        },
                        moment
                            .movers
                            .iter()
                            .filter(|&v| v != mover)
                            .cloned()
                            .chain(std::iter::once(FutureTree {
                                node: stored_child_node,
                                index: mover.index,
                                id: beam.top_id() + 2,
                            }))
                            .collect(),
                    ));
                    beam.log_prob += stored_prob + child_prob + config.move_prob;
                    if beam.record_rules() {
                        beam.push_rule(Rule::UnmoveFromMover {
                            parent: moment.tree.id,
                            child_id: beam.top_id() + 1,
                            stored_id: beam.top_id() + 2,
                            storage: mover.id,
                        });
                    }
                    *beam.top_id_mut() += 2;
                    beam.inc();
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
    B: Beam<T> + Eq,
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

    beam.push_moment(ParseMoment::new(
        FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(Direction::Right),
            id: beam.top_id() + 1,
        },
        clone_push(
            &moment.movers,
            FutureTree {
                node: stored,
                index: moment.tree.index.clone_push(Direction::Left),
                id: beam.top_id() + 2,
            },
        ),
    ));

    beam.log_prob += child_prob + rule_prob;
    if beam.record_rules() {
        beam.push_rule(Rule::Unmove {
            child_id: beam.top_id() + 1,
            stored_id: beam.top_id() + 2,
            parent: moment.tree.id,
        });
    }
    *beam.top_id_mut() += 2;
    beam.inc();
    v.push(beam);
    Ok(())
}

pub fn expand<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Beam<T> + Eq + Clone + 'a,
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
                    B::scan(extender, &moment, beam, s, child_node, p);
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
#[cfg(test)]
mod tests;
mod trees;
pub use trees::MAX_STEPS;
