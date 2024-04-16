use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use crate::{Direction, ParseHeap};
use allocator_api2::alloc::Allocator;
use anyhow::Result;
use beam::Beam;
use itertools::repeat_n;
use petgraph::graph::NodeIndex;
use thin_vec::{thin_vec, ThinVec};
pub(crate) use trees::FutureTree;
pub(crate) use trees::GornIndex;
pub(crate) use trees::ParseMoment;

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

fn clone_push<T: Clone + Default>(v: &[T], x: T) -> ThinVec<T> {
    let mut v: ThinVec<T> = ThinVec::from(v);
    v.push(x);
    v
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmerge_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    L: Lexiconable<T, Category>,
    B: Beam<T, Probability = L::Probability> + Clone,
    A: Allocator,
>(
    v: &mut ParseHeap<T, B, A>,
    lexicon: &L,
    moment: &ParseMoment,
    beam: &B,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: L::Probability,
    rule_prob: L::Probability,
    inverse_rule_prob: L::Probability,
) -> L::Probability {
    let mut output = lexicon.probability_of_one();
    for mover in moment.movers.iter() {
        for (stored_prob, stored_child_node) in lexicon.children_of(mover.node) {
            let stored = lexicon.get(stored_child_node).unwrap();
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

                    beam.add_to_log_prob(stored_prob);
                    beam.add_to_log_prob(child_prob.clone());
                    beam.add_to_log_prob(rule_prob.clone());
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
                    output = inverse_rule_prob.clone();
                }
                _ => (),
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmerge<
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    L: Lexiconable<T, Category>,
    B: Beam<T, Probability = L::Probability> + Clone,
    A: Allocator,
>(
    v: &mut ParseHeap<T, B, A>,
    lexicon: &L,
    moment: &ParseMoment,
    beam: B,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: L::Probability,
    rule_prob: L::Probability,
) -> Result<()> {
    let complements = lexicon.find_category(cat)?;
    for ((complement_prob, complement), mut beam) in
        complements.iter().zip(repeat_n(beam, complements.len()))
    {
        beam.push_moment(ParseMoment::new(
            FutureTree {
                node: *complement,
                index: moment.tree.index.clone_push(*dir)?,
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
                index: moment.tree.index.clone_push(dir.flip())?,
                id: beam.top_id() + 2,
            },
            match dir {
                Direction::Right => thin_vec![],
                Direction::Left => moment.movers.clone(),
            },
        ));

        beam.add_to_log_prob(complement_prob.clone());
        beam.add_to_log_prob(child_prob.clone());
        beam.add_to_log_prob(rule_prob.clone());
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
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    L: Lexiconable<T, Category>,
    B: Beam<T, Probability = L::Probability> + Clone,
    A: Allocator,
>(
    v: &mut ParseHeap<T, B, A>,
    lexicon: &L,
    moment: &ParseMoment,
    beam: &B,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: L::Probability,
    rule_prob: L::Probability,
    inverse_rule_prob: L::Probability,
) -> L::Probability {
    let mut output = lexicon.probability_of_one();
    for mover in moment.movers.iter() {
        for (stored_prob, stored_child_node) in lexicon.children_of(mover.node) {
            let stored = lexicon.get(stored_child_node).unwrap();
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

                    beam.add_to_log_prob(stored_prob);
                    beam.add_to_log_prob(child_prob.clone());
                    beam.add_to_log_prob(rule_prob.clone());
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
                    output = inverse_rule_prob.clone();
                }
                _ => (),
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove<
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    L: Lexiconable<T, Category>,
    B: Beam<T, Probability = L::Probability> + Clone,
    A: Allocator,
>(
    v: &mut ParseHeap<T, B, A>,
    lexicon: &L,
    moment: &ParseMoment,
    beam: B,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: L::Probability,
    rule_prob: L::Probability,
) -> Result<()> {
    let storeds = lexicon.find_licensee(cat)?;

    for ((stored_prob, stored), mut beam) in storeds.iter().zip(repeat_n(beam, storeds.len())) {
        beam.push_moment(ParseMoment::new(
            FutureTree {
                node: child_node,
                index: moment.tree.index.clone_push(Direction::Right)?,
                id: beam.top_id() + 1,
            },
            clone_push(
                &moment.movers,
                FutureTree {
                    node: *stored,
                    index: moment.tree.index.clone_push(Direction::Left)?,
                    id: beam.top_id() + 2,
                },
            ),
        ));
        beam.add_to_log_prob(stored_prob.clone());
        beam.add_to_log_prob(child_prob.clone());
        beam.add_to_log_prob(rule_prob.clone());

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
    }
    Ok(())
}

pub fn expand<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    L: Lexiconable<T, Category>,
    B: Beam<T, Probability = L::Probability> + Clone + 'a,
    A: Allocator,
>(
    extender: &mut ParseHeap<'a, T, B, A>,
    moment: ParseMoment,
    beam: B,
    lexicon: &'a L,
    probability_of_moving: L::Probability,
    probability_of_merging: L::Probability,
) {
    let n_children = lexicon.n_children(moment.tree.node);
    let new_beams = itertools::repeat_n(beam, n_children);

    new_beams
        .zip(lexicon.children_of(moment.tree.node))
        .for_each(|(beam, (child_prob, child_node))| {
            let child = lexicon.get(child_node).unwrap();
            match &child {
                FeatureOrLemma::Lemma(s) if moment.no_movers() => {
                    B::scan(extender, &moment, beam, s, child_node, child_prob);
                }
                FeatureOrLemma::Feature(Feature::Selector(cat, dir)) => {
                    let rule_prob = unmerge_from_mover(
                        extender,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        child_prob.clone(),
                        probability_of_moving.clone(),
                        probability_of_merging.clone(),
                    );
                    let _ = unmerge(
                        extender, lexicon, &moment, beam, cat, dir, child_node, child_prob,
                        rule_prob,
                    );
                }
                FeatureOrLemma::Feature(Feature::Licensor(cat)) => {
                    let rule_prob = unmove_from_mover(
                        extender,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        child_prob.clone(),
                        probability_of_moving.clone(),
                        probability_of_merging.clone(),
                    );
                    let _ = unmove(
                        extender, lexicon, &moment, beam, cat, child_node, child_prob, rule_prob,
                    );
                }
                _ => (),
            }
        });
}

pub mod beam;
#[cfg(test)]
mod tests;
mod trees;
pub use trees::MAX_STEPS;
