use crate::lexicon::{Feature, FeatureOrLemma, Lexicon};
use crate::Direction;
use anyhow::Result;
use beam::Beam;
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;
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

fn clone_push<T: Clone>(v: &[T], x: T) -> Vec<T> {
    let mut v = v.to_vec();
    v.push(x);
    v
}

fn scan<T: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<Beam<T>>,
    moment: &ParseMoment,
    mut beam: Beam<T>,
    s: &Option<T>,
    child_node: NodeIndex,
    child_prob: f64,
) {
    let should_push = if let Some(s) = s {
        if let Some(x) = beam.sentence.last() {
            if s == x {
                beam.sentence.pop();
                true
            } else {
                //The word was the wrong word.
                false
            }
        } else {
            //The sentence is empty but we need a word.
            false
        }
    } else {
        //Invisible word to scan
        true
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
fn reverse_scan<T: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<Beam<T>>,
    moment: &ParseMoment,
    mut beam: Beam<T>,
    s: &Option<T>,
    child_node: NodeIndex,
    child_prob: f64,
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

#[allow(clippy::too_many_arguments)]
fn unmerge_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
>(
    v: &mut Vec<Beam<T>>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    beam: &Beam<T>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
    rule_prob: f64,
) {
    for mover_id in 0..moment.movers.len() {
        for stored_child_node in lexicon.children_of(moment.movers[mover_id].node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Category(stored)) if stored == cat => {
                    let mut beam = beam.clone();
                    beam.queue.push(Reverse(ParseMoment {
                        tree: FutureTree {
                            node: child_node,
                            index: moment.tree.index.clone(),
                            id: beam.top_id + 1,
                        },
                        movers: vec![],
                    }));
                    beam.queue.push(Reverse(ParseMoment {
                        tree: FutureTree {
                            node: stored_child_node,
                            index: moment.movers[mover_id].index.clone(),
                            id: beam.top_id + 2,
                        },
                        movers: moment
                            .movers
                            .iter()
                            .enumerate()
                            .filter(|&(i, _v)| i != mover_id)
                            .map(|(_, v)| v.clone())
                            .collect(),
                    }));

                    beam.log_probability += stored_prob + child_prob + rule_prob;
                    beam.rules.push(Rule::UnmergeFromMover {
                        child: child_node,
                        child_id: beam.top_id + 1,
                        stored_id: beam.top_id + 2,
                        parent: moment.tree.id,
                        storage: moment.movers[mover_id].id,
                    });
                    beam.top_id += 2;
                    beam.steps += 1;
                    v.push(beam);
                }
                _ => (),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unmerge<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
>(
    v: &mut Vec<Beam<T>>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: Beam<T>,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: f64,
    rule_prob: f64,
) -> Result<()> {
    let complement = lexicon.find_category(cat.clone())?;
    beam.queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: complement,
            index: moment.tree.index.clone_push(*dir),
            id: beam.top_id + 1,
        },
        movers: match dir {
            Direction::Right => moment.movers.clone(),
            Direction::Left => vec![],
        },
    }));
    beam.queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(dir.flip()),
            id: beam.top_id + 2,
        },
        movers: match dir {
            Direction::Right => vec![],
            Direction::Left => moment.movers.clone(),
        },
    }));

    beam.log_probability += child_prob + rule_prob;
    beam.rules.push(Rule::Unmerge {
        child: child_node,
        parent: moment.tree.id,
        child_id: beam.top_id + 2,
        complement_id: beam.top_id + 1,
    });
    beam.top_id += 2;
    beam.steps += 1;
    v.push(beam);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn unmove_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
>(
    v: &mut Vec<Beam<T>>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    beam: &Beam<T>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
    rule_prob: f64,
) {
    for mover_id in 0..moment.movers.len() {
        for stored_child_node in lexicon.children_of(moment.movers[mover_id].node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Licensee(s)) if cat == s => {
                    let mut beam = beam.clone();
                    beam.queue.push(Reverse(ParseMoment {
                        tree: FutureTree {
                            node: child_node,
                            index: moment.tree.index.clone(),
                            id: beam.top_id + 1,
                        },
                        movers: moment
                            .movers
                            .iter()
                            .enumerate()
                            .filter(|&(i, _v)| i != mover_id)
                            .map(|(_, v)| v.clone())
                            .chain(std::iter::once(FutureTree {
                                node: stored_child_node,
                                index: moment.movers[mover_id].index.clone(),
                                id: beam.top_id + 2,
                            }))
                            .collect(),
                    }));
                    beam.log_probability += stored_prob + child_prob + rule_prob;
                    beam.rules.push(Rule::UnmoveFromMover {
                        parent: moment.tree.id,
                        child_id: beam.top_id + 1,
                        stored_id: beam.top_id + 2,
                        storage: moment.movers[mover_id].id,
                    });
                    beam.top_id += 2;
                    beam.steps += 1;
                    v.push(beam);
                }
                _ => (),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unmove<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
>(
    v: &mut Vec<Beam<T>>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: Beam<T>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
    rule_prob: f64,
) -> Result<()> {
    let stored = lexicon.find_licensee(cat.clone())?;

    beam.queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(Direction::Right),
            id: beam.top_id + 1,
        },
        movers: clone_push(
            &moment.movers,
            FutureTree {
                node: stored,
                index: moment.tree.index.clone_push(Direction::Left),
                id: beam.top_id + 2,
            },
        ),
    }));

    beam.log_probability += child_prob + rule_prob;
    beam.rules.push(Rule::Unmove {
        child_id: beam.top_id + 1,
        stored_id: beam.top_id + 2,
        parent: moment.tree.id,
    });
    beam.top_id += 2;
    beam.steps += 1;
    v.push(beam);
    Ok(())
}

pub fn expand_generate<
    'a,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
>(
    moment: &'a ParseMoment,
    beam: Beam<T>,
    lexicon: &'a Lexicon<T, Category>,
    probability_of_moving: f64,
    probability_of_merging: f64,
) -> impl Iterator<Item = Beam<T>> + 'a {
    #[cfg(test)]
    {
        let c = if probability_of_moving > probability_of_merging {
            probability_of_moving
        } else {
            probability_of_merging
        };
        let n = c + ((probability_of_merging - c).exp() + (probability_of_moving - c).exp()).ln();
        assert_eq!(n, 0.0_f64)
    }

    let children = lexicon
        .children_of(moment.tree.node)
        .map(|nx| (nx, lexicon.get(nx).unwrap()));
    let new_beams = itertools::repeat_n(beam, lexicon.n_children(moment.tree.node));

    children
        .zip(new_beams)
        .flat_map(move |((child_node, (child, child_prob)), beam)| {
            let mut v: Vec<_> = vec![];
            match &child {
                FeatureOrLemma::Lemma(s) if moment.movers.is_empty() => {
                    reverse_scan(&mut v, moment, beam, s, child_node, child_prob);
                }
                FeatureOrLemma::Feature(Feature::Selector(cat, dir)) => {
                    unmerge_from_mover(
                        &mut v,
                        lexicon,
                        moment,
                        &beam,
                        cat,
                        child_node,
                        child_prob,
                        probability_of_moving,
                    );
                    let rule_prob = if v.len() > 1 {
                        probability_of_merging
                    } else {
                        0_f64
                    };
                    //Right now we just ignore the error, it means no beam will be added.
                    let _ = unmerge(
                        &mut v, lexicon, moment, beam, cat, dir, child_node, child_prob, rule_prob,
                    );
                }
                FeatureOrLemma::Feature(Feature::Licensor(cat)) => {
                    unmove_from_mover(
                        &mut v,
                        lexicon,
                        moment,
                        &beam,
                        cat,
                        child_node,
                        child_prob,
                        probability_of_moving,
                    );
                    let rule_prob = if v.len() > 1 {
                        probability_of_merging
                    } else {
                        0_f64
                    };
                    let _ = unmove(
                        &mut v, lexicon, moment, beam, cat, child_node, child_prob, rule_prob,
                    );
                }
                _ => (),
            }
            v.into_iter()
        })
}

pub fn expand_parse<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    moment: ParseMoment,
    beam: Beam<&'a T>,
    lexicon: &'a Lexicon<T, Category>,
    probability_of_moving: f64,
    probability_of_merging: f64,
) -> impl Iterator<Item = Beam<&'a T>> + 'a {
    #[cfg(test)]
    {
        let c = if probability_of_moving > probability_of_merging {
            probability_of_moving
        } else {
            probability_of_merging
        };
        let n = c + ((probability_of_merging - c).exp() + (probability_of_moving - c).exp()).ln();
        assert_eq!(n, 0.0_f64)
    }
    let children = lexicon
        .children_of(moment.tree.node)
        .map(|nx| (nx, lexicon.get(nx).unwrap()));
    let new_beams = itertools::repeat_n(beam, lexicon.n_children(moment.tree.node));

    children
        .zip(new_beams)
        .flat_map(move |((child_node, (child, child_prob)), beam)| {
            let mut v: Vec<_> = vec![];
            match &child {
                FeatureOrLemma::Lemma(s) if moment.movers.is_empty() => {
                    scan(&mut v, &moment, beam, &s.as_ref(), child_node, child_prob);
                }
                FeatureOrLemma::Feature(Feature::Selector(cat, dir)) => {
                    unmerge_from_mover(
                        &mut v,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        child_prob,
                        probability_of_moving,
                    );
                    let rule_prob = if v.len() > 1 {
                        probability_of_merging
                    } else {
                        0_f64
                    };
                    let _ = unmerge(
                        &mut v, lexicon, &moment, beam, cat, dir, child_node, child_prob, rule_prob,
                    );
                }
                FeatureOrLemma::Feature(Feature::Licensor(cat)) => {
                    unmove_from_mover(
                        &mut v,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        child_prob,
                        probability_of_moving,
                    );
                    let rule_prob = if v.len() > 1 {
                        probability_of_merging
                    } else {
                        0_f64
                    };
                    let _ = unmove(
                        &mut v, lexicon, &moment, beam, cat, child_node, child_prob, rule_prob,
                    );
                }
                _ => (),
            }
            v.into_iter()
        })
}

pub mod beam;
#[cfg(test)]
mod tests;
mod trees;
