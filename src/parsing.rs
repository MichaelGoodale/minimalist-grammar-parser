use crate::lexicon::{Feature, FeatureOrLemma, Lexicon};
use crate::Direction;
use beam::Beam;
use beam::ParseBeam;
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;
use trees::{FutureTree, ParseMoment};

use self::beam::GenerationBeam;

#[derive(Debug, Clone)]
pub enum Rule {
    Start(NodeIndex),
    Scan(NodeIndex),
    Unmerge {
        child: NodeIndex,
        complement: NodeIndex,
    },
    UnmergeFromMover {
        child: NodeIndex,
        complement: NodeIndex,
    },
    Unmove {
        child: NodeIndex,
        stored: NodeIndex,
    },
    UnmoveFromMover {
        child: NodeIndex,
        stored: NodeIndex,
    },
}

fn clone_push<T: Clone>(v: &[T], x: T) -> Vec<T> {
    let mut v = v.to_vec();
    v.push(x);
    v
}

fn scan<T: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<ParseBeam<T>>,
    beam: &Beam,
    s: &Option<T>,
    child_node: NodeIndex,
    child_prob: f64,
) {
    if let Some(x) = beam.sentence().first() {
        if let Some(s) = s {
            if s == x {
                v.push(B::new(
                    beam.log_probability() + child_prob,
                    beam.queue().clone(),
                    beam.sentence()[1..].to_vec(),
                    clone_push(&beam.rules(), Rule::Scan(child_node)),
                ))
            }
        } else {
            //Invisible word to scan
            v.push(B::new(
                beam.log_probability() + child_prob,
                beam.queue().clone(),
                beam.sentence().to_vec(),
                clone_push(&beam.rules(), Rule::Scan(child_node)),
            ));
        }
    };
}
fn reverse_scan<T: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<GenerationBeam<T>>,
    beam: &GenerationBeam<T>,
    s: &Option<T>,
    child_node: NodeIndex,
    child_prob: f64,
) {
    if let Some(s) = s {
        let mut sentence = beam.sentence.clone();
        sentence.push(s.clone());
        v.push(GenerationBeam {
            log_probability: beam.log_probability() + child_prob,
            queue: beam.queue().clone(),
            sentence,
            rules: clone_push(&beam.rules(), Rule::Scan(child_node)),
        });
    } else {
        //Invisible word to scan
        v.push(GenerationBeam {
            queue: beam.queue.clone(),
            log_probability: beam.log_probability + child_prob,
            sentence: beam.sentence.clone(),
            rules: clone_push(&beam.rules, Rule::Scan(child_node)),
        })
    };
}

#[allow(clippy::too_many_arguments)]
fn unmerge_from_mover<
    B: Beam<T>,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug,
>(
    v: &mut Vec<B>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &B,
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
                    let mut queue = beam.queue().clone();
                    queue.push(Reverse(ParseMoment {
                        tree: FutureTree {
                            node: child_node,
                            index: moment.tree.index.clone(),
                        },
                        movers: vec![],
                    }));
                    queue.push(Reverse(ParseMoment {
                        tree: FutureTree {
                            node: stored_child_node,
                            index: moment.movers[mover_id].index.clone(),
                        },
                        movers: moment
                            .movers
                            .iter()
                            .enumerate()
                            .filter(|&(i, _v)| i != mover_id)
                            .map(|(_, v)| v.clone())
                            .collect(),
                    }));
                    v.push(B::new(
                        beam.log_probability() + stored_prob + child_prob + rule_prob,
                        queue,
                        beam.sentence().to_vec(),
                        clone_push(
                            beam.rules(),
                            Rule::UnmergeFromMover {
                                child: child_node,
                                complement: stored_child_node,
                            },
                        ),
                    ));
                }
                _ => (),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unmerge<B: Beam<T>, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<B>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &B,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: f64,
    rule_prob: f64,
) {
    let complement = lexicon.find_category(cat.clone()).unwrap();
    let mut queue = beam.queue().clone();
    queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: complement,
            index: moment.tree.index.clone_push(*dir),
        },
        movers: match dir {
            Direction::Right => moment.movers.clone(),
            Direction::Left => vec![],
        },
    }));
    queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(dir.flip()),
        },
        movers: match dir {
            Direction::Right => vec![],
            Direction::Left => moment.movers.clone(),
        },
    }));

    v.push(B::new(
        beam.log_probability() + child_prob + rule_prob,
        queue,
        beam.sentence().to_vec(),
        clone_push(
            beam.rules(),
            Rule::Unmerge {
                child: child_node,
                complement,
            },
        ),
    ));
}

#[allow(clippy::too_many_arguments)]
fn unmove_from_mover<
    B: Beam<T>,
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug,
>(
    v: &mut Vec<B>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &B,
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
                    let mut queue = beam.queue().clone();
                    queue.push(Reverse(ParseMoment {
                        tree: FutureTree {
                            node: child_node,
                            index: moment.tree.index.clone(),
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
                            }))
                            .collect(),
                    }));
                    v.push(B::new(
                        beam.log_probability() + stored_prob + child_prob + rule_prob,
                        queue,
                        beam.sentence().to_vec(),
                        clone_push(
                            beam.rules(),
                            Rule::UnmoveFromMover {
                                child: child_node,
                                stored: stored_child_node,
                            },
                        ),
                    ));
                }
                _ => (),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unmove<B: Beam<T>, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<B>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &B,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
    rule_prob: f64,
) {
    let stored = lexicon.find_licensee(cat.clone()).unwrap();
    let mut queue = beam.queue().clone();

    queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(Direction::Right),
        },
        movers: clone_push(
            &moment.movers,
            FutureTree {
                node: stored,
                index: moment.tree.index.clone_push(Direction::Left),
            },
        ),
    }));
    v.push(B::new(
        beam.log_probability() + child_prob + rule_prob,
        queue,
        beam.sentence().to_vec(),
        clone_push(
            beam.rules(),
            Rule::Unmove {
                child: child_node,
                stored,
            },
        ),
    ));
}

pub fn expand_parse<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    moment: &'a ParseMoment,
    beam: &'a ParseBeam<T>,
    lexicon: &'a Lexicon<T, Category>,
    probability_of_moving: f64,
    probability_of_merging: f64,
) -> impl Iterator<Item = ParseBeam<'a, T>> + 'a {
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

    lexicon
        .children_of(moment.tree.node)
        .map(|nx| (nx, lexicon.get(nx).unwrap()))
        .flat_map(move |(child_node, (child, child_prob))| {
            let mut v: Vec<_> = vec![];
            match &child {
                FeatureOrLemma::Lemma(s) if moment.movers.is_empty() => {
                    scan(&mut v, beam, s, child_node, child_prob);
                }
                FeatureOrLemma::Feature(Feature::Selector(cat, dir)) => {
                    unmerge_from_mover(
                        &mut v,
                        lexicon,
                        moment,
                        beam,
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
                    unmerge(
                        &mut v, lexicon, moment, beam, cat, dir, child_node, child_prob, rule_prob,
                    );
                }
                FeatureOrLemma::Feature(Feature::Licensor(cat)) => {
                    unmove_from_mover(
                        &mut v,
                        lexicon,
                        moment,
                        beam,
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
                    unmove(
                        &mut v, lexicon, moment, beam, cat, child_node, child_prob, rule_prob,
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
