use crate::lexicon::{Feature, FeatureOrLemma, Lexicon};
use crate::Direction;
use beam::ParseBeam;
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;
use trees::{clone_push, FutureTree, ParseMoment};

fn scan<T: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<ParseBeam<T>>,
    beam: &ParseBeam<T>,
    s: &Option<T>,
    child_prob: f64,
) {
    if let Some(x) = beam.sentence.first() {
        if let Some(s) = s {
            if s == x {
                v.push(ParseBeam::<T> {
                    queue: beam.queue.clone(),
                    log_probability: beam.log_probability + child_prob,
                    sentence: beam.sentence[1..].to_vec(),
                })
            }
        } else {
            //Invisible word to scan
            v.push(ParseBeam::<T> {
                queue: beam.queue.clone(),
                log_probability: beam.log_probability + child_prob,
                sentence: beam.sentence.clone(),
            })
        }
    };
}
fn reverse_scan<T: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<ParseBeam<T>>,
    beam: &ParseBeam<T>,
    s: &Option<T>,
    child_prob: f64,
) {
    if let Some(s) = s {
        let mut sentence = beam.sentence.clone();
        sentence.push(s.clone());
        v.push(ParseBeam::<T> {
            queue: beam.queue.clone(),
            log_probability: beam.log_probability + child_prob,
            sentence,
        })
    } else {
        //Invisible word to scan
        v.push(ParseBeam::<T> {
            queue: beam.queue.clone(),
            log_probability: beam.log_probability + child_prob,
            sentence: beam.sentence.clone(),
        })
    };
}

fn unmerge_from_mover<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug>(
    v: &mut Vec<ParseBeam<T>>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &ParseBeam<T>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
) {
    for mover_id in 0..moment.movers.len() {
        for stored_child_node in lexicon.children_of(moment.movers[mover_id].node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Category(stored)) if stored == cat => {
                    let mut queue = beam.queue.clone();
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
                    v.push(ParseBeam {
                        log_probability: beam.log_probability + stored_prob + child_prob
                            - 2_f64.ln(),
                        queue,
                        sentence: beam.sentence.clone(),
                    });
                }
                _ => (),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unmerge<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<ParseBeam<T>>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &ParseBeam<T>,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: f64,
) {
    let complement = lexicon.find_category(cat.clone()).unwrap();
    let mut queue = beam.queue.clone();
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

    v.push(ParseBeam::<T> {
        queue,
        log_probability: beam.log_probability + child_prob,
        sentence: beam.sentence.clone(),
    });
}

fn unmove_from_mover<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug>(
    v: &mut Vec<ParseBeam<T>>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &ParseBeam<T>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
) {
    for mover_id in 0..moment.movers.len() {
        for stored_child_node in lexicon.children_of(moment.movers[mover_id].node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Licensee(s)) if cat == s => {
                    let mut queue = beam.queue.clone();
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
                    v.push(ParseBeam {
                        log_probability: beam.log_probability + stored_prob + child_prob,
                        queue,
                        sentence: beam.sentence.clone(),
                    });
                }
                _ => (),
            }
        }
    }
}

fn unmove<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    v: &mut Vec<ParseBeam<T>>,
    lexicon: &Lexicon<T, Category>,
    moment: &ParseMoment,
    beam: &ParseBeam<T>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: f64,
) {
    let stored = lexicon.find_licensee(cat.clone()).unwrap();
    let mut queue = beam.queue.clone();

    queue.push(Reverse(ParseMoment {
        tree: FutureTree {
            node: child_node,
            index: moment.tree.index.clone_push(Direction::Right),
        },
        movers: clone_push(
            &moment.movers,
            stored,
            moment.tree.index.clone_push(Direction::Left),
        ),
    }));
    v.push(ParseBeam::<T> {
        log_probability: beam.log_probability + child_prob,
        queue,
        sentence: beam.sentence.clone(),
    });
}

pub fn expand<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    moment: &'a ParseMoment,
    beam: &'a ParseBeam<T>,
    lexicon: &'a Lexicon<T, Category>,
    use_reverse_scan: bool,
) -> impl Iterator<Item = ParseBeam<T>> + 'a {
    lexicon
        .children_of(moment.tree.node)
        .map(|nx| (nx, lexicon.get(nx).unwrap()))
        .flat_map(move |(child_node, (child, child_prob))| {
            let mut v: Vec<_> = vec![];
            match &child {
                FeatureOrLemma::Lemma(s) if moment.movers.is_empty() => {
                    if use_reverse_scan {
                        reverse_scan(&mut v, beam, s, child_prob);
                    } else {
                        scan(&mut v, beam, s, child_prob);
                    }
                }
                FeatureOrLemma::Feature(Feature::Selector(cat, dir)) => {
                    unmerge_from_mover(&mut v, lexicon, moment, beam, cat, child_node, child_prob);
                    unmerge(
                        &mut v, lexicon, moment, beam, cat, dir, child_node, child_prob,
                    );
                }
                FeatureOrLemma::Feature(Feature::Licensor(cat)) => {
                    unmove_from_mover(&mut v, lexicon, moment, beam, cat, child_node, child_prob);
                    unmove(&mut v, lexicon, moment, beam, cat, child_node, child_prob);
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
