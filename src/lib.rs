use std::collections::BinaryHeap;

use anyhow::{bail, Result};
use lexicon::Lexicon;
use lexicon::{Feature, FeatureOrLemma};
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    Left,
    Right,
}

impl Direction {
    fn flip(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
struct GornIndex {
    index: Vec<Direction>,
}

impl GornIndex {
    fn clone_push(&self, d: Direction) -> Self {
        let mut v = self.clone();
        v.index.push(d);
        v
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FutureTree {
    node: NodeIndex,
    index: GornIndex,
}

impl PartialOrd for FutureTree {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FutureTree {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.index.cmp(&other.index) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.node.cmp(&other.node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParseMoment {
    tree: FutureTree,
    movers: Vec<FutureTree>,
}

fn clone_push(v: &[FutureTree], node: NodeIndex, index: GornIndex) -> Vec<FutureTree> {
    let mut v = v.to_vec();
    v.push(FutureTree { node, index });
    v
}

impl ParseMoment {
    fn least_index(&self) -> &GornIndex {
        let mut least = &self.tree.index;
        for m in self.movers.iter() {
            if m.index < *least {
                least = &m.index
            }
        }
        least
    }
}

impl PartialOrd for ParseMoment {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ParseMoment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.least_index().cmp(other.least_index())
    }
}

#[derive(Debug, Clone)]
struct ParseBeam<T> {
    log_probability: f64,
    queue: BinaryHeap<Reverse<ParseMoment>>,
    sentence: Vec<T>,
}

impl<T: Eq + std::fmt::Debug> PartialEq for ParseBeam<T> {
    fn eq(&self, other: &Self) -> bool {
        self.log_probability == other.log_probability
            && self.sentence == other.sentence
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}
impl<T: Eq + std::fmt::Debug> PartialOrd for ParseBeam<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug> Eq for ParseBeam<T> {}

impl<T: Eq + std::fmt::Debug> Ord for ParseBeam<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_probability
            .partial_cmp(&other.log_probability)
            .unwrap()
    }
}

impl<T: Eq + std::fmt::Debug> ParseBeam<T> {
    fn new<Category: Eq + std::fmt::Debug>(
        lexicon: &Lexicon<T, Category>,
        initial_category: Category,
        sentence: Vec<T>,
    ) -> Result<ParseBeam<T>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(initial_category)?;

        queue.push(Reverse(ParseMoment {
            tree: FutureTree {
                node: category_index,
                index: GornIndex::default(),
            },
            movers: vec![],
        }));

        Ok(ParseBeam {
            log_probability: 0_f64,
            sentence,
            queue,
        })
    }

    fn good_parse(&self) -> bool {
        self.queue.is_empty() && self.sentence.is_empty()
    }
    fn pop(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }
}

fn scan<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug>(
    moment: &'_ ParseMoment,
    beam: &'a ParseBeam<T>,
    lexicon: &'a Lexicon<T, Category>,
) -> impl Iterator<Item = ParseBeam<T>> + 'a {
    lexicon
        .children_of(moment.tree.node)
        .map(|child| lexicon.get(child).unwrap())
        .filter_map(|(child, child_prob)| match child {
            FeatureOrLemma::Lemma(s) => {
                if let Some(x) = beam.sentence.first() {
                    if let Some(s) = s {
                        if s == x {
                            Some(ParseBeam::<T> {
                                queue: beam.queue.clone(),
                                log_probability: beam.log_probability + child_prob,
                                sentence: beam.sentence[1..].to_vec(),
                            })
                        } else {
                            None
                        }
                    } else {
                        //Invisible word to scan
                        Some(ParseBeam::<T> {
                            queue: beam.queue.clone(),
                            log_probability: beam.log_probability + child_prob,
                            sentence: beam.sentence.clone(),
                        })
                    }
                } else {
                    None
                }
            }
            _ => None,
        })
}

fn unmerge<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>(
    moment: &'a ParseMoment,
    beam: &'a ParseBeam<T>,
    lexicon: &'a Lexicon<T, Category>,
) -> impl Iterator<Item = ParseBeam<T>> + 'a {
    lexicon
        .children_of(moment.tree.node)
        .map(|nx| (nx, lexicon.get(nx).unwrap()))
        .filter_map(|(child_node, (child, child_prob))| match &child {
            FeatureOrLemma::Feature(Feature::Selector(cat, dir)) => {
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

                Some(ParseBeam::<T> {
                    queue,
                    log_probability: beam.log_probability + child_prob,
                    sentence: beam.sentence.clone(),
                })
            }
            FeatureOrLemma::Feature(Feature::Licensor(cat)) => {
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
                Some(ParseBeam::<T> {
                    log_probability: beam.log_probability + child_prob,
                    queue,
                    sentence: beam.sentence.clone(),
                })
            }
            _ => None,
        })
}

fn expand<'a, T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone>(
    moment: &'a ParseMoment,
    beam: &'a ParseBeam<T>,
    lexicon: &'a Lexicon<T, Category>,
) -> impl Iterator<Item = ParseBeam<T>> + 'a {
    scan(moment, beam, lexicon).chain(unmerge(moment, beam, lexicon))
}

pub fn parse<T: Eq + std::fmt::Debug + Clone, Category: Eq + Clone + std::fmt::Debug>(
    lexicon: &Lexicon<T, Category>,
    initial_category: Category,
    sentence: Vec<T>,
) -> Result<()> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(ParseBeam::<T>::new(lexicon, initial_category, sentence)?);
    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(expand(&moment, &beam, lexicon));
        } else {
            if beam.good_parse() {
                println!("Found parse");
                return Ok(());
            }
            break;
        }
    }
    bail!("No parse found :(")
}

mod grammars;
pub mod lexicon;
#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::lexicon::SimpleLexicalEntry;

    use super::*;

    #[test]
    fn simple_scan() -> Result<()> {
        let v = vec![SimpleLexicalEntry::parse("hello::h")?];
        let lexicon = Lexicon::new(v);
        parse(&lexicon, 'h', vec!["hello".to_string()])
    }

    #[test]
    fn simple_merge() -> Result<()> {
        let v = vec![
            SimpleLexicalEntry::parse("the::n= d")?,
            SimpleLexicalEntry::parse("man::n")?,
            SimpleLexicalEntry::parse("drinks::d= =d v")?,
            SimpleLexicalEntry::parse("beer::n")?,
        ];
        let lexicon = Lexicon::new(v);
        parse(&lexicon, 'd', vec!["the".to_string(), "man".to_string()])?;
        parse(
            &lexicon,
            'v',
            "the man drinks the beer"
                .split(' ')
                .map(|x| x.to_string())
                .collect(),
        )?;
        assert!(parse(
            &lexicon,
            'd',
            "drinks the man the beer"
                .split(' ')
                .map(|x| x.to_string())
                .collect(),
        )
        .is_err());

        Ok(())
    }

    #[test]
    fn index_order() -> Result<()> {
        let a = GornIndex {
            index: vec![Direction::Left, Direction::Left, Direction::Left],
        };
        let b = GornIndex {
            index: vec![Direction::Left, Direction::Left, Direction::Right],
        };
        assert!(a < b);

        let b = GornIndex {
            index: vec![Direction::Left, Direction::Left],
        };

        assert!(b < a);
        let b = GornIndex {
            index: vec![Direction::Right],
        };
        assert!(a < b);

        let a = ParseMoment {
            tree: FutureTree {
                node: NodeIndex::new(0),
                index: GornIndex {
                    index: vec![Direction::Left, Direction::Right],
                },
            },
            movers: vec![FutureTree {
                node: NodeIndex::new(0),
                index: GornIndex {
                    index: vec![Direction::Right],
                },
            }],
        };

        assert_eq!(
            a.least_index(),
            &GornIndex {
                index: vec![Direction::Left, Direction::Right]
            }
        );

        Ok(())
    }
}
