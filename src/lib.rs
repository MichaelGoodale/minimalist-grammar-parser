use std::collections::BinaryHeap;

use anyhow::Result;
use lexicon::FeatureOrLemma;
use lexicon::Lexicon;
use petgraph::graph::NodeIndex;
use std::cmp::Reverse;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
struct GornIndex {
    index: Vec<Direction>,
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

    fn bad_parse(&self) -> bool {
        self.queue.is_empty() && !self.sentence.is_empty()
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
        .filter_map(|child| match lexicon.get(child) {
            Some((FeatureOrLemma::Lemma(s), p)) => {
                if let Some(x) = beam.sentence.first() {
                    if x == s {
                        Some(ParseBeam::<T> {
                            queue: beam.queue.clone(),
                            log_probability: beam.log_probability + p,
                            sentence: beam.sentence[1..].to_vec(),
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => {
                panic!("THIS SHOULD BE IMPOSSIBLE!")
            }
            _ => None,
        })
    //        node = moment.node
    //    movers = moment.movers
    //    for child in node.children:
    //        if child.is_leaf() and len(movers) == 0:
    //            if child.value == "":
    //                return [copy_push(child.prob(), [], beam, ("scan", child))]
    //            elif len(beam.s) > 0 and child.value == beam.s[0]:
    //                beam = copy.copy(beam)
    //                beam.p = beam.p * child.prob()
    //                beam.rules.append(("scan", child))
    //                beam.s.pop(0)
    //                return [beam]
    //    return []
}

fn expand<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug>(
    moment: ParseMoment,
    beam: ParseBeam<T>,
    lexicon: &Lexicon<T, Category>,
) -> Vec<ParseBeam<T>> {
    scan(&moment, &beam, lexicon).collect()
}

fn parse<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug>(
    lexicon: Lexicon<T, Category>,
    initial_category: Category,
    sentence: Vec<T>,
) -> Result<()> {
    let mut parse_heap = BinaryHeap::new();
    parse_heap.push(ParseBeam::<T>::new(&lexicon, initial_category, sentence)?);

    while let Some(mut beam) = parse_heap.pop() {
        if let Some(moment) = beam.pop() {
            parse_heap.extend(expand(moment, beam, &lexicon).into_iter());
        } else {
            if beam.good_parse() {
                println!("Found parse");
            }
            break;
        }
    }
    Ok(())
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
        parse(lexicon, 'h', vec!["hello".to_string()])
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
