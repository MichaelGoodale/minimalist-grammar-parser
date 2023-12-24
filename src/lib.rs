use std::collections::BinaryHeap;

use anyhow::Result;
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

struct ParseBeam {
    log_probability: f64,
    queue: BinaryHeap<Reverse<ParseMoment>>,
}

impl ParseBeam {
    fn new<T: Eq + std::fmt::Debug, Category: Eq + std::fmt::Debug>(
        lexicon: Lexicon<T, Category>,
        initial_category: Category,
    ) -> Result<ParseBeam> {
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
            queue,
        })
    }
}

mod grammars;
pub mod lexicon;
#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

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
