use std::fmt::Debug;

use super::{Feature, FeatureOrLemma, Lexicon};
use ahash::AHashSet;
use chumsky::prelude::todo;
use petgraph::{
    Direction::{Incoming, Outgoing},
    graph::NodeIndex,
    visit::EdgeCount,
};

#[derive(Debug)]
struct AccessibilityChecker<'a, T: Eq, Category: Eq> {
    stack: Vec<NodeIndex>,
    seen: AHashSet<NodeIndex>,
    unsatisfiable: AHashSet<NodeIndex>,
    lex: &'a Lexicon<T, Category>,
}

impl<'a, T, C> AccessibilityChecker<'a, T, C>
where
    T: Eq + Debug + Clone,
    C: Eq + Debug + Clone,
{
    fn new(node: NodeIndex, lex: &'a Lexicon<T, C>) -> Self {
        Self {
            stack: lex.graph.neighbors_directed(node, Outgoing).collect(),
            seen: [lex.root, node].into_iter().collect(),
            unsatisfiable: AHashSet::default(),
            lex,
        }
    }

    fn pop(&mut self) -> Option<NodeIndex> {
        match self.stack.pop() {
            Some(x) => {
                self.seen.insert(x);
                Some(x)
            }
            None => None,
        }
    }

    fn add_direct_children(&mut self, node: NodeIndex) {
        self.stack.extend(
            self.lex
                .graph
                .neighbors_directed(node, Outgoing)
                .filter(|x| !self.seen.contains(x)),
        );
    }

    fn add_indirect_children(&mut self, node: NodeIndex) {
        match self.lex.graph.node_weight(node).unwrap() {
            FeatureOrLemma::Feature(Feature::Selector(c, _)) | FeatureOrLemma::Complement(c, _) => {
                match self.lex.find_category(c) {
                    Ok(x) => {
                        self.stack.push(x);
                    }
                    Err(_) => {
                        self.unsatisfiable.insert(node);
                    }
                }
            }
            FeatureOrLemma::Feature(Feature::Licensor(c)) => {
                match self.lex.find_licensee(c) {
                    //Does not check if the licensee can be built
                    //internally  (e.g. if we have type z= w+ c but the -w is not in buildable
                    //starting from z.
                    Ok(x) => {
                        self.stack.push(x);
                    }
                    Err(_) => {
                        self.unsatisfiable.insert(node);
                    }
                }
            }
            FeatureOrLemma::Root
            | FeatureOrLemma::Lemma(_)
            | FeatureOrLemma::Feature(Feature::Licensee(_))
            | FeatureOrLemma::Feature(Feature::Category(_)) => (),
        }
    }
}

impl<T, C> Lexicon<T, C>
where
    T: Eq + Debug + Clone,
    C: Eq + Debug + Clone,
{
    pub fn prune(&mut self, start: &C) {
        loop {
            let start = match self.find_category(start) {
                Ok(x) => x,
                Err(_) => {
                    self.graph.retain_nodes(|g, n| {
                        matches!(g.node_weight(n).unwrap(), FeatureOrLemma::Root)
                    });
                    self.leaves.clear();
                    return;
                }
            };
            let mut checker = AccessibilityChecker::new(start, self);

            while let Some(node) = checker.pop() {
                checker.add_direct_children(node);
                checker.add_indirect_children(node);
            }

            if checker.unsatisfiable.is_empty() && checker.seen.len() == self.graph.node_count()
            //-1 since we're not gonna see the root
            {
                break;
            } else {
                self.graph.retain_nodes(|_, n| {
                    checker.seen.contains(&n) & !checker.unsatisfiable.contains(&n)
                });
            }
        }

        self.leaves = self
            .graph
            .node_indices()
            .filter_map(|x| {
                if matches!(self.graph.node_weight(x).unwrap(), FeatureOrLemma::Lemma(_)) {
                    let w = self
                        .graph
                        .edges_directed(x, Incoming)
                        .next()
                        .unwrap()
                        .weight();
                    Some((x, w.into_inner()))
                } else {
                    None
                }
            })
            .collect();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pruning() -> anyhow::Result<()> {
        let mut lex = Lexicon::parse("A::c= s\nB::d\nC::c")?;
        lex.prune(&"s");
        dbg!(lex);
        panic!();
        Ok(())
    }
}
