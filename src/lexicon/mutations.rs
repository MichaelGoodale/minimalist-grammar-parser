use std::fmt::Debug;

use super::{Feature, FeatureOrLemma, Lexicon};
use ahash::AHashSet;
use petgraph::{
    Direction::{Incoming, Outgoing},
    graph::NodeIndex,
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

    ///Climb up a lexeme marking it unsatisfiable until you reach a branch.
    fn mark_unsatisfiable(&mut self, mut node: NodeIndex) {
        self.unsatisfiable.insert(node);
        let get_parent = |node| self.lex.graph.neighbors_directed(node, Incoming).next();
        while let Some(parent) = get_parent(node) {
            //Check for sisters
            if self.lex.graph.neighbors_directed(node, Outgoing).count() > 1 {
                break;
            } else if !matches!(
                self.lex.graph.node_weight(parent).unwrap(),
                FeatureOrLemma::Root
            ) {
                self.unsatisfiable.insert(parent);
            }
            node = parent;
        }
    }

    fn add_indirect_children(&mut self, node: NodeIndex) {
        match self.lex.graph.node_weight(node).unwrap() {
            FeatureOrLemma::Feature(Feature::Selector(c, _)) | FeatureOrLemma::Complement(c, _) => {
                match self.lex.find_category(c) {
                    Ok(x) => {
                        if !self.seen.contains(&x) {
                            self.stack.push(x);
                        }
                    }
                    Err(_) if !self.lex.has_moving_category(c) => self.mark_unsatisfiable(node),
                    Err(_) => (),
                }
            }
            FeatureOrLemma::Feature(Feature::Licensor(c)) => {
                match self.lex.find_licensee(c) {
                    //Does not check if the licensee can be built
                    //internally  (e.g. if we have type z= w+ c but the -w is not in buildable
                    //starting from z.
                    Ok(x) => {
                        if !self.seen.contains(&x) {
                            self.stack.push(x);
                        }
                    }
                    Err(_) => self.mark_unsatisfiable(node),
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
    fn has_moving_category(&self, cat: &C) -> bool {
        let mut stack: Vec<_> = self
            .graph
            .neighbors_directed(self.root, Outgoing)
            .filter(|a| {
                matches!(
                    self.graph.node_weight(*a).unwrap(),
                    FeatureOrLemma::Feature(Feature::Licensee(_))
                )
            })
            .collect();
        while let Some(x) = stack.pop() {
            for x in self.graph.neighbors_directed(x, Outgoing) {
                match &self.graph[x] {
                    FeatureOrLemma::Feature(Feature::Licensee(_)) => stack.push(x),
                    FeatureOrLemma::Feature(Feature::Category(c)) if c == cat => return true,
                    _ => (),
                }
            }
        }
        false
    }

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
        assert_eq!(lex.to_string(), "A::c= s\nC::c");

        let mut lex = Lexicon::parse("A::z= c= s\nB::d\nC::c")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "");

        let mut lex = Lexicon::parse("A::z= c= s\nB::d\nC::d= c\nD::z")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "A::z= c= s\nB::d\nC::d= c\nD::z");
        let mut lex = Lexicon::parse("A::z= +w s\nD::z -w")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "A::z= +w s\nD::z -w");
        Ok(())
    }
}
