use std::collections::HashMap;
use std::fmt::Display;

use crate::lexicon::Feature;
use crate::lexicon::FeatureOrLemma;
use crate::lexicon::Lexicon;
use crate::parsing::Rule;

use petgraph::dot::Dot;
use petgraph::graph::DiGraph;

struct Empty {}
impl Display for Empty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

pub fn build_tree<T, C>(lexicon: &Lexicon<T, C>, rules: &[Rule])
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug,
    C: Eq + std::fmt::Debug,
{
    let mut g = DiGraph::<&FeatureOrLemma<T, C>, Empty>::new();
    let mut id2node = HashMap::new();
    for rule in rules.iter() {
        match rule {
            Rule::Start(nx) => {
                let c = lexicon.get(*nx).unwrap().0;
                id2node.insert(0, g.add_node(c));
            }
            Rule::Unmerge {
                parent,
                child_id,
                complement_id,
                child,
                ..
            } => {
                let child = lexicon.get_category(*child).unwrap();
                let child = g.add_node(child);
                id2node.insert(*child_id, child);
                id2node.insert(*complement_id, child);
                g.add_edge(id2node[parent], child, Empty {});
            }
            Rule::UnmergeFromMover {
                child,
                complement,
                storage,
                parent,
                child_id,
                stored_id,
            } => {
                let child = lexicon.get(*child).unwrap().0;
                let complement = lexicon.get(*complement).unwrap().0;
                let child = g.add_node(child);
                let complement = g.add_node(complement);
                id2node.insert(*child_id, child);
                id2node.insert(*stored_id, child);
                g.add_edge(id2node[storage], complement, Empty {});
                g.add_edge(id2node[parent], child, Empty {});
                g.add_edge(id2node[parent], complement, Empty {});
            }
            Rule::UnmoveFromMover {
                child,
                stored,
                storage,
                parent,
                child_id,
                stored_id,
            } => {
                let child = lexicon.get(*child).unwrap().0;
                let complement = lexicon.get(*stored).unwrap().0;
                let child = g.add_node(child);
                let complement = g.add_node(complement);
                id2node.insert(*child_id, child);
                id2node.insert(*stored_id, child);
                g.add_edge(id2node[storage], complement, Empty {});
                g.add_edge(id2node[parent], child, Empty {});
                g.add_edge(id2node[parent], complement, Empty {});
            }
            Rule::Unmove {
                child,
                stored,
                child_id,
                stored_id,
                parent,
            } => {
                let child = lexicon.get(*child).unwrap().0;
                let complement = lexicon.get(*stored).unwrap().0;
                let child = g.add_node(child);
                let complement = g.add_node(complement);
                id2node.insert(*child_id, child);
                id2node.insert(*stored_id, child);
                g.add_edge(id2node[parent], child, Empty {});
                g.add_edge(id2node[parent], complement, Empty {});
            }
            Rule::Scan { node, parent } => {
                let parent_category = lexicon.get_category(*node).unwrap();
                let node = lexicon.get(*node).unwrap().0;
                let node = g.add_node(node);
                let parent_category = g.add_node(parent_category);
                g.add_edge(id2node[parent], parent_category, Empty {});
                g.add_edge(parent_category, node, Empty {});
            }
        }
    }
    g.remove_node(id2node[&0]);
    let d = Dot::new(&g);
    println!("{}", d);
}
