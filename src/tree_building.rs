use std::collections::HashMap;
use std::fmt::Display;

use crate::lexicon::{FeatureOrLemma, Lexicon};
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
    let mut g = DiGraph::<String, Empty>::new();
    let mut id2node = HashMap::new();
    for rule in rules.iter() {
        match rule {
            Rule::Start(nx) => {
                let c = lexicon.get(*nx).unwrap().0;
                id2node.insert(0, g.add_node(format!("{c}P").to_string()));
            }
            Rule::Unmerge {
                parent,
                child_id,
                complement_id,
                child,
            } => {
                let child = lexicon.get_category(*child).unwrap().to_string();
                let parent_cat = &g[id2node[parent]];
                let child = if parent_cat.as_str() == format!("{}P", &child) {
                    let child = g.add_node(format!("{child}'"));
                    g.add_edge(id2node[parent], child, Empty {});
                    child
                } else if parent_cat.as_str() == "temp" {
                    let p = id2node[parent];
                    g[p] = format!("{child}P");
                    p
                } else {
                    let child = g.add_node(format!("{child}P"));
                    g.add_edge(id2node[parent], child, Empty {});
                    child
                };

                id2node.insert(*child_id, child);
                id2node.insert(*complement_id, child);
            }
            Rule::UnmergeFromMover {
                child,
                storage,
                parent,
                child_id,
                stored_id,
            } => {
                let child = lexicon.get_category(*child).unwrap().to_string();
                let parent_cat = &g[id2node[parent]];

                let child = if parent_cat.as_str() == format!("{}P", &child) {
                    let child = g.add_node(format!("{child}'"));
                    g.add_edge(id2node[parent], child, Empty {});
                    child
                } else if parent_cat.as_str() == "temp" {
                    let p = id2node[parent];
                    g[p] = format!("{child}P");
                    p
                } else {
                    let child = g.add_node(format!("{child}P"));
                    g.add_edge(id2node[parent], child, Empty {});
                    child
                };
                id2node.insert(*child_id, child);
                id2node.insert(*stored_id, id2node[storage]);

                let trace = g.add_node("t".to_string());
                g.add_edge(child, trace, Empty {});
                g.add_edge(trace, id2node[storage], Empty {});
            }
            Rule::UnmoveFromMover {
                parent,
                child_id,
                stored_id,
            } => {
                id2node.insert(*child_id, id2node[parent]);
                let temp = g.add_node("temp".to_string());
                g.add_edge(id2node[parent], temp, Empty {});
                id2node.insert(*stored_id, temp);
            }
            Rule::Unmove {
                child_id,
                stored_id,
                parent,
            } => {
                id2node.insert(*child_id, id2node[parent]);
                let temp = g.add_node("temp".to_string());
                g.add_edge(id2node[parent], temp, Empty {});
                id2node.insert(*stored_id, temp);
            }
            Rule::Scan { node, parent } => {
                let parent_category = lexicon.get_category(*node).unwrap();
                let node = lexicon.get(*node).unwrap().0;
                let node = g.add_node(node.to_string());
                let parent_node = g.add_node(parent_category.to_string());

                if g[id2node[parent]] != format!("{parent_category}P")
                    && g[id2node[parent]] != format!("{parent_category}'")
                {
                    let p = g.add_node(format!("{parent_category}P"));
                    g.add_edge(id2node[parent], p, Empty {});
                    g.add_edge(p, parent_node, Empty {});
                } else {
                    g.add_edge(id2node[parent], parent_node, Empty {});
                }

                g.add_edge(parent_node, node, Empty {});
            }
        }
    }
    //g.remove_node(id2node[&0]);
    let d = Dot::new(&g);
    println!("{}", d);
}
