use std::collections::HashMap;
use std::fmt::Display;

use crate::lexicon::{FeatureOrLemma, Lexicon};
use crate::parsing::Rule;

use petgraph::graph::DiGraph;

pub struct Empty {}
impl Display for Empty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

pub fn build_tree<T, C>(lexicon: &Lexicon<T, C>, rules: &[Rule]) -> DiGraph<String, Empty>
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone,
    C: Eq + std::fmt::Debug + std::clone::Clone,
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
                storage,
            } => {
                id2node.insert(*child_id, id2node[parent]);
                id2node.insert(*stored_id, id2node[storage]);
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

                if g[id2node[parent]] == "temp" {
                    g[id2node[parent]] = format!("{parent_category}P");
                    g.add_edge(id2node[parent], parent_node, Empty {});
                } else if g[id2node[parent]] != format!("{parent_category}P")
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
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::STABLER2011;
    use crate::lexicon::Lexicon;
    use crate::lexicon::SimpleLexicalEntry;
    use crate::{Parser, ParsingConfig};
    use anyhow::Result;
    use petgraph::dot::Dot;
    use std::f64::consts::LN_2;

    const CONFIG: ParsingConfig = ParsingConfig {
        min_log_prob: -64.0,
        merge_log_prob: -LN_2,
        move_log_prob: -LN_2,
        max_steps: 10000,
        max_beams: 100,
    };

    #[test]
    fn tree_building() -> Result<()> {
        let v: Vec<_> = STABLER2011
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()?;
        let lex = Lexicon::new(v);
        let rules = Parser::new(
            &lex,
            'C',
            &"which queen prefers the wine"
                .split(' ')
                .collect::<Vec<_>>(),
            &CONFIG,
        )?
        .next()
        .unwrap()
        .1;
        let parse = build_tree(&lex, &rules);
        let dot = Dot::new(&parse);
        assert_eq!(
            dot.to_string(),
            "digraph {
    0 [ label = \"CP\" ]
    1 [ label = \"DP\" ]
    2 [ label = \"C'\" ]
    3 [ label = \"VP\" ]
    4 [ label = \"t\" ]
    5 [ label = \"which\" ]
    6 [ label = \"D\" ]
    7 [ label = \"queen\" ]
    8 [ label = \"N\" ]
    9 [ label = \"NP\" ]
    10 [ label = \"ε\" ]
    11 [ label = \"C\" ]
    12 [ label = \"V'\" ]
    13 [ label = \"prefers\" ]
    14 [ label = \"V\" ]
    15 [ label = \"DP\" ]
    16 [ label = \"the\" ]
    17 [ label = \"D\" ]
    18 [ label = \"wine\" ]
    19 [ label = \"N\" ]
    20 [ label = \"NP\" ]
    0 -> 1 [ label = \"\" ]
    0 -> 2 [ label = \"\" ]
    2 -> 3 [ label = \"\" ]
    3 -> 4 [ label = \"\" ]
    4 -> 1 [ label = \"\" ]
    1 -> 6 [ label = \"\" ]
    6 -> 5 [ label = \"\" ]
    1 -> 9 [ label = \"\" ]
    9 -> 8 [ label = \"\" ]
    8 -> 7 [ label = \"\" ]
    2 -> 11 [ label = \"\" ]
    11 -> 10 [ label = \"\" ]
    3 -> 12 [ label = \"\" ]
    12 -> 14 [ label = \"\" ]
    14 -> 13 [ label = \"\" ]
    12 -> 15 [ label = \"\" ]
    15 -> 17 [ label = \"\" ]
    17 -> 16 [ label = \"\" ]
    15 -> 20 [ label = \"\" ]
    20 -> 19 [ label = \"\" ]
    19 -> 18 [ label = \"\" ]
}
"
        );
        Ok(())
    }
}
