use std::collections::HashMap;

use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;

use crate::lexicon::Feature;
use crate::lexicon::FeatureOrLemma;
use crate::lexicon::LexicalEntry;
use crate::lexicon::Lexicon;

#[cfg(feature = "semantics")]
use crate::lexicon::SemanticLexicon;
#[cfg(feature = "semantics")]
use simple_semantics::{
    lambda::{LambdaPool, RootedLambdaPool},
    language::Expr,
};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RuleIndex(usize);
impl RuleIndex {
    pub fn one() -> Self {
        RuleIndex(1)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct TraceId(usize);

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Rule {
    Start {
        node: NodeIndex,
        child: RuleIndex,
    },
    UnmoveTrace(TraceId),
    Scan {
        node: NodeIndex,
    },
    Unmerge {
        child: NodeIndex,
        child_id: RuleIndex,
        complement_id: RuleIndex,
    },
    UnmergeFromMover {
        child: NodeIndex,
        child_id: RuleIndex,
        stored_id: RuleIndex,
        trace_id: TraceId,
    },
    Unmove {
        child_id: RuleIndex,
        stored_id: RuleIndex,
    },
    UnmoveFromMover {
        child_id: RuleIndex,
        stored_id: RuleIndex,
        trace_id: TraceId,
    },
}

impl Rule {
    fn children(&self) -> (Option<RuleIndex>, Option<RuleIndex>) {
        match self {
            Rule::Start { child, .. } => (Some(*child), None),
            Rule::Unmerge {
                child_id,
                complement_id,
                ..
            }
            | Rule::UnmergeFromMover {
                child_id,
                stored_id: complement_id,
                ..
            }
            | Rule::Unmove {
                child_id,
                stored_id: complement_id,
            }
            | Rule::UnmoveFromMover {
                child_id,
                stored_id: complement_id,
                ..
            } => (Some(*child_id), Some(*complement_id)),
            Rule::UnmoveTrace(_) | Rule::Scan { .. } => (None, None),
        }
    }

    fn to_name<T, C>(self, lex: &Lexicon<T, C>) -> String
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        match self {
            Rule::Start { .. } => "Start".to_string(),
            Rule::UnmoveTrace(trace_id) => trace_id.to_string(),
            Rule::Scan { node } => lex.get(node).unwrap().0.to_string(),
            Rule::Unmerge { .. } => "Merge".to_string(),
            Rule::UnmergeFromMover { .. } => "MergeFromMover".to_string(),
            Rule::Unmove { .. } => "Move".to_string(),
            Rule::UnmoveFromMover { .. } => "MoveFromMover".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartialRulePool {
    pool: Vec<Option<Rule>>,
    n_traces: usize,
}

impl PartialRulePool {
    pub fn fresh(&mut self) -> RuleIndex {
        let id = RuleIndex(self.pool.len()); //Get fresh ID
        self.pool.push(None);
        id
    }
    pub fn fresh_trace(&mut self) -> TraceId {
        let i = TraceId(self.n_traces);
        self.n_traces += 1;
        i
    }

    pub fn n_steps(&self) -> usize {
        self.pool.len()
    }

    pub fn push_rule(&mut self, r: Rule, r_i: RuleIndex, trace: Option<(TraceId, RuleIndex)>) {
        if let Some((trace, i)) = trace {
            self.pool[i.0] = Some(Rule::UnmoveTrace(trace));
        }
        self.pool[r_i.0] = Some(r);
    }

    pub fn start_from_category(cat: NodeIndex) -> Self {
        PartialRulePool {
            pool: vec![
                Some(Rule::Start {
                    node: cat,
                    child: RuleIndex(1),
                }),
                None,
            ],
            n_traces: 0,
        }
    }

    pub fn into_rule_pool(self) -> RulePool {
        RulePool(self.pool.into_iter().collect::<Option<Vec<_>>>().unwrap())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RulePool(Vec<Rule>);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MGEdge {
    Move,
    Merge,
}

impl std::fmt::Display for MGEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MGEdge::Move => "move",
            MGEdge::Merge => "",
        };
        write!(f, "{}", s)
    }
}

impl RulePool {
    pub fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }
    pub fn to_x_bar_graph<T, C>(&self, lex: &Lexicon<T, C>) -> DiGraph<String, MGEdge>
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let mut g = DiGraph::<String, MGEdge>::new();
        let mut trace_h = HashMap::new();
        let mut rule_h: HashMap<RuleIndex, NodeIndex> = HashMap::new();
        inner_to_x_bar_graph(&mut g, lex, self, RuleIndex(0), &mut trace_h, &mut rule_h);
        for (a, b) in trace_h.into_iter().filter_map(|(_, x)| {
            if let (Some(a), Some(b)) = x {
                Some((*rule_h.get(&a).unwrap(), b))
            } else {
                None
            }
        }) {
            g.add_edge(a, b, MGEdge::Move);
        }
        g
    }

    pub fn to_graph<T, C>(&self, lex: &Lexicon<T, C>) -> DiGraph<String, MGEdge>
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        let mut g = DiGraph::<String, MGEdge>::new();
        let mut trace_h = HashMap::new();
        let mut rule_h: HashMap<RuleIndex, NodeIndex> = HashMap::new();
        inner_to_graph(&mut g, lex, self, RuleIndex(0), &mut trace_h, &mut rule_h);
        for (a, b) in trace_h.into_iter().filter_map(|(_, x)| {
            if let (Some(a), Some(b)) = x {
                Some((*rule_h.get(&a).unwrap(), b))
            } else {
                None
            }
        }) {
            g.add_edge(a, b, MGEdge::Move);
        }
        g
    }

    #[cfg(feature = "semantics")]
    pub fn to_interpretation<T: Eq, C: Eq>(
        &self,
        lex: &SemanticLexicon<T, C>,
    ) -> anyhow::Result<RootedLambdaPool<Expr>> {
        let mut trace_h = HashMap::default();
        let (mut pool, _) = inner_interpretation(self, lex, RuleIndex(0), &mut trace_h);
        pool.reduce()?;
        Ok(pool)
    }
}

#[cfg(feature = "semantics")]
fn inner_interpretation<T: Eq, C: Eq>(
    rules: &RulePool,
    lex: &SemanticLexicon<T, C>,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, RootedLambdaPool<Expr>>,
) -> (RootedLambdaPool<Expr>, Option<TraceId>) {
    let rule = rules.get(index);
    match rule {
        Rule::Scan { node } => (lex.interpretation(*node).clone(), None),
        Rule::Start { child, .. } => inner_interpretation(rules, lex, *child, trace_h),
        Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => {
            let complement = inner_interpretation(rules, lex, *complement_id, trace_h).0;
            let child = inner_interpretation(rules, lex, *child_id, trace_h).0;
            (child.merge(complement).unwrap(), None)
        }
        Rule::UnmoveTrace(trace_id) => (trace_h.remove(trace_id).unwrap(), Some(*trace_id)),
        Rule::UnmergeFromMover {
            child_id,
            stored_id,
            trace_id,
            ..
        } => {
            let child = inner_interpretation(rules, lex, *child_id, trace_h).0;
            let stored_value = inner_interpretation(rules, lex, *stored_id, trace_h).0;
            trace_h.insert(*trace_id, stored_value);

            dbg!(trace_h);
            //TODO: Actually apply the novel abstraction
            (child, None)
        }
        Rule::Unmove {
            child_id,
            stored_id,
        } =>
        //We add the lambda extraction to child_id
        {
            let child = inner_interpretation(rules, lex, *child_id, trace_h).0;
            let (stored_value, trace_id) = inner_interpretation(rules, lex, *stored_id, trace_h);
            dbg!(child, trace_id, stored_id);
            todo!()
        }
        Rule::UnmoveFromMover {
            child_id,
            stored_id,
            trace_id,
        } => todo!(),
    }
}
fn inner_to_x_bar_graph<T, C>(
    g: &mut DiGraph<String, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
) -> (NodeIndex, Vec<Feature<C>>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let (node, features) = match rules.get(index) {
        Rule::UnmoveTrace(trace_id) => {
            let node = g.add_node(trace_id.to_string());
            trace_h.entry(*trace_id).or_default().1 = Some(node);
            (node, vec![])
        }
        Rule::UnmergeFromMover {
            trace_id,
            stored_id,
            ..
        }
        | Rule::UnmoveFromMover {
            trace_id,
            stored_id,
            ..
        } => {
            trace_h.entry(*trace_id).or_default().0 = Some(*stored_id);

            (g.add_node(trace_id.to_string()), vec![])
        }
        Rule::Scan { node } => {
            let lexeme = lex.get_lexical_entry(*node).unwrap();

            let mut node = g.add_node(match lexeme.lemma {
                Some(x) => x.to_string(),
                None => "ε".to_string(),
            });

            if let Some(Feature::Category(c)) = lexeme.features.first() {
                let parent = g.add_node(format!("{c}P"));
                g.add_edge(parent, node, MGEdge::Merge);
                node = parent;
            }

            (node, lexeme.features)
        }
        Rule::Unmove {
            child_id,
            stored_id: complement_id,
        }
        | Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => {
            let (child_node, mut child_features) =
                inner_to_x_bar_graph(g, lex, rules, *child_id, trace_h, rules_h);
            let current_feature = child_features.pop();

            let (complement_node, _complement_features) =
                inner_to_x_bar_graph(g, lex, rules, *complement_id, trace_h, rules_h);

            let node = g.add_node(
                current_feature
                    .map(FeatureOrLemma::Feature)
                    .unwrap()
                    .to_string(),
            );

            g.add_edge(node, child_node, MGEdge::Merge);
            g.add_edge(node, complement_node, MGEdge::Merge);

            (node, child_features)
        }
        Rule::Start { child, .. } => inner_to_x_bar_graph(g, lex, rules, *child, trace_h, rules_h),
    };
    rules_h.insert(index, node);

    (node, features)
}

fn inner_to_graph<T, C>(
    g: &mut DiGraph<String, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
) -> NodeIndex
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone,
    C: Eq + std::fmt::Debug + std::clone::Clone,
{
    let rule = rules.get(index);
    let node = g.add_node(rule.to_name(lex));
    rules_h.insert(index, node);

    match rule {
        Rule::UnmoveTrace(trace_id) => trace_h.entry(*trace_id).or_default().1 = Some(node),
        Rule::UnmergeFromMover {
            trace_id,
            stored_id,
            ..
        }
        | Rule::UnmoveFromMover {
            trace_id,
            stored_id,
            ..
        } => trace_h.entry(*trace_id).or_default().0 = Some(*stored_id),
        Rule::Unmove { .. } | Rule::Start { .. } | Rule::Scan { .. } | Rule::Unmerge { .. } => (),
    };

    let (child_a, child_b) = rule.children();
    if let Some(child_a) = child_a {
        let child = inner_to_graph(g, lex, rules, child_a, trace_h, rules_h);
        g.add_edge(node, child, MGEdge::Merge);
    };
    if let Some(child_b) = child_b {
        let child = inner_to_graph(g, lex, rules, child_b, trace_h, rules_h);
        g.add_edge(node, child, MGEdge::Merge);
    };
    node
}

#[cfg(test)]
mod test {
    use logprob::LogProb;

    use super::*;
    use crate::grammars::STABLER2011;
    use crate::{Parser, ParsingConfig};
    use petgraph::dot::Dot;

    #[test]
    fn to_graph() -> anyhow::Result<()> {
        let lex = Lexicon::parse(STABLER2011)?;
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        for sentence in vec![
            "the king drinks the beer",
            "which wine the queen prefers",
            "which queen prefers the wine",
            "the queen knows the king drinks the beer",
            "the queen knows the king knows the queen drinks the beer",
        ]
        .into_iter()
        {
            let (_, _, rules) =
                Parser::new(&lex, "C", &sentence.split(' ').collect::<Vec<_>>(), &config)?
                    .next()
                    .unwrap();
            let g = rules.to_graph(&lex);
            let _dot = Dot::new(&g);
        }
        //TODO: Decide on a formatting and stick with it.
        Ok(())
    }

    #[test]
    fn no_movement_xbar() -> anyhow::Result<()> {
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lex = Lexicon::parse(STABLER2011)?;
        let rules = Parser::new(
            &lex,
            "C",
            &"the queen prefers the wine".split(' ').collect::<Vec<_>>(),
            &config,
        )?
        .next()
        .unwrap()
        .2;
        let parse = rules.to_x_bar_graph(&lex);
        let dot = Dot::new(&parse);
        println!("{}", dot.to_string());
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

    #[test]
    fn tree_building() -> anyhow::Result<()> {
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lex = Lexicon::parse(STABLER2011)?;
        let rules = Parser::new(
            &lex,
            "C",
            &"which queen prefers the wine"
                .split(' ')
                .collect::<Vec<_>>(),
            &config,
        )?
        .next()
        .unwrap()
        .2;
        let parse = rules.to_x_bar_graph(&lex);
        let dot = Dot::new(&parse);
        println!("{}", dot.to_string());
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
