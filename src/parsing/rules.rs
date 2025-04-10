use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

use crate::lexicon::Feature;
use crate::lexicon::FeatureOrLemma;
use crate::lexicon::Lexicon;

#[cfg(feature = "semantics")]
use crate::lexicon::SemanticLexicon;
use crate::Direction;
#[cfg(feature = "semantics")]
use simple_semantics::{lambda::RootedLambdaPool, language::Expr};

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
    Merge(Option<Direction>),
}

impl std::fmt::Display for MGEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MGEdge::Move => "move",
            MGEdge::Merge(None) => "",
            MGEdge::Merge(Some(Direction::Left)) => "",
            MGEdge::Merge(Some(Direction::Right)) => "",
        };
        write!(f, "{}", s)
    }
}

impl RulePool {
    pub fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }
    pub fn to_x_bar_graph<T, C>(&self, lex: &Lexicon<T, C>) -> StableDiGraph<String, MGEdge>
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let mut g = StableDiGraph::<String, MGEdge>::new();
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
            let a_parent_edge = g
                .edges_directed(a, petgraph::Direction::Incoming)
                .next()
                .unwrap();
            let a_parent_node = a_parent_edge.source();

            let b_parent_edge = g
                .edges_directed(b, petgraph::Direction::Incoming)
                .next()
                .unwrap();
            let b_parent_node = b_parent_edge.source();
            let b_parent_id = b_parent_edge.id();

            let edge_weight = g.remove_edge(a_parent_edge.id()).unwrap();
            g.add_edge(a_parent_node, b, edge_weight);

            let edge_weight = g.remove_edge(b_parent_id).unwrap();
            g.add_edge(b_parent_node, a, edge_weight);

            g.add_edge(b, a, MGEdge::Move);
        }
        g
    }

    pub fn to_graph<T, C>(&self, lex: &Lexicon<T, C>) -> StableDiGraph<String, MGEdge>
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        let mut g = StableDiGraph::<String, MGEdge>::new();
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
        let (mut pool, _) = inner_interpretation(self, lex, RuleIndex(0), &mut trace_h)?;
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
) -> anyhow::Result<(RootedLambdaPool<Expr>, Option<TraceId>)> {
    let rule = rules.get(index);
    Ok(match rule {
        Rule::Scan { node } => (lex.interpretation(*node).clone(), None),
        Rule::Start { child, .. } => inner_interpretation(rules, lex, *child, trace_h)?,
        Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => {
            let complement = inner_interpretation(rules, lex, *complement_id, trace_h)?.0;
            let child = inner_interpretation(rules, lex, *child_id, trace_h)?.0;
            let merged = child.merge(complement).unwrap();
            (merged, None)
        }
        Rule::UnmoveTrace(trace_id) => (trace_h.remove(trace_id).unwrap(), Some(*trace_id)),
        Rule::UnmergeFromMover {
            child_id,
            stored_id,
            trace_id,
            ..
        } => {
            let mut child = inner_interpretation(rules, lex, *child_id, trace_h)?.0;
            let stored_value = inner_interpretation(rules, lex, *stored_id, trace_h)?.0;
            trace_h.insert(*trace_id, stored_value);
            child.apply_new_free_variable(trace_id.0)?;
            (child, None)
        }
        Rule::Unmove {
            child_id,
            stored_id,
        } =>
        //We add the lambda extraction to child_id
        {
            let mut child = inner_interpretation(rules, lex, *child_id, trace_h)?.0;
            let (stored_value, trace_id) = inner_interpretation(rules, lex, *stored_id, trace_h)?;
            child.lambda_abstract_free_variable(trace_id.unwrap().0)?;
            let merged = stored_value.merge(child).unwrap();
            (merged, None)
        }
        Rule::UnmoveFromMover {
            child_id,
            stored_id,
            trace_id,
        } => todo!(),
    })
}

fn x_bar_helper<T, C>(
    child_id: RuleIndex,
    complement_id: RuleIndex,
    g: &mut StableDiGraph<String, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
) -> (NodeIndex, Vec<Feature<C>>, Option<C>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let (child_node, mut child_features, child_category) =
        inner_to_x_bar_graph(g, lex, rules, child_id, trace_h, rules_h);
    let feature = child_features.pop();

    let (complement_node, _, _) =
        inner_to_x_bar_graph(g, lex, rules, complement_id, trace_h, rules_h);

    let node = if let Some(Feature::Category(_)) = child_features.last() {
        child_features.pop();
        g.add_node(format!("{}P", child_category.as_ref().unwrap()))
    } else {
        g.add_node(format!("{}'", child_category.as_ref().unwrap()))
    };

    let (child_dir, complement_dir) = if let Some(Feature::Selector(_, dir)) = feature {
        (dir.flip(), dir)
    } else {
        (Direction::Left, Direction::Right)
    };

    g.add_edge(node, child_node, MGEdge::Merge(Some(child_dir)));
    g.add_edge(node, complement_node, MGEdge::Merge(Some(complement_dir)));

    (node, child_features, child_category)
}

fn inner_to_x_bar_graph<T, C>(
    g: &mut StableDiGraph<String, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
) -> (NodeIndex, Vec<Feature<C>>, Option<C>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let (mut node, features, category) = match rules.get(index) {
        Rule::UnmoveTrace(trace_id) => {
            let node = g.add_node(trace_id.to_string());
            trace_h.entry(*trace_id).or_default().1 = Some(node);
            (node, vec![], None)
        }
        Rule::UnmergeFromMover {
            trace_id,
            stored_id,
            child_id,
            ..
        }
        | Rule::UnmoveFromMover {
            trace_id,
            stored_id,
            child_id,
            ..
        } => {
            trace_h.entry(*trace_id).or_default().0 = Some(*stored_id);
            x_bar_helper(*child_id, *stored_id, g, lex, rules, trace_h, rules_h)
        }
        Rule::Scan { node } => {
            let mut lexeme = lex.get_lexical_entry(*node).unwrap();
            let category = lexeme.category().clone();
            let node = g.add_node(match lexeme.lemma {
                Some(x) => x.to_string(),
                None => "ε".to_string(),
            });

            let parent = g.add_node(category.to_string());
            g.add_edge(parent, node, MGEdge::Merge(None));

            lexeme.features.reverse();
            (parent, lexeme.features, Some(category))
        }
        Rule::Unmove {
            child_id,
            stored_id: complement_id,
        }
        | Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => x_bar_helper(*child_id, *complement_id, g, lex, rules, trace_h, rules_h),
        Rule::Start { child, .. } => {
            let (node, _, category) = inner_to_x_bar_graph(g, lex, rules, *child, trace_h, rules_h);

            (node, vec![], category)
        }
    };
    if let Some(Feature::Category(c)) = features.last() {
        let new_node = g.add_node(format!("{c}P"));
        g.add_edge(new_node, node, MGEdge::Merge(None));
        node = new_node
    }

    rules_h.insert(index, node);

    (node, features, category)
}

fn inner_to_graph<T, C>(
    g: &mut StableDiGraph<String, MGEdge>,
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
        g.add_edge(node, child, MGEdge::Merge(None));
    };
    if let Some(child_b) = child_b {
        let child = inner_to_graph(g, lex, rules, child_b, trace_h, rules_h);
        g.add_edge(node, child, MGEdge::Merge(None));
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
        println!("{}", dot);
        assert_eq!(
            dot.to_string(),
            "digraph {\n    0 [ label = \"ε\" ]\n    1 [ label = \"C\" ]\n    2 [ label = \"prefers\" ]\n    3 [ label = \"V\" ]\n    4 [ label = \"the\" ]\n    5 [ label = \"D\" ]\n    6 [ label = \"wine\" ]\n    7 [ label = \"N\" ]\n    8 [ label = \"NP\" ]\n    9 [ label = \"DP\" ]\n    10 [ label = \"V'\" ]\n    11 [ label = \"the\" ]\n    12 [ label = \"D\" ]\n    13 [ label = \"queen\" ]\n    14 [ label = \"N\" ]\n    15 [ label = \"NP\" ]\n    16 [ label = \"DP\" ]\n    17 [ label = \"VP\" ]\n    18 [ label = \"CP\" ]\n    1 -> 0 [ label = \"\" ]\n    3 -> 2 [ label = \"\" ]\n    5 -> 4 [ label = \"\" ]\n    7 -> 6 [ label = \"\" ]\n    8 -> 7 [ label = \"\" ]\n    9 -> 5 [ label = \"\" ]\n    9 -> 8 [ label = \"\" ]\n    10 -> 3 [ label = \"\" ]\n    10 -> 9 [ label = \"\" ]\n    12 -> 11 [ label = \"\" ]\n    14 -> 13 [ label = \"\" ]\n    15 -> 14 [ label = \"\" ]\n    16 -> 12 [ label = \"\" ]\n    16 -> 15 [ label = \"\" ]\n    17 -> 10 [ label = \"\" ]\n    17 -> 16 [ label = \"\" ]\n    18 -> 1 [ label = \"\" ]\n    18 -> 17 [ label = \"\" ]\n}\n"
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
        println!("{}", dot);
        assert_eq!(
            dot.to_string(),
 "digraph {\n    0 [ label = \"ε\" ]\n    1 [ label = \"C\" ]\n    2 [ label = \"prefers\" ]\n    3 [ label = \"V\" ]\n    4 [ label = \"the\" ]\n    5 [ label = \"D\" ]\n    6 [ label = \"wine\" ]\n    7 [ label = \"N\" ]\n    8 [ label = \"NP\" ]\n    9 [ label = \"DP\" ]\n    10 [ label = \"V'\" ]\n    11 [ label = \"which\" ]\n    12 [ label = \"D\" ]\n    13 [ label = \"queen\" ]\n    14 [ label = \"N\" ]\n    15 [ label = \"NP\" ]\n    16 [ label = \"DP\" ]\n    17 [ label = \"VP\" ]\n    18 [ label = \"C'\" ]\n    19 [ label = \"t0\" ]\n    20 [ label = \"CP\" ]\n    1 -> 0 [ label = \"\" ]\n    3 -> 2 [ label = \"\" ]\n    5 -> 4 [ label = \"\" ]\n    7 -> 6 [ label = \"\" ]\n    8 -> 7 [ label = \"\" ]\n    9 -> 5 [ label = \"\" ]\n    9 -> 8 [ label = \"\" ]\n    10 -> 3 [ label = \"\" ]\n    10 -> 9 [ label = \"\" ]\n    12 -> 11 [ label = \"\" ]\n    14 -> 13 [ label = \"\" ]\n    15 -> 14 [ label = \"\" ]\n    16 -> 12 [ label = \"\" ]\n    16 -> 15 [ label = \"\" ]\n    17 -> 10 [ label = \"\" ]\n    17 -> 19 [ label = \"\" ]\n    18 -> 1 [ label = \"\" ]\n    18 -> 17 [ label = \"\" ]\n    20 -> 18 [ label = \"\" ]\n    20 -> 16 [ label = \"\" ]\n    19 -> 16 [ label = \"move\" ]\n}\n",
        );
        Ok(())
    }
}
