use std::collections::HashMap;

use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;

use crate::lexicon::FeatureOrLemma;
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
        let mut pool = inner_interpretation(self, lex, RuleIndex(0));
        pool.reduce()?;
        Ok(pool)
    }
}

#[cfg(feature = "semantics")]
fn inner_interpretation<T: Eq, C: Eq>(
    rules: &RulePool,
    lex: &SemanticLexicon<T, C>,
    index: RuleIndex,
) -> RootedLambdaPool<Expr> {
    let rule = rules.get(index);
    match rule {
        Rule::Scan { node } => lex.interpretation(*node).clone(),
        Rule::Start { child, .. } => inner_interpretation(rules, lex, *child),
        Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => {
            let complement = inner_interpretation(rules, lex, *complement_id);
            let child = inner_interpretation(rules, lex, *child_id);
            child.merge(complement).unwrap()
        }
        _ => todo!(),
    }
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
            let dot = Dot::new(&g);
            println!("{}", dot);
        }
        panic!();
        Ok(())
    }
}
