use std::collections::VecDeque;
use std::fmt::Display;

use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;

use crate::lexicon;
use crate::lexicon::FeatureOrLemma;
use crate::lexicon::Lexicon;

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
            Rule::UnmergeFromMover { .. } => "Merge".to_string(),
            Rule::Unmove { .. } => "Move".to_string(),
            Rule::UnmoveFromMover { .. } => "Move".to_string(),
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct Empty {}
impl std::fmt::Display for Empty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl RulePool {
    pub fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }

    pub fn iter(&self) -> RulePoolBFSIterator {
        RulePoolBFSIterator {
            pool: self,
            queue: VecDeque::from([RuleIndex(0)]),
        }
    }

    pub fn to_graph<T, C>(&self, lex: &Lexicon<T, C>) -> DiGraph<String, Empty>
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        let mut g = DiGraph::<String, Empty>::new();
        link(&mut g, lex, self, RuleIndex(0));
        g
    }
}

fn link<T, C>(
    g: &mut DiGraph<String, Empty>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    index: RuleIndex,
) -> NodeIndex
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone,
    C: Eq + std::fmt::Debug + std::clone::Clone,
{
    let rule = rules.get(index);
    let node = g.add_node(rule.to_name(lex));
    let (child_a, child_b) = rule.children();
    if let Some(child_a) = child_a {
        let child = link(g, lex, rules, child_a);
        g.add_edge(node, child, Empty::default());
    };
    if let Some(child_b) = child_b {
        let child = link(g, lex, rules, child_b);
        g.add_edge(node, child, Empty::default());
    };
    node
}

pub struct RulePoolBFSIterator<'a> {
    pool: &'a RulePool,
    queue: VecDeque<RuleIndex>,
}

impl Iterator for RulePoolBFSIterator<'_> {
    type Item = Rule;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(r) = self.queue.pop_front() {
            let rule = self.pool.get(r);
            let (a, b) = rule.children();
            if let Some(a) = a {
                self.queue.push_back(a);
            };
            if let Some(b) = b {
                self.queue.push_back(b);
            };
            Some(*rule)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn to_graph() -> anyhow::Result<()> {
        panic!();
        Ok(())
    }
}
