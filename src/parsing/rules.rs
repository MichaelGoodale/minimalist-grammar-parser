use petgraph::graph::NodeIndex;
use std::{fmt::Debug, hash::Hash};

#[cfg(feature = "pretty")]
mod printing;

#[cfg(feature = "pretty")]
pub use printing::{MGEdge, MgNode, PackagedRulePool};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RuleIndex(usize);
impl RuleIndex {
    pub fn one() -> Self {
        RuleIndex(1)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct TraceId(usize);

impl TraceId {
    pub fn index(&self) -> usize {
        self.0
    }
}

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
        destination_id: RuleIndex,
        trace_id: TraceId,
    },
    Unmove {
        child_id: RuleIndex,
        stored_id: RuleIndex,
    },
    UnmoveFromMover {
        child_id: RuleIndex,
        stored_id: RuleIndex,
        destination_id: RuleIndex,
        trace_id: TraceId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PartialIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleHolder {
    rule: Rule,
    index: RuleIndex,
    parent: Option<PartialIndex>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct PartialRulePool {
    n_traces: usize,
    n_nodes: usize,
    most_recent: PartialIndex,
}

impl PartialRulePool {
    pub fn fresh(&mut self) -> RuleIndex {
        let id = RuleIndex(self.n_nodes); //Get fresh ID
        self.n_nodes += 1;
        id
    }
    pub fn fresh_trace(&mut self) -> TraceId {
        let i = TraceId(self.n_traces);
        self.n_traces += 1;
        i
    }

    pub fn n_steps(&self) -> usize {
        self.n_nodes
    }

    pub(crate) fn push_rule(&mut self, pool: &mut Vec<RuleHolder>, rule: Rule, index: RuleIndex) {
        pool.push(RuleHolder {
            rule,
            index,
            parent: Some(self.most_recent),
        });
        self.most_recent = PartialIndex(pool.len() - 1);
    }

    pub fn default_pool(cat: NodeIndex) -> Vec<RuleHolder> {
        let mut v = Vec::with_capacity(100_000);
        v.push(RuleHolder {
            rule: Rule::Start {
                node: cat,
                child: RuleIndex(1),
            },
            index: RuleIndex(0),
            parent: None,
        });
        v
    }

    pub fn into_rule_pool(self, big_pool: &[RuleHolder]) -> RulePool {
        let mut pool = vec![None; self.n_nodes];
        let mut i = Some(self.most_recent);

        while i.is_some() {
            let RuleHolder {
                rule,
                index,
                parent,
            } = big_pool[i.unwrap().0];
            match rule {
                Rule::UnmoveFromMover {
                    destination_id,
                    trace_id,
                    ..
                }
                | Rule::UnmergeFromMover {
                    destination_id,
                    trace_id,
                    ..
                } => {
                    pool[destination_id.0] = Some(Rule::UnmoveTrace(trace_id));
                }
                _ => (),
            }
            pool[index.0] = Some(rule);
            i = parent;
        }

        RulePool(pool.into_iter().collect::<Option<Vec<_>>>().unwrap())
    }
}

impl Default for PartialRulePool {
    fn default() -> Self {
        PartialRulePool {
            n_traces: 0,
            n_nodes: 2,
            most_recent: PartialIndex(0),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RulePool(Vec<Rule>);

#[cfg(feature = "semantics")]
pub mod semantics;

impl RulePool {
    #[cfg(any(feature = "pretty", feature = "semantics"))]
    fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }

    #[cfg(feature = "semantics")]
    fn children(&self, x: RuleIndex) -> impl Iterator<Item = RuleIndex> {
        match self.get(x) {
            Rule::Start { child, .. } => [Some(*child), None],
            Rule::UnmoveTrace(_) | Rule::Scan { .. } => [None, None],
            Rule::Unmove {
                child_id: a,
                stored_id: b,
            }
            | Rule::Unmerge {
                child_id: a,
                complement_id: b,
                ..
            }
            | Rule::UnmergeFromMover {
                child_id: a,
                stored_id: b,
                ..
            }
            | Rule::UnmoveFromMover {
                child_id: a,
                stored_id: b,
                ..
            } => [Some(*a), Some(*b)],
        }
        .into_iter()
        .flatten()
    }
}
