//! Defines helper functions which allow one to record the structure of a parse.
use ahash::HashSet;
use petgraph::graph::NodeIndex;
use std::{fmt::Debug, hash::Hash};

#[cfg(feature = "pretty")]
mod printing;
#[cfg(feature = "pretty")]
mod serialization;

#[cfg(feature = "pretty")]
pub use printing::Derivation;
#[cfg(feature = "pretty")]
pub use serialization::{Tree, TreeEdge, TreeNode, TreeWithMovement};

use crate::{Direction, lexicon::LexemeId};

use super::trees::GornIndex;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub(crate) struct RuleIndex(usize);
impl RuleIndex {
    pub fn one() -> Self {
        RuleIndex(1)
    }
}

///This struct record the ID of each trace in a derivation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct TraceId(usize);

impl TraceId {
    ///Gets the inner value of the trace as a `usize`
    pub fn index(&self) -> usize {
        self.0
    }
}

impl From<TraceId> for usize {
    fn from(value: TraceId) -> Self {
        value.0
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum StolenInfo {
    Normal,
    Stolen(RuleIndex, Direction),
    Stealer,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum Rule {
    Start {
        node: NodeIndex,
        child: RuleIndex,
    },
    UnmoveTrace(TraceId),
    Scan {
        lexeme: LexemeId,
        stolen: StolenInfo,
    },
    Unmerge {
        child: NodeIndex,
        child_id: RuleIndex,
        complement_id: RuleIndex,
        dir: Direction,
        affix: bool,
    },
    UnmergeFromMover {
        child: NodeIndex,
        child_id: RuleIndex,
        stored_id: RuleIndex,
        destination_id: RuleIndex,
        dir: Direction,
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

impl Rule {
    #[cfg(any(feature = "semantics", feature = "pretty"))]
    fn children_directed(&self) -> impl DoubleEndedIterator<Item = RuleIndex> {
        match self {
            Rule::Start { child, .. } => [Some(*child), None],
            Rule::UnmoveTrace(_) | Rule::Scan { .. } => [None, None],
            Rule::Unmove {
                child_id: a,
                stored_id: b,
            }
            | Rule::UnmoveFromMover {
                child_id: a,
                stored_id: b,
                ..
            } => [Some(*b), Some(*a)],
            Rule::Unmerge {
                child_id: a,
                complement_id: b,
                dir,
                ..
            }
            | Rule::UnmergeFromMover {
                child_id: a,
                stored_id: b,
                dir,
                ..
            } => match dir {
                Direction::Left => [Some(*b), Some(*a)],
                Direction::Right => [Some(*a), Some(*b)],
            },
        }
        .into_iter()
        .flatten()
    }

    #[cfg(feature = "semantics")]
    fn children(&self) -> impl DoubleEndedIterator<Item = RuleIndex> {
        match self {
            Rule::Start { child, .. } => [Some(*child), None],
            Rule::UnmoveTrace(_) | Rule::Scan { .. } => [None, None],
            Rule::Unmove {
                child_id: a,
                stored_id: b,
            }
            | Rule::UnmoveFromMover {
                child_id: a,
                stored_id: b,
                ..
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
            } => [Some(*a), Some(*b)],
        }
        .into_iter()
        .flatten()
    }

    fn is_scan(&self) -> bool {
        matches!(self, Rule::Scan { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PartialIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuleHolder {
    rule: Rule,
    index: RuleIndex,
    parent: Option<PartialIndex>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct PartialRulePool {
    n_traces: usize,
    n_nodes: usize,
    most_recent: PartialIndex,
}

impl PartialRulePool {
    ///Check what the next call to [`PartialRulePool::fresh`] will return without modifying
    pub(crate) fn peek_fresh(&self) -> RuleIndex {
        RuleIndex(self.n_nodes)
    }

    pub(crate) fn fresh(&mut self) -> RuleIndex {
        let id = RuleIndex(self.n_nodes); //Get fresh ID
        self.n_nodes += 1;
        id
    }

    pub(crate) fn fresh_trace(&mut self) -> TraceId {
        let i = TraceId(self.n_traces);
        self.n_traces += 1;
        i
    }

    pub(crate) fn n_steps(&self) -> usize {
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

    pub(crate) fn default_pool(cat: NodeIndex) -> Vec<RuleHolder> {
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

    pub(crate) fn into_rule_pool(self, big_pool: &[RuleHolder]) -> RulePool {
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

///This struct holds the history of which rules were used to generate a parse and thus can be used
///to plot trees or look at the syntactic structure of a parse.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RulePool(Vec<Rule>);

#[cfg(feature = "semantics")]
pub mod semantics;

impl RulePool {
    #[cfg(feature = "pretty")]
    pub(crate) fn complement_last(
        &self,
        x: RuleIndex,
    ) -> impl DoubleEndedIterator<Item = RuleIndex> {
        let x = self.get(x);
        match x {
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
            } => match (self.get(*a).is_scan(), self.get(*b).is_scan()) {
                (true, false) => [Some(*a), Some(*b)],
                (false, true) => [Some(*b), Some(*a)],
                (false, false) => [Some(*b), Some(*a)],
                (true, true) => [Some(*b), Some(*a)],
            },
        }
        .into_iter()
        .flatten()
    }

    ///The number of steps in the derivation
    pub fn n_steps(&self) -> usize {
        self.0.len()
    }

    #[cfg(any(feature = "pretty", feature = "semantics"))]
    fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }

    fn iter(&self) -> impl Iterator<Item = &Rule> {
        self.0.iter()
    }

    ///Gets an iterator of all used leaves.
    pub fn used_lemmas(&self) -> impl Iterator<Item = LexemeId> {
        self.0.iter().filter_map(|x| {
            if let Rule::Scan { lexeme, .. } = x {
                Some(*lexeme)
            } else {
                None
            }
        })
    }

    ///Records the maximum number of moving pieces stored in memory at a single time.
    pub fn max_memory_load(&self) -> usize {
        let mut max = 0;
        let mut memory = HashSet::default();
        for rule in &self.0 {
            match rule {
                Rule::UnmoveTrace(trace_id) => {
                    memory.insert(*trace_id);
                    if memory.len() > max {
                        max = memory.len();
                    }
                }
                Rule::UnmergeFromMover { trace_id, .. }
                | Rule::UnmoveFromMover { trace_id, .. } => {
                    memory.remove(trace_id);
                }
                _ => (),
            }
        }
        max
    }
}

#[cfg(test)]
mod test {
    use crate::{Lexicon, PhonContent};

    #[test]
    fn memory_load() -> anyhow::Result<()> {
        let grammar = Lexicon::from_string("a::b= c= +a +e C\nb::b -a\nc::c -e")?;

        let (_, _, rules) = grammar
            .parse(
                &[
                    PhonContent::Normal("c"),
                    PhonContent::Normal("b"),
                    PhonContent::Normal("a"),
                ],
                "C",
                &crate::ParsingConfig::default(),
            )?
            .next()
            .unwrap();

        assert_eq!(rules.max_memory_load(), 2);
        let grammar = Lexicon::from_string("a::b= +a c= +e C\nb::b -a\nc::c -e")?;

        let (_, _, rules) = grammar
            .parse(
                &[
                    PhonContent::Normal("c"),
                    PhonContent::Normal("b"),
                    PhonContent::Normal("a"),
                ],
                "C",
                &crate::ParsingConfig::default(),
            )?
            .next()
            .unwrap();

        assert_eq!(rules.max_memory_load(), 1);
        Ok(())
    }
}
