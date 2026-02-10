//! Defines helper functions which allow one to record the structure of a parse.
use ahash::HashSet;
use petgraph::graph::NodeIndex;
use std::{collections::BTreeMap, fmt::Debug, hash::Hash};

#[cfg(feature = "pretty")]
mod printing;
#[cfg(feature = "pretty")]
mod serialization;

#[cfg(feature = "pretty")]
pub use printing::Derivation;
#[cfg(feature = "pretty")]
pub use serialization::{Tree, TreeEdge, TreeNode, TreeWithMovement};

use crate::{
    Direction, Lexicon,
    lexicon::{Feature, FeatureOrLemma, LexemeId},
};

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
    #[must_use]
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
    Stolen(RuleIndex, GornIndex),
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

///A description of each step in the derivation. The usizes are indices pointing to the other rules
///in the tree.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash)]
pub enum DerivationStep<T> {
    ///A lemma is merged here
    Lemma(Option<T>),

    ///Merge
    Merge {
        ///The child (towards the head)
        child: usize,
        ///The argument of the merge
        argument: usize,
        ///Whether it merged left or right
        direction: Direction,
    },
    ///Merge right and move the head of the complement
    MergeAffix {
        ///The child (towards head)
        child: usize,
        ///The argument of the merge
        argument: usize,
        ///Whether the head of the affix was put to the right or left of the child
        direction: Direction,
    },
    ///Move
    Move {
        ///The child of movement
        child: usize,
        ///That which has been moved
        mover: usize,
    },
}

impl RulePool {
    ///This outputs a [`Vec`] of [`DerivationStep`]s which indicates the steps needed to make the
    ///derivation. The usizes on the enum are indexes that indicates where the tree next points
    ///
    ///Warning, when something is moved twice, both move steps point to the origin!
    pub fn as_derivation<T: Eq + Clone + Debug, C: Eq + Debug>(
        &self,
        lex: &Lexicon<T, C>,
    ) -> Vec<DerivationStep<T>> {
        let mut stack = vec![(RuleIndex(1), 0)];
        let mut rules = vec![None];
        let mut movers: BTreeMap<RuleIndex, Vec<usize>> = BTreeMap::new();

        while let Some((x, i)) = stack.pop() {
            match self.get(x) {
                Rule::Start { .. } => unimplemented!("Should be skipped!"),
                Rule::UnmoveTrace(_) => (),
                Rule::Scan { lexeme, .. } => {
                    let lemma = lex.leaf_to_lemma(*lexeme).unwrap();
                    *rules.get_mut(i).unwrap() = Some(DerivationStep::Lemma(lemma.clone()));
                }
                Rule::Unmerge {
                    child_id,
                    complement_id,
                    dir,
                    affix,
                    child: nx,
                } => {
                    let child = rules.len();
                    let argument = rules.len() + 1;
                    rules.extend([None, None]);
                    stack.extend([(*child_id, child), (*complement_id, argument)]);
                    *rules.get_mut(i).unwrap() = Some(if *affix {
                        let FeatureOrLemma::Feature(Feature::Affix(_, dir)) =
                            lex.get(*nx).unwrap().0
                        else {
                            panic!("Can't affix a non-affix")
                        };
                        DerivationStep::MergeAffix {
                            child,
                            argument,
                            direction: *dir,
                        }
                    } else {
                        DerivationStep::Merge {
                            child,
                            argument,
                            direction: *dir,
                        }
                    });
                }
                Rule::UnmergeFromMover {
                    child_id,
                    stored_id,
                    dir,
                    destination_id,
                    ..
                } => {
                    let child = rules.len();
                    let argument = rules.len() + 1;
                    rules.extend([None, None]);
                    stack.extend([(*child_id, child), (*stored_id, argument)]);
                    let targets = movers.get(destination_id).unwrap();
                    for target in targets {
                        let Some(DerivationStep::Move { mover, .. }) =
                            rules.get_mut(*target).unwrap()
                        else {
                            panic!("Problem as all targets must be Move!")
                        };
                        *mover = argument;
                    }
                    *rules.get_mut(i).unwrap() = Some(DerivationStep::Merge {
                        child,
                        argument,
                        direction: *dir,
                    });
                }
                Rule::Unmove {
                    child_id,
                    stored_id,
                } => {
                    let child = rules.len();
                    movers.insert(*stored_id, vec![i]);
                    rules.push(None);
                    stack.push((*child_id, child));
                    *rules.get_mut(i).unwrap() = Some(DerivationStep::Move { child, mover: 0 });
                }
                Rule::UnmoveFromMover {
                    child_id,
                    stored_id,
                    destination_id,
                    ..
                } => {
                    let child = rules.len();
                    let mut future_moves = movers.remove(destination_id).unwrap();
                    future_moves.push(i);
                    movers.insert(*stored_id, future_moves);
                    rules.push(None);
                    stack.push((*child_id, child));
                    *rules.get_mut(i).unwrap() = Some(DerivationStep::Move { child, mover: 0 });
                }
            }
        }
        rules.into_iter().map(|x| x.unwrap()).collect()
    }
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
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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
                (false, true) | (false, false) | (true, true) => [Some(*b), Some(*a)],
            },
        }
        .into_iter()
        .flatten()
    }

    ///The number of steps in the derivation
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.0.len()
    }

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
    #[must_use]
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
    use crate::{
        Direction, Lexicon, ParsingConfig, PhonContent, grammars, parsing::rules::DerivationStep,
    };

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

    #[test]
    fn convert_to_derivation() -> anyhow::Result<()> {
        let lexicon = Lexicon::from_string("a::b= s\nb::c= b\nc::c")?;
        for (_, _, rules) in lexicon.generate("s", &ParsingConfig::default())? {
            assert_eq!(
                rules.as_derivation(&lexicon),
                vec![
                    DerivationStep::Merge {
                        child: 1,
                        argument: 2,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(Some("a")),
                    DerivationStep::Merge {
                        child: 3,
                        argument: 4,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(Some("b")),
                    DerivationStep::Lemma(Some("c"))
                ]
            );
        }

        println!("The king knows which beer the queen drinks");
        let lexicon = Lexicon::from_string(grammars::STABLER2011)?;
        for (_, _, rules) in lexicon.parse(
            &PhonContent::from([
                "the", "king", "knows", "which", "beer", "the", "queen", "drinks",
            ]),
            "C",
            &ParsingConfig::default(),
        )? {
            let d = rules.as_derivation(&lexicon);

            for (i, x) in d.iter().enumerate() {
                println!("{i}\t{x:?}");
            }

            assert_eq!(
                d,
                vec![
                    DerivationStep::Merge {
                        child: 1,
                        argument: 2,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(None),
                    DerivationStep::Merge {
                        child: 3,
                        argument: 4,
                        direction: Direction::Left
                    },
                    DerivationStep::Merge {
                        child: 7,
                        argument: 8,
                        direction: Direction::Right
                    },
                    DerivationStep::Merge {
                        child: 5,
                        argument: 6,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(Some("the")),
                    DerivationStep::Lemma(Some("king")),
                    DerivationStep::Lemma(Some("knows")),
                    DerivationStep::Move {
                        child: 9,
                        mover: 17
                    },
                    DerivationStep::Merge {
                        child: 10,
                        argument: 11,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(None),
                    DerivationStep::Merge {
                        child: 12,
                        argument: 13,
                        direction: Direction::Left
                    },
                    DerivationStep::Merge {
                        child: 16,
                        argument: 17,
                        direction: Direction::Right
                    },
                    DerivationStep::Merge {
                        child: 14,
                        argument: 15,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(Some("the")),
                    DerivationStep::Lemma(Some("queen")),
                    DerivationStep::Lemma(Some("drinks")),
                    DerivationStep::Merge {
                        child: 18,
                        argument: 19,
                        direction: Direction::Right
                    },
                    DerivationStep::Lemma(Some("which")),
                    DerivationStep::Lemma(Some("beer"))
                ]
            );
        }

        let lexicon = Lexicon::from_string(grammars::COPY_LANGUAGE)?;
        println!("ABAB");

        for (_, _, rules) in lexicon.parse(
            &PhonContent::from(["a", "b", "a", "b"]),
            "T",
            &ParsingConfig::default(),
        )? {
            let d = rules.as_derivation(&lexicon);

            for (i, x) in d.iter().enumerate() {
                println!("{i}\t{x:?}");
            }
            assert_eq!(
                d,
                vec![
                    DerivationStep::Move { child: 1, mover: 4 },
                    DerivationStep::Move { child: 2, mover: 7 },
                    DerivationStep::Merge {
                        child: 3,
                        argument: 4,
                        direction: Direction::Left
                    },
                    DerivationStep::Lemma(None),
                    DerivationStep::Move {
                        child: 5,
                        mover: 10
                    },
                    DerivationStep::Merge {
                        child: 6,
                        argument: 7,
                        direction: Direction::Left
                    },
                    DerivationStep::Lemma(Some("b")),
                    DerivationStep::Move {
                        child: 8,
                        mover: 13
                    },
                    DerivationStep::Merge {
                        child: 9,
                        argument: 10,
                        direction: Direction::Left
                    },
                    DerivationStep::Lemma(Some("b")),
                    DerivationStep::Move {
                        child: 11,
                        mover: 16
                    },
                    DerivationStep::Merge {
                        child: 12,
                        argument: 13,
                        direction: Direction::Left
                    },
                    DerivationStep::Lemma(Some("a")),
                    DerivationStep::Move {
                        child: 14,
                        mover: 16
                    },
                    DerivationStep::Merge {
                        child: 15,
                        argument: 16,
                        direction: Direction::Left
                    },
                    DerivationStep::Lemma(Some("a")),
                    DerivationStep::Lemma(None)
                ]
            );
        }

        Ok(())
    }
}
