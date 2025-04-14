use itertools::Itertools;
use petgraph::graph::NodeIndex;
use std::{fmt::Debug, hash::Hash};

#[cfg(feature = "semantics")]
use crate::lexicon::SemanticLexicon;
#[cfg(feature = "semantics")]
use ahash::HashMap;
#[cfg(feature = "semantics")]
use simple_semantics::{lambda::RootedLambdaPool, language::Expr};

#[cfg(feature = "pretty")]
mod printing;

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

#[derive(Debug, Clone, Copy)]
struct PartialIndex(usize);

#[derive(Debug, Clone, Copy)]
pub struct RuleHolder {
    rule: Rule,
    index: RuleIndex,
    parent: Option<PartialIndex>,
}

#[derive(Debug, Copy, Clone)]
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

    pub fn push_rule(&mut self, pool: &mut Vec<RuleHolder>, rule: Rule, index: RuleIndex) {
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

impl RulePool {
    pub fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(feature = "semantics")]
    pub fn to_interpretation<'a, T, C>(
        &'a self,
        lex: &'a SemanticLexicon<T, C>,
    ) -> impl Iterator<Item = RootedLambdaPool<Expr>> + 'a
    where
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        SemanticDerivation::interpret(self, lex).filter_map(|mut pool| {
            if pool.reduce().is_ok() {
                Some(pool)
            } else {
                None
            }
        })
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "semantics")]
struct SemanticState {
    expr: RootedLambdaPool<Expr>,
    movers: HashMap<TraceId, RootedLambdaPool<Expr>>,
}

#[cfg(feature = "semantics")]
impl SemanticState {
    fn new(alpha: RootedLambdaPool<Expr>) -> Self {
        SemanticState {
            expr: alpha,
            movers: HashMap::default(),
        }
    }

    fn merge(alpha: Self, beta: Self) -> Option<Self> {
        let overlapping_traces = alpha.movers.keys().any(|k| beta.movers.contains_key(k));
        if overlapping_traces {
            return None;
        }

        let SemanticState {
            expr: alpha,
            movers: mut alpha_movers,
        } = alpha;
        let SemanticState {
            expr: beta,
            movers: beta_movers,
        } = beta;
        let alpha = alpha.merge(beta).unwrap();
        for (k, v) in beta_movers {
            alpha_movers.insert(k, v);
        }
        Some(SemanticState {
            expr: alpha,
            movers: alpha_movers,
        })
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "semantics")]
struct SemanticDerivation<'a, T: Eq, C: Eq> {
    lexicon: &'a SemanticLexicon<T, C>,
    rules: &'a RulePool,
}

#[cfg(feature = "semantics")]
impl<T, C> SemanticDerivation<'_, T, C>
where
    T: Eq + std::fmt::Debug + std::clone::Clone,
    C: Eq + std::fmt::Debug + std::clone::Clone,
{
    fn interpret(
        rules: &RulePool,
        lex: &SemanticLexicon<T, C>,
    ) -> impl Iterator<Item = RootedLambdaPool<Expr>> {
        let mut derivation = SemanticDerivation {
            rules,
            lexicon: lex,
        };

        //We jump to rule 1 since the start rule is superfluous
        let last_derivation = derivation.get_previous_rules(RuleIndex(1));

        last_derivation.into_iter().filter_map(|x| {
            if x.movers.is_empty() {
                Some(x.expr)
            } else {
                None
            }
        })
    }

    fn get_previous_rules(&mut self, index: RuleIndex) -> Vec<SemanticState> {
        let rule = self.rules.get(index);
        match rule {
            Rule::Scan { node } => {
                vec![SemanticState::new(
                    self.lexicon.interpretation(*node).clone(),
                )]
            }
            Rule::Start { .. } => {
                panic!("The start rule should always be skipped");
            } // This shouldn't be called.
            Rule::Unmerge {
                child_id,
                complement_id,
                ..
            } => {
                let complements = self.get_previous_rules(*complement_id);
                let children = self.get_previous_rules(*child_id);

                children
                    .into_iter()
                    .cartesian_product(complements)
                    .filter_map(|(alpha, beta)| SemanticState::merge(alpha, beta))
                    .collect::<Vec<_>>()
            }
            Rule::UnmoveTrace(_) => panic!("Traces shouldn't directly be accessed"),
            Rule::UnmergeFromMover {
                child_id,
                stored_id,
                trace_id,
                ..
            } => {
                let stored = self.get_previous_rules(*stored_id);
                let children = self.get_previous_rules(*child_id);

                children
                    .into_iter()
                    .cartesian_product(stored)
                    .filter_map(|(mut alpha, beta)| {
                        if alpha.expr.apply_new_free_variable(trace_id.0).is_ok() {
                            alpha.movers.insert(*trace_id, beta.expr.clone());
                            Some(alpha)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            }
            Rule::Unmove {
                child_id,
                stored_id,
            } =>
            //We add the lambda extraction to child_id
            {
                let trace_id = match self.rules.get(*stored_id) {
                    Rule::UnmoveTrace(trace_id) => trace_id,
                    _ => panic!("Ill-formed tree"),
                };
                self.get_previous_rules(*child_id)
                    .into_iter()
                    .filter_map(|alpha| {
                        let mut alpha = alpha.clone();
                        if alpha.expr.lambda_abstract_free_variable(trace_id.0).is_ok() {
                            alpha.movers.remove(trace_id).and_then(|stored_value| {
                                let SemanticState { expr, movers } = alpha;
                                expr.merge(stored_value)
                                    .map(|expr| SemanticState { expr, movers })
                            })
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            }
            _ => todo!(), /*
                          Rule::UnmergeFromMover {
                              child_id,
                              stored_id,
                              trace_id,
                              ..
                          } => {
                              let mut child = inner_interpretation(rules, lex, *child_id, trace_h)?.0;
                              let stored_value = inner_interpretation(rules, lex, *stored_id, trace_h)?.0;
                              dbg!(child.get_type()?, &stored_value.get_type()?);
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
                              let (stored_value, trace_id) =
                                  inner_interpretation(rules, lex, *stored_id, trace_h)?;
                              child.lambda_abstract_free_variable(trace_id.unwrap().0)?;
                              let merged = stored_value.merge(child).unwrap();
                              (merged, None)
                          }
                          Rule::UnmoveFromMover {
                              child_id,
                              stored_id,
                              trace_id,
                              ..
                          } => {
                              let mut child = inner_interpretation(rules, lex, *child_id, trace_h)?.0;
                              let (stored_value, next_trace_id) =
                                  inner_interpretation(rules, lex, *stored_id, trace_h)?;
                              trace_h.insert(*trace_id, stored_value);

                              child.apply_new_free_variable(trace_id.0)?;
                              child.lambda_abstract_free_variable(next_trace_id.unwrap().0)?;
                              (child, None)
                          }*/
        }
    }
}
