use petgraph::graph::NodeIndex;
use std::fmt::Debug;

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

impl RulePool {
    pub fn get(&self, x: RuleIndex) -> &Rule {
        &self.0[x.0]
    }

    #[cfg(feature = "semantics")]
    pub fn to_interpretation<T, C>(
        &self,
        lex: &SemanticLexicon<T, C>,
    ) -> anyhow::Result<RootedLambdaPool<Expr>>
    where
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        let mut trace_h = HashMap::default();
        let (mut pool, _) = inner_interpretation(self, lex, RuleIndex(0), &mut trace_h)?;
        pool.reduce()?;
        Ok(pool)
    }
}

#[cfg(feature = "semantics")]
fn inner_interpretation<T, C>(
    rules: &RulePool,
    lex: &SemanticLexicon<T, C>,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, RootedLambdaPool<Expr>>,
) -> anyhow::Result<(RootedLambdaPool<Expr>, Option<TraceId>)>
where
    T: Eq + std::fmt::Debug + std::clone::Clone,
    C: Eq + std::fmt::Debug + std::clone::Clone,
{
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
        Rule::UnmoveFromMover { .. } => todo!(),
    })
}
