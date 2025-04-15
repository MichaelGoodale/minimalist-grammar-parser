use crate::lexicon::SemanticLexicon;
use ahash::HashMap;
use itertools::Itertools;
use simple_semantics::{lambda::RootedLambdaPool, language::Expr};

use super::{Rule, RuleIndex, RulePool, TraceId};

impl RulePool {
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
        if let Some(alpha) = alpha.merge(beta) {
            alpha_movers.extend(beta_movers);
            Some(SemanticState {
                expr: alpha,
                movers: alpha_movers,
            })
        } else {
            None
        }
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
            Rule::Scan { node } => [SemanticState::new(
                self.lexicon.interpretation(*node).clone(),
            )]
            .into(),
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
                    .collect()
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
                let product = children.into_iter().cartesian_product(stored);
                product
                    .clone()
                    .filter_map(|(mut alpha, beta)| {
                        if alpha.expr.apply_new_free_variable(trace_id.0).is_ok() {
                            alpha.movers.extend(beta.movers);
                            alpha.movers.insert(*trace_id, beta.expr.clone());
                            Some(alpha)
                        } else {
                            None
                        }
                    })
                    .chain(product.filter_map(|(alpha, beta)| SemanticState::merge(alpha, beta)))
                    .collect()
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
                let children = self.get_previous_rules(*child_id);

                children
                    .into_iter()
                    .filter_map(|mut alpha| {
                        if let Some(stored_value) = alpha.movers.remove(trace_id) {
                            alpha
                                .expr
                                .lambda_abstract_free_variable(trace_id.0)
                                .unwrap();
                            let SemanticState { expr, movers } = alpha;
                            expr.merge(stored_value)
                                .map(|expr| SemanticState { expr, movers })
                        } else {
                            Some(alpha)
                        }
                    })
                    .collect()
            }

            Rule::UnmoveFromMover {
                child_id,
                stored_id,
                trace_id,
                ..
            } => {
                let children = self.get_previous_rules(*child_id);
                let old_trace_id = match self.rules.get(*stored_id) {
                    Rule::UnmoveTrace(trace_id) => trace_id,
                    _ => panic!("Ill-formed tree"),
                };
                children
                    .clone()
                    .into_iter()
                    .map(|mut x| {
                        if let Some(stored_value) = x.movers.remove(old_trace_id) {
                            x.movers.insert(*trace_id, stored_value);
                            x.expr
                                .lambda_abstract_free_variable(old_trace_id.0)
                                .unwrap();
                            x.expr.apply_new_free_variable(trace_id.0).unwrap();
                        }
                        x
                    })
                    .chain(children.into_iter().filter_map(|mut x| {
                        if let Some(stored_value) = x.movers.remove(old_trace_id) {
                            let SemanticState { mut expr, movers } = x;
                            expr.lambda_abstract_free_variable(old_trace_id.0).unwrap();
                            expr.merge(stored_value)
                                .map(|expr| SemanticState { expr, movers })
                        } else {
                            None
                        }
                    }))
                    .collect()
            }
        }
    }
}
