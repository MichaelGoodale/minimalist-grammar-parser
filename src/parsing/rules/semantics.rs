use crate::lexicon::SemanticLexicon;
use ahash::HashMap;
use itertools::Itertools;
use simple_semantics::{lambda::RootedLambdaPool, language::Expr};

use super::{Rule, RuleIndex, RulePool, TraceId};

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SemanticRule {
    FunctionalApplication,
    Store,
    Identity,
    ApplyFromStorage,
    UpdateTrace,
    Trace,
    Scan,
}

impl RulePool {
    pub fn to_interpretation<'a, T, C>(
        &'a self,
        lex: &'a SemanticLexicon<T, C>,
    ) -> impl Iterator<Item = (RootedLambdaPool<Expr>, Vec<SemanticRule>)> + 'a
    where
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        SemanticDerivation::interpret(self, lex).filter_map(|(mut pool, history)| {
            if pool.reduce().is_ok() {
                Some((pool, history))
            } else {
                None
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct HistoryId(usize);

#[derive(Debug, Clone, Copy)]
struct HistoryNode {
    rule_id: RuleIndex,
    rule: SemanticRule,
    children: [Option<HistoryId>; 2],
}

#[derive(Debug, Clone)]
struct SemanticState {
    expr: RootedLambdaPool<Expr>,
    movers: HashMap<TraceId, RootedLambdaPool<Expr>>,
}

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
            ..
        } = alpha;
        let SemanticState {
            expr: beta,
            movers: beta_movers,
            ..
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
struct SemanticDerivation<'a, T: Eq, C: Eq> {
    lexicon: &'a SemanticLexicon<T, C>,
    rules: &'a RulePool,
    semantic_history: Vec<HistoryNode>,
}

#[derive(Debug, Clone)]
enum ApplyFromStorageResult<T> {
    SuccesfulMerge(T),
    FailedMerge,
    NoTrace(T),
}

impl<'a, T, C> SemanticDerivation<'a, T, C>
where
    T: Eq + std::fmt::Debug + std::clone::Clone,
    C: Eq + std::fmt::Debug + std::clone::Clone,
{
    fn interpret(
        rules: &'a RulePool,
        lex: &'a SemanticLexicon<T, C>,
    ) -> impl Iterator<Item = (RootedLambdaPool<Expr>, Vec<SemanticRule>)> + 'a {
        let mut derivation = SemanticDerivation {
            rules,
            lexicon: lex,
            semantic_history: vec![],
        };

        //We jump to rule 1 since the start rule is superfluous
        let last_derivation = derivation.get_previous_rules(RuleIndex(1));

        last_derivation.into_iter().filter_map(move |(x, root)| {
            if x.movers.is_empty() {
                let history = derivation.get_history(root);
                Some((x.expr, history))
            } else {
                None
            }
        })
    }

    fn get_history(&self, root: HistoryId) -> Vec<SemanticRule> {
        let mut stack = vec![root];
        let mut history: Vec<Option<SemanticRule>> = self
            .rules
            .0
            .iter()
            .map(|rule| match rule {
                Rule::Start { .. } => Some(SemanticRule::Identity),
                Rule::UnmoveTrace(_) => Some(SemanticRule::Trace),
                _ => None,
            })
            .collect();

        while let Some(node) = stack.pop() {
            let HistoryNode {
                rule_id,
                rule,
                children,
            } = self.semantic_history.get(node.0).unwrap();
            history[rule_id.0] = Some(*rule);
            stack.extend(children.iter().filter_map(|x| *x));
        }

        history.into_iter().collect::<Option<Vec<_>>>().unwrap()
    }

    fn history_node(
        &mut self,
        rule_id: RuleIndex,
        semantic: SemanticRule,
        child_a: Option<HistoryId>,
        child_b: Option<HistoryId>,
    ) -> HistoryId {
        self.semantic_history.push(HistoryNode {
            rule_id,
            rule: semantic,
            children: [child_a, child_b],
        });
        HistoryId(self.semantic_history.len() - 1)
    }

    fn identity(
        &mut self,
        rule_id: RuleIndex,
        child: (SemanticState, HistoryId),
    ) -> (SemanticState, HistoryId) {
        let (alpha, child_a) = child;
        (
            alpha,
            self.history_node(rule_id, SemanticRule::Identity, Some(child_a), None),
        )
    }

    fn functional_application(
        &mut self,
        rule_id: RuleIndex,
        child: (SemanticState, HistoryId),
        complement: (SemanticState, HistoryId),
    ) -> Option<(SemanticState, HistoryId)> {
        let (alpha, alpha_id) = child;
        let (beta, beta_id) = complement;
        SemanticState::merge(alpha, beta).map(|x| {
            (
                x,
                self.history_node(
                    rule_id,
                    SemanticRule::FunctionalApplication,
                    Some(alpha_id),
                    Some(beta_id),
                ),
            )
        })
    }

    fn store(
        &mut self,
        rule_id: RuleIndex,
        child: (SemanticState, HistoryId),
        complement: (SemanticState, HistoryId),
        trace_id: TraceId,
    ) -> Option<(SemanticState, HistoryId)> {
        let (mut alpha, alpha_id) = child;
        let (beta, beta_id) = complement;
        if alpha.expr.apply_new_free_variable(trace_id.0).is_ok() {
            alpha.movers.extend(beta.movers);
            alpha.movers.insert(trace_id, beta.expr.clone());
            Some((
                alpha,
                self.history_node(rule_id, SemanticRule::Store, Some(alpha_id), Some(beta_id)),
            ))
        } else {
            None
        }
    }

    fn update_trace(
        &mut self,
        rule_id: RuleIndex,
        child: (SemanticState, HistoryId),
        old_trace_id: TraceId,
        trace_id: TraceId,
    ) -> (SemanticState, HistoryId) {
        let (mut alpha, alpha_child) = child;
        if let Some(stored_value) = alpha.movers.remove(&old_trace_id) {
            alpha.movers.insert(trace_id, stored_value);
            alpha
                .expr
                .lambda_abstract_free_variable(old_trace_id.0)
                .unwrap();
            alpha.expr.apply_new_free_variable(trace_id.0).unwrap();
        }
        (
            alpha,
            self.history_node(rule_id, SemanticRule::UpdateTrace, Some(alpha_child), None),
        )
    }

    fn apply_from_storage(
        &mut self,
        rule_id: RuleIndex,
        child: (SemanticState, HistoryId),
        trace_id: TraceId,
    ) -> ApplyFromStorageResult<(SemanticState, HistoryId)> {
        let (mut alpha, alpha_id) = child;
        if let Some(stored_value) = alpha.movers.remove(&trace_id) {
            alpha
                .expr
                .lambda_abstract_free_variable(trace_id.0)
                .unwrap();
            let SemanticState { expr, movers } = alpha;
            match expr.merge(stored_value).map(|expr| {
                (
                    SemanticState { expr, movers },
                    self.history_node(
                        rule_id,
                        SemanticRule::ApplyFromStorage,
                        Some(alpha_id),
                        None,
                    ),
                )
            }) {
                Some(x) => ApplyFromStorageResult::SuccesfulMerge(x),
                None => ApplyFromStorageResult::FailedMerge,
            }
        } else {
            ApplyFromStorageResult::NoTrace((alpha, alpha_id))
        }
    }

    fn get_trace(&mut self, trace_id: RuleIndex) -> TraceId {
        match self.rules.get(trace_id) {
            Rule::UnmoveTrace(trace_id) => *trace_id,
            _ => panic!("Ill-formed tree"),
        }
    }

    fn get_previous_rules(&mut self, rule_id: RuleIndex) -> Vec<(SemanticState, HistoryId)> {
        let rule = self.rules.get(rule_id);
        match rule {
            Rule::Scan { node } => [(
                SemanticState::new(self.lexicon.interpretation(*node).clone()),
                self.history_node(rule_id, SemanticRule::Scan, None, None),
            )]
            .into(),
            // These shouldn't be called.
            Rule::UnmoveTrace(_) => panic!("Traces shouldn't directly be accessed"),
            Rule::Start { .. } => panic!("The start rule must always be skipped"),
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
                    .filter_map(|(child, complement)| {
                        self.functional_application(rule_id, child, complement)
                    })
                    .collect()
            }
            Rule::UnmergeFromMover {
                child_id,
                stored_id,
                trace_id,
                ..
            } => {
                let stored = self.get_previous_rules(*stored_id);
                let children = self.get_previous_rules(*child_id);
                let product = children.into_iter().cartesian_product(stored);
                let mut new_states = product
                    .clone()
                    .filter_map(|(child, complement)| {
                        self.store(rule_id, child, complement, *trace_id)
                    })
                    .collect::<Vec<_>>();

                new_states.extend(product.filter_map(|(child, complement)| {
                    self.functional_application(rule_id, child, complement)
                }));
                new_states
            }
            Rule::Unmove {
                child_id,
                stored_id,
            } =>
            //We add the lambda extraction to child_id
            {
                let trace_id = self.get_trace(*stored_id);
                let children = self.get_previous_rules(*child_id);

                children
                    .into_iter()
                    .filter_map(
                        |child| match self.apply_from_storage(rule_id, child, trace_id) {
                            ApplyFromStorageResult::SuccesfulMerge(x) => Some(x),
                            ApplyFromStorageResult::FailedMerge => None,
                            ApplyFromStorageResult::NoTrace(child) => {
                                Some(self.identity(rule_id, child))
                            }
                        },
                    )
                    .collect()
            }

            Rule::UnmoveFromMover {
                child_id,
                stored_id,
                trace_id,
                ..
            } => {
                let children = self.get_previous_rules(*child_id);
                let old_trace_id = self.get_trace(*stored_id);
                let mut states = children
                    .clone()
                    .into_iter()
                    .map(|child| self.update_trace(rule_id, child, old_trace_id, *trace_id))
                    .collect::<Vec<_>>();
                states.extend(children.into_iter().filter_map(
                    |child| match self.apply_from_storage(rule_id, child, old_trace_id) {
                        ApplyFromStorageResult::SuccesfulMerge(x) => Some(x),
                        ApplyFromStorageResult::FailedMerge
                        | ApplyFromStorageResult::NoTrace(_) => None, //We don't percolate up if
                                                                      //the trace is missing because the previous rule handles that.
                    },
                ));
                states
            }
        }
    }
}
