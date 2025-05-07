use std::fmt::Display;

use crate::lexicon::SemanticLexicon;
use ahash::HashMap;
use itertools::Itertools;
use simple_semantics::{
    lambda::{RootedLambdaPool, types::LambdaType},
    language::Expr,
};

use super::{Rule, RuleIndex, RulePool, TraceId};

#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub enum SemanticRule {
    FunctionalApplication,
    Store,
    Identity,
    ApplyFromStorage,
    UpdateTrace,
    Trace,
    Scan,
}

impl std::fmt::Display for SemanticRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SemanticRule::FunctionalApplication => "FA",
                SemanticRule::Store => "Store",
                SemanticRule::Identity => "Id",
                SemanticRule::ApplyFromStorage => "ApplyFromStorage",
                SemanticRule::UpdateTrace => "UpdateTrace",
                SemanticRule::Trace => "Trace",
                SemanticRule::Scan => "LexicalEntry",
            }
        )
    }
}

impl RulePool {
    pub fn to_interpretation<'a, T, C>(
        &'a self,
        lex: &'a SemanticLexicon<T, C>,
    ) -> impl Iterator<Item = (RootedLambdaPool<Expr>, SemanticHistory)> + 'a
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HistoryId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HistoryNode {
    rule_id: RuleIndex,
    rule: SemanticRule,
    children: [Option<HistoryId>; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticHistory {
    Rich(Vec<(SemanticRule, Option<SemanticState>)>),
    Simple(Vec<SemanticRule>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SemanticNode {
    Rich(SemanticRule, Option<SemanticState>),
    Simple(SemanticRule),
}

impl SemanticHistory {
    pub fn semantic_node(&self, i: RuleIndex) -> Option<SemanticNode> {
        match self {
            SemanticHistory::Rich(items) => items
                .get(i.0)
                .map(|(rule, interp)| SemanticNode::Rich(*rule, interp.clone())),
            SemanticHistory::Simple(items) => {
                items.get(i.0).map(|rule| SemanticNode::Simple(*rule))
            }
        }
    }

    pub fn into_rich<T, C>(self, lexicon: &SemanticLexicon<T, C>, rules: &RulePool) -> Self
    where
        T: Eq + std::fmt::Debug + std::clone::Clone,
        C: Eq + std::fmt::Debug + std::clone::Clone,
    {
        match self {
            SemanticHistory::Rich(items) => SemanticHistory::Rich(items),
            SemanticHistory::Simple(semantic_rules) => {
                let mut items = semantic_rules.into_iter().map(|x| (x, None)).collect_vec();

                let mut derivation = SemanticDerivation {
                    rules,
                    lexicon,
                    semantic_history: vec![],
                };

                derivation.redo_history(RuleIndex(0), &mut items);

                SemanticHistory::Rich(items)
            }
        }
    }
}

impl Display for SemanticNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticNode::Rich(semantic_rule, Some(interp)) => {
                write!(f, "\\semanticRule[{}]{{{}}}", semantic_rule, interp)
            }
            SemanticNode::Rich(semantic_rule, None) => {
                write!(f, "{{[\\textsc{{{}}}]}}", semantic_rule)
            }
            SemanticNode::Simple(semantic_rule) => write!(f, "{semantic_rule}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticState {
    expr: RootedLambdaPool<Expr>,
    movers: HashMap<TraceId, (RootedLambdaPool<Expr>, LambdaType)>,
}

impl Display for SemanticState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr)
    }
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
    ) -> impl Iterator<Item = (RootedLambdaPool<Expr>, SemanticHistory)> + 'a {
        let mut derivation = SemanticDerivation {
            rules,
            lexicon: lex,
            semantic_history: vec![],
        };

        //We jump to rule 1 since the start rule is superfluous
        let last_derivation = derivation.get_previous_rules(RuleIndex(1));

        last_derivation.into_iter().filter_map(move |(x, root)| {
            if x.movers.is_empty() {
                Some((
                    x.expr,
                    SemanticHistory::Simple(derivation.get_history(root)),
                ))
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
        if let Ok(trace_type) = alpha.expr.apply_new_free_variable(trace_id.0) {
            alpha.movers.extend(beta.movers);
            alpha
                .movers
                .insert(trace_id, (beta.expr.clone(), trace_type));
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
        if let Some((stored_value, stored_type)) = alpha.movers.remove(&old_trace_id) {
            alpha
                .movers
                .insert(trace_id, (stored_value, stored_type.clone()));

            alpha
                .expr
                .lambda_abstract_free_variable(old_trace_id.0, stored_type, true)
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
        if let Some((stored_value, stored_type)) = alpha.movers.remove(&trace_id) {
            alpha
                .expr
                .lambda_abstract_free_variable(trace_id.0, stored_type, false)
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

    fn redo_history(
        &mut self,
        rule_id: RuleIndex,
        history: &mut [(SemanticRule, Option<SemanticState>)],
    ) {
        let rule = *self.rules.get(rule_id);
        let semantic_rule = history.get(rule_id.0).unwrap().0;
        let children: Vec<_> = self.rules.children(rule_id).collect();
        for child in children.iter() {
            self.redo_history(*child, history);
        }

        let get_child = |i: usize| {
            (
                history
                    .get(children.get(i).unwrap().0)
                    .unwrap()
                    .1
                    .clone()
                    .unwrap(),
                HistoryId(0),
            )
        };

        let trace_id = match &rule {
            Rule::UnmergeFromMover { trace_id, .. } | Rule::UnmoveFromMover { trace_id, .. } => {
                Some(*trace_id)
            }
            _ => None,
        };

        let value = match semantic_rule {
            SemanticRule::FunctionalApplication => {
                let child = get_child(0);
                let complement = get_child(1);

                self.functional_application(rule_id, child, complement)
            }
            SemanticRule::Store => {
                let child = get_child(0);
                let complement = get_child(1);
                self.store(rule_id, child, complement, trace_id.unwrap())
            }
            SemanticRule::Identity => {
                let child = get_child(0);
                Some(self.identity(rule_id, child))
            }
            SemanticRule::ApplyFromStorage => {
                let child = get_child(0);
                let trace_id = self.get_trace(children[1]);
                match self.apply_from_storage(rule_id, child, trace_id) {
                    ApplyFromStorageResult::SuccesfulMerge(x) => Some(x),
                    ApplyFromStorageResult::FailedMerge | ApplyFromStorageResult::NoTrace(_) => {
                        None
                    }
                }
            }
            SemanticRule::UpdateTrace => {
                let child = get_child(0);
                let old_trace_id = self.get_trace(children[1]);
                Some(self.update_trace(rule_id, child, old_trace_id, trace_id.unwrap()))
            }
            SemanticRule::Trace => {
                return;
            }
            SemanticRule::Scan => {
                let node = match rule {
                    Rule::Scan { node } => node,
                    _ => panic!(
                        "The scan semantic rule should only happen with scanning when parsing"
                    ),
                };
                Some((
                    SemanticState::new(self.lexicon.interpretation(node).clone()),
                    HistoryId(0),
                ))
            }
        };
        let s = history.get_mut(rule_id.0).unwrap();
        let state = &mut s.1;
        let mut value = value.unwrap().0;
        value.expr.reduce().unwrap();

        *state = Some(value);
    }
}

#[cfg(test)]
mod tests {

    use crate::lexicon::SemanticLexicon;
    use crate::{Parser, ParsingConfig};

    #[test]
    fn doesnt_crash_with_bad_typed_double_movement() -> anyhow::Result<()> {
        let (lexicon, _) = SemanticLexicon::parse(
            "mary::0 -1 -1::a0\n::=0 +1 0::lambda <e,e> x_l (a1)\nran::=0 +1 0::a1",
        )?;
        for (_, _, r) in Parser::new(
            lexicon.lexicon(),
            "0",
            &["mary", "ran"],
            &ParsingConfig::default(),
        )? {
            for (x, _h) in r.to_interpretation(&lexicon).take(10) {
                println!("{x}");
            }
        }
        Ok(())
    }
}
