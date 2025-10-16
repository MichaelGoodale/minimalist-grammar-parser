use std::cmp::Ordering;
use std::fmt::{Debug, Display};

use ahash::HashMap;
use serde::Serialize;

use super::{Rule, RuleIndex};
use crate::lexicon::{Feature, FeatureOrLemma, LexemeId};
use crate::parsing::rules::{StolenInfo, TraceId};
use crate::parsing::trees::GornIndex;
use crate::{Direction, Lexicon, RulePool};

struct FlatTree {
    rules: RulePool,
    order: Vec<RuleIndex>,
}

///A representation of a node in a derivation for export.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
enum MgNode<T, C: Eq + Display> {
    ///A dummy node for the root.
    Start,
    ///A normal node in the derivation
    Node {
        ///Current features at this node.
        features: Vec<Feature<C>>,
    },
    ///A lemma/leaf node.
    Leaf {
        ///The lemma displayed
        lemma: Lemma<T>,
        ///The full features of the lexical entry.
        features: Vec<Feature<C>>,
    },
    ///A trace (note usually the target of movement rather than the origin as is typical).
    Trace {
        ///The node that is moved here will have the same [`TraceId`]
        trace: TraceId,
    },
}

///Representation of a lemma for display or printing
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
enum Lemma<T> {
    ///A normal lemma
    Single(Option<T>),
    ///A head created by affixing multiple heads.
    Multi(Vec<Option<T>>),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DepthRuleOrder {
    Todo(RuleIndex),
    Done(RuleIndex),
}

impl RulePool {
    fn get_node<T: Eq + Clone, C: Eq + Display + Clone>(
        &self,
        lex: &Lexicon<T, C>,
        rule: RuleIndex,
        lemma_lookup: &LemmaLookup,
    ) -> MgNode<T, C> {
        match self.get(rule) {
            Rule::Start { .. } => MgNode::Start,
            Rule::UnmoveTrace(trace_id) => MgNode::Trace { trace: *trace_id },
            Rule::Scan { lexeme, .. } => {
                let features = lex.leaf_to_features(*lexeme).unwrap().collect();
                let lemma = lemma_lookup.get_lemma(lex, &rule);
                MgNode::Leaf { lemma, features }
            }
            Rule::Unmerge { child, .. } | Rule::UnmergeFromMover { child, .. } => MgNode::Node {
                features: lex.node_to_features(*child).collect(),
            },
            Rule::Unmove { child_id, .. } | Rule::UnmoveFromMover { child_id, .. } => {
                let mut child = *child_id;
                let mut offset = 1;

                //We loop down until we find the last feature before the movement.
                loop {
                    match self.get(child) {
                        Rule::Start { .. } | Rule::UnmoveTrace(..) | Rule::Scan { .. } => {
                            panic!("Can't move out of a leaf node or the start node.")
                        }
                        Rule::Unmerge { child, .. } | Rule::UnmergeFromMover { child, .. } => {
                            return MgNode::Node {
                                features: lex.node_to_features(*child).skip(offset).collect(),
                            };
                        }
                        Rule::Unmove { child_id, .. } | Rule::UnmoveFromMover { child_id, .. } => {
                            child = *child_id;
                            offset += 1;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyValue {
    Normal(LexemeId),
    Stealer(RuleIndex),
    Stolen(RuleIndex, GornIndex),
}

#[derive(Debug, Default, Clone)]
struct LemmaLookup {
    h: HashMap<RuleIndex, KeyValue>,
    v: HashMap<RuleIndex, Vec<(GornIndex, LexemeId, RuleIndex)>>,
}

impl LemmaLookup {
    fn add(&mut self, rule_index: RuleIndex, stolen: &StolenInfo, lexeme: LexemeId) {
        match stolen {
            StolenInfo::Normal => {
                self.h.insert(rule_index, KeyValue::Normal(lexeme));
            }
            StolenInfo::Stolen(target_index, gorn_index) => {
                self.h
                    .insert(rule_index, KeyValue::Stolen(*target_index, *gorn_index));
                self.v
                    .entry(*target_index)
                    .or_default()
                    .push((*gorn_index, lexeme, rule_index));
            }
            StolenInfo::Stealer => {
                self.h.insert(rule_index, KeyValue::Stealer(rule_index));
                self.v.entry(rule_index).or_default().push((
                    GornIndex::default(),
                    lexeme,
                    rule_index,
                ));
            }
        }
    }

    fn organize(&mut self) {
        self.v
            .values_mut()
            .for_each(|x| x.sort_by(|(x, _, _), (y, _, _)| GornIndex::infix_order(x, y)));
    }

    fn get_lemma<T: Eq + Clone, C: Eq>(
        &self,
        lex: &Lexicon<T, C>,
        rule_index: &RuleIndex,
    ) -> Lemma<T> {
        match self.h.get(rule_index).expect("Missing rule index") {
            KeyValue::Normal(lexeme_id) => Lemma::Single(
                lex.leaf_to_lemma(*lexeme_id)
                    .expect("Missing word in lexicon!")
                    .clone(),
            ),
            KeyValue::Stealer(target_rule) => Lemma::Multi(
                self.v
                    .get(target_rule)
                    .expect("Bad lemma lookup")
                    .iter()
                    .map(|(_, x, _)| lex.leaf_to_lemma(*x).unwrap().clone())
                    .collect(),
            ),
            KeyValue::Stolen(target_rule, gorn_index) => Lemma::Multi(
                self.v
                    .get(target_rule)
                    .expect("Bad lemma lookup")
                    .iter()
                    .filter_map(|(g, x, _)| {
                        if g == gorn_index || gorn_index.is_parent_of(*g) {
                            Some(lex.leaf_to_lemma(*x).unwrap().clone())
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
        }
    }

    fn new(rules: &RulePool) -> LemmaLookup {
        let mut affixes = LemmaLookup::default();

        rules
            .iter()
            .enumerate()
            .filter_map(|(x, y)| match y {
                Rule::Scan { lexeme, stolen } => Some((RuleIndex(x), stolen, *lexeme)),
                _ => None,
            })
            .for_each(|(rule_index, stolen, lexeme)| affixes.add(rule_index, stolen, lexeme));
        affixes.organize();
        dbg!(&affixes);
        affixes
    }
}

impl<T: Eq + Debug + Clone + Display, C: Eq + Debug + Clone + Display> Lexicon<T, C> {
    fn check_tree(&self, rules: RulePool) {
        let mut stack = vec![DepthRuleOrder::Todo(RuleIndex(0))];
        let mut order = vec![];
        let lemma_lookup = LemmaLookup::new(&rules);

        let nodes: Vec<_> = (0..rules.0.len())
            .map(|i| rules.get_node(self, RuleIndex(i), &lemma_lookup))
            .collect();

        while let Some(x) = stack.pop() {
            match x {
                DepthRuleOrder::Todo(rule_index) => {
                    stack.push(DepthRuleOrder::Done(rule_index));
                    stack.extend(rules.complement_last(rule_index).map(DepthRuleOrder::Todo));
                }
                DepthRuleOrder::Done(rule_index) => order.push(rule_index),
            }
        }
        println!("{:?}", order);
        for o in order {
            //println!("{:?}", rules.get(o));
            println!("{:?}", nodes[o.0]);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{Lexicon, ParsingConfig, PhonContent};

    #[test]
    fn output_tree() -> anyhow::Result<()> {
        let lex = Lexicon::from_string("C::=>a b= +y c\nA::d<= a\nB::b -y\nD::d")?;
        for (_, x, r) in lex.generate("c", &ParsingConfig::default()).unwrap() {
            println!("{x:?}");

            lex.check_tree(r);
        }
        println!("{lex}");

        panic!();
        Ok(())
    }
}
