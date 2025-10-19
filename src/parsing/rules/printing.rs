use std::collections::BTreeMap;
use std::fmt::{Debug, Display};

#[cfg(not(feature = "semantics"))]
use std::marker::PhantomData;

#[cfg(feature = "semantics")]
use crate::lexicon::SemanticLexicon;
#[cfg(feature = "semantics")]
use crate::parsing::rules::semantics::SemanticHistory;

use ahash::HashMap;
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use serde::Serialize;

use super::serialization::Tree;
use super::{Rule, RuleIndex};
use crate::lexicon::{Feature, LexemeId};
use crate::parsing::rules::{StolenInfo, TraceId};
use crate::parsing::trees::GornIndex;
use crate::{Lexicon, RulePool};

//TODO:Indicate whether a head has been moved in a given point in a derivation.
//TODO: Helper functions for TreeNode
//TODO: Window function for derivation

///A representation of a node in a derivation for export.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub enum MgNode<T, C: Eq + Display> {
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
pub enum Lemma<T> {
    ///A normal lemma
    Single(Option<T>),
    ///A head created by affixing multiple heads.
    Multi {
        heads: Vec<Option<T>>,
        original_head: usize,
    },
}

impl<T: Display> Display for Lemma<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lemma::Single(Some(x)) => write!(f, "{x}"),
            Lemma::Single(None) => write!(f, "ε"),
            Lemma::Multi { heads, .. } => write!(
                f,
                "{}",
                heads
                    .iter()
                    .map(|x| x
                        .as_ref()
                        .map(|x| x.to_string())
                        .unwrap_or_else(|| "ε".to_string()))
                    .collect::<Vec<_>>()
                    .join("-")
            ),
        }
    }
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

impl Rule {
    fn node(&self, rules: &RulePool) -> Option<NodeIndex> {
        match self {
            Rule::UnmoveTrace(..) => None,
            Rule::Unmove { child_id, .. } | Rule::UnmoveFromMover { child_id, .. } => {
                rules.get(*child_id).node(rules)
            }
            Rule::Scan { lexeme, .. } => Some(lexeme.0),
            Rule::Start { node: child, .. }
            | Rule::UnmergeFromMover { child, .. }
            | Rule::Unmerge { child, .. } => Some(*child),
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
            KeyValue::Stealer(target_rule) => {
                let mut original_head = 0;
                let heads = self
                    .v
                    .get(target_rule)
                    .expect("Bad lemma lookup")
                    .iter()
                    .enumerate()
                    .map(|(i, (_, x, r))| {
                        if r == rule_index {
                            original_head = i;
                        }
                        lex.leaf_to_lemma(*x).unwrap().clone()
                    })
                    .collect();

                Lemma::Multi {
                    heads,
                    original_head,
                }
            }
            KeyValue::Stolen(target_rule, gorn_index) => {
                let mut original_head = 0;
                let heads = self
                    .v
                    .get(target_rule)
                    .expect("Bad lemma lookup")
                    .iter()
                    .filter(|(g, _, _)| g == gorn_index || gorn_index.is_parent_of(*g))
                    .enumerate()
                    .map(|(i, (_, x, r))| {
                        if r == rule_index {
                            original_head = i;
                        }
                        lex.leaf_to_lemma(*x).unwrap().clone()
                    })
                    .collect();

                Lemma::Multi {
                    heads,
                    original_head,
                }
            }
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
        affixes
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct TraceHistory {
    source: RuleIndex,
    destination: RuleIndex,
}

fn set_trace_destinations(trace_origins: &mut Vec<TraceHistory>, i: RuleIndex, rules: &RulePool) {
    match rules.get(i) {
        Rule::UnmergeFromMover {
            stored_id,
            destination_id,
            ..
        }
        | Rule::UnmoveFromMover {
            stored_id,
            destination_id,
            ..
        } => {
            trace_origins.push(TraceHistory {
                source: *stored_id,
                destination: *destination_id,
            });
        }
        _ => (),
    }
}

/// Reorganises <(a,b), (b,c), (c,d), (e, f)> to <<a,b,c,d>, <e,f>>
/// Can't handle orders like <(a,b), (c,d), (b,c)> so we have to sort beforehand
fn organise_movements(mut v: Vec<TraceHistory>) -> Vec<Vec<RuleIndex>> {
    v.sort_by_key(|x| std::cmp::Reverse((x.source.0, x.destination.0)));
    let mut threads: Vec<Vec<RuleIndex>> = vec![];
    for TraceHistory {
        source,
        destination,
    } in v
    {
        if let Some(x) = threads
            .iter_mut()
            .find(|x| x.last().map(|x| *x == source).unwrap_or(false))
        {
            x.push(destination);
        } else {
            threads.push(vec![source, destination]);
        }
    }
    threads
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct MovementHelper<C> {
    pub(crate) trace_origins: Vec<Vec<RuleIndex>>,
    movement_features: Vec<Vec<C>>,
    movement_ids: HashMap<RuleIndex, (usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Storage<C> {
    h: BTreeMap<usize, Vec<C>>,
}

impl<C: Display + Eq> Display for Storage<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.values()
                .map(|x| x.iter().map(Feature::Licensee).join(" "))
                .join(", ")
        )
    }
}

impl<C> Default for Storage<C> {
    fn default() -> Self {
        Self {
            h: Default::default(),
        }
    }
}

impl<C: Clone> Storage<C> {
    fn update_storage(&mut self, r: RuleIndex, m: &MovementHelper<C>) {
        if let Some((s_id, f_id)) = m.movement_ids.get(&r) {
            let features = &m.movement_features[*s_id];
            let value = features.iter().skip(*f_id).cloned().collect::<Vec<_>>();
            if value.is_empty() {
                self.h.remove(s_id);
            } else {
                self.h.insert(*s_id, value);
            }
        };
    }

    fn add_from(&mut self, other: &Storage<C>) {
        self.h.extend(other.h.iter().map(|(a, b)| (*a, b.clone())));
    }
}

impl<C> Storage<C> {
    pub fn values(&self) -> std::collections::btree_map::Values<'_, usize, Vec<C>> {
        self.h.values()
    }

    pub fn len(&self) -> usize {
        self.h.len()
    }

    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }
}

fn movement_helpers<T: Eq, C: Eq + Clone>(
    trace_origins: Vec<TraceHistory>,
    rules: &RulePool,
    lex: &Lexicon<T, C>,
) -> MovementHelper<C> {
    let trace_origins = organise_movements(trace_origins);
    let movement_features = trace_origins
        .iter()
        .map(|x| {
            dbg!(x.iter().map(|x| rules.get(*x)).collect::<Vec<_>>());
            lex.node_to_features(rules.get(*x.first().unwrap()).node(rules).unwrap())
                .skip(1) //skip category where merge happened
                .map(|x| x.into_inner())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut movement_ids = HashMap::default();
    rules.iter().enumerate().for_each(|(i, r)| match r {
        Rule::Start { .. } | Rule::UnmoveTrace(_) | Rule::Scan { .. } | Rule::Unmerge { .. } => (),
        Rule::UnmergeFromMover { stored_id, .. }
        | Rule::Unmove { stored_id, .. }
        | Rule::UnmoveFromMover { stored_id, .. } => {
            let i = RuleIndex(i);
            movement_ids.insert(
                i,
                trace_origins
                    .iter()
                    .enumerate()
                    .find_map(|(movement_id, v)| {
                        v.iter()
                            .position(|x| x == stored_id)
                            .map(|feature_id| (movement_id, feature_id))
                    })
                    .unwrap(),
            );
        }
    });

    MovementHelper {
        trace_origins,
        movement_features,
        movement_ids,
    }
}

impl<T: Eq + Debug + Clone + Display, C: Eq + Debug + Clone + Display> Lexicon<T, C> {
    ///Converts a [`RulePool`] into a [`Derivation`] which allows the construction of [`Tree`]s
    ///which can be used to plot syntactic trees throughout the derivation of the parse.
    pub fn derivation(&self, rules: RulePool) -> Derivation<'static, T, C> {
        let mut stack = vec![DepthRuleOrder::Todo(RuleIndex(0))];
        let mut order = vec![];
        let lemma_lookup = LemmaLookup::new(&rules);
        let mut trace_origins = Vec::default();

        let nodes: Vec<_> = (0..rules.0.len())
            .map(|i| {
                let i = RuleIndex(i);
                set_trace_destinations(&mut trace_origins, i, &rules);
                rules.get_node(self, i, &lemma_lookup)
            })
            .collect();

        let movement = movement_helpers(trace_origins, &rules, self);

        while let Some(x) = stack.pop() {
            match x {
                DepthRuleOrder::Todo(rule_index) => {
                    stack.push(DepthRuleOrder::Done(rule_index));
                    stack.extend(rules.complement_last(rule_index).map(DepthRuleOrder::Todo));
                }
                DepthRuleOrder::Done(rule_index) => order.push(rule_index),
            }
        }
        let Some(RuleIndex(0)) = order.pop() else {
            panic!("Malformed rules!")
        };

        Derivation {
            order,
            nodes,
            rules,
            movement,
            #[cfg(feature = "semantics")]
            semantics: None,
            #[cfg(not(feature = "semantics"))]
            semantics: PhantomData,
        }
    }
}

#[cfg(feature = "semantics")]
impl<'src, T: Eq + Debug + Clone + Display, C: Eq + Debug + Clone + Display>
    SemanticLexicon<'src, T, C>
{
    ///Converts a [`RulePool`] into a [`Derivation`] which allows the construction of [`Tree`]s
    ///which can be used to plot syntactic trees throughout the derivation of the parse. This
    ///version allows for semantic information in the derivation as well.
    pub fn derivation(&self, rules: RulePool, h: SemanticHistory<'src>) -> Derivation<'src, T, C> {
        let Derivation {
            order,
            rules,
            nodes,
            movement,
            semantics: _,
        } = self.lexicon().derivation(rules);
        Derivation {
            order,
            rules,
            nodes,
            movement,
            semantics: Some(h),
        }
    }
}

///A representation of all of the steps in the derivation of a parse.
///This can be used to generate a [`Tree`] of any step of the parse.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Derivation<'src, T, C: Eq + Display> {
    order: Vec<RuleIndex>,
    rules: RulePool,
    nodes: Vec<MgNode<T, C>>,
    pub(super) movement: MovementHelper<C>,
    #[cfg(feature = "semantics")]
    semantics: Option<SemanticHistory<'src>>,
    #[cfg(not(feature = "semantics"))]
    semantics: PhantomData<&'src ()>,
}

impl<'src, T: Clone + Debug, C: Clone + Eq + Display + Debug> Derivation<'src, T, C> {
    ///Get all possible [`Tree`]s in bottom-up order of a parse.
    pub fn trees(&self) -> impl DoubleEndedIterator<Item = Tree<'src, T, C>> {
        (0..self.order.len()).map(|x| self.tree_at(self.order[x], x))
    }

    ///Get a [`Tree`] representation of the final parse.
    pub fn tree(&self) -> Tree<'src, T, C> {
        self.tree_at(*self.order.last().unwrap(), self.order.len() - 1)
    }

    fn tree_at(&self, mut rule: RuleIndex, max_order: usize) -> Tree<'src, T, C> {
        //Handles moving nodes up or down depending on the position in the derivation
        if let Some(movement) = self
            .movement
            .trace_origins
            .iter()
            .find(|x| x.contains(&rule))
        {
            let valid_rules = &self.order[..max_order];

            let n = movement
                .iter()
                .enumerate()
                .find_map(|(i, rule)| {
                    if !valid_rules.contains(rule) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .unwrap_or(movement.len());

            if n != 0 {
                let pos = (movement.iter().position(|x| *x == rule).unwrap() + 1) % n;
                rule = movement[pos]
            }
        };

        let node = self.nodes[rule.0].clone();
        let mut children = vec![];
        let mut storage = Storage::default();
        for child in self
            .rules
            .get(rule)
            .children_directed()
            .map(|rule| self.tree_at(rule, max_order))
        {
            storage.add_from(child.storage());
            children.push(child);
        }
        storage.update_storage(rule, &self.movement);

        #[cfg(feature = "semantics")]
        if let Some(semantics) = &self.semantics {
            Tree::new_with_semantics(node, semantics.semantic_node(rule), storage, children, rule)
        } else {
            Tree::new(node, storage, children, rule)
        }
        #[cfg(not(feature = "semantics"))]
        Tree::new(node, storage, children, rule)
    }
}

#[cfg(test)]
mod test {
    use crate::{Lexicon, ParsingConfig};
    use logprob::LogProb;

    use crate::PhonContent;
    use crate::grammars::{COPY_LANGUAGE, STABLER2011};

    #[test]
    fn output_tree() -> anyhow::Result<()> {
        let grammar = [
            "John::d -k -q",
            "Mary::d -k -q",
            "some::n= d -k -q",
            "vase::n",
            "dance::V",
            "see::d= +k V",
            "break::d= V",
            "fall::d= v",
            "::=>V v",
            "::=>v =d +k voice",
            "s::=>voice +q t",
            "::=>V +q agrO",
            "::=>V +k +q agrO",
            "::=>agrO v",
            "::=>v +k voice",
        ]
        .join("\n");

        let lex =
            Lexicon::from_string(grammar.as_str()).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        for (_, _, r) in lex
            .generate("t", &ParsingConfig::default())
            .unwrap()
            .take(2)
        {
            for tree in lex.derivation(r).trees() {
                dbg!(tree);
            }
        }
        println!("{lex}");
        Ok(())
    }

    #[test]
    fn to_graph() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        for sentence in vec!["which wine the queen prefers"].into_iter() {
            let (_, _, rules) = lex
                .parse(
                    &sentence
                        .split(' ')
                        .map(PhonContent::Normal)
                        .collect::<Vec<_>>(),
                    "C",
                    &config,
                )?
                .next()
                .unwrap();
            let latex = lex.derivation(rules).tree().latex();

            println!("{}", latex);
            assert_eq!(
                latex,
                "\\begin{forest}[\\der{C} [\\der{D -W} [\\plainlex{N= D -W}{which}] [\\plainlex{N}{wine}]] [\\der{+W C} [\\plainlex{V= +W C}{$\\epsilon$}] [\\der{V} [\\der{D} [\\plainlex{N= D}{the}] [\\plainlex{N}{queen}]] [\\der{=D V} [\\plainlex{D= =D V}{prefers}] [$t_0$]]]]]\\end{forest}"
            );
        }
        Ok(())
    }

    #[test]
    fn double_movement_graph() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(COPY_LANGUAGE)?;
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        for sentence in vec!["a b a b"].into_iter() {
            let (_, _, rules) = lex
                .parse(
                    &sentence
                        .split(' ')
                        .map(PhonContent::Normal)
                        .collect::<Vec<_>>(),
                    "T",
                    &config,
                )?
                .next()
                .unwrap();
            let latex = lex.derivation(rules).tree().latex();

            println!("{}", latex);
            assert_eq!(
                latex,
                "\\begin{forest}[\\der{T} [\\der{T -l} [\\der{T -l} [\\plainlex{T -r -l}{$\\epsilon$}] [\\der{+l T -l} [$t_3$] [\\plainlex{=A +l T -l}{a}]]] [\\der{+l T -l} [$t_1$] [\\plainlex{=B +l T -l}{b}]]] [\\der{+l T} [\\der{B -r} [\\der{A -r} [$t_4$] [\\der{+r A -r} [$t_5$] [\\plainlex{=T +r A -r}{a}]]] [\\der{+r B -r} [$t_2$] [\\plainlex{=T +r B -r}{b}]]] [\\der{+r +l T} [$t_0$] [\\plainlex{=T +r +l T}{$\\epsilon$}]]]]\\end{forest}"
            );
        }
        Ok(())
    }
}
