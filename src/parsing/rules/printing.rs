use itertools::Itertools;
use petgraph::Direction::Incoming;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::EdgeRef;
use serde::Serialize;
use serde::ser::SerializeSeq;
use serde::ser::SerializeStructVariant;

use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;

#[cfg(not(feature = "semantics"))]
use std::marker::PhantomData;

use thiserror::Error;

use crate::lexicon::Feature;
use crate::lexicon::FeatureOrLemma;
use crate::lexicon::LexemeId;
use crate::lexicon::LexicalEntry;
use crate::lexicon::Lexicon;
use crate::parsing::trees::GornIndex;

use super::{Rule, RuleIndex, RulePool, StolenInfo, TraceId};
use crate::Direction;

#[cfg(feature = "semantics")]
use crate::lexicon::SemanticLexicon;

#[cfg(feature = "semantics")]
use super::semantics::{SemanticHistory, SemanticNode};

#[cfg(feature = "semantics")]
use regex::Regex;

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
///Represents a line in a tree, and stores whether it was movement or a merge (and in which
///direction)
pub enum MGEdge {
    ///The consituent moved
    Move,
    ///Merge and direction (where [`None`] refers to merging from movement)
    Merge(Option<Direction>),
}

impl std::fmt::Display for MGEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MGEdge::Move => "move",
            MGEdge::Merge(None) => "",
            MGEdge::Merge(Some(Direction::Left)) => "",
            MGEdge::Merge(Some(Direction::Right)) => "",
        };
        write!(f, "{s}")
    }
}

/// Reorganises <(a,b), (b,c), (c,d), (e, f)> to <<a,b,c,d>, <e,f>>
/// Can't handle orders like <(a,b), (c,d), (b,c)> so we have to sort beforehand
fn organise_movements<T: PartialEq + Copy>(v: &[(T, T)]) -> Vec<Vec<T>> {
    let mut threads: Vec<Vec<T>> = vec![];
    for (start, end) in v {
        if let Some(x) = threads
            .iter_mut()
            .find(|x| x.last().map(|x| x == start).unwrap_or(false))
        {
            x.push(*end);
        } else {
            threads.push(vec![*start, *end]);
        }
    }
    threads
}

impl RulePool {
    #[allow(clippy::type_complexity)]
    fn to_graph<T, C>(
        &self,
        lex: &Lexicon<T, C>,
    ) -> (
        StableDiGraph<MgNode<T, C>, MGEdge>,
        NodeIndex,
        HashMap<NodeIndex, RuleIndex>,
    )
    where
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let mut g = StableDiGraph::<MgNode<T, C>, MGEdge>::new();
        let mut trace_h = HashMap::new();
        let mut rule_h: HashMap<RuleIndex, NodeIndex> = HashMap::new();
        let mut nodes_h: HashMap<NodeIndex, RuleIndex> = HashMap::new();

        let (root, _, _, _) = inner_to_graph(
            &mut g,
            lex,
            self,
            RuleIndex(0),
            &mut trace_h,
            &mut rule_h,
            &mut nodes_h,
        );

        //Complicated code to add edges between traces and their origins as well as for canceling
        //the features of movers.
        let mut movements = trace_h
            .into_iter()
            .filter_map(|(t, x)| {
                if let (Some(unmove_origin_rule), Some(trace_node)) = x {
                    Some((t, trace_node, *rule_h.get(&unmove_origin_rule).unwrap()))
                } else {
                    None
                }
            })
            .collect_vec();

        movements.sort_by_key(|(x, _, _)| *x);
        let movements: Vec<_> = movements.into_iter().map(|(_, x, y)| (x, y)).collect();

        for (trace_node, _trace_origin) in movements.iter() {
            let trace_id = match g.node_weight(*trace_node).unwrap() {
                MgNode::Trace { trace, .. } => *trace,
                _ => panic!("trace_node must be a MgNode::Trace!"),
            };

            let parent = g
                .neighbors_directed(*trace_node, petgraph::Direction::Incoming)
                .next()
                .unwrap();

            let sister = g
                .neighbors_directed(parent, petgraph::Direction::Outgoing)
                .find(|x| *x != *trace_node)
                .unwrap();

            match g.node_weight_mut(sister).unwrap() {
                MgNode::Node { movement, .. } | MgNode::Leaf { movement, .. } => {
                    let m = movement
                        .iter_mut()
                        .find(|x| x.trace_id == trace_id)
                        .unwrap();
                    m.canceled = true;
                }
                MgNode::Trace { .. } => (),
            };
        }
        let movements = organise_movements(&movements);

        for traces in movements.into_iter() {
            let origin = *traces.first().unwrap();
            let dest = *traces.last().unwrap();

            let (origin_node, dest_node) = g.index_twice_mut(origin, dest);
            let origin_t = origin_node.destination_trace_mut().unwrap();
            let dest_t = dest_node.start_trace_mut().unwrap();
            std::mem::swap(origin_t, dest_t);

            let parent = |x: NodeIndex| {
                g.edges_directed(x, Incoming)
                    .map(|x| (x.source(), x.id()))
                    .next()
                    .unwrap()
            };
            let (p_origin, e1) = parent(origin);
            let (p_dest, e2) = parent(dest);
            let w = g.remove_edge(e1).unwrap();
            g.add_edge(p_origin, dest, w);
            let w = g.remove_edge(e2).unwrap();
            g.add_edge(p_dest, origin, w);

            for x in traces.windows(2) {
                g.add_edge(*x.first().unwrap(), *x.last().unwrap(), MGEdge::Move);
            }
        }

        (g, root, nodes_h)
    }

    #[cfg(feature = "semantics")]
    ///Converts the [`RulePool`] to a json format combined with semantics.
    pub fn to_semantic_tree<'src, T, C>(
        &self,
        semantic_lex: &'_ SemanticLexicon<'src, T, C>,
        history: &'_ SemanticHistory<'src>,
    ) -> Tree<'src, T, C>
    where
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
    {
        let (g, root, node_to_rule) = self.to_graph(semantic_lex.lexicon());
        let g = g.map(
            |i, node| SemanticMGNode {
                node: node.clone(),
                semantic: {
                    history
                        .semantic_node(*node_to_rule.get(&i).unwrap())
                        .unwrap()
                },
            },
            |_, e| *e,
        );

        Tree::new_semantic(&g, root)
    }

    ///Converts a [`RulePool`] to a [`Tree<'a, T, C>`] using its [`Lexicon<T,C>`].
    pub fn to_tree<'a, T, C>(&'a self, lex: &Lexicon<T, C>) -> Tree<'a, T, C>
    where
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize + 'a,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize + 'a,
    {
        let (g, root, _) = self.to_graph(lex);

        Tree::new(&g, root)
    }
    ///Converts a [`RulePool`] to a petgraph [`StableDiGraph`]. Returns graph and its root.
    pub fn to_petgraph<'a, T, C>(
        &'a self,
        lex: &Lexicon<T, C>,
    ) -> (StableDiGraph<MgNode<T, C>, MGEdge>, NodeIndex)
    where
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize + 'a,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize + 'a,
    {
        let (g, root, _) = self.to_graph(lex);
        (g, root)
    }
}

fn get_children<N>(g: &StableDiGraph<N, MGEdge>, x: NodeIndex) -> Vec<NodeIndex> {
    g.edges_directed(x, petgraph::Direction::Outgoing)
        .sorted_by_key(|x| x.weight())
        .filter_map(|e| match e.weight() {
            MGEdge::Move => None,
            MGEdge::Merge(_) => Some(e.target()),
        })
        .collect()
}

impl<'a, T: Display, C: Eq + Display> Tree<'a, T, C> {
    pub fn to_latex(&self) -> String {
        format!("\\begin{{forest}}{}\\end{{forest}}", self.to_latex_inner())
    }

    fn to_latex_inner(&self) -> String {
        let node = self.node.to_latex();

        let children: Vec<_> = self.children.iter().map(|x| x.to_latex_inner()).collect();
        if children.is_empty() {
            format!("[{node}]")
        } else {
            format!("[{node} {}]", children.join(" "))
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tree<'a, T, C: Eq + Display> {
    node: TreeNode<'a, T, C>,
    children: Vec<Tree<'a, T, C>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct TreeNode<'a, T, C: Eq + Display> {
    node: MgNode<T, C>,

    #[cfg(feature = "semantics")]
    semantics: Option<SemanticNode<'a>>,

    #[cfg(not(feature = "semantics"))]
    data: PhantomData<&'a ()>,
}

impl<'a, T: Display, C: Eq + Display> TreeNode<'a, T, C> {
    fn to_latex(&self) -> String {
        match &self.node {
            MgNode::Node {
                features, trace, ..
            } => {
                let s = {
                    let features = features.iter().map(|x| x.to_string()).join(" ");

                    #[cfg(feature = "semantics")]
                    if let Some(meaning) = &self.semantics {
                        match meaning {
                            SemanticNode::Rich(..) => {
                                return format!(
                                    "\\semder{{{features}}}{{\\texttt{{{}}}}}",
                                    clean_up_expr(meaning.to_string())
                                );
                            }
                            SemanticNode::Simple(_) => {
                                return format!("\\semder{{{features}}}{{\\textsc{{{meaning}}}}}");
                            }
                        }
                    }
                    format!("\\der{{{features}}}")
                };
                if let Some(trace) = trace {
                    format!("{s}, name=node{trace}")
                } else {
                    s
                }
            }
            MgNode::Leaf {
                lemma, features, ..
            } => {
                let features = features.iter().map(|x| x.to_string()).join(" ");
                let lemma = lemma.to_string("$\\epsilon$", "-");
                #[cfg(feature = "semantics")]
                if let Some(meaning) = &self.semantics {
                    match meaning {
                        SemanticNode::Rich(..) => {
                            return format!(
                                "\\lex{{{features}}}{{{lemma}}}{{\\texttt{{{}}}}}",
                                clean_up_expr(meaning.to_string())
                            );
                        }
                        SemanticNode::Simple(_) => {
                            return format!(
                                "\\lex{{{features}}}{{{lemma}}}{{\\textsc{{{meaning}}}}}"
                            );
                        }
                    }
                }
                format!("\\plainlex{{{features}}}{{{lemma}}}")
            }
            MgNode::Trace { trace, new_trace } => match new_trace {
                Some(trace) => format!("$t$, name=trace{trace}"),
                None => format!("$t$, name=trace{trace}"),
            },
        }
    }
}

impl<T, C: Eq> Serialize for TreeNode<'_, T, C>
where
    C: Display,
    Mover<C>: Serialize,
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.node {
            MgNode::Node {
                features,
                movement,
                trace,
                ..
            } => {
                #[cfg(not(feature = "semantics"))]
                let n = 3;

                #[cfg(feature = "semantics")]
                let n = if self.semantics.is_some() { 4 } else { 3 };

                let mut seq = serializer.serialize_struct_variant("MgNode", 0, "Node", n)?;

                seq.serialize_field("features", features)?;
                seq.serialize_field("movement", movement)?;
                seq.serialize_field("trace", trace)?;

                #[cfg(feature = "semantics")]
                if let Some(semantics) = &self.semantics {
                    seq.serialize_field("semantics", &semantics)?;
                }

                seq.end()
            }

            MgNode::Leaf {
                lemma,
                features,
                movement,
                trace,
                ..
            } => {
                #[cfg(not(feature = "semantics"))]
                let n = 4;

                #[cfg(feature = "semantics")]
                let n = if self.semantics.is_some() { 5 } else { 4 };

                let mut seq = serializer.serialize_struct_variant("MgNode", 1, "Leaf", n)?;

                seq.serialize_field("features", features)?;
                seq.serialize_field("movement", movement)?;
                seq.serialize_field("lemma", lemma)?;
                seq.serialize_field("trace", trace)?;

                #[cfg(feature = "semantics")]
                if let Some(semantics) = &self.semantics {
                    seq.serialize_field("semantics", semantics)?;
                }

                seq.end()
            }

            MgNode::Trace { trace, new_trace } => {
                #[cfg(not(feature = "semantics"))]
                let n = 2;

                #[cfg(feature = "semantics")]
                let n = if self.semantics.is_some() { 3 } else { 2 };

                let mut seq = serializer.serialize_struct_variant("MgNode", 2, "Trace", n)?;

                seq.serialize_field("trace", trace)?;
                seq.serialize_field("new_trace", new_trace)?;

                #[cfg(feature = "semantics")]
                if let Some(semantics) = &self.semantics {
                    seq.serialize_field("semantics", semantics)?;
                }

                seq.end()
            }
        }
    }
}

impl<T, C: Eq> Serialize for Tree<'_, T, C>
where
    C: Display,
    Mover<C>: Serialize,
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        //Slightly odd serialization order for compatibility with Typst's Cetz library.
        if self.children.is_empty() {
            self.node.serialize(serializer)
        } else {
            let mut seq = serializer.serialize_seq(Some(self.children.len() + 1))?;
            seq.serialize_element(&self.node)?;
            for tree in &self.children {
                seq.serialize_element(tree)?;
            }
            seq.end()
        }
    }
}

impl<T, C> Tree<'_, T, C>
where
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
    MgNode<T, C>: Serialize,
{
    fn new<'a>(g: &'_ StableDiGraph<MgNode<T, C>, MGEdge>, x: NodeIndex) -> Tree<'a, T, C> {
        Tree {
            node: TreeNode {
                node: g.node_weight(x).unwrap().clone(),

                #[cfg(feature = "semantics")]
                semantics: None,
                #[cfg(not(feature = "semantics"))]
                data: PhantomData,
            },
            children: get_children(g, x)
                .into_iter()
                .map(|x| Self::new(g, x))
                .collect(),
        }
    }

    #[cfg(feature = "semantics")]
    fn new_semantic<'a>(
        g: &'_ StableDiGraph<SemanticMGNode<'a, T, C>, MGEdge>,
        x: NodeIndex,
    ) -> Tree<'a, T, C> {
        let node = g.node_weight(x).unwrap();
        Tree {
            node: TreeNode {
                node: node.node.clone(),

                #[cfg(feature = "semantics")]
                semantics: Some(node.semantic.clone()),
                #[cfg(not(feature = "semantics"))]
                data: PhantomData,
            },
            children: get_children(g, x)
                .into_iter()
                .map(|x| Self::new_semantic(g, x))
                .collect(),
        }
    }
}

impl Serialize for TraceId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq, Hash)]
pub struct Mover<C: Eq + Display> {
    trace_id: TraceId,
    canceled: bool,
    features: Vec<Feature<C>>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PrintError {
    #[error("The alleged lemma is not a leaf!")]
    NotALeaf,
    #[error("The alleged trace is not a trace!")]
    NotATrace,
}

impl<C: Eq + Display> Mover<C> {
    pub fn features(&self) -> &[Feature<C>] {
        &self.features
    }
}

///Representation of a lemma for display or printing
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub enum Lemma<T> {
    ///A normal lemma
    Single(Option<T>),
    ///A head that has been stolen and what it would otherwise have been.
    Stolen(Option<T>),
    ///A head created by affixing multiple heads.
    Multi(Vec<Option<T>>),
}

impl<T: Display> Lemma<T> {
    pub fn to_string(&self, empty_string: &str, join: &str) -> String {
        match self {
            Lemma::Single(Some(x)) => x.to_string(),
            Lemma::Single(None) | Lemma::Stolen(_) => empty_string.to_string(),
            Lemma::Multi(items) => items
                .iter()
                .map(|x| {
                    x.as_ref()
                        .map(|x| x.to_string())
                        .unwrap_or_else(|| empty_string.to_string())
                })
                .collect::<Vec<_>>()
                .join(join),
        }
    }
}

///A representation of a node in a derivation for export.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub enum MgNode<T, C: Eq + Display> {
    ///A normal node in the derivation
    Node {
        ///Current features at this node.
        features: Vec<Feature<C>>,
        ///Current present moving features.
        movement: Vec<Mover<C>>,
        ///If this node is moved, what is its target trace?
        trace: Option<TraceId>,

        ///Whether a node is root.
        #[serde(skip_serializing)]
        root: bool,
    },
    ///A lemma/leaf node.
    Leaf {
        ///The lemma displayed
        lemma: Lemma<T>,
        ///The full features of the lexical entry.
        features: Vec<Feature<C>>,
        ///Whether this lemma holds any movers
        movement: Vec<Mover<C>>,
        ///If this node is moved, what is its target trace?
        trace: Option<TraceId>,
        ///Whether a node is root.
        #[serde(skip_serializing)]
        root: bool,
    },
    ///A trace (note usually the target of movement rather than the origin as is typical).
    Trace {
        ///The node that is moved here will have the same [`TraceId`]
        trace: TraceId,
        ///If this trace is moved again, where does it go?
        new_trace: Option<TraceId>,
    },
}

impl<C: Eq + Display> Serialize for Feature<C> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

impl<T, C: Eq + Display> MgNode<T, C> {
    ///The features at a given node
    pub fn features(&self) -> &[Feature<C>] {
        match self {
            MgNode::Node { features, .. } => features,
            MgNode::Leaf { features, .. } => features,
            MgNode::Trace { .. } => &[],
        }
    }

    ///Get the trace ID of a node, if it is one
    pub fn trace(&self) -> Result<TraceId, PrintError> {
        match self {
            MgNode::Trace { trace, .. } => Ok(*trace),
            MgNode::Node { .. } | MgNode::Leaf { .. } => Err(PrintError::NotATrace),
        }
    }

    fn start_trace_mut(&mut self) -> Option<&mut TraceId> {
        match self {
            MgNode::Node { trace, .. } => trace.as_mut(),
            MgNode::Leaf { trace, .. } => trace.as_mut(),
            MgNode::Trace { new_trace, .. } => new_trace.as_mut(),
        }
    }

    fn destination_trace_mut(&mut self) -> Option<&mut TraceId> {
        if let MgNode::Trace { trace, .. } = self {
            Some(trace)
        } else {
            None
        }
    }

    ///Get the lemma of a node, if it is a leaf
    pub fn lemma(&self) -> Result<&Lemma<T>, PrintError> {
        match self {
            MgNode::Leaf { lemma, .. } => Ok(lemma),
            MgNode::Node { .. } | MgNode::Trace { .. } => Err(PrintError::NotALeaf),
        }
    }
}

#[cfg(feature = "semantics")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SemanticMGNode<'a, T, C: Eq + Display> {
    node: MgNode<T, C>,
    semantic: SemanticNode<'a>,
}

#[cfg(feature = "semantics")]
fn clean_up_expr(s: String) -> String {
    let re = Regex::new(r"lambda (?<t>[eat,< >]+) ").unwrap();
    let s = s
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("#", "\\#");
    re.replace_all(s.as_str(), "{$\\lambda_{$t}$}")
        .to_string()
        .replace("<", "\\left\\langle ")
        .replace(">", "\\right\\rangle ")
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum LemmaNode {
    Normal(LexemeId),
    Moved(LexemeId),
    Affix(Vec<LexemeId>),
}

fn get_lemmas_nodes(
    rules: &RulePool,
    rule: RuleIndex,
    lexeme: LexemeId,
    stolen: StolenInfo,
) -> LemmaNode {
    match stolen {
        StolenInfo::Normal => LemmaNode::Normal(lexeme),
        StolenInfo::Stolen(_, _) => LemmaNode::Moved(lexeme),
        StolenInfo::Stealer => {
            let mut affixes = vec![(GornIndex::default(), lexeme)];
            affixes.extend(rules.iter().filter_map(|x| match x {
                Rule::Scan {
                    lexeme,
                    stolen: StolenInfo::Stolen(rule_index, index),
                } if *rule_index == rule => Some((*index, *lexeme)),
                _ => None,
            }));
            affixes.sort_by_key(|(i, _n)| std::cmp::Reverse(*i));
            LemmaNode::Affix(affixes.into_iter().map(|(_, n)| n).collect())
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn graph_helper<T, C>(
    current_rule: Rule,
    child_id: RuleIndex,
    complement_id: RuleIndex,
    g: &mut StableDiGraph<MgNode<T, C>, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
    nodes_h: &mut HashMap<NodeIndex, RuleIndex>,
) -> (NodeIndex, Vec<Feature<C>>, Vec<Mover<C>>, Option<TraceId>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let (complement_node, complement_features, mut complement_movers, comp_trace_id) =
        inner_to_graph(g, lex, rules, complement_id, trace_h, rules_h, nodes_h);

    let (child_node, mut features, mut movement, _) =
        inner_to_graph(g, lex, rules, child_id, trace_h, rules_h, nodes_h);

    movement.append(&mut complement_movers);
    match current_rule {
        Rule::UnmergeFromMover { trace_id, .. } => {
            movement.push(Mover {
                features: complement_features,
                trace_id,
                canceled: false,
            });

            //Make it so that the child has evidence of its use in movement.
            match g.node_weight_mut(complement_node).unwrap() {
                MgNode::Node { trace, .. } => *trace = Some(trace_id),
                MgNode::Leaf { trace, .. } => *trace = Some(trace_id),
                MgNode::Trace { .. } => panic!("impossible to do"),
            };
        }
        Rule::Unmove { .. } => {
            let trace_id = comp_trace_id.unwrap();
            let i = movement
                .iter()
                .find_position(|x| x.trace_id == trace_id)
                .unwrap()
                .0;
            movement.remove(i);
        }
        Rule::UnmoveFromMover { trace_id, .. } => {
            let comp_trace_id = comp_trace_id.unwrap();
            let mover = movement
                .iter_mut()
                .find(|x| x.trace_id == comp_trace_id)
                .unwrap();
            mover.features.remove(0);
            mover.trace_id = trace_id;

            //Make it so that the child has evidence of its use in movement.
            match g.node_weight_mut(complement_node).unwrap() {
                MgNode::Trace { new_trace, .. } => *new_trace = Some(trace_id),
                MgNode::Node { .. } | MgNode::Leaf { .. } => {
                    panic!("impossible to do!")
                }
            };
        }
        _ => (),
    }

    let complement_direction = match g.node_weight(child_node).unwrap() {
        MgNode::Node { features, .. } | MgNode::Leaf { features, .. } => {
            match features.first().unwrap() {
                Feature::Selector(_, direction) => Some(direction),
                Feature::Affix(_, _) => Some(&Direction::Right),
                _ => None,
            }
        }
        MgNode::Trace { .. } => panic!("a trace can't be a direct child"),
    };
    let (child_dir, complement_dir) = if let Some(dir) = complement_direction {
        (dir.flip(), *dir)
    } else {
        (Direction::Right, Direction::Left)
    };

    let node = g.add_node(MgNode::Node {
        features: features.clone(),
        movement: movement.clone(),
        root: false,
        trace: None,
    });

    g.add_edge(node, child_node, MGEdge::Merge(Some(child_dir)));
    g.add_edge(node, complement_node, MGEdge::Merge(Some(complement_dir)));

    features.remove(0);
    (node, features, movement, None)
}

#[allow(clippy::type_complexity)]
fn inner_to_graph<T, C>(
    g: &mut StableDiGraph<MgNode<T, C>, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
    nodes_h: &mut HashMap<NodeIndex, RuleIndex>,
) -> (NodeIndex, Vec<Feature<C>>, Vec<Mover<C>>, Option<TraceId>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let rule = rules.get(index);
    let (node, features, movers, trace_id) = match rule {
        Rule::UnmoveTrace(trace_id) => {
            let node = g.add_node(MgNode::Trace {
                trace: *trace_id,
                new_trace: None,
            });
            trace_h.entry(*trace_id).or_default().1 = Some(node);
            (node, vec![], vec![], Some(*trace_id))
        }
        Rule::UnmergeFromMover {
            trace_id,
            stored_id,
            child_id,
            ..
        }
        | Rule::UnmoveFromMover {
            trace_id,
            stored_id,
            child_id,
            ..
        } => {
            trace_h.entry(*trace_id).or_default().0 = Some(*stored_id);
            graph_helper(
                *rule, *child_id, *stored_id, g, lex, rules, trace_h, rules_h, nodes_h,
            )
        }
        Rule::Scan { lexeme, stolen } => {
            let lemma = get_lemmas_nodes(rules, index, *lexeme, *stolen);
            let LexicalEntry {
                lemma: _,
                mut features,
            } = lex.get_lexical_entry(*lexeme).unwrap();

            let lemma = match lemma {
                LemmaNode::Normal(node_index) => {
                    Lemma::Single(lex.leaf_to_lemma(node_index).unwrap().clone())
                }
                LemmaNode::Moved(node_index) => {
                    Lemma::Stolen(lex.leaf_to_lemma(node_index).unwrap().clone())
                }
                LemmaNode::Affix(items) => Lemma::Multi(
                    items
                        .into_iter()
                        .map(|nx| lex.leaf_to_lemma(nx).unwrap())
                        .cloned()
                        .collect(),
                ),
            };
            let node = g.add_node(MgNode::Leaf {
                root: false,
                lemma,
                features: features.clone(),
                movement: vec![],
                trace: None,
            });

            features.remove(0);
            (node, features, vec![], None)
        }
        Rule::Unmove {
            child_id,
            stored_id: complement_id,
        }
        | Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => graph_helper(
            *rule,
            *child_id,
            *complement_id,
            g,
            lex,
            rules,
            trace_h,
            rules_h,
            nodes_h,
        ),
        Rule::Start { child, .. } => {
            let (node, features, movers, trace_id) =
                inner_to_graph(g, lex, rules, *child, trace_h, rules_h, nodes_h);
            match g.node_weight_mut(node).unwrap() {
                MgNode::Node { root, .. } | MgNode::Leaf { root, .. } => *root = true,
                MgNode::Trace { .. } => (),
            };
            (node, features, movers, trace_id)
        }
    };

    rules_h.insert(index, node);
    nodes_h.entry(node).or_insert(index);
    (node, features, movers, trace_id)
}

#[cfg(test)]
mod test {
    use logprob::LogProb;

    use super::*;
    use crate::grammars::{COPY_LANGUAGE, STABLER2011};
    use crate::{ParsingConfig, PhonContent};

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
            println!("{}", rules.to_tree(&lex).to_latex());
            assert_eq!(
                rules.to_tree(&lex).to_latex(),
                "\\begin{forest}[\\der{C} [\\der{D -W}, name=nodet0 [\\plainlex{N= D -W}{which}] [\\plainlex{N}{wine}]] [\\der{+W C} [\\plainlex{V= +W C}{$\\epsilon$}] [\\der{V} [\\der{D} [\\plainlex{N= D}{the}] [\\plainlex{N}{queen}]] [\\der{=D V} [\\plainlex{D= =D V}{prefers}] [$t$, name=tracet0]]]]]\\end{forest}"
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
            println!("{}", rules.to_tree(&lex).to_latex());
            assert_eq!(
                rules.to_tree(&lex).to_latex(),
                "\\begin{forest}[\\der{T} [\\der{T -l}, name=nodet0 [\\der{T -l}, name=nodet2 [\\plainlex{T -r -l}{$\\epsilon$}] [\\der{+l T -l} [$t$, name=tracet3] [\\plainlex{=A +l T -l}{a}]]] [\\der{+l T -l} [$t$, name=tracet1] [\\plainlex{=B +l T -l}{b}]]] [\\der{+l T} [\\der{B -r}, name=nodet1 [\\der{A -r}, name=nodet3 [$t$, name=tracet4] [\\der{+r A -r} [$t$, name=tracet5] [\\plainlex{=T +r A -r}{a}]]] [\\der{+r B -r} [$t$, name=tracet2] [\\plainlex{=T +r B -r}{b}]]] [\\der{+r +l T} [$t$, name=tracet0] [\\plainlex{=T +r +l T}{$\\epsilon$}]]]]\\end{forest}"
            );
        }
        Ok(())
    }

    #[test]
    fn edge_ordering() -> anyhow::Result<()> {
        let order = vec![
            MGEdge::Move,
            MGEdge::Merge(None),
            MGEdge::Merge(Some(Direction::Left)),
            MGEdge::Merge(Some(Direction::Right)),
        ];
        let mut new_order = order.clone();
        new_order.sort();
        assert_eq!(new_order, order);
        Ok(())
    }
}
