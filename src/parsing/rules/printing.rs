use itertools::Itertools;
use petgraph::graph::NodeIndex;
use petgraph::prelude::DiGraphMap;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use serde::Serialize;
use serde::ser::SerializeSeq;
use serde::ser::SerializeStructVariant;

#[cfg(feature = "semantics")]
use simple_semantics::LabelledScenarios;

use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use thiserror::Error;

use crate::lexicon::Feature;
use crate::lexicon::FeatureOrLemma;
use crate::lexicon::LexicalEntry;
use crate::lexicon::Lexicon;

use super::{Rule, RuleIndex, RulePool, TraceId};
use crate::Direction;

#[cfg(feature = "semantics")]
use crate::lexicon::SemanticLexicon;

#[cfg(feature = "semantics")]
use super::semantics::{SemanticHistory, SemanticNode};

#[cfg(feature = "semantics")]
use regex::Regex;

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum MGEdge {
    Move,
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
        write!(f, "{}", s)
    }
}

///Awful helper function to make chains from pairs, e.g. [(a,b), (b,c), (c,d), (z, w)] ->
///[(a,b,c,d), (z,w)]
fn to_chains<I: Iterator<Item = (NodeIndex, NodeIndex)>>(
    x: I,
) -> impl Iterator<Item = Vec<NodeIndex>> {
    let mut trace_paths = DiGraphMap::<NodeIndex, ()>::new();
    x.map(|(a, b)| {
        trace_paths.add_node(a);
        trace_paths.add_node(b);
        trace_paths.add_edge(a, b, ());
        a
    })
    .collect_vec()
    .into_iter()
    .filter_map(move |mut x| {
        if trace_paths
            .edges_directed(x, petgraph::Direction::Incoming)
            .next()
            .is_none()
        {
            let mut path = vec![x];
            let get_child = |x| {
                trace_paths
                    .edges_directed(x, petgraph::Direction::Outgoing)
                    .next()
                    .map(|(_start, end, _edge_weight)| end)
            };
            while let Some(next) = get_child(x) {
                path.push(next);
                x = next;
            }
            Some(path)
        } else {
            None
        }
    })
}

impl RulePool {
    pub fn to_x_bar_graph<T, C>(&self, lex: &Lexicon<T, C>) -> StableDiGraph<String, MGEdge>
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let mut g = StableDiGraph::<String, MGEdge>::new();
        let mut trace_h = HashMap::new();
        let mut rule_h: HashMap<RuleIndex, NodeIndex> = HashMap::new();
        inner_to_x_bar_graph(&mut g, lex, self, RuleIndex(0), &mut trace_h, &mut rule_h);

        let paths = to_chains(trace_h.into_iter().filter_map(|(_, x)| {
            if let (Some(a), Some(b)) = x {
                let a = *rule_h.get(&a).unwrap();
                Some((a, b))
            } else {
                None
            }
        }));

        for mut path in paths {
            let parents = path
                .iter()
                .map(|x| {
                    let parent_edge = g
                        .edges_directed(*x, petgraph::Direction::Incoming)
                        .next()
                        .unwrap();
                    let parent = parent_edge.source();
                    let w = g.remove_edge(parent_edge.id()).unwrap();
                    (parent, w)
                })
                .collect_vec();
            path.rotate_left(1);

            for (x, (p, w)) in path.iter().zip(parents.into_iter()) {
                g.add_edge(p, *x, w);
            }
            for (x, y) in path.into_iter().tuple_windows() {
                g.add_edge(x, y, MGEdge::Move);
            }
        }
        g
    }

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
        FeatureOrLemma<T, C>: std::fmt::Display,
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
        let movements = trace_h
            .into_iter()
            .filter_map(|(_, x)| {
                if let (Some(unmove_origin_rule), Some(trace_node)) = x {
                    Some((*rule_h.get(&unmove_origin_rule).unwrap(), trace_node))
                } else {
                    None
                }
            })
            .collect_vec();

        for (_trace_origin, trace_node) in movements.iter() {
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

        for (trace_origin, trace_node) in movements.into_iter() {
            g.add_edge(trace_origin, trace_node, MGEdge::Move);
        }
        (g, root, nodes_h)
    }

    #[cfg(feature = "semantics")]
    pub fn to_semantic_json<T, C>(
        &self,
        semantic_lex: &SemanticLexicon<T, C>,
        history: &SemanticHistory,
    ) -> String
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
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

        serde_json::to_string(&Tree::new_semantic(&g, root)).unwrap()
    }

    pub fn to_json<T, C>(&self, lex: &Lexicon<T, C>) -> String
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
    {
        let (g, root, _) = self.to_graph(lex);
        serde_json::to_string(&Tree::new(&g, root)).unwrap()
    }

    #[cfg(feature = "semantics")]
    pub fn to_semantic_latex<T, C>(
        &self,
        semantic_lex: &SemanticLexicon<T, C>,
        history: &SemanticHistory,
    ) -> String
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let (g, root, node_to_rule) = self.to_graph(semantic_lex.lexicon());
        let g = g.map(
            |i, node| SemanticMGNode {
                node: node.clone(),
                semantic: history
                    .semantic_node(*node_to_rule.get(&i).unwrap())
                    .unwrap(),
            },
            |_, e| *e,
        );
        self.inner_latex(&g, root)
    }

    fn inner_latex<N, T, C>(&self, g: &StableDiGraph<N, MGEdge>, root: NodeIndex) -> String
    where
        N: LaTeXify,
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let mut g = g.map(|_, n| n.to_latex(), |_, e| *e);
        let movement_edges = g
            .edge_references()
            .filter_map(|x| {
                if matches!(x.weight(), MGEdge::Move) {
                    Some([x.source(), x.target()])
                } else {
                    None
                }
            })
            .collect_vec();

        movement_edges
            .iter()
            .flatten()
            .unique()
            .sorted()
            .for_each(|x| {
                g.node_weight_mut(*x)
                    .unwrap()
                    .push_str(&format!(",name=node{}", x.index()));
            });

        let mut s = String::new();
        recursive_latex(&g, root, &mut s, 0);
        s = format!("\\begin{{forest}}\n{s}");
        for [a, b] in movement_edges.into_iter().sorted() {
            s.push_str(&format!(
                "\n\\draw[densely dotted,->] (node{}) to[out=west,in=south west] (node{});",
                a.index(),
                b.index()
            ))
        }
        s.push_str("\n\\end{forest}");
        s
    }

    pub fn to_latex<T, C>(&self, lex: &Lexicon<T, C>) -> String
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let (g, root, _) = self.to_graph(lex);
        self.inner_latex(&g, root)
    }

    pub fn to_tree<T, C>(
        &self,
        lex: &Lexicon<T, C>,
    ) -> (StableDiGraph<MgNode<T, C>, MGEdge>, NodeIndex)
    where
        FeatureOrLemma<T, C>: std::fmt::Display,
        T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
        C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    {
        let (g, root, _) = self.to_graph(lex);

        (g, root)
    }
}

fn recursive_latex(
    g: &StableDiGraph<String, MGEdge>,
    node: NodeIndex,
    s: &mut String,
    depth: usize,
) {
    let indent = (0..depth).map(|_| '\t').collect::<String>();
    s.push_str(&format!("{}[{}", indent, g.node_weight(node).unwrap()));
    let children = get_children(g, node);
    if !children.is_empty() {
        s.push('\n');
        let n_children = children.len();
        for (i, child) in children.into_iter().enumerate() {
            recursive_latex(g, child, s, depth + 1);
            if i < n_children - 1 {
                s.push('\n');
            }
        }
    }
    s.push_str(" ]");
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

enum Tree<'a, T, C: Eq + Display> {
    Node {
        node: MgNode<T, C>,

        #[cfg(feature = "semantics")]
        semantics: Option<SemanticNode<'a>>,
    },
    Children(Vec<Tree<'a, T, C>>),
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
        match self {
            Tree::Node {
                node,
                #[cfg(feature = "semantics")]
                semantics,
                ..
            } => {
                match node {
                    MgNode::Node {
                        features,
                        movement,
                        trace,
                        ..
                    } => {
                        #[cfg(not(feature = "semantics"))]
                        let n = 3;

                        #[cfg(feature = "semantics")]
                        let n = if semantics.is_some() { 4 } else { 3 };

                        let mut seq =
                            serializer.serialize_struct_variant("MgNode", 0, "Node", n)?;

                        seq.serialize_field("features", features)?;
                        seq.serialize_field("movement", movement)?;
                        seq.serialize_field("trace", trace)?;

                        #[cfg(feature = "semantics")]
                        if let Some(semantics) = semantics {
                            seq.serialize_field("semantics", semantics)?;
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
                        let n = if semantics.is_some() { 5 } else { 4 };

                        let mut seq =
                            serializer.serialize_struct_variant("MgNode", 1, "Leaf", n)?;

                        seq.serialize_field("features", features)?;
                        seq.serialize_field("movement", movement)?;
                        seq.serialize_field("lemma", lemma)?;
                        seq.serialize_field("trace", trace)?;

                        #[cfg(feature = "semantics")]
                        if let Some(semantics) = semantics {
                            seq.serialize_field("semantics", semantics)?;
                        }

                        seq.end()
                    }

                    MgNode::Trace { trace, new_trace } => {
                        #[cfg(not(feature = "semantics"))]
                        let n = 2;

                        #[cfg(feature = "semantics")]
                        let n = if semantics.is_some() { 3 } else { 2 };

                        let mut seq =
                            serializer.serialize_struct_variant("MgNode", 2, "Trace", n)?;

                        seq.serialize_field("trace", trace)?;
                        seq.serialize_field("new_trace", new_trace)?;

                        #[cfg(feature = "semantics")]
                        if let Some(semantics) = semantics {
                            seq.serialize_field("semantics", semantics)?;
                        }

                        seq.end()
                    }
                }

                //typst_data.serialize(serializer)?;
            }

            Tree::Children(trees) => {
                let mut seq = serializer.serialize_seq(Some(trees.len()))?;
                for tree in trees {
                    seq.serialize_element(tree)?;
                }
                seq.end()
            }
        }
    }
}

impl<T, C> Tree<'_, T, C>
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display + Serialize,
    MgNode<T, C>: Serialize,
{
    fn new(g: &StableDiGraph<MgNode<T, C>, MGEdge>, x: NodeIndex) -> Tree<T, C> {
        let node = g.node_weight(x).unwrap();
        let children: Vec<_> = get_children(g, x);

        let node = Tree::Node {
            node: node.clone(),

            #[cfg(feature = "semantics")]
            semantics: None,
        };
        if children.is_empty() {
            node
        } else {
            let mut row = vec![node];
            row.extend(children.into_iter().map(|x| Self::new(g, x)));
            Tree::Children(row)
        }
    }

    #[cfg(feature = "semantics")]
    fn new_semantic<'a>(
        g: &'a StableDiGraph<SemanticMGNode<T, C>, MGEdge>,
        x: NodeIndex,
    ) -> Tree<'a, T, C> {
        let node = g.node_weight(x).unwrap();
        let children: Vec<_> = get_children(g, x);

        let node = Tree::Node {
            node: node.node.clone(),
            semantics: Some(node.semantic.clone()),
        };
        if children.is_empty() {
            node
        } else {
            let mut row = vec![node];
            row.extend(children.into_iter().map(|x| Self::new_semantic(g, x)));
            Tree::Children(row)
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

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct Mover<C: Eq + Display> {
    trace_id: TraceId,
    canceled: bool,
    features: Vec<Feature<C>>,
}

impl<C: Eq + std::fmt::Display> Display for Mover<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.features.len() {
            0 => write!(f, ""),
            1 => write!(
                f,
                "\\mover{{{}}}{{{}}}",
                match self.canceled {
                    true => format!("\\cancel{{{}}}", self.features.first().unwrap()),
                    false => self.features.first().unwrap().to_string(),
                },
                self.trace_id.0
            ),
            _ => write!(
                f,
                "\\mover[{{{}}}]{{{}}}{{{}}}",
                self.features
                    .iter()
                    .skip(1)
                    .map(|x| x.to_string())
                    .join(" "),
                match self.canceled {
                    true => format!("\\cancel{{{}}}", self.features.first().unwrap()),
                    false => self.features.first().unwrap().to_string(),
                },
                self.trace_id.0,
            ),
        }
    }
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum MgNode<T, C: Eq + Display> {
    Node {
        features: Vec<Feature<C>>,
        movement: Vec<Mover<C>>,
        trace: Option<TraceId>,
        #[serde(skip_serializing)]
        root: bool,
    },
    Leaf {
        lemma: Option<T>,
        features: Vec<Feature<C>>,
        movement: Vec<Mover<C>>,
        trace: Option<TraceId>,
        #[serde(skip_serializing)]
        root: bool,
    },
    Trace {
        trace: TraceId,
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
    pub fn features(&self) -> &[Feature<C>] {
        match self {
            MgNode::Node { features, .. } => features,
            MgNode::Leaf { features, .. } => features,
            MgNode::Trace { .. } => &[],
        }
    }

    pub fn trace(&self) -> Result<TraceId, PrintError> {
        match self {
            MgNode::Trace { trace, .. } => Ok(*trace),
            MgNode::Node { .. } | MgNode::Leaf { .. } => Err(PrintError::NotATrace),
        }
    }

    pub fn lemma(&self) -> Result<Option<&T>, PrintError> {
        match self {
            MgNode::Leaf { lemma, .. } => Ok(lemma.as_ref()),
            MgNode::Node { .. } | MgNode::Trace { .. } => Err(PrintError::NotALeaf),
        }
    }
}

#[cfg(feature = "semantics")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticMGNode<'a, T, C: Eq + Display> {
    node: MgNode<T, C>,
    semantic: SemanticNode<'a>,
}

fn feature_vec_to_string<C: Eq + Display>(v: &[Feature<C>], canceled_features: bool) -> String {
    v.iter()
        .enumerate()
        .map(|(i, x)| {
            if i == 0 && canceled_features {
                format!("\\cancel{{{}}}", x)
            } else {
                x.to_string()
            }
        })
        .join(" ")
}

trait LaTeXify {
    fn to_latex(&self) -> String;
}

#[cfg(feature = "semantics")]
fn clean_up_expr(s: String) -> String {
    let re = Regex::new(r"lambda (?<t>[eat,< >]+) ").unwrap();
    let s = s.replace("&", "\\&").replace("_", "\\_");
    re.replace_all(s.as_str(), "{$\\lambda_{$t}$}")
        .to_string()
        .replace("<", "\\left\\langle ")
        .replace(">", "\\right\\rangle ")
}

#[cfg(feature = "semantics")]
impl<T: Eq + Debug + Display, C: Eq + Debug + Display> LaTeXify for SemanticMGNode<'_, T, C> {
    fn to_latex(&self) -> String {
        match (&self.node, &self.semantic) {
            (
                MgNode::Node {
                    features,
                    movement,
                    root,
                    ..
                },
                SemanticNode::Simple(_),
            ) => format!(
                "{{\\rulesemder{}{{{}}}{{{}}} }}",
                if !movement.is_empty() {
                    format!(
                        "[{{{}}}]",
                        movement.iter().map(|x| x.to_string()).join(", ")
                    )
                } else {
                    "".to_string()
                },
                feature_vec_to_string(features, !root),
                self.semantic
            ),
            (
                MgNode::Node {
                    features,
                    movement,
                    root,
                    ..
                },
                SemanticNode::Rich(_, _, _),
            ) => format!(
                "{{\\semder{}{{{}}}{{{}}} }}",
                if !movement.is_empty() {
                    format!(
                        "[{{{}}}]",
                        movement.iter().map(|x| x.to_string()).join(", ")
                    )
                } else {
                    "".to_string()
                },
                feature_vec_to_string(features, !root),
                clean_up_expr(self.semantic.to_string())
            ),
            (
                MgNode::Leaf {
                    lemma,
                    features,
                    root,
                    ..
                },
                SemanticNode::Simple(_),
            ) => {
                format!(
                    "{{\\plainlex{{{}}}{{{}}} }}",
                    match lemma {
                        Some(x) => x.to_string(),
                        None => "$\\epsilon$".to_string(),
                    },
                    feature_vec_to_string(features, !root),
                )
            }
            (
                MgNode::Leaf {
                    lemma,
                    features,
                    root,
                    ..
                },
                SemanticNode::Rich(_, _, _),
            ) => {
                format!(
                    "{{\\semlex{{{}}}{{{}}}{{{}}} }}",
                    match lemma {
                        Some(x) => x.to_string(),
                        None => "$\\epsilon$".to_string(),
                    },
                    feature_vec_to_string(features, !root),
                    clean_up_expr(self.semantic.to_string())
                )
            }
            (MgNode::Trace { trace, .. }, _) => format!("{{${}$}}", trace),
        }
    }
}

impl<T: Eq + Debug + Display, C: Eq + Debug + Display> LaTeXify for MgNode<T, C> {
    fn to_latex(&self) -> String {
        match self {
            MgNode::Node {
                features,
                movement,
                root,
                ..
            } => format!(
                "{{\\der{}{{{}}}}}",
                if !movement.is_empty() {
                    format!(
                        "[{{{}}}]",
                        movement.iter().map(|x| x.to_string()).join(", ")
                    )
                } else {
                    "".to_string()
                },
                feature_vec_to_string(features, !root),
            ),
            MgNode::Leaf {
                lemma,
                features,
                root,
                ..
            } => {
                format!(
                    "{{\\plainlex{{{}}}{{{}}}}}",
                    match lemma {
                        Some(x) => x.to_string(),
                        None => "$\\epsilon$".to_string(),
                    },
                    feature_vec_to_string(features, !root),
                )
            }
            MgNode::Trace { trace, .. } => format!("{{${}$}}", trace),
        }
    }
}

fn x_bar_helper<T, C>(
    child_id: RuleIndex,
    complement_id: RuleIndex,
    g: &mut StableDiGraph<String, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
) -> (NodeIndex, Vec<Feature<C>>, Option<C>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let (child_node, mut child_features, child_category) =
        inner_to_x_bar_graph(g, lex, rules, child_id, trace_h, rules_h);
    let feature = child_features.pop();

    let (complement_node, _, _) =
        inner_to_x_bar_graph(g, lex, rules, complement_id, trace_h, rules_h);

    let node = if let Some(Feature::Category(_)) = child_features.last() {
        child_features.pop();
        g.add_node(format!("{}P", child_category.as_ref().unwrap()))
    } else {
        g.add_node(format!("{}'", child_category.as_ref().unwrap()))
    };

    let (child_dir, complement_dir) = if let Some(Feature::Selector(_, dir)) = feature {
        (dir.flip(), dir)
    } else {
        (Direction::Left, Direction::Right)
    };

    g.add_edge(node, child_node, MGEdge::Merge(Some(child_dir)));
    g.add_edge(node, complement_node, MGEdge::Merge(Some(complement_dir)));

    (node, child_features, child_category)
}

fn inner_to_x_bar_graph<T, C>(
    g: &mut StableDiGraph<String, MGEdge>,
    lex: &Lexicon<T, C>,
    rules: &RulePool,
    index: RuleIndex,
    trace_h: &mut HashMap<TraceId, (Option<RuleIndex>, Option<NodeIndex>)>,
    rules_h: &mut HashMap<RuleIndex, NodeIndex>,
) -> (NodeIndex, Vec<Feature<C>>, Option<C>)
where
    FeatureOrLemma<T, C>: std::fmt::Display,
    T: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
    C: Eq + std::fmt::Debug + std::clone::Clone + std::fmt::Display,
{
    let (mut node, features, category) = match rules.get(index) {
        Rule::UnmoveTrace(trace_id) => {
            let node = g.add_node(trace_id.to_string());
            trace_h.entry(*trace_id).or_default().1 = Some(node);
            (node, vec![], None)
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
            x_bar_helper(*child_id, *stored_id, g, lex, rules, trace_h, rules_h)
        }
        Rule::Scan { node } => {
            let mut lexeme = lex.get_lexical_entry(*node).unwrap();
            let category = lexeme.category().clone();
            let node = g.add_node(match lexeme.lemma {
                Some(x) => x.to_string(),
                None => "ε".to_string(),
            });

            let parent = g.add_node(category.to_string());
            g.add_edge(parent, node, MGEdge::Merge(None));

            lexeme.features.reverse();
            (parent, lexeme.features, Some(category))
        }
        Rule::Unmove {
            child_id,
            stored_id: complement_id,
        }
        | Rule::Unmerge {
            child_id,
            complement_id,
            ..
        } => x_bar_helper(*child_id, *complement_id, g, lex, rules, trace_h, rules_h),
        Rule::Start { child, .. } => {
            let (node, _, category) = inner_to_x_bar_graph(g, lex, rules, *child, trace_h, rules_h);

            (node, vec![], category)
        }
    };
    if let Some(Feature::Category(c)) = features.last() {
        let new_node = g.add_node(format!("{c}P"));
        g.add_edge(new_node, node, MGEdge::Merge(None));
        node = new_node
    }

    rules_h.insert(index, node);

    (node, features, category)
}

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
        Rule::Scan { node } => {
            let LexicalEntry {
                lemma,
                mut features,
            } = lex.get_lexical_entry(*node).unwrap();
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
    use crate::ParsingConfig;
    use crate::grammars::{COPY_LANGUAGE, STABLER2011};
    use petgraph::dot::Dot;

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
                .parse(&sentence.split(' ').collect::<Vec<_>>(), "C", &config)?
                .next()
                .unwrap();
            let (g, _, _) = rules.to_graph(&lex);
            let g = g.map(|_, n| n.to_latex(), |_, e| e);
            let dot = Dot::new(&g);
            println!("{dot}");
            println!("{}", rules.to_latex(&lex));
            assert_eq!(
                rules.to_latex(&lex),
                "\\begin{forest}\n[{\\der{C}}\n\t[{$t0$},name=node0 ]\n\t[{\\der[{\\mover{\\cancel{-W}}{0}}]{\\cancel{+W} C}}\n\t\t[{\\plainlex{$\\epsilon$}{\\cancel{V=} +W C}} ]\n\t\t[{\\der[{\\mover{-W}{0}}]{\\cancel{V}}}\n\t\t\t[{\\der{\\cancel{D}}}\n\t\t\t\t[{\\plainlex{the}{\\cancel{N=} D}} ]\n\t\t\t\t[{\\plainlex{queen}{\\cancel{N}}} ] ]\n\t\t\t[{\\der[{\\mover{-W}{0}}]{\\cancel{=D} V}}\n\t\t\t\t[{\\plainlex{prefers}{\\cancel{D=} =D V}} ]\n\t\t\t\t[{\\der{\\cancel{D} -W}},name=node6\n\t\t\t\t\t[{\\plainlex{which}{\\cancel{N=} D -W}} ]\n\t\t\t\t\t[{\\plainlex{wine}{\\cancel{N}}} ] ] ] ] ] ]\n\\draw[densely dotted,->] (node6) to[out=west,in=south west] (node0);\n\\end{forest}"
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
                .parse(&sentence.split(' ').collect::<Vec<_>>(), "T", &config)?
                .next()
                .unwrap();
            let (g, _, _) = rules.to_graph(&lex);
            let g = g.map(|_, n| n.to_latex(), |_, e| e);
            let dot = Dot::new(&g);
            println!("{dot}");
            println!("{}", rules.to_latex(&lex));
            assert_eq!(
                rules.to_latex(&lex),
                "\\begin{forest}\n[{\\der{T}}\n\t[{$t0$},name=node0 ]\n\t[{\\der[{\\mover{\\cancel{-l}}{0}}]{\\cancel{+l} T}}\n\t\t[{$t1$},name=node1 ]\n\t\t[{\\der[{\\mover{\\cancel{-r}}{1}, \\mover{-l}{0}}]{\\cancel{+r} +l T}}\n\t\t\t[{\\der[{\\mover{-r}{1}}]{\\cancel{T} -l}},name=node18\n\t\t\t\t[{$t2$},name=node2 ]\n\t\t\t\t[{\\der[{\\mover{\\cancel{-l}}{2}, \\mover{-r}{1}}]{\\cancel{+l} T -l}}\n\t\t\t\t\t[{\\der[{\\mover{-l}{2}}]{\\cancel{B} -r}},name=node15\n\t\t\t\t\t\t[{$t3$},name=node3 ]\n\t\t\t\t\t\t[{\\der[{\\mover{\\cancel{-r}}{3}, \\mover{-l}{2}}]{\\cancel{+r} B -r}}\n\t\t\t\t\t\t\t[{\\der[{\\mover{-r}{3}}]{\\cancel{T} -l}},name=node12\n\t\t\t\t\t\t\t\t[{$t4$},name=node4 ]\n\t\t\t\t\t\t\t\t[{\\der[{\\mover{\\cancel{-l}}{4}, \\mover{-r}{3}}]{\\cancel{+l} T -l}}\n\t\t\t\t\t\t\t\t\t[{\\der[{\\mover{-l}{4}}]{\\cancel{A} -r}},name=node9\n\t\t\t\t\t\t\t\t\t\t[{$t5$},name=node5 ]\n\t\t\t\t\t\t\t\t\t\t[{\\der[{\\mover[{-l}]{\\cancel{-r}}{5}}]{\\cancel{+r} A -r}}\n\t\t\t\t\t\t\t\t\t\t\t[{\\plainlex{$\\epsilon$}{\\cancel{T} -r -l}},name=node6 ]\n\t\t\t\t\t\t\t\t\t\t\t[{\\plainlex{a}{\\cancel{=T} +r A -r}} ] ] ]\n\t\t\t\t\t\t\t\t\t[{\\plainlex{a}{\\cancel{=A} +l T -l}} ] ] ]\n\t\t\t\t\t\t\t[{\\plainlex{b}{\\cancel{=T} +r B -r}} ] ] ]\n\t\t\t\t\t[{\\plainlex{b}{\\cancel{=B} +l T -l}} ] ] ]\n\t\t\t[{\\plainlex{$\\epsilon$}{\\cancel{=T} +r +l T}} ] ] ] ]\n\\draw[densely dotted,->] (node5) to[out=west,in=south west] (node4);\n\\draw[densely dotted,->] (node6) to[out=west,in=south west] (node5);\n\\draw[densely dotted,->] (node9) to[out=west,in=south west] (node3);\n\\draw[densely dotted,->] (node12) to[out=west,in=south west] (node2);\n\\draw[densely dotted,->] (node15) to[out=west,in=south west] (node1);\n\\draw[densely dotted,->] (node18) to[out=west,in=south west] (node0);\n\\end{forest}"
            );
        }
        Ok(())
    }

    #[test]
    fn no_movement_xbar() -> anyhow::Result<()> {
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lex = Lexicon::from_string(STABLER2011)?;
        let rules = lex
            .parse(
                &"the queen prefers the wine".split(' ').collect::<Vec<_>>(),
                "C",
                &config,
            )?
            .next()
            .unwrap()
            .2;
        let parse = rules.to_x_bar_graph(&lex);
        let dot = Dot::new(&parse);
        println!("{}", dot);
        assert_eq!(
            dot.to_string(),
            "digraph {\n    0 [ label = \"ε\" ]\n    1 [ label = \"C\" ]\n    2 [ label = \"prefers\" ]\n    3 [ label = \"V\" ]\n    4 [ label = \"the\" ]\n    5 [ label = \"D\" ]\n    6 [ label = \"wine\" ]\n    7 [ label = \"N\" ]\n    8 [ label = \"NP\" ]\n    9 [ label = \"DP\" ]\n    10 [ label = \"V'\" ]\n    11 [ label = \"the\" ]\n    12 [ label = \"D\" ]\n    13 [ label = \"queen\" ]\n    14 [ label = \"N\" ]\n    15 [ label = \"NP\" ]\n    16 [ label = \"DP\" ]\n    17 [ label = \"VP\" ]\n    18 [ label = \"CP\" ]\n    1 -> 0 [ label = \"\" ]\n    3 -> 2 [ label = \"\" ]\n    5 -> 4 [ label = \"\" ]\n    7 -> 6 [ label = \"\" ]\n    8 -> 7 [ label = \"\" ]\n    9 -> 5 [ label = \"\" ]\n    9 -> 8 [ label = \"\" ]\n    10 -> 3 [ label = \"\" ]\n    10 -> 9 [ label = \"\" ]\n    12 -> 11 [ label = \"\" ]\n    14 -> 13 [ label = \"\" ]\n    15 -> 14 [ label = \"\" ]\n    16 -> 12 [ label = \"\" ]\n    16 -> 15 [ label = \"\" ]\n    17 -> 10 [ label = \"\" ]\n    17 -> 16 [ label = \"\" ]\n    18 -> 1 [ label = \"\" ]\n    18 -> 17 [ label = \"\" ]\n}\n"
        );
        Ok(())
    }

    #[test]
    fn tree_building() -> anyhow::Result<()> {
        let config = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lex = Lexicon::from_string(STABLER2011)?;
        let rules = lex
            .parse(
                &"which queen prefers the wine"
                    .split(' ')
                    .collect::<Vec<_>>(),
                "C",
                &config,
            )?
            .next()
            .unwrap()
            .2;
        let parse = rules.to_x_bar_graph(&lex);
        let dot = Dot::new(&parse);
        println!("{}", dot);
        assert_eq!(
            dot.to_string(),
            "digraph {\n    0 [ label = \"ε\" ]\n    1 [ label = \"C\" ]\n    2 [ label = \"prefers\" ]\n    3 [ label = \"V\" ]\n    4 [ label = \"the\" ]\n    5 [ label = \"D\" ]\n    6 [ label = \"wine\" ]\n    7 [ label = \"N\" ]\n    8 [ label = \"NP\" ]\n    9 [ label = \"DP\" ]\n    10 [ label = \"V'\" ]\n    11 [ label = \"which\" ]\n    12 [ label = \"D\" ]\n    13 [ label = \"queen\" ]\n    14 [ label = \"N\" ]\n    15 [ label = \"NP\" ]\n    16 [ label = \"DP\" ]\n    17 [ label = \"VP\" ]\n    18 [ label = \"C'\" ]\n    19 [ label = \"t0\" ]\n    20 [ label = \"CP\" ]\n    1 -> 0 [ label = \"\" ]\n    3 -> 2 [ label = \"\" ]\n    5 -> 4 [ label = \"\" ]\n    7 -> 6 [ label = \"\" ]\n    8 -> 7 [ label = \"\" ]\n    9 -> 5 [ label = \"\" ]\n    9 -> 8 [ label = \"\" ]\n    10 -> 3 [ label = \"\" ]\n    10 -> 9 [ label = \"\" ]\n    12 -> 11 [ label = \"\" ]\n    14 -> 13 [ label = \"\" ]\n    15 -> 14 [ label = \"\" ]\n    16 -> 12 [ label = \"\" ]\n    16 -> 15 [ label = \"\" ]\n    17 -> 10 [ label = \"\" ]\n    20 -> 16 [ label = \"\" ]\n    18 -> 1 [ label = \"\" ]\n    18 -> 17 [ label = \"\" ]\n    20 -> 18 [ label = \"\" ]\n    17 -> 19 [ label = \"\" ]\n    19 -> 16 [ label = \"move\" ]\n}\n"
        );
        Ok(())
    }

    #[test]
    fn multimove() -> anyhow::Result<()> {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );

        let lex = Lexicon::from_string("a::v= +k +q z\nb::v -k -q")?;
        let (_, _s, r) = lex.parse(&["b", "a"], "z", &config)?.next().unwrap();
        let g = r.to_x_bar_graph(&lex);
        println!("{}", Dot::new(&g));
        assert_eq!(
            Dot::new(&g).to_string(),
            "digraph {\n    0 [ label = \"a\" ]\n    1 [ label = \"z\" ]\n    2 [ label = \"b\" ]\n    3 [ label = \"v\" ]\n    4 [ label = \"vP\" ]\n    5 [ label = \"z'\" ]\n    6 [ label = \"t1\" ]\n    7 [ label = \"z'\" ]\n    8 [ label = \"t0\" ]\n    9 [ label = \"zP\" ]\n    1 -> 0 [ label = \"\" ]\n    3 -> 2 [ label = \"\" ]\n    4 -> 3 [ label = \"\" ]\n    5 -> 1 [ label = \"\" ]\n    9 -> 4 [ label = \"\" ]\n    7 -> 5 [ label = \"\" ]\n    7 -> 8 [ label = \"\" ]\n    9 -> 7 [ label = \"\" ]\n    5 -> 6 [ label = \"\" ]\n    6 -> 8 [ label = \"move\" ]\n    8 -> 4 [ label = \"move\" ]\n}\n"
        );
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
