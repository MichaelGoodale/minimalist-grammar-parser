use crate::{
    Direction,
    lexicon::Feature,
    parsing::{
        RuleIndex,
        rules::{
            Derivation, TraceId,
            printing::{Lemma, MgNode, Storage},
        },
    },
};
use ahash::HashMap;
use itertools::Itertools;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{
    Serialize,
    ser::{SerializeSeq, SerializeStructVariant},
};
use std::fmt::{Debug, Display};
use std::{collections::VecDeque, hash::Hash};

#[cfg(not(feature = "semantics"))]
use std::marker::PhantomData;

#[cfg(feature = "semantics")]
use regex::Regex;

#[cfg(feature = "semantics")]
use super::semantics::SemanticNode;

///A structured tree representation of a parse.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tree<'src, T, C: Eq + Display> {
    node: TreeNode<'src, T, C>,
    children: Vec<Tree<'src, T, C>>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum TreeEdge {
    Merge(Direction),
    Move,
}

impl Display for TreeEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeEdge::Merge(Direction::Left) => write!(f, "Merge(Left)"),
            TreeEdge::Merge(Direction::Right) => write!(f, "Merge(Right)"),
            TreeEdge::Move => write!(f, "Move"),
        }
    }
}

impl<'src, T: Clone, C: Clone + Eq + Display> Tree<'src, T, C> {
    ///Get a parse as a [petgraph](https://crates.io/crates/petgraph) DiGraph
    pub fn petgraph(
        &self,
        d: &Derivation<'src, T, C>,
    ) -> (DiGraph<TreeNode<'src, T, C>, TreeEdge>, NodeIndex) {
        let mut h = HashMap::default();
        let mut g = DiGraph::new();
        let root = g.add_node(self.node.clone());
        h.insert(self.node.rule, root);
        let mut stack: VecDeque<_> = self
            .children
            .iter()
            .enumerate()
            .map(|(i, x)| (i, x, root))
            .collect();
        while let Some((pos, tree, par)) = stack.pop_front() {
            let node = g.add_node(tree.node.clone());
            h.insert(tree.node.rule, node);
            g.add_edge(
                par,
                node,
                match pos {
                    0 => TreeEdge::Merge(Direction::Left),
                    1 => TreeEdge::Merge(Direction::Right),
                    _ => panic!("The library should only have binary branching!"),
                },
            );
            stack.extend(tree.children.iter().enumerate().map(|(i, x)| (i, x, node)));
        }
        for x in d.movement.trace_origins.iter() {
            let mut x = x.clone();
            x.rotate_right(1);
            for (src, dst) in x.iter().tuple_windows::<(_, _)>() {
                let src = *h.get(src).unwrap();
                let dst = *h.get(dst).unwrap();
                g.add_edge(src, dst, TreeEdge::Move);
            }
        }

        (g, root)
    }
}

impl<'src, T, C: Eq + Display> Tree<'src, T, C> {
    pub(crate) fn new(
        node: MgNode<T, C>,
        storage: Storage<C>,
        children: Vec<Tree<'src, T, C>>,
        rule: RuleIndex,
    ) -> Self {
        Tree {
            node: TreeNode::new(node, storage, rule),
            children,
        }
    }

    #[cfg(feature = "semantics")]
    pub(crate) fn new_with_semantics(
        node: MgNode<T, C>,
        semantic_node: Option<SemanticNode<'src>>,
        storage: Storage<C>,
        children: Vec<Tree<'src, T, C>>,
        rule: RuleIndex,
    ) -> Self {
        Tree {
            node: TreeNode::new_semantics(node, storage, semantic_node, rule),
            children,
        }
    }

    ///The storage of the tree at the end (will be empty if the tree is complete)
    pub fn storage(&self) -> &Storage<C> {
        &self.node.storage
    }
}
impl<'src, T: Display, C: Eq + Display> Tree<'src, T, C> {
    ///The representation of the tree as a LaTeX forest tree. Requires using [the following commands in the preamble](https://github.com/MichaelGoodale/python-mg/blob/master/latex-commands.tex)
    pub fn latex(&self) -> String {
        format!("\\begin{{forest}}{}\\end{{forest}}", self.latex_inner())
    }
    fn latex_inner(&self) -> String {
        let node = self.node.latex();

        let children: Vec<_> = self.children.iter().map(|x| x.latex_inner()).collect();
        if children.is_empty() {
            format!("[{node}]")
        } else {
            format!("[{node} {}]", children.join(" "))
        }
    }
}

///A node inside of a [`Tree`] which contains information about the features of a current lexical
///entry, anything in storage as well as its semantic information, if the semantics feature is
///enabled.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TreeNode<'src, T, C: Eq + Display> {
    node: MgNode<T, C>,
    rule: RuleIndex,

    storage: Storage<C>,

    #[cfg(feature = "semantics")]
    semantics: Option<SemanticNode<'src>>,

    #[cfg(not(feature = "semantics"))]
    semantics: PhantomData<&'src ()>,
}

impl<'src, T: Display, C: Eq + Display> Display for TreeNode<'src, T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.node {
            MgNode::Start => {
                write!(f, "Start")?;
            }
            MgNode::Node { features } => {
                write!(f, "{}", features.iter().join(" "))?;
                #[cfg(feature = "semantics")]
                if let Some(SemanticNode::Rich(_, Some(state))) = &self.semantics {
                    write!(f, "::{state}")?;
                }
            }

            MgNode::Leaf { lemma, features } => {
                write!(f, "{}::{}", lemma, features.iter().join(" "))?;

                #[cfg(feature = "semantics")]
                if let Some(SemanticNode::Rich(_, Some(state))) = &self.semantics {
                    write!(f, "::{state}")?;
                }
            }
            MgNode::Trace { trace } => {
                write!(f, "{trace}")?;
            }
        }
        Ok(())
    }
}

impl<T: Display> Lemma<T> {
    fn to_string(&self, empty_string: &str, join: &str) -> String {
        match self {
            Lemma::Single(Some(x)) => x.to_string(),
            Lemma::Single(None) => empty_string.to_string(),
            Lemma::Multi { heads, .. } => heads
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

impl<'src, T: Display, C: Eq + Display> TreeNode<'src, T, C> {
    fn latex(&self) -> String {
        match &self.node {
            MgNode::Node { features } => {
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
            }
            MgNode::Start => "\\textsc{Start}".to_string(),
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
            MgNode::Trace { trace } => format!("$t_{}$", trace.0),
        }
    }
}

impl<'src, T, C: Eq + Display> TreeNode<'src, T, C> {
    fn new(node: MgNode<T, C>, storage: Storage<C>, rule: RuleIndex) -> TreeNode<'static, T, C> {
        TreeNode {
            node,
            rule,
            storage,

            #[cfg(feature = "semantics")]
            semantics: None,

            #[cfg(not(feature = "semantics"))]
            semantics: PhantomData,
        }
    }

    #[cfg(feature = "semantics")]
    fn new_semantics(
        node: MgNode<T, C>,
        storage: Storage<C>,
        semantics: Option<SemanticNode<'src>>,
        rule: RuleIndex,
    ) -> Self {
        TreeNode {
            node,
            rule,
            storage,
            semantics,
        }
    }
}

impl<C: Eq + Display> Serialize for Feature<C> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
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

impl<C: Eq + Display + Clone> Serialize for Storage<C> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for mover in self.values() {
            seq.serialize_element(
                &mover
                    .iter()
                    .map(|c| Feature::Licensee(c.clone()))
                    .collect::<Vec<_>>(),
            )?;
        }
        seq.end()
    }
}

impl<T, C: Eq + Clone> Serialize for TreeNode<'_, T, C>
where
    C: Display,
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.node {
            MgNode::Node { features, .. } => {
                #[cfg(not(feature = "semantics"))]
                let n = 3;

                #[cfg(feature = "semantics")]
                let n = if self.semantics.is_some() { 4 } else { 3 };

                let mut seq = serializer.serialize_struct_variant("MgNode", 0, "Node", n)?;

                seq.serialize_field("features", features)?;
                seq.serialize_field("movement", &self.storage)?;

                #[cfg(feature = "semantics")]
                if let Some(semantics) = &self.semantics {
                    seq.serialize_field("semantics", &semantics)?;
                }

                seq.end()
            }
            MgNode::Leaf { lemma, features } => {
                #[cfg(not(feature = "semantics"))]
                let n = 4;

                #[cfg(feature = "semantics")]
                let n = if self.semantics.is_some() { 5 } else { 4 };

                let mut seq = serializer.serialize_struct_variant("MgNode", 1, "Leaf", n)?;

                seq.serialize_field("features", features)?;
                seq.serialize_field("lemma", lemma)?;

                #[cfg(feature = "semantics")]
                if let Some(semantics) = &self.semantics {
                    seq.serialize_field("semantics", semantics)?;
                }

                seq.end()
            }
            MgNode::Trace { trace } => {
                #[cfg(not(feature = "semantics"))]
                let n = 2;

                #[cfg(feature = "semantics")]
                let n = if self.semantics.is_some() { 3 } else { 2 };

                let mut seq = serializer.serialize_struct_variant("MgNode", 2, "Trace", n)?;

                seq.serialize_field("trace", trace)?;

                #[cfg(feature = "semantics")]
                if let Some(semantics) = &self.semantics {
                    seq.serialize_field("semantics", semantics)?;
                }

                seq.end()
            }
            MgNode::Start => "Start".serialize(serializer),
        }
    }
}

impl<T, C: Eq + Clone> Serialize for Tree<'_, T, C>
where
    C: Display,
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

impl Serialize for TraceId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(test)]
mod test {
    use crate::grammars::STABLER2011;
    use crate::{Lexicon, ParsingConfig, PhonContent};
    use petgraph::dot::Dot;

    #[test]
    fn petgraph() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;
        let (_, _, r) = lex
            .parse(
                &PhonContent::from(["which", "queen", "the", "king", "prefers"]),
                "C",
                &ParsingConfig::default(),
            )?
            .next()
            .unwrap();

        let d = lex.derivation(r);
        let (g, _root) = d.tree().petgraph(&d);

        let s = format!("{}", Dot::new(&g));
        println!("{s}");
        assert_eq!(
            s,
            "digraph {\n    0 [ label = \"C\" ]\n    1 [ label = \"D -W\" ]\n    2 [ label = \"+W C\" ]\n    3 [ label = \"which::N= D -W\" ]\n    4 [ label = \"queen::N\" ]\n    5 [ label = \"ε::V= +W C\" ]\n    6 [ label = \"V\" ]\n    7 [ label = \"D\" ]\n    8 [ label = \"=D V\" ]\n    9 [ label = \"the::N= D\" ]\n    10 [ label = \"king::N\" ]\n    11 [ label = \"prefers::D= =D V\" ]\n    12 [ label = \"t0\" ]\n    0 -> 1 [ label = \"Merge(Left)\" ]\n    0 -> 2 [ label = \"Merge(Right)\" ]\n    1 -> 3 [ label = \"Merge(Left)\" ]\n    1 -> 4 [ label = \"Merge(Right)\" ]\n    2 -> 5 [ label = \"Merge(Left)\" ]\n    2 -> 6 [ label = \"Merge(Right)\" ]\n    6 -> 7 [ label = \"Merge(Left)\" ]\n    6 -> 8 [ label = \"Merge(Right)\" ]\n    7 -> 9 [ label = \"Merge(Left)\" ]\n    7 -> 10 [ label = \"Merge(Right)\" ]\n    8 -> 11 [ label = \"Merge(Left)\" ]\n    8 -> 12 [ label = \"Merge(Right)\" ]\n    12 -> 1 [ label = \"Move\" ]\n}\n"
        );
        Ok(())
    }
}
