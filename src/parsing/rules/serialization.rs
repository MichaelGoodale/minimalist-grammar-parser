use crate::{
    lexicon::Feature,
    parsing::rules::{
        TraceId,
        printing::{Lemma, MgNode, Storage},
    },
};
use itertools::Itertools;
use serde::{
    Serialize,
    ser::{SerializeSeq, SerializeStructVariant},
};
use std::fmt::{Debug, Display};

#[cfg(not(feature = "semantics"))]
use std::marker::PhantomData;

#[cfg(feature = "semantics")]
use regex::Regex;

#[cfg(feature = "semantics")]
use super::semantics::SemanticNode;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tree<'src, T, C: Eq + Display> {
    node: TreeNode<'src, T, C>,
    children: Vec<Tree<'src, T, C>>,
}

impl<'src, T, C: Eq + Display> Tree<'src, T, C> {
    pub(crate) fn new(
        node: MgNode<T, C>,
        storage: Storage<C>,
        children: Vec<Tree<'src, T, C>>,
    ) -> Self {
        Tree {
            node: TreeNode::new(node, storage),
            children,
        }
    }

    #[cfg(feature = "semantics")]
    pub(crate) fn new_with_semantics(
        node: MgNode<T, C>,
        semantic_node: Option<SemanticNode<'src>>,
        storage: Storage<C>,
        children: Vec<Tree<'src, T, C>>,
    ) -> Self {
        Tree {
            node: TreeNode::new_semantics(node, storage, semantic_node),
            children,
        }
    }

    pub fn storage(&self) -> &Storage<C> {
        &self.node.storage
    }
}
impl<'src, T: Display, C: Eq + Display> Tree<'src, T, C> {
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct TreeNode<'src, T, C: Eq + Display> {
    node: MgNode<T, C>,

    storage: Storage<C>,

    #[cfg(feature = "semantics")]
    semantics: Option<SemanticNode<'src>>,

    #[cfg(not(feature = "semantics"))]
    semantics: PhantomData<&'src ()>,
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
    fn new(node: MgNode<T, C>, storage: Storage<C>) -> TreeNode<'static, T, C> {
        TreeNode {
            node,
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
    ) -> Self {
        TreeNode {
            node,
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
