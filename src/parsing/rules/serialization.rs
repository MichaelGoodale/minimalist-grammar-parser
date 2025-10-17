use crate::parsing::rules::novel_printing::MgNode;
use serde::Serialize;
use std::fmt::{Debug, Display};

#[cfg(feature = "semantics")]
use super::semantics::{SemanticHistory, SemanticNode};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tree<'src, T, C: Eq + Display> {
    node: TreeNode<'src, T, C>,
    children: Vec<Tree<'src, T, C>>,
}

impl<'src, T, C: Eq + Display> Tree<'src, T, C> {
    pub(crate) fn new(node: MgNode<T, C>, children: Vec<Tree<'src, T, C>>) -> Self {
        Tree {
            node: TreeNode::new(node),
            children,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct TreeNode<'src, T, C: Eq + Display> {
    node: MgNode<T, C>,

    #[cfg(feature = "semantics")]
    semantics: Option<SemanticNode<'src>>,

    #[cfg(not(feature = "semantics"))]
    data: PhantomData<&'a ()>,
}

impl<'src, T, C: Eq + Display> TreeNode<'src, T, C> {
    fn new(node: MgNode<T, C>) -> Self {
        TreeNode {
            node,

            #[cfg(feature = "semantics")]
            semantics: None,

            #[cfg(not(feature = "semantics"))]
            data: PhantomData,
        }
    }
}
