//! Module which defines the core functions to create or modify MG lexicons.

use crate::Direction;
use crate::parsing::HeadTree;
use chumsky::{extra::ParserExtra, label::LabelError, text::TextExpected, util::MaybeRef};
use chumsky::{
    prelude::*,
    text::{inline_whitespace, newline},
};
use itertools::Itertools;
use logprob::{LogProb, Softmax};
use petgraph::dot::Dot;
use petgraph::prelude::StableDiGraph;
use petgraph::{
    graph::NodeIndex,
    visit::{EdgeRef, IntoNodeReferences},
};
use std::result::Result;
use std::{
    fmt::{Debug, Display},
    hash::Hash,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
///A possible features of a lexical entry
pub enum Feature<Category: Eq> {
    ///The category of a lexical entry.
    Category(Category),
    ///Poss;ble complements or specifiers (e.g. =d or d=)
    Selector(Category, Direction),
    ///Head movement
    Affix(Category, Direction),
    ///Possible places for movers to go to (e.g. +wh)
    Licensor(Category),
    ///Marks that a lexical entry can move (e.g. -wh)
    Licensee(Category),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
///Returns either a licensee or category: used only in [`ParsingError`]
#[allow(missing_docs)]
pub enum LicenseeOrCategory<C> {
    Licensee(C),
    Category(C),
}

impl<C: Debug> Display for LicenseeOrCategory<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseeOrCategory::Licensee(c) => write!(f, "-{c:?}"),
            LicenseeOrCategory::Category(c) => write!(f, "{c:?}"),
        }
    }
}

#[derive(Error, Clone, Copy, Debug)]
///Indicates that there is something wrong a lexicon
pub enum LexiconError {
    ///Reference a node that doesn't exist
    #[error("There is no node ({0:?})")]
    MissingNode(NodeIndex),

    ///Treats a node index as a leaf when it is not one.
    #[error("Node ({0:?}) is not a leaf")]
    NotALeaf(NodeIndex),

    ///Lexicon is malformed since a node isn't a descendent from the root.
    #[error("Following the parents of node, ({0:?}), doesn't lead to root")]
    DoesntGoToRoot(NodeIndex),

    ///The parents of a node go beyond the root or go through a leaf.
    #[error("Following the parents of node, ({0:?}), we pass through the root or another lemma")]
    InvalidOrder(NodeIndex),
}

///Indicates a problem with parsing.
#[derive(Error, Clone, Copy, Debug)]
pub enum ParsingError<C>
where
    C: Debug,
{
    ///We try to get a licensor or category of a feature that is not in the grammar.
    #[error("There is nothing with feature ({0})")]
    NoLicensorOrCategory(LicenseeOrCategory<C>),
}

impl<A: Debug> ParsingError<A> {
    ///Module to convert errors to their owned variants:
    ///
    ///```
    /// # use minimalist_grammar_parser::ParsingError;
    /// # use minimalist_grammar_parser::lexicon::LicenseeOrCategory;
    /// let x: ParsingError<&str> = ParsingError::NoLicensorOrCategory(LicenseeOrCategory::Category("s"));
    /// let y: ParsingError<String> = x.inner_into();
    ///
    ///```
    pub fn inner_into<B: From<A> + Debug>(self: ParsingError<A>) -> ParsingError<B> {
        match self {
            ParsingError::NoLicensorOrCategory(LicenseeOrCategory::Licensee(c)) => {
                ParsingError::NoLicensorOrCategory(LicenseeOrCategory::Licensee(c.into()))
            }
            ParsingError::NoLicensorOrCategory(LicenseeOrCategory::Category(c)) => {
                ParsingError::NoLicensorOrCategory(LicenseeOrCategory::Category(c.into()))
            }
        }
    }
}

fn grammar_parser<'src>()
-> impl Parser<'src, &'src str, Lexicon<&'src str, &'src str>, extra::Err<Rich<'src, char>>> {
    entry_parser::<extra::Err<Rich<'src, char>>>()
        .separated_by(newline())
        .allow_leading()
        .allow_trailing()
        .collect::<Vec<_>>()
        .map(Lexicon::new)
        .then_ignore(end())
}

fn entry_parser<'src, E>() -> impl Parser<'src, &'src str, LexicalEntry<&'src str, &'src str>, E>
where
    E: ParserExtra<'src, &'src str>,
    E::Error: LabelError<'src, &'src str, TextExpected<'src, &'src str>>
        + LabelError<'src, &'src str, MaybeRef<'src, char>>
        + LabelError<'src, &'src str, &'static str>,
{
    let feature_name = any()
        .and_is(none_of([
            '\t', '\n', ' ', '+', '-', '=', '>', '<', ':', '\r',
        ]))
        .repeated()
        .at_least(1)
        .labelled("feature name")
        .to_slice();

    let affix = choice((
        feature_name
            .then_ignore(just("<="))
            .map(|x| Feature::Affix(x, Direction::Right))
            .labelled("right affix"),
        just("=>")
            .ignore_then(feature_name)
            .map(|x| Feature::Affix(x, Direction::Left))
            .labelled("left affix"),
    ));

    let pre_category_features = choice((
        feature_name
            .then_ignore(just("="))
            .map(|x| Feature::Selector(x, Direction::Right))
            .labelled("right selector"),
        just("=")
            .ignore_then(feature_name)
            .map(|x| Feature::Selector(x, Direction::Left))
            .labelled("left selector"),
        just("+")
            .ignore_then(feature_name)
            .map(Feature::Licensor)
            .labelled("licensor"),
    ));

    choice((
        just("ε").to(None),
        (none_of(['\t', '\n', ' ', '+', '-', '=', ':', '\r'])
            .repeated()
            .at_least(1)
            .to_slice())
        .or_not(),
    ))
    .labelled("lemma")
    .then_ignore(
        just("::")
            .padded_by(inline_whitespace())
            .labelled("lemma feature seperator"),
    )
    .then(
        affix
            .or_not()
            .then_ignore(inline_whitespace().or_not())
            .then(
                pre_category_features
                    .separated_by(inline_whitespace())
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .labelled("pre category features"),
            )
            .map(|(affix, mut feats)| {
                if let Some(affix) = affix {
                    feats.insert(0, affix);
                }
                feats
            }),
    )
    .then(
        feature_name
            .map(Feature::Category)
            .labelled("category feature"),
    )
    .then(
        just('-')
            .ignore_then(feature_name)
            .map(Feature::Licensee)
            .labelled("licensee")
            .separated_by(inline_whitespace())
            .allow_leading()
            .allow_trailing()
            .collect::<Vec<_>>()
            .labelled("licensees"),
    )
    .map(|(((lemma, mut features), category), mut licensees)| {
        features.push(category);
        features.append(&mut licensees);
        LexicalEntry::new(lemma, features)
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum FeatureOrLemma<T: Eq, Category: Eq> {
    Root,
    Lemma(Option<T>),
    Feature(Feature<Category>),
    Complement(Category, Direction),
}

#[cfg(feature = "sampling")]
impl<T: Eq, C: Eq + Clone> FeatureOrLemma<T, C> {
    fn to_complement_mut(&mut self) {
        if let FeatureOrLemma::Feature(Feature::Selector(c, d)) = self {
            *self = FeatureOrLemma::Complement(c.clone(), *d);
        }
    }
}

impl<T: Eq, Category: Eq> From<LexicalEntry<T, Category>> for Vec<FeatureOrLemma<T, Category>> {
    fn from(value: LexicalEntry<T, Category>) -> Self {
        let LexicalEntry { lemma, features } = value;
        std::iter::once(lemma)
            .map(|x| FeatureOrLemma::Lemma(x))
            .chain(
                features
                    .into_iter()
                    .enumerate()
                    .map(|(i, feat)| match feat {
                        //A feature is a complement iff its the first
                        //selector (i.e. the moment the head first is merged)
                        Feature::Selector(c, d) if i == 0 => FeatureOrLemma::Complement(c, d),
                        _ => FeatureOrLemma::Feature(feat),
                    }),
            )
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
///A representation of a lexical entry in a grammar.
pub struct LexicalEntry<T: Eq, Category: Eq> {
    pub(crate) lemma: Option<T>,
    pub(crate) features: Vec<Feature<Category>>,
}

impl<T: Eq, Category: Eq> LexicalEntry<T, Category> {
    ///Creates a new lexical entry
    pub fn new(lemma: Option<T>, features: Vec<Feature<Category>>) -> LexicalEntry<T, Category> {
        LexicalEntry { lemma, features }
    }

    ///Get the lemma (possibly `None`) of a lexical entry
    pub fn lemma(&self) -> &Option<T> {
        &self.lemma
    }

    ///Gets all features of a lexical entry
    pub fn features(&self) -> &[Feature<Category>] {
        &self.features
    }

    ///Gets the category of a lexical entry
    pub fn category(&self) -> &Category {
        let mut cat = None;
        for lex in self.features.iter() {
            if let Feature::Category(c) = lex {
                cat = Some(c);
                break;
            }
        }
        cat.expect("All lexical entries must have one category")
    }
}

impl<T: Display + PartialEq + Eq, Category: Display + PartialEq + Eq> Display
    for LexicalEntry<T, Category>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(t) = &self.lemma {
            write!(f, "{t}::")?;
        } else {
            write!(f, "ε::")?;
        }
        for (i, feature) in self.features.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{feature}")?;
        }
        Ok(())
    }
}

impl<Category: Display + Eq> Display for Feature<Category> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Feature::Category(x) => write!(f, "{x}"),
            Feature::Selector(x, Direction::Left) => write!(f, "={x}"),
            Feature::Selector(x, Direction::Right) => write!(f, "{x}="),
            Feature::Affix(x, Direction::Left) => write!(f, "=>{x}"),
            Feature::Affix(x, Direction::Right) => write!(f, "{x}<="),
            Feature::Licensor(x) => write!(f, "+{x}"),
            Feature::Licensee(x) => write!(f, "-{x}"),
        }
    }
}

///The struct which represents a specific MG.
///They can be created by parsing:
///```
///# use minimalist_grammar_parser::Lexicon;
///let grammar: Lexicon<&str, &str> = Lexicon::from_string("John::d\nsaw::d= =d v\nMary::d")?;
///# Ok::<(), anyhow::Error>(())
///```
///It is also possible to make a [`Lexicon`] which use different types for categories or lemmas, such as integers or [`String`]s.
///These can be made with [`Lexicon::remap_lexicon`] or by passing a vector of [`LexicalEntry`]
///using [`Lexicon::new`].
///
///## Lemmas
///
///Lemmas are representated as `Option<T>`, so empty strings are `None` rather than an empty
///string to allow for arbitrary lemma type `T`.
///
///## Useful functions
///
/// - [`Lexicon::parse`] parse a string.
/// - [`Lexicon::parse_multiple`] parse multiple strings simultaneously.
/// - [`Lexicon::generate`] generate some of the strings of a grammar.
/// - [`Lexicon::valid_continuations`] find the valid next tokens of a grammar given a prefix
///
///
///## Representation
///The struct represents a lexicon as a graph, as per Stabler (2013).
///This means certain functions exploit this, such as [`Lexicon::n_nodes`] which doesn't reference
///the number of features of the grammar as written out, but rather the number of nodes in the graph representation.
///
/// - Stabler, E. (2013). Two Models of Minimalist, Incremental Syntactic Analysis. Topics in Cognitive Science, 5(3), 611–633. <https://doi.org/10.1111/tops.12031>
#[derive(Debug, Clone)]
pub struct Lexicon<T: Eq, Category: Eq> {
    graph: StableDiGraph<FeatureOrLemma<T, Category>, LogProb<f64>>,
    root: NodeIndex,

    leaves: Vec<NodeIndex>,
}

impl<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone> PartialEq
    for Lexicon<T, Category>
{
    //Super inefficient but we can leave it for now
    fn eq(&self, other: &Self) -> bool {
        let lexemes = self.lexemes().unwrap();
        for other_l in other.lexemes().unwrap() {
            if !lexemes.contains(&other_l) {
                return false;
            }
        }
        true
    }
}
impl<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone> Eq
    for Lexicon<T, Category>
{
}
impl<T, Category> Lexicon<T, Category>
where
    T: Eq + std::fmt::Debug + Clone + Display,
    Category: Eq + std::fmt::Debug + Clone + Display,
{
    ///Prints a lexicon as a GraphViz dot file.
    pub fn graphviz(&self) -> String {
        let dot = Dot::new(&self.graph);
        format!("{dot}")
    }
}

fn renormalise_weights<T: Eq + Clone, C: Eq + Clone>(
    mut graph: StableDiGraph<FeatureOrLemma<T, C>, f64>,
) -> StableDiGraph<FeatureOrLemma<T, C>, LogProb<f64>> {
    //Renormalise probabilities to sum to one.
    for node_index in graph.node_indices().collect_vec() {
        let edges: Vec<_> = graph
            .edges_directed(node_index, petgraph::Direction::Outgoing)
            .map(|e| (*e.weight(), e.id()))
            .collect();

        match edges.len().cmp(&1) {
            std::cmp::Ordering::Equal => {
                for (_, edge) in edges {
                    graph[edge] = 0.0;
                }
            }
            std::cmp::Ordering::Greater => {
                let dist = edges
                    .iter()
                    .map(|(w, _edge)| *w)
                    .softmax()
                    .unwrap()
                    .map(|x| x.into_inner());

                for (new_weight, (_weight, edge)) in dist.zip(edges.iter()) {
                    graph[*edge] = new_weight;
                }
            }
            _ => (),
        }
    }

    //TODO: Get rid of this annoying clone.
    graph.map(|_, n| n.clone(), |_, e| LogProb::new(*e).unwrap())
}

impl<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone> Lexicon<T, Category> {
    pub(crate) fn possible_heads(
        &self,
        nx: NodeIndex,
        depth: usize,
    ) -> Result<Vec<HeadTree>, ParsingError<Category>> {
        if depth > 10 {
            return Ok(vec![]);
        }
        let Some(FeatureOrLemma::Feature(Feature::Affix(category, direction))) =
            self.graph.node_weight(nx)
        else {
            panic!("Node must be an affix!")
        };

        //TODO: Consider whether the heads of moved phrases can themselves be moved??
        //TODO: Make faster by weaving some of this logic into beams
        let mut children = vec![];
        let x: NodeIndex = self.find_category(category)?;

        let mut stack = vec![x];

        while let Some(nx) = stack.pop() {
            match self.graph.node_weight(nx).unwrap() {
                FeatureOrLemma::Feature(Feature::Affix(..)) => {
                    children.extend(self.possible_heads(nx, depth + 1)?.into_iter())
                }
                FeatureOrLemma::Lemma(_) => children.push(HeadTree::just_heads(nx)),
                _ => stack.extend(self.children_of(nx)),
            }
        }

        let mut heads = vec![];
        for child in self.children_of(nx).filter(|child| {
            matches!(
                self.graph.node_weight(*child).unwrap(),
                FeatureOrLemma::Lemma(_)
            )
        }) {
            heads.extend(
                children
                    .clone()
                    .into_iter()
                    .map(|x| HeadTree::just_heads(child).merge(x, *direction)),
            )
        }
        Ok(heads)
    }
}

impl<T: Eq, Category: Eq> Lexicon<T, Category> {
    ///Gets the lemma of a leaf.
    pub fn leaf_to_lemma(&self, nx: NodeIndex) -> Option<&Option<T>> {
        match self.graph.node_weight(nx) {
            Some(x) => {
                if let FeatureOrLemma::Lemma(l) = x {
                    Some(l)
                } else {
                    None
                }
            }
            None => None,
        }
    }
}

impl<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone> Lexicon<T, Category> {
    ///Checks if `nx` is a complement
    pub fn is_complement(&self, nx: NodeIndex) -> bool {
        matches!(
            self.graph.node_weight(nx).unwrap(),
            FeatureOrLemma::Complement(_, _) | FeatureOrLemma::Feature(Feature::Affix(_, _))
        )
    }

    ///The number of nodes in a lexicon.
    pub fn n_nodes(&self) -> usize {
        self.graph.node_count()
    }

    ///Returns the leaves of a grammar
    pub fn leaves(&self) -> &[NodeIndex] {
        &self.leaves
    }

    ///Returns all leaves with their sibling nodes (e.g. lexemes that are identical except for
    ///their lemma)
    pub fn sibling_leaves(&self) -> Vec<Vec<NodeIndex>> {
        let mut leaves = self.leaves.clone();
        let mut result = vec![];
        while let Some(leaf) = leaves.pop() {
            let siblings: Vec<_> = self
                .children_of(self.parent_of(leaf).unwrap())
                .filter(|&a| matches!(self.graph.node_weight(a).unwrap(), FeatureOrLemma::Lemma(_)))
                .collect();
            leaves.retain(|x| !siblings.contains(x));
            result.push(siblings);
        }
        result
    }

    ///Turns a lexicon into a `Vec<LexicalEntry<T, Category>>` which can be useful for printing or
    ///investigating individual lexical entries.
    pub fn lexemes(&self) -> Result<Vec<LexicalEntry<T, Category>>, LexiconError> {
        let mut v = vec![];

        //NOTE: Must guarantee to iterate in this order.
        for leaf in self.leaves.iter() {
            if let FeatureOrLemma::Lemma(lemma) = &self.graph[*leaf] {
                let mut features = vec![];
                let mut nx = *leaf;
                while let Some(parent) = self.parent_of(nx) {
                    if parent == self.root {
                        break;
                    } else if let FeatureOrLemma::Feature(f) = &self.graph[parent] {
                        features.push(f.clone());
                    } else if let FeatureOrLemma::Complement(c, d) = &self.graph[parent] {
                        features.push(Feature::Selector(c.clone(), *d));
                    }
                    nx = parent;
                }

                v.push(LexicalEntry {
                    lemma: lemma.as_ref().cloned(),
                    features,
                })
            } else {
                return Err(LexiconError::NotALeaf(*leaf));
            }
        }
        Ok(v)
    }

    ///Given a leaf node, return its [`LexicalEntry`]
    pub fn get_lexical_entry(
        &self,
        nx: NodeIndex,
    ) -> Result<LexicalEntry<T, Category>, LexiconError> {
        let Some(lemma) = self.graph.node_weight(nx) else {
            return Err(LexiconError::MissingNode(nx));
        };
        let lemma = match lemma {
            FeatureOrLemma::Lemma(lemma) => lemma,
            FeatureOrLemma::Feature(_)
            | FeatureOrLemma::Root
            | FeatureOrLemma::Complement(_, _) => return Err(LexiconError::NotALeaf(nx)),
        }
        .clone();

        let mut parent = self
            .parent_of(nx)
            .expect("A lemma must always have a parent node");
        let mut parent_node = self
            .graph
            .node_weight(parent)
            .expect("A lemma must always have a parent node");
        let mut features = vec![];
        while !(matches!(parent_node, FeatureOrLemma::Root)) {
            match parent_node {
                FeatureOrLemma::Feature(feature) => {
                    features.push(feature.clone());
                }
                FeatureOrLemma::Complement(cat, d) => {
                    features.push(Feature::Selector(cat.clone(), *d))
                }
                FeatureOrLemma::Root | FeatureOrLemma::Lemma(_) => {
                    return Err(LexiconError::InvalidOrder(nx));
                }
            }

            match self.parent_of(parent) {
                Some(p) => {
                    parent = p;
                    parent_node = self
                        .graph
                        .node_weight(parent)
                        .expect("parent_of returning invalid node index!")
                }
                None => return Err(LexiconError::DoesntGoToRoot(nx)),
            }
        }
        Ok(LexicalEntry { features, lemma })
    }

    ///Get all lemmas of a grammar
    pub fn lemmas(&self) -> impl Iterator<Item = &Option<T>> {
        self.leaves.iter().filter_map(|x| match &self.graph[*x] {
            FeatureOrLemma::Lemma(x) => Some(x),
            _ => None,
        })
    }

    ///Create a new grammar from a [`Vec`] of [`LexicalEntry`]
    pub fn new(items: Vec<LexicalEntry<T, Category>>) -> Self {
        let n_items = items.len();
        Self::new_with_weights(items, std::iter::repeat_n(1.0, n_items).collect())
    }

    ///Create a new grammar from a [`Vec`] of [`LexicalEntry`] where lexical entries are weighted
    ///in probability
    pub fn new_with_weights(items: Vec<LexicalEntry<T, Category>>, weights: Vec<f64>) -> Self {
        let mut graph = StableDiGraph::new();
        let root_index = graph.add_node(FeatureOrLemma::Root);
        let mut leaves = vec![];

        for (lexeme, weight) in items.into_iter().zip(weights.into_iter()) {
            let lexeme: Vec<FeatureOrLemma<T, Category>> = lexeme.into();
            let mut node_index = root_index;

            for feature in lexeme.into_iter().rev() {
                //We go down the feature list and add nodes and edges corresponding to features
                //If they exist already, we just follow the path until we have to start adding.
                node_index = if let Some((nx, edge_index)) = graph
                    .edges_directed(node_index, petgraph::Direction::Outgoing)
                    .find(|x| graph[x.target()] == feature)
                    .map(|x| (x.target(), x.id()))
                {
                    graph[edge_index] += weight;
                    nx
                } else {
                    let new_node_index = graph.add_node(feature);
                    graph.add_edge(node_index, new_node_index, weight);
                    new_node_index
                };

                if let FeatureOrLemma::Lemma(_) = graph[node_index] {
                    leaves.push(node_index);
                };
            }
        }
        let graph = renormalise_weights(graph);
        Lexicon {
            graph,
            leaves,
            root: root_index,
        }
    }

    ///Get the node corresponding to a category
    pub(crate) fn find_category(
        &self,
        category: &Category,
    ) -> Result<NodeIndex, ParsingError<Category>> {
        match self
            .graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                FeatureOrLemma::Feature(Feature::Category(c)) => c == category,
                _ => false,
            }) {
            Some(x) => Ok(x),
            None => Err(ParsingError::NoLicensorOrCategory(
                LicenseeOrCategory::Category(category.clone()),
            )),
        }
    }

    ///Get the node corresponding to a licensee
    pub(crate) fn find_licensee(
        &self,
        category: &Category,
    ) -> Result<NodeIndex, ParsingError<Category>> {
        match self
            .graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                FeatureOrLemma::Feature(Feature::Licensee(c)) => c == category,
                _ => false,
            }) {
            Some(x) => Ok(x),
            None => Err(ParsingError::NoLicensorOrCategory(
                LicenseeOrCategory::Licensee(category.clone()),
            )),
        }
    }

    pub(crate) fn get(
        &self,
        nx: NodeIndex,
    ) -> Option<(&FeatureOrLemma<T, Category>, LogProb<f64>)> {
        if let Some(x) = self.graph.node_weight(nx) {
            let p = self
                .graph
                .edges_directed(nx, petgraph::Direction::Incoming)
                .next()
                .unwrap()
                .weight();
            Some((x, *p))
        } else {
            None
        }
    }

    ///Get the feature of a node, if it has one.
    pub(crate) fn get_feature_category(&self, nx: NodeIndex) -> Option<&Category> {
        self.graph.node_weight(nx).and_then(|x| match x {
            FeatureOrLemma::Root => None,
            FeatureOrLemma::Lemma(_) => None,
            FeatureOrLemma::Feature(feature) => match feature {
                Feature::Category(c)
                | Feature::Selector(c, _)
                | Feature::Licensor(c)
                | Feature::Licensee(c)
                | Feature::Affix(c, _) => Some(c),
            },
            FeatureOrLemma::Complement(c, _) => Some(c),
        })
    }

    ///Get the parent of a node.
    pub(crate) fn parent_of(&self, nx: NodeIndex) -> Option<NodeIndex> {
        self.graph
            .edges_directed(nx, petgraph::Direction::Incoming)
            .next()
            .map(|e| e.source())
    }

    ///Iterate over all categories of a grammar
    pub fn categories(&self) -> impl Iterator<Item = &Category> {
        self.graph.node_references().filter_map(|(_, x)| match x {
            FeatureOrLemma::Feature(Feature::Category(x)) => Some(x),
            _ => None,
        })
    }

    ///Iterate over all licensors of a grammar
    pub fn licensor_types(&self) -> impl Iterator<Item = &Category> {
        self.graph.node_references().filter_map(|(_, x)| match x {
            FeatureOrLemma::Feature(Feature::Licensor(x))
            | FeatureOrLemma::Feature(Feature::Licensee(x)) => Some(x),
            _ => None,
        })
    }

    ///The number of children of a node
    pub(crate) fn n_children(&self, nx: NodeIndex) -> usize {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .count()
    }

    ///The children of a node
    pub(crate) fn children_of(&self, nx: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .map(|e| e.target())
    }
}

impl<'src> Lexicon<&'src str, &'src str> {
    ///Parse a grammar and return a [`Lexicon<&str, &str>`]
    pub fn from_string(s: &'src str) -> Result<Self, LexiconParsingError<'src>> {
        grammar_parser()
            .padded()
            .then_ignore(end())
            .parse(s)
            .into_result()
            .map_err(|x| x.into())
    }
}

impl<T: Eq, C: Eq> Lexicon<T, C> {
    ///Converts a `Lexicon<T,C>` to a `Lexicon<T2, C2>` according to two functions which remap
    ///them.
    pub fn remap_lexicon<'lex, T2: Eq, C2: Eq>(
        &'lex self,
        lemma_map: impl Fn(&'lex T) -> T2,
        category_map: impl Fn(&'lex C) -> C2,
    ) -> Lexicon<T2, C2> {
        let Lexicon {
            graph,
            root,
            leaves,
        } = self;

        let graph = graph.map(
            |_, x| match x {
                FeatureOrLemma::Root => FeatureOrLemma::Root,
                FeatureOrLemma::Lemma(Some(s)) => FeatureOrLemma::Lemma(Some(lemma_map(s))),
                FeatureOrLemma::Lemma(None) => FeatureOrLemma::Lemma(None),
                FeatureOrLemma::Feature(Feature::Category(c)) => {
                    FeatureOrLemma::Feature(Feature::Category(category_map(c)))
                }
                FeatureOrLemma::Feature(Feature::Licensor(c)) => {
                    FeatureOrLemma::Feature(Feature::Licensor(category_map(c)))
                }
                FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                    FeatureOrLemma::Feature(Feature::Licensee(category_map(c)))
                }
                FeatureOrLemma::Feature(Feature::Affix(c, d)) => {
                    FeatureOrLemma::Feature(Feature::Affix(category_map(c), *d))
                }
                FeatureOrLemma::Feature(Feature::Selector(c, d)) => {
                    FeatureOrLemma::Feature(Feature::Selector(category_map(c), *d))
                }
                FeatureOrLemma::Complement(c, direction) => {
                    FeatureOrLemma::Complement(category_map(c), *direction)
                }
            },
            |_, e| *e,
        );
        Lexicon {
            graph,
            root: *root,
            leaves: leaves.clone(),
        }
    }
}

impl<'a> Lexicon<&'a str, &'a str> {
    ///Converts from `Lexicon<&str, &str>` to `Lexicon<String, String>`
    pub fn to_owned_values(&self) -> Lexicon<String, String> {
        let Lexicon {
            graph,
            root,
            leaves,
        } = self;

        let graph = graph.map(
            |_, x| match *x {
                FeatureOrLemma::Root => FeatureOrLemma::Root,
                FeatureOrLemma::Lemma(s) => FeatureOrLemma::Lemma(s.map(|x| x.to_owned())),
                FeatureOrLemma::Feature(Feature::Category(c)) => {
                    FeatureOrLemma::Feature(Feature::Category(c.to_owned()))
                }
                FeatureOrLemma::Feature(Feature::Affix(c, d)) => {
                    FeatureOrLemma::Feature(Feature::Affix(c.to_owned(), d))
                }
                FeatureOrLemma::Feature(Feature::Licensor(c)) => {
                    FeatureOrLemma::Feature(Feature::Licensor(c.to_owned()))
                }
                FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                    FeatureOrLemma::Feature(Feature::Licensee(c.to_owned()))
                }
                FeatureOrLemma::Feature(Feature::Selector(c, d)) => {
                    FeatureOrLemma::Feature(Feature::Selector(c.to_owned(), d))
                }
                FeatureOrLemma::Complement(c, direction) => {
                    FeatureOrLemma::Complement(c.to_owned(), direction)
                }
            },
            |_, e| *e,
        );
        Lexicon {
            graph,
            root: *root,
            leaves: leaves.clone(),
        }
    }
}

///Problem parsing a grammar; returns a vector of all [`Rich`] errors from Chumsky
#[derive(Error, Debug, Clone)]
pub struct LexiconParsingError<'a>(pub Vec<Rich<'a, char>>);

impl<'a> Display for LexiconParsingError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{}",
            self.0
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

impl<'a> From<Vec<Rich<'a, char>>> for LexiconParsingError<'a> {
    fn from(value: Vec<Rich<'a, char>>) -> Self {
        LexiconParsingError(value)
    }
}

impl LexicalEntry<&str, &str> {
    ///Parses a single lexical entry and returns it as a [`LexicalEntry`].
    pub fn parse(s: &str) -> Result<LexicalEntry<&str, &str>, LexiconParsingError> {
        entry_parser::<extra::Err<Rich<char>>>()
            .parse(s)
            .into_result()
            .map_err(|e| e.into())
    }
}

impl<T, C> std::fmt::Display for FeatureOrLemma<T, C>
where
    T: Eq + Display,
    C: Eq + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureOrLemma::Root => write!(f, "root"),
            FeatureOrLemma::Lemma(lemma) => {
                if let Some(l) = lemma {
                    write!(f, "{l}")
                } else {
                    write!(f, "ε")
                }
            }
            FeatureOrLemma::Feature(feature) => write!(f, "{feature}"),
            FeatureOrLemma::Complement(c, d) => write!(f, "{}", Feature::Selector(c, *d)),
        }
    }
}

impl<T, C> Display for Lexicon<T, C>
where
    T: Eq + Display + std::fmt::Debug + Clone,
    C: Eq + Display + std::fmt::Debug + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.lexemes()
                .unwrap()
                .iter()
                .map(|l| l.to_string())
                .join("\n")
        )
    }
}

#[cfg(feature = "semantics")]
mod semantics;

#[cfg(feature = "semantics")]
pub use semantics::SemanticLexicon;

pub mod mdl;

#[cfg(feature = "sampling")]
pub mod mutations;

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use anyhow::Result;

    use super::*;

    #[test]
    fn categories() -> Result<()> {
        let lex = Lexicon::from_string("::V= +Z C -W")?;

        assert_eq!(vec![&"C"], lex.categories().collect::<Vec<_>>());
        let mut lice = lex.licensor_types().collect::<Vec<_>>();
        lice.sort();
        assert_eq!(vec![&"W", &"Z"], lice);
        Ok(())
    }

    type SimpleLexicalEntry<'a> = LexicalEntry<&'a str, &'a str>;
    #[test]
    fn parsing() {
        assert_eq!(
            SimpleLexicalEntry::parse("John::d").unwrap(),
            SimpleLexicalEntry {
                lemma: Some("John"),
                features: vec![Feature::Category("d")]
            }
        );
        assert_eq!(
            SimpleLexicalEntry::parse("eats::d= =d V").unwrap(),
            SimpleLexicalEntry {
                lemma: Some("eats"),
                features: vec![
                    Feature::Selector("d", Direction::Right),
                    Feature::Selector("d", Direction::Left),
                    Feature::Category("V")
                ]
            }
        );
        assert_eq!(
            SimpleLexicalEntry::parse("::d= =d V").unwrap(),
            SimpleLexicalEntry::new(
                None,
                vec![
                    Feature::Selector("d", Direction::Right),
                    Feature::Selector("d", Direction::Left),
                    Feature::Category("V")
                ]
            )
        );
        assert_eq!(
            SimpleLexicalEntry::parse("ε::d= =d V").unwrap(),
            SimpleLexicalEntry {
                lemma: None,
                features: vec![
                    Feature::Selector("d", Direction::Right),
                    Feature::Selector("d", Direction::Left),
                    Feature::Category("V")
                ]
            }
        );
        assert_eq!(
            SimpleLexicalEntry::parse("ε::d<= V").unwrap(),
            SimpleLexicalEntry {
                lemma: None,
                features: vec![
                    Feature::Affix("d", Direction::Right),
                    Feature::Category("V")
                ]
            }
        );
    }

    #[test]
    fn get_lexical_entry() -> anyhow::Result<()> {
        let entries: HashSet<_> = STABLER2011
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<_, LexiconParsingError>>()?;
        let lex = Lexicon::from_string(STABLER2011)?;
        for nx in &lex.leaves {
            let lexical_entry = lex.get_lexical_entry(*nx)?;
            assert!(entries.contains(&lexical_entry));
        }
        Ok(())
    }
    #[test]
    fn get_category() {
        assert_eq!(
            *SimpleLexicalEntry::parse("eats::d= =d V")
                .unwrap()
                .category(),
            "V"
        );
        assert_eq!(
            *SimpleLexicalEntry::parse("eats::d= V -d")
                .unwrap()
                .category(),
            "V"
        );
        assert_eq!(
            *SimpleLexicalEntry::parse("eats::C -d").unwrap().category(),
            "C"
        );
        assert_eq!(
            *SimpleLexicalEntry::parse("eats::+w Z -d")
                .unwrap()
                .category(),
            "Z"
        );
    }

    #[test]
    fn convert_to_vec() {
        let x: Vec<FeatureOrLemma<_, _>> =
            SimpleLexicalEntry::parse("eats::d= =d V").unwrap().into();
        assert_eq!(
            x,
            vec![
                FeatureOrLemma::Lemma(Some("eats")),
                FeatureOrLemma::Complement("d", Direction::Right),
                FeatureOrLemma::Feature(Feature::Selector("d", Direction::Left)),
                FeatureOrLemma::Feature(Feature::Category("V")),
            ]
        );
    }

    use crate::grammars::{COPY_LANGUAGE, STABLER2011};
    use petgraph::dot::Dot;

    #[test]
    fn siblings() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;
        let siblings = lex.sibling_leaves();
        let siblings = siblings
            .into_iter()
            .map(|x| {
                x.into_iter()
                    .map(|x| *lex.leaf_to_lemma(x).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            siblings,
            vec![
                vec![Some("which")],
                vec![Some("the")],
                vec![Some("queen"), Some("beer"), Some("wine"), Some("king")],
                vec![Some("drinks"), Some("prefers")],
                vec![Some("says"), Some("knows")],
                vec![None],
                vec![None]
            ]
        );
        Ok(())
    }

    #[test]
    fn initialize_lexicon() -> anyhow::Result<()> {
        let strings: Vec<&str> = STABLER2011.split('\n').collect();

        let lex = Lexicon::from_string(STABLER2011)?;
        for lex in lex.lexemes()? {
            assert!(
                strings.contains(&format!("{}", lex).as_str())
                    || strings.contains(&format!("{}", lex).replace('ε', "").as_str())
            )
        }
        let lex_2 = Lexicon::new(lex.lexemes().unwrap());
        assert_eq!(lex, lex_2);
        assert_eq!(
            format!("{}", Dot::new(&lex.graph)),
            "digraph {
    0 [ label = \"root\" ]
    1 [ label = \"C\" ]
    2 [ label = \"V=\" ]
    3 [ label = \"ε\" ]
    4 [ label = \"+W\" ]
    5 [ label = \"V=\" ]
    6 [ label = \"ε\" ]
    7 [ label = \"V\" ]
    8 [ label = \"=D\" ]
    9 [ label = \"C=\" ]
    10 [ label = \"knows\" ]
    11 [ label = \"says\" ]
    12 [ label = \"D=\" ]
    13 [ label = \"prefers\" ]
    14 [ label = \"drinks\" ]
    15 [ label = \"N\" ]
    16 [ label = \"king\" ]
    17 [ label = \"wine\" ]
    18 [ label = \"beer\" ]
    19 [ label = \"queen\" ]
    20 [ label = \"D\" ]
    21 [ label = \"N=\" ]
    22 [ label = \"the\" ]
    23 [ label = \"-W\" ]
    24 [ label = \"D\" ]
    25 [ label = \"N=\" ]
    26 [ label = \"which\" ]
    0 -> 1 [ label = \"-2.8042006992676702\" ]
    1 -> 2 [ label = \"-0.6931471805599453\" ]
    2 -> 3 [ label = \"0\" ]
    1 -> 4 [ label = \"-0.6931471805599453\" ]
    4 -> 5 [ label = \"0\" ]
    5 -> 6 [ label = \"0\" ]
    0 -> 7 [ label = \"-0.8042006992676702\" ]
    7 -> 8 [ label = \"0\" ]
    8 -> 9 [ label = \"-0.6931471805599454\" ]
    9 -> 10 [ label = \"-0.6931471805599453\" ]
    9 -> 11 [ label = \"-0.6931471805599453\" ]
    8 -> 12 [ label = \"-0.6931471805599454\" ]
    12 -> 13 [ label = \"-0.6931471805599453\" ]
    12 -> 14 [ label = \"-0.6931471805599453\" ]
    0 -> 15 [ label = \"-0.8042006992676702\" ]
    15 -> 16 [ label = \"-1.3862943611198906\" ]
    15 -> 17 [ label = \"-1.3862943611198906\" ]
    15 -> 18 [ label = \"-1.3862943611198906\" ]
    15 -> 19 [ label = \"-1.3862943611198906\" ]
    0 -> 20 [ label = \"-3.8042006992676702\" ]
    20 -> 21 [ label = \"0\" ]
    21 -> 22 [ label = \"0\" ]
    0 -> 23 [ label = \"-3.8042006992676702\" ]
    23 -> 24 [ label = \"0\" ]
    24 -> 25 [ label = \"0\" ]
    25 -> 26 [ label = \"0\" ]
}
"
        );
        let n_categories = lex.categories().collect::<HashSet<_>>().len();
        assert_eq!(4, n_categories);
        let n_licensors = lex.licensor_types().collect::<HashSet<_>>().len();
        assert_eq!(1, n_licensors);
        let mut lemmas: Vec<_> = lex.lemmas().collect();
        lemmas.sort();
        assert_eq!(
            lemmas,
            vec![
                &None,
                &None,
                &Some("beer"),
                &Some("drinks"),
                &Some("king"),
                &Some("knows"),
                &Some("prefers"),
                &Some("queen"),
                &Some("says"),
                &Some("the"),
                &Some("which"),
                &Some("wine")
            ]
        );

        let lex = Lexicon::from_string(COPY_LANGUAGE)?;
        assert_ne!(lex, lex_2);
        assert_eq!(
            "digraph {
    0 [ label = \"root\" ]
    1 [ label = \"T\" ]
    2 [ label = \"+l\" ]
    3 [ label = \"+r\" ]
    4 [ label = \"=T\" ]
    5 [ label = \"ε\" ]
    6 [ label = \"-l\" ]
    7 [ label = \"-r\" ]
    8 [ label = \"T\" ]
    9 [ label = \"ε\" ]
    10 [ label = \"T\" ]
    11 [ label = \"+l\" ]
    12 [ label = \"=A\" ]
    13 [ label = \"a\" ]
    14 [ label = \"-r\" ]
    15 [ label = \"A\" ]
    16 [ label = \"+r\" ]
    17 [ label = \"=T\" ]
    18 [ label = \"a\" ]
    19 [ label = \"=B\" ]
    20 [ label = \"b\" ]
    21 [ label = \"B\" ]
    22 [ label = \"+r\" ]
    23 [ label = \"=T\" ]
    24 [ label = \"b\" ]
    0 -> 1 [ label = \"-2.4076059644443806\" ]
    1 -> 2 [ label = \"0\" ]
    2 -> 3 [ label = \"0\" ]
    3 -> 4 [ label = \"0\" ]
    4 -> 5 [ label = \"0\" ]
    0 -> 6 [ label = \"-0.4076059644443806\" ]
    6 -> 7 [ label = \"-1.3132616875182228\" ]
    7 -> 8 [ label = \"0\" ]
    8 -> 9 [ label = \"0\" ]
    6 -> 10 [ label = \"-0.3132616875182228\" ]
    10 -> 11 [ label = \"0\" ]
    11 -> 12 [ label = \"-0.6931471805599453\" ]
    12 -> 13 [ label = \"0\" ]
    0 -> 14 [ label = \"-1.4076059644443804\" ]
    14 -> 15 [ label = \"-0.6931471805599453\" ]
    15 -> 16 [ label = \"0\" ]
    16 -> 17 [ label = \"0\" ]
    17 -> 18 [ label = \"0\" ]
    11 -> 19 [ label = \"-0.6931471805599453\" ]
    19 -> 20 [ label = \"0\" ]
    14 -> 21 [ label = \"-0.6931471805599453\" ]
    21 -> 22 [ label = \"0\" ]
    22 -> 23 [ label = \"0\" ]
    23 -> 24 [ label = \"0\" ]
}
",
            format!("{}", Dot::new(&lex.graph))
        );
        let n_categories = lex.categories().collect::<HashSet<_>>().len();
        assert_eq!(3, n_categories);
        let n_licensors = lex.licensor_types().collect::<HashSet<_>>().len();
        assert_eq!(2, n_licensors);
        Ok(())
    }

    #[test]
    fn conversion() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;
        lex.to_owned_values();
        Ok(())
    }
    #[test]
    fn conversion2() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;
        let lex2 = lex.remap_lexicon(|x| x.to_string(), |c| c.to_string());
        let lex3 = lex2.remap_lexicon(|x| x.as_str(), |c| c.as_str());
        assert_eq!(lex, lex3);

        let x = ParsingError::NoLicensorOrCategory(LicenseeOrCategory::Category("s"));
        let _y: ParsingError<String> = x.inner_into();

        Ok(())
    }
}
