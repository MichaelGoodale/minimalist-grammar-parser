//! Module which defines the core functions to create or modify MG lexicons.

use ahash::{HashMap, HashSet};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{Direction, Pronounciation};
use chumsky::{extra::ParserExtra, label::LabelError, text::TextExpected, util::MaybeRef};
use chumsky::{
    prelude::*,
    text::{inline_whitespace, newline},
};
use itertools::Itertools;

use logprob::{LogProb, Softmax};

use logprob::LogSumExp;

use petgraph::Direction::Outgoing;
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

impl<C: Eq> Feature<C> {
    ///Get a reference to the inner category of a feature.
    pub fn inner(&self) -> &C {
        match self {
            Feature::Category(c)
            | Feature::Selector(c, _)
            | Feature::Affix(c, _)
            | Feature::Licensor(c)
            | Feature::Licensee(c) => c,
        }
    }

    ///Convert the [`Feature<C>`] to its inner category.
    pub fn into_inner(self) -> C {
        match self {
            Feature::Category(c)
            | Feature::Selector(c, _)
            | Feature::Affix(c, _)
            | Feature::Licensor(c)
            | Feature::Licensee(c) => c,
        }
    }

    ///Checks whether this features is a [`Feature::Selector`] or not.
    pub fn is_selector(&self) -> bool {
        matches!(self, Self::Selector(_, _))
    }
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

    ///Reference a lexeme that doesn't exist
    #[error("{0:?} is not a lexeme")]
    MissingLexeme(LexemeId),

    ///Lexicon is malformed since a node isn't a descendent from the root.
    #[error("Following the parents of node, ({0:?}), doesn't lead to root")]
    DoesntGoToRoot(NodeIndex),

    ///The parents of a node go beyond the root or go through a leaf.
    #[error("Following the parents of node, ({0:?}), we pass through the root or another lemma")]
    InvalidOrder(NodeIndex),
}

///Indicates a problem with parsing.
#[derive(Error, Clone, Copy, Debug)]
pub enum ParsingError<C> {
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
        .map(|x| Lexicon::new(x, true))
        .then_ignore(end())
}

fn entry_parser<'src, E>() -> impl Parser<'src, &'src str, LexicalEntry<&'src str, &'src str>, E>
where
    E: ParserExtra<'src, &'src str>,
    E::Error: LabelError<'src, &'src str, TextExpected<&'src str>>
        + LabelError<'src, &'src str, MaybeRef<'src, char>>
        + LabelError<'src, &'src str, &'static str>
        + LabelError<'src, &'src str, TextExpected<()>>,
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
        LexicalEntry::new(lemma.into(), features)
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum FeatureOrLemma<T: Eq, Category: Eq> {
    Root,
    Lemma(Pronounciation<T>),
    Feature(Feature<Category>),
    Complement(Category, Direction),
}

impl<T: Eq, C: Eq> FeatureOrLemma<T, C> {
    fn is_lemma(&self) -> bool {
        matches!(self, FeatureOrLemma::Lemma(_))
    }
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
    pub(crate) lemma: Pronounciation<T>,
    pub(crate) features: Vec<Feature<Category>>,
}
impl<T: Eq, Category: Eq> LexicalEntry<T, Category> {
    ///Creates a new lexical entry
    pub fn new(
        lemma: Pronounciation<T>,
        features: Vec<Feature<Category>>,
    ) -> LexicalEntry<T, Category> {
        LexicalEntry { lemma, features }
    }

    ///Get the lemma (possibly `None`) of a lexical entry
    pub fn lemma(&self) -> &Pronounciation<T> {
        &self.lemma
    }

    ///Gets all features of a lexical entry
    pub fn features(&self) -> &[Feature<Category>] {
        &self.features
    }

    ///Change the type of a [`LexicalEntry`]
    pub fn remap<'lex, T2: Eq, C2: Eq>(
        &'lex self,
        lemma_map: impl Fn(&'lex T) -> T2,
        category_map: impl Fn(&'lex Category) -> C2,
    ) -> LexicalEntry<T2, C2> {
        let lemma = self.lemma.as_ref().map(lemma_map);
        let features = self
            .features
            .iter()
            .map(|x| match x {
                Feature::Category(c) => Feature::Category(category_map(c)),
                Feature::Selector(c, direction) => Feature::Selector(category_map(c), *direction),
                Feature::Affix(c, direction) => Feature::Affix(category_map(c), *direction),
                Feature::Licensor(c) => Feature::Licensor(category_map(c)),
                Feature::Licensee(c) => Feature::Licensee(category_map(c)),
            })
            .collect();
        LexicalEntry { lemma, features }
    }

    ///Gets the category of a lexical entry
    pub fn category(&self) -> &Category {
        let mut cat = None;
        for lex in &self.features {
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
        write!(f, "{}::", &self.lemma)?;
        write!(
            f,
            "{}",
            self.features.iter().map(|x| x.to_string()).join(" ")
        )
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
    leaves: Vec<LexemeId>,
}

impl Serialize for Lexicon<&str, &str> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

impl<'de, 'a> Deserialize<'de> for Lexicon<&'a str, &'a str>
where
    'de: 'a,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = <&'de str>::deserialize(deserializer)?;
        Lexicon::from_string(s).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
///An ID for each lexeme in a grammar
pub struct LexemeId(pub(crate) NodeIndex);

#[cfg(feature = "pretty")]
impl Serialize for LexemeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTupleStruct;

        let mut s = serializer.serialize_tuple_struct("LexemeId", 1)?;
        s.serialize_field(&self.0.index())?;
        s.end()
    }
}

impl<T: Eq, Category: Eq> PartialEq for Lexicon<T, Category> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.leaves == other.leaves
            && self.graph.node_weights().eq(other.graph.node_weights())
            && self.graph.edge_weights().eq(other.graph.edge_weights())
    }
}
impl<T: Eq, Category: Eq> Eq for Lexicon<T, Category> {}
impl<T, Category> Lexicon<T, Category>
where
    T: Eq + Display,
    Category: Eq + Display,
{
    ///Prints a lexicon as a `GraphViz` dot file.
    #[must_use]
    pub fn graphviz(&self) -> String {
        let dot = Dot::new(&self.graph);
        format!("{dot}")
    }
}

fn renormalise_weights<T: Eq, C: Eq>(
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
                    .map(logprob::LogProb::into_inner);

                for (new_weight, (_weight, edge)) in dist.zip(edges.iter()) {
                    graph[*edge] = new_weight;
                }
            }
            _ => (),
        }
    }

    graph.map_owned(|_, n| n, |_, e| LogProb::new(e).unwrap())
}

///Iterator that climbs up a graph until it reaches the root.
pub struct Climber<'a, T: Eq, C: Eq> {
    lex: &'a Lexicon<T, C>,
    pos: NodeIndex,
}

impl<T: Eq, C: Eq + Clone> Iterator for Climber<'_, T, C> {
    type Item = Feature<C>;

    fn next(&mut self) -> Option<Self::Item> {
        self.pos = self.lex.parent_of(self.pos)?;

        match self.lex.graph.node_weight(self.pos) {
            Some(x) => match x {
                FeatureOrLemma::Root => None,
                FeatureOrLemma::Lemma(_) => None,
                FeatureOrLemma::Feature(feature) => Some(feature.clone()),
                FeatureOrLemma::Complement(c, direction) => {
                    Some(Feature::Selector(c.clone(), *direction))
                }
            },
            None => None,
        }
    }
}

pub(crate) fn fix_weights_per_node<T: Eq, C: Eq>(
    graph: &mut StableDiGraph<FeatureOrLemma<T, C>, LogProb<f64>>,
    node_index: NodeIndex,
) {
    let sum = graph
        .edges_directed(node_index, Outgoing)
        .map(|x| x.weight())
        .log_sum_exp_float_no_alloc();

    if sum != 0.0 {
        let edges: Vec<_> = graph
            .edges_directed(node_index, Outgoing)
            .map(|e| e.id())
            .collect();
        for e in edges {
            graph[e] = LogProb::new(graph[e].into_inner() - sum).unwrap();
        }
    }
}

pub(crate) fn fix_weights<T: Eq, C: Eq>(
    graph: &mut StableDiGraph<FeatureOrLemma<T, C>, LogProb<f64>>,
) {
    //Renormalise probabilities to sum to one.
    for node_index in graph.node_indices().collect_vec() {
        fix_weights_per_node(graph, node_index);
    }
}

///A struct which records whether a given licensee must always occur with a lexeme of a given
///category
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ObligatoryMover<C: Eq> {
    ///The licensee which always occurs with the category.
    pub licensee: C,
    ///The category that requires a specific licensee
    pub category: C,
}
impl<T: Eq + Debug, Category: Eq + Hash + Clone> Lexicon<T, Category> {
    ///This function goes through a lexicon and returns all licensees which are *always* correlated
    ///with a *specific* category.
    ///```
    /// # use minimalist_grammar_parser::Lexicon;
    /// # use minimalist_grammar_parser::lexicon::ObligatoryMover;
    /// # use anyhow;
    /// # fn main() -> anyhow::Result<()> {
    /// let lex = Lexicon::from_string("John::d -k -q\nwho::d -k -q -w\nsaw::d= +k v")?;
    /// assert_eq!(
    ///     lex.obligatory_movers(),
    ///     vec![
    ///         ObligatoryMover {
    ///             licensee: "k",
    ///             category: "d",
    ///         },
    ///         ObligatoryMover {
    ///             licensee: "q",
    ///             category: "d",
    ///         },
    ///     ]
    /// );
    /// let lex = Lexicon::from_string("John::d -k -q\nwho::d -k -w\nsaw::d= +k v")?;
    /// assert_eq!(
    ///     lex.obligatory_movers(),
    ///     vec![ObligatoryMover {
    ///         licensee: "k",
    ///         category: "d",
    ///     }]
    /// );
    /// let lex = Lexicon::from_string("John::d -q\nwho::d -k -q -w\nsaw::d= +k v")?;
    /// assert_eq!(lex.obligatory_movers(), vec![]);
    ///
    /// let lex = Lexicon::from_string("John::d -k -q\nwho::d -k -q -w\nsaw::d= +k v -q")?;
    /// assert_eq!(
    ///     lex.obligatory_movers(),
    ///     vec![ObligatoryMover {
    ///         licensee: "k",
    ///         category: "d",
    ///     }]
    /// );
    ///# Ok(())
    ///# }
    ///
    ///```
    #[must_use]
    pub fn obligatory_movers(&self) -> Vec<ObligatoryMover<Category>> {
        let mut stack = vec![(self.root, vec![])];
        let mut histories: HashMap<&Category, Vec<Vec<&Category>>> = HashMap::default();

        while let Some((x, hist)) = stack.pop() {
            for child in self.children_of(x) {
                match self.get(child).unwrap().0 {
                    FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                        let mut hist = hist.clone();
                        hist.push(c);
                        stack.push((child, hist));
                    }
                    FeatureOrLemma::Feature(Feature::Category(c)) => {
                        histories.entry(c).or_default().push(hist.clone());
                    }
                    _ => (),
                }
            }
        }

        let mut movers = vec![];
        let mut used_licensees = HashSet::default();
        for (category, mut licensees) in histories {
            while let Ok(Some(licensee)) = licensees
                .iter_mut()
                .map(std::vec::Vec::pop)
                .all_equal_value()
            {
                if used_licensees.contains(licensee) {
                    movers.retain(|x: &ObligatoryMover<Category>| &x.licensee != licensee);
                } else {
                    used_licensees.insert(licensee);
                    movers.push(ObligatoryMover {
                        licensee: licensee.clone(),
                        category: category.clone(),
                    });
                }
            }
        }

        movers
    }
}

impl<T: Eq, Category: Eq> Lexicon<T, Category> {
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

    ///Goes over all categories than can be parsed. Crucially, will exclude any category that may
    ///must necessarily be moved.
    pub fn root_categories(&self) -> impl Iterator<Item = &Category> {
        self.children_of(self.root).filter_map(|x| {
            if let FeatureOrLemma::Feature(Feature::Category(c)) =
                self.graph.node_weight(x).unwrap()
            {
                Some(c)
            } else {
                None
            }
        })
    }

    pub(crate) fn add_lexical_entry(
        &mut self,
        lexical_entry: LexicalEntry<T, Category>,
    ) -> Option<LexemeId> {
        let LexicalEntry { lemma, features } = lexical_entry;

        let mut new_nodes = vec![FeatureOrLemma::Lemma(lemma)];
        for (i, f) in features.into_iter().enumerate() {
            let f = if i == 0 && f.is_selector() {
                let Feature::Selector(c, d) = f else {
                    panic!("We checked it was a selector!")
                };
                FeatureOrLemma::Complement(c, d)
            } else {
                FeatureOrLemma::Feature(f)
            };
            new_nodes.push(f);
        }

        let mut pos = self.root;
        while let Some(f) = new_nodes.pop() {
            if let Some(p) = self.children_of(pos).find(|x| self.find(*x) == Some(&f)) {
                pos = p;
            } else {
                let new_pos = self.graph.add_node(f);
                self.graph.add_edge(pos, new_pos, LogProb::prob_of_one());
                fix_weights_per_node(&mut self.graph, pos);
                pos = new_pos;
            }
        }
        if self.leaves.contains(&LexemeId(pos)) {
            None
        } else {
            self.leaves.push(LexemeId(pos));
            Some(LexemeId(pos))
        }
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

    ///Finds a node's feature
    pub(crate) fn find(&self, nx: NodeIndex) -> Option<&FeatureOrLemma<T, Category>> {
        self.graph.node_weight(nx)
    }

    ///Climb up a node over all of its features
    pub(crate) fn node_to_features(&self, nx: NodeIndex) -> Climber<'_, T, Category> {
        Climber { lex: self, pos: nx }
    }

    ///Climb up a lexeme over all of its features
    #[must_use]
    pub fn leaf_to_features(&self, lexeme: LexemeId) -> Option<Climber<'_, T, Category>> {
        if !matches!(
            self.graph.node_weight(lexeme.0),
            Some(FeatureOrLemma::Lemma(_))
        ) {
            return None;
        }

        Some(Climber {
            lex: self,
            pos: lexeme.0,
        })
    }

    ///Gets the lemma of a lexeme.
    #[must_use]
    pub fn leaf_to_lemma(&self, lexeme_id: LexemeId) -> Option<&Pronounciation<T>> {
        match self.graph.node_weight(lexeme_id.0) {
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

    ///Get the category of a lexeme.
    #[must_use]
    pub fn category(&self, lexeme_id: LexemeId) -> Option<&Category> {
        let mut pos = lexeme_id.0;
        while !matches!(
            self.graph[pos],
            FeatureOrLemma::Feature(Feature::Category(_))
        ) {
            let new_pos = self.parent_of(pos)?;
            pos = new_pos;
        }
        let FeatureOrLemma::Feature(Feature::Category(cat)) = &self.graph[pos] else {
            return None;
        };
        Some(cat)
    }

    ///Get the parent of a node.
    pub(crate) fn parent_of(&self, nx: NodeIndex) -> Option<NodeIndex> {
        self.graph
            .edges_directed(nx, petgraph::Direction::Incoming)
            .next()
            .map(|e| e.source())
    }
}

impl<T: Eq, Category: Eq> Lexicon<T, Category> {
    ///Checks if `nx` is a complement
    #[must_use]
    pub fn is_complement(&self, nx: NodeIndex) -> bool {
        matches!(
            self.graph.node_weight(nx).unwrap(),
            FeatureOrLemma::Complement(_, _) | FeatureOrLemma::Feature(Feature::Affix(_, _))
        )
    }

    ///The number of nodes in a lexicon.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.graph.node_count()
    }

    ///Returns the leaves of a grammar
    #[must_use]
    pub fn leaves(&self) -> &[LexemeId] {
        &self.leaves
    }

    ///Get all lemmas of a grammar
    pub fn lemmas(&self) -> impl Iterator<Item = &Pronounciation<T>> {
        self.leaves.iter().filter_map(|x| match &self.graph[x.0] {
            FeatureOrLemma::Lemma(x) => Some(x),
            _ => None,
        })
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
}

impl<T: Eq + Clone, Category: Eq + Clone> Lexicon<T, Category> {
    ///Returns all leaves with their sibling nodes (e.g. exemes that are identical except for
    ///their lemma)
    #[must_use]
    pub fn sibling_leaves(&self) -> Vec<Vec<LexemeId>> {
        let mut leaves = self.leaves.clone();
        let mut result = vec![];
        while let Some(leaf) = leaves.pop() {
            let siblings: Vec<_> = self
                .children_of(self.parent_of(leaf.0).unwrap())
                .filter_map(|a| {
                    if matches!(self.graph.node_weight(a).unwrap(), FeatureOrLemma::Lemma(_)) {
                        Some(LexemeId(a))
                    } else {
                        None
                    }
                })
                .collect();
            leaves.retain(|x| !siblings.contains(x));
            result.push(siblings);
        }
        result
    }

    ///Gets each lexical entry along with its ID.
    pub fn lexemes_and_ids(
        &self,
    ) -> Result<impl Iterator<Item = (LexemeId, LexicalEntry<T, Category>)>, LexiconError> {
        Ok(self.leaves().iter().copied().zip(self.lexemes()?))
    }

    ///Turns a lexicon into a `Vec<LexicalEntry<T, Category>>` which can be useful for printing or
    ///investigating individual lexical entries.
    pub fn lexemes(&self) -> Result<Vec<LexicalEntry<T, Category>>, LexiconError> {
        let mut v = vec![];

        //NOTE: Must guarantee to iterate in this order.
        for leaf in &self.leaves {
            if let FeatureOrLemma::Lemma(lemma) = &self.graph[leaf.0] {
                let mut features = vec![];
                let mut nx = leaf.0;
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
                    lemma: lemma.clone(),
                    features,
                });
            } else {
                return Err(LexiconError::MissingLexeme(*leaf));
            }
        }
        Ok(v)
    }

    ///Given a leaf node, return its [`LexicalEntry`]
    pub fn get_lexical_entry(
        &self,
        lexeme_id: LexemeId,
    ) -> Result<LexicalEntry<T, Category>, LexiconError> {
        let Some(lemma) = self.graph.node_weight(lexeme_id.0) else {
            return Err(LexiconError::MissingLexeme(lexeme_id));
        };
        let lemma = match lemma {
            FeatureOrLemma::Lemma(lemma) => lemma,
            FeatureOrLemma::Feature(_)
            | FeatureOrLemma::Root
            | FeatureOrLemma::Complement(_, _) => {
                return Err(LexiconError::MissingLexeme(lexeme_id));
            }
        }
        .clone();

        let features = self.leaf_to_features(lexeme_id).unwrap().collect();
        Ok(LexicalEntry { lemma, features })
    }

    ///Create a new grammar from a [`Vec`] of [`LexicalEntry`]
    #[must_use]
    pub fn new(items: Vec<LexicalEntry<T, Category>>, collapse_lemmas: bool) -> Self {
        let n_items = items.len();
        Self::new_with_weights(
            items,
            std::iter::repeat_n(1.0, n_items).collect(),
            collapse_lemmas,
        )
    }

    ///Create a new grammar from a [`Vec`] of [`LexicalEntry`] where lexical entries are weighted
    ///in probability
    pub fn new_with_weights(
        items: Vec<LexicalEntry<T, Category>>,
        weights: Vec<f64>,
        collapse_lemmas: bool,
    ) -> Self {
        let mut graph = StableDiGraph::new();
        let root_index = graph.add_node(FeatureOrLemma::Root);
        let mut leaves = vec![];

        for (lexeme, weight) in items.into_iter().zip(weights.into_iter()) {
            let lexeme: Vec<FeatureOrLemma<T, Category>> = lexeme.into();
            let mut node_index = root_index;

            for feature in lexeme.into_iter().rev() {
                //We go down the feature list and add nodes and edges corresponding to features
                //If they exist already, we just follow the path until we have to start adding.
                node_index = if (!feature.is_lemma() || collapse_lemmas)
                    && let Some((nx, edge_index)) = graph
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
                    leaves.push(LexemeId(node_index));
                }
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

    ///Iterate over all categories of a grammar whether as selectors or categories. This goes over
    ///the entire lexicon so it will repeat elements multiple times.
    pub fn categories(&self) -> impl Iterator<Item = &Category> {
        self.graph.node_references().filter_map(|(_, x)| match x {
            FeatureOrLemma::Feature(
                Feature::Category(x) | Feature::Selector(x, _) | Feature::Affix(x, _),
            )
            | FeatureOrLemma::Complement(x, _) => Some(x),
            _ => None,
        })
    }

    ///Iterate over all licensors of a grammar
    pub fn licensor_types(&self) -> impl Iterator<Item = &Category> {
        self.graph.node_references().filter_map(|(_, x)| match x {
            FeatureOrLemma::Feature(Feature::Licensor(x) | Feature::Licensee(x)) => Some(x),
            _ => None,
        })
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
            .map_err(std::convert::Into::into)
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

        //Per StableGraph documentation, the node indices are still valid on the mapped data
        //structure.
        let graph = graph.map(
            |_, x| match x {
                FeatureOrLemma::Root => FeatureOrLemma::Root,
                FeatureOrLemma::Lemma(Pronounciation::Pronounced(s)) => {
                    FeatureOrLemma::Lemma(Pronounciation::Pronounced(lemma_map(s)))
                }
                FeatureOrLemma::Lemma(Pronounciation::Unpronounced) => {
                    FeatureOrLemma::Lemma(Pronounciation::Unpronounced)
                }
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
    #[must_use]
    pub fn to_owned_values(&self) -> Lexicon<String, String> {
        let Lexicon {
            graph,
            root,
            leaves,
        } = self;

        let graph = graph.map(
            |_, x| match *x {
                FeatureOrLemma::Root => FeatureOrLemma::Root,
                FeatureOrLemma::Lemma(s) => {
                    FeatureOrLemma::Lemma(s.map(std::borrow::ToOwned::to_owned))
                }
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

impl Display for LexiconParsingError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{}",
            self.0
                .iter()
                .map(std::string::ToString::to_string)
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
    pub fn parse(s: &str) -> Result<LexicalEntry<&str, &str>, LexiconParsingError<'_>> {
        entry_parser::<extra::Err<Rich<char>>>()
            .parse(s)
            .into_result()
            .map_err(std::convert::Into::into)
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
            FeatureOrLemma::Lemma(lemma) => write!(f, "{lemma}"),
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
                .map(std::string::ToString::to_string)
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

        let mut cats = lex.categories().collect::<Vec<_>>();
        cats.sort();
        assert_eq!(vec![&"C", &"V"], cats);
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
                lemma: Pronounciation::Pronounced("John"),
                features: vec![Feature::Category("d")]
            }
        );
        assert_eq!(
            SimpleLexicalEntry::parse("eats::d= =d V").unwrap(),
            SimpleLexicalEntry {
                lemma: Pronounciation::Pronounced("eats"),
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
                Pronounciation::Unpronounced,
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
                lemma: Pronounciation::Unpronounced,
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
                lemma: Pronounciation::Unpronounced,
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
                FeatureOrLemma::Lemma(Pronounciation::Pronounced("eats")),
                FeatureOrLemma::Complement("d", Direction::Right),
                FeatureOrLemma::Feature(Feature::Selector("d", Direction::Left)),
                FeatureOrLemma::Feature(Feature::Category("V")),
            ]
        );
    }

    use crate::grammars::{COPY_LANGUAGE, STABLER2011};
    use petgraph::dot::Dot;

    #[test]
    fn category() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;
        let leaves = lex
            .leaves()
            .iter()
            .map(|&x| (*lex.leaf_to_lemma(x).unwrap(), *lex.category(x).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            leaves,
            vec![
                (Pronounciation::Unpronounced, "C"),
                (Pronounciation::Unpronounced, "C"),
                (Pronounciation::Pronounced("knows"), "V"),
                (Pronounciation::Pronounced("says"), "V"),
                (Pronounciation::Pronounced("prefers"), "V"),
                (Pronounciation::Pronounced("drinks"), "V"),
                (Pronounciation::Pronounced("king"), "N"),
                (Pronounciation::Pronounced("wine"), "N"),
                (Pronounciation::Pronounced("beer"), "N"),
                (Pronounciation::Pronounced("queen"), "N"),
                (Pronounciation::Pronounced("the"), "D"),
                (Pronounciation::Pronounced("which"), "D")
            ]
        );
        Ok(())
    }

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
                vec![Pronounciation::Pronounced("which")],
                vec![Pronounciation::Pronounced("the")],
                vec![
                    Pronounciation::Pronounced("queen"),
                    Pronounciation::Pronounced("beer"),
                    Pronounciation::Pronounced("wine"),
                    Pronounciation::Pronounced("king")
                ],
                vec![
                    Pronounciation::Pronounced("drinks"),
                    Pronounciation::Pronounced("prefers")
                ],
                vec![
                    Pronounciation::Pronounced("says"),
                    Pronounciation::Pronounced("knows")
                ],
                vec![Pronounciation::Unpronounced],
                vec![Pronounciation::Unpronounced]
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
        let lex_2 = Lexicon::new(lex.lexemes().unwrap(), false);
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
                &Pronounciation::Unpronounced,
                &Pronounciation::Unpronounced,
                &Pronounciation::Pronounced("beer"),
                &Pronounciation::Pronounced("drinks"),
                &Pronounciation::Pronounced("king"),
                &Pronounciation::Pronounced("knows"),
                &Pronounciation::Pronounced("prefers"),
                &Pronounciation::Pronounced("queen"),
                &Pronounciation::Pronounced("says"),
                &Pronounciation::Pronounced("the"),
                &Pronounciation::Pronounced("which"),
                &Pronounciation::Pronounced("wine")
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
        let _ = lex.to_owned_values();
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
