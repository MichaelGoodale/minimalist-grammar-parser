use crate::Direction;
use anyhow::{Context, Result, bail};
use chumsky::{extra::ParserExtra, label::LabelError, text::TextExpected, util::MaybeRef};
use chumsky::{
    prelude::*,
    text::{inline_whitespace, newline},
};
use logprob::{LogProb, Softmax};
use petgraph::{
    graph::DiGraph,
    graph::NodeIndex,
    visit::{EdgeRef, IntoNodeReferences},
};
use std::{fmt::Display, hash::Hash};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature<Category: Eq> {
    Category(Category),
    Selector(Category, Direction),
    Licensor(Category),
    Licensee(Category),
}
fn grammar_parser<'src>()
-> impl Parser<'src, &'src str, Lexicon<&'src str, &'src str>, extra::Err<Rich<'src, char>>> {
    entry_parser::<extra::Err<Rich<'src, char>>>()
        .separated_by(newline())
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
        .and_is(none_of(['\t', '\n', ' ', '+', '-', '=', ':', '\r']))
        .repeated()
        .at_least(1)
        .labelled("feature name")
        .to_slice();

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

    choice((just("ε").to(None), text::ident().or_not()))
        .labelled("lemma")
        .then_ignore(just("::").padded().labelled("lemma feature seperator"))
        .then(
            pre_category_features
                .separated_by(inline_whitespace())
                .allow_trailing()
                .collect::<Vec<_>>()
                .labelled("pre category features"),
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
pub enum FeatureOrLemma<T: Eq, Category: Eq> {
    Root,
    Lemma(Option<T>),
    Feature(Feature<Category>),
    Complement(Category, Direction),
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
pub struct LexicalEntry<T: Eq, Category: Eq> {
    pub(crate) lemma: Option<T>,
    pub(crate) features: Vec<Feature<Category>>,
}

impl<T: Eq, Category: Eq> LexicalEntry<T, Category> {
    pub fn new(lemma: Option<T>, features: Vec<Feature<Category>>) -> LexicalEntry<T, Category> {
        LexicalEntry { lemma, features }
    }

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
            write!(f, "{}::", t)?;
        } else {
            write!(f, "ε::")?;
        }
        for (i, feature) in self.features.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", feature)?;
        }
        Ok(())
    }
}

impl<Category: Display + Eq> Display for Feature<Category> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Feature::Category(x) => write!(f, "{}", x),
            Feature::Selector(x, Direction::Left) => write!(f, "={}", x),
            Feature::Selector(x, Direction::Right) => write!(f, "{}=", x),
            Feature::Licensor(x) => write!(f, "+{}", x),
            Feature::Licensee(x) => write!(f, "-{}", x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lexicon<T: Eq, Category: Eq> {
    graph: DiGraph<FeatureOrLemma<T, Category>, LogProb<f64>>,
    root: NodeIndex,
    leaves: Vec<(NodeIndex, f64)>,
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

impl<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone> Lexicon<T, Category> {
    pub fn is_complement(&self, nx: NodeIndex) -> bool {
        matches!(
            self.graph.node_weight(nx).unwrap(),
            FeatureOrLemma::Complement(_, _)
        )
    }

    pub fn lexemes(&self) -> Result<Vec<(LexicalEntry<T, Category>, f64)>> {
        let mut v = vec![];
        for (leaf, weight) in self.leaves.iter() {
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

                v.push((
                    LexicalEntry {
                        lemma: lemma.as_ref().cloned(),
                        features,
                    },
                    *weight,
                ))
            } else {
                bail!("Bad lexicon!");
            }
        }
        Ok(v)
    }

    pub fn get_lexical_entry(&self, nx: NodeIndex) -> anyhow::Result<LexicalEntry<T, Category>> {
        let lemma = self.graph.node_weight(nx).context("No such node")?;
        let lemma = match lemma {
            FeatureOrLemma::Lemma(lemma) => lemma,
            FeatureOrLemma::Feature(_)
            | FeatureOrLemma::Root
            | FeatureOrLemma::Complement(_, _) => {
                bail!("Node is not a lemma node!")
            }
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
                FeatureOrLemma::Root | FeatureOrLemma::Lemma(_) => bail!(
                    "We should never have a lemma or root accessed this way, the lexicon is mal-formed"
                ),
            }
            parent = self
                .parent_of(parent)
                .expect("All features must end before the root!");
            parent_node = self
                .graph
                .node_weight(parent)
                .expect("All features must end before the root!");
        }
        Ok(LexicalEntry { features, lemma })
    }

    pub fn lemmas(&self) -> impl Iterator<Item = &Option<T>> {
        self.leaves
            .iter()
            .filter_map(|(x, _w)| match &self.graph[*x] {
                FeatureOrLemma::Lemma(x) => Some(x),
                _ => None,
            })
    }

    pub fn new(items: Vec<LexicalEntry<T, Category>>) -> Self {
        let n_items = items.len();
        Self::new_with_weights(items, std::iter::repeat_n(1.0, n_items).collect())
    }

    pub fn new_with_weights(items: Vec<LexicalEntry<T, Category>>, weights: Vec<f64>) -> Self {
        let mut graph = DiGraph::new();
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
                    leaves.push((node_index, weight));
                };
            }
        }
        //Renormalise probabilities to sum to one.
        for node_index in graph.node_indices() {
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
        let graph = graph.map(|_, n| n.clone(), |_, e| LogProb::new(*e).unwrap());
        Lexicon {
            graph,
            leaves,
            root: root_index,
        }
    }

    pub fn find_category(&self, category: &Category) -> Result<NodeIndex> {
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                FeatureOrLemma::Feature(Feature::Category(c)) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }

    pub fn find_licensee(&self, category: &Category) -> Result<NodeIndex> {
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                FeatureOrLemma::Feature(Feature::Licensee(c)) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }

    pub fn get(&self, nx: NodeIndex) -> Option<(&FeatureOrLemma<T, Category>, LogProb<f64>)> {
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

    pub fn get_feature_category(&self, nx: NodeIndex) -> Option<&Category> {
        self.graph.node_weight(nx).and_then(|x| match x {
            FeatureOrLemma::Root => None,
            FeatureOrLemma::Lemma(_) => None,
            FeatureOrLemma::Feature(feature) => match feature {
                Feature::Category(c)
                | Feature::Selector(c, _)
                | Feature::Licensor(c)
                | Feature::Licensee(c) => Some(c),
            },
            FeatureOrLemma::Complement(c, _) => Some(c),
        })
    }

    pub fn parent_of(&self, nx: NodeIndex) -> Option<NodeIndex> {
        self.graph
            .edges_directed(nx, petgraph::Direction::Incoming)
            .next()
            .map(|e| e.source())
    }

    pub fn get_category(&self, mut nx: NodeIndex) -> Option<&FeatureOrLemma<T, Category>> {
        while let Some(e) = self
            .graph
            .edges_directed(nx, petgraph::Direction::Incoming)
            .next()
        {
            nx = e.source();
            let cat = &self.graph[nx];
            if let FeatureOrLemma::Feature(Feature::Category(_)) = cat {
                return Some(cat);
            }
        }

        None
    }

    pub fn categories(&self) -> impl Iterator<Item = &Category> {
        self.graph.node_references().filter_map(|(_, x)| match x {
            FeatureOrLemma::Feature(Feature::Category(x)) => Some(x),
            _ => None,
        })
    }

    pub fn licensor_types(&self) -> impl Iterator<Item = &Category> {
        self.graph.node_references().filter_map(|(_, x)| match x {
            FeatureOrLemma::Feature(Feature::Licensor(x))
            | FeatureOrLemma::Feature(Feature::Licensee(x)) => Some(x),
            _ => None,
        })
    }

    pub fn n_children(&self, nx: NodeIndex) -> usize {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .count()
    }

    pub fn children_of(&self, nx: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .map(|e| e.target())
    }
}

impl<T, Category> std::fmt::Display for Lexicon<T, Category>
where
    T: Eq,
    Category: Eq,
    FeatureOrLemma<T, Category>: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", petgraph::dot::Dot::new(&self.graph))
    }
}
impl<'src> Lexicon<&'src str, &'src str> {
    pub fn parse(s: &'src str) -> Result<Self> {
        grammar_parser()
            .then_ignore(end())
            .parse(s)
            .into_result()
            .map_err(|x| {
                anyhow::Error::msg(
                    x.into_iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })
    }
}

pub type SimpleLexicalEntry<'a> = LexicalEntry<&'a str, &'a str>;

impl LexicalEntry<&str, &str> {
    pub fn parse(s: &str) -> Result<LexicalEntry<&str, &str>> {
        entry_parser::<extra::Err<Rich<char>>>()
            .parse(s)
            .into_result()
            .map_err(|x| {
                anyhow::Error::msg(
                    x.into_iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })
    }
}

impl std::fmt::Display for FeatureOrLemma<&str, &str> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureOrLemma::Root => write!(f, "root"),
            FeatureOrLemma::Lemma(lemma) => {
                if let Some(l) = lemma {
                    write!(f, "{}", l)
                } else {
                    write!(f, "ε")
                }
            }
            FeatureOrLemma::Feature(feature) => write!(f, "{}", feature),
            FeatureOrLemma::Complement(c, d) => write!(f, "{}", Feature::Selector(c, *d)),
        }
    }
}

#[cfg(feature = "semantics")]
mod semantics;

#[cfg(feature = "semantics")]
pub use semantics::SemanticLexicon;

mod mdl;

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use anyhow::Result;

    use super::*;

    #[test]
    fn categories() -> Result<()> {
        let lex = Lexicon::parse("::V= +Z C -W")?;

        assert_eq!(vec![&"C"], lex.categories().collect::<Vec<_>>());
        let mut lice = lex.licensor_types().collect::<Vec<_>>();
        lice.sort();
        assert_eq!(vec![&"W", &"Z"], lice);
        Ok(())
    }

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
    }

    #[test]
    fn get_lexical_entry() -> anyhow::Result<()> {
        let entries: HashSet<_> = STABLER2011
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<_>>()?;
        let lex = Lexicon::parse(STABLER2011)?;
        for (nx, _) in &lex.leaves {
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
    fn initialize_lexicon() -> anyhow::Result<()> {
        let strings: Vec<&str> = STABLER2011.split('\n').collect();

        let lex = Lexicon::parse(STABLER2011)?;
        for (lex, weight) in lex.lexemes()? {
            assert_eq!(weight, 1.0);
            assert!(
                strings.contains(&format!("{}", lex).as_str())
                    || strings.contains(&format!("{}", lex).replace('ε', "").as_str())
            )
        }
        let lex_2 = Lexicon::new(
            lex.lexemes()
                .unwrap()
                .into_iter()
                .map(|(word, _weight)| word)
                .collect::<Vec<_>>(),
        );
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

        let lex = Lexicon::parse(COPY_LANGUAGE)?;
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
}
