use crate::Direction;
use ahash::AHashSet;
use anyhow::{bail, Context, Result};
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

impl Feature<char> {
    fn parse(s: &str) -> Result<Feature<char>> {
        if s.chars().nth(2).is_some() {
            bail!("This feature has too many characters!")
        } else if s.starts_with('+') {
            Ok(Feature::Licensor(
                s.chars().nth(1).context("No character!")?,
            ))
        } else if s.starts_with('-') {
            Ok(Feature::Licensee(
                s.chars().nth(1).context("No character!")?,
            ))
        } else if s.starts_with('=') {
            Ok(Feature::Selector(
                s.chars().nth(1).context("No character!")?,
                Direction::Left,
            ))
        } else if s.ends_with('=') {
            Ok(Feature::Selector(
                s.chars().next().context("No character!")?,
                Direction::Right,
            ))
        } else {
            if s.chars().nth(1).is_some() {
                bail!("This feature has too many characters!");
            }
            Ok(Feature::Category(
                s.chars().next().context("No character!")?,
            ))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeatureOrLemma<T: Eq, Category: Eq> {
    Root,
    Lemma(Option<T>),
    Feature(Feature<Category>),
}

impl<T: Eq, Category: Eq> From<LexicalEntry<T, Category>> for Vec<FeatureOrLemma<T, Category>> {
    fn from(value: LexicalEntry<T, Category>) -> Self {
        let LexicalEntry { lemma, features } = value;
        std::iter::once(lemma)
            .map(|x| FeatureOrLemma::Lemma(x))
            .chain(features.into_iter().map(|x| FeatureOrLemma::Feature(x)))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LexicalEntry<T: Eq, Category: Eq> {
    pub lemma: Option<T>,
    pub features: Vec<Feature<Category>>,
}

impl<T: Eq, Category: Eq> LexicalEntry<T, Category> {
    pub fn new(lemma: Option<T>, features: Vec<Feature<Category>>) -> LexicalEntry<T, Category> {
        LexicalEntry { lemma, features }
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
    graph: DiGraph<FeatureOrLemma<T, Category>, f64>,
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

pub trait SymbolCost {
    ///This is the length of a member where each sub-unit has [``n_phonemes``] possible encodings.
    /// # Example
    /// Let Phon = $\{a,b,c\}$ so, `n_phonemes` should be 3 (passed to [``mdl_score``].
    /// The string, abcabc should have symbol_cost 6.
    fn symbol_cost(&self) -> Result<u16>;
}

impl SymbolCost for &str {
    fn symbol_cost(&self) -> Result<u16> {
        Ok(self.len().try_into()?)
    }
}

impl SymbolCost for char {
    fn symbol_cost(&self) -> Result<u16> {
        Ok(1)
    }
}

///Number of types of features, e.g. the space of possible features in [``Feature``] enum.
///Here it is five to account for left and right attachment.
const MG_TYPES: u16 = 5;

impl<
        T: Eq + std::fmt::Debug + Clone + SymbolCost,
        Category: Eq + std::fmt::Debug + Clone + Hash,
    > Lexicon<T, Category>
{
    ///Returns the MDL Score of the lexicon as per Ermolaeva 2021
    ///
    /// # Arguments
    /// * `n_phonemes` - The size of required to encode a symbol of the phonology. E.g.
    /// in English orthography, it would be 26.
    pub fn mdl_score(&self, n_phonemes: u16) -> Result<f64> {
        let mut category_symbols = AHashSet::new();
        let mut lemma_costs: u16 = 0;

        for (leaf, _weight) in self.leaves.iter() {
            println!("{:?}", self.graph[*leaf]);
            if let FeatureOrLemma::Lemma(lemma) = &self.graph[*leaf] {
                let mut nx = *leaf;
                let lemma_cost: u16 = if let Some(lemma) = lemma {
                    lemma.symbol_cost()?
                } else {
                    0
                };

                let mut n_features = 0;
                while let Some(parent) = self.parent_of(nx) {
                    if parent == self.root {
                        break;
                    } else if let FeatureOrLemma::Feature(f) = &self.graph[parent] {
                        println!("{:?}", f);
                        category_symbols.insert(match f {
                            Feature::Category(c) | Feature::Licensor(c) | Feature::Licensee(c) => c,
                            Feature::Selector(c, _d) => c,
                        });
                        n_features += 1;
                    }
                    nx = parent;
                }

                lemma_costs += lemma_cost + 2 * n_features + 1;
            } else {
                bail!("Bad lexicon!");
            }
        }
        let n_categories: u16 = category_symbols.len().try_into()?;
        let symbol_cost: f64 = (MG_TYPES + n_categories + n_phonemes + 1).try_into()?;
        let symbol_cost = symbol_cost.log2();

        let lemma_costs: f64 = lemma_costs.try_into()?;

        Ok(lemma_costs * symbol_cost)
    }
}

impl<T: Eq + std::fmt::Debug + Clone, Category: Eq + std::fmt::Debug + Clone> Lexicon<T, Category> {
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
        Self::new_with_weights(items, std::iter::repeat(1.0).take(n_items).collect())
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
                    let n: f64 = edges.iter().map(|(w, _)| w).sum::<f64>().ln();
                    for (weight, edge) in edges {
                        graph[edge] = weight.ln() - n;
                    }
                }
                _ => (),
            }
        }
        Lexicon {
            graph,
            leaves,
            root: root_index,
        }
    }

    pub fn find_category(&self, category: Category) -> Result<NodeIndex> {
        let category = FeatureOrLemma::Feature(Feature::Category(category));
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| self.graph[*i] == category)
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }

    pub fn find_licensee(&self, category: Category) -> Result<NodeIndex> {
        let category = &FeatureOrLemma::Feature(Feature::Licensee(category));
        self.graph
            .edges_directed(self.root, petgraph::Direction::Outgoing)
            .map(|e| (e.target(), &self.graph[e.target()]))
            .find_map(move |(i, x)| if x == category { Some(i) } else { None })
            .with_context(|| format!("{category:?} is not a valid licensee in the lexicon!"))
    }

    pub fn get(&self, nx: NodeIndex) -> Option<(&FeatureOrLemma<T, Category>, f64)> {
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

pub type SimpleLexicalEntry<'a> = LexicalEntry<&'a str, char>;

impl LexicalEntry<&str, char> {
    pub fn parse(s: &str) -> Result<LexicalEntry<&str, char>> {
        if let Some((lemma, features)) = s.split_once("::") {
            Ok(LexicalEntry {
                lemma: if lemma == "ε" || lemma.is_empty() {
                    None
                } else {
                    Some(lemma)
                },
                features: features
                    .split(' ')
                    .map(Feature::<char>::parse)
                    .collect::<Result<Vec<_>>>()?,
            })
        } else {
            bail!("No :: divider!");
        }
    }
}

impl std::fmt::Display for FeatureOrLemma<&str, char> {
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
        }
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use anyhow::Result;
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn categories() -> Result<()> {
        let g: &str = "::V= +Z C -W";

        let strings: Vec<&str> = g.split('\n').collect();
        let v: Vec<_> = strings
            .iter()
            .copied()
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()?;
        let lex = Lexicon::new(v);

        assert_eq!(vec![&'C'], lex.categories().collect::<Vec<_>>());
        let mut lice = lex.licensor_types().collect::<Vec<_>>();
        lice.sort();
        assert_eq!(vec![&'W', &'Z'], lice);
        Ok(())
    }

    #[test]
    fn parsing() {
        assert_eq!(
            SimpleLexicalEntry::parse("John::d").unwrap(),
            SimpleLexicalEntry {
                lemma: Some("John"),
                features: vec![Feature::Category('d')]
            }
        );
        assert_eq!(
            SimpleLexicalEntry::parse("eats::d= =d V").unwrap(),
            SimpleLexicalEntry {
                lemma: Some("eats"),
                features: vec![
                    Feature::Selector('d', Direction::Right),
                    Feature::Selector('d', Direction::Left),
                    Feature::Category('V')
                ]
            }
        );
        assert_eq!(
            SimpleLexicalEntry::parse("::d= =d V").unwrap(),
            SimpleLexicalEntry::new(
                None,
                vec![
                    Feature::Selector('d', Direction::Right),
                    Feature::Selector('d', Direction::Left),
                    Feature::Category('V')
                ]
            )
        );
        assert_eq!(
            SimpleLexicalEntry::parse("ε::d= =d V").unwrap(),
            SimpleLexicalEntry {
                lemma: None,
                features: vec![
                    Feature::Selector('d', Direction::Right),
                    Feature::Selector('d', Direction::Left),
                    Feature::Category('V')
                ]
            }
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
                FeatureOrLemma::Feature(Feature::Selector('d', Direction::Right)),
                FeatureOrLemma::Feature(Feature::Selector('d', Direction::Left)),
                FeatureOrLemma::Feature(Feature::Category('V')),
            ]
        );
    }

    use crate::grammars::{COPY_LANGUAGE, STABLER2011};
    use petgraph::dot::Dot;

    #[test]
    fn initialize_lexicon() -> anyhow::Result<()> {
        let strings: Vec<&str> = STABLER2011.split('\n').collect();
        let v: Vec<_> = strings
            .iter()
            .copied()
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()?;

        let lex = Lexicon::new(v);
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
    0 -> 1 [ label = \"-1.791759469228055\" ]
    1 -> 2 [ label = \"-0.6931471805599453\" ]
    2 -> 3 [ label = \"0\" ]
    1 -> 4 [ label = \"-0.6931471805599453\" ]
    4 -> 5 [ label = \"0\" ]
    5 -> 6 [ label = \"0\" ]
    0 -> 7 [ label = \"-1.0986122886681098\" ]
    7 -> 8 [ label = \"0\" ]
    8 -> 9 [ label = \"-0.6931471805599453\" ]
    9 -> 10 [ label = \"-0.6931471805599453\" ]
    9 -> 11 [ label = \"-0.6931471805599453\" ]
    8 -> 12 [ label = \"-0.6931471805599453\" ]
    12 -> 13 [ label = \"-0.6931471805599453\" ]
    12 -> 14 [ label = \"-0.6931471805599453\" ]
    0 -> 15 [ label = \"-1.0986122886681098\" ]
    15 -> 16 [ label = \"-1.3862943611198906\" ]
    15 -> 17 [ label = \"-1.3862943611198906\" ]
    15 -> 18 [ label = \"-1.3862943611198906\" ]
    15 -> 19 [ label = \"-1.3862943611198906\" ]
    0 -> 20 [ label = \"-2.4849066497880004\" ]
    20 -> 21 [ label = \"0\" ]
    21 -> 22 [ label = \"0\" ]
    0 -> 23 [ label = \"-2.4849066497880004\" ]
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

        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()?;
        let lex = Lexicon::new(v);
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
    0 -> 1 [ label = \"-1.791759469228055\" ]
    1 -> 2 [ label = \"0\" ]
    2 -> 3 [ label = \"0\" ]
    3 -> 4 [ label = \"0\" ]
    4 -> 5 [ label = \"0\" ]
    0 -> 6 [ label = \"-0.6931471805599452\" ]
    6 -> 7 [ label = \"-1.0986122886681098\" ]
    7 -> 8 [ label = \"0\" ]
    8 -> 9 [ label = \"0\" ]
    6 -> 10 [ label = \"-0.4054651081081645\" ]
    10 -> 11 [ label = \"0\" ]
    11 -> 12 [ label = \"-0.6931471805599453\" ]
    12 -> 13 [ label = \"0\" ]
    0 -> 14 [ label = \"-1.0986122886681096\" ]
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
    fn mdl_score_test() -> Result<()> {
        let ga: &str = "mary::d -k
laughs::=d +k t
laughed::=d +k t
jumps::=d +k t
jumped::=d +k t";
        let gb: &str = "mary::d -k
laugh::=d v
jump::=d v
s::=v +k t
ed::=v +k t";
        for (g, n_categories, size) in [(ga, 3, 61_f64), (gb, 4, 45_f64)] {
            let strings: Vec<&str> = g.split('\n').collect();
            let v: Vec<_> = strings
                .iter()
                .copied()
                .map(SimpleLexicalEntry::parse)
                .collect::<Result<Vec<_>>>()?;
            let lex = Lexicon::new(v);
            for alphabet_size in [26, 32, 37] {
                let bit_per_symbol: f64 =
                    (MG_TYPES + n_categories + alphabet_size + 1).try_into()?;
                assert_relative_eq!(lex.mdl_score(alphabet_size)?, size * bit_per_symbol.log2());
            }
        }
        Ok(())
    }
}
