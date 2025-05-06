use std::fmt::Debug;

use crate::Direction;

use super::{Feature, FeatureOrLemma, Lexicon, renormalise_weights};
use ahash::AHashSet;
use petgraph::{
    Direction::{Incoming, Outgoing},
    graph::{DiGraph, NodeIndex},
};
use rand::{Rng, seq::IndexedRandom};
use rand_distr::{Distribution, Geometric};

#[derive(Debug)]
struct AccessibilityChecker<'a, T: Eq, Category: Eq> {
    stack: Vec<NodeIndex>,
    seen: AHashSet<NodeIndex>,
    unsatisfiable: AHashSet<NodeIndex>,
    lex: &'a Lexicon<T, Category>,
}

impl<'a, T, C> AccessibilityChecker<'a, T, C>
where
    T: Eq + Debug + Clone,
    C: Eq + Debug + Clone,
{
    fn new(node: NodeIndex, lex: &'a Lexicon<T, C>) -> Self {
        Self {
            stack: lex.graph.neighbors_directed(node, Outgoing).collect(),
            seen: [lex.root, node].into_iter().collect(),
            unsatisfiable: AHashSet::default(),
            lex,
        }
    }

    fn pop(&mut self) -> Option<NodeIndex> {
        match self.stack.pop() {
            Some(x) => {
                self.seen.insert(x);
                Some(x)
            }
            None => None,
        }
    }

    fn add_direct_children(&mut self, node: NodeIndex) {
        self.stack.extend(
            self.lex
                .graph
                .neighbors_directed(node, Outgoing)
                .filter(|x| !self.seen.contains(x)),
        );
    }

    ///Climb up a lexeme marking it unsatisfiable until you reach a branch.
    fn mark_unsatisfiable(&mut self, mut node: NodeIndex) {
        self.unsatisfiable.insert(node);
        let get_parent = |node| self.lex.graph.neighbors_directed(node, Incoming).next();
        while let Some(parent) = get_parent(node) {
            //Check for sisters
            if self.lex.graph.neighbors_directed(node, Outgoing).count() > 1 {
                break;
            } else if !matches!(
                self.lex.graph.node_weight(parent).unwrap(),
                FeatureOrLemma::Root
            ) {
                self.unsatisfiable.insert(parent);
            }
            node = parent;
        }
    }

    fn add_indirect_children(&mut self, node: NodeIndex) {
        match self.lex.graph.node_weight(node).unwrap() {
            FeatureOrLemma::Feature(Feature::Selector(c, _)) | FeatureOrLemma::Complement(c, _) => {
                match self.lex.find_category(c) {
                    Ok(x) => {
                        if !self.seen.contains(&x) {
                            self.stack.push(x);
                        }
                    }
                    Err(_) if !self.lex.has_moving_category(c) => self.mark_unsatisfiable(node),
                    Err(_) => (),
                }
            }
            FeatureOrLemma::Feature(Feature::Licensor(c)) => {
                match self.lex.find_licensee(c) {
                    //Does not check if the licensee can be built
                    //internally  (e.g. if we have type z= w+ c but the -w is not in buildable
                    //starting from z.
                    Ok(x) => {
                        if !self.seen.contains(&x) {
                            self.stack.push(x);
                        }
                    }
                    Err(_) => self.mark_unsatisfiable(node),
                }
            }
            FeatureOrLemma::Root
            | FeatureOrLemma::Lemma(_)
            | FeatureOrLemma::Feature(Feature::Licensee(_))
            | FeatureOrLemma::Feature(Feature::Category(_)) => (),
        }
    }
}

impl<T, C> Lexicon<T, C>
where
    T: Eq + Debug + Clone,
    C: Eq + Debug + Clone,
{
    fn has_moving_category(&self, cat: &C) -> bool {
        let mut stack: Vec<_> = self
            .graph
            .neighbors_directed(self.root, Outgoing)
            .filter(|a| {
                matches!(
                    self.graph.node_weight(*a).unwrap(),
                    FeatureOrLemma::Feature(Feature::Licensee(_))
                )
            })
            .collect();
        while let Some(x) = stack.pop() {
            for x in self.graph.neighbors_directed(x, Outgoing) {
                match &self.graph[x] {
                    FeatureOrLemma::Feature(Feature::Licensee(_)) => stack.push(x),
                    FeatureOrLemma::Feature(Feature::Category(c)) if c == cat => return true,
                    _ => (),
                }
            }
        }
        false
    }

    pub fn prune(&mut self, start: &C) {
        loop {
            let start = match self.find_category(start) {
                Ok(x) => x,
                Err(_) => {
                    self.graph.retain_nodes(|g, n| {
                        matches!(g.node_weight(n).unwrap(), FeatureOrLemma::Root)
                    });
                    self.leaves.clear();
                    return;
                }
            };
            let mut checker = AccessibilityChecker::new(start, self);

            while let Some(node) = checker.pop() {
                checker.add_direct_children(node);
                checker.add_indirect_children(node);
            }

            if checker.unsatisfiable.is_empty() && checker.seen.len() == self.graph.node_count()
            //-1 since we're not gonna see the root
            {
                break;
            } else {
                self.graph.retain_nodes(|_, n| {
                    checker.seen.contains(&n) & !checker.unsatisfiable.contains(&n)
                });
            }
        }

        self.leaves = self
            .graph
            .node_indices()
            .filter(|x| {
                matches!(
                    self.graph.node_weight(*x).unwrap(),
                    FeatureOrLemma::Lemma(_)
                )
            })
            .collect();
    }
}

pub trait FreshCategory: Sized {
    fn fresh(categories: &[Self]) -> Self;
}

impl FreshCategory for usize {
    fn fresh(categories: &[Self]) -> Self {
        match categories.iter().max() {
            Some(x) => x + 1,
            None => 0,
        }
    }
}

impl FreshCategory for u8 {
    fn fresh(categories: &[Self]) -> Self {
        match categories.iter().max() {
            Some(x) => x + 1,
            None => 0,
        }
    }
}
impl FreshCategory for u16 {
    fn fresh(categories: &[Self]) -> Self {
        match categories.iter().max() {
            Some(x) => x + 1,
            None => 0,
        }
    }
}
impl FreshCategory for u32 {
    fn fresh(categories: &[Self]) -> Self {
        match categories.iter().max() {
            Some(x) => x + 1,
            None => 0,
        }
    }
}

impl FreshCategory for u64 {
    fn fresh(categories: &[Self]) -> Self {
        match categories.iter().max() {
            Some(x) => x + 1,
            None => 0,
        }
    }
}

impl<T, C> Lexicon<T, C>
where
    T: Eq + Debug + Clone,
    C: Eq + Debug + Clone + FreshCategory,
{
    pub fn random(
        base_category: &C,
        lemmas: &[T],
        config: Option<LexicalProbConfig>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut graph = DiGraph::new();
        let root = graph.add_node(FeatureOrLemma::Root);
        let mut leaves = vec![];
        let config = config.unwrap_or_default();
        let mut probs = LexicalProbs::new(&config, base_category.clone());

        random_branch(lemmas, root, &mut leaves, &mut graph, &mut probs, rng);

        for leaf in leaves.iter() {
            let parent = graph.neighbors_directed(*leaf, Incoming).next().unwrap();
            let weight = graph.node_weight_mut(parent).unwrap();
            if let FeatureOrLemma::Feature(Feature::Selector(c, d)) = weight {
                *weight = FeatureOrLemma::Complement(c.clone(), *d);
            }
        }

        Lexicon {
            graph: renormalise_weights(graph),
            root,
            leaves,
        }
    }
}

fn random_branch<C: Eq + Clone + Debug + FreshCategory, T: Eq + Clone + Debug>(
    lemmas: &[T],
    root: NodeIndex,
    leaves: &mut Vec<NodeIndex>,
    graph: &mut DiGraph<FeatureOrLemma<T, C>, f64>,
    probs: &mut LexicalProbs<C>,
    rng: &mut impl Rng,
) {
    while let Some(branch_root) = probs.to_branch.pop() {
        let nx = graph.add_node(FeatureOrLemma::Feature(match branch_root {
            MoverOrSelector::Selector(category) => Feature::Category(category),
            MoverOrSelector::Mover(category) => Feature::Licensee(category),
        }));

        graph.add_edge(root, nx, 1.0);
        let mut stack = vec![nx];
        while let Some(n) = stack.pop() {
            probs.set_state(graph.node_weight(n).unwrap());

            let n_children = probs.n_children(rng);

            for _ in 0..n_children {
                match probs.state {
                    ParentNodeType::Root | ParentNodeType::Licensee => {
                        let node = if probs.is_licensee(rng) {
                            let c = probs.choose_category_for_licensee(rng);
                            graph.add_node(FeatureOrLemma::Feature(Feature::Licensee(c.clone())))
                        } else {
                            let c = probs.choose_category_for_category(rng);
                            graph.add_node(FeatureOrLemma::Feature(Feature::Category(c)))
                        };
                        graph.add_edge(n, node, 1.0);
                        stack.push(node);
                    }
                    ParentNodeType::Category
                    | ParentNodeType::Licensor
                    | ParentNodeType::Selector => {
                        if probs.is_lemma(rng) {
                            let lemma = if probs.is_empty(rng) {
                                None
                            } else {
                                lemmas.choose(rng).cloned()
                            };
                            let node = graph.add_node(FeatureOrLemma::Lemma(lemma));
                            graph.add_edge(n, node, 1.0);
                            leaves.push(node);
                        } else {
                            let node = if probs.is_licensor(rng) {
                                let c = probs.choose_category_for_licensor(rng);
                                graph
                                    .add_node(FeatureOrLemma::Feature(Feature::Licensor(c.clone())))
                            } else {
                                let c = probs.choose_category_for_feature(rng);
                                graph.add_node(FeatureOrLemma::Feature(Feature::Selector(
                                    c.clone(),
                                    probs.direction(rng),
                                )))
                            };
                            graph.add_edge(n, node, 1.0);
                            stack.push(node);
                        }
                    }
                }
            }
        }
    }
}
#[derive(Debug, Clone)]
enum MoverOrSelector<C> {
    Selector(C),
    Mover(C),
}

#[derive(Debug, Clone, Copy)]
pub struct LexicalProbConfig {
    children_width: f64,
    lemma_prob: f64,
    empty_prob: f64,
    left_prob: f64,
    add_cat_prob: f64,
    mover_prob: f64,
    licensee_prob: f64,
}

impl Default for LexicalProbConfig {
    fn default() -> Self {
        Self {
            children_width: 0.8,
            lemma_prob: 0.75,
            empty_prob: 0.25,
            left_prob: 0.5,
            add_cat_prob: 0.65,
            mover_prob: 0.2,
            licensee_prob: 0.05,
        }
    }
}

impl<'a, C: Eq + FreshCategory + Clone + Debug> LexicalProbs<'a, C> {
    fn new(config: &'a LexicalProbConfig, category: C) -> Self {
        LexicalProbs {
            children_distr: Geometric::new(config.children_width).unwrap(),
            categories: vec![category.clone()],
            licensee_features: vec![],
            to_branch: vec![MoverOrSelector::Selector(category)],
            config,
            state: ParentNodeType::Category,
        }
    }

    fn n_children(&self, rng: &mut impl Rng) -> u64 {
        self.children_distr.sample(rng) + 1
    }

    fn is_lemma(&self, rng: &mut impl Rng) -> bool {
        match self.state {
            ParentNodeType::Licensor | ParentNodeType::Licensee | ParentNodeType::Root => false,
            ParentNodeType::Category | ParentNodeType::Selector => {
                rng.random_bool(self.config.lemma_prob)
            }
        }
    }

    fn is_empty(&self, rng: &mut impl Rng) -> bool {
        rng.random_bool(self.config.empty_prob)
    }

    fn is_licensor(&mut self, rng: &mut impl Rng) -> bool {
        match self.state {
            ParentNodeType::Licensee | ParentNodeType::Root => false,
            ParentNodeType::Category | ParentNodeType::Licensor | ParentNodeType::Selector => {
                rng.random_bool(self.config.mover_prob)
            }
        }
    }

    fn is_licensee(&mut self, rng: &mut impl Rng) -> bool {
        match self.state {
            ParentNodeType::Licensee | ParentNodeType::Root => {
                rng.random_bool(self.config.licensee_prob)
            }
            ParentNodeType::Category | ParentNodeType::Licensor | ParentNodeType::Selector => false,
        }
    }

    fn direction(&self, rng: &mut impl Rng) -> Direction {
        if rng.random_bool(self.config.left_prob) {
            Direction::Left
        } else {
            Direction::Right
        }
    }

    fn choose_category_for_category(&mut self, rng: &mut impl Rng) -> C {
        self.categories.choose(rng).cloned().unwrap()
    }
    fn choose_category_for_licensee(&mut self, rng: &mut impl Rng) -> C {
        self.licensee_features.choose(rng).cloned().unwrap()
    }

    fn choose_category_for_licensor(&mut self, rng: &mut impl Rng) -> C {
        if self.licensee_features.is_empty()
            || rng.random_bool(self.config.add_cat_prob / self.licensee_features.len() as f64)
        {
            let new_cat = C::fresh(
                &[
                    self.categories.as_slice(),
                    self.licensee_features.as_slice(),
                ]
                .concat(),
            );
            self.licensee_features.push(new_cat.clone());
            self.to_branch.push(MoverOrSelector::Mover(new_cat.clone()));
            new_cat
        } else {
            self.licensee_features.choose(rng).cloned().unwrap()
        }
    }

    fn choose_category_for_feature(&mut self, rng: &mut impl Rng) -> C {
        if rng.random_bool(self.config.add_cat_prob / self.categories.len() as f64) {
            let new_cat = C::fresh(
                &[
                    self.categories.as_slice(),
                    self.licensee_features.as_slice(),
                ]
                .concat(),
            );
            self.categories.push(new_cat.clone());
            self.to_branch
                .push(MoverOrSelector::Selector(new_cat.clone()));
            new_cat
        } else {
            self.categories.choose(rng).cloned().unwrap()
        }
    }

    fn set_state<T: Eq>(&mut self, feature: &FeatureOrLemma<T, C>) {
        self.state = match feature {
            FeatureOrLemma::Feature(Feature::Selector(_, _)) => ParentNodeType::Selector,
            FeatureOrLemma::Feature(Feature::Category(_)) => ParentNodeType::Category,
            FeatureOrLemma::Feature(Feature::Licensor(_)) => ParentNodeType::Licensor,
            FeatureOrLemma::Feature(Feature::Licensee(_)) => ParentNodeType::Licensee,
            FeatureOrLemma::Root => ParentNodeType::Root,
            FeatureOrLemma::Complement(_, _) | FeatureOrLemma::Lemma(_) => {
                panic!("These should not be encountered in random generation")
            }
        };
    }
}

struct LexicalProbs<'a, C: Eq> {
    children_distr: Geometric,
    categories: Vec<C>,
    to_branch: Vec<MoverOrSelector<C>>,
    licensee_features: Vec<C>,
    config: &'a LexicalProbConfig,
    state: ParentNodeType,
}

enum ParentNodeType {
    Licensor,
    Licensee,
    Category,
    Selector,
    Root,
}

#[cfg(feature = "semantics")]
mod semantics;

#[cfg(feature = "semantics")]
pub use semantics::{LearntSemanticLexicon, TypeConstraintData};

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::{Generator, ParsingConfig};

    use super::*;

    #[test]
    fn pruning() -> anyhow::Result<()> {
        let mut lex = Lexicon::parse("A::c= s\nB::d\nC::c")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "A::c= s\nC::c");

        let mut lex = Lexicon::parse("A::z= c= s\nB::d\nC::c")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "");

        let mut lex = Lexicon::parse("A::z= c= s\nB::d\nC::d= c\nD::z")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "A::z= c= s\nB::d\nC::d= c\nD::z");
        let mut lex = Lexicon::parse("A::z= +w s\nD::z -w")?;
        lex.prune(&"s");
        assert_eq!(lex.to_string(), "A::z= +w s\nD::z -w");
        Ok(())
    }

    #[test]
    fn random_lexicon() -> anyhow::Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..100 {
            let x = Lexicon::<_, usize>::random(&0, &["the", "dog", "runs"], None, &mut rng);
            println!("{}", x);
            let config = ParsingConfig::default();
            Generator::new(x, 0, &config)?
                .take(50)
                .for_each(|(p, s, _)| println!("{:.2}: {}", p, s.join(" ")));
            println!("_______________________________________________");
        }
        Ok(())
    }
}
