use std::{f64::consts::LN_2, fmt::Debug, hash::Hash};

use crate::Direction;

use super::{Feature, FeatureOrLemma, Lexicon};
use ahash::{AHashMap, AHashSet};
use itertools::Itertools;
use logprob::{LogProb, LogSumExp};
use petgraph::{
    Direction::{Incoming, Outgoing},
    graph::NodeIndex,
    prelude::StableDiGraph,
    visit::EdgeRef,
};
use rand::{
    Rng,
    seq::{IndexedRandom, IteratorRandom},
};
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

fn fix_weights_per_node<T: Eq + Clone, C: Eq + Clone>(
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

fn fix_weights<T: Eq + Clone, C: Eq + Clone>(
    graph: &mut StableDiGraph<FeatureOrLemma<T, C>, LogProb<f64>>,
) {
    //Renormalise probabilities to sum to one.
    for node_index in graph.node_indices().collect_vec() {
        fix_weights_per_node(graph, node_index);
    }
}

impl<T, C> Lexicon<T, C>
where
    T: Eq + Debug + Clone + Hash,
    C: Eq + Debug + Clone + FreshCategory + Hash,
{
    pub fn add_new_lexeme_randomly(&mut self, lemma: T, rng: &mut impl Rng) -> Option<NodeIndex> {
        if let Some(&leaf) = self
            .leaves
            .iter()
            .filter(|&&x| matches!(self.graph.node_weight(x).unwrap(), FeatureOrLemma::Lemma(Some(s)) if s!=&lemma))
            .choose(rng)
        {
            let parent = self.parent_of(leaf).unwrap();
            let node = self.graph.add_node(FeatureOrLemma::Lemma(Some(lemma)));
            self.graph.add_edge(parent, node, LogProb::prob_of_one());
            fix_weights_per_node(&mut self.graph, parent);
            self.leaves.push(node);
            Some(leaf)
        }else{
            None
        }
    }

    pub fn delete_from_node(&mut self, rng: &mut impl Rng) {
        if let Some(&node) = self
            .graph
            .node_indices()
            .filter(|&nx| match self.graph.node_weight(nx).unwrap() {
                FeatureOrLemma::Root => false,
                _ => {
                    let parent = self.parent_of(nx).unwrap();
                    //We can only delete a branch if there's at least one sibling.
                    self.children_of(parent).count() > 1
                }
            })
            .collect::<Vec<_>>()
            .choose(rng)
        {
            let parent = self.parent_of(node).unwrap();
            let mut stack = vec![node];
            while let Some(nx) = stack.pop() {
                stack.extend(self.children_of(nx));
                self.graph.remove_node(nx);
            }
            fix_weights_per_node(&mut self.graph, parent);
            self.leaves.retain(|&x| self.graph.contains_node(x));
        }
    }

    pub fn delete_node(&mut self, rng: &mut impl Rng) -> Option<NodeIndex> {
        if let Some(&node) = self
            .graph
            .node_indices()
            .filter(|&nx| match self.graph.node_weight(nx).unwrap() {
                FeatureOrLemma::Root | FeatureOrLemma::Feature(Feature::Category(_)) => false,
                FeatureOrLemma::Lemma(_) => {
                    let parent = self.parent_of(nx).unwrap();
                    //If the lemma node has no siblings, deleting it will make an invalid grammar.
                    self.graph.edges_directed(parent, Outgoing).count() > 1
                }
                FeatureOrLemma::Complement(_, _) => {
                    let parent = self.parent_of(nx).unwrap();
                    //If the parent of a complement is a licensor, we can't delete it
                    !matches!(
                        self.graph[parent],
                        FeatureOrLemma::Feature(Feature::Licensor(_))
                    )
                }
                _ => true,
            })
            .collect::<Vec<_>>()
            .choose(rng)
        {
            let e = self.graph.edges_directed(node, Incoming).next().unwrap();
            let parent = e.source();
            let w = *e.weight();

            let edges = self
                .graph
                .edges_directed(node, Outgoing)
                .map(|e| (e.target(), w + e.weight()))
                .collect::<Vec<_>>();

            if !edges.is_empty() {
                if matches!(
                    self.graph.node_weight(parent).unwrap(),
                    FeatureOrLemma::Feature(Feature::Selector(_, _))
                ) {
                    let mut complement_edges = vec![];
                    let mut selector_edges = vec![];

                    for (child, w) in edges {
                        if matches!(
                            self.graph.node_weight(child).unwrap(),
                            FeatureOrLemma::Lemma(_)
                        ) {
                            complement_edges.push((child, w));
                        } else {
                            selector_edges.push((child, w));
                        }
                    }
                    if complement_edges.is_empty() {
                        for (child, w) in selector_edges {
                            self.graph.add_edge(parent, child, w);
                        }
                    } else if selector_edges.is_empty()
                        && self.graph.edges_directed(parent, Outgoing).count() == 1
                    {
                        for (child, w) in complement_edges {
                            self.graph.add_edge(parent, child, w);
                        }
                        self.graph[parent].into_complement();
                    } else {
                        let mut f = self.graph[parent].clone();
                        f.into_complement();
                        let alt_parent = self.graph.add_node(f);
                        let grand_parent = self.parent_of(parent).unwrap();
                        let parent_e = self
                            .graph
                            .edges_directed(grand_parent, Incoming)
                            .next()
                            .unwrap()
                            .id();

                        self.graph[parent_e] += LogProb::new(-LN_2).unwrap();
                        self.graph
                            .add_edge(grand_parent, alt_parent, self.graph[parent_e]);

                        for (child, w) in selector_edges {
                            self.graph.add_edge(parent, child, w);
                        }
                        for (child, w) in complement_edges {
                            self.graph.add_edge(alt_parent, child, w);
                        }
                    }
                } else {
                    for (child, w) in edges {
                        self.graph.add_edge(parent, child, w);
                    }
                }
            }

            if matches!(self.graph[node], FeatureOrLemma::Lemma(_)) {
                self.leaves.retain(|&a| a != node);
            }
            self.graph.remove_node(node);
            self.clean_up();
            Some(node)
        } else {
            None
        }
    }

    pub fn random(
        base_category: &C,
        lemmas: &[T],
        config: Option<LexicalProbConfig>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut graph = StableDiGraph::new();
        let root = graph.add_node(FeatureOrLemma::Root);
        let node = graph.add_node(FeatureOrLemma::Feature(Feature::Category(
            base_category.clone(),
        )));
        graph.add_edge(root, node, LogProb::prob_of_one());

        let mut lexicon = Lexicon {
            graph,
            root,
            leaves: vec![],
        };
        let config = config.unwrap_or_default();
        let mut probs = LexicalProbs::from_lexicon(&mut lexicon, lemmas, &config);
        probs.descend_from(node, rng);
        probs.add_novel_branches(rng);
        lexicon.clean_up();
        lexicon
    }

    pub fn change_feature(
        &mut self,
        lemmas: &[T],
        config: Option<LexicalProbConfig>,
        rng: &mut impl Rng,
    ) {
        let config = config.unwrap_or_default();
        if let Some(&node) = self
            .graph
            .node_indices()
            .filter(|nx| !matches!(self.graph.node_weight(*nx).unwrap(), FeatureOrLemma::Root))
            .collect::<Vec<_>>()
            .choose(rng)
        {
            let mut probs = LexicalProbs::from_lexicon(self, lemmas, &config);
            probs.set_node(node, rng);
            self.clean_up();
        }
    }

    pub fn resample_below_node(
        &mut self,
        lemmas: &[T],
        config: Option<LexicalProbConfig>,
        rng: &mut impl Rng,
    ) {
        let config = config.unwrap_or_default();
        if let Some(&node) = self
            .graph
            .node_indices()
            .filter(|nx| {
                !matches!(
                    self.graph.node_weight(*nx).unwrap(),
                    FeatureOrLemma::Root | FeatureOrLemma::Lemma(_)
                )
            })
            .collect::<Vec<_>>()
            .choose(rng)
        {
            let mut children: Vec<_> = self.children_of(node).collect();

            let mut probs = LexicalProbs::from_lexicon(self, lemmas, &config);
            probs.descend_from(node, rng);
            probs.add_novel_branches(rng);
            while let Some(child) = children.pop() {
                children.extend(
                    self.graph
                        .edges_directed(child, Outgoing)
                        .map(|x| x.target()),
                );
                if matches!(
                    self.graph.node_weight(child).unwrap(),
                    FeatureOrLemma::Lemma(_)
                ) {
                    self.leaves.retain(|&x| x != child);
                }
                self.graph.remove_node(child);
            }
            self.clean_up();
        }
    }

    fn clean_up(&mut self) {
        fix_weights(&mut self.graph);

        let mut stack = vec![self.root];

        while let Some(n) = stack.pop() {
            let mut features: AHashMap<_, Vec<_>> = AHashMap::default();

            for child in self.children_of(n) {
                let feature = self.graph.node_weight(child).unwrap().clone();
                features.entry(feature).or_default().push(child);
            }

            for (key, mut nodes_to_merge) in features.into_iter() {
                if nodes_to_merge.len() == 1 {
                    stack.push(nodes_to_merge.pop().unwrap());
                } else if matches!(key, FeatureOrLemma::Lemma(_)) {
                    stack.extend(nodes_to_merge);
                } else {
                    let sum = nodes_to_merge
                        .iter()
                        .flat_map(|&a| self.graph.edges_directed(a, Outgoing))
                        .map(|x| x.weight())
                        .log_sum_exp_float_no_alloc();

                    let incoming_weight = nodes_to_merge
                        .iter()
                        .flat_map(|&a| self.graph.edges_directed(a, Incoming))
                        .map(|x| x.weight())
                        .log_sum_exp_clamped_no_alloc();

                    let node_to_keep = nodes_to_merge.pop().unwrap();

                    let new_edges: Vec<_> = nodes_to_merge
                        .iter()
                        .flat_map(|&a| self.graph.edges_directed(a, Outgoing))
                        .map(|e| {
                            (
                                e.target(),
                                LogProb::new(e.weight().into_inner() - sum).unwrap(),
                            )
                        })
                        .collect();

                    for (e, p) in self
                        .graph
                        .edges_directed(node_to_keep, Outgoing)
                        .map(|e| (e.id(), LogProb::new(e.weight().into_inner() - sum).unwrap()))
                        .collect::<Vec<_>>()
                    {
                        self.graph[e] = p;
                    }

                    new_edges.into_iter().for_each(|(tgt, weight)| {
                        self.graph.add_edge(node_to_keep, tgt, weight);
                    });

                    if let Some(e) = self
                        .graph
                        .edges_directed(node_to_keep, Incoming)
                        .next()
                        .map(|e| e.id())
                    {
                        self.graph[e] = incoming_weight;
                    }
                    nodes_to_merge.into_iter().for_each(|x| {
                        self.graph.remove_node(x);
                    });
                    stack.push(node_to_keep);
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

impl<'a, 'b, 'c, T: Eq + Clone + Debug, C: Eq + FreshCategory + Clone + Debug>
    LexicalProbs<'a, 'b, 'c, T, C>
{
    fn from_lexicon(
        lexicon: &'b mut Lexicon<T, C>,
        lemmas: &'c [T],
        config: &'a LexicalProbConfig,
    ) -> Self {
        LexicalProbs {
            children_distr: Geometric::new(config.children_width).unwrap(),
            categories: lexicon.categories().cloned().collect(),
            licensee_features: lexicon.licensor_types().cloned().collect(),
            to_branch: vec![],
            config,
            lemmas,
            lexicon,
        }
    }
    fn n_children(&self, rng: &mut impl Rng) -> u64 {
        self.children_distr.sample(rng) + 1
    }

    fn is_lemma(&self, rng: &mut impl Rng) -> bool {
        rng.random_bool(self.config.lemma_prob)
    }

    fn get_feature(&mut self, rng: &mut impl Rng) -> FeatureOrLemma<T, C> {
        if rng.random_bool(self.config.mover_prob) {
            let c = self.choose_category_for_licensor(rng);
            FeatureOrLemma::Feature(Feature::Licensor(c))
        } else {
            let c = self.choose_category_for_feature(rng);
            if self.is_lemma(rng) {
                FeatureOrLemma::Complement(c, self.direction(rng))
            } else {
                FeatureOrLemma::Feature(Feature::Selector(c, self.direction(rng)))
            }
        }
    }

    fn get_licensee_or_category(&mut self, rng: &mut impl Rng) -> Feature<C> {
        if rng.random_bool(self.config.licensee_prob) {
            let c = self.choose_category_for_licensee(rng);
            Feature::Licensee(c)
        } else {
            let c = self.choose_category_for_category(rng);
            Feature::Category(c)
        }
    }

    fn add_novel_branches(&mut self, rng: &mut impl Rng) {
        while let Some(p) = self.to_branch.pop() {
            let f = FeatureOrLemma::Feature(match p {
                MoverOrSelector::Selector(c) => Feature::Category(c),
                MoverOrSelector::Mover(c) => Feature::Licensee(c),
            });
            let node = self.lexicon.graph.add_node(f);
            self.lexicon
                .graph
                .add_edge(self.lexicon.root, node, LogProb::prob_of_one());
            self.descend_from(node, rng);
        }
    }

    fn descend_from(&mut self, node: NodeIndex, rng: &mut impl Rng) {
        let mut stack = vec![node];

        while let Some(node) = stack.pop() {
            let feature = self.lexicon.graph.node_weight(node).unwrap();
            match feature {
                FeatureOrLemma::Root => unimplemented!(),
                FeatureOrLemma::Lemma(_) => (),
                FeatureOrLemma::Feature(Feature::Licensee(_)) => {
                    let n_children = self.n_children(rng);
                    for _ in 0..n_children {
                        let feature = FeatureOrLemma::Feature(self.get_licensee_or_category(rng));
                        let child = self.lexicon.graph.add_node(feature);
                        self.lexicon
                            .graph
                            .add_edge(node, child, LogProb::prob_of_one());
                        stack.push(child);
                    }
                }
                FeatureOrLemma::Feature(Feature::Licensor(_))
                | FeatureOrLemma::Feature(Feature::Selector(_, _)) => {
                    let n_children = self.n_children(rng);
                    for _ in 0..n_children {
                        let feature = self.get_feature(rng);
                        let child = self.lexicon.graph.add_node(feature);
                        self.lexicon
                            .graph
                            .add_edge(node, child, LogProb::prob_of_one());
                        stack.push(child);
                    }
                }
                FeatureOrLemma::Feature(Feature::Category(_)) => {
                    let n_children = self.n_children(rng);
                    for _ in 0..n_children {
                        let is_lemma = self.is_lemma(rng);
                        let feature = if is_lemma {
                            FeatureOrLemma::Lemma(self.get_lemma(rng))
                        } else {
                            self.get_feature(rng)
                        };
                        let child = self.lexicon.graph.add_node(feature);
                        self.lexicon
                            .graph
                            .add_edge(node, child, LogProb::prob_of_one());
                        if is_lemma {
                            self.lexicon.leaves.push(child);
                        } else {
                            stack.push(child);
                        }
                    }
                }
                FeatureOrLemma::Complement(_, _) => {
                    let n_children = self.n_children(rng);
                    for _ in 0..n_children {
                        let child = self
                            .lexicon
                            .graph
                            .add_node(FeatureOrLemma::Lemma(self.get_lemma(rng)));
                        self.lexicon
                            .graph
                            .add_edge(node, child, LogProb::prob_of_one());
                        self.lexicon.leaves.push(child);
                    }
                }
            }
        }
    }

    fn get_lemma(&self, rng: &mut impl Rng) -> Option<T> {
        if rng.random_bool(self.config.empty_prob) {
            None
        } else {
            self.lemmas.choose(rng).cloned()
        }
    }

    fn direction(&self, rng: &mut impl Rng) -> Direction {
        if rng.random_bool(self.config.left_prob) {
            Direction::Left
        } else {
            Direction::Right
        }
    }

    fn set_node(&mut self, node: NodeIndex, rng: &mut impl Rng) {
        let n = self.lexicon.graph.node_weight(node).unwrap();

        let new_feature = match n {
            FeatureOrLemma::Root => FeatureOrLemma::Root,
            FeatureOrLemma::Lemma(_) => FeatureOrLemma::Lemma(self.get_lemma(rng)),
            FeatureOrLemma::Feature(f) => FeatureOrLemma::Feature(match f {
                Feature::Category(_) => Feature::Category(self.choose_category_for_category(rng)),
                Feature::Licensee(_) => Feature::Licensee(self.choose_category_for_licensee(rng)),
                _ => {
                    let lemma_children =
                        self.lexicon
                            .graph
                            .neighbors_directed(node, Outgoing)
                            .any(|x| {
                                matches!(
                                    self.lexicon.graph.node_weight(x).unwrap(),
                                    FeatureOrLemma::Lemma(_)
                                )
                            });

                    if rng.random_bool(self.config.mover_prob) && !lemma_children {
                        let feature = self.choose_category_for_licensor(rng);
                        Feature::Licensor(feature)
                    } else {
                        let feature = self.choose_category_for_feature(rng);
                        Feature::Selector(feature, self.direction(rng))
                    }
                }
            }),
            FeatureOrLemma::Complement(..) => FeatureOrLemma::Complement(
                self.choose_category_for_feature(rng),
                self.direction(rng),
            ),
        };
        *self.lexicon.graph.node_weight_mut(node).unwrap() = new_feature;
    }

    fn choose_category_for_category(&mut self, rng: &mut impl Rng) -> C {
        if self.categories.is_empty() {
            self.categories = vec![C::fresh(&self.licensee_features)];
        }
        self.categories.choose(rng).cloned().unwrap()
    }

    fn choose_category_for_licensee(&mut self, rng: &mut impl Rng) -> C {
        if self.licensee_features.is_empty() {
            self.licensee_features = vec![C::fresh(&self.categories)];
        }
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
}

struct LexicalProbs<'a, 'b, 'c, T: Eq, C: Eq> {
    children_distr: Geometric,
    categories: Vec<C>,
    to_branch: Vec<MoverOrSelector<C>>,
    licensee_features: Vec<C>,
    config: &'a LexicalProbConfig,
    lemmas: &'c [T],
    lexicon: &'b mut Lexicon<T, C>,
}

#[cfg(test)]
mod test {
    use anyhow::bail;
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn total_prob<T: Eq, C: Eq + Debug>(lex: &Lexicon<T, C>, node: NodeIndex) -> f64 {
        lex.graph
            .edges_directed(node, Outgoing)
            .map(|x| x.weight())
            .log_sum_exp_float()
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Position {
        Root,
        PreCategory,
        Category,
        PostCategory,
        NextIsLemma,
        Done,
    }

    impl Position {
        fn next_pos<T: Eq + Debug, C: Eq + Debug>(
            &self,
            f: &FeatureOrLemma<T, C>,
        ) -> anyhow::Result<Self> {
            match self {
                Position::Root | Position::PreCategory => match f {
                    FeatureOrLemma::Feature(Feature::Category(_)) => Ok(Position::Category),
                    FeatureOrLemma::Feature(Feature::Licensee(_)) => Ok(Position::PreCategory),
                    _ => bail!("can't go from {:?} to {:?} !", self, f),
                },
                Position::Category => match f {
                    FeatureOrLemma::Feature(Feature::Selector(..))
                    | FeatureOrLemma::Feature(Feature::Licensor(_)) => Ok(Position::PostCategory),
                    FeatureOrLemma::Complement(..) => Ok(Position::NextIsLemma),
                    FeatureOrLemma::Lemma(_) => Ok(Position::Done),
                    _ => bail!("can't go from {:?} to {:?} !", self, f),
                },
                Position::PostCategory => match f {
                    FeatureOrLemma::Feature(Feature::Selector(..))
                    | FeatureOrLemma::Feature(Feature::Licensor(_)) => Ok(Position::PostCategory),
                    FeatureOrLemma::Complement(..) => Ok(Position::NextIsLemma),
                    _ => bail!("can't go from {:?} to {:?} !", self, f),
                },
                Position::NextIsLemma => match f {
                    FeatureOrLemma::Lemma(_) => Ok(Position::Done),
                    _ => bail!("can't go from {:?} to {:?} !", self, f),
                },
                Position::Done => bail!("Done can't be continued"),
            }
        }
    }

    fn validate_lexicon<T: Eq + Debug + Clone, C: Eq + Debug + Clone>(
        lex: &Lexicon<T, C>,
    ) -> anyhow::Result<()> {
        let mut at_least_one_category = false;
        let mut found_leaves = AHashSet::default();
        let mut found_root = None;
        let mut stack = vec![(lex.root, Position::Root)];

        while let Some((nx, pos)) = stack.pop() {
            let children = lex.children_of(nx).collect::<Vec<_>>();

            if pos == Position::Done {
                assert!(children.is_empty())
            } else {
                assert!(!children.is_empty())
            }
            for child in children {
                let f = lex.graph.node_weight(child).unwrap();
                if matches!(f, FeatureOrLemma::Feature(Feature::Category(_))) {
                    at_least_one_category = true;
                }
                let next_pos = pos.next_pos(f)?;
                stack.push((child, next_pos))
            }
        }
        assert!(at_least_one_category);

        for node in lex.graph.node_indices() {
            let mut parent_iter = lex.graph.neighbors_directed(node, Incoming);
            let parent = parent_iter.next();
            assert!(parent_iter.next().is_none());
            let children: Vec<_> = lex.graph.neighbors_directed(node, Outgoing).collect();
            let de_duped: Vec<_> = children
                .iter()
                .map(|&x| lex.graph.node_weight(x))
                .filter(|x| !matches!(x.unwrap(), FeatureOrLemma::Lemma(_)))
                .dedup()
                .collect();

            assert_eq!(
                de_duped.len(),
                children
                    .iter()
                    .filter(|&&a| !matches!(
                        lex.graph.node_weight(a).unwrap(),
                        FeatureOrLemma::Lemma(_)
                    ))
                    .count()
            );

            match lex.graph.node_weight(node).unwrap() {
                FeatureOrLemma::Root => {
                    assert!(parent.is_none());
                    if found_root.is_some() {
                        panic!("Multiple roots!");
                    }
                    found_root = Some(node);
                }
                FeatureOrLemma::Lemma(_) => {
                    assert!(parent.is_some());
                    found_leaves.insert(node);

                    assert!(matches!(
                        lex.graph.node_weight(parent.unwrap()).unwrap(),
                        FeatureOrLemma::Complement(_, _)
                            | FeatureOrLemma::Feature(Feature::Category(_))
                    ));
                    assert!(children.is_empty());
                }
                FeatureOrLemma::Feature(_) => {
                    assert!(parent.is_some());
                    assert!(!children.is_empty());
                    approx::assert_relative_eq!(total_prob(lex, node), 0.0, epsilon = 1e-10);
                }
                FeatureOrLemma::Complement(_, _) => {
                    assert!(parent.is_some());
                    assert!(!children.is_empty());
                    assert!(children.into_iter().all(|x| matches!(
                        lex.graph.node_weight(x).unwrap(),
                        FeatureOrLemma::Lemma(_)
                    )));
                    approx::assert_relative_eq!(total_prob(lex, node), 0.0, epsilon = 1e-10);
                }
            }
        }

        let leaves: AHashSet<_> = lex.leaves.iter().copied().collect();
        assert_eq!(leaves, found_leaves);
        assert_eq!(leaves.len(), lex.leaves.len());

        Ok(())
    }

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
        for _ in 0..1000 {
            let x = Lexicon::<_, usize>::random(&0, &["the", "dog", "runs"], None, &mut rng);
            dbg!(&x);
            validate_lexicon(&x)?;
        }
        Ok(())
    }

    #[test]
    fn random_redone_lexicon() -> anyhow::Result<()> {
        let mut main_rng = ChaCha8Rng::seed_from_u64(0);
        let lemmas = &["the", "dog", "runs"];
        for _ in 0..1000 {
            let mut rng = ChaCha8Rng::seed_from_u64(497);
            let mut lex = Lexicon::<_, usize>::random(&0, lemmas, None, &mut rng);
            validate_lexicon(&lex)?;
            lex.resample_below_node(lemmas, None, &mut main_rng);
            validate_lexicon(&lex)?;
        }
        Ok(())
    }

    #[test]
    fn random_delete_branch() -> anyhow::Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let lemmas = &["the", "dog", "runs"];
        for _ in 0..10000 {
            let mut lex = Lexicon::<_, usize>::random(&0, lemmas, None, &mut rng);
            validate_lexicon(&lex)?;
            lex.delete_from_node(&mut rng);
            validate_lexicon(&lex)?;
        }
        Ok(())
    }

    #[test]
    fn random_delete_feat() -> anyhow::Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let lemmas = &["the", "dog", "runs"];
        for _ in 0..10000 {
            let mut lex = Lexicon::<_, usize>::random(&0, lemmas, None, &mut rng);
            validate_lexicon(&lex)?;
            lex.delete_node(&mut rng);
            validate_lexicon(&lex)?;
        }
        Ok(())
    }
    #[test]
    fn random_add_lexeme() -> anyhow::Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let lemmas = &["the", "dog", "runs"];
        for _ in 0..10000 {
            let mut lex = Lexicon::<_, usize>::random(&0, lemmas, None, &mut rng);
            validate_lexicon(&lex)?;
            lex.add_new_lexeme_randomly("lbarg", &mut rng);
            validate_lexicon(&lex)?;
        }
        Ok(())
    }

    #[test]
    fn random_change_feat() -> anyhow::Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let lemmas = &["the", "dog", "runs"];
        for _ in 0..10000 {
            let mut lex = Lexicon::<_, usize>::random(&0, lemmas, None, &mut rng);
            println!("{}", lex);
            validate_lexicon(&lex)?;
            println!("NEW VERSION");
            lex.change_feature(lemmas, None, &mut rng);
            validate_lexicon(&lex)?;
            println!("{lex}");
            println!("_______________________________________________");
        }
        Ok(())
    }
}
