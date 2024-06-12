use std::collections::BTreeSet;

use super::loss::NeuralConfig;
use super::neural_beam::{NodeFeature, StringProbHistory};
use super::parameterization::GrammarParameterization;
use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::HashMap;
use anyhow::Context;
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use itertools::Itertools;
use logprob::LogProb;
use petgraph::visit::EdgeRef;
use petgraph::{
    graph::DiGraph,
    graph::{EdgeIndex, NodeIndex},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum EdgeHistory {
    AtLeastNLicenseses {
        lexeme_idx: usize,
        n_licensees: usize,
    },
    AtLeastNCategories {
        lexeme_idx: usize,
        n_categories: usize,
    },
    AtMostNLicenseses {
        lexeme_idx: usize,
        n_licensees: usize,
    },
    AtMostNCategories {
        lexeme_idx: usize,
        n_categories: usize,
    },
}

impl EdgeHistory {
    pub fn lexeme_idx(&self) -> usize {
        match self {
            EdgeHistory::AtLeastNLicenseses {
                lexeme_idx,
                n_licensees: _,
            }
            | EdgeHistory::AtMostNLicenseses {
                lexeme_idx,
                n_licensees: _,
            }
            | EdgeHistory::AtLeastNCategories {
                lexeme_idx,
                n_categories: _,
            }
            | EdgeHistory::AtMostNCategories {
                lexeme_idx,
                n_categories: _,
            } => *lexeme_idx,
        }
    }

    pub fn is_compatible(&self, max_n_licensees: usize, max_n_features: usize) -> bool {
        match self {
            EdgeHistory::AtLeastNLicenseses { n_licensees, .. } => *n_licensees <= max_n_licensees,
            EdgeHistory::AtLeastNCategories { n_categories, .. } => *n_categories <= max_n_features,
            EdgeHistory::AtMostNLicenseses { n_licensees, .. } => max_n_licensees == *n_licensees,
            EdgeHistory::AtMostNCategories { n_categories, .. } => max_n_features == *n_categories,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum NeuralProbabilityRecord {
    Lexeme {
        node: NodeIndex,
        id: usize,
        n_features: usize,
        n_licensees: usize,
    },
    Node(NodeIndex),
    MergeRuleProb,
    MoveRuleProb,
    OneProb,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NeuralProbability(
    pub NeuralProbabilityRecord,
    pub Option<EdgeHistory>,
    pub LogProb<f64>,
);

pub type NeuralFeature = FeatureOrLemma<usize, usize>;
type NeuralGraph = DiGraph<(NeuralFeature, LogProb<f64>), EdgeHistory>;

#[derive(Debug, Clone)]
pub struct NeuralLexicon<B: Backend> {
    weights: HashMap<NeuralProbabilityRecord, Tensor<B, 1>>,
    index_to_id: HashMap<NodeIndex, usize>,
    categories: HashMap<usize, Vec<(NeuralProbability, NodeIndex)>>,
    licensees: HashMap<usize, Vec<(NeuralProbability, NodeIndex)>>,
    alternatives: HashMap<NodeIndex, Vec<NodeIndex>>,
    n_lexemes: usize,
    graph: NeuralGraph,
    device: B::Device,
}

fn tensor_to_log_prob<B: Backend>(x: &Tensor<B, 1>) -> anyhow::Result<LogProb<f64>> {
    Ok(LogProb::new(clamp_prob(x.clone().into_scalar().elem())?)?)
}

fn add_alternatives(map: &mut HashMap<NodeIndex, Vec<NodeIndex>>, nodes: &[NodeIndex]) {
    for a in nodes.iter() {
        map.insert(
            *a,
            nodes
                .iter()
                .filter_map(|x| if x == a { None } else { Some(*x) })
                .collect(),
        );
    }
}

impl<B: Backend> NeuralLexicon<B> {
    pub fn node_to_id(&self, nx: &NodeIndex) -> usize {
        *self
            .index_to_id
            .get(nx)
            .expect("Passing a node which is not a child of a root should be impossible")
    }

    pub fn get_weight(&self, n: &NeuralProbabilityRecord) -> Option<&Tensor<B, 1>> {
        self.weights.get(n)
    }

    pub fn get_feature(&self, e: EdgeIndex) -> &NeuralFeature {
        &self.graph[self.graph.edge_endpoints(e).unwrap().1].0
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    //TODO: look into if weights should be done only over categories

    pub fn new_superimposed(
        grammar_params: &GrammarParameterization<B>,
        config: &NeuralConfig,
    ) -> anyhow::Result<Self> {
        let mut licensees_map = HashMap::default();
        let mut categories_map = HashMap::default();
        let mut weights_map = HashMap::default();
        let mut alternative_map = HashMap::default();
        let mut index_to_id = HashMap::default();

        let mut graph: NeuralGraph = DiGraph::new();

        for lexeme_idx in 0..grammar_params.n_lexemes() {
            let lexeme_root = graph.add_node((FeatureOrLemma::Root, LogProb::new(0.0).unwrap()));
            let first_features: Vec<_> = (0..grammar_params.n_categories())
                .map(|c| {
                    (
                        1,
                        FeatureOrLemma::Feature(Feature::Licensee(c)),
                        grammar_params
                            .licensee_categories()
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, 0..1, c..c + 1])
                            .reshape([1]),
                    )
                })
                .chain((0..grammar_params.n_categories()).map(|c| {
                    (
                        0,
                        FeatureOrLemma::Feature(Feature::Category(c)),
                        grammar_params
                            .categories()
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, c..c + 1])
                            .reshape([1]),
                    )
                }))
                .collect();
            let mut all_categories = vec![];
            let mut parent_licensees = vec![];
            for (n_licensees, feature, prob) in first_features {
                let lexeme_p = prob.clone();
                let log_prob = tensor_to_log_prob(&lexeme_p)?;

                let node = graph.add_node((feature, log_prob));
                index_to_id.insert(node, lexeme_idx);
                let edge_history = match n_licensees {
                    0 => EdgeHistory::AtMostNLicenseses {
                        lexeme_idx,
                        n_licensees,
                    },
                    _ => EdgeHistory::AtLeastNLicenseses {
                        lexeme_idx,
                        n_licensees,
                    },
                };
                graph.add_edge(lexeme_root, node, edge_history);
                let n_features = 0..=grammar_params.n_features();
                let n_licensees = match n_licensees {
                    0 => 0..1,
                    _ => 1..grammar_params.n_licensees() + 1,
                };
                let ps = n_features
                    .cartesian_product(n_licensees)
                    .filter_map(|(n_features, n_licensees)| {
                        let log_prob = log_prob
                            + tensor_to_log_prob(
                                &(grammar_params
                                    .included_features()
                                    .clone()
                                    .slice([lexeme_idx..lexeme_idx + 1, n_features..n_features + 1])
                                    .reshape([1])
                                    + grammar_params
                                        .included_licensees()
                                        .clone()
                                        .slice([
                                            lexeme_idx..lexeme_idx + 1,
                                            n_licensees..n_licensees + 1,
                                        ])
                                        .reshape([1])),
                            )
                            .unwrap();
                        match config.parsing_config.min_log_prob {
                            Some(min) if log_prob < min => None,
                            _ => Some((
                                NeuralProbability(
                                    NeuralProbabilityRecord::Lexeme {
                                        node,
                                        id: lexeme_idx,
                                        n_features,
                                        n_licensees,
                                    },
                                    Some(edge_history),
                                    log_prob,
                                ),
                                node,
                            )),
                        }
                    })
                    .collect_vec();
                weights_map.insert(NeuralProbabilityRecord::Node(node), lexeme_p);
                let feature = &graph[node].0;
                match feature {
                    FeatureOrLemma::Feature(Feature::Category(c)) => {
                        all_categories.push(node);
                        categories_map.entry(*c).or_insert(vec![]).extend(ps);
                    }
                    FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                        parent_licensees.push(node);
                        licensees_map.entry(*c).or_insert(vec![]).extend(ps);
                    }
                    _ => {
                        panic!("Invalid first feature!")
                    }
                };
            }

            add_alternatives(&mut alternative_map, &all_categories);
            add_alternatives(&mut alternative_map, &parent_licensees);

            for (licensee, category) in parent_licensees.iter().zip(all_categories.iter()) {
                graph.add_edge(
                    *licensee,
                    *category,
                    EdgeHistory::AtMostNLicenseses {
                        lexeme_idx,
                        n_licensees: 1,
                    },
                );
            }

            for n_licensees in 1..grammar_params.n_licensees() {
                let mut new_parent_licensees = vec![];
                let licensees = (0..grammar_params.n_categories())
                    .map(|c| {
                        (
                            FeatureOrLemma::Feature(Feature::Licensee(c)),
                            grammar_params
                                .licensee_categories()
                                .clone()
                                .slice([
                                    lexeme_idx..lexeme_idx + 1,
                                    n_licensees..n_licensees + 1,
                                    c..c + 1,
                                ])
                                .reshape([1]),
                        )
                    })
                    .collect::<Vec<_>>();
                for (feature, prob) in licensees.into_iter() {
                    let node = graph.add_node((feature, tensor_to_log_prob(&prob)?));
                    weights_map.insert(NeuralProbabilityRecord::Node(node), prob);

                    for parent in parent_licensees.iter() {
                        graph.add_edge(
                            *parent,
                            node,
                            EdgeHistory::AtLeastNLicenseses {
                                lexeme_idx,
                                n_licensees: n_licensees + 1,
                            },
                        );
                    }
                    for category in all_categories.iter() {
                        graph.add_edge(
                            node,
                            *category,
                            EdgeHistory::AtMostNLicenseses {
                                lexeme_idx,
                                n_licensees: n_licensees + 1,
                            },
                        );
                    }
                    new_parent_licensees.push(node);
                }
                add_alternatives(&mut alternative_map, &new_parent_licensees);
                parent_licensees = new_parent_licensees;
            }

            let silent_p = grammar_params
                .silent_probabilities()
                .clone()
                .slice([lexeme_idx..lexeme_idx + 1, 0..1])
                .reshape([1]);
            let nonsilent_p = grammar_params
                .silent_probabilities()
                .clone()
                .slice([lexeme_idx..lexeme_idx + 1, 1..2])
                .reshape([1]);

            let lemma = graph.add_node((
                NeuralFeature::Lemma(Some(lexeme_idx)),
                tensor_to_log_prob(&nonsilent_p)?,
            ));
            let silent_lemma =
                graph.add_node((NeuralFeature::Lemma(None), tensor_to_log_prob(&silent_p)?));
            weights_map.insert(NeuralProbabilityRecord::Node(lemma), nonsilent_p);
            weights_map.insert(NeuralProbabilityRecord::Node(silent_lemma), silent_p);

            for category in all_categories.iter() {
                graph.add_edge(
                    *category,
                    lemma,
                    EdgeHistory::AtMostNCategories {
                        lexeme_idx,
                        n_categories: 0,
                    },
                );
                graph.add_edge(
                    *category,
                    silent_lemma,
                    EdgeHistory::AtMostNCategories {
                        lexeme_idx,
                        n_categories: 0,
                    },
                );
            }

            let lemmas = [lemma, silent_lemma];
            add_alternatives(&mut alternative_map, &lemmas);

            let mut parents = all_categories;
            for n_categories in 0..grammar_params.n_features() {
                let new_parents: Vec<_> = (0..grammar_params.n_categories())
                    .cartesian_product(0..N_TYPES)
                    .collect();

                let new_parents = new_parents
                    .into_iter()
                    .map(|(c, t)| {
                        let feature = to_feature(t, c);
                        let p = (grammar_params.type_categories().clone().slice([
                            lexeme_idx..lexeme_idx + 1,
                            n_categories..n_categories + 1,
                            c..c + 1,
                        ]) + grammar_params.types().clone().slice([
                            lexeme_idx..lexeme_idx + 1,
                            n_categories..n_categories + 1,
                            t..t + 1,
                        ]))
                        .reshape([1]);
                        let node = graph.add_node((feature, tensor_to_log_prob(&p)?));
                        weights_map.insert(NeuralProbabilityRecord::Node(node), p);
                        Ok(node)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                for (node, parent) in new_parents.iter().cartesian_product(parents.iter()) {
                    graph.add_edge(
                        *parent,
                        *node,
                        EdgeHistory::AtLeastNCategories {
                            lexeme_idx,
                            n_categories: n_categories + 1,
                        },
                    );
                }

                for (node, lemma) in new_parents.iter().cartesian_product(lemmas.iter()) {
                    graph.add_edge(
                        *node,
                        *lemma,
                        EdgeHistory::AtMostNCategories {
                            lexeme_idx,
                            n_categories: n_categories + 1,
                        },
                    );
                }

                add_alternatives(&mut alternative_map, &new_parents);
                parents = new_parents;
            }
        }

        Ok(NeuralLexicon {
            graph,
            index_to_id,
            licensees: licensees_map,
            categories: categories_map,
            weights: weights_map,
            alternatives: alternative_map,
            n_lexemes: grammar_params.n_lexemes(),
            device: grammar_params.device(),
        })
    }

    pub fn n_lexemes(&self) -> usize {
        self.n_lexemes
    }

    pub fn has_alternative(&self, nx: &NodeIndex, path: &BTreeSet<NodeFeature>) -> bool {
        self.alternatives
            .get(nx)
            .unwrap_or(&vec![])
            .iter()
            .any(|x| path.contains(&NodeFeature::Node(*x)))
    }

    pub fn grammar_features(&self, s: &StringProbHistory) -> Vec<Vec<NeuralFeature>> {
        let nodes = s.attested_nodes();
        nodes
            .iter()
            .filter_map(|x| match x {
                NodeFeature::Node(_) => None,
                NodeFeature::NFeats { node, .. } => {
                    let mut v = vec![self.get(*node).unwrap().clone()];
                    let mut current = Some(*node);
                    while current.is_some() {
                        let mut next = None;
                        for child in self
                            .graph
                            .neighbors_directed(current.unwrap(), petgraph::Direction::Outgoing)
                        {
                            if nodes.contains(&NodeFeature::Node(child)) {
                                v.push(self.get(child).unwrap().clone());
                                next = Some(child);
                                break;
                            }
                        }
                        current = next;
                    }
                    Some(v)
                }
            })
            .collect()
    }
}

impl<B: Backend> Lexiconable<usize, usize> for NeuralLexicon<B> {
    type Probability = NeuralProbability;

    fn probability_of_one(&self) -> Self::Probability {
        NeuralProbability(
            NeuralProbabilityRecord::OneProb,
            None,
            LogProb::new(0.0).unwrap(),
        )
    }

    fn n_children(&self, nx: petgraph::prelude::NodeIndex) -> usize {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .count()
    }

    fn children_of(
        &self,
        nx: petgraph::prelude::NodeIndex,
    ) -> impl Iterator<Item = (Self::Probability, petgraph::prelude::NodeIndex)> + '_ {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .map(|e| {
                let n = e.target();
                let log_prob = self.graph[n].1;
                let p = NeuralProbability(
                    NeuralProbabilityRecord::Node(e.target()),
                    Some(*e.weight()),
                    log_prob,
                );
                (p, n)
            })
    }

    fn get(&self, nx: petgraph::prelude::NodeIndex) -> Option<&NeuralFeature> {
        self.graph.node_weight(nx).map(|x| &x.0)
    }

    fn find_licensee(&self, category: &usize) -> anyhow::Result<&[(Self::Probability, NodeIndex)]> {
        self.licensees
            .get(category)
            .map(Vec::as_slice)
            .with_context(|| format!("{category:?} is not a valid licensee in the lexicon!"))
    }

    fn find_category(&self, category: &usize) -> anyhow::Result<&[(Self::Probability, NodeIndex)]> {
        self.categories
            .get(category)
            .map(Vec::as_slice)
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::Tensor;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::{GrammarParameterization, N_TYPES};

    #[test]
    fn get_param() -> anyhow::Result<()> {
        let n_lexemes = 2;
        let n_pos = 5;
        let n_licensee = 2;
        let n_categories = 5;
        let n_lemmas = 6;
        let lemmas = Tensor::<NdArray, 2>::random(
            [n_lexemes, n_lemmas],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );

        let types = Tensor::<NdArray, 3>::random(
            [n_lexemes, n_pos, N_TYPES],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );

        let type_categories = Tensor::<NdArray, 3>::random(
            [n_lexemes, n_pos, N_TYPES],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );

        let licensee_categories = Tensor::<NdArray, 3>::random(
            [n_lexemes, n_licensee, n_categories],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let included_licensees = Tensor::<NdArray, 2>::random(
            [n_lexemes, n_licensee + 1],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let included_features = Tensor::<NdArray, 2>::random(
            [n_lexemes, n_pos + 1],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );

        let categories = Tensor::<NdArray, 2>::random(
            [n_lexemes, n_categories],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let weights = Tensor::<NdArray, 1>::random(
            [n_lexemes],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let silent_probabilities = Tensor::<NdArray, 2>::random(
            [n_lexemes, 2],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let include_lemma = Tensor::<NdArray, 1>::random(
            [n_lexemes],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let pad_vector = Tensor::<NdArray, 1>::from_floats(
            [10., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            &NdArrayDevice::default(),
        );
        let end_vector = Tensor::<NdArray, 1>::from_floats(
            [0., 10., 0., 0., 0., 0., 0., 0., 0., 0.],
            &NdArrayDevice::default(),
        );
        let mut rng = StdRng::seed_from_u64(0);

        let _g = GrammarParameterization::new(
            types,
            type_categories,
            licensee_categories,
            included_features,
            included_licensees,
            lemmas,
            silent_probabilities,
            categories,
            weights,
            include_lemma,
            pad_vector,
            end_vector,
            1.0,
            true,
            &mut rng,
        );
        Ok(())
    }
}
