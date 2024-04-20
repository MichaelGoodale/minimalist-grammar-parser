use std::collections::BTreeSet;

use super::loss::NeuralConfig;
use super::neural_beam::{NodeFeature, StringProbHistory};
use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::HashMap;
use anyhow::{bail, Context};
use burn::tensor::{activation::log_softmax, backend::Backend, Device, ElementConversion, Tensor};
use burn::tensor::{Data, Shape};
use itertools::Itertools;
use logprob::LogProb;
use petgraph::visit::EdgeRef;
use petgraph::{
    graph::DiGraph,
    graph::{EdgeIndex, NodeIndex},
};
use rand::prelude::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Gumbel};

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
    categories: HashMap<usize, Vec<(NeuralProbability, NodeIndex)>>,
    licensees: HashMap<usize, Vec<(NeuralProbability, NodeIndex)>>,
    alternatives: HashMap<NodeIndex, Vec<NodeIndex>>,
    n_lexemes: usize,
    graph: NeuralGraph,
    device: B::Device,
}

#[derive(Debug, Clone)]
pub struct GrammarParameterization<B: Backend> {
    types: Tensor<B, 3>,                //(lexeme, lexeme_pos, type_distribution)
    type_categories: Tensor<B, 3>,      //(lexeme, lexeme_pos, categories_position)
    licensee_categories: Tensor<B, 3>,  //(lexeme, lexeme_pos, categories_position)
    lemmas: Tensor<B, 2>,               //(lexeme, lemma_distribution)
    categories: Tensor<B, 2>,           //(lexeme, categories)
    included_features: Tensor<B, 2>,    //(lexeme, n_features)
    included_licensees: Tensor<B, 2>,   //(lexeme, n_licensees)
    weights: Tensor<B, 1>,              //(lexeme)
    unnormalized_weights: Tensor<B, 1>, //(lexeme)
    include_lemma: Tensor<B, 2>,        // (lexeme, 2)
    lemma_lookups: HashMap<(usize, usize), LogProb<f64>>,
    lexeme_weights: HashMap<usize, LogProb<f64>>,
    n_lexemes: usize,
    n_features: usize,
    n_licensees: usize,
    n_categories: usize,
    n_lemmas: usize,
    pad_vector: Tensor<B, 1>, //(n_lemmas)
    end_vector: Tensor<B, 1>, //(n_lemmas)
    silent_probabilities: Tensor<B, 2>,
}

fn clamp_prob(x: f64) -> anyhow::Result<f64> {
    if x > 0.0 {
        eprintln!("Clampling prob {x} to 0.0");
        Ok(0.0)
    } else if x.is_nan() {
        bail!("Probability is NaN");
    } else if x.is_infinite() {
        bail!("Probability is NaN or inf");
    } else {
        Ok(x)
    }
}

fn tensor_to_log_prob<B: Backend>(x: &Tensor<B, 1>) -> anyhow::Result<LogProb<f64>> {
    Ok(LogProb::new(clamp_prob(x.clone().into_scalar().elem())?)?)
}

fn gumbel_vector<B: Backend, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
    rng: &mut impl Rng,
) -> Tensor<B, D> {
    let n = shape.dims.iter().product();
    let g = Gumbel::<f64>::new(0.0, 1.0).unwrap();
    let v = g.sample_iter(rng).take(n).collect::<Vec<_>>();
    let data = Data::from(v.as_slice());
    Tensor::from_data(data.convert(), device).reshape(shape)
}

fn gumbel_activation<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    inverse_temperature: f64,
    rng: &mut impl Rng,
) -> Tensor<B, D> {
    let shape = tensor.shape();
    let device = tensor.device();

    log_softmax(
        (log_softmax(tensor, dim) + gumbel_vector(shape, &device, rng)) * inverse_temperature,
        dim,
    )
}

fn activation_function<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    inverse_temperature: f64,
    gumbel: bool,
    rng: &mut impl Rng,
) -> Tensor<B, D> {
    if gumbel {
        gumbel_activation(tensor, dim, inverse_temperature, rng)
    } else {
        log_softmax(tensor, dim)
    }
}

impl<B: Backend> GrammarParameterization<B> {
    pub fn new(
        types: Tensor<B, 3>,                // (lexeme, n_features, types)
        type_categories: Tensor<B, 3>,      // (lexeme, n_features, N_TYPES)
        licensee_categories: Tensor<B, 3>,  // (lexeme, n_licensee, categories)
        included_features: Tensor<B, 2>,    // (lexeme, n_features)
        included_licensees: Tensor<B, 2>,   // (lexeme, n_licensees)
        lemmas: Tensor<B, 2>,               // (lexeme, n_lemmas)
        silent_probabilities: Tensor<B, 2>, //(lexeme, 2)
        categories: Tensor<B, 2>,           // (lexeme, n_categories)
        weights: Tensor<B, 1>,              // (lexeme)
        include_lemma: Tensor<B, 2>,
        pad_vector: Tensor<B, 1>,
        end_vector: Tensor<B, 1>,
        temperature: f64,
        gumbel: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<GrammarParameterization<B>> {
        let [n_lexemes, n_features, n_categories] = type_categories.shape().dims;
        let n_lemmas = lemmas.shape().dims[1];
        let n_licensees = licensee_categories.shape().dims[1];

        if n_lemmas <= 1 {
            bail!(format!("The model needs at least 2 lemmas, not {n_lemmas} since idx=0 is the silent lemma."))
        }
        if types.shape().dims[2] != N_TYPES {
            bail!(format!(
                "Tensor types must be of shape [n_lexemes, n_features, {N_TYPES}]"
            ));
        }

        let inverse_temperature = 1.0 / temperature;

        let included_features =
            activation_function(included_features, 1, inverse_temperature, gumbel, rng);
        let included_licensees =
            activation_function(included_licensees, 1, inverse_temperature, gumbel, rng);
        let include_lemma = activation_function(include_lemma, 1, inverse_temperature, gumbel, rng);

        let types = activation_function(types, 2, inverse_temperature, gumbel, rng);
        let type_categories =
            activation_function(type_categories, 2, inverse_temperature, gumbel, rng);
        let lemmas = activation_function(lemmas, 1, inverse_temperature, gumbel, rng);
        let unnormalized_weights = weights.clone();
        let weights = log_softmax(weights, 0);
        let licensee_categories =
            activation_function(licensee_categories, 2, inverse_temperature, gumbel, rng);
        let categories = activation_function(categories, 1, inverse_temperature, gumbel, rng);

        let pad_vector = log_softmax(pad_vector, 0);
        let end_vector = log_softmax(end_vector, 0);

        let silent_probabilities =
            gumbel_activation(silent_probabilities, 1, inverse_temperature, rng);

        let mut lemma_lookups = HashMap::default();
        let mut lexeme_weights = HashMap::default();
        for lexeme_i in 0..n_lexemes {
            let v = lemmas
                .clone()
                .slice([lexeme_i..lexeme_i + 1, 0..n_lemmas])
                .to_data()
                .convert::<f64>()
                .value;
            for (lemma_i, v) in v.into_iter().enumerate() {
                lemma_lookups.insert((lexeme_i, lemma_i), LogProb::new(clamp_prob(v)?).unwrap());
            }
            lexeme_weights.insert(
                lexeme_i,
                LogProb::new(clamp_prob(
                    weights
                        .clone()
                        .slice([lexeme_i..lexeme_i + 1])
                        .into_scalar()
                        .elem(),
                )?)
                .unwrap(),
            );
        }

        Ok(GrammarParameterization {
            types,
            type_categories,
            lemmas,
            licensee_categories,
            included_features,
            included_licensees,
            include_lemma,
            weights,
            unnormalized_weights,
            lemma_lookups,
            n_lexemes,
            n_features,
            lexeme_weights,
            n_licensees,
            n_lemmas,
            n_categories,
            categories,
            pad_vector,
            end_vector,
            silent_probabilities,
        })
    }

    pub fn pad_vector(&self) -> &Tensor<B, 1> {
        &self.pad_vector
    }

    pub fn end_vector(&self) -> &Tensor<B, 1> {
        &self.end_vector
    }

    pub fn lemma_lookups(&self) -> &HashMap<(usize, usize), LogProb<f64>> {
        &self.lemma_lookups
    }

    pub fn lexeme_weights(&self) -> &HashMap<usize, LogProb<f64>> {
        &self.lexeme_weights
    }

    pub fn include_lemma(&self) -> &Tensor<B, 2> {
        &self.include_lemma
    }
    pub fn included_licensees(&self) -> &Tensor<B, 2> {
        &self.included_licensees
    }
    pub fn included_features(&self) -> &Tensor<B, 2> {
        &self.included_features
    }

    pub fn unnormalized_weights(&self) -> &Tensor<B, 1> {
        &self.unnormalized_weights
    }

    pub fn device(&self) -> Device<B> {
        self.types.device()
    }

    pub fn n_lemmas(&self) -> usize {
        self.n_lemmas
    }

    pub fn n_lexemes(&self) -> usize {
        self.n_lexemes
    }

    pub fn n_categories(&self) -> usize {
        self.n_categories
    }

    pub fn lemmas(&self) -> &Tensor<B, 2> {
        &self.lemmas
    }
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
        rng: &mut impl Rng,
        config: &NeuralConfig,
    ) -> anyhow::Result<Self> {
        let mut licensees_map = HashMap::default();
        let mut categories_map = HashMap::default();
        let mut weights_map = HashMap::default();
        let mut alternative_map = HashMap::default();

        let mut graph: NeuralGraph = DiGraph::new();

        let mut order = (0..grammar_params.n_lexemes).collect_vec();
        order.shuffle(rng);
        for lexeme_idx in order {
            let lexeme_root = graph.add_node((FeatureOrLemma::Root, LogProb::new(0.0).unwrap()));
            let mut first_features: Vec<_> = (0..grammar_params.n_categories)
                .map(|c| {
                    (
                        1,
                        FeatureOrLemma::Feature(Feature::Licensee(c)),
                        grammar_params
                            .licensee_categories
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, 0..1, c..c + 1])
                            .reshape([1]),
                    )
                })
                .chain((0..grammar_params.n_categories).map(|c| {
                    (
                        0,
                        FeatureOrLemma::Feature(Feature::Category(c)),
                        grammar_params
                            .categories
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, c..c + 1])
                            .reshape([1]),
                    )
                }))
                .collect();
            let mut all_categories = vec![];
            let mut parent_licensees = vec![];
            first_features.shuffle(rng);
            for (n_licensees, feature, prob) in first_features {
                let lexeme_p = prob.clone();
                let lexeme_weight = grammar_params
                    .weights
                    .clone()
                    .slice([lexeme_idx..lexeme_idx + 1]);

                let log_prob = tensor_to_log_prob(&lexeme_p)?;
                let node = graph.add_node((feature, log_prob));
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
                let n_features = 0..=grammar_params.n_features;
                let n_licensees = match n_licensees {
                    0 => 0..1,
                    _ => 1..grammar_params.n_licensees + 1,
                };
                let mut ps = n_features
                    .cartesian_product(n_licensees)
                    .filter_map(|(n_features, n_licensees)| {
                        let log_prob = log_prob
                            + tensor_to_log_prob(
                                &(grammar_params
                                    .included_features
                                    .clone()
                                    .slice([lexeme_idx..lexeme_idx + 1, n_features..n_features + 1])
                                    .reshape([1])
                                    + grammar_params
                                        .included_licensees
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
                weights_map.insert(NeuralProbabilityRecord::Node(node), lexeme_weight);
                let feature = &graph[node].0;
                ps.shuffle(rng);
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

            for n_licensees in 1..grammar_params.n_licensees {
                let mut new_parent_licensees = vec![];
                let mut licensees = (0..grammar_params.n_categories)
                    .map(|c| {
                        (
                            FeatureOrLemma::Feature(Feature::Licensee(c)),
                            grammar_params
                                .licensee_categories
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
                licensees.shuffle(rng);
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
                .silent_probabilities
                .clone()
                .slice([lexeme_idx..lexeme_idx + 1, 0..1])
                .reshape([1]);
            let nonsilent_p = grammar_params
                .silent_probabilities
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

            let mut lemmas = [lemma, silent_lemma];
            lemmas.shuffle(rng);
            add_alternatives(&mut alternative_map, &lemmas);

            let mut parents = all_categories;
            for n_categories in 0..grammar_params.n_features {
                let new_parents: Vec<_> = (0..grammar_params.n_categories)
                    .cartesian_product(0..N_TYPES)
                    .collect();

                let mut new_parents = new_parents
                    .into_iter()
                    .map(|(c, t)| {
                        let feature = to_feature(t, c);
                        let p = (grammar_params.type_categories.clone().slice([
                            lexeme_idx..lexeme_idx + 1,
                            n_categories..n_categories + 1,
                            c..c + 1,
                        ]) + grammar_params.types.clone().slice([
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
                new_parents.shuffle(rng);
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
            licensees: licensees_map,
            categories: categories_map,
            weights: weights_map,
            alternatives: alternative_map,
            n_lexemes: grammar_params.n_lexemes,
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

    use super::{GrammarParameterization, N_TYPES};
    use rand::SeedableRng;

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
        let include_lemma = Tensor::<NdArray, 2>::random(
            [n_lexemes, 2],
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(32);

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
