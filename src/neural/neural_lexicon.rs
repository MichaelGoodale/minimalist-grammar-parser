use std::ops::Range;

use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::HashMap;
use anyhow::{bail, Context};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{activation::log_softmax, backend::Backend, Device, ElementConversion, Tensor};
use burn::tensor::{Data, Int};
use itertools::Itertools;
use logprob::LogProb;
use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef};
use rand::Rng;
use rand_distr::WeightedAliasIndex;
use rand_distr::{Bernoulli, Distribution};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NeuralProbabilityRecord {
    Feature(NodeIndex),
    MergeRuleProb,
    MoveRuleProb,
    OneProb,
}

pub type NeuralFeature = FeatureOrLemma<usize, usize>;
type NeuralGraph = DiGraph<
    (
        Option<(NeuralProbabilityRecord, LogProb<f64>)>,
        NeuralFeature,
    ),
    (),
>;

#[derive(Debug, Clone)]
pub struct NeuralLexicon<B: Backend> {
    weights: HashMap<NeuralProbabilityRecord, Tensor<B, 1>>,
    graph: NeuralGraph,
    root: NodeIndex,
    device: B::Device,
}

pub struct GrammarParameterization<B: Backend> {
    types: Tensor<B, 3>,               //(lexeme, lexeme_pos, type_distribution)
    type_categories: Tensor<B, 3>,     //(lexeme, lexeme_pos, categories_position)
    licensee_categories: Tensor<B, 3>, //(lexeme, lexeme_pos, categories_position)
    lemmas: Tensor<B, 2>,              //(lexeme, lemma_distribution)
    categories: Tensor<B, 2>,          //(lexeme, categories)
    weights: Tensor<B, 1>,             //(lexeme, lexeme_weight)
    included_features: Tensor<B, 2>,   //(lexeme, n_licensee + n_features)
    opposite_included_features: Tensor<B, 2>, //(lexeme, n_licensee + n_features)
    n_lexemes: usize,
    n_features: usize,
    n_licensees: usize,
    n_lemmas: usize,
    type_distributions: HashMap<(usize, usize), WeightedAliasIndex<f64>>,
    type_category_distributions: HashMap<(usize, usize), WeightedAliasIndex<f64>>,
    licensee_category_distributions: HashMap<(usize, usize), WeightedAliasIndex<f64>>,
    category_distributions: HashMap<usize, WeightedAliasIndex<f64>>,
    silent_lemma_probability: HashMap<usize, Bernoulli>,
    included_probs: HashMap<(usize, usize), Bernoulli>,
    silent_probabilities: Tensor<B, 2>,
}

fn get_distribution<B: Backend, const D: usize>(
    x: &Tensor<B, D>,
    slice: [Range<usize>; D],
) -> WeightedAliasIndex<f64> {
    WeightedAliasIndex::new(
        x.clone()
            .slice(slice)
            .exp()
            .to_data()
            .convert::<f64>()
            .value,
    )
    .unwrap()
}

impl<B: Backend> GrammarParameterization<B> {
    pub fn new(
        types: Tensor<B, 3>,               // (lexeme, n_features, types)
        type_categories: Tensor<B, 3>,     // (lexeme, n_features, N_TYPES)
        licensee_categories: Tensor<B, 3>, // (lexeme, n_licensee, categories)
        included_features: Tensor<B, 2>,   // (lexeme, n_licensee + n_features)
        lemmas: Tensor<B, 2>,              // (lexeme, n_lemmas)
        categories: Tensor<B, 2>,          // (lexeme, n_categories)
        weights: Tensor<B, 1>,             // (lexeme)
        temperature: f64,
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

        let mut type_distributions = HashMap::default();
        let mut type_category_distributions = HashMap::default();
        let mut category_distributions = HashMap::default();
        let mut licensee_category_distributions = HashMap::default();
        let mut silent_lemma_probability = HashMap::default();
        let mut included_probs = HashMap::default();

        let heated_categories = log_softmax(categories.clone() / temperature, 1);
        let heated_type_categories = log_softmax(type_categories.clone() / temperature, 2);
        let heated_types = log_softmax(types.clone() / temperature, 2);
        let heated_lemmas = log_softmax(lemmas.clone() / temperature, 1);
        let heated_licensee_categories = log_softmax(licensee_categories.clone() / temperature, 2);

        let included_features = log_sigmoid(included_features);
        let opposite_included_features = (-included_features.clone().exp()).log1p();

        let types = log_softmax(types, 2);
        let type_categories = log_softmax(type_categories, 2);
        let lemmas = log_softmax(lemmas, 1);
        let licensee_categories = log_softmax(licensee_categories, 1);
        let categories = log_softmax(categories, 1);

        let silent_probability: Tensor<B, 1> =
            lemmas.clone().slice([0..n_lexemes, 0..1]).squeeze(1);
        let non_silent_probability: Tensor<B, 1> =
            log_sum_exp_dim::<B, 2, 1>(lemmas.clone().slice([0..n_lexemes, 1..n_lemmas]), 1);

        for lexeme_idx in 0..n_lexemes {
            category_distributions.insert(
                lexeme_idx,
                get_distribution(
                    &heated_categories,
                    [lexeme_idx..lexeme_idx + 1, 0..n_categories],
                ),
            );

            silent_lemma_probability.insert(
                lexeme_idx,
                Bernoulli::new(
                    heated_lemmas
                        .clone()
                        .slice([lexeme_idx..lexeme_idx + 1, 0..1])
                        .exp()
                        .into_scalar()
                        .elem(),
                )?,
            );
            for position in 0..n_licensees {
                licensee_category_distributions.insert(
                    (lexeme_idx, position),
                    get_distribution(
                        &heated_licensee_categories,
                        [
                            lexeme_idx..lexeme_idx + 1,
                            position..position + 1,
                            0..n_categories,
                        ],
                    ),
                );
                included_probs.insert(
                    (lexeme_idx, position),
                    Bernoulli::new(
                        included_features
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, position..position + 1])
                            .exp()
                            .into_scalar()
                            .elem(),
                    )?,
                );
            }

            for position in 0..n_features {
                included_probs.insert(
                    (lexeme_idx, position + n_licensees),
                    Bernoulli::new(
                        included_features
                            .clone()
                            .slice([
                                lexeme_idx..lexeme_idx + 1,
                                n_licensees + position..n_licensees + position + 1,
                            ])
                            .exp()
                            .into_scalar()
                            .elem(),
                    )?,
                );
                type_distributions.insert(
                    (lexeme_idx, position),
                    get_distribution(
                        &heated_types,
                        [
                            lexeme_idx..lexeme_idx + 1,
                            position..position + 1,
                            0..N_TYPES,
                        ],
                    ),
                );
                type_category_distributions.insert(
                    (lexeme_idx, position),
                    get_distribution(
                        &heated_type_categories,
                        [
                            lexeme_idx..lexeme_idx + 1,
                            position..position + 1,
                            0..n_categories,
                        ],
                    ),
                );
            }
        }

        Ok(GrammarParameterization {
            types,
            type_categories,
            lemmas,
            weights,
            licensee_categories,
            included_features,
            opposite_included_features,
            n_lexemes,
            n_features,
            n_licensees,
            n_lemmas,
            categories,
            licensee_category_distributions,
            category_distributions,
            type_distributions,
            included_probs,
            type_category_distributions,
            silent_lemma_probability,
            silent_probabilities: Tensor::cat(
                vec![
                    silent_probability.unsqueeze_dim(1),
                    non_silent_probability.unsqueeze_dim(1),
                ],
                1,
            ),
        })
    }

    pub fn device(&self) -> Device<B> {
        self.types.device()
    }

    fn sample_type(&self, lexeme: usize, position: usize, rng: &mut impl Rng) -> usize {
        self.type_distributions
            .get(&(lexeme, position))
            .unwrap()
            .sample(rng)
    }

    fn sample_licensee_category(
        &self,
        lexeme: usize,
        position: usize,
        rng: &mut impl Rng,
    ) -> usize {
        self.licensee_category_distributions
            .get(&(lexeme, position))
            .unwrap()
            .sample(rng)
    }

    fn sample_type_category(&self, lexeme: usize, position: usize, rng: &mut impl Rng) -> usize {
        self.type_category_distributions
            .get(&(lexeme, position))
            .unwrap()
            .sample(rng)
    }

    fn sample_category(&self, lexeme: usize, rng: &mut impl Rng) -> usize {
        self.category_distributions
            .get(&lexeme)
            .unwrap()
            .sample(rng)
    }

    fn is_licensee(&self, lexeme: usize, n_feature: usize, rng: &mut impl Rng) -> bool {
        self.included_probs
            .get(&(lexeme, n_feature))
            .unwrap()
            .sample(rng)
    }

    fn is_feature(&self, lexeme: usize, n_feature: usize, rng: &mut impl Rng) -> bool {
        self.included_probs
            .get(&(lexeme, self.n_licensees + n_feature))
            .unwrap()
            .sample(rng)
    }

    fn is_silent(&self, lexeme: usize, rng: &mut impl Rng) -> bool {
        self.silent_lemma_probability
            .get(&lexeme)
            .unwrap()
            .sample(rng)
    }

    fn get_lexeme_probability(
        &self,
        lexeme_idx: usize,
        licensees: &[(usize, usize)],
        features: &[(usize, usize, usize)],
        lexeme_category: &usize,
        is_silent: &bool,
    ) -> Tensor<B, 1> {
        let mut probability = Tensor::zeros([1], &self.device());
        let mut feature_positions: Vec<u32> = vec![];
        for (licensee_pos, category) in licensees {
            feature_positions.push(*licensee_pos as u32);
            probability = probability
                + self
                    .licensee_categories
                    .clone()
                    .slice([
                        lexeme_idx..lexeme_idx + 1,
                        *licensee_pos..licensee_pos + 1,
                        *category..category + 1,
                    ])
                    .reshape([1]);
        }
        for (feature_pos, category, feature_type) in features {
            feature_positions.push((feature_pos + self.n_licensees) as u32);
            probability = probability
                + self
                    .type_categories
                    .clone()
                    .slice([
                        lexeme_idx..lexeme_idx + 1,
                        *feature_pos..feature_pos + 1,
                        *category..category + 1,
                    ])
                    .reshape([1])
                + self
                    .types
                    .clone()
                    .slice([
                        lexeme_idx..lexeme_idx + 1,
                        *feature_pos..feature_pos + 1,
                        *feature_type..feature_type + 1,
                    ])
                    .reshape([1]);
        }

        probability = probability
            + self
                .categories
                .clone()
                .slice([
                    lexeme_idx..lexeme_idx + 1,
                    *lexeme_category..lexeme_category + 1,
                ])
                .squeeze(1);

        probability = probability
            + self
                .silent_probabilities
                .clone()
                .slice([
                    lexeme_idx..lexeme_idx + 1,
                    match is_silent {
                        true => 0..1,
                        false => 1..2,
                    },
                ])
                .squeeze(1);

        let mut curr_feat = feature_positions.iter();
        let mut next = curr_feat.next();
        let opposite_idx = (0..(self.n_licensees + self.n_features) as u32)
            .filter(|i| {
                if let Some(pos) = next {
                    if pos == i {
                        next = curr_feat.next();
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .collect_vec();

        let idx = Tensor::<B, 1, Int>::from_data(
            Data::<_, 1>::from(feature_positions.as_slice()).convert(),
            &self.device(),
        );

        let opposite_idx = Tensor::<B, 1, Int>::from_data(
            Data::<_, 1>::from(opposite_idx.as_slice()).convert(),
            &self.device(),
        );

        probability
            + self
                .included_features
                .clone()
                .slice([
                    lexeme_idx..lexeme_idx + 1,
                    0..(self.n_features + self.n_licensees),
                ])
                .select(1, idx)
                .sum()
            + self
                .opposite_included_features
                .clone()
                .slice([
                    lexeme_idx..lexeme_idx + 1,
                    0..(self.n_features + self.n_licensees),
                ])
                .select(1, opposite_idx)
                .sum()
    }

    pub fn n_lemmas(&self) -> usize {
        self.n_lemmas
    }
    pub fn lemmas(&self) -> &Tensor<B, 2> {
        &self.lemmas
    }
}

impl<B: Backend> NeuralLexicon<B> {
    pub fn get_weight(&self, n: &NeuralProbabilityRecord) -> Option<&Tensor<B, 1>> {
        self.weights.get(n)
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn new_random(
        grammar_params: &GrammarParameterization<B>,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 1>, Vec<Vec<NeuralFeature>>, Self) {
        type PositionLog = Vec<usize>;
        let mut graph: DiGraph<(Option<PositionLog>, NeuralFeature), _> = DiGraph::new();

        let root = graph.add_node((None, FeatureOrLemma::Root));
        let mut grammar_prob = Tensor::<B, 1>::zeros([1], &grammar_params.device());
        let mut lexemes = vec![];
        for lexeme_idx in 0..grammar_params.n_lexemes {
            let licensees: Vec<_> = if lexeme_idx > 0 {
                (0..grammar_params.n_licensees)
                    .filter_map(|i| match grammar_params.is_licensee(lexeme_idx, i, rng) {
                        true => Some((
                            i,
                            grammar_params.sample_licensee_category(lexeme_idx, i, rng),
                        )),
                        false => None,
                    })
                    .collect()
            } else {
                //Ensure there is always a lexeme of category 0 without licensees
                vec![]
            };

            let features: Vec<_> = (0..grammar_params.n_features)
                .filter_map(|i| match grammar_params.is_feature(lexeme_idx, i, rng) {
                    true => Some((
                        i,
                        grammar_params.sample_type_category(lexeme_idx, i, rng),
                        grammar_params.sample_type(lexeme_idx, i, rng),
                    )),
                    false => None,
                })
                .collect();

            let category = if lexeme_idx == 0 {
                //Ensure there is always a lexeme of category 0
                0
            } else {
                grammar_params.sample_category(lexeme_idx, rng)
            };
            let is_silent = grammar_params.is_silent(lexeme_idx, rng);
            grammar_prob = grammar_prob
                + grammar_params.get_lexeme_probability(
                    lexeme_idx, &licensees, &features, &category, &is_silent,
                );

            let mut lexeme: Vec<NeuralFeature> = licensees
                .into_iter()
                .map(|(_pos, cat)| NeuralFeature::Feature(Feature::Licensee(cat)))
                .collect();
            lexeme.push(NeuralFeature::Feature(Feature::Category(category)));
            lexeme.extend(
                features
                    .into_iter()
                    .map(|(_pos, cat, feature_type)| to_feature(feature_type, cat)),
            );
            lexeme.push(match is_silent {
                true => NeuralFeature::Lemma(None),
                false => NeuralFeature::Lemma(Some(lexeme_idx)),
            });
            lexemes.push(lexeme);
        }

        for (lexeme_idx, lexeme) in lexemes.iter().enumerate() {
            let mut parent = root;
            for feature in lexeme {
                let mut node = None;
                for child in graph
                    .neighbors_directed(parent, petgraph::Direction::Outgoing)
                    .collect_vec()
                    .into_iter()
                {
                    if let (Some(log), child_feature) = graph.node_weight_mut(child).unwrap() {
                        if child_feature == feature {
                            node = Some(child);
                            log.push(lexeme_idx);
                        }
                    }
                }

                parent = node.unwrap_or_else(|| {
                    let node = graph.add_node((Some(vec![lexeme_idx]), feature.clone()));
                    graph.add_edge(parent, node, ());
                    node
                });
            }
        }

        let mut weights_map: HashMap<NeuralProbabilityRecord, Tensor<B, 1>> = HashMap::default();
        let mut node_map: HashMap<NodeIndex, Option<(NeuralProbabilityRecord, LogProb<f64>)>> =
            HashMap::default();
        node_map.insert(root, None);

        //Renormalise probabilities to sum to one.
        for node_index in graph.node_indices() {
            let neighbours: Vec<_> = graph
                .neighbors_directed(node_index, petgraph::Direction::Outgoing)
                .collect();

            match neighbours.len() {
                0 => (),
                1 => {
                    node_map.insert(
                        *neighbours.first().unwrap(),
                        Some((NeuralProbabilityRecord::OneProb, LogProb::new(0.0).unwrap())),
                    );
                }
                _ => {
                    let mut weights: Tensor<B, 1> =
                        Tensor::zeros([neighbours.len()], &grammar_params.device());
                    for (n_i, n) in neighbours.iter().enumerate() {
                        let log: &PositionLog = graph[*n].0.as_ref().unwrap();
                        let values = log
                            .iter()
                            .map(|i| grammar_params.weights.clone().slice([*i..i + 1]))
                            .fold(Tensor::zeros([1], &grammar_params.device()), |c, acc| {
                                acc + c
                            });
                        weights = weights.slice_assign([n_i..n_i + 1], values);
                    }
                    weights = log_softmax(weights, 0);
                    for (n_i, n) in neighbours.iter().enumerate() {
                        let record = NeuralProbabilityRecord::Feature(*n);
                        let weight = weights.clone().slice([n_i..n_i + 1]);
                        let log_prob = LogProb::new(weight.clone().into_scalar().elem())
                            .unwrap_or_else(|_| LogProb::new(0.0).unwrap());
                        node_map.insert(*n, Some((record, log_prob)));
                        weights_map.insert(record, weight);
                    }
                }
            }
        }

        //Get actual weights that have been log normalised.
        let graph = graph.map(
            |n, (_log, feat)| (node_map.remove(&n).unwrap(), feat.clone()),
            |_, _: &()| (),
        );
        (
            grammar_prob,
            lexemes,
            NeuralLexicon {
                graph,
                weights: weights_map,
                root,
                device: grammar_params.device(),
            },
        )
    }
}

impl<B: Backend> Lexiconable<usize, usize> for NeuralLexicon<B> {
    type Probability = (NeuralProbabilityRecord, LogProb<f64>);

    fn probability_of_one(&self) -> Self::Probability {
        (NeuralProbabilityRecord::OneProb, LogProb::new(0.0).unwrap())
    }

    fn n_children(&self, nx: petgraph::prelude::NodeIndex) -> usize {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .count()
    }

    fn children_of(
        &self,
        nx: petgraph::prelude::NodeIndex,
    ) -> impl Iterator<Item = petgraph::prelude::NodeIndex> + '_ {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .map(|e| e.target())
    }

    fn get(&self, nx: petgraph::prelude::NodeIndex) -> Option<(&NeuralFeature, Self::Probability)> {
        if let Some((Some(p), feature)) = self.graph.node_weight(nx) {
            Some((feature, *p))
        } else {
            None
        }
    }

    fn find_licensee(&self, category: &usize) -> anyhow::Result<petgraph::prelude::NodeIndex> {
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                (_, FeatureOrLemma::Feature(Feature::Licensee(c))) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }

    fn find_category(&self, category: &usize) -> anyhow::Result<petgraph::prelude::NodeIndex> {
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                (_, FeatureOrLemma::Feature(Feature::Category(c))) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::Tensor;

    use super::{GrammarParameterization, N_TYPES};

    #[test]
    fn get_param() -> anyhow::Result<()> {
        let n_lexemes = 2;
        let n_pos = 5;
        let n_licensee = 2;
        let n_categories = 5;
        let n_lemmas = 10;
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
        let included_features = Tensor::<NdArray, 2>::random(
            [n_lexemes, n_licensee + n_pos],
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

        let _g = GrammarParameterization::new(
            types,
            type_categories,
            licensee_categories,
            included_features,
            lemmas,
            categories,
            weights,
            1.0,
        );
        Ok(())
    }
}
