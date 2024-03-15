use std::ops::Range;

use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::HashMap;
use anyhow::Context;
use bitvec::array::BitArray;
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{activation::log_softmax, backend::Backend, Device, ElementConversion, Tensor};
use itertools::Itertools;
use logprob::LogProb;
use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::WeightedAliasIndex;

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
    licensor_categories: Tensor<B, 3>, //(lexeme, lexeme_pos, categories_position)
    lemmas: Tensor<B, 2>,              //(lexeme, lemma_distribution)
    categories: Tensor<B, 2>,          //(lexeme, categories)
    weights: Tensor<B, 1>,             //(lexeme, lexeme_weight)
    included_features: Tensor<B, 2>,   //(lexeme, n_licensor + n_features)
    n_lexemes: usize,
    n_features: usize,
    n_categories: usize,
    n_licensors: usize,
    n_lemmas: usize,
    type_distributions: HashMap<(usize, usize), WeightedAliasIndex<f64>>,
    type_category_distributions: HashMap<(usize, usize), WeightedAliasIndex<f64>>,
    licensor_category_distributions: HashMap<(usize, usize), WeightedAliasIndex<f64>>,
    category_distributions: HashMap<usize, WeightedAliasIndex<f64>>,
    silent_lemma_probability: HashMap<usize, f64>,
    included_probs: HashMap<(usize, usize), f64>,
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
        type_categories: Tensor<B, 3>,     // (lexeme, n_features, categories)
        licensor_categories: Tensor<B, 3>, // (lexeme, n_licensor, categories)
        included_features: Tensor<B, 2>,   // (lexeme, n_licensor + n_features)
        lemmas: Tensor<B, 2>,              // (lexeme, n_lemmas)
        categories: Tensor<B, 2>,          // (lexeme, n_categories)
        weights: Tensor<B, 1>,             // (lexeme)
        temperature: f64,
    ) -> GrammarParameterization<B> {
        let [n_lexemes, n_features, n_categories] = type_categories.shape().dims;
        let n_lemmas = lemmas.shape().dims[1];
        let n_licensors = licensor_categories.shape().dims[1];
        let mut type_distributions = HashMap::default();
        let mut type_category_distributions = HashMap::default();
        let mut category_distributions = HashMap::default();
        let mut licensor_category_distributions = HashMap::default();
        let mut silent_lemma_probability = HashMap::default();
        let mut included_probs = HashMap::default();

        let heated_categories = log_softmax(categories.clone() / temperature, 1);
        let heated_type_categories = log_softmax(type_categories.clone() / temperature, 2);
        let heated_types = log_softmax(types.clone() / temperature, 2);
        let heated_lemmas = log_softmax(lemmas.clone() / temperature, 1);
        let heated_licensor_categories = log_softmax(licensor_categories.clone() / temperature, 2);

        let included_features = log_sigmoid(included_features);

        let types = log_softmax(types, 2);
        let type_categories = log_softmax(type_categories, 2);
        let lemmas = log_softmax(lemmas, 1);
        let licensor_categories = log_softmax(licensor_categories, 1);
        let categories = log_softmax(categories, 1);

        let silent_probability = lemmas.clone().slice([0..n_lexemes, 0..1]);
        let non_silent_probability =
            log_sum_exp_dim(lemmas.clone().slice([0..n_lexemes, 1..n_lemmas]), 1);

        for lexeme_idx in 0..n_lexemes {
            category_distributions.insert(
                lexeme_idx,
                get_distribution(&heated_categories, [lexeme_idx..lexeme_idx + 1, 0..N_TYPES]),
            );

            silent_lemma_probability.insert(
                lexeme_idx,
                heated_lemmas
                    .clone()
                    .slice([lexeme_idx..lexeme_idx + 1, 0..1])
                    .exp()
                    .into_scalar()
                    .elem(),
            );
            for position in 0..n_licensors {
                licensor_category_distributions.insert(
                    (lexeme_idx, position),
                    get_distribution(
                        &heated_licensor_categories,
                        [
                            lexeme_idx..lexeme_idx + 1,
                            position..position + 1,
                            0..n_categories,
                        ],
                    ),
                );
                included_probs.insert(
                    (lexeme_idx, position),
                    included_features
                        .clone()
                        .slice([lexeme_idx..lexeme_idx + 1, position..position + 1])
                        .exp()
                        .into_scalar()
                        .elem(),
                );
            }

            for position in 0..n_features {
                included_probs.insert(
                    (lexeme_idx, position),
                    included_features
                        .clone()
                        .slice([
                            lexeme_idx..lexeme_idx + 1,
                            n_licensors + position..n_licensors + position + 1,
                        ])
                        .exp()
                        .into_scalar()
                        .elem(),
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
                            0..N_TYPES,
                        ],
                    ),
                );
            }
        }

        GrammarParameterization {
            types,
            type_categories,
            lemmas,
            weights,
            licensor_categories,
            included_features,
            n_lexemes,
            n_features,
            n_categories,
            n_licensors,
            n_lemmas,
            categories,
            licensor_category_distributions,
            category_distributions,
            type_distributions,
            included_probs,
            type_category_distributions,
            silent_lemma_probability,
            silent_probabilities: Tensor::cat(vec![silent_probability, non_silent_probability], 1),
        }
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

    fn sample_licensor_category(
        &self,
        lexeme: usize,
        position: usize,
        rng: &mut impl Rng,
    ) -> usize {
        self.licensor_category_distributions
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

    fn is_licensor(&self, lexeme: usize, n_feature: usize, rng: &mut impl Rng) -> bool {
        rng.gen_bool(*self.included_probs.get(&(lexeme, n_feature)).unwrap())
    }

    fn is_feature(&self, lexeme: usize, n_feature: usize, rng: &mut impl Rng) -> bool {
        rng.gen_bool(
            *self
                .included_probs
                .get(&(lexeme, self.n_licensors + n_feature))
                .unwrap(),
        )
    }

    fn is_silent(&self, lexeme: usize, rng: &mut impl Rng) -> bool {
        !rng.gen_bool(*self.silent_lemma_probability.get(&lexeme).unwrap())
    }

    fn get_lexeme_probability(
        &self,
        licensors: &[(usize, usize)],
        features: &[(usize, usize, usize)],
        category: &usize,
        is_silent: &bool,
    ) -> Tensor<B, 1> {
        todo!();
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
            let licensors: Vec<_> = (0..grammar_params.n_licensors)
                .filter_map(|i| match grammar_params.is_licensor(lexeme_idx, i, rng) {
                    true => Some((
                        i,
                        grammar_params.sample_licensor_category(lexeme_idx, i, rng),
                    )),
                    false => None,
                })
                .collect();

            let features: Vec<_> = (0..grammar_params.n_licensors)
                .filter_map(|i| match grammar_params.is_feature(lexeme_idx, i, rng) {
                    true => Some((
                        i,
                        grammar_params.sample_type_category(lexeme_idx, i, rng),
                        grammar_params.sample_type(lexeme_idx, i, rng),
                    )),
                    false => None,
                })
                .collect();

            let category = grammar_params.sample_category(lexeme_idx, rng);
            let is_silent = grammar_params.is_silent(lexeme_idx, rng);
            grammar_prob = grammar_prob
                + grammar_params
                    .get_lexeme_probability(&licensors, &features, &category, &is_silent);

            let mut lexeme: Vec<NeuralFeature> = licensors
                .into_iter()
                .map(|(_pos, cat)| NeuralFeature::Feature(Feature::Licensor(cat)))
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
                parent = node
                    .unwrap_or_else(|| graph.add_node((Some(vec![lexeme_idx]), feature.clone())));
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
                        let log_prob = LogProb::new(weights.clone().into_scalar().elem()).unwrap();
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
    use burn::tensor::activation::log_softmax;
    use burn::tensor::Tensor;
    use rand::SeedableRng;

    use super::{NeuralLexicon, N_TYPES};

    #[test]
    fn sample_grammar() -> anyhow::Result<()> {
        let lemmas = log_softmax(
            Tensor::<NdArray, 3>::random(
                [3, 5, 3],
                burn::tensor::Distribution::Default,
                &NdArrayDevice::default(),
            ),
            2,
        );
        let types = log_softmax(
            Tensor::<NdArray, 3>::random(
                [3, 5, N_TYPES],
                burn::tensor::Distribution::Default,
                &NdArrayDevice::default(),
            ),
            2,
        );
        let categories = log_softmax(
            Tensor::<NdArray, 3>::random(
                [3, 5, 2],
                burn::tensor::Distribution::Default,
                &NdArrayDevice::default(),
            ),
            2,
        );
        let weights = Tensor::<NdArray, 2>::random(
            [3, 5],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );

        let mut r = rand::rngs::StdRng::seed_from_u64(123);

        NeuralLexicon::new_random(
            &types,
            &types,
            &categories,
            &categories,
            &lemmas,
            &weights,
            &mut r,
        );
        Ok(())
    }
}
