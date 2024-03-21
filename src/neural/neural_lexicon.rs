use std::ops::Range;

use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::HashMap;
use anyhow::{bail, Context};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{activation::log_softmax, backend::Backend, Device, ElementConversion, Tensor};
use burn::tensor::{Data, Int};
use itertools::Itertools;
use logprob::log_sum_exp_float;
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
    categories: HashMap<usize, Vec<NodeIndex>>,
    licensees: HashMap<usize, Vec<NodeIndex>>,
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
    n_categories: usize,
    n_lemmas: usize,
    pad_vector: Tensor<B, 1>, //(n_lemmas)
    end_vector: Tensor<B, 1>, //(n_lemmas)
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

fn clamp_prob(x: f64) -> anyhow::Result<f64> {
    if x > 1.0 {
        eprintln!("Clampling prob {x} to 1.0");
        Ok(1.0)
    } else if x < 0.0 {
        eprintln!("Clampling prob {x} to 0.0");
        Ok(0.0)
    } else if x.is_nan() {
        bail!("Probability is NaN");
    } else {
        Ok(x)
    }
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
        pad_vector: Tensor<B, 1>,
        end_vector: Tensor<B, 1>,
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

        let weights = log_softmax(weights, 0);
        let types = log_softmax(types, 2);
        let pad_vector = log_softmax(pad_vector, 0);
        let end_vector = log_softmax(end_vector, 0);
        let type_categories = log_softmax(type_categories, 2);
        let lemmas = log_softmax(lemmas, 1);
        let licensee_categories = log_softmax(licensee_categories, 2);
        let categories = log_softmax(categories, 1);

        let silent_probability: Tensor<B, 1> =
            lemmas.clone().slice([0..n_lexemes, 2..3]).squeeze(1);
        let non_silent_probability: Tensor<B, 1> = (-silent_probability.clone().exp()).log1p();

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
                Bernoulli::new(clamp_prob(
                    heated_lemmas
                        .clone()
                        .slice([lexeme_idx..lexeme_idx + 1, 0..1])
                        .exp()
                        .into_scalar()
                        .elem(),
                )?)?,
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
                    Bernoulli::new(clamp_prob(
                        included_features
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, position..position + 1])
                            .exp()
                            .into_scalar()
                            .elem(),
                    )?)?,
                );
            }

            for position in 0..n_features {
                included_probs.insert(
                    (lexeme_idx, position + n_licensees),
                    Bernoulli::new(clamp_prob(
                        included_features
                            .clone()
                            .slice([
                                lexeme_idx..lexeme_idx + 1,
                                n_licensees + position..n_licensees + position + 1,
                            ])
                            .exp()
                            .into_scalar()
                            .elem(),
                    )?)?,
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
            n_categories,
            categories,
            licensee_category_distributions,
            category_distributions,
            type_distributions,
            pad_vector,
            end_vector,
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

    pub fn pad_vector(&self) -> &Tensor<B, 1> {
        &self.pad_vector
    }

    pub fn end_vector(&self) -> &Tensor<B, 1> {
        &self.end_vector
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

    //TODO: look into if weights should be done only over categories

    pub fn new_superimposed(
        grammar_params: &GrammarParameterization<B>,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 1>, Self) {
        todo!(); /*
                         let mut licensees_map = HashMap::default();
                         let mut categories_map = HashMap::default();

                         let mut grammar_prob = Tensor::<B, 1>::zeros([1], &grammar_params.device());
                         let mut lexemes = vec![];
                         let mut graph: DiGraph<(Option<(usize, Tensor<B, 1>)>, NeuralFeature), ()> = DiGraph::new();
                         let root = graph.add_node((None, FeatureOrLemma::Root));

                         for lexeme_idx in 0..grammar_params.n_lexemes {
                             let mut parents = vec![];
                             let first_features = (0..grammar_params.n_licensees)
                                 .map(|c| {
                                     (
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
                                         FeatureOrLemma::Feature(Feature::Category(c)),
                                         grammar_params
                                             .categories
                                             .clone()
                                             .slice([lexeme_idx..lexeme_idx + 1, c..c + 1])
                                             .reshape([1])
                                             + grammar_params //all parses go through weights
                                                 .weights
                                                 .slice([lexeme_idx..lexeme_idx + 1])
                                                 .clone(),
                                     )
                                 }));

                             let all_licensees = vec![];
                             let all_categories = vec![];
                             let parent_licensees = vec![];
                             for (feature, prob) in first_features {
                                 let node = graph.add_node((Some((lexeme_idx, prob)), feature));
                                 let feature = &graph[node].1;
                                 match feature {
                                     FeatureOrLemma::Feature(Feature::Category(c)) => {
                                         all_categories.push(node);
                                         categories_map.entry(*c).or_insert(vec![]).push(node);
                                     }
                                     FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                                         parent_licensees.push(node);
                                         all_licensees.push(node);
                                         licensees_map.entry(*c).or_insert(vec![]).push(node);
                                     }
                                     _ => {
                                         panic!("Invalid first feature!")
                                     }
                                 };
                                 graph.add_edge(root, node, ());
                                 parents.push(node);
                             }

                             let mut new_parent_licensees = vec![];
                             for i in 1..grammar_params.n_licensees {
                                 let licensees = (0..grammar_params.n_licensees).map(|c| {
                                     (
                                         FeatureOrLemma::Feature(Feature::Licensee(c)),
                                         grammar_params
                                             .licensee_categories
                                             .clone()
                                             .slice([lexeme_idx..lexeme_idx + 1, 0..1, c..c + 1])
                                             .reshape([1]),
                                     )
                                 });
                                 for (feature, prob) in licensees {
                                     let node = graph.add_node((Some((lexeme_idx, prob)), feature));
                                     all_licensees.push(node);
                                     for parent in parent_licensees {
                                         graph.add_edge(node, parent, ());
                                     }
                                 }
                             }

                             for (category, licensee) in all_categories.into_iter().zip(all_licensees.into_iter()) {
                                 graph.add_edge(licensee, category, ());
                             }

                             for (i, feature, prob) in first_features {
                                 let node = graph.add_node((Some((lexeme_idx, prob)), feature));
                                 let feature = &graph[node].1;
                                 match feature {
                                     FeatureOrLemma::Feature(Feature::Category(c)) => {
                                         categories_map.entry(*c).or_insert(vec![]).push(node);
                                     }
                                     FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                                         licensees_map.entry(*c).or_insert(vec![]).push(node);
                                     }
                                     _ => {
                                         panic!("Invalid first feature!")
                                     }
                                 };
                                 graph.add_edge(root, node, ());
                                 parents.push(node);
                             }

                             for (_position, possibles) in features {
                                 let mut nodes = vec![];

                                 for (feature, prob) in possibles.into_iter() {
                                     nodes.push(graph.add_node((Some((lexeme_idx, prob.clone())), feature.clone())));
                                 }
                                 for (parent, child) in parents.iter().cartesian_product(nodes.iter()) {
                                     if !graph.contains_edge(*parent, *child) {
                                         graph.add_edge(*parent, *child, ());
                                     }
                                 }
                                 parents = nodes;
                             }
                         }

                         //Renormalise probabilities to sum to one.
                         //
                         //

                         let mut weights_map: HashMap<NeuralProbabilityRecord, Tensor<B, 1>> = HashMap::default();
                         let mut ree: HashMap<NodeIndex, usize> = HashMap::default();
                         let graph = graph.map(
                             |nx, (x, feat)| match x {
                                 Some((lexeme_idx, prob)) => {
                                     let mut prob = prob.clone();
                                     if graph.contains_edge(root, nx) {
                                         prob = prob
                                             + grammar_params
                                                 .weights
                                                 .clone()
                                                 .slice([*lexeme_idx..lexeme_idx + 1]);
                                     }
                                     ree.insert(nx, *lexeme_idx);
                                     let record = NeuralProbabilityRecord::Feature(nx);
                                     let log_prob = LogProb::new(prob.clone().into_scalar().elem())
                                         .unwrap_or_else(|_| LogProb::new(0.0).unwrap());
                                     weights_map.insert(record, prob);
                                     (Some((record, log_prob)), feat.clone())
                                 }
                                 None => (None, feat.clone()),
                             },
                             |_, _: &()| (),
                         );

                         (
                             grammar_prob,
                             NeuralLexicon {
                                 graph,
                                 licensees: licensees_map,
                                 categories: categories_map,
                                 weights: weights_map,
                                 root,
                                 device: grammar_params.device(),
                             },
                         )
                 */
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
    ) -> impl Iterator<Item = (Self::Probability, petgraph::prelude::NodeIndex)> + '_ {
        std::iter::once(todo!())
        /*
                self.graph
                    .edges_directed(nx, petgraph::Direction::Outgoing)
                    .map(|e| e.target())
        */
    }

    fn get(&self, nx: petgraph::prelude::NodeIndex) -> Option<&NeuralFeature> {
        if let Some((_, feature)) = self.graph.node_weight(nx) {
            Some(feature)
        } else {
            None
        }
    }

    fn find_licensee(&self, category: &usize) -> anyhow::Result<&[(Self::Probability, NodeIndex)]> {
        todo!();
        //self.licensees
        //    .get(category)
        //    .map(Vec::as_slice)
        //    .with_context(|| format!("{category:?} is not a valid licensee in the lexicon!"))
    }

    fn find_category(&self, category: &usize) -> anyhow::Result<&[(Self::Probability, NodeIndex)]> {
        todo!();
        //self.categories
        //    .get(category)
        //    .map(Vec::as_slice)
        //    .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
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
        let pad_vector = Tensor::<NdArray, 1>::from_floats(
            [10., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            &NdArrayDevice::default(),
        );
        let end_vector = Tensor::<NdArray, 1>::from_floats(
            [0., 10., 0., 0., 0., 0., 0., 0., 0., 0.],
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
            pad_vector,
            end_vector,
            1.0,
        );
        Ok(())
    }
}
