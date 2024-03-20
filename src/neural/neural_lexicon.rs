use std::ops::Range;

use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::{HashMap, HashSet};
use anyhow::{bail, Context};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{activation::log_softmax, backend::Backend, Device, ElementConversion, Tensor};
use burn::tensor::{Bool, Data, Int};
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

        let types = log_softmax(types, 2);
        let pad_vector = log_softmax(pad_vector, 0);
        let end_vector = log_softmax(end_vector, 0);
        let type_categories = log_softmax(type_categories, 2);
        let lemmas = log_softmax(lemmas, 1);
        let licensee_categories = log_softmax(licensee_categories, 1);
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

    pub fn new_superimposed(
        grammar_params: &GrammarParameterization<B>,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 1>, Self) {
        type PositionLog<B> = Vec<(usize, Option<usize>, Tensor<B, 1>)>;
        let mut graph: DiGraph<(Option<PositionLog<B>>, NeuralFeature), _> = DiGraph::new();
        let mut licensees_map = HashMap::default();
        let mut categories_map = HashMap::default();

        let root = graph.add_node((None, FeatureOrLemma::Root));
        let mut grammar_prob = Tensor::<B, 1>::zeros([1], &grammar_params.device());
        let mut lexemes = vec![];
        for lexeme_idx in 0..grammar_params.n_lexemes {
            let mut unattested_licensees = vec![];
            let licensees: Vec<_> = if lexeme_idx > 0 {
                (0..grammar_params.n_licensees)
                    .filter_map(|i| match grammar_params.is_licensee(lexeme_idx, i, rng) {
                        true => Some((
                            Some(i),
                            (0..grammar_params.n_categories)
                                .map(|x| {
                                    (
                                        FeatureOrLemma::Feature(Feature::Licensee(x)),
                                        grammar_params
                                            .licensee_categories
                                            .clone()
                                            .slice([lexeme_idx..lexeme_idx + 1, i..i + 1, x..x + 1])
                                            .reshape([1]),
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )),
                        false => {
                            unattested_licensees.push(i);
                            None
                        }
                    })
                    .collect()
            } else {
                unattested_licensees.extend(0..grammar_params.n_licensees);
                //Ensure there is always a lexeme of category 0 without licensees
                vec![]
            };

            for (x, _) in licensees.iter() {
                let x = x.unwrap();
                grammar_prob = grammar_prob
                    + grammar_params
                        .included_features
                        .clone()
                        .slice([lexeme_idx..lexeme_idx + 1, x..x + 1])
                        .reshape([1]);
            }
            for i in unattested_licensees.iter() {
                grammar_prob = grammar_prob
                    + (-(grammar_params
                        .included_features
                        .clone()
                        .slice([lexeme_idx..lexeme_idx + 1, *i..i + 1])
                        .reshape([1])
                        .exp()))
                    .log1p();
            }

            let categories = (
                None,
                (0..grammar_params.n_categories)
                    .map(|c| {
                        (
                            FeatureOrLemma::Feature(Feature::Category(c)),
                            grammar_params
                                .categories
                                .clone()
                                .slice([lexeme_idx..lexeme_idx + 1, c..c + 1])
                                .reshape([1]),
                        )
                    })
                    .collect::<Vec<_>>(),
            );

            let mut unattested_features = vec![];
            let features: Vec<_> = (0..grammar_params.n_features)
                .filter_map(|i| match grammar_params.is_feature(lexeme_idx, i, rng) {
                    true => Some((
                        Some(i),
                        (0..grammar_params.n_categories)
                            .cartesian_product(0..N_TYPES)
                            .map(|(c, t)| {
                                (
                                    to_feature(t, c),
                                    (grammar_params.type_categories.clone().slice([
                                        lexeme_idx..lexeme_idx + 1,
                                        i..i + 1,
                                        c..c + 1,
                                    ]) + grammar_params.types.clone().slice([
                                        lexeme_idx..lexeme_idx + 1,
                                        i..i + 1,
                                        t..t + 1,
                                    ]))
                                    .reshape([1]),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )),
                    false => {
                        unattested_features.push(i);
                        None
                    }
                })
                .collect();

            for (x, _) in features.iter() {
                let x = x.unwrap();
                grammar_prob = grammar_prob
                    + grammar_params
                        .included_features
                        .clone()
                        .slice([
                            lexeme_idx..lexeme_idx + 1,
                            x + grammar_params.n_licensees..x + grammar_params.n_licensees + 1,
                        ])
                        .reshape([1]);
            }

            for i in unattested_features.iter() {
                grammar_prob = grammar_prob
                    + (-(grammar_params
                        .included_features
                        .clone()
                        .slice([
                            lexeme_idx..lexeme_idx + 1,
                            i + grammar_params.n_licensees..i + grammar_params.n_licensees + 1,
                        ])
                        .reshape([1])
                        .exp()))
                    .log1p();
            }

            let lemma = (
                None,
                vec![
                    (
                        FeatureOrLemma::Lemma(None),
                        grammar_params
                            .silent_probabilities
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, 0..1])
                            .reshape([1]),
                    ),
                    (
                        FeatureOrLemma::Lemma(Some(lexeme_idx)),
                        grammar_params
                            .silent_probabilities
                            .clone()
                            .slice([lexeme_idx..lexeme_idx + 1, 1..2])
                            .reshape([1]),
                    ),
                ],
            );

            let mut lexeme = licensees;
            lexeme.push(categories);
            lexeme.extend(features);
            lexeme.push(lemma);
            lexemes.push(lexeme);
        }

        for (lexeme_idx, features) in lexemes.into_iter().enumerate() {
            let mut features = features.into_iter();
            let mut parents = vec![];
            let (position, first_features) = features.next().unwrap();
            for (i, (feature, prob)) in first_features.into_iter().enumerate() {
                let node = graph.add_node((Some(vec![(lexeme_idx, position, prob)]), feature));
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
            }

            for (position, possibles) in features {
                let mut nodes = vec![];

                for (feature, prob) in possibles.into_iter() {
                    nodes.push(graph.add_node((
                        Some(vec![(lexeme_idx, position, prob.clone())]),
                        feature.clone(),
                    )));
                }
                for (parent, child) in parents.iter().cartesian_product(nodes.iter()) {
                    if !graph.contains_edge(*parent, *child) {
                        graph.add_edge(*parent, *child, ());
                    }
                }
                parents = nodes;
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
                    let mut probs = vec![];
                    let mut impossible_lexemes: Vec<_> = std::iter::repeat(true)
                        .take(grammar_params.n_lexemes)
                        .collect();
                    let mut attested_lexemes = vec![];

                    for n in neighbours.iter() {
                        let (log, _feat) = &graph[*n];
                        let log: &PositionLog<B> = log.as_ref().unwrap();
                        let mut attested: Vec<u32> = vec![];
                        let prob_values: Vec<Tensor<B, 1>> = log
                            .iter()
                            .map(|(i, _feature_pos, p)| (i, p))
                            .fold(vec![], |mut acc, c| {
                                acc.push(c.1.clone());
                                attested.push(*c.0 as u32);
                                impossible_lexemes[*c.0] = false;
                                acc
                            });
                        attested_lexemes.push(attested);
                        probs.push(Tensor::cat(prob_values, 0));
                    }

                    let n_attested: usize = impossible_lexemes.iter().map(|x| !x as usize).sum();

                    let mut final_weights = vec![];
                    if n_attested == 1 {
                        final_weights = probs;
                    } else {
                        let mask = Tensor::<B, 1, Bool>::from_data(
                            Data::from(impossible_lexemes.as_slice()),
                            &grammar_params.device(),
                        );

                        let weights = log_softmax(
                            {
                                grammar_params.weights.clone()
                                    + Tensor::zeros_like(&grammar_params.weights)
                                        .mask_fill(mask, -20.0)
                            },
                            0,
                        );

                        for (p, a) in probs.iter().zip(&attested_lexemes) {
                            let idx = Tensor::<B, 1, Int>::from_data(
                                Data::from(a.as_slice()).convert(),
                                &grammar_params.device(),
                            );
                            let p = p.clone();
                            let weights = weights.clone().select(0, idx);
                            let z = if a.len() == 1 {
                                p + weights
                            } else {
                                let p_of_none = (-((p + weights).exp())).log1p().sum();
                                (-(p_of_none.exp())).log1p()
                            };
                            final_weights.push(z);
                        }
                    }

                    for (n, w) in neighbours.iter().zip(final_weights.into_iter()) {
                        let record = NeuralProbabilityRecord::Feature(*n);
                        let log_prob = LogProb::new(w.clone().into_scalar().elem())
                            .unwrap_or_else(|_| LogProb::new(0.0).unwrap());
                        node_map.insert(*n, Some((record, log_prob)));
                        weights_map.insert(record, w);
                    }
                }
            }
        }
        /*
        let graph2 = graph.clone();
        let graph2 = graph2.map(
            |n_i, (_, n)| {
                format!(
                    "{n} {:.2}",
                    match node_map.get(&n_i).unwrap() {
                        Some((_, p)) => p.into_inner(),
                        None => 0.0,
                    }
                )
            },
            |_, _| "",
        );*/

        let graph = graph.map(
            |n, (_log, feat)| (node_map.remove(&n).unwrap(), feat.clone()),
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
    }

    pub fn new_random(
        grammar_params: &GrammarParameterization<B>,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 1>, Vec<Vec<NeuralFeature>>, Self) {
        type PositionLog = Vec<(usize, Option<usize>)>;
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

            let mut lexeme: Vec<(NeuralFeature, Option<usize>)> = licensees
                .into_iter()
                .map(|(pos, cat)| (NeuralFeature::Feature(Feature::Licensee(cat)), Some(pos)))
                .collect();
            lexeme.push((NeuralFeature::Feature(Feature::Category(category)), None));
            lexeme.extend(
                features
                    .into_iter()
                    .map(|(pos, cat, feature_type)| (to_feature(feature_type, cat), Some(pos))),
            );
            lexeme.push((
                match is_silent {
                    true => NeuralFeature::Lemma(None),
                    false => NeuralFeature::Lemma(Some(lexeme_idx)),
                },
                None,
            ));
            lexemes.push(lexeme);
        }

        for (lexeme_idx, lexeme) in lexemes.iter().enumerate() {
            let mut parent = root;
            for (feature, position) in lexeme {
                let mut node = None;
                for child in graph
                    .neighbors_directed(parent, petgraph::Direction::Outgoing)
                    .collect_vec()
                    .into_iter()
                {
                    if let (Some(log), child_feature) = graph.node_weight_mut(child).unwrap() {
                        if child_feature == feature {
                            node = Some(child);
                            log.push((lexeme_idx, *position));
                        }
                    }
                }

                parent = node.unwrap_or_else(|| {
                    let node =
                        graph.add_node((Some(vec![(lexeme_idx, *position)]), feature.clone()));
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
                    let mut p: Tensor<B, 1> =
                        Tensor::zeros([neighbours.len()], &grammar_params.device());
                    for (n_i, n) in neighbours.iter().enumerate() {
                        let (log, feat) = &graph[*n];
                        let log: &PositionLog = log.as_ref().unwrap();
                        let (weight_values, p_values) = log
                            .iter()
                            .map(|(i, feature_pos)| {
                                let v = grammar_params.weights.clone().slice([*i..i + 1]);
                                let mut p: Tensor<B, 1> =
                                    Tensor::zeros([1], &grammar_params.device());
                                match feat {
                                    FeatureOrLemma::Root => (),
                                    FeatureOrLemma::Lemma(lemma) => {
                                        p = grammar_params
                                            .silent_probabilities
                                            .clone()
                                            .slice([
                                                *i..i + 1,
                                                match lemma.is_none() {
                                                    true => 0..1,
                                                    false => 1..2,
                                                },
                                            ])
                                            .squeeze::<1>(1);
                                    }
                                    FeatureOrLemma::Feature(feat) => {
                                        match feat {
                                            Feature::Category(c) => {
                                                p = grammar_params
                                                    .categories
                                                    .clone()
                                                    .slice([*i..i + 1, *c..c + 1])
                                                    .squeeze(1);
                                            }
                                            Feature::Licensee(c) => {
                                                let feature_pos = feature_pos.unwrap();
                                                p = grammar_params
                                                    .licensee_categories
                                                    .clone()
                                                    .slice([
                                                        *i..i + 1,
                                                        feature_pos..feature_pos + 1,
                                                        *c..c + 1,
                                                    ])
                                                    .reshape([1])
                                                    + grammar_params
                                                        .included_features
                                                        .clone()
                                                        .slice([
                                                            *i..i + 1,
                                                            feature_pos..feature_pos + 1,
                                                        ])
                                                        .reshape([1]);
                                            }
                                            _ => {
                                                let (type_of_feat, category) = from_feature(feat);
                                                let feature_pos = feature_pos.unwrap();
                                                p = grammar_params
                                                    .type_categories
                                                    .clone()
                                                    .slice([
                                                        *i..i + 1,
                                                        feature_pos..feature_pos + 1,
                                                        category..category + 1,
                                                    ])
                                                    .reshape([1])
                                                    + grammar_params
                                                        .types
                                                        .clone()
                                                        .slice([
                                                            *i..*i + 1,
                                                            feature_pos..feature_pos + 1,
                                                            type_of_feat..type_of_feat + 1,
                                                        ])
                                                        .reshape([1])
                                                    + grammar_params
                                                        .included_features
                                                        .clone()
                                                        .slice([
                                                            *i..i + 1,
                                                            feature_pos + grammar_params.n_licensees
                                                                ..feature_pos
                                                                    + grammar_params.n_licensees
                                                                    + 1,
                                                        ])
                                                        .reshape([1]);
                                            }
                                        };
                                    }
                                };
                                (v, p)
                            })
                            .fold(
                                (
                                    Tensor::zeros([1], &grammar_params.device()),
                                    Tensor::zeros([1], &grammar_params.device()),
                                ),
                                |(v, p), (v_acc, p_acc)| ((v_acc + v), (p_acc + p)),
                            );
                        weights = weights.slice_assign([n_i..n_i + 1], weight_values);
                        p = p.slice_assign([n_i..n_i + 1], p_values);
                    }
                    weights = log_softmax(weights, 0);
                    for (n_i, n) in neighbours.iter().enumerate() {
                        let record = NeuralProbabilityRecord::Feature(*n);
                        let weight = weights.clone().slice([n_i..n_i + 1]);
                        let log_prob = LogProb::new(weight.clone().into_scalar().elem())
                            .unwrap_or_else(|_| LogProb::new(0.0).unwrap());
                        node_map.insert(*n, Some((record, log_prob)));
                        let p = p.clone().slice([n_i..n_i + 1]);
                        weights_map.insert(record, weight + p.clone() - p.detach());
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
            lexemes
                .into_iter()
                .map(|v| v.into_iter().map(|(a, _b)| a).collect())
                .collect(),
            NeuralLexicon {
                graph,
                licensees: HashMap::default(),
                categories: HashMap::default(),
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

    fn find_licensee(&self, category: &usize) -> anyhow::Result<&[petgraph::prelude::NodeIndex]> {
        self.licensees
            .get(category)
            .map(Vec::as_slice)
            .with_context(|| format!("{category:?} is not a valid licensee in the lexicon!"))
    }

    fn find_category(&self, category: &usize) -> anyhow::Result<&[petgraph::prelude::NodeIndex]> {
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
