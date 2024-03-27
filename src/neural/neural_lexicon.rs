use super::{utils::*, N_TYPES};
use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use ahash::{HashMap, HashSet};
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
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Gumbel};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NeuralProbabilityRecord {
    Lexeme(NodeIndex, usize),
    Feature(NodeIndex),
    Edge(EdgeIndex),
    EdgeAndFeature((NodeIndex, EdgeIndex)),
    MergeRuleProb,
    MoveRuleProb,
    OneProb,
}

//TODO: Refactor so that we don't go through siblings and that there is no reptition

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NeuralProbability(pub (NeuralProbabilityRecord, LogProb<f64>));

impl NeuralProbability {
    fn mul_by_edge(
        &self,
        edge: EdgeIndex,
        edge_p: LogProb<f64>,
    ) -> anyhow::Result<NeuralProbability> {
        let NeuralProbability((record, p)) = &self;
        match record {
            NeuralProbabilityRecord::Feature(node) => Ok(NeuralProbability((
                NeuralProbabilityRecord::EdgeAndFeature((*node, edge)),
                edge_p + p,
            ))),
            NeuralProbabilityRecord::OneProb => Ok(*self),
            _ => bail!("Invalid edge multiplication!"),
        }
    }
}

pub type NeuralFeature = FeatureOrLemma<usize, usize>;
type NeuralGraph = DiGraph<(Option<LogProb<f64>>, NeuralFeature), LogProb<f64>>;

#[derive(Debug, Clone)]
pub struct NeuralLexicon<B: Backend> {
    weights: HashMap<NeuralProbabilityRecord, Tensor<B, 1>>,
    categories: HashMap<usize, Vec<(NeuralProbability, NodeIndex)>>,
    licensees: HashMap<usize, Vec<(NeuralProbability, NodeIndex)>>,
    graph: NeuralGraph,
    device: B::Device,
}

pub struct GrammarParameterization<B: Backend> {
    types: Tensor<B, 3>,               //(lexeme, lexeme_pos, type_distribution)
    type_categories: Tensor<B, 3>,     //(lexeme, lexeme_pos, categories_position)
    licensee_categories: Tensor<B, 3>, //(lexeme, lexeme_pos, categories_position)
    lemmas: Tensor<B, 2>,              //(lexeme, lemma_distribution)
    categories: Tensor<B, 2>,          //(lexeme, categories)
    included_features: Tensor<B, 3>,   //(lexeme, n_licensee + n_features, true/false)
    weights: Tensor<B, 1>,             //(lexeme)
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
        included_features: Tensor<B, 2>,    // (lexeme, n_licensee + n_features)
        lemmas: Tensor<B, 2>,               // (lexeme, n_lemmas)
        silent_probabilities: Tensor<B, 1>, //(lexeme)
        categories: Tensor<B, 2>,           // (lexeme, n_categories)
        weights: Tensor<B, 1>,              // (lexeme)
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

        let included_features = activation_function(
            Tensor::stack::<3>([included_features.clone(), -included_features].to_vec(), 2),
            2,
            inverse_temperature,
            gumbel,
            rng,
        );

        let types = activation_function(types, 2, inverse_temperature, gumbel, rng);
        let type_categories =
            activation_function(type_categories, 2, inverse_temperature, gumbel, rng);
        let lemmas = activation_function(lemmas, 1, inverse_temperature, gumbel, rng);
        let weights = log_softmax(weights, 0);
        let licensee_categories =
            activation_function(licensee_categories, 2, inverse_temperature, gumbel, rng);
        let categories = activation_function(categories, 1, inverse_temperature, gumbel, rng);

        let pad_vector = log_softmax(pad_vector, 0);
        let end_vector = log_softmax(end_vector, 0);

        let silent_probabilities = gumbel_activation(
            Tensor::stack::<2>(
                [silent_probabilities.clone(), -silent_probabilities].to_vec(),
                1,
            ),
            1,
            inverse_temperature,
            rng,
        );

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
            weights,
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

    pub fn device(&self) -> Device<B> {
        self.types.device()
    }

    pub fn n_lemmas(&self) -> usize {
        self.n_lemmas
    }
    pub fn lemmas(&self) -> &Tensor<B, 2> {
        &self.lemmas
    }

    fn prob_of_n_licensees(&self, lexeme_idx: usize, n: usize) -> Tensor<B, 1> {
        match n {
            0 => self
                .included_features
                .clone()
                .slice([lexeme_idx..lexeme_idx + 1, 0..1, 1..2])
                .reshape([1]),
            _ => self
                .included_features
                .clone()
                .slice([lexeme_idx..lexeme_idx + 1, (n - 1)..n, 0..1])
                .reshape([1]),
        }
    }
    fn prob_of_not_n_licensees(&self, lexeme_idx: usize, n: usize) -> Tensor<B, 1> {
        self.included_features
            .clone()
            .slice([lexeme_idx..lexeme_idx + 1, n..n + 1, 1..2])
            .reshape([1])
    }

    fn prob_of_not_n_features(&self, lexeme_idx: usize, n: usize) -> Tensor<B, 1> {
        self.included_features
            .clone()
            .slice([
                lexeme_idx..lexeme_idx + 1,
                self.n_licensees + n..self.n_licensees + n + 1,
                1..2,
            ])
            .reshape([1])
    }

    fn prob_of_n_features(&self, lexeme_idx: usize, n: usize) -> Tensor<B, 1> {
        match n {
            0 => self
                .included_features
                .clone()
                .slice([
                    lexeme_idx..lexeme_idx + 1,
                    self.n_licensees..self.n_licensees + 1,
                    1..2,
                ])
                .reshape([1]),
            _ => self
                .included_features
                .clone()
                .slice([
                    lexeme_idx..lexeme_idx + 1,
                    self.n_licensees + n - 1..self.n_licensees + n,
                    0..1,
                ])
                .reshape([1]),
        }
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
    ) -> anyhow::Result<(Self, HashMap<EdgeIndex, Vec<EdgeIndex>>)> {
        let mut licensees_map = HashMap::default();
        let mut categories_map = HashMap::default();
        let mut weights_map = HashMap::default();

        let mut graph: NeuralGraph = DiGraph::new();

        for lexeme_idx in 0..grammar_params.n_lexemes {
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

            first_features.shuffle(rng);

            let mut all_categories = vec![];
            let mut parent_licensees = vec![];
            for (n_licensees, feature, prob) in first_features {
                let p: LogProb<f64> = tensor_to_log_prob(&prob).context("First layer")?;
                let node = graph.add_node((Some(p), feature));
                let prob_of_n_licensees =
                    grammar_params.prob_of_n_licensees(lexeme_idx, n_licensees);

                let lexeme_p = prob + prob_of_n_licensees;
                let p = NeuralProbability((
                    NeuralProbabilityRecord::Lexeme(node, lexeme_idx),
                    tensor_to_log_prob(&lexeme_p).context("First n_licensees")?,
                ));
                let lexeme_weight = grammar_params
                    .weights
                    .clone()
                    .slice([lexeme_idx..lexeme_idx + 1]);
                weights_map.insert(
                    NeuralProbabilityRecord::Lexeme(node, lexeme_idx),
                    lexeme_weight,
                );
                weights_map.insert(NeuralProbabilityRecord::Feature(node), lexeme_p);

                let feature = &graph[node].1;
                match feature {
                    FeatureOrLemma::Feature(Feature::Category(c)) => {
                        all_categories.push(node);
                        categories_map.entry(*c).or_insert(vec![]).push((p, node));
                    }
                    FeatureOrLemma::Feature(Feature::Licensee(c)) => {
                        parent_licensees.push(node);
                        licensees_map.entry(*c).or_insert(vec![]).push((p, node));
                    }
                    _ => {
                        panic!("Invalid first feature!")
                    }
                };
            }

            for (licensee, category) in parent_licensees.iter().zip(all_categories.iter()) {
                if grammar_params.n_licensees == 1 {
                    let e_prob = Tensor::zeros([1], &grammar_params.device());
                    let e = graph.add_edge(
                        *licensee,
                        *category,
                        tensor_to_log_prob(&e_prob).context("Licensee to cat")?,
                    );
                    weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                } else {
                    let e_prob = grammar_params.prob_of_not_n_licensees(lexeme_idx, 1);
                    let e = graph.add_edge(
                        *licensee,
                        *category,
                        tensor_to_log_prob(&e_prob).context("Licensee to cat")?,
                    );
                    weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                }
            }

            for i in 1..grammar_params.n_licensees {
                let mut new_parent_licensees = vec![];
                let mut licensees = (0..grammar_params.n_categories)
                    .map(|c| {
                        (
                            FeatureOrLemma::Feature(Feature::Licensee(c)),
                            grammar_params
                                .licensee_categories
                                .clone()
                                .slice([lexeme_idx..lexeme_idx + 1, i..i + 1, c..c + 1])
                                .reshape([1]),
                        )
                    })
                    .collect::<Vec<_>>();
                licensees.shuffle(rng);
                for (feature, prob) in licensees.into_iter() {
                    let node = graph.add_node((
                        Some(tensor_to_log_prob(&prob).context("Licensee")?),
                        feature,
                    ));
                    for parent in parent_licensees.iter() {
                        let e_prob = grammar_params.prob_of_n_licensees(lexeme_idx, i + 1);
                        let e = graph.add_edge(
                            *parent,
                            node,
                            tensor_to_log_prob(&e_prob).context("Licensee to licensee")?,
                        );
                        weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                    }
                    for category in all_categories.iter() {
                        if i == grammar_params.n_licensees - 1 {
                            let e_prob = Tensor::zeros([1], &grammar_params.device());
                            let e = graph.add_edge(
                                node,
                                *category,
                                tensor_to_log_prob(&e_prob).context("Zero Licensee to cat")?,
                            );
                            weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                        } else {
                            let e_prob = grammar_params.prob_of_not_n_licensees(lexeme_idx, i + 1);
                            let e = graph.add_edge(
                                node,
                                *category,
                                tensor_to_log_prob(&e_prob).context("Licensee to cat")?,
                            );
                            weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                        }
                    }
                    weights_map.insert(NeuralProbabilityRecord::Feature(node), prob);
                    new_parent_licensees.push(node);
                }
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

            let lemma = NeuralFeature::Lemma(Some(lexeme_idx));
            let silent_lemma = NeuralFeature::Lemma(None);
            let lemma = graph.add_node((
                Some(tensor_to_log_prob(&nonsilent_p).context("lemma")?),
                lemma,
            ));
            let silent_lemma = graph.add_node((
                Some(tensor_to_log_prob(&silent_p).context("silent lemma")?),
                silent_lemma,
            ));

            for category in all_categories.iter() {
                let e_prob = grammar_params.prob_of_not_n_features(lexeme_idx, 0);
                let e = graph.add_edge(
                    *category,
                    lemma,
                    tensor_to_log_prob(&e_prob).context("cat to lemma")?,
                );
                weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob.clone());

                let e = graph.add_edge(
                    *category,
                    silent_lemma,
                    tensor_to_log_prob(&e_prob).context("cat to silent lemma")?,
                );
                weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
            }
            weights_map.insert(NeuralProbabilityRecord::Feature(lemma), nonsilent_p);
            weights_map.insert(NeuralProbabilityRecord::Feature(silent_lemma), silent_p);

            let lemmas = [lemma, silent_lemma];

            let mut parents = all_categories;
            for i in 0..grammar_params.n_features {
                let mut new_parents: Vec<_> = (0..grammar_params.n_categories)
                    .cartesian_product(0..N_TYPES)
                    .collect();
                new_parents.shuffle(rng);

                let new_parents = new_parents
                    .into_iter()
                    .map(|(c, t)| {
                        let feature = to_feature(t, c);
                        let p = (grammar_params.type_categories.clone().slice([
                            lexeme_idx..lexeme_idx + 1,
                            i..i + 1,
                            c..c + 1,
                        ]) + grammar_params.types.clone().slice([
                            lexeme_idx..lexeme_idx + 1,
                            i..i + 1,
                            t..t + 1,
                        ]))
                        .reshape([1]);
                        let node = graph
                            .add_node((Some(tensor_to_log_prob(&p).context("feature")?), feature));
                        weights_map.insert(NeuralProbabilityRecord::Feature(node), p);
                        Ok(node)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                for (node, parent) in new_parents.iter().cartesian_product(parents.iter()) {
                    let e_prob = grammar_params.prob_of_n_features(lexeme_idx, i + 1);
                    let e = graph.add_edge(
                        *parent,
                        *node,
                        tensor_to_log_prob(&e_prob).context("feat to feat")?,
                    );
                    weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                }
                for (node, lemma) in new_parents.iter().cartesian_product(lemmas.iter()) {
                    if i == grammar_params.n_features - 1 {
                        let e_prob = Tensor::zeros([1], &grammar_params.device());
                        let e = graph.add_edge(
                            *node,
                            *lemma,
                            tensor_to_log_prob(&e_prob).context("feat to lemma")?,
                        );
                        weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                    } else {
                        let e_prob = grammar_params.prob_of_not_n_features(lexeme_idx, i + 1);
                        let e = graph.add_edge(
                            *node,
                            *lemma,
                            tensor_to_log_prob(&e_prob).context("feat to lemma")?,
                        );
                        weights_map.insert(NeuralProbabilityRecord::Edge(e), e_prob);
                    }
                }

                parents = new_parents;
            }
        }

        let mut alternative_map = HashMap::default();
        for node in graph.node_indices() {
            let alternatives: Vec<_> = graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .map(|e| e.id())
                .collect();
            for a in alternatives.iter() {
                alternative_map.insert(
                    *a,
                    alternatives
                        .iter()
                        .filter_map(|x| if x == a { None } else { Some(*x) })
                        .collect(),
                );
            }
        }

        /*
                let g = graph.clone();

                let g = g.map(
                    |_i, (p, f)| format!("{} {:2}", f, p.unwrap_or(LogProb::new(0.0).unwrap())),
                    |e_i, e| e,
                );
                println!("{}", petgraph::dot::Dot::new(&g));
        */

        Ok((
            NeuralLexicon {
                graph,
                licensees: licensees_map,
                categories: categories_map,
                weights: weights_map,
                device: grammar_params.device(),
            },
            alternative_map,
        ))
    }
}

impl<B: Backend> Lexiconable<usize, usize> for NeuralLexicon<B> {
    type Probability = NeuralProbability;

    fn probability_of_one(&self) -> Self::Probability {
        NeuralProbability((NeuralProbabilityRecord::OneProb, LogProb::new(0.0).unwrap()))
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
                let target = e.target();
                let p = NeuralProbability((
                    NeuralProbabilityRecord::Feature(target),
                    self.graph[target].0.unwrap(),
                ))
                .mul_by_edge(e.id(), *e.weight())
                .unwrap();
                (p, target)
            })
    }

    fn get(&self, nx: petgraph::prelude::NodeIndex) -> Option<&NeuralFeature> {
        if let Some((_, feature)) = self.graph.node_weight(nx) {
            Some(feature)
        } else {
            None
        }
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
        let silent_probabilities = Tensor::<NdArray, 1>::random(
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(32);

        let _g = GrammarParameterization::new(
            types,
            type_categories,
            licensee_categories,
            included_features,
            lemmas,
            silent_probabilities,
            categories,
            weights,
            pad_vector,
            end_vector,
            1.0,
            true,
            &mut rng,
        );
        Ok(())
    }
}
