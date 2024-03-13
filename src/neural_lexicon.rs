use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use anyhow::Context;
use bitvec::array::BitArray;
use burn::tensor::{
    activation::log_softmax, backend::Backend, Bool, Data, ElementConversion, Tensor,
};
use itertools::Itertools;
use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef};
use rand::{seq::SliceRandom, Rng};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NeuralProbabilityRecord {
    Feature {
        lexemes: BitArray<u64>,
        position: usize,
    },
    MergeRuleProb,
    MoveRuleProb,
    OneProb,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TensorPosition(usize);

const LEMMA_POS: TensorPosition = TensorPosition(0);
const CATEGORY_POS: TensorPosition = TensorPosition(1);
const LICENSOR_POS: TensorPosition = TensorPosition(2);
const LICENSEE_POS: TensorPosition = TensorPosition(3);
const LEFT_SELECTOR_POS: TensorPosition = TensorPosition(4);
const RIGHT_SELECTOR_POS: TensorPosition = TensorPosition(5);

const POSITIONS: [TensorPosition; 6] = [
    LEMMA_POS,
    CATEGORY_POS,
    LICENSOR_POS,
    LICENSEE_POS,
    LEFT_SELECTOR_POS,
    RIGHT_SELECTOR_POS,
];
pub const N_TYPES: usize = 6;

type NeuralFeature = FeatureOrLemma<(usize, usize), usize>;
type NeuralGraph<B> = DiGraph<
    (
        Option<(NeuralProbabilityRecord, Tensor<B, 1>)>,
        NeuralFeature,
    ),
    (),
>;

#[derive(Debug)]
pub struct NeuralLexicon<B: Backend> {
    lemmas: Tensor<B, 3>, //(lexeme, lexeme_pos, lemma_distribution)
    graph: NeuralGraph<B>,
    root: NodeIndex,
    device: B::Device,
}

fn to_feature(pos: TensorPosition, category: usize) -> NeuralFeature {
    match pos {
        CATEGORY_POS => FeatureOrLemma::Feature(Feature::Category(category)),
        LICENSEE_POS => FeatureOrLemma::Feature(Feature::Licensee(category)),
        LICENSOR_POS => FeatureOrLemma::Feature(Feature::Licensor(category)),
        LEFT_SELECTOR_POS => {
            FeatureOrLemma::Feature(Feature::Selector(category, crate::Direction::Left))
        }
        RIGHT_SELECTOR_POS => {
            FeatureOrLemma::Feature(Feature::Selector(category, crate::Direction::Right))
        }
        _ => panic!("Invalid position!"),
    }
}

impl<B: Backend> NeuralLexicon<B> {
    pub fn lemma_at_position(&self, lexeme: usize, pos: usize) -> Tensor<B, 1> {
        let n_lemmas = self.lemmas.shape().dims[2];
        self.lemmas
            .clone()
            .slice([lexeme..lexeme + 1, pos..pos + 1, 0..n_lemmas])
            .reshape([-1])
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn new_random(
        types: &Tensor<B, 3>, //(lexeme, lexeme_pos, type_distribution)
        type_sampling: &Tensor<B, 3>,
        categories: &Tensor<B, 3>, //(lexeme, lexeme_pos, categories_position)
        categories_sampling: &Tensor<B, 3>,
        lemmas: &Tensor<B, 3>,  //(lexeme, lexeme_pos, lemma_distribution)
        weights: &Tensor<B, 2>, //(lexeme, lexeme_weight)
        rng: &mut impl Rng,
    ) -> (Tensor<B, 1>, Self) {
        let [n_lexemes, n_positions, n_categories] = categories.shape().dims;
        let category_ids = (0..n_categories).collect_vec();

        type PositionLog = Vec<(usize, usize)>;
        let mut graph: DiGraph<(Option<PositionLog>, NeuralFeature), _> = DiGraph::new();

        let root = graph.add_node((None, FeatureOrLemma::Root));
        let device = lemmas.device();
        let mut grammar_prob = Tensor::<B, 1>::zeros([1], &types.device());

        let mut has_start_category = false;
        let mut attested_at_position: Vec<_> = std::iter::repeat(true)
            .take(n_lexemes * n_positions)
            .collect_vec();

        for lexeme in 0..n_lexemes {
            let mut parent = root;
            let mut has_category_already = false;
            let mut has_lemma_already = false;

            for position in 0..n_positions {
                if has_lemma_already {
                    break;
                };
                attested_at_position[n_positions * lexeme + position] = false;

                let possible_type = if !has_lemma_already && position == n_positions - 1 {
                    //Make sure we have a lemma if we don't already and its the last stop.
                    LEMMA_POS
                } else if !has_start_category && position == 0 && lexeme == n_lexemes - 1 {
                    CATEGORY_POS
                } else {
                    //Sample a type
                    let type_distribution: Vec<f64> = type_sampling
                        .clone()
                        .slice([lexeme..lexeme + 1, position..position + 1, 0..N_TYPES])
                        .exp()
                        .to_data()
                        .convert::<f64>()
                        .value;

                    let samples: Vec<_> = POSITIONS
                        .into_iter()
                        .filter(|&x| {
                            has_category_already ^ (x == CATEGORY_POS || x == LICENSEE_POS)
                        })
                        .collect();

                    *samples
                        .choose_weighted(rng, |i| {
                            let x: f64 = type_distribution[i.0];
                            x
                        })
                        .unwrap()
                };

                has_category_already = possible_type == CATEGORY_POS || has_category_already;
                has_lemma_already = possible_type == LEMMA_POS || has_lemma_already;

                let mut lemma_structure_p = types
                    .clone()
                    .slice([
                        lexeme..lexeme + 1,
                        position..position + 1,
                        possible_type.0..possible_type.0 + 1,
                    ])
                    .reshape([1]);

                let feature: FeatureOrLemma<(usize, usize), usize> = match possible_type {
                    LEMMA_POS => {
                        let p_of_unpronounced_lemma = lemmas
                            .clone()
                            .slice([lexeme..lexeme + 1, position..position + 1, 0..1])
                            .reshape([1]);

                        let p: f64 = p_of_unpronounced_lemma.clone().into_scalar().elem();
                        if rng.gen_bool(p.exp()) {
                            lemma_structure_p = lemma_structure_p + p_of_unpronounced_lemma;
                            FeatureOrLemma::Lemma(None)
                        } else {
                            lemma_structure_p = lemma_structure_p
                                + (-p_of_unpronounced_lemma.clone().exp()).log1p();
                            FeatureOrLemma::Lemma(Some((lexeme, position)))
                        }
                    }

                    _ => {
                        let cat_weights = categories_sampling
                            .clone()
                            .slice([lexeme..lexeme + 1, position..position + 1, 0..n_categories])
                            .exp()
                            .to_data()
                            .convert::<f64>()
                            .value;
                        let mut cat = *category_ids
                            .choose_weighted(rng, |i| cat_weights[*i])
                            .unwrap();

                        if position == 0 && lexeme == n_lexemes - 1 && !has_start_category {
                            //If we're the last lexeme and there's been no start category, we force
                            //the last lexeme to start with a category.
                            cat = 0
                        }
                        has_start_category = has_start_category
                            || (possible_type == CATEGORY_POS && cat == 0 && position == 0);

                        lemma_structure_p = lemma_structure_p
                            + categories
                                .clone()
                                .slice([lexeme..lexeme + 1, position..position + 1, cat..cat + 1])
                                .reshape([1]);
                        to_feature(possible_type, cat)
                    }
                };

                //Check if the parent has a child with the same feature.
                //If so, merge current and previous.
                let mut node = None;

                for sibling in graph
                    .neighbors_directed(parent, petgraph::Direction::Outgoing)
                    .collect_vec()
                    .into_iter()
                {
                    if let (Some(lexemes), sibling_feature) = &mut graph[sibling] {
                        if feature == *sibling_feature {
                            lexemes.push((lexeme, position));
                            node = Some(sibling);
                            break;
                        }
                    }
                }
                let node =
                    node.unwrap_or(graph.add_node((Some(vec![(lexeme, position)]), feature)));

                graph.add_edge(parent, node, ());
                parent = node;

                grammar_prob = grammar_prob + lemma_structure_p;
            }
        }

        let attested_mask = Tensor::<B, 1, Bool>::from_data(
            Data::from(attested_at_position.as_slice()),
            &types.device(),
        )
        .reshape([n_lexemes, n_positions]);

        let weights = log_softmax(
            Tensor::ones_like(weights).mask_fill(attested_mask, -999) + weights.clone(),
            0,
        );

        //Get actual weights that have been log normalised.
        let graph = graph.map(
            |_, (idx, feat)| match idx {
                Some(positions) => {
                    let (lex, pos) = positions.first().unwrap();
                    let mut tensor = weights
                        .clone()
                        .slice([*lex..lex + 1, *pos..pos + 1])
                        .reshape([1]);
                    let mut included_lexemes = BitArray::ZERO;
                    included_lexemes.set(*lex, true);
                    for (lex, pos) in positions.iter().skip(1) {
                        included_lexemes.set(*lex, true);
                        //TODO: Switch to log sum exp
                        tensor = tensor
                            + weights
                                .clone()
                                .slice([*lex..*lex + 1, *pos..*pos + 1])
                                .reshape([1]);
                    }

                    let neural_feature = NeuralProbabilityRecord::Feature {
                        lexemes: included_lexemes,
                        position: *pos,
                    };
                    (Some((neural_feature, tensor)), feat.clone())
                }
                None => (None, feat.clone()),
            },
            |_, _| (),
        );
        (
            grammar_prob,
            NeuralLexicon {
                lemmas: lemmas.clone(),
                graph,
                root,
                device,
            },
        )
    }
}

impl<B: Backend> Lexiconable<(usize, usize), usize> for NeuralLexicon<B> {
    type Probability = (NeuralProbabilityRecord, Tensor<B, 1>);

    fn probability_of_one(&self) -> Self::Probability {
        (
            NeuralProbabilityRecord::OneProb,
            Tensor::<B, 1>::zeros([1], &self.device),
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
    ) -> impl Iterator<Item = petgraph::prelude::NodeIndex> + '_ {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .map(|e| e.target())
    }

    fn get(
        &self,
        nx: petgraph::prelude::NodeIndex,
    ) -> Option<(
        &crate::lexicon::FeatureOrLemma<(usize, usize), usize>,
        Self::Probability,
    )> {
        if let Some((Some((prob_record, p_tensor)), feature)) = self.graph.node_weight(nx) {
            Some((feature, (*prob_record, p_tensor.clone())))
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
