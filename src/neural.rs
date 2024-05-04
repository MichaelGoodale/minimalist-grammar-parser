use ahash::HashSet;
use burn::tensor::{activation::log_softmax, backend::Backend, Bool, Data, Int, Tensor};
use itertools::Itertools;

use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};

use self::{
    loss::NeuralConfig,
    neural_beam::{NodeFeature, StringPath, StringProbHistory},
    neural_lexicon::{NeuralLexicon, NeuralProbabilityRecord},
    parameterization::GrammarParameterization,
};

pub mod loss;
pub mod neural_beam;
pub mod neural_lexicon;
pub mod parameterization;
pub mod pathfinder;
mod utils;

#[derive(Debug, Clone)]
pub struct CompletedParse {
    parse: StringPath,
    history: StringProbHistory,
    grammar_details: Vec<LexemeTypes>,
    valid: bool,
}

#[derive(Debug, Copy, Clone)]
struct LexemeTypes {
    licensee: bool,
    lexeme_idx: usize,
    category: usize,
}

impl CompletedParse {
    pub fn len(&self) -> usize {
        self.parse.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parse.is_empty()
    }

    pub fn new<B: Backend>(
        parse: StringPath,
        history: StringProbHistory,
        valid: bool,
        lexicon: &NeuralLexicon<B>,
    ) -> Self {
        let grammar_details = history
            .attested_nodes()
            .iter()
            .filter_map(|n| match n {
                NodeFeature::Node(_) => None,
                NodeFeature::NFeats {
                    node, lexeme_idx, ..
                } => Some(match lexicon.get(*node).unwrap() {
                    FeatureOrLemma::Feature(Feature::Category(category)) => LexemeTypes {
                        licensee: false,
                        lexeme_idx: *lexeme_idx,
                        category: *category,
                    },
                    FeatureOrLemma::Feature(Feature::Licensee(category)) => LexemeTypes {
                        licensee: true,
                        lexeme_idx: *lexeme_idx,
                        category: *category,
                    },

                    _ => panic!("this should not happen!"),
                }),
            })
            .collect();

        CompletedParse {
            parse,
            history,
            valid,
            grammar_details,
        }
    }

    pub fn grammar_prob<B: Backend>(
        &self,
        g: &GrammarParameterization<B>,
        lexicon: &NeuralLexicon<B>,
    ) -> Tensor<B, 1> {
        Tensor::cat(
            self.history
                .attested_nodes()
                .iter()
                .map(|n| match n {
                    NodeFeature::Node(n) => lexicon
                        .get_weight(&NeuralProbabilityRecord::Node(*n))
                        .unwrap()
                        .clone(),
                    NodeFeature::NFeats {
                        lexeme_idx,
                        n_features,
                        n_licensees,
                        ..
                    } => {
                        g.included_features()
                            .clone()
                            .slice([*lexeme_idx..lexeme_idx + 1, *n_features..n_features + 1])
                            + g.included_licensees()
                                .clone()
                                .slice([*lexeme_idx..lexeme_idx + 1, *n_licensees..n_licensees + 1])
                    }
                    .reshape([1]),
                })
                .collect::<Vec<_>>(),
            0,
        )
        .sum_dim(0)
    }

    fn string_prob<B: Backend>(
        &self,
        g: &GrammarParameterization<B>,
        lexicon: &NeuralLexicon<B>,
        neural_config: &NeuralConfig,
        others: Option<&[&Self]>,
    ) -> Tensor<B, 1> {
        let move_p: f64 = neural_config.parsing_config.move_prob.into_inner();
        let merge_p: f64 = neural_config
            .parsing_config
            .move_prob
            .opposite_prob()
            .into_inner();

        let mut cats: Tensor<B, 3, Bool> =
            Tensor::<B, 3, Int>::ones([g.n_lexemes(), g.n_categories(), 2], &g.device()).bool();
        let false_tensor = Tensor::from_data(
            Data::new(vec![false], burn::tensor::Shape { dims: [1, 1, 1] }),
            &g.device(),
        );
        let mut valids: HashSet<_> = HashSet::default();

        for LexemeTypes {
            licensee,
            lexeme_idx,
            category,
        } in self.grammar_details.iter()
        {
            let t = if *licensee { 1 } else { 0 };
            let slice = [
                *lexeme_idx..lexeme_idx + 1,
                *category..category + 1,
                t..t + 1,
            ];

            valids.insert(slice.clone());
            cats = cats.slice_assign(slice, false_tensor.clone());
        }

        let weights: Tensor<B, 3> = log_softmax(
            g.unnormalized_weights()
                .clone()
                .unsqueeze_dims(&[1, 2])
                .repeat(1, g.n_categories())
                .repeat(2, 2)
                .mask_fill(cats, -500.0),
            0,
        );

        if let Some(others) = others {
            Tensor::cat(
                others
                    .iter()
                    .map(|p| prob_of_string(p, move_p, merge_p, &weights, lexicon))
                    .collect_vec(),
                0,
            )
        } else {
            prob_of_string(self, move_p, merge_p, &weights, lexicon)
        }
    }
}

fn prob_of_string<B: Backend>(
    parse: &CompletedParse,
    move_p: f64,
    merge_p: f64,
    weights: &Tensor<B, 3>,
    lexicon: &NeuralLexicon<B>,
) -> Tensor<B, 1> {
    let mut p: Tensor<B, 1> = Tensor::zeros([1], &weights.device());
    for (prob_type, count) in parse.history.iter() {
        match prob_type {
            NeuralProbabilityRecord::MoveRuleProb => p = p.add_scalar(move_p * (*count as f64)),
            NeuralProbabilityRecord::MergeRuleProb => p = p.add_scalar(merge_p * (*count as f64)),
            NeuralProbabilityRecord::Lexeme {
                node: lexeme,
                id: lexeme_idx,
                ..
            } => {
                let (c, t) = match lexicon.get(*lexeme).unwrap() {
                    FeatureOrLemma::Feature(Feature::Category(c)) => (*c, 0),
                    FeatureOrLemma::Feature(Feature::Licensee(c)) => (*c, 1),
                    _ => panic!("Should be impossible!"),
                };
                let slice = [*lexeme_idx..lexeme_idx + 1, c..c + 1, t..t + 1];
                p = p + weights.clone().slice(slice).mul_scalar(*count).reshape([1]);
            }
            _ => (),
        }
    }
    p
}

pub const N_TYPES: usize = 3;
