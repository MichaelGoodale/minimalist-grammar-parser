use super::neural_beam::{StringPath, StringProbHistory};
use super::neural_lexicon::{
    GrammarParameterization, NeuralFeature, NeuralLexicon, NeuralProbabilityRecord,
};
use super::utils::log_sum_exp_dim;
use crate::{NeuralGenerator, ParsingConfig};
use ahash::HashMap;
use burn::tensor::Int;
use burn::tensor::{activation::log_softmax, backend::Backend, Tensor};
use moka::sync::Cache;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::hash_map::Entry;

pub struct NeuralConfig {
    pub n_grammars: usize,
    pub n_strings_per_grammar: usize,
    pub n_strings_to_sample: usize,
    pub padding_length: usize,
    pub negative_weight: Option<f64>,
    pub temperature: f64,
    pub parsing_config: ParsingConfig,
}

///Not technically correct as it treats the number of categories as independent from the expected
///depth.
//fn expected_mdl_score<B: Backend>(
//    types: Tensor<B, 3>,
//    categories: Tensor<B, 3>,
//    lemma_inclusion: Tensor<B, 3>,
//) -> B::FloatElem {
//    todo!();
//}

fn retrieve_strings<B: Backend>(
    lexicon: &NeuralLexicon<B>,
    neural_config: &NeuralConfig,
) -> (Vec<StringPath>, Vec<StringProbHistory>) {
    let mut grammar_strings: Vec<_> = vec![];
    let mut string_paths: Vec<_> = vec![];

    for (s, h) in NeuralGenerator::new(lexicon, &neural_config.parsing_config)
        .filter(|(s, _h)| s.len() < neural_config.padding_length)
        .take(neural_config.n_strings_per_grammar)
    {
        string_paths.push(h);
        grammar_strings.push(s);
    }
    (grammar_strings, string_paths)
}

fn string_path_to_tensor<B: Backend>(
    strings: &[StringPath],
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
) -> Tensor<B, 3> {
    let mut s_tensor: Tensor<B, 3> = log_softmax(
        Tensor::zeros([1, g.n_lemmas()], &g.device())
            .slice_assign([0..1, 0..1], Tensor::full([1, 1], 10, &g.device())),
        1,
    )
    .repeat(0, neural_config.padding_length)
    .unsqueeze_dim(0)
    .repeat(0, strings.len());

    for (s_i, s) in strings.iter().enumerate() {
        for (w_i, lexeme) in s.iter().enumerate() {
            let values = g
                .lemmas()
                .clone()
                .slice([*lexeme..lexeme + 1, 0..g.n_lemmas()])
                .unsqueeze_dim(0);
            s_tensor = s_tensor.slice_assign([s_i..s_i + 1, w_i..w_i + 1, 0..g.n_lemmas()], values)
        }
    }
    s_tensor
}

fn get_string_prob<B: Backend>(
    string_paths: &[StringProbHistory],
    lexicon: &NeuralLexicon<B>,
    neural_config: &NeuralConfig,
    device: &B::Device,
) -> Tensor<B, 2> {
    let move_p: f64 = neural_config.parsing_config.move_prob.into_inner();
    let merge_p: f64 = neural_config
        .parsing_config
        .move_prob
        .opposite_prob()
        .into_inner();

    let mut string_path_tensor = Tensor::<B, 2>::zeros([1, string_paths.len()], device);
    for (i, string_path) in string_paths.iter().enumerate() {
        let mut p: Tensor<B, 1> = Tensor::zeros([1], device);
        for (prob_type, count) in string_path.iter() {
            match prob_type {
                NeuralProbabilityRecord::OneProb => (),
                NeuralProbabilityRecord::MoveRuleProb => p = p.add_scalar(move_p * (*count as f64)),
                NeuralProbabilityRecord::MergeRuleProb => {
                    p = p.add_scalar(merge_p * (*count as f64))
                }
                NeuralProbabilityRecord::Feature(lexeme) => {
                    let feature = NeuralProbabilityRecord::Feature(*lexeme);
                    p = p.add(
                        lexicon
                            .get_weight(&feature)
                            .unwrap()
                            .clone()
                            .mul_scalar(*count),
                    );
                }
            }
        }
        string_path_tensor = string_path_tensor.slice_assign([0..1, i..i + 1], p.unsqueeze_dim(0));
    }
    string_path_tensor
}

fn get_reward_loss<B: Backend>(
    target_set: &mut HashMap<Vec<u32>, bool>,
    grammar: &Tensor<B, 3>,
    n_samples: usize,
    rng: &mut impl Rng,
) -> f32 {
    target_set.iter_mut().for_each(|(_k, v)| *v = false);

    let [n_strings, length, n_lemmas] = grammar.shape().dims;
    let lemmas_ids: Vec<_> = (0..n_lemmas).collect();

    for string_i in 0..n_strings {
        let dists: Vec<Vec<f64>> = (0..length)
            .map(|i| {
                grammar
                    .clone()
                    .slice([string_i..string_i + 1, i..i + 1, 0..n_lemmas])
                    .exp()
                    .to_data()
                    .convert::<f64>()
                    .value
            })
            .collect();

        for _ in 0..n_samples {
            let mut sample: Vec<u32> = std::iter::repeat(0).take(length).collect();
            for (j, word) in sample.iter_mut().enumerate() {
                *word = (*lemmas_ids.choose_weighted(rng, |i| dists[j][*i]).unwrap())
                    .try_into()
                    .unwrap();
            }

            match target_set.entry(sample) {
                Entry::Occupied(v) => {
                    *v.into_mut() = true;
                }
                Entry::Vacant(_) => {}
            }
        }
    }

    target_set
        .values()
        .map(|in_grammar| {
            let x: f32 = (*in_grammar).into();
            x
        })
        .sum::<f32>()
}

pub type NeuralGrammarCache =
    Cache<Vec<Vec<NeuralFeature>>, (Vec<StringPath>, Vec<StringProbHistory>)>;

pub fn get_grammar<B: Backend>(
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
    rng: &mut impl Rng,
    cache: &NeuralGrammarCache,
) -> (
    Option<(Tensor<B, 3>, Tensor<B, 1>)>,
    Tensor<B, 1>,
    Vec<Vec<NeuralFeature>>,
) {
    let (p_of_lex, lexemes, lexicon) = NeuralLexicon::new_random(g, rng);

    let entry = cache
        .entry(lexemes.clone())
        .or_insert_with(|| retrieve_strings(&lexicon, neural_config));
    let (strings, string_probs) = entry.value();

    (
        if strings.is_empty() {
            None
        } else {
            //(1, n_grammar_strings)
            let string_probs =
                get_string_prob(string_probs, &lexicon, neural_config, &g.device()).squeeze(0);

            //(n_grammar_strings, padding_length, n_lemmas)
            Some((
                string_path_to_tensor(strings, g, neural_config),
                string_probs,
            ))
        },
        p_of_lex,
        lexemes,
    )
}

pub fn get_neural_outputs<B: Backend>(
    g: &GrammarParameterization<B>,
    targets: Tensor<B, 2, Int>,
    neural_config: &NeuralConfig,
    rng: &mut impl Rng,
    cache: &NeuralGrammarCache,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let n_targets = targets.shape().dims[0];
    let target_length = targets.shape().dims[1];

    let mut target_set: HashMap<_, _> = (0..n_targets)
        .map(|i| {
            let data = targets
                .clone()
                .slice([i..i + 1, 0..target_length])
                .squeeze::<1>(0)
                .to_data()
                .convert::<u32>();
            (data.value, false)
        })
        .collect();
    //(n_targets, n_grammar_strings, padding_length, n_lemmas)
    let targets: Tensor<B, 4, Int> = targets.unsqueeze_dim::<3>(2).unsqueeze_dim(1);

    let mut loss = Tensor::zeros([1], &targets.device());
    let mut alternate_loss = Tensor::zeros([1], &targets.device());
    let mut valid_grammars = 0.0001;

    for _ in 0..neural_config.n_grammars {
        let (p_of_lex, lexemes, lexicon) = NeuralLexicon::new_random(g, rng);

        let entry = cache
            .entry(lexemes)
            .or_insert_with(|| retrieve_strings(&lexicon, neural_config));
        let (strings, string_probs) = entry.value();

        if strings.is_empty() {
            if let Some(weight) = neural_config.negative_weight {
                loss = loss + (p_of_lex.clone().add_scalar(weight));
                valid_grammars += 1.0;
            }
        } else {
            let n_grammar_strings = strings.len();

            //(1, n_grammar_strings)
            let string_probs =
                get_string_prob(string_probs, &lexicon, neural_config, &targets.device());

            //(n_grammar_strings, padding_length, n_lemmas)
            let grammar = string_path_to_tensor(strings, g, neural_config);

            let reward: f32 = get_reward_loss(
                &mut target_set,
                &grammar.clone(),
                neural_config.n_strings_to_sample,
                rng,
            );

            //(n_targets, n_grammar_strings, padding_length, n_lemmas)
            let grammar: Tensor<B, 4> = grammar.unsqueeze().repeat(0, n_targets);

            //Probability of generating every string for this grammar.
            //(n_targets, n_grammar_strings)
            let grammar_loss = grammar
                .gather(3, targets.clone().repeat(1, n_grammar_strings))
                .squeeze::<3>(3)
                .sum_dim(2)
                .squeeze::<2>(2)
                + string_probs;

            let grammar_loss: Tensor<B, 1> = log_sum_exp_dim(grammar_loss, 1).sum_dim(0);

            alternate_loss = alternate_loss + (-p_of_lex.clone() * reward);
            loss = loss + (grammar_loss * -p_of_lex);
        }
    }
    (
        -loss / valid_grammars,
        alternate_loss / (neural_config.n_grammars as f32),
    )
}
