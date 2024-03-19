use super::neural_beam::{StringPath, StringProbHistory};
use super::neural_lexicon::{
    GrammarParameterization, NeuralFeature, NeuralLexicon, NeuralProbabilityRecord,
};
use super::utils::log_sum_exp_dim;
use crate::{neural, NeuralGenerator, ParsingConfig};
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
    let mut s_tensor: Tensor<B, 3> = g
        .pad_vector()
        .clone()
        .unsqueeze_dim::<2>(0)
        .repeat(0, neural_config.padding_length)
        .unsqueeze_dim(0)
        .repeat(0, neural_config.n_strings_per_grammar);

    for (s_i, s) in strings.iter().enumerate() {
        for (w_i, lexeme) in s.iter().enumerate() {
            let values = g
                .lemmas()
                .clone()
                .slice([*lexeme..lexeme + 1, 0..g.n_lemmas()])
                .unsqueeze_dim(0);
            s_tensor = s_tensor.slice_assign([s_i..s_i + 1, w_i..w_i + 1, 0..g.n_lemmas()], values)
        }
        s_tensor = s_tensor.slice_assign(
            [s_i..s_i + 1, s.len()..s.len() + 1, 0..g.n_lemmas()],
            g.end_vector().clone().unsqueeze_dims(&[0, 1]),
        )
    }
    s_tensor
}

const EPSILON: f64 = -10.0;
const EPSILON_INV: f64 = -0.000045400960370489214;

fn get_string_prob<B: Backend>(
    string_paths: &[StringProbHistory],
    lexicon: &NeuralLexicon<B>,
    neural_config: &NeuralConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let move_p: f64 = neural_config.parsing_config.move_prob.into_inner();
    let merge_p: f64 = neural_config
        .parsing_config
        .move_prob
        .opposite_prob()
        .into_inner();

    let n_strings: f64 = (string_paths.len() as f64).ln();
    let n_fakes: f64 = ((neural_config.n_strings_per_grammar - string_paths.len()) as f64).ln();
    let mut string_path_tensor = Tensor::<B, 1>::full(
        [neural_config.n_strings_per_grammar],
        EPSILON_INV - n_strings,
        device,
    );

    if neural_config.n_strings_per_grammar > string_paths.len() {
        string_path_tensor = string_path_tensor.slice_assign(
            [string_paths.len()..neural_config.n_strings_per_grammar],
            Tensor::<B, 1>::full(
                [neural_config.n_strings_per_grammar - string_paths.len()],
                EPSILON - n_fakes,
                device,
            ),
        );
    }

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
        string_path_tensor = string_path_tensor.slice_assign([i..i + 1], p);
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
            let string_probs = get_string_prob(string_probs, &lexicon, neural_config, &g.device());

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
) -> Tensor<B, 1> {
    let n_targets = targets.shape().dims[0];

    //(n_grammar_strings, padding_length, n_lemmas)
    let mut grammars = vec![];
    let mut probs = vec![];

    for _ in 0..neural_config.n_grammars {
        let (p_of_lex, lexicon) = NeuralLexicon::new_superimposed(g, rng);

        let (strings, string_probs) = retrieve_strings(&lexicon, neural_config);
        //let entry = cache
        //    .entry(lexemes)
        //    .or_insert_with(|| retrieve_strings(&lexicon, neural_config));
        //let (strings, string_probs) = entry.value();

        let grammar = string_path_to_tensor(&strings, g, neural_config);
        let string_probs = if strings.is_empty() {
            Tensor::full(
                [neural_config.n_strings_per_grammar],
                EPSILON - (neural_config.n_strings_per_grammar as f64).ln(),
                &g.device(),
            )
        } else {
            get_string_prob(&string_probs, &lexicon, neural_config, &targets.device())
        };
        grammars.push(grammar + p_of_lex.unsqueeze_dims(&[1, 2]));
        probs.push(string_probs);
    }
    //(n_grammars, n_strings_per_grammar);
    let string_probs: Tensor<B, 2> = Tensor::stack(probs, 1);

    //(n_grammar_strings, n_grammars, padding_length, n_lemmas)
    let grammar: Tensor<B, 4> = Tensor::stack(grammars, 1);

    //(n_targets, n_grammar_strings, n_grammars, padding_length, n_lemmas)
    let grammar: Tensor<B, 5> = grammar.unsqueeze_dim(0).repeat(0, n_targets);

    let targets: Tensor<B, 5, Int> = targets
        .unsqueeze_dims(&[1, 2, 4])
        .repeat(1, neural_config.n_strings_per_grammar)
        .repeat(2, neural_config.n_grammars);

    //Probability of generating every string for each grammar.
    //(n_targets, n_grammar_strings, n_grammars)
    let loss: Tensor<B, 3> = grammar
        .gather(4, targets)
        .squeeze::<4>(4)
        .sum_dim(3)
        .squeeze::<3>(3)
        + string_probs.unsqueeze_dim(0);

    let loss: Tensor<B, 2> = log_sum_exp_dim(loss, 2) - (neural_config.n_grammars as f64).ln();

    //Probability of generating each of the strings
    let loss: Tensor<B, 1> = log_sum_exp_dim(loss, 1).sum_dim(0);
    -loss
}
