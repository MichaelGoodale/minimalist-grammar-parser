use std::collections::{BTreeMap, BTreeSet};

use super::neural_beam::{StringPath, StringProbHistory};
use super::neural_lexicon::{NeuralFeature, NeuralLexicon};
use super::parameterization::GrammarParameterization;
use super::pathfinder::{rules_to_prob, NeuralGenerator};
use super::CompletedParse;
use crate::ParsingConfig;
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::{Bool, Data, ElementConversion, Int};
use itertools::Itertools;
use moka::sync::Cache;

#[derive(Clone, Debug)]
pub struct NeuralConfig {
    pub n_strings_per_grammar: usize,
    pub compatible_weight: f64,
    pub padding_length: usize,
    pub temperature: f64,
    pub parsing_config: ParsingConfig,
}

pub fn retrieve_strings<B: Backend>(
    lexicon: &NeuralLexicon<B>,
    g: &GrammarParameterization<B>,
    targets: Option<&[Vec<usize>]>,
    rule_logits: Tensor<B, 2>,
    lexeme_logits: Tensor<B, 2>,
    neural_config: &NeuralConfig,
    valid_only: bool,
) -> (Vec<CompletedParse>, Option<Tensor<B, 1>>) {
    let mut parses: Vec<_> = vec![];

    let mut rules = vec![];
    for (result, rule_prob) in NeuralGenerator::new(
        lexicon,
        g,
        targets,
        rule_logits,
        lexeme_logits,
        neural_config.padding_length,
        valid_only,
        neural_config,
    )
    .take(neural_config.n_strings_per_grammar)
    {
        parses.push(result);
        rules.push(rule_prob);
    }
    (
        parses,
        if rules.is_empty() {
            None
        } else {
            Some(Tensor::cat(rules, 0))
        },
    )
}

fn string_path_to_tensor<B: Backend>(
    strings: &[CompletedParse],
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
) -> Tensor<B, 3> {
    let mut s_tensor: Tensor<B, 3> = g
        .pad_vector()
        .clone()
        .unsqueeze_dim::<2>(0)
        .repeat(0, neural_config.padding_length)
        .unsqueeze_dim(0)
        .repeat(0, strings.len());

    for (s_i, s) in strings.iter().enumerate() {
        for (w_i, lexeme) in s.parse.iter().enumerate() {
            let values = g
                .lemmas()
                .clone()
                .slice([*lexeme..lexeme + 1, 0..g.n_lemmas()])
                .unsqueeze_dim(0);
            s_tensor = s_tensor.slice_assign([s_i..s_i + 1, w_i..w_i + 1, 0..g.n_lemmas()], values)
        }
        s_tensor = s_tensor.slice_assign(
            [
                s_i..s_i + 1,
                s.parse.len()..s.parse.len() + 1,
                0..g.n_lemmas(),
            ],
            g.end_vector().clone().unsqueeze_dims(&[0, 1]),
        )
    }
    s_tensor
}

///Returns (n_targets, n_strings), (n_targets, n_strings)
///Left has if it is a compatible string, and right has its prob of being generated.
fn compatible_strings<B: Backend>(
    strings: &[CompletedParse],
    targets: &[Vec<usize>],
    g: &GrammarParameterization<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut n_compatible = Vec::with_capacity(strings.len());
    let mut compatible_grammars =
        Tensor::<B, 2>::full([targets.len(), strings.len()], -999.0, &g.device());

    for (string_i, s) in strings.iter().enumerate() {
        let mut n_compatible_per_string = vec![];
        if s.valid {
            for (target_i, target) in targets.iter().enumerate() {
                let mut string_target_map = BTreeMap::default();
                let mut compatible = s.parse.len() == target.len();
                if compatible {
                    for (c, c_t) in s.parse.iter().zip(target.iter()) {
                        match string_target_map.entry(*c) {
                            std::collections::btree_map::Entry::Vacant(v) => {
                                v.insert(*c_t);
                            }
                            std::collections::btree_map::Entry::Occupied(v) => {
                                if v.get() != c_t {
                                    compatible = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                n_compatible_per_string.push(if compatible { 1.0 } else { 0.0 });
                if compatible {
                    let mut p = Tensor::<B, 2>::zeros([1, 1], &g.device());
                    for (key, value) in string_target_map {
                        p = p + g.lemmas().clone().slice([key..key + 1, value..value + 1]);
                    }
                    compatible_grammars = compatible_grammars
                        .slice_assign([target_i..target_i + 1, string_i..string_i + 1], p);
                };
            }
        } else {
            n_compatible_per_string = vec![0.0; targets.len()];
        }
        n_compatible.push(Tensor::<B, 1>::from_data(
            Data::from(n_compatible_per_string.as_slice()).convert(),
            &g.device(),
        ));
    }

    (Tensor::stack(n_compatible, 1), compatible_grammars)
}

pub type NeuralGrammarCache =
    Cache<Vec<Vec<NeuralFeature>>, (Vec<StringPath>, Vec<StringProbHistory>)>;

pub fn target_to_vec<B: Backend>(targets: &Tensor<B, 2, Int>) -> Vec<Vec<usize>> {
    targets
        .clone()
        .iter_dim(0)
        .map(|x| {
            x.iter_dim(1)
                .map(|x| x.into_scalar().elem())
                .take_while(|&x| x != 1)
                .map(|x: u32| x as usize)
                .collect()
        })
        .collect()
}

#[allow(dead_code)]
fn prefix_loss<B: Backend>(
    strings: &[CompletedParse],
    targets: Tensor<B, 2, Int>,
    target_vec: &[Vec<usize>],
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
    mask_wrong_lengths: bool,
) -> Tensor<B, 2> {
    let n_targets = targets.shape().dims[0];

    //(n_grammar_strings, padding_length, n_lemmas)
    let grammar = string_path_to_tensor(strings, g, neural_config);

    //(n_targets, n_grammar_strings, padding_length, n_lemmas)
    let grammar: Tensor<B, 4> = grammar.unsqueeze_dim(0).repeat(0, n_targets);

    //(n_targets, n_grammar_strings, padding_length, 1)
    let targets: Tensor<B, 4, Int> = targets.unsqueeze_dims(&[1, 3]).repeat(1, strings.len());

    //Probability of generating every target for each string.
    //(n_targets, n_grammar_strings, 1)
    let prefix_loss: Tensor<B, 2> = grammar
        .gather(3, targets)
        .squeeze::<3>(3)
        .sum_dim(2)
        .squeeze(2);

    if mask_wrong_lengths {
        let target_s_ids: Tensor<B, 2, Bool> = Tensor::<B, 1, Bool>::stack(
            target_vec
                .iter()
                .map(|t| {
                    let l = t.len();
                    let v = strings
                        .iter()
                        .map(|x| x.parse.len() != l)
                        .collect::<Vec<_>>();
                    let ids =
                        Tensor::<B, 1, Bool>::from_data(Data::from(v.as_slice()), &g.device());
                    ids
                })
                .collect(),
            0,
        );
        prefix_loss.mask_fill(target_s_ids, -999.0)
    } else {
        prefix_loss
    }
}

pub fn get_all_parses<B: Backend>(
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
    targets: Option<Tensor<B, 2, Int>>,
    rule_logits: Tensor<B, 2>,
    lexeme_logits: Tensor<B, 2>,
    neural_config: &NeuralConfig,
    valid_only: bool,
) -> (Vec<CompletedParse>, Option<Tensor<B, 1>>) {
    match targets {
        Some(targets) => {
            let target_vec = target_to_vec(&targets);
            retrieve_strings(
                lexicon,
                g,
                Some(&target_vec),
                rule_logits,
                lexeme_logits,
                neural_config,
                valid_only,
            )
        }
        None => retrieve_strings(
            lexicon,
            g,
            None,
            rule_logits,
            lexeme_logits,
            neural_config,
            valid_only,
        ),
    }
}

pub fn get_neural_outputs<B: Backend>(
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
    parses: &[CompletedParse],
    rule_probs: Tensor<B, 1>,
    target_vec: &[Vec<usize>],
    neural_config: &NeuralConfig,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let (n_compatible, compatible_loss) = compatible_strings(parses, target_vec, g);
    let grammar_probs = Tensor::cat(
        parses
            .iter()
            .map(|p| p.grammar_prob(g, lexicon))
            .collect_vec(),
        0,
    )
    let validity = Tensor::<B, 1>::from_data(
        Data::from(
            parses
                .iter()
                .map(|p| if p.valid { 1.0 } else { 0.0 })
                .collect_vec()
                .as_slice(),
        )
        .convert(),
        &g.device(),
    );

    let string_probs = parses
        .iter()
        .map(|p| p.string_prob(g, lexicon, neural_config, None))
        .collect_vec();

    let string_probs: Tensor<B, 1> = Tensor::cat(string_probs, 0);

    let rewards = n_compatible.clone();
    let rewards = rewards.mask_fill(
        validity
            .unsqueeze_dim(0)
            .repeat(0, target_vec.len())
            .equal_elem(-1.0),
        -0.0001,
    );
    let p_of_s =
        rewards.clone() * (compatible_loss + (string_probs + grammar_probs).unsqueeze_dim(0));

    let (max_reward, _idx) = Tensor::max_dim_with_indices(rewards, 0);
    //let idx = idx.squeeze(0);
    //let p_of_s: Tensor<B, 2> = p_of_s.select(0, idx.clone());
    //let loss = log_sum_exp_dim(
    //    compatible_loss + (string_probs + grammar_probs).unsqueeze_dim(0),
    //    1,
    //);
    (p_of_s.mean().reshape([1]), max_reward.mean())
}
