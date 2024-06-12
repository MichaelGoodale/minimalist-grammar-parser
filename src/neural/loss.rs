use std::collections::{BTreeMap, BTreeSet};

use super::neural_beam::{StringPath, StringProbHistory};
use super::neural_lexicon::{NeuralFeature, NeuralLexicon};
use super::parameterization::GrammarParameterization;
use super::pathfinder::NeuralGenerator;
use super::utils::log_sum_exp_dim;
use super::CompletedParse;
use crate::ParsingConfig;
use ahash::HashMap;
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
    neural_config: &NeuralConfig,
) -> Vec<CompletedParse> {
    let mut parses: Vec<_> = vec![];
    let lens: Option<BTreeSet<usize>> = targets.map(|x| x.iter().map(|x| x.len()).collect());

    for result in NeuralGenerator::new(
        lexicon,
        g,
        targets,
        neural_config.padding_length,
        true,
        neural_config,
    )
    .filter(|x| {
        if let Some(lens) = lens.as_ref() {
            lens.contains(&x.len())
        } else {
            true
        }
    })
    .take(neural_config.n_strings_per_grammar)
    {
        parses.push(result)
    }
    parses
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

fn split_into_grammars<B: Backend>(
    string_paths: &[CompletedParse],
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
) -> (
    Tensor<B, 1>,
    Vec<(usize, Tensor<B, 1, Int>, BTreeSet<usize>)>,
) {
    let mut grammar_sets = HashMap::default();
    for (i, string_path) in string_paths.iter().enumerate().filter(|(_, x)| x.valid) {
        grammar_sets
            .entry(string_path.history.attested_nodes().clone())
            .or_insert(BTreeSet::default())
            .insert(i as u32);
    }

    let mut output = vec![];
    let mut grammar_probs = Tensor::<B, 1>::full([grammar_sets.len()], 0.0, &g.device());
    for (i, (key, ids)) in grammar_sets.iter().enumerate() {
        let key_id = *ids.first().unwrap() as usize;
        let mut all_valid_strings = ids.clone();
        for (key2, ids) in grammar_sets
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, x)| x)
        {
            if key2.is_subset(key) {
                all_valid_strings.extend(ids);
            }
        }
        let mut all_valid_strings = all_valid_strings.into_iter().collect_vec();
        all_valid_strings.sort();
        let string_idx = Tensor::<B, 1, Int>::from_data(
            Data::from(all_valid_strings.as_slice()).convert(),
            &g.device(),
        );
        let p = string_paths[key_id].grammar_prob(g, lexicon);
        grammar_probs = grammar_probs.slice_assign([i..i + 1], p);

        output.push((
            key_id,
            string_idx,
            all_valid_strings
                .iter()
                .map(|x| (*x).try_into().unwrap())
                .collect(),
        ));
    }
    (grammar_probs, output)
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

pub fn get_grammar_with_targets<B: Backend>(
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
    targets: Tensor<B, 2, Int>,
    neural_config: &NeuralConfig,
) -> anyhow::Result<(
    Vec<(Vec<Vec<NeuralFeature>>, Tensor<B, 3>, Tensor<B, 1>)>,
    Tensor<B, 1>,
    Tensor<B, 1>,
)> {
    let target_vec = target_to_vec(&targets);
    let parses = retrieve_strings(lexicon, g, Some(&target_vec), neural_config)
        .into_iter()
        .filter(|x| x.valid)
        .collect_vec();

    let (_, compatible_loss) = compatible_strings(&parses, &target_vec, g);
    let (grammar_probs, grammar_idx) = split_into_grammars(&parses, g, lexicon);
    let string_tensor = string_path_to_tensor(&parses, g, neural_config);
    let mut grammars = vec![];
    let mut loss = vec![];
    for (full_grammar_string_id, grammar_id, grammar_set) in grammar_idx {
        loss.push(log_sum_exp_dim(
            compatible_loss.clone().select(1, grammar_id.clone()),
            1,
        ));
        let parse = &parses[full_grammar_string_id];
        let grammar = lexicon.grammar_features(&parse.history);
        //(1, n_grammar_strings)
        let slice: Vec<&CompletedParse> = parses
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if grammar_set.contains(&i) {
                    Some(p)
                } else {
                    None
                }
            })
            .collect_vec();
        grammars.push((
            grammar,
            string_tensor.clone().select(0, grammar_id.clone()),
            parse.string_prob(g, lexicon, neural_config, Some(&slice)),
        ));
    }

    //(n_grammar_strings, padding_length, n_lemmas)
    Ok((
        grammars,
        Tensor::cat(loss, 1).sum_dim(0).squeeze(0),
        grammar_probs,
    ))
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
    neural_config: &NeuralConfig,
) -> Vec<CompletedParse> {
    match targets {
        Some(targets) => {
            let target_vec = target_to_vec(&targets);
            retrieve_strings(lexicon, g, Some(&target_vec), neural_config)
        }
        None => retrieve_strings(lexicon, g, None, neural_config),
    }
}

pub fn get_neural_outputs<B: Backend>(
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
    parses: &[CompletedParse],
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
    );
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
    )
    .unsqueeze_dim(0);
    let string_probs = parses
        .iter()
        .map(|p| p.string_prob(g, lexicon, neural_config, None))
        .collect_vec();

    let string_probs: Tensor<B, 1> = Tensor::cat(string_probs, 0);

    let rewards = string_probs.clone().unsqueeze_dim(0).exp() * validity * n_compatible.clone();
    //let rewards = rewards.mask_fill(
    //    validity
    //        .unsqueeze_dim(0)
    //        .repeat(0, target_vec.len())
    //        .equal_elem(-1.0),
    //    -1.0,
    //);
    let p_of_s = (rewards.clone()
        - (compatible_loss + (string_probs + grammar_probs).unsqueeze_dim(0)).exp())
    .powf_scalar(2.0);
    let (max_reward, _idx) = Tensor::max_dim_with_indices(rewards, 0);
    //let idx = idx.squeeze(0);
    //let p_of_s: Tensor<B, 2> = p_of_s.select(0, idx.clone());
    //let loss = log_sum_exp_dim(
    //    compatible_loss + (string_probs + grammar_probs).unsqueeze_dim(0),
    //    1,
    //);
    (p_of_s.mean().reshape([1]), max_reward.mean())
}
