use std::collections::{BTreeMap, BTreeSet};
use std::f32::consts::LN_2;

use super::neural_beam::{NodeFeature, StringPath, StringProbHistory};
use super::neural_lexicon::{
    GrammarParameterization, NeuralFeature, NeuralLexicon, NeuralProbabilityRecord,
};
use super::utils::log_sum_exp_dim;
use crate::lexicon::Lexiconable;
use crate::lexicon::{Feature, FeatureOrLemma};
use crate::{NeuralGenerator, ParsingConfig};
use ahash::{HashMap, HashSet};
use burn::tensor::activation::softmax;
use burn::tensor::{activation::log_softmax, backend::Backend, Tensor};
use burn::tensor::{Bool, Data, ElementConversion, Int};
use itertools::Itertools;
use moka::sync::Cache;

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
) -> (Vec<StringPath>, Vec<StringProbHistory>) {
    let mut grammar_strings: Vec<_> = vec![];
    let mut string_paths: Vec<_> = vec![];
    let max_string_len = targets.map(|x| x.iter().map(|x| x.len()).max().unwrap());

    for (s, h) in NeuralGenerator::new(
        lexicon,
        g,
        targets,
        max_string_len,
        &neural_config.parsing_config,
    )
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
        s_tensor = s_tensor.slice_assign(
            [s_i..s_i + 1, s.len()..s.len() + 1, 0..g.n_lemmas()],
            g.end_vector().clone().unsqueeze_dims(&[0, 1]),
        )
    }
    s_tensor
}

///Returns (n_targets, n_strings), (n_targets, n_strings)
///Left has if it is a compatible string, and right has its prob of being generated.
fn compatible_strings<B: Backend>(
    strings: &[StringPath],
    targets: &[Vec<usize>],
    g: &GrammarParameterization<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut n_compatible = Vec::with_capacity(strings.len());
    let mut compatible_loss =
        Tensor::<B, 2>::full([targets.len(), strings.len()], -999.0, &g.device());

    for (string_i, s) in strings.iter().enumerate() {
        let mut n_compatible_per_string = vec![];
        for (target_i, target) in targets.iter().enumerate() {
            let mut string_target_map = BTreeMap::default();
            let mut compatible = s.len() == target.len();
            if compatible {
                for (c, c_t) in s.iter().zip(target.iter()) {
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
                let mut p = Tensor::zeros([1, 1], &g.device());
                for (c, c_t) in string_target_map.iter() {
                    p = p + g.lemmas().clone().slice([*c..c + 1, *c_t..c_t + 1]);
                }
                compatible_loss = compatible_loss
                    .slice_assign([target_i..target_i + 1, string_i..string_i + 1], p);
            }
        }
        n_compatible.push(Tensor::<B, 1>::from_data(
            Data::from(n_compatible_per_string.as_slice()).convert(),
            &g.device(),
        ));
    }

    (Tensor::stack(n_compatible, 1), compatible_loss)
}

#[derive(Debug, Copy, Clone)]
struct LexemeTypes {
    licensee: bool,
    lexeme_idx: usize,
    category: usize,
}

fn get_prob_of_grammar<B: Backend>(
    nodes: &BTreeSet<NodeFeature>,
    lexicon: &NeuralLexicon<B>,
    g: &GrammarParameterization<B>,
) -> (Tensor<B, 1>, Vec<LexemeTypes>) {
    let mut g_types = vec![];
    let mut unattested = std::iter::repeat(true)
        .take(g.n_lexemes())
        .collect::<Vec<_>>();
    let mut attested = vec![];
    let p: Tensor<B, 1> = Tensor::cat(
        nodes
            .iter()
            .map(|n| match n {
                NodeFeature::Node(n) => lexicon
                    .get_weight(&NeuralProbabilityRecord::Node(*n))
                    .unwrap()
                    .clone(),
                NodeFeature::NFeats {
                    node,
                    lexeme_idx,
                    n_features,
                    n_licensees,
                } => {
                    let x = match lexicon.get(*node).unwrap() {
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
                    };
                    unattested[*lexeme_idx] = false;
                    attested.push(*lexeme_idx as u32);
                    g_types.push(x);

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
    .sum_dim(0);

    let unattested = {
        let unattested = unattested
            .into_iter()
            .enumerate()
            .filter_map(|(i, x)| if x { Some(i as u32) } else { None })
            .collect::<Vec<_>>();
        if unattested.is_empty() {
            Tensor::<B, 1>::zeros([1], &g.device())
        } else {
            let unattested = Tensor::<B, 1, Int>::from_data(
                Data::from(unattested.as_slice()).convert(),
                &g.device(),
            );
            g.include_lemma()
                .clone()
                .slice([0..g.n_lexemes(), 0..1])
                .select(0, unattested)
                .sum_dim(0)
                .reshape([1])
        }
    };
    let attested = {
        if attested.is_empty() {
            Tensor::<B, 1>::zeros([1], &g.device())
        } else {
            let attested = Tensor::<B, 1, Int>::from_data(
                Data::from(attested.as_slice()).convert(),
                &g.device(),
            );
            g.include_lemma()
                .clone()
                .slice([0..g.n_lexemes(), 1..2])
                .select(0, attested)
                .sum_dim(0)
                .reshape([1])
        }
    };

    (p + attested + unattested, g_types)
}

fn get_grammar_per_string<B: Backend>(
    string_paths: &[StringProbHistory],
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
) -> (Tensor<B, 1>, Vec<Vec<LexemeTypes>>) {
    let mut grammar_probs = Tensor::<B, 1>::full([string_paths.len()], 0.0, &g.device());
    let mut cats = vec![];
    for (i, string_path) in string_paths.iter().enumerate() {
        let (p, g) = get_prob_of_grammar(string_path.attested_nodes(), lexicon, g);
        grammar_probs = grammar_probs.slice_assign([i..i + 1], p);
        cats.push(g);
    }
    (grammar_probs, cats)
}

fn get_grammar_probs<B: Backend>(
    string_paths: &[StringProbHistory],
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
) -> (
    Tensor<B, 1>,
    Vec<(usize, Tensor<B, 1, Int>, BTreeSet<usize>, Vec<LexemeTypes>)>,
) {
    let mut grammar_sets = HashMap::default();
    for (i, string_path) in string_paths.iter().enumerate() {
        grammar_sets
            .entry(string_path.attested_nodes().clone())
            .or_insert(vec![])
            .push(i as u32);
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
        all_valid_strings.sort();
        let string_idx = Tensor::<B, 1, Int>::from_data(
            Data::from(all_valid_strings.as_slice()).convert(),
            &g.device(),
        );
        let (p, g_types) = get_prob_of_grammar(key, lexicon, g);
        grammar_probs = grammar_probs.slice_assign([i..i + 1], p);

        output.push((
            key_id,
            string_idx,
            all_valid_strings
                .iter()
                .map(|x| (*x).try_into().unwrap())
                .collect(),
            g_types,
        ));
    }
    (grammar_probs, output)
}

fn get_string_prob<B: Backend>(
    string_paths: &[StringProbHistory],
    lexicon: &NeuralLexicon<B>,
    g: &GrammarParameterization<B>,
    include: Option<&BTreeSet<usize>>,
    grammar_cats: &[LexemeTypes],
    neural_config: &NeuralConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let move_p: f64 = neural_config.parsing_config.move_prob.into_inner();
    let merge_p: f64 = neural_config
        .parsing_config
        .move_prob
        .opposite_prob()
        .into_inner();

    let mut string_path_tensor = Tensor::<B, 1>::full(
        [include.map(|s| s.len()).unwrap_or(string_paths.len())],
        0.0,
        device,
    );
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
    } in grammar_cats
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

    for (i, string_path) in string_paths
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            if include.map(|s| s.contains(&i)).unwrap_or(true) {
                Some(x)
            } else {
                None
            }
        })
        .enumerate()
    {
        let mut p: Tensor<B, 1> = Tensor::zeros([1], device);
        for (prob_type, count) in string_path.iter() {
            match prob_type {
                NeuralProbabilityRecord::MoveRuleProb => p = p.add_scalar(move_p * (*count as f64)),
                NeuralProbabilityRecord::MergeRuleProb => {
                    p = p.add_scalar(merge_p * (*count as f64))
                }
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
                    if !valids.contains(&slice) {
                        print!("Cannot find {slice:?} in {valids:?}");
                    }
                    p = p + weights.clone().slice(slice).mul_scalar(*count).reshape([1]);
                }
                _ => (),
            }
        }
        string_path_tensor = string_path_tensor.slice_assign([i..i + 1], p);
    }
    string_path_tensor
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
    let (strings, string_probs) = retrieve_strings(lexicon, g, Some(&target_vec), neural_config);
    let (losses, _, grammar_losses, grammar_idx, _) = get_grammar_losses(
        g,
        lexicon,
        &strings,
        &string_probs,
        &target_vec,
        targets,
        neural_config,
        true,
    );
    let string_tensor = string_path_to_tensor(&strings, g, neural_config);
    let mut grammars = vec![];
    for (full_grammar_string_id, grammar_id, grammar_set, grammar_cats) in grammar_idx {
        let grammar = lexicon.grammar_features(&string_probs[full_grammar_string_id]);
        //(1, n_grammar_strings)
        let string_probs = get_string_prob(
            &string_probs,
            lexicon,
            g,
            Some(&grammar_set),
            &grammar_cats,
            neural_config,
            &g.device(),
        );
        grammars.push((
            grammar,
            string_tensor.clone().select(0, grammar_id.clone()),
            string_probs,
        ));
    }

    //(n_grammar_strings, padding_length, n_lemmas)
    Ok((grammars, losses.sum_dim(0).squeeze(0), grammar_losses))
}

fn get_grammar_losses<B: Backend>(
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
    strings: &[StringPath],
    string_probs: &[StringProbHistory],
    target_vec: &[Vec<usize>],
    targets: Tensor<B, 2, Int>,
    neural_config: &NeuralConfig,
    grammar_splitting: bool,
) -> (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 1>,
    Vec<(usize, Tensor<B, 1, Int>, BTreeSet<usize>, Vec<LexemeTypes>)>,
    Tensor<B, 1>,
) {
    let n_targets = targets.shape().dims[0];

    let (n_compatible, compatible_loss) = compatible_strings(strings, target_vec, g);

    let target_s_ids: Tensor<B, 2, Bool> = Tensor::<B, 1, Bool>::stack(
        target_vec
            .iter()
            .map(|t| {
                let l = t.len();
                let v = strings.iter().map(|x| x.len() != l).collect::<Vec<_>>();
                let ids = Tensor::<B, 1, Bool>::from_data(Data::from(v.as_slice()), &g.device());
                ids
            })
            .collect(),
        0,
    );
    let n_strings = strings.len();

    //(n_grammar_strings, padding_length, n_lemmas)
    let grammar = string_path_to_tensor(strings, g, neural_config);

    //(n_targets, n_grammar_strings, padding_length, n_lemmas)
    let grammar: Tensor<B, 4> = grammar.unsqueeze_dim(0).repeat(0, n_targets);

    //(n_targets, n_grammar_strings, padding_length, 1)
    let targets: Tensor<B, 4, Int> = targets.unsqueeze_dims(&[1, 3]).repeat(1, n_strings);

    //Probability of generating every target for each string.
    //(n_targets, n_grammar_strings, 1)
    let loss: Tensor<B, 2> = grammar
        .gather(3, targets)
        .squeeze::<3>(3)
        .sum_dim(2)
        .squeeze(2)
        .mask_fill(target_s_ids, -999.0);

    let loss: Tensor<B, 2> = log_sum_exp_dim(
        Tensor::cat(
            vec![
                loss.unsqueeze_dim::<3>(2) + (0.5_f32).ln(),
                compatible_loss.unsqueeze_dim(2) + (0.5_f32).ln(),
            ],
            2,
        ),
        2,
    )
    .squeeze(2);

    if grammar_splitting {
        let (grammar_probs, grammar_idx) = get_grammar_probs(string_probs, g, lexicon);
        let mut loss_per_grammar = vec![];
        let mut compatible_per_grammar = vec![];
        for (_full_grammar_string_id, grammar_id, grammar_set, grammar_cats) in grammar_idx.iter() {
            let string_probs = get_string_prob(
                string_probs,
                lexicon,
                g,
                Some(grammar_set),
                grammar_cats,
                neural_config,
                &loss.device(),
            );

            let loss = loss.clone().select(1, grammar_id.clone()) + string_probs.unsqueeze_dim(0);
            //Probability of generating each of the targets
            loss_per_grammar.push(log_sum_exp_dim(loss, 1));

            let n_compatible = n_compatible
                .clone()
                .select(1, grammar_id.clone())
                .sum_dim(1)
                .squeeze(1);
            let n_compatible = n_compatible
                .clone()
                .mask_fill(n_compatible.greater_equal_elem(1.0), 1.0)
                .sum_dim(0);
            //How many compatible strings in the grammar?
            compatible_per_grammar.push(n_compatible);
            //(n_strings_per_grammar);
        }
        (
            Tensor::cat(loss_per_grammar, 1),
            Tensor::zeros([1, 1], &g.device()),
            grammar_probs,
            grammar_idx,
            Tensor::cat(compatible_per_grammar, 0),
        )
    } else {
        let n_compatible = n_compatible.sum_dim(0).squeeze(0);
        let n_compatible = n_compatible
            .clone()
            .mask_fill(n_compatible.greater_equal_elem(1.0), 1.0);
        let (grammar_probs, grammar_cats) = get_grammar_per_string(string_probs, g, lexicon);
        let mut s = vec![];
        for (i, cats) in grammar_cats.into_iter().enumerate() {
            s.push(get_string_prob(
                string_probs,
                lexicon,
                g,
                Some(&BTreeSet::from([i])),
                &cats,
                neural_config,
                &g.device(),
            ));
        }
        (
            loss,
            Tensor::stack(s, 1),
            grammar_probs,
            vec![],
            n_compatible.clone(),
        )
    }
}

pub fn get_all_parses<B: Backend>(
    g: &GrammarParameterization<B>,
    lexicon: &NeuralLexicon<B>,
    targets: Option<Tensor<B, 2, Int>>,
    neural_config: &NeuralConfig,
) -> (Vec<StringPath>, Vec<StringProbHistory>) {
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
    strings: &[StringPath],
    string_probs: &[StringProbHistory],
    target_vec: &[Vec<usize>],
    targets: Tensor<B, 2, Int>,
    neural_config: &NeuralConfig,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let (loss_per_grammar, string_probs, grammar_losses, _, n_compatible) = get_grammar_losses(
        g,
        lexicon,
        strings,
        string_probs,
        target_vec,
        targets,
        neural_config,
        false,
    );

    let max_n_compatible: f32 = n_compatible.clone().max_dim(0).into_scalar().elem();
    let idx: Vec<u32> = n_compatible
        .clone()
        .equal_elem(max_n_compatible)
        .iter_dim(0)
        .enumerate()
        .filter_map(|(i, x)| {
            if x.into_scalar() {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect();

    let idx = Tensor::<B, 1, Int>::from_data(Data::from(idx.as_slice()).convert(), &g.device());

    let s_w: Tensor<B, 2> = if max_n_compatible == 1.0 {
        softmax(string_probs.clone().select(1, idx.clone()), 1)
    } else {
        Tensor::ones([1, idx.shape().dims[0]], &g.device())
    };
    let grammar =
        string_probs.clone() + loss_per_grammar.clone() + grammar_losses.clone().unsqueeze_dim(0);

    (
        -(log_sum_exp_dim(grammar.clone().select(1, idx), 1).squeeze(1)).mean_dim(0),
        -log_sum_exp_dim(grammar, 1).squeeze(1).mean_dim(0),
    )
}
