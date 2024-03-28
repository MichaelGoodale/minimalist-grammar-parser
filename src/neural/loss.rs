use std::collections::BTreeSet;

use super::neural_beam::{StringPath, StringProbHistory};
use super::neural_lexicon::{
    GrammarParameterization, NeuralFeature, NeuralLexicon, NeuralProbabilityRecord,
};
use super::utils::log_sum_exp_dim;
use crate::lexicon::{Feature, FeatureOrLemma};
use crate::{NeuralGenerator, ParsingConfig};
use ahash::HashMap;
use anyhow::bail;
use burn::tensor::{activation::log_softmax, backend::Backend, Tensor};
use burn::tensor::{Bool, Data, Int};
use logprob::LogProb;
use moka::sync::Cache;
use petgraph::graph::EdgeIndex;
use rand::Rng;

pub struct NeuralConfig {
    pub n_grammars: usize,
    pub n_strings_per_grammar: usize,
    pub n_strings_to_sample: usize,
    pub padding_length: usize,
    pub negative_weight: Option<f64>,
    pub temperature: f64,
    pub parsing_config: ParsingConfig,
}

fn retrieve_strings<B: Backend>(
    lexicon: &NeuralLexicon<B>,
    targets: &Vec<Vec<usize>>,
    lemma_lookups: &HashMap<(usize, usize), LogProb<f64>>,
    lexeme_weights: &HashMap<usize, LogProb<f64>>,
    alternatives: &HashMap<EdgeIndex, Vec<EdgeIndex>>,
    neural_config: &NeuralConfig,
) -> (Vec<StringPath>, Vec<StringProbHistory>) {
    let mut grammar_strings: Vec<_> = vec![];
    let mut string_paths: Vec<_> = vec![];

    for (s, h) in NeuralGenerator::new(
        lexicon,
        targets,
        lemma_lookups,
        lexeme_weights,
        alternatives,
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

fn get_grammar_probs<B: Backend>(
    string_paths: &[StringProbHistory],
    lexicon: &NeuralLexicon<B>,
    device: &B::Device,
) -> (
    Tensor<B, 1>,
    Vec<(
        Tensor<B, 1, Int>,
        BTreeSet<usize>,
        Vec<(usize, NeuralFeature)>,
    )>,
) {
    let mut grammar_sets = HashMap::default();
    for (i, string_path) in string_paths.iter().enumerate() {
        grammar_sets
            .entry(string_path.attested_nodes().clone())
            .or_insert(vec![])
            .push(i as u32);
    }

    let mut output = vec![];
    let mut grammar_probs = Tensor::<B, 1>::full([grammar_sets.len()], 0.0, device);
    for (i, (key, ids)) in grammar_sets.iter().enumerate() {
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
        let string_idx = Tensor::<B, 1, Int>::from_data(
            Data::from(all_valid_strings.as_slice()).convert(),
            device,
        );
        let p: Tensor<B, 1> = Tensor::cat(
            key.iter()
                .map(|e| {
                    lexicon
                        .get_weight(&NeuralProbabilityRecord::Edge(*e))
                        .unwrap()
                        .clone()
                })
                .collect::<Vec<_>>(),
            0,
        )
        .sum();
        grammar_probs = grammar_probs.slice_assign([i..i + 1], p);

        let lexemes = string_paths[ids[0] as usize]
            .keys()
            .filter_map(|x| {
                if let NeuralProbabilityRecord::Lexeme(e, lexeme_idx) = x {
                    Some((*lexeme_idx, lexicon.get_feature(*e).clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        output.push((
            string_idx,
            all_valid_strings
                .iter()
                .map(|x| (*x).try_into().unwrap())
                .collect(),
            lexemes,
        ));
    }
    (grammar_probs, output)
}

fn get_string_prob<B: Backend>(
    string_paths: &[StringProbHistory],
    lexicon: &NeuralLexicon<B>,
    g: &GrammarParameterization<B>,
    include: Option<&BTreeSet<usize>>,
    grammar_cats: &Vec<(usize, NeuralFeature)>,
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
    let cats = std::iter::repeat(true)
        .take(g.n_lexemes() * g.n_categories() * 2)
        .collect::<Vec<_>>();
    let mut cats: Tensor<B, 3, Bool> = Tensor::from_data(
        Data::new(
            cats,
            burn::tensor::Shape {
                dims: [g.n_lexemes(), g.n_categories(), 2],
            },
        ),
        &g.device(),
    );
    let false_tensor = Tensor::from_data(
        Data::new(vec![false], burn::tensor::Shape { dims: [1, 1, 1] }),
        &g.device(),
    );
    for (lexeme_idx, feature) in grammar_cats {
        let (cat, t) = match feature {
            FeatureOrLemma::Feature(Feature::Category(c)) => (*c, 0),
            FeatureOrLemma::Feature(Feature::Licensee(c)) => (*c, 1),
            _ => panic!("Should be impossible!"),
        };
        cats = cats.slice_assign(
            [*lexeme_idx..lexeme_idx + 1, cat..cat + 1, t..t + 1],
            false_tensor.clone(),
        );
    }

    let weights: Tensor<B, 3> = log_softmax(
        g.unnormalized_weights()
            .clone()
            .unsqueeze_dims(&[1, 2])
            .repeat(1, g.n_categories())
            .repeat(2, 2)
            .mask_fill(cats, -100.0),
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
                NeuralProbabilityRecord::Lexeme(lexeme, lexeme_idx) => {
                    let (c, t) = match lexicon.get_feature(*lexeme) {
                        FeatureOrLemma::Feature(Feature::Category(c)) => (*c, 0),
                        FeatureOrLemma::Feature(Feature::Licensee(c)) => (*c, 1),
                        _ => panic!("Should be impossible!"),
                    };
                    p = p + weights
                        .clone()
                        .slice([*lexeme_idx..lexeme_idx + 1, c..c + 1, t..t + 1])
                        .mul_scalar(*count)
                        .reshape([1]);
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

pub fn get_grammar<B: Backend>(
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<(Tensor<B, 3>, Tensor<B, 1>)>, Tensor<B, 1>)> {
    let (lexicon, alternatives) = NeuralLexicon::new_superimposed(g, rng)?;
    let (strings, string_probs) = retrieve_strings(
        &lexicon,
        &vec![],
        g.lemma_lookups(),
        g.lexeme_weights(),
        &alternatives,
        neural_config,
    );
    let strings = string_path_to_tensor(&strings, g, neural_config);

    let (grammar_probs, grammar_idx) = get_grammar_probs(&string_probs, &lexicon, &g.device());
    let mut grammars = vec![];
    for (grammar_id, grammar_set, grammar_cats) in grammar_idx {
        //(1, n_grammar_strings)
        let string_probs = get_string_prob(
            &string_probs,
            &lexicon,
            g,
            Some(&grammar_set),
            &grammar_cats,
            neural_config,
            &g.device(),
        );
        grammars.push((strings.clone().select(0, grammar_id.clone()), string_probs));
    }

    //(n_grammar_strings, padding_length, n_lemmas)
    Ok((grammars, grammar_probs))
}

pub fn get_neural_outputs<B: Backend>(
    g: &GrammarParameterization<B>,
    targets: Tensor<B, 2, Int>,
    neural_config: &NeuralConfig,
    rng: &mut impl Rng,
) -> anyhow::Result<Tensor<B, 1>> {
    let n_targets = targets.shape().dims[0];

    let (lexicon, alternatives) = NeuralLexicon::new_superimposed(g, rng)?;
    let target_vec = (0..n_targets)
        .map(|i| {
            let v: Vec<usize> = targets
                .clone()
                .slice([i..i + 1])
                .to_data()
                .convert::<u32>()
                .value
                .into_iter()
                .take_while(|&x| x != 1)
                .map(|x| x as usize)
                .collect();
            v
        })
        .collect();

    dbg!(&target_vec);

    let (strings, string_probs) = retrieve_strings(
        &lexicon,
        &target_vec,
        g.lemma_lookups(),
        g.lexeme_weights(),
        &alternatives,
        neural_config,
    );
    let n_strings = strings.len();

    if n_strings == 0 {
        bail!("Zero outputted strings!");
    }

    //(n_grammar_strings, padding_length, n_lemmas)
    let grammar = string_path_to_tensor(&strings, g, neural_config);

    let (grammar_probs, grammar_idx) =
        get_grammar_probs(&string_probs, &lexicon, &targets.device());

    //(n_targets, n_grammar_strings, padding_length, n_lemmas)
    let grammar: Tensor<B, 4> = grammar.unsqueeze_dim(0).repeat(0, n_targets);

    //(n_targets, n_strings_per_grammar, padding_length, 1)
    let targets: Tensor<B, 4, Int> = targets.unsqueeze_dims(&[1, 3]).repeat(1, n_strings);

    //Probability of generating every target for each string.
    //(n_targets, n_grammar_strings)
    let loss: Tensor<B, 2> = grammar
        .gather(3, targets)
        .squeeze::<3>(3)
        .sum_dim(2)
        .squeeze(2);

    let mut loss_per_grammar = vec![];
    for (grammar_id, grammar_set, grammar_cats) in grammar_idx {
        let string_probs = get_string_prob(
            &string_probs,
            &lexicon,
            g,
            Some(&grammar_set),
            &grammar_cats,
            neural_config,
            &loss.device(),
        );
        //Probability of generating each of the targets
        loss_per_grammar.push(log_sum_exp_dim(
            loss.clone().select(1, grammar_id) + string_probs.unsqueeze_dim(0),
            1,
        ));
        //(n_strings_per_grammar);
    }
    let loss_per_grammar = Tensor::cat(loss_per_grammar, 1) + grammar_probs.unsqueeze_dim(0);

    //Probability of generating each of the grammars
    let loss: Tensor<B, 1> = log_sum_exp_dim(loss_per_grammar, 1).squeeze(1);
    Ok(-loss.sum())
}
