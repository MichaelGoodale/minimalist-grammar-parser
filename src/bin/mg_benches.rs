#![allow(unused_variables, dead_code)]

use std::{collections::BTreeMap, time::Instant};

use anyhow::Result;
use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    tensor::{backend::Backend, Data, Int, Tensor},
};
use itertools::Itertools;
use logprob::LogProb;
use minimalist_grammar_parser::{
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{Lexicon, SimpleLexicalEntry},
    neural::{
        loss::{get_all_parses, NeuralConfig},
        neural_lexicon::NeuralLexicon,
        parameterization::GrammarParameterization,
        N_TYPES,
    },
    Generator, Parser, ParsingConfig,
};
use rand::{Rng, SeedableRng};

fn main() {
    let config: ParsingConfig = ParsingConfig::new(
        LogProb::new(-256.0).unwrap(),
        LogProb::from_raw_prob(0.5).unwrap(),
        100,
        1000,
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(32);
    let targets = (1..5)
        .map(|i| {
            let mut s: [u32; 11] = [0; 11];
            s.iter_mut().take(i).for_each(|x| *x = 3);
            s[i] = 1;
            Tensor::<NdArray, 1, Int>::from_data(Data::from(s).convert(), &NdArrayDevice::default())
        })
        .collect::<Vec<_>>();
    let g = get_grammar_structure::<NdArray>(&mut rng, 1.0, NdArrayDevice::default()).unwrap();

    for depth in [1, 5, 10, 20] {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-512.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            depth,
            5000,
        );
        let neural_config = NeuralConfig {
            compatible_weight: 0.5,
            n_strings_per_grammar: 1000000,
            padding_length: 11,
            temperature: 1.0,
            parsing_config: config,
        };
        let start = Instant::now();
        let lexicon = NeuralLexicon::new_superimposed(&g, &neural_config).unwrap();
        let (strings, string_probs) = get_all_parses(&g, &lexicon, None, &neural_config);
        let elapse = start.elapsed();
        let n = strings.len();
        let mut counts = BTreeMap::default();
        strings
            .iter()
            .for_each(|x| *counts.entry(x.len()).or_insert(0) += 1);

        println!(
            "DONE, {depth} has {n} parses in {} ms and the following counts: {counts:?}",
            elapse.as_millis()
        );
    }
}

fn get_grammar() -> Lexicon<&'static str, char> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Lexicon::new(v)
}

fn parse_copy_language_together(config: &ParsingConfig, record_rules: bool) {
    let (lex, strings) = {
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let lex = Lexicon::new(v);

        let mut strings = Vec::<Vec<&str>>::new();
        strings.push(vec![]);

        for i in 1..=5 {
            strings.extend(
                itertools::repeat_n(vec!["a", "b"].into_iter(), i)
                    .multi_cartesian_product()
                    .map(|mut x| {
                        x.append(&mut x.clone());
                        x
                    }),
            );
        }
        (lex, strings)
    };

    (if record_rules {
        Parser::new_multiple
    } else {
        Parser::new_skip_rules_multiple
    })(&lex, 'T', &strings, config)
    .unwrap()
    .take(strings.len())
    .for_each(|_| ());
}

fn parse_long_sentence(config: &ParsingConfig, record_rules: bool) {
    let g = get_grammar();
    let sentence: Vec<&str> = "which king knows the queen knows which beer the king drinks"
        .split(' ')
        .collect();

    (if record_rules {
        Parser::new
    } else {
        Parser::new_skip_rules
    })(&g, 'C', &sentence, config)
    .unwrap()
    .next()
    .unwrap();
}

fn generate_sentence(config: &ParsingConfig, record_rules: bool) {
    let g = get_grammar();
    (if record_rules {
        Generator::new
    } else {
        Generator::new_skip_rules
    })(&g, 'C', config)
    .unwrap()
    .take(100)
    .count();
}

fn parse_copy_language(config: &ParsingConfig, record_rules: bool) {
    let v: Vec<_> = COPY_LANGUAGE
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let lex = Lexicon::new(v);

    let mut strings = Vec::<Vec<&str>>::new();
    strings.push(vec![]);

    for i in 1..=5 {
        strings.extend(
            itertools::repeat_n(vec!["a", "b"].into_iter(), i)
                .multi_cartesian_product()
                .map(|mut x| {
                    x.append(&mut x.clone());
                    x
                }),
        );
    }

    for s in strings.iter() {
        (if record_rules {
            Parser::new
        } else {
            Parser::new_skip_rules
        })(&lex, 'T', s, config)
        .unwrap()
        .next()
        .unwrap();
    }
}

fn generate_copy_language(config: &ParsingConfig, record_rules: bool) {
    let v: Vec<_> = COPY_LANGUAGE
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let lex = Lexicon::new(v);
    (if record_rules {
        Generator::new
    } else {
        Generator::new_skip_rules
    })(&lex, 'T', config)
    .unwrap()
    .take(100)
    .count();
}

fn get_grammar_structure<B: Backend>(
    rng: &mut impl Rng,
    temperature: f64,
    device: B::Device,
) -> anyhow::Result<GrammarParameterization<B>> {
    let n_lexemes = 20;
    let n_pos = 2;
    let n_licensee = 1;
    let n_categories = 2;
    let n_lemmas = 2000;
    let lemmas = Tensor::<B, 2>::zeros([n_lexemes, n_lemmas], &device);

    let types = Tensor::<B, 3>::zeros([n_lexemes, n_pos, N_TYPES], &device);

    let type_categories = Tensor::<B, 3>::zeros([n_lexemes, n_pos, n_categories], &device);

    let licensee_categories = Tensor::<B, 3>::zeros([n_lexemes, n_licensee, n_categories], &device);
    let included_licensees = Tensor::<B, 2>::zeros([n_lexemes, n_licensee + 1], &device);

    let included_features = Tensor::<B, 2>::zeros([n_lexemes, n_pos + 1], &device);

    let categories = Tensor::<B, 2>::zeros([n_lexemes, n_categories], &device);
    let weights = Tensor::<B, 1>::zeros([n_lexemes], &device);

    let silent_lemmas = Tensor::<B, 2>::zeros([n_lexemes, 2], &device);
    let include_lemmas = Tensor::<B, 1>::zeros([n_lexemes], &device);

    let pad_vector =
        Tensor::<B, 1>::from_floats([10., 0., 0., 0., 0., 0., 0., 0., 0., 0.], &device);
    let end_vector =
        Tensor::<B, 1>::from_floats([0., 10., 0., 0., 0., 0., 0., 0., 0., 0.], &device);
    GrammarParameterization::new(
        types.clone(),
        type_categories.clone(),
        licensee_categories.clone(),
        included_features.clone(),
        included_licensees.clone(),
        lemmas.clone(),
        silent_lemmas.clone(),
        categories.clone(),
        weights.clone(),
        include_lemmas.clone(),
        pad_vector.clone(),
        end_vector.clone(),
        temperature,
        false,
        rng,
    )
}
