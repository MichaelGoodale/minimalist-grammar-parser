#![allow(unused_variables, dead_code)]

use anyhow::Result;
use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    tensor::Tensor,
};
use itertools::Itertools;
use logprob::LogProb;
use minimalist_grammar_parser::{
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{Lexicon, SimpleLexicalEntry},
    neural::{
        loss::{get_neural_outputs, NeuralConfig},
        neural_lexicon::GrammarParameterization,
        N_TYPES,
    },
    Generator, Parser, ParsingConfig,
};
use rand::SeedableRng;

fn main() {
    let config: ParsingConfig = ParsingConfig::new(
        LogProb::new(-256.0).unwrap(),
        LogProb::from_raw_prob(0.5).unwrap(),
        100,
        1000,
    );
    for _ in 0..10 {
        random_neural_generation().unwrap();
        //parse_long_sentence(&config, false);
        //parse_copy_language(&config, false);
        //generate_sentence(&config, false);
        //generate_copy_language(&config, false);
        //parse_copy_language_together(&config, false);
    }
    println!("DONE");
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

fn random_neural_generation() -> Result<()> {
    let n_lexemes = 5;
    let n_pos = 3;
    let n_licensee = 2;
    let n_categories = 4;
    let n_lemmas = 10;
    let lemmas = Tensor::<NdArray, 2>::zeros([n_lexemes, n_lemmas], &NdArrayDevice::default());

    let types = Tensor::<NdArray, 3>::zeros([n_lexemes, n_pos, N_TYPES], &NdArrayDevice::default());

    let type_categories =
        Tensor::<NdArray, 3>::zeros([n_lexemes, n_pos, n_categories], &NdArrayDevice::default());

    let licensee_categories = Tensor::<NdArray, 3>::zeros(
        [n_lexemes, n_licensee, n_categories],
        &NdArrayDevice::default(),
    );
    let included_features = Tensor::<NdArray, 3>::zeros(
        [n_lexemes, n_licensee + n_pos, 2],
        &NdArrayDevice::default(),
    );

    let categories =
        Tensor::<NdArray, 2>::zeros([n_lexemes, n_categories], &NdArrayDevice::default());
    let weights = Tensor::<NdArray, 1>::zeros([n_lexemes], &NdArrayDevice::default());

    let silent_lemmas = Tensor::<NdArray, 2>::zeros([n_lexemes, 2], &NdArrayDevice::default());
    let include_lemmas = Tensor::<NdArray, 2>::zeros([n_lexemes, 2], &NdArrayDevice::default());

    let pad_vector = Tensor::<NdArray, 1>::from_floats(
        [10., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        &NdArrayDevice::default(),
    );
    let end_vector = Tensor::<NdArray, 1>::from_floats(
        [0., 10., 0., 0., 0., 0., 0., 0., 0., 0.],
        &NdArrayDevice::default(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(32);

    for temperature in [0.1, 0.5, 1.0] {
        let g = GrammarParameterization::new(
            types.clone(),
            type_categories.clone(),
            licensee_categories.clone(),
            included_features.clone(),
            lemmas.clone(),
            silent_lemmas.clone(),
            categories.clone(),
            weights.clone(),
            include_lemmas.clone(),
            pad_vector.clone(),
            end_vector.clone(),
            temperature,
            false,
            &mut rng,
        )?;
        let targets = Tensor::<NdArray, 2, _>::ones([10, 10], &NdArrayDevice::default()).tril(0);
        let config = NeuralConfig {
            n_grammars: 1,
            n_strings_per_grammar: 10_000,
            padding_length: 10,
            temperature: 1.0,
            n_strings_to_sample: 5,
            negative_weight: None,
            parsing_config: ParsingConfig::new(
                LogProb::new(-256.0).unwrap(),
                LogProb::from_raw_prob(0.5).unwrap(),
                20,
                1000,
            ),
        };
        get_neural_outputs(&g, targets, &config, &mut rng)?;
    }
    Ok(())
}
