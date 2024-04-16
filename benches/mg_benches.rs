use anyhow::Result;
use bumpalo::Bump;
use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    tensor::{Data, ElementConversion, Int, Tensor},
};
use itertools::Itertools;
use lazy_static::lazy_static;
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
    // Run registered benchmarks.
    divan::main();
}

fn get_grammar() -> Lexicon<&'static str, char> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Lexicon::new(v)
}

lazy_static! {
    static ref CONFIG: ParsingConfig = ParsingConfig::new(
        LogProb::new(-256.0).unwrap(),
        LogProb::from_raw_prob(0.5).unwrap(),
        100,
        1000,
    );
}

#[divan::bench(args = [true, false])]
fn parse_long_sentence(record_rules: bool) {
    let g = divan::black_box(get_grammar());
    let sentence: Vec<&str> = divan::black_box(
        "which king knows the queen knows which beer the king drinks"
            .split(' ')
            .collect(),
    );
    divan::black_box(if record_rules {
        Parser::new
    } else {
        Parser::new_skip_rules
    })(&g, 'C', &sentence, &CONFIG)
    .unwrap()
    .next()
    .unwrap();
}

#[divan::bench(args=[true, false])]
fn generate_sentence(record_rules: bool) {
    let g = divan::black_box(get_grammar());
    divan::black_box(if record_rules {
        Generator::new
    } else {
        Generator::new_skip_rules
    })(&g, 'C', &CONFIG)
    .unwrap()
    .take(100)
    .count();
}
#[divan::bench]
fn generate_sentence_arena() {
    let g = divan::black_box(get_grammar());
    let bump = Bump::new();
    Generator::new_skip_rules_bump(&g, 'C', &CONFIG, &bump)
        .unwrap()
        .take(100)
        .count();
}

#[divan::bench(args = [true, false])]
fn parse_copy_language(record_rules: bool) {
    let (lex, strings) = divan::black_box({
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
    });

    for s in strings.iter() {
        divan::black_box(if record_rules {
            Parser::new
        } else {
            Parser::new_skip_rules
        })(&lex, 'T', s, &CONFIG)
        .unwrap()
        .next()
        .unwrap();
    }
}

#[divan::bench(args = [true, false])]
fn parse_copy_language_together(record_rules: bool) {
    let (lex, strings) = divan::black_box({
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
    });

    divan::black_box(if record_rules {
        Parser::new_multiple
    } else {
        Parser::new_skip_rules_multiple
    })(&lex, 'T', &strings, &CONFIG)
    .unwrap()
    .take(strings.len())
    .for_each(|_| ());
}

#[divan::bench(args = [true, false])]
fn generate_copy_language(record_rules: bool) {
    let lex = divan::black_box({
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        Lexicon::new(v)
    });

    divan::black_box(if record_rules {
        Generator::new
    } else {
        Generator::new_skip_rules
    })(&lex, 'T', &CONFIG)
    .unwrap()
    .take(100)
    .count();
}

#[divan::bench]
fn neural_loss() {
    let (
        types,
        type_categories,
        licensee_categories,
        included_features,
        lemmas,
        silent_lemmas,
        categories,
        weights,
        pad_vector,
        end_vector,
        targets,
        mut rng,
        config,
        include_lemmas,
        included_licensees,
    ) = divan::black_box({
        let n_lexemes = 2;
        let n_pos = 3;
        let n_categories = 2;
        let n_licensees = 1;

        let categories = Tensor::<Autodiff<NdArray>, 2>::zeros(
            [n_lexemes, n_categories],
            &NdArrayDevice::default(),
        )
        .slice_assign(
            [0..n_lexemes, 0..1],
            Tensor::full([n_lexemes, 1], 5.0, &NdArrayDevice::default()),
        );

        let mut types = Tensor::<Autodiff<NdArray>, 3>::zeros(
            [n_lexemes, n_pos, N_TYPES],
            &NdArrayDevice::default(),
        );
        let slices = [[1..2, 0..1, 1..2]];

        let dev = types.device();
        for slice in slices {
            types = types.slice_assign(
                slice.clone(),
                Tensor::full(slice.map(|x| x.len()), 10.0, &dev),
            );
        }

        let type_categories = Tensor::<Autodiff<NdArray>, 3>::full(
            [n_lexemes, n_pos, n_categories],
            5.0,
            &NdArrayDevice::default(),
        )
        .slice_assign(
            [0..n_lexemes, 0..n_pos, 1..2],
            Tensor::zeros([n_lexemes, n_pos, 1], &NdArrayDevice::default()),
        );

        let lemmas =
            Tensor::<Autodiff<NdArray>, 2>::zeros([n_lexemes, 4], &NdArrayDevice::default())
                .slice_assign(
                    [0..n_lexemes, 3..4],
                    Tensor::full([n_lexemes, 1], 30.0, &NdArrayDevice::default()),
                );
        let weights = Tensor::<Autodiff<NdArray>, 1>::zeros([n_lexemes], &NdArrayDevice::default());

        let licensee_categories = Tensor::<Autodiff<NdArray>, 3>::zeros(
            [n_lexemes, n_licensees, n_categories],
            &NdArrayDevice::default(),
        );

        let included_licensees = Tensor::<Autodiff<NdArray>, 2>::full(
            [n_lexemes, n_licensees + 1],
            -20.0,
            &NdArrayDevice::default(),
        )
        .slice_assign(
            [0..n_lexemes, 0..1],
            Tensor::<Autodiff<NdArray>, 2>::full([n_lexemes, 1], 10.0, &NdArrayDevice::default()),
        );

        let included_features = Tensor::<Autodiff<NdArray>, 2>::full(
            [n_lexemes, n_pos + 1],
            -10,
            &NdArrayDevice::default(),
        )
        .slice_assign([1..2, 1..2], Tensor::full([1, 1], 10, &dev))
        .slice_assign([0..1, 0..1], Tensor::full([1, 1], 10, &dev));

        let targets = (1..9)
            .map(|i| {
                let mut s: [u32; 10] = [0; 10];
                s.iter_mut().take(i).for_each(|x| *x = 3);
                s[i] = 1;
                Tensor::<Autodiff<NdArray>, 1, Int>::from_data(
                    Data::from(s).convert(),
                    &NdArrayDevice::default(),
                )
            })
            .collect::<Vec<_>>();

        let targets = Tensor::stack(targets, 0);
        let rng = rand::rngs::StdRng::seed_from_u64(1);

        let config = NeuralConfig {
            n_strings_per_grammar: 20,
            padding_length: 10,
            temperature: 1.0,
            compatible_weight: 0.99,
            parsing_config: ParsingConfig::new(
                LogProb::new(-200.0).unwrap(),
                LogProb::from_raw_prob(0.5).unwrap(),
                200,
                200,
            ),
        };
        let silent_lemmas =
            Tensor::<Autodiff<NdArray>, 2>::full([n_lexemes, 2], -20.0, &NdArrayDevice::default())
                .slice_assign(
                    [0..n_lexemes, 1..2],
                    Tensor::full([n_lexemes, 1], 50.0, &NdArrayDevice::default()),
                );
        let include_lemmas =
            Tensor::<Autodiff<NdArray>, 2>::full([n_lexemes, 2], -20.0, &NdArrayDevice::default())
                .slice_assign(
                    [0..n_lexemes, 1..2],
                    Tensor::full([n_lexemes, 1], 50.0, &NdArrayDevice::default()),
                );
        let pad_vector = Tensor::<Autodiff<NdArray>, 1>::from_floats(
            [50., 0., 0., 0.],
            &NdArrayDevice::default(),
        );
        let end_vector = Tensor::<Autodiff<NdArray>, 1>::from_floats(
            [0., 50., 0., 0.],
            &NdArrayDevice::default(),
        );

        (
            types,
            type_categories,
            licensee_categories,
            included_features,
            lemmas,
            silent_lemmas,
            categories,
            weights,
            pad_vector,
            end_vector,
            targets,
            rng,
            config,
            include_lemmas,
            included_licensees,
        )
    });

    for t in [0.1, 0.25, 0.5, 1.0, 5.0] {
        let g = GrammarParameterization::new(
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
            t,
            true,
            &mut rng,
        )
        .unwrap();
        get_neural_outputs(&g, targets.clone(), &config, &mut rng).unwrap();
    }
}

#[divan::bench]
fn generate_copy_language_arena() {
    let lex = divan::black_box({
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        Lexicon::new(v)
    });

    let bump = Bump::new();
    Generator::new_skip_rules_bump(&lex, 'T', &CONFIG, &bump)
        .unwrap()
        .take(100)
        .count();
}
