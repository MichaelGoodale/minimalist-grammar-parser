use crate::{
    lexicon::Feature,
    neural::loss::{get_neural_outputs, NeuralConfig},
};
use anyhow::Result;
use rand::SeedableRng;

use self::neural::neural_lexicon::GrammarParameterization;

use super::*;
use crate::{
    grammars::SIMPLESTABLER2011,
    lexicon::{LexicalEntry, Lexicon, SimpleLexicalEntry},
};
use lazy_static::lazy_static;
use std::{collections::HashSet, f64::consts::LN_2};

lazy_static! {
    static ref CONFIG: ParsingConfig = ParsingConfig::new(
        LogProb::new(-256.0).unwrap(),
        LogProb::from_raw_prob(0.5).unwrap(),
        100,
        1000,
    );
}

#[test]
fn simple_scan() -> Result<()> {
    let v = vec![SimpleLexicalEntry::parse("hello::h")?];
    let lexicon = Lexicon::new(v);
    let s: Vec<_> = vec!["hello"];
    Parser::new(&lexicon, 'h', &s, &CONFIG)?.next().unwrap();
    Ok(())
}

#[test]
fn simple_merge() -> Result<()> {
    let v = vec![
        SimpleLexicalEntry::parse("the::n= d")?,
        SimpleLexicalEntry::parse("man::n")?,
        SimpleLexicalEntry::parse("drinks::d= =d v")?,
        SimpleLexicalEntry::parse("beer::n")?,
    ];
    let lexicon = Lexicon::new(v);
    Parser::new(&lexicon, 'd', &["the", "man"], &CONFIG)?
        .next()
        .unwrap();
    Parser::new(
        &lexicon,
        'v',
        &"the man drinks the beer".split(' ').collect::<Vec<_>>(),
        &CONFIG,
    )?
    .next()
    .unwrap();
    assert!(Parser::new(
        &lexicon,
        'd',
        &"drinks the man the beer".split(' ').collect::<Vec<_>>(),
        &CONFIG
    )?
    .next()
    .is_none());
    Ok(())
}

use crate::grammars::STABLER2011;
#[test]
fn moving_parse() -> anyhow::Result<()> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()?;
    let lex = Lexicon::new(v);
    for sentence in vec![
        "the king drinks the beer",
        "which wine the queen prefers",
        "which queen prefers the wine",
        "the queen knows the king drinks the beer",
        "the queen knows the king knows the queen drinks the beer",
    ]
    .into_iter()
    {
        Parser::new(&lex, 'C', &sentence.split(' ').collect::<Vec<_>>(), &CONFIG)?
            .next()
            .unwrap();
    }

    for bad_sentence in vec![
        "the king the drinks the beer",
        "which the queen prefers the wine",
        "which queen prefers king the",
    ]
    .into_iter()
    {
        let bad_sentence: Vec<_> = bad_sentence.split(' ').collect();

        let v: Vec<_> = STABLER2011
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .filter_map(|x| {
                if let Ok(LexicalEntry { lemma, features }) = x {
                    if let Some(lemma) = lemma {
                        if bad_sentence.contains(&lemma) {
                            Some(LexicalEntry {
                                lemma: Some(lemma),
                                features,
                            })
                        } else {
                            None
                        }
                    } else {
                        Some(LexicalEntry { lemma, features })
                    }
                } else {
                    None
                }
            })
            .collect();

        let mut categories: Vec<_> = vec![];
        v.iter().for_each(|x| {
            for feature in &x.features {
                match feature {
                    Feature::Category(c) => categories.push(*c),
                    Feature::Licensee(c) => categories.push(*c),
                    _ => {}
                }
            }
        });
        let (v, _) = v.into_iter().partition(|x| {
            for feature in &x.features {
                match feature {
                    Feature::Selector(c, _) if !categories.contains(c) => {
                        return false;
                    }
                    Feature::Licensor(c) if !categories.contains(c) => {
                        return false;
                    }
                    _ => (),
                }
            }
            true
        });
        let lex = Lexicon::new(v);

        assert!(Parser::new(&lex, 'C', &bad_sentence, &CONFIG)?
            .next()
            .is_none());
    }
    Ok(())
}

#[test]
fn generation() -> Result<()> {
    let v: Vec<_> = SIMPLESTABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()?;
    let lex = Lexicon::new(v);
    let mut v: Vec<_> = Generator::new(
        &lex,
        'C',
        &ParsingConfig {
            min_log_prob: LogProb::new(-64.0).unwrap(),
            move_prob: LogProb::from_raw_prob(0.5).unwrap(),
            max_steps: 100,
            max_beams: 1000,
            global_steps: None,
            max_length: None,
        },
    )?
    .collect();

    let mut x = vec![
        (
            -2.0794415416798357,
            vec!["the", "king", "likes", "the", "queen"],
        ),
        (
            -2.0794415416798357,
            vec!["the", "king", "likes", "the", "king"],
        ),
        (
            -2.0794415416798357,
            vec!["the", "queen", "likes", "the", "king"],
        ),
        (
            -2.0794415416798357,
            vec!["the", "queen", "likes", "the", "queen"],
        ),
        (
            -2.772588722239781,
            vec!["which", "king", "likes", "the", "queen"],
        ),
        (
            -2.772588722239781,
            vec!["which", "queen", "likes", "the", "king"],
        ),
        (
            -2.772588722239781,
            vec!["which", "queen", "likes", "the", "queen"],
        ),
        (
            -2.772588722239781,
            vec!["which", "king", "likes", "the", "king"],
        ),
        (
            -3.4657359027997265,
            vec!["which", "queen", "the", "queen", "likes"],
        ),
        (
            -3.4657359027997265,
            vec!["which", "king", "the", "queen", "likes"],
        ),
        (
            -3.4657359027997265,
            vec!["which", "king", "the", "king", "likes"],
        ),
        (
            -3.4657359027997265,
            vec!["which", "queen", "the", "king", "likes"],
        ),
    ];

    assert_eq!(v.len(), x.len());
    x.sort_by(|a, b| a.1.cmp(&b.1));
    v.sort_by(|a, b| a.1.cmp(&b.1));
    let strings: Vec<_> = x.iter().map(|(_, s)| s.as_slice()).collect();
    let mut outputs: Vec<_> = Parser::new_multiple(&lex, 'C', &strings, &CONFIG)?
        .map(|(p, s, _)| (p.into_inner(), s.to_vec()))
        .collect();
    outputs.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(outputs, x);

    let mut outputs: Vec<_> = FuzzyParser::new(&lex, 'C', &strings, &CONFIG)?
        .map(|(p, s, _)| (p.into_inner(), s.to_vec()))
        .collect();
    outputs.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(outputs, x);

    for ((p, sentence, _), (correct_p, correct_sentence)) in v.into_iter().zip(x) {
        let correct_sentence = correct_sentence.into_iter().collect();
        assert_eq!((p.into_inner(), &sentence), (correct_p, &correct_sentence));
        Parser::new(&lex, 'C', &sentence, &CONFIG)?.next().unwrap();
    }
    Ok(())
}
use grammars::COPY_LANGUAGE;
use itertools::{self, Itertools};

#[test]
fn copy_language() -> anyhow::Result<()> {
    let v: Vec<_> = COPY_LANGUAGE
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()?;
    let lex = Lexicon::new(v);
    let mut strings = HashSet::<Vec<&str>>::new();
    strings.insert(vec![]);

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

    let generated: HashSet<_> = Generator::new(&lex, 'T', &CONFIG)?
        .take(strings.len())
        .map(|(_, s, _)| s)
        .collect();

    let ordered_strings: Vec<_> = strings.iter().collect();
    let generated_guided: HashSet<_> = FuzzyParser::new(&lex, 'T', &ordered_strings, &CONFIG)?
        .take(strings.len())
        .map(|(_, s, _)| s)
        .collect();

    assert_eq!(generated, strings);
    assert_eq!(generated_guided, strings);
    for s in strings.iter() {
        Parser::new(&lex, 'T', s, &CONFIG)?.next().unwrap();
    }
    let rules = Generator::new(&lex, 'T', &CONFIG)?
        .take(1)
        .last()
        .unwrap()
        .2;

    assert_eq!(
        "digraph {
    0 [ label = \"TP\" ]
    1 [ label = \"TP\" ]
    2 [ label = \"T'\" ]
    3 [ label = \"t\" ]
    4 [ label = \"ε\" ]
    5 [ label = \"T\" ]
    6 [ label = \"ε\" ]
    7 [ label = \"T\" ]
    0 -> 1 [ label = \"\" ]
    0 -> 2 [ label = \"\" ]
    2 -> 3 [ label = \"\" ]
    3 -> 1 [ label = \"\" ]
    1 -> 5 [ label = \"\" ]
    5 -> 4 [ label = \"\" ]
    2 -> 7 [ label = \"\" ]
    7 -> 6 [ label = \"\" ]
}
",
        format!(
            "{}",
            petgraph::dot::Dot::new(&tree_building::build_tree(&lex, &rules))
        )
    );
    Ok(())
}

#[test]
fn degenerate_grammar() -> Result<()> {
    let lexicon = Lexicon::new(vec![LexicalEntry::parse("a::=c c")?]);
    let x: Vec<_> = Generator::new(&lexicon, 'c', &CONFIG)?.take(50).collect();
    assert_eq!(x, vec![]);
    Ok(())
}

#[test]
fn degenerate_strings() -> Result<()> {
    let s = "a::c
b::c
d::a= c
e::c -d
f::+g c";
    let s = s
        .split('\n')
        .map(LexicalEntry::parse)
        .collect::<Result<Vec<_>>>()?;
    let lexicon = Lexicon::new(s);
    let x: Vec<_> = Generator::new(&lexicon, 'c', &CONFIG)?
        .take(50)
        .map(|(_, x, _)| x)
        .collect();
    assert_eq!(x, vec![vec!["a"], vec!["b"]]);
    Ok(())
}

#[test]
fn capped_beams() -> Result<()> {
    let lexicon = Lexicon::new(vec![
        LexicalEntry::parse("a::=b c")?,
        LexicalEntry::parse("a::=d c")?,
        LexicalEntry::parse("b::=c b")?,
        LexicalEntry::parse("d::=c d")?,
    ]);
    let max_beams = 12;
    let g: Vec<_> = Generator::new(
        &lexicon,
        'c',
        &ParsingConfig::new(
            LogProb::new(-128.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            max_beams,
        ),
    )?
    .take(50)
    .collect();
    assert_eq!(g, vec![]);
    Ok(())
}
#[test]
fn simple_movement() -> Result<()> {
    let lexicon = [
        "john::d -k",
        "will::v= +k t",
        "eat::d= d= v",
        "the::n= d",
        "cake::n",
    ];
    let lexicon = Lexicon::new(
        lexicon
            .into_iter()
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()?,
    );
    let v: Vec<_> = Generator::new(&lexicon, 't', &CONFIG)?
        .take(50)
        .map(|(_, s, _)| s)
        .collect();
    assert_eq!(v, vec![["john", "will", "eat", "the", "cake"]]);
    Parser::new(
        &lexicon,
        't',
        &["john", "will", "eat", "the", "cake"],
        &CONFIG,
    )?
    .next()
    .unwrap();
    Ok(())
}

#[test]
fn proper_distributions() -> Result<()> {
    let lexicon = ["a::1= +3 0", "a::1= 1", "ε::1 -3"];
    let lexicon = Lexicon::new(
        lexicon
            .into_iter()
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()?,
    );
    let v = vec![
        (-LN_2, vec!["a"]),
        (-1.3862943611198906, vec!["a", "a"]),
        (-2.0794415416798357, vec!["a", "a", "a"]),
        (-2.772588722239781, vec!["a", "a", "a", "a"]),
        (-3.4657359027997265, vec!["a", "a", "a", "a", "a"]),
        (-4.1588830833596715, vec!["a", "a", "a", "a", "a", "a"]),
        (-4.852030263919617, vec!["a", "a", "a", "a", "a", "a", "a"]),
        (
            -5.545177444479562,
            vec!["a", "a", "a", "a", "a", "a", "a", "a"],
        ),
    ];

    for (prob, s) in v.iter() {
        let (p, _, _) = Parser::new(
            &lexicon,
            '0',
            s,
            &ParsingConfig::new(
                LogProb::new(-128.0).unwrap(),
                LogProb::from_raw_prob(0.5).unwrap(),
                100,
                50,
            ),
        )?
        .next()
        .unwrap();
        assert_eq!(p.into_inner(), *prob);
    }

    let g: Vec<_> = Generator::new(
        &lexicon,
        '0',
        &ParsingConfig::new(
            LogProb::new(-128.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            50,
        ),
    )?
    .take(8)
    .map(|(p, s, _)| (p.into_inner(), s))
    .collect();
    let generated_sentences: Vec<_> = v.iter().map(|(_, s)| s).collect();

    let parse: Vec<_> = Parser::new_multiple(
        &lexicon,
        '0',
        &generated_sentences,
        &ParsingConfig::new(
            LogProb::new(-128.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            50,
        ),
    )?
    .map(|(p, s, _)| (p.into_inner(), s.to_vec()))
    .collect();
    assert_eq!(v, g);
    assert_eq!(v, parse);
    Ok(())
}

use burn::tensor::{Data, ElementConversion, Int, Tensor};
use burn::{
    backend::Autodiff,
    backend::{ndarray::NdArrayDevice, NdArray},
};
use neural::N_TYPES;

#[test]
fn test_loss() -> Result<()> {
    let n_lexemes = 2;
    let n_pos = 3;
    let n_categories = 2;
    let n_licensees = 1;

    let categories =
        Tensor::<Autodiff<NdArray>, 2>::ones([n_lexemes, n_categories], &NdArrayDevice::default());

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

    let type_categories = Tensor::<Autodiff<NdArray>, 3>::ones(
        [n_lexemes, n_pos, n_categories],
        &NdArrayDevice::default(),
    );

    let lemmas = Tensor::<Autodiff<NdArray>, 2>::zeros([n_lexemes, 4], &NdArrayDevice::default())
        .slice_assign(
            [0..n_lexemes, 3..4],
            Tensor::full([n_lexemes, 1], 10.0, &NdArrayDevice::default()),
        );
    let weights = Tensor::<Autodiff<NdArray>, 1>::zeros([n_lexemes], &NdArrayDevice::default());

    let licensee_categories = Tensor::<Autodiff<NdArray>, 3>::zeros(
        [n_lexemes, n_licensees, n_categories],
        &NdArrayDevice::default(),
    );

    let included_features = Tensor::<Autodiff<NdArray>, 2>::full(
        [n_lexemes, n_licensees + n_pos],
        -10,
        &NdArrayDevice::default(),
    )
    .slice_assign(
        [1..2, n_licensees..n_licensees + 1],
        Tensor::full([1, 1], 10, &dev),
    );

    let targets = (1..6)
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

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let config = NeuralConfig {
        n_grammars: 1,
        n_strings_per_grammar: 100,
        padding_length: 10,
        n_strings_to_sample: 5,
        temperature: 1.0,
        negative_weight: None,
        parsing_config: ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            200,
            200,
        ),
    };
    let silent_lemmas =
        Tensor::<Autodiff<NdArray>, 1>::full([n_lexemes], -10.0, &NdArrayDevice::default());
    let pad_vector =
        Tensor::<Autodiff<NdArray>, 1>::from_floats([10., 0., 0., 0.], &NdArrayDevice::default());
    let end_vector =
        Tensor::<Autodiff<NdArray>, 1>::from_floats([0., 10., 0., 0.], &NdArrayDevice::default());
    let mut loss: Vec<f32> = vec![];
    for t in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0] {
        let mut avg = 0.0;
        for _ in 0..3 {
            let g = GrammarParameterization::new(
                types.clone(),
                type_categories.clone(),
                licensee_categories.clone(),
                included_features.clone(),
                lemmas.clone(),
                silent_lemmas.clone(),
                categories.clone(),
                weights.clone(),
                pad_vector.clone(),
                end_vector.clone(),
                t,
                &mut rng,
            )?;
            avg += get_neural_outputs(&g, targets.clone(), &config, &mut rng)?
                .into_scalar()
                .elem::<f32>();
        }
        loss.push(avg / 3.0);
    }

    let stored_losses = [
        95.489075, 25.114962, 41.96997, 20.512835, 21.748817, 28.503601,
    ];
    dbg!(&loss);
    for (loss, stored_loss) in loss.into_iter().zip(stored_losses) {
        approx::assert_relative_eq!(loss, stored_loss, epsilon = 1e-5);
    }
    Ok(())
}

#[test]
fn random_neural_generation() -> Result<()> {
    let n_lexemes = 1;
    let n_pos = 2;
    let n_licensee = 2;
    let n_categories = 2;
    let n_lemmas = 10;
    let lemmas = Tensor::<NdArray, 2>::zeros([n_lexemes, n_lemmas], &NdArrayDevice::default());

    let types = Tensor::<NdArray, 3>::zeros([n_lexemes, n_pos, N_TYPES], &NdArrayDevice::default());

    let type_categories =
        Tensor::<NdArray, 3>::zeros([n_lexemes, n_pos, n_categories], &NdArrayDevice::default());

    let licensee_categories = Tensor::<NdArray, 3>::zeros(
        [n_lexemes, n_licensee, n_categories],
        &NdArrayDevice::default(),
    );
    let included_features =
        Tensor::<NdArray, 2>::zeros([n_lexemes, n_licensee + n_pos], &NdArrayDevice::default());

    let categories =
        Tensor::<NdArray, 2>::zeros([n_lexemes, n_categories], &NdArrayDevice::default());
    let weights = Tensor::<NdArray, 1>::zeros([n_lexemes], &NdArrayDevice::default());

    let silent_lemmas = Tensor::<NdArray, 1>::zeros([n_lexemes], &NdArrayDevice::default());

    let pad_vector = Tensor::<NdArray, 1>::from_floats(
        [10., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        &NdArrayDevice::default(),
    );
    let end_vector = Tensor::<NdArray, 1>::from_floats(
        [10., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
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
            pad_vector.clone(),
            end_vector.clone(),
            temperature,
            &mut rng,
        )?;
        let targets = Tensor::<NdArray, 2, _>::ones([10, 10], &NdArrayDevice::default()).tril(0);
        let config = NeuralConfig {
            n_grammars: 1,
            n_strings_per_grammar: 20,
            padding_length: 10,
            temperature: 1.0,
            n_strings_to_sample: 5,
            negative_weight: None,
            parsing_config: ParsingConfig::new_with_global_steps(
                LogProb::new(-256.0).unwrap(),
                LogProb::from_raw_prob(0.5).unwrap(),
                500,
                20,
                5000,
            ),
        };
        get_neural_outputs(&g, targets, &config, &mut rng)?;
    }
    Ok(())
}
