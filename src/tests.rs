use crate::lexicon::Feature;
use anyhow::Result;

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

use burn::backend::{ndarray::NdArrayDevice, NdArray};
use burn::tensor::Tensor;
use neural_lexicon::N_TYPES;
#[test]
fn neural_generation() -> Result<()> {
    let lemmas = Tensor::<NdArray, 3>::random(
        [3, 3, 3],
        burn::tensor::Distribution::Default,
        &NdArrayDevice::default(),
    );
    let types = Tensor::<NdArray, 3>::random(
        [3, 3, N_TYPES],
        burn::tensor::Distribution::Default,
        &NdArrayDevice::default(),
    );
    let categories = Tensor::<NdArray, 3>::random(
        [3, 3, 2],
        burn::tensor::Distribution::Default,
        &NdArrayDevice::default(),
    );
    let lexeme_weights = Tensor::<NdArray, 1>::random(
        [3],
        burn::tensor::Distribution::Default,
        &NdArrayDevice::default(),
    );
    let lexicon = NeuralLexicon::new(types, lexeme_weights, lemmas, categories);

    let x: Vec<_> = NeuralGenerator::new(
        &lexicon,
        &ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            100,
        ),
    )
    .take(50)
    .collect();
    println!("{x:?}");
    panic!();
}
