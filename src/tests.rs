use crate::lexicon::{Feature, LexiconParsingError};
use anyhow::Result;

use super::*;
use crate::{
    grammars::SIMPLESTABLER2011,
    lexicon::{LexicalEntry, Lexicon},
};
use lazy_static::lazy_static;
use std::{collections::HashSet, f64::consts::LN_2};

type SimpleLexicalEntry<'a> = LexicalEntry<&'a str, &'a str>;

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
    let s: Vec<_> = PhonContent::new(vec!["hello"]);
    lexicon.parse(&s, "h", &CONFIG)?.next().unwrap();
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
    lexicon
        .parse(&PhonContent::from(["the", "man"]), "d", &CONFIG)?
        .next()
        .unwrap();
    lexicon
        .parse(
            &"the man drinks the beer"
                .split(' ')
                .map(PhonContent::Normal)
                .collect::<Vec<_>>(),
            "v",
            &CONFIG,
        )?
        .next()
        .unwrap();
    assert!(
        lexicon
            .parse(
                &"drinks the man the beer"
                    .split(' ')
                    .map(PhonContent::Normal)
                    .collect::<Vec<_>>(),
                "d",
                &CONFIG
            )?
            .next()
            .is_none()
    );
    Ok(())
}

use crate::grammars::STABLER2011;
#[test]
fn moving_parse() -> anyhow::Result<()> {
    let lex = Lexicon::from_string(STABLER2011)?;

    for sentence in vec![
        "the king drinks the beer",
        "which wine the queen prefers",
        "the queen knows the king drinks the beer",
        "the queen knows the king knows the queen drinks the beer",
        "which king knows the queen knows which beer the king drinks",
        "which queen prefers the wine",
    ]
    .into_iter()
    {
        println!("{sentence}");
        lex.parse(
            &sentence
                .split(' ')
                .map(PhonContent::Normal)
                .collect::<Vec<_>>(),
            "C",
            &CONFIG,
        )?
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

        assert!(
            lex.parse(&PhonContent::new(bad_sentence), "C", &CONFIG)?
                .next()
                .is_none()
        );
    }
    Ok(())
}

#[test]
fn generation() -> Result<()> {
    let lex = Lexicon::from_string(SIMPLESTABLER2011)?;
    dbg!(&lex);
    let mut v: Vec<_> = lex
        .generate(
            "C",
            &ParsingConfig {
                min_log_prob: LogProb::new(-64.0).unwrap(),
                move_prob: LogProb::from_raw_prob(0.5).unwrap(),
                dont_move_prob: LogProb::from_raw_prob(0.5).unwrap(),
                max_steps: 100,
                max_beams: 1000,
                max_time: None,
            },
        )?
        .collect();

    let mut x = vec![
        (
            -2.0794415416798357,
            PhonContent::new(vec!["the", "king", "likes", "the", "queen"]),
        ),
        (
            -2.0794415416798357,
            PhonContent::new(vec!["the", "king", "likes", "the", "king"]),
        ),
        (
            -2.0794415416798357,
            PhonContent::new(vec!["the", "queen", "likes", "the", "king"]),
        ),
        (
            -2.0794415416798357,
            PhonContent::new(vec!["the", "queen", "likes", "the", "queen"]),
        ),
        (
            -2.772588722239781,
            PhonContent::new(vec!["which", "king", "likes", "the", "queen"]),
        ),
        (
            -2.772588722239781,
            PhonContent::new(vec!["which", "queen", "likes", "the", "king"]),
        ),
        (
            -2.772588722239781,
            PhonContent::new(vec!["which", "queen", "likes", "the", "queen"]),
        ),
        (
            -2.772588722239781,
            PhonContent::new(vec!["which", "king", "likes", "the", "king"]),
        ),
        (
            -3.4657359027997265,
            PhonContent::new(vec!["which", "queen", "the", "queen", "likes"]),
        ),
        (
            -3.4657359027997265,
            PhonContent::new(vec!["which", "king", "the", "queen", "likes"]),
        ),
        (
            -3.4657359027997265,
            PhonContent::new(vec!["which", "king", "the", "king", "likes"]),
        ),
        (
            -3.4657359027997265,
            PhonContent::new(vec!["which", "queen", "the", "king", "likes"]),
        ),
    ];

    dbg!(&v);
    assert_eq!(v.len(), x.len());
    x.sort_by(|a, b| a.1.cmp(&b.1));
    v.sort_by(|a, b| a.1.cmp(&b.1));
    let strings: Vec<_> = x.iter().map(|(_, s)| s.as_slice()).collect();
    let mut outputs: Vec<_> = lex
        .parse_multiple(&strings, "C", &CONFIG)?
        .map(|(p, s, _)| (p.into_inner(), s.to_vec()))
        .collect();
    outputs.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(outputs, x);

    let mut outputs: Vec<_> = lex
        .fuzzy_parse(&strings, "C", &CONFIG)?
        .map(|(p, s, _)| (p.into_inner(), s.to_vec()))
        .collect();
    outputs.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(outputs, x);

    for ((p, sentence, _), (correct_p, correct_sentence)) in v.into_iter().zip(x) {
        let correct_sentence = correct_sentence.into_iter().collect();
        assert_eq!((p.into_inner(), &sentence), (correct_p, &correct_sentence));
        lex.parse(&sentence, "C", &CONFIG)?.next().unwrap();
    }
    Ok(())
}
use grammars::{ALT_COPY_LANGUAGE, COPY_LANGUAGE};
use itertools::{self, Itertools};

#[test]
fn clear_copy_language() -> anyhow::Result<()> {
    let lex = Lexicon::from_string(ALT_COPY_LANGUAGE)?;
    let mut strings = HashSet::<Vec<_>>::new();
    strings.insert(PhonContent::new(vec!["S", "E"]));

    for i in 1..=5 {
        strings.extend(
            itertools::repeat_n(vec!["a", "b"].into_iter(), i)
                .multi_cartesian_product()
                .map(|mut x| {
                    x.append(&mut x.clone());
                    x.insert(0, "S");
                    x.push("E");
                    PhonContent::new(x)
                }),
        );
    }

    let generated: HashSet<_> = lex
        .generate("T", &CONFIG)?
        .take(strings.len())
        .map(|(_, s, _)| s)
        .collect();

    assert_eq!(generated, strings);
    Ok(())
}

#[test]
fn copy_language() -> anyhow::Result<()> {
    let lex = Lexicon::from_string(COPY_LANGUAGE)?;
    let mut strings = HashSet::<Vec<_>>::new();
    strings.insert(vec![]);

    for i in 1..=5 {
        strings.extend(
            itertools::repeat_n(vec!["a", "b"].into_iter(), i)
                .multi_cartesian_product()
                .map(|mut x| {
                    x.append(&mut x.clone());
                    PhonContent::new(x)
                }),
        );
    }

    let generated: HashSet<_> = lex
        .generate("T", &CONFIG)?
        .take(strings.len())
        .map(|(_, s, _)| s)
        .collect();

    let ordered_strings: Vec<_> = strings.iter().collect();
    let generated_guided: HashSet<_> = lex
        .fuzzy_parse(&ordered_strings, "T", &CONFIG)?
        .take(strings.len())
        .map(|(_, s, _)| s)
        .collect();

    assert_eq!(generated, strings);
    assert_eq!(generated_guided, strings);
    for s in strings.iter() {
        lex.parse(s, "T", &CONFIG)?.next().unwrap();
    }
    Ok(())
}

#[test]
fn degenerate_grammar() -> Result<()> {
    let lexicon = Lexicon::from_string("a::=c c")?;
    let x: Vec<_> = lexicon.generate("c", &CONFIG)?.take(50).collect();
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
    let lexicon = Lexicon::from_string(s)?;
    let x: Vec<_> = lexicon
        .generate("c", &CONFIG)?
        .take(50)
        .map(|(_, x, _)| x)
        .collect();
    assert_eq!(
        x,
        vec![
            vec![PhonContent::Normal("a")],
            vec![PhonContent::Normal("b")]
        ]
    );
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
    let g: Vec<_> = lexicon
        .generate(
            "c",
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
        "eat::d= =d v",
        "the::n= d",
        "cake::n",
    ];
    let lexicon = Lexicon::new(
        lexicon
            .into_iter()
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>, LexiconParsingError>>()?,
    );
    let v: Vec<_> = lexicon
        .generate("t", &CONFIG)?
        .take(50)
        .map(|(_, s, _)| s)
        .collect();

    assert_eq!(
        v,
        vec![
            PhonContent::from(["john", "will", "eat", "the", "cake"]),
            PhonContent::from(["john", "will", "the", "cake", "eat"])
        ]
    );
    lexicon
        .parse(
            &PhonContent::from(["john", "will", "eat", "the", "cake"]),
            "t",
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
            .collect::<Result<Vec<_>, LexiconParsingError>>()?,
    );
    let v = vec![
        (-LN_2, PhonContent::new(vec!["a"])),
        (-1.3862943611198906, PhonContent::new(vec!["a", "a"])),
        (-2.0794415416798357, PhonContent::new(vec!["a", "a", "a"])),
        (
            -2.772588722239781,
            PhonContent::new(vec!["a", "a", "a", "a"]),
        ),
        (
            -3.4657359027997265,
            PhonContent::new(vec!["a", "a", "a", "a", "a"]),
        ),
        (
            -4.1588830833596715,
            PhonContent::new(vec!["a", "a", "a", "a", "a", "a"]),
        ),
        (
            -4.852030263919617,
            PhonContent::new(vec!["a", "a", "a", "a", "a", "a", "a"]),
        ),
        (
            -5.545177444479562,
            PhonContent::new(vec!["a", "a", "a", "a", "a", "a", "a", "a"]),
        ),
    ];

    for (prob, s) in v.iter() {
        let (p, _, _) = lexicon
            .parse(
                s,
                "0",
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

    let g: Vec<_> = lexicon
        .generate(
            "0",
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

    let parse: Vec<_> = lexicon
        .parse_multiple(
            &generated_sentences,
            "0",
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

#[test]
fn simple_head_movement() -> Result<()> {
    let lexicon = [
        "john::d -k",
        "is::prog= +k t",
        "laugh::=d v",
        "ing::=>v prog",
    ];

    let lexicon = Lexicon::new(
        lexicon
            .into_iter()
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>, LexiconParsingError>>()?,
    );
    let v: Vec<_> = lexicon
        .generate("t", &CONFIG)?
        .take(50)
        .map(|(_, s, _)| s)
        .collect();

    assert_eq!(
        v,
        vec![
            PhonContent::from(["john", "is ", "eat", "the", "cake"]),
            PhonContent::from(["john", "will", "the", "cake", "eat"])
        ]
    );
    lexicon
        .parse(
            &[
                PhonContent::Normal("john"),
                PhonContent::Normal("is"),
                PhonContent::Affixed(vec!["laugh", "ing"]),
            ],
            "t",
            &CONFIG,
        )?
        .next()
        .unwrap();
    Ok(())
}
