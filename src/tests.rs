use crate::lexicon::Feature;
use anyhow::Result;

use crate::{
    grammars::SIMPLESTABLER2011,
    lexicon::{LexicalEntry, Lexicon, SimpleLexicalEntry},
};
use std::f64::consts::LN_2;

use super::*;

const CONFIG: ParsingConfig = ParsingConfig {
    min_log_prob: -64.0,
    merge_log_prob: -LN_2,
    move_log_prob: -LN_2,
    max_parses: 1000,
};

#[test]
fn simple_scan() -> Result<()> {
    let v = vec![SimpleLexicalEntry::parse("hello::h")?];
    let lexicon = Lexicon::new(v);
    parse(&lexicon, 'h', vec!["hello".to_string()], &CONFIG)
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
    parse(
        &lexicon,
        'd',
        vec!["the".to_string(), "man".to_string()],
        &CONFIG,
    )?;
    parse(
        &lexicon,
        'v',
        "the man drinks the beer"
            .split(' ')
            .map(|x| x.to_string())
            .collect(),
        &CONFIG,
    )?;
    assert!(parse(
        &lexicon,
        'd',
        "drinks the man the beer"
            .split(' ')
            .map(|x| x.to_string())
            .collect(),
        &CONFIG
    )
    .is_err());
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
        parse(
            &lex,
            'C',
            sentence.split(' ').map(|x| x.to_string()).collect(),
            &CONFIG,
        )?;
    }

    for bad_sentence in vec![
        "the king the drinks the beer",
        "which the queen prefers the wine",
        "which queen prefers king the",
    ]
    .into_iter()
    {
        let bad_sentence: Vec<_> = bad_sentence.split(' ').map(|x| x.to_string()).collect();

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

        assert!(parse(&lex, 'C', bad_sentence, &CONFIG).is_err());
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
    let v = generate(&lex, 'C', &CONFIG);

    let x = vec![
        (
            -2.0794415416798357,
            vec!["the", "queen", "likes", "the", "king"],
        ),
        (
            -2.0794415416798357,
            vec!["the", "queen", "likes", "the", "queen"],
        ),
        (
            -2.0794415416798357,
            vec!["the", "king", "likes", "the", "king"],
        ),
        (
            -2.0794415416798357,
            vec!["the", "king", "likes", "the", "queen"],
        ),
        (
            -2.772588722239781,
            vec!["which", "queen", "likes", "the", "king"],
        ),
        (
            -2.772588722239781,
            vec!["which", "king", "likes", "the", "king"],
        ),
        (
            -2.772588722239781,
            vec!["which", "king", "likes", "the", "queen"],
        ),
        (
            -2.772588722239781,
            vec!["which", "queen", "likes", "the", "queen"],
        ),
        (
            -2.772588722239781,
            vec!["which", "king", "the", "king", "likes"],
        ),
        (
            -2.772588722239781,
            vec!["which", "king", "the", "queen", "likes"],
        ),
        (
            -2.772588722239781,
            vec!["which", "queen", "the", "queen", "likes"],
        ),
        (
            -2.772588722239781,
            vec!["which", "queen", "the", "king", "likes"],
        ),
    ];

    for ((p, sentence), (correct_p, correct_sentence)) in v.into_iter().zip(x) {
        let correct_sentence = correct_sentence
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        assert_eq!((p, &sentence), (correct_p, &correct_sentence));
        parse(&lex, 'C', sentence, &CONFIG)?;
    }
    Ok(())
}
