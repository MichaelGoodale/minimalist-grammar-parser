use anyhow::Result;
use bumpalo::Bump;
use itertools::Itertools;
use lazy_static::lazy_static;
use logprob::LogProb;
use minimalist_grammar_parser::{
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{Lexicon, SimpleLexicalEntry},
    Generator, Parser, ParsingConfig,
};

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
    .take(2000)
    .count();
}
#[divan::bench]
fn generate_sentence_arena() {
    let g = divan::black_box(get_grammar());
    let bump = Bump::new();
    Generator::new_skip_rules_bump(&g, 'C', &CONFIG, &bump)
        .unwrap()
        .take(2000)
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
