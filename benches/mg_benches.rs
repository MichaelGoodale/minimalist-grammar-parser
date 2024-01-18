fn main() {
    // Run registered benchmarks.
    divan::main();
}

use anyhow::Result;
use itertools::Itertools;
use minimalist_grammar_parser::{
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{Lexicon, SimpleLexicalEntry},
    Generator, Parser, ParsingConfig,
};

fn get_grammar() -> Lexicon<&'static str, char> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Lexicon::new(v)
}

const CONFIG: ParsingConfig = ParsingConfig {
    min_log_prob: -256.0,
    move_prob: 0.5,
    max_steps: 100,
    max_beams: 1000,
};

#[divan::bench]
fn parse_long_sentence() {
    let g = divan::black_box(get_grammar());
    let sentence: Vec<&str> = divan::black_box(
        "which king knows the queen knows which beer the king drinks"
            .split(' ')
            .collect(),
    );
    Parser::new(&g, 'C', &sentence, &CONFIG)
        .unwrap()
        .next()
        .unwrap();
}

#[divan::bench]
fn generate_sentence() {
    let g = divan::black_box(get_grammar());
    let sentence: Vec<&str> = divan::black_box(
        "which king knows the queen knows which beer the king drinks"
            .split(' ')
            .collect(),
    );
    Generator::new(&g, 'C', &CONFIG).unwrap().take(100).count();
}

#[divan::bench]
fn parse_copy_language() {
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
        Parser::new(&lex, 'T', s, &CONFIG).unwrap().next().unwrap();
    }
}

#[divan::bench]
fn generate_copy_language() {
    let lex = divan::black_box({
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        Lexicon::new(v)
    });

    Generator::new(&lex, 'T', &CONFIG)
        .unwrap()
        .take(100)
        .count();
}
