use itertools::Itertools;
use lazy_static::lazy_static;
use logprob::LogProb;
use minimalist_grammar_parser::{
    ParsingConfig,
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{Lexicon, SimpleLexicalEntry},
};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

fn get_grammar() -> Lexicon<&'static str, &'static str> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>, _>>()
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

#[divan::bench]
fn parse_long_sentence() {
    let g = divan::black_box(get_grammar());
    let sentence: Vec<&str> = divan::black_box(
        "which king knows the queen knows which beer the king drinks"
            .split(' ')
            .collect(),
    );
    g.parse(&sentence, "C", &CONFIG).unwrap().next().unwrap();
}

#[divan::bench]
fn generate_sentence() {
    let g = divan::black_box(get_grammar());
    g.generate("C", &CONFIG).unwrap().take(100).count();
}

#[divan::bench]
fn parse_copy_language() {
    let (lex, strings) = divan::black_box({
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>, _>>()
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
        lex.parse(s, "T", &CONFIG).unwrap().next().unwrap();
    }
}

#[divan::bench]
fn parse_copy_language_together() {
    let (lex, strings) = divan::black_box({
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>, _>>()
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

    lex.parse_multiple(&strings, "T", &CONFIG)
        .unwrap()
        .take(strings.len())
        .for_each(|_| ());
}

#[divan::bench]
fn generate_copy_language() {
    let lex = divan::black_box({
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(SimpleLexicalEntry::parse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        Lexicon::new(v)
    });

    lex.generate("T", &CONFIG).unwrap().take(100).count();
}
