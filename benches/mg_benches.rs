use itertools::Itertools;
use lazy_static::lazy_static;
use logprob::LogProb;
use minimalist_grammar_parser::{
    ParsingConfig, PhonContent,
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{LexicalEntry, Lexicon},
};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

fn get_grammar() -> Lexicon<&'static str, &'static str> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(LexicalEntry::parse)
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
fn parse_long_sentence(bencher: divan::Bencher) {
    let g = get_grammar();
    let sentence: Vec<_> = "which king knows the queen knows which beer the king drinks"
        .split(' ')
        .map(PhonContent::Normal)
        .collect();
    bencher.bench(|| g.parse(&sentence, "C", &CONFIG).unwrap().next().unwrap());
}

#[divan::bench]
fn generate_sentence(bencher: divan::Bencher) {
    let g = get_grammar();
    bencher.bench(|| g.generate("C", &CONFIG).unwrap().take(100).count());
}

#[divan::bench]
fn parse_copy_language(bencher: divan::Bencher) {
    let (lex, strings) = {
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(LexicalEntry::parse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let lex = Lexicon::new(v);

        let mut strings = Vec::<Vec<_>>::new();
        strings.push(vec![]);

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
        (lex, strings)
    };

    bencher.bench(|| {
        for s in strings.iter() {
            lex.parse(s, "T", &CONFIG).unwrap().next().unwrap();
        }
    });
}

#[divan::bench]
fn parse_copy_language_together(bencher: divan::Bencher) {
    let (lex, strings) = {
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(LexicalEntry::parse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let lex = Lexicon::new(v);

        let mut strings = Vec::<Vec<_>>::new();
        strings.push(vec![]);

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
        (lex, strings)
    };

    bencher.bench(|| {
        lex.parse_multiple(&strings, "T", &CONFIG)
            .unwrap()
            .take(strings.len())
            .for_each(|_| ());
    });
}

#[divan::bench]
fn generate_copy_language(bencher: divan::Bencher) {
    let lex = {
        let v: Vec<_> = COPY_LANGUAGE
            .split('\n')
            .map(LexicalEntry::parse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        Lexicon::new(v)
    };

    bencher.bench(|| {
        lex.generate("T", &CONFIG).unwrap().take(100).count();
    });
}
