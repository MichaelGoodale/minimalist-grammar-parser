use anyhow::Result;
use itertools::Itertools;
use logprob::LogProb;
use minimalist_grammar_parser::{
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{Lexicon, SimpleLexicalEntry},
    Generator, Parser, ParsingConfig,
};

fn main() {
    let config: ParsingConfig = ParsingConfig::new(
        LogProb::new(-256.0).unwrap(),
        LogProb::from_raw_prob(0.5).unwrap(),
        100,
        1000,
    );
    for _ in 0..10000 {
        parse_long_sentence(&config);
        parse_copy_language(&config);
        generate_sentence(&config);
        generate_copy_language(&config);
        parse_copy_language_together(&config);
    }
    println!("DONE");
}

fn get_grammar() -> Lexicon<&'static str, &'static str> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Lexicon::new(v)
}

fn parse_copy_language_together(config: &ParsingConfig) {
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

    Parser::new_multiple(&lex, "T", &strings, config)
        .unwrap()
        .take(strings.len())
        .for_each(|_| ());
}

fn parse_long_sentence(config: &ParsingConfig) {
    let g = get_grammar();
    let sentence: Vec<&str> = "which king knows the queen knows which beer the king drinks"
        .split(' ')
        .collect();

    Parser::new(&g, "C", &sentence, config)
        .unwrap()
        .next()
        .unwrap();
}

fn generate_sentence(config: &ParsingConfig) {
    let g = get_grammar();

    Generator::new(&g, "C", config).unwrap().take(100).count();
}

fn parse_copy_language(config: &ParsingConfig) {
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
        Parser::new(&lex, "T", s, config).unwrap().next().unwrap();
    }
}

fn generate_copy_language(config: &ParsingConfig) {
    let v: Vec<_> = COPY_LANGUAGE
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let lex = Lexicon::new(v);
    Generator::new(&lex, "T", config).unwrap().take(100).count();
}
