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
        //parse_long_sentence(&config, false);
        //parse_copy_language(&config, false);
        //generate_sentence(&config, false);
        //generate_copy_language(&config, false);
        parse_copy_language_together(&config, false);
    }
    println!("DONE");
}

fn get_grammar() -> Lexicon<&'static str, char> {
    let v: Vec<_> = STABLER2011
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Lexicon::new(v)
}

fn parse_copy_language_together(config: &ParsingConfig, record_rules: bool) {
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

    (if record_rules {
        Parser::new_multiple
    } else {
        Parser::new_skip_rules_multiple
    })(&lex, 'T', &strings, config)
    .unwrap()
    .take(strings.len())
    .for_each(|_| ());
}

fn parse_long_sentence(config: &ParsingConfig, record_rules: bool) {
    let g = get_grammar();
    let sentence: Vec<&str> = "which king knows the queen knows which beer the king drinks"
        .split(' ')
        .collect();

    (if record_rules {
        Parser::new
    } else {
        Parser::new_skip_rules
    })(&g, 'C', &sentence, config)
    .unwrap()
    .next()
    .unwrap();
}

fn generate_sentence(config: &ParsingConfig, record_rules: bool) {
    let g = get_grammar();
    (if record_rules {
        Generator::new
    } else {
        Generator::new_skip_rules
    })(&g, 'C', config)
    .unwrap()
    .take(100)
    .count();
}

fn parse_copy_language(config: &ParsingConfig, record_rules: bool) {
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
        (if record_rules {
            Parser::new
        } else {
            Parser::new_skip_rules
        })(&lex, 'T', s, config)
        .unwrap()
        .next()
        .unwrap();
    }
}

fn generate_copy_language(config: &ParsingConfig, record_rules: bool) {
    let v: Vec<_> = COPY_LANGUAGE
        .split('\n')
        .map(SimpleLexicalEntry::parse)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let lex = Lexicon::new(v);
    (if record_rules {
        Generator::new
    } else {
        Generator::new_skip_rules
    })(&lex, 'T', config)
    .unwrap()
    .take(100)
    .count();
}
