use ahash::HashMap;
use itertools::Itertools;
use lazy_static::lazy_static;
use logprob::LogProb;
use minimalist_grammar_parser::{
    ParsingConfig, PhonContent,
    grammars::{COPY_LANGUAGE, STABLER2011},
    lexicon::{LexicalEntry, Lexicon, SemanticLexicon},
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

#[divan::bench]
fn semantic_bench(bencher: divan::Bencher) {
    let bad_grammar="ε::0= 0::lambda t phi pa_man(a_John) | (~every_e(x, pe_likes, PatientOf(a_Phil, x)) & phi)
ε::0= 0::lambda t phi every_e(x, pe_laughs, phi) & some(x, pa_man, (every(y, pa_woman, some(z, all_a, some(a, lambda a b some_e(c, pe_laughs, pe_laughs(c)), some(b, pa_woman, some_e(c, lambda e d pe_helps(c), PatientOf(b, c)) | ~(pa_man(a) | every(c, all_a, pa_woman(x))))))) & pa_woman(a_Phil) & some(y, all_a, every_e(z, pe_runs, every_e(a, pe_runs, PatientOf(a_Mary, a))))) | every(y, all_a, phi)) & some(x, all_a, some(y, all_a, some(z, lambda a a phi, some(a, all_a, every_e(b, pe_likes, pa_woman(x)))) & phi))
laughs::0::(pa_woman(a_Sue) | every_e(x, pe_runs, ~pa_woman(a_Mary))) & pa_woman(a_Sue)
sleeps::0::pa_woman(a_Phil) | ((every(x, lambda a y every_e(z, pe_likes, ~~pe_runs(z)), some(y, lambda a z pa_woman(y) & every_e(a, all_e, pe_runs(a)), ~(pa_woman(a_John) & ~every(z, all_a, ~((pa_man(z) & some_e(a, pe_runs, AgentOf(y, a)) & (pa_man(y) | ~pa_man(a_Phil) | ~pa_woman(x) | pa_man(a_Mary)) & (pa_man(a_John) | every_e(a, pe_helps, pe_loves(a))) & pa_man(a_John) & every(a, pa_woman, pa_man(a_Sue))) | (~some_e(a, all_e, pe_sleeps(a)) & every_e(a, pe_runs, some(b, all_a, PatientOf(a_Mary, a)) | pe_loves(a) | pe_loves(a) | pe_laughs(a)) & some_e(a, pe_sleeps, pe_loves(a)))))))) | ~~(~pa_man(a_John) | pa_woman(a_Mary) | pa_woman(a_Sue))) & pa_man(a_Sue) & ~every_e(x, pe_helps, (~some_e(y, pe_loves, some(z, pa_woman, pe_loves(x))) & pa_man(a_Mary) & pa_man(a_Phil)) | PatientOf(a_John, x)) & (every_e(x, lambda e y AgentOf(a_John, y), pe_sleeps(x)) | some(x, all_a, ~some_e(y, pe_likes, pe_sleeps(y) | pe_runs(y)))))
Mary::0= 0::lambda t phi phi";

    let fast_grammar = "ε::0= 0::lambda t phi (~every_e(x, pe_likes, PatientOf(a_Phil, x)) & phi)
ε::0= 0::lambda t phi every_e(x, pe_laughs, phi)
laughs::0::every_e(x, pe_runs, ~pa_woman(a_Mary)) & pa_woman(a_Sue)
sleeps::0::pa_woman(a_Phil)
Mary::0= 0::lambda t phi phi";

    //fast grammar is 1000x times faster. literally 1000x.

    let semantics = SemanticLexicon::parse(bad_grammar).unwrap();

    let config = ParsingConfig::new(
        LogProb::new(-32.0).unwrap(),
        LogProb::from_raw_prob(0.5).unwrap(),
        20,
        50,
    );

    let sentences = [
        vec!["Mary", "laughs"],
        vec!["Mary", "sleeps"],
        vec!["John", "sleeps"],
        vec!["John", "helps Mary"],
        vec!["Mary", "helps John"],
    ]
    .map(PhonContent::new);

    bencher.bench(|| {
        if let Ok(parser) = semantics.lexicon().parse_multiple(&sentences, "0", &config) {
            let mut map = HashMap::default();
            for (p, s, r) in parser.take(20) {
                let parses = map.entry(s).or_insert((p, vec![]));
                let interp = r
                    .to_interpretation(&semantics)
                    .take(10)
                    .filter_map(|(pool, h)| {
                        if let Ok(expr) = pool.into_pool() {
                            Some((expr, h.into_rich(&semantics, &r)))
                        } else {
                            None
                        }
                    });
                parses.1.extend(interp);
            }
        }
    });
}
