//!Module which defines how the different beams used by [`Lexicon::parse`] or [`Lexicon::generate`]
//!work.

use crate::{
    ParseHeap, ParsingConfig, PhonContent, expand,
    lexicon::{Lexicon, ParsingError},
    parsing::FilledHeadTree,
};

use super::{BeamWrapper, RuleHolder, rules::RulePool};
use ahash::HashSet;
use logprob::LogProb;
use std::{fmt::Debug, hash::Hash};

///A trait which allows a struct to be used as by the parsing algorithm by defining how scanning
///works. Parsing checks the next string corresponds to a parse, whereas generation uses scan to
///iteratively build strings.
pub(crate) trait Scanner<T>: Sized {
    fn scan(&mut self, s: &Option<T>) -> bool;

    fn multiscan(&mut self, heads: &FilledHeadTree<T>) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParseScan<'a, T> {
    pub sentence: Vec<(&'a [PhonContent<T>], usize)>,
}

impl<'a, T> Scanner<T> for ParseScan<'a, T>
where
    T: std::cmp::Eq + std::fmt::Debug,
{
    fn scan(&mut self, s: &Option<T>) -> bool {
        self.sentence.retain_mut(|(sentence, position)| match s {
            Some(s) => {
                if let Some(PhonContent::Normal(string)) = sentence.get(*position) {
                    if s == string {
                        *position += 1;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            None => true,
        });
        !self.sentence.is_empty()
    }

    fn multiscan(&mut self, heads: &FilledHeadTree<T>) -> bool {
        let heads = heads.inorder();
        if heads.is_empty() {
            return true;
        }

        self.sentence.retain_mut(|(sentence, position)| {
            if let Some(s) = sentence.get(*position) {
                match s {
                    PhonContent::Normal(s) => {
                        if heads.len() == 1 && heads.first().unwrap() == &s {
                            *position += 1;
                            true
                        } else {
                            false
                        }
                    }
                    PhonContent::Affixed(string) => {
                        if heads.iter().zip(string.iter()).all(|(a, b)| *a == b) {
                            *position += 1;
                            true
                        } else {
                            false
                        }
                    }
                }
            } else {
                false
            }
        });
        !self.sentence.is_empty()
    }
}

impl<'a, T: Eq + std::fmt::Debug + Clone> ParseScan<'a, T> {
    #[allow(clippy::complexity)]
    pub(crate) fn yield_good_parse(
        b: BeamWrapper<T, Self>,
        rules: &[RuleHolder],
    ) -> Option<(
        impl Iterator<Item = &'a [PhonContent<T>]> + 'a,
        LogProb<f64>,
        RulePool,
    )> {
        if b.is_empty() {
            Some((
                b.beam
                    .sentence
                    .into_iter()
                    .filter(|(s, pos)| s.len() == *pos)
                    .map(|(s, _)| s),
                b.log_prob,
                b.rules.into_rule_pool(rules),
            ))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FuzzyScan<'b, T> {
    pub generated_sentences: Vec<PhonContent<T>>,
    pub sentence_guides: Vec<(&'b [PhonContent<T>], usize)>,
}

impl<'a, T: Eq + std::fmt::Debug + Clone> FuzzyScan<'a, T> {
    pub fn yield_good_parse(
        b: BeamWrapper<T, Self>,
        rules: &[RuleHolder],
    ) -> Option<(LogProb<f64>, Vec<PhonContent<T>>, RulePool)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.generated_sentences.to_vec(),
                b.rules.into_rule_pool(rules),
            ))
        } else {
            None
        }
    }
}

impl<T> Scanner<T> for FuzzyScan<'_, T>
where
    T: std::cmp::Eq + std::fmt::Debug + Clone,
{
    fn scan(&mut self, s: &Option<T>) -> bool {
        if let Some(s) = s {
            self.generated_sentences
                .push(PhonContent::Normal(s.clone()));
        }
        self.sentence_guides
            .retain_mut(|(sentence, position)| match s {
                Some(s) => {
                    if let Some(PhonContent::Normal(string)) = sentence.get(*position) {
                        if s == string {
                            *position += 1;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                None => true,
            });
        true
    }

    fn multiscan(&mut self, heads: &FilledHeadTree<T>) -> bool {
        let mut heads: Vec<_> = heads.inorder().into_iter().cloned().collect();
        self.sentence_guides.retain_mut(|(sentence, position)| {
            if let Some(s) = sentence.get(*position) {
                match s {
                    PhonContent::Normal(s) => {
                        if heads.len() == 1 && heads.first().unwrap() == s {
                            *position += 1;
                            true
                        } else {
                            false
                        }
                    }
                    PhonContent::Affixed(string) => {
                        if heads.iter().zip(string.iter()).all(|(a, b)| a == b) {
                            *position += 1;
                            true
                        } else {
                            false
                        }
                    }
                }
            } else {
                false
            }
        });
        if !heads.is_empty() {
            if heads.len() == 1 {
                self.generated_sentences
                    .push(PhonContent::Normal(heads.pop().unwrap()));
            } else {
                self.generated_sentences.push(PhonContent::Affixed(heads));
            }
        }
        true
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct GeneratorScan<T> {
    pub sentence: Vec<PhonContent<T>>,
}

impl<T: Clone> Scanner<T> for GeneratorScan<T>
where
    T: std::cmp::Eq + std::fmt::Debug,
{
    fn scan(&mut self, s: &Option<T>) -> bool {
        if let Some(s) = s {
            //If the word was None then adding it does nothing
            self.sentence.push(PhonContent::Normal(s.clone()));
        }
        true
    }

    fn multiscan(&mut self, heads: &FilledHeadTree<T>) -> bool {
        let mut heads: Vec<_> = heads.inorder().into_iter().cloned().collect();
        if !heads.is_empty() {
            if heads.len() == 1 {
                self.sentence
                    .push(PhonContent::Normal(heads.pop().unwrap()));
            } else {
                self.sentence.push(PhonContent::Affixed(heads));
            }
        }
        true
    }
}

impl<T: Eq + std::fmt::Debug + Clone> GeneratorScan<T> {
    pub(crate) fn yield_good_parse(
        b: BeamWrapper<T, Self>,
        rules: &[RuleHolder],
    ) -> Option<(LogProb<f64>, Vec<PhonContent<T>>, RulePool)> {
        if b.is_empty() {
            Some((
                b.log_prob,
                b.beam.sentence.to_vec(),
                b.rules.into_rule_pool(rules),
            ))
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct ContinuationScan<'a, T> {
    prefix: &'a [PhonContent<T>],
    position: usize,
    final_char: Option<Continuation<T>>,
}

impl<T> Scanner<T> for ContinuationScan<'_, T>
where
    T: std::cmp::Eq + std::fmt::Debug + Clone,
{
    fn scan(&mut self, word: &Option<T>) -> bool {
        match word {
            Some(word) => {
                if let Some(PhonContent::Normal(string)) = self.prefix.get(self.position) {
                    if word == string {
                        self.position += 1;
                        true
                    } else {
                        false
                    }
                } else if self.position == self.prefix.len() {
                    self.final_char = Some(Continuation::Word(word.clone()));
                    self.position += 1;
                    true
                } else {
                    self.position += 1;
                    true
                }
            }
            None => true,
        }
    }

    fn multiscan(&mut self, heads: &FilledHeadTree<T>) -> bool {
        let heads = heads.inorder();
        if heads.is_empty() {
            return true;
        }

        if let Some(s) = self.prefix.get(self.position) {
            match s {
                PhonContent::Normal(s) => {
                    if heads.len() == 1 && heads.first().unwrap() == &s {
                        self.position += 1;
                        true
                    } else {
                        false
                    }
                }
                PhonContent::Affixed(string) => {
                    if heads.iter().zip(string.iter()).all(|(a, b)| *a == b) {
                        self.position += 1;
                        true
                    } else {
                        false
                    }
                }
            }
        } else if self.position == self.prefix.len() {
            self.final_char = Some(Continuation::AffixedWord(
                heads.into_iter().cloned().collect(),
            ));
            self.position += 1;
            true
        } else {
            self.position += 1;
            true
        }
    }
}

impl<'a, T: Eq + Debug + Clone> ContinuationScan<'a, T> {
    pub fn yield_good_parse(b: BeamWrapper<T, Self>) -> Option<Continuation<T>> {
        if b.is_empty() {
            match b.beam.final_char {
                Some(x) => Some(x),
                None if b.beam.position == b.beam.prefix.len() => Some(Continuation::EndOfSentence),
                None => None,
            }
        } else {
            None
        }
    }
}

///Enum that describes a possible token of a grammar
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Continuation<T> {
    ///The following word is a valid token given the prefix in [`Lexicon::valid_continuations`].
    Word(T),
    ///The following affxied word is a valid token given the prefix in [`Lexicon::valid_continuations`].
    AffixedWord(Vec<T>),
    ///Has the sentence ended
    EndOfSentence,
}

impl<T, C> Lexicon<T, C>
where
    T: Eq + std::fmt::Debug + Clone + Hash,
    C: Eq + Clone + std::fmt::Debug + Hash,
{
    ///Given a grammar and a prefix string, return a [`HashSet`] of the possible [`Continuation`]s (i.e. next words) that are valid
    ///in the grammar.
    ///Returns an [`ParsingError`] if there is no node with the category of `initial_category`.
    ///
    ///```
    ///# use minimalist_grammar_parser::{ParsingConfig, Lexicon, PhonContent};
    ///# use ahash::HashSet;
    ///# use minimalist_grammar_parser::parsing::beam::Continuation;
    ///
    ///let lex = Lexicon::from_string("a::S= b= S\n::S\nb::b")?;
    ///let continuations = lex.valid_continuations("S", &PhonContent::from(["a"]), &ParsingConfig::default())?;
    ///assert_eq!(continuations, HashSet::from_iter([Continuation::Word("b"), Continuation::Word("a")].into_iter()));
    ///let continuations = lex.valid_continuations("S", &PhonContent::from(["a", "b"]), &ParsingConfig::default())?;
    ///assert_eq!(continuations, HashSet::from_iter([Continuation::EndOfSentence]));
    ///# Ok::<(), anyhow::Error>(())
    /// ```
    pub fn valid_continuations(
        &self,
        initial_category: C,
        prefix: &[PhonContent<T>],
        config: &ParsingConfig,
    ) -> Result<HashSet<Continuation<T>>, ParsingError<C>> {
        let cat = self.find_category(&initial_category)?;

        let cont = ContinuationScan {
            prefix,
            position: 0,
            final_char: None,
        };

        let mut valid_chars: HashSet<Continuation<T>> = HashSet::default();

        let mut parse_heap = ParseHeap::new(BeamWrapper::new(cont, cat), config, cat);

        while let Some(mut beam) = parse_heap.pop() {
            if let Some(word) = beam.beam.final_char.as_ref()
                && valid_chars.contains(word)
            {
                //We don't care since there's already a successful parse with that character.
                continue;
            }

            if let Some(moment) = beam.pop_moment() {
                expand(&mut parse_heap, moment, beam, self, config);
            } else if let Some(cont) = ContinuationScan::yield_good_parse(beam) {
                valid_chars.insert(cont);
            }
        }
        Ok(valid_chars)
    }
}

#[cfg(test)]
mod test {

    use crate::{
        ParsingConfig, PhonContent,
        grammars::{DYCK_LANGUAGE, STABLER2011},
        lexicon::Lexicon,
        parsing::beam::Continuation,
    };

    #[test]
    fn continuations() -> anyhow::Result<()> {
        let lex = Lexicon::from_string(STABLER2011)?;

        let strings = [
            "the",
            "the king",
            "which",
            "which king",
            "the king knows",
            "the king drinks the beer",
        ]
        .map(|x| x.split(" ").collect::<Vec<_>>());

        let continuations = [
            vec![
                Continuation::Word("king"),
                Continuation::Word("beer"),
                Continuation::Word("wine"),
                Continuation::Word("queen"),
            ],
            vec![
                Continuation::Word("knows"),
                Continuation::Word("says"),
                Continuation::Word("drinks"),
                Continuation::Word("prefers"),
            ],
            vec![
                Continuation::Word("wine"),
                Continuation::Word("king"),
                Continuation::Word("beer"),
                Continuation::Word("queen"),
            ],
            vec![
                Continuation::Word("drinks"),
                Continuation::Word("knows"),
                Continuation::Word("the"),
                Continuation::Word("says"),
                Continuation::Word("prefers"),
            ],
            vec![Continuation::Word("which"), Continuation::Word("the")],
            vec![Continuation::EndOfSentence],
        ]
        .into_iter()
        .map(|x| x.into_iter().collect());

        for (s, valid) in strings.into_iter().map(PhonContent::new).zip(continuations) {
            let cont = lex.valid_continuations("C", &s, &ParsingConfig::default())?;
            assert_eq!(cont, valid);
        }
        let lex = Lexicon::from_string(DYCK_LANGUAGE)?;

        let strings = ["(", "( )", "( ( )", "( ( ) )", "( ) ( )", "( ( ( ) )"]
            .map(|x| x.split(" ").collect::<Vec<_>>());

        let continuations = [
            vec![Continuation::Word(")"), Continuation::Word("(")],
            vec![Continuation::Word("("), Continuation::EndOfSentence],
            vec![Continuation::Word(")"), Continuation::Word("(")],
            vec![Continuation::Word("("), Continuation::EndOfSentence],
            vec![Continuation::Word("("), Continuation::EndOfSentence],
            vec![Continuation::Word(")"), Continuation::Word("(")],
        ]
        .into_iter()
        .map(|x| x.into_iter().collect());

        for (s, valid) in strings.into_iter().map(PhonContent::new).zip(continuations) {
            let cont = lex.valid_continuations("S", &s, &ParsingConfig::default())?;
            assert_eq!(cont, valid);
        }

        let lex = Lexicon::from_string("a::S= b= S\n::S\nb::b")?;

        let mut strings: Vec<_> = ["a", "a b", "a a b", "a a b b"]
            .iter()
            .map(|x| x.split(" ").collect::<Vec<_>>())
            .collect();
        strings.push(vec![]);

        let continuations = [
            vec![Continuation::Word("b"), Continuation::Word("a")],
            vec![Continuation::EndOfSentence],
            vec![Continuation::Word("b")],
            vec![Continuation::EndOfSentence],
            vec![Continuation::Word("a"), Continuation::EndOfSentence],
        ]
        .into_iter()
        .map(|x| x.into_iter().collect());

        for (s, valid) in strings.into_iter().map(PhonContent::new).zip(continuations) {
            let cont = lex.valid_continuations("S", &s, &ParsingConfig::default())?;
            assert_eq!(cont, valid);
        }
        Ok(())
    }
}
