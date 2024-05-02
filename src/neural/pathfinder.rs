use super::{
    neural_beam::{NeuralBeam, StringPath, StringProbHistory},
    neural_lexicon::{NeuralLexicon, NeuralProbability, NeuralProbabilityRecord},
    parameterization::GrammarParameterization,
};
use crate::{
    parsing::{beam::Beam, expand},
    ParseHeap, ParsingConfig,
};
use burn::prelude::*;
use min_max_heap::MinMaxHeap;
use std::collections::BTreeSet;

struct NeuralParses {}

struct NeuralParseAttempt {}

#[derive(Debug)]
pub struct NeuralGenerator<'a, B: Backend> {
    lexicon: &'a NeuralLexicon<B>,
    parse_heap: ParseHeap<'a, usize, NeuralBeam<'a, B>>,
    target_lens: Option<BTreeSet<usize>>,
    move_log_prob: NeuralProbability,
    merge_log_prob: NeuralProbability,
}

impl<'a, B: Backend> NeuralGenerator<'a, B> {
    pub fn new(
        lexicon: &'a NeuralLexicon<B>,
        g: &'a GrammarParameterization<B>,
        targets: Option<&'a [Vec<usize>]>,
        max_string_length: Option<usize>,
        config: &'a ParsingConfig,
    ) -> NeuralGenerator<'a, B> {
        let mut parse_heap = MinMaxHeap::with_capacity(config.max_beams.unwrap_or(100000));
        let target_lens = targets.map(|x| x.iter().map(|x| x.len()).collect());
        parse_heap.extend(NeuralBeam::new(lexicon, g, 0, targets, max_string_length).unwrap());
        NeuralGenerator {
            lexicon,
            target_lens,
            move_log_prob: NeuralProbability(
                NeuralProbabilityRecord::MoveRuleProb,
                None,
                config.move_prob,
            ),
            merge_log_prob: NeuralProbability(
                NeuralProbabilityRecord::MergeRuleProb,
                None,
                config.move_prob.opposite_prob(),
            ),

            parse_heap: ParseHeap::new(parse_heap, config),
        }
    }
}

impl<B: Backend> Iterator for NeuralGenerator<'_, B> {
    type Item = (StringPath, StringProbHistory);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parse_heap.pop() {
            if let Some(moment) = beam.pop_moment() {
                expand(
                    &mut self.parse_heap,
                    moment,
                    beam,
                    self.lexicon,
                    self.move_log_prob,
                    self.merge_log_prob,
                );
            } else if let Some(sentence) = beam.yield_good_parse() {
                if self
                    .target_lens
                    .as_ref()
                    .map_or(true, |x| x.contains(&sentence.0.len()))
                {
                    return Some(sentence);
                }
            }
        }
        None
    }
}
