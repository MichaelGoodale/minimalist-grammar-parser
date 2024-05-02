use std::collections::BinaryHeap;

use super::{
    loss::NeuralConfig,
    neural_beam::NeuralBeam,
    neural_lexicon::{NeuralLexicon, NeuralProbability, NeuralProbabilityRecord},
    parameterization::GrammarParameterization,
    CompletedParse,
};
use crate::parsing::{beam::Beam, expand, ParseHolder};
use burn::prelude::*;
use min_max_heap::MinMaxHeap;

#[derive(Debug, Clone)]
struct NeuralParseHolder<'a, B: Backend> {
    upcoming_parses: MinMaxHeap<NeuralBeam<'a, B>>,
    parse_buffer: Vec<NeuralBeam<'a, B>>,
    config: &'a NeuralConfig,
    global_steps: usize,
}

impl<'a, B: Backend> NeuralParseHolder<'a, B> {
    fn pop(&mut self) -> Option<NeuralBeam<'a, B>> {
        self.upcoming_parses.pop_max()
    }
    fn choose(&mut self) {
        if let Some(max) = self.config.parsing_config.max_beams {
            for x in self.parse_buffer.drain(..) {
                if self.upcoming_parses.len() >= max {
                    self.upcoming_parses.push_pop_min(x);
                } else {
                    self.upcoming_parses.push(x);
                }
            }
        } else {
            self.upcoming_parses.extend(self.parse_buffer.drain(..))
        }
    }
}

impl<'a, B: Backend> ParseHolder<usize, NeuralBeam<'a, B>> for NeuralParseHolder<'a, B> {
    fn add(&mut self, beam: NeuralBeam<'a, B>) {
        self.global_steps += 1;
        let mut pushable = true;
        if let Some(max_steps) = self.config.parsing_config.global_steps {
            pushable = self.global_steps < max_steps;
        }

        if let Some(min_log_prob) = self.config.parsing_config.min_log_prob {
            if beam.log_prob() < min_log_prob {
                pushable = false;
            }
        }

        if let Some(max_steps) = self.config.parsing_config.max_steps {
            if beam.n_steps() > max_steps {
                pushable = false;
            }
        }
        if pushable && beam.pushable(&self.config.parsing_config) {
            self.parse_buffer.push(beam);
        }
    }
}

#[derive(Debug)]
pub struct NeuralGenerator<'a, B: Backend> {
    lexicon: &'a NeuralLexicon<B>,
    parses: NeuralParseHolder<'a, B>,
    move_log_prob: NeuralProbability,
    merge_log_prob: NeuralProbability,
}

impl<'a, B: Backend> NeuralGenerator<'a, B> {
    pub fn new(
        lexicon: &'a NeuralLexicon<B>,
        g: &'a GrammarParameterization<B>,
        targets: Option<&'a [Vec<usize>]>,
        max_string_length: Option<usize>,
        config: &'a NeuralConfig,
    ) -> NeuralGenerator<'a, B> {
        let mut parses = Vec::with_capacity(config.parsing_config.max_beams.unwrap_or(100000));
        parses.extend(NeuralBeam::new(lexicon, g, 0, targets, max_string_length).unwrap());
        let mut parses = NeuralParseHolder {
            upcoming_parses: MinMaxHeap::new(),
            parse_buffer: parses,
            global_steps: 0,
            config,
        };
        parses.choose();
        NeuralGenerator {
            lexicon,
            move_log_prob: NeuralProbability(
                NeuralProbabilityRecord::MoveRuleProb,
                None,
                config.parsing_config.move_prob,
            ),
            merge_log_prob: NeuralProbability(
                NeuralProbabilityRecord::MergeRuleProb,
                None,
                config.parsing_config.move_prob.opposite_prob(),
            ),
            parses,
        }
    }
}

impl<'a, B: Backend> Iterator for NeuralGenerator<'a, B> {
    type Item = CompletedParse;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parses.pop() {
            if let Some(moment) = beam.pop_moment() {
                expand(
                    &mut self.parses,
                    moment,
                    beam,
                    self.lexicon,
                    self.move_log_prob,
                    self.merge_log_prob,
                );
                self.parses.choose();
            } else if let Some((parse, history)) = beam.yield_good_parse() {
                return Some(CompletedParse {
                    parse,
                    history,
                    valid: true,
                });
            }
        }
        None
    }
}
