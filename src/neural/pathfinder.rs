use std::collections::BinaryHeap;

use super::{
    loss::NeuralConfig,
    neural_beam::NeuralBeam,
    neural_lexicon::{NeuralLexicon, NeuralProbability, NeuralProbabilityRecord},
    parameterization::GrammarParameterization,
    CompletedParse,
};
use crate::{
    handlers::ParseHeap,
    parsing::{beam::Beam, expand, ParseHolder},
};
use burn::prelude::*;
use itertools::Itertools;
use min_max_heap::MinMaxHeap;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, WeightedIndex};

#[derive(Debug, Clone)]
struct NeuralParseHolder<'a, B: Backend> {
    next_parse: Option<NeuralBeam<'a, B>>,
    parse_buffer: Vec<NeuralBeam<'a, B>>,
    starts: Vec<NeuralBeam<'a, B>>,
    upcoming_parses: BinaryHeap<NeuralBeam<'a, B>>,
    config: &'a NeuralConfig,
    global_steps: usize,
    only_sample: bool,
    rng: StdRng,
    temperature: f64,
}

fn sample<'a, B: Backend>(
    v: impl Iterator<Item = &'a NeuralBeam<'a, B>>,
    temperature: f64,
    rng: &mut impl Rng,
) -> usize {
    let p = v
        .map(|x| x.log_prob().into_inner().exp() / temperature)
        .collect_vec();
    let dist = WeightedIndex::new(p).unwrap();
    dist.sample(rng)
}

impl<'a, B: Backend> NeuralParseHolder<'a, B> {
    fn pop(&mut self) -> Option<NeuralBeam<'a, B>> {
        self.global_steps += 1;
        if let Some(max_steps) = self.config.parsing_config.global_steps {
            if self.global_steps > max_steps {
                return None;
            }
        }
        self.choose();
        let mut n = None;
        std::mem::swap(&mut self.next_parse, &mut n);
        n
    }

    fn choose(&mut self) {
        if self.parse_buffer.is_empty() {
            if self.only_sample {
                let sampled_i = sample(self.starts.iter(), self.temperature, &mut self.rng);
                self.next_parse = Some(self.starts[sampled_i].clone());
            } else {
                self.next_parse = self.parse_buffer.pop();
            }
            return;
        }

        if self.parse_buffer.len() == 1 {
            self.next_parse = self.parse_buffer.pop();
            return;
        }
        let sampled_i = sample(self.parse_buffer.iter(), self.temperature, &mut self.rng);
        if self.only_sample {
            for (i, x) in self.parse_buffer.drain(..).enumerate() {
                if sampled_i == i {
                    self.next_parse = Some(x);
                    break;
                }
            }
        } else {
            self.upcoming_parses
                .extend(
                    self.parse_buffer
                        .drain(..)
                        .enumerate()
                        .filter_map(|(i, x)| {
                            if sampled_i == i {
                                self.next_parse = Some(x);
                                None
                            } else {
                                Some(x)
                            }
                        }),
                );
        }
    }
}

impl<'a, B: Backend> ParseHolder<usize, NeuralBeam<'a, B>> for NeuralParseHolder<'a, B> {
    fn add(&mut self, beam: NeuralBeam<'a, B>) {
        let mut pushable = true;

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
    valid_only: bool,
}

impl<'a, B: Backend> NeuralGenerator<'a, B> {
    pub fn new(
        lexicon: &'a NeuralLexicon<B>,
        g: &'a GrammarParameterization<B>,
        targets: Option<&'a [Vec<usize>]>,
        max_string_length: usize,
        valid_only: bool,
        only_sample: bool,
        config: &'a NeuralConfig,
    ) -> NeuralGenerator<'a, B> {
        let mut parses = Vec::with_capacity(config.parsing_config.max_beams.unwrap_or(100000));
        parses.extend(NeuralBeam::new(lexicon, g, 0, targets, max_string_length).unwrap());
        let parses = if only_sample {
            NeuralParseHolder {
                temperature: g.temperature(),
                rng: StdRng::from_entropy(),
                upcoming_parses: BinaryHeap::new(),
                next_parse: None,
                parse_buffer: vec![],
                starts: parses,
                global_steps: 0,
                only_sample: true,
                config,
            }
        } else {
            NeuralParseHolder {
                temperature: g.temperature(),
                rng: StdRng::from_entropy(),
                upcoming_parses: BinaryHeap::new(),
                next_parse: None,
                parse_buffer: parses,
                starts: vec![],
                global_steps: 0,
                only_sample: false,
                config,
            }
        };
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
            valid_only,
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
            } else {
                let (parse, history, valid) = beam.into_completed_parse();
                if self.valid_only {
                    if valid {
                        return Some(CompletedParse::new(
                            parse,
                            history,
                            valid,
                            None::<&mut StdRng>,
                            //        Some(&mut self.parses.rng),
                            self.lexicon,
                        ));
                    }
                } else {
                    return Some(CompletedParse::new(
                        parse,
                        history,
                        valid,
                        None::<&mut StdRng>,
                        //Some(&mut self.parses.rng),
                        self.lexicon,
                    ));
                }
            }
        }
        None
    }
}
