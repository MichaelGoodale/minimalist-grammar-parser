use std::collections::BinaryHeap;

use super::{
    loss::NeuralConfig,
    neural_beam::NeuralBeam,
    neural_lexicon::{NeuralLexicon, NeuralProbability, NeuralProbabilityRecord},
    parameterization::GrammarParameterization,
    CompletedParse,
};
use crate::parsing::{beam::Beam, expand, ParseHolder, Rule};
use burn::prelude::*;
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, WeightedIndex};

#[derive(Debug, Clone)]
struct NeuralParseHolder<'a, B: Backend> {
    next_parse: Option<NeuralBeam<'a, B>>,
    parse_buffer: Vec<NeuralBeam<'a, B>>,
    upcoming_parses: BinaryHeap<NeuralBeam<'a, B>>,
    config: &'a NeuralConfig,
    global_steps: usize,
    rng: StdRng,
    temperature: f64,
    lexicon: &'a NeuralLexicon<B>,
}

fn rule_to_prob<B: Backend>(
    rule_logits: &[f64],
    lexeme_logits: &[f64],
    rule: &Rule,
    lexicon: &NeuralLexicon<B>,
) -> f64 {
    // We will need to eventually allow for parsing to
    // decide between movement categories
    match rule {
        Rule::Start(node) => rule_logits[0] + lexeme_logits[lexicon.node_to_id(node)],
        Rule::Scan { .. } => rule_logits[1],
        Rule::Unmerge { target, .. } => rule_logits[2] + lexeme_logits[lexicon.node_to_id(target)],
        Rule::UnmergeFromMover { .. } => rule_logits[2],
        Rule::Unmove { target, .. } => rule_logits[3] + lexeme_logits[lexicon.node_to_id(target)],
        Rule::UnmoveFromMover { .. } => rule_logits[4],
    }
}

fn sample<'a, B: Backend>(
    v: impl Iterator<Item = &'a NeuralBeam<'a, B>>,
    rule_logits: &[f64],
    lexeme_logits: &[f64],
    temperature: f64,
    lexicon: &NeuralLexicon<B>,
    rng: &mut impl Rng,
) -> usize {
    let mut p = v
        .map(|x| {
            ((rule_to_prob(rule_logits, lexeme_logits, x.latest_rule(), lexicon)
                + x.log_prob().into_inner())
                / temperature)
                .exp()
        })
        .collect_vec();
    let s = p.iter().sum::<f64>();
    p.iter_mut().for_each(|x| *x /= s);
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
            self.next_parse = self.parse_buffer.pop();
            return;
        }

        if self.parse_buffer.len() == 1 {
            self.next_parse = self.parse_buffer.pop();
            return;
        }
        let rule_logits = [0.0; 5];
        let lexeme_logits = [0.0; 50];
        let rules = self
            .parse_buffer
            .iter()
            .map(|x| x.latest_rule())
            .collect_vec();
        dbg!(rules);

        let sampled_i = sample(
            self.parse_buffer.iter(),
            &rule_logits,
            &lexeme_logits,
            self.temperature,
            self.lexicon,
            &mut self.rng,
        );
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
        config: &'a NeuralConfig,
    ) -> NeuralGenerator<'a, B> {
        let mut parses = Vec::with_capacity(config.parsing_config.max_beams.unwrap_or(100000));
        parses.extend(NeuralBeam::new(lexicon, g, 0, targets, max_string_length).unwrap());
        let parses = NeuralParseHolder {
            temperature: g.temperature(),
            rng: StdRng::from_entropy(),
            upcoming_parses: BinaryHeap::new(),
            next_parse: None,
            parse_buffer: parses,
            global_steps: 0,
            lexicon,
            config,
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
                        return Some(CompletedParse::new(parse, history, valid, self.lexicon));
                    }
                } else {
                    return Some(CompletedParse::new(parse, history, valid, self.lexicon));
                }
            }
        }
        None
    }
}
