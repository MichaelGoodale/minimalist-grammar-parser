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
    position: usize,
    valid_only: bool,
    rule_logits: Tensor<B, 2>,
    lexeme_logits: Tensor<B, 2>,
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
        Rule::UnmergeFromMover { .. } => rule_logits[3],
        Rule::Unmove { target, .. } => rule_logits[4] + lexeme_logits[lexicon.node_to_id(target)],
        Rule::UnmoveFromMover { .. } => rule_logits[5],
    }
}

///The differentiable log probability of sampling some sequence of rules
pub(crate) fn rules_to_prob<B: Backend>(
    rules: &[Rule],
    rule_logits: Tensor<B, 2>,
    lexeme_logits: Tensor<B, 2>,
    lexicon: &NeuralLexicon<B>,
) -> Tensor<B, 1> {
    let mut rule_ids: Vec<u32> = vec![];
    let mut lexeme_used: Vec<u32> = vec![];
    let mut lexeme_ids: Vec<u32> = vec![];
    for (i, rule) in rules.iter().enumerate() {
        match rule {
            Rule::Start(node) => {
                lexeme_used.push(i as u32);
                lexeme_ids.push(lexicon.node_to_id(node) as u32);
                rule_ids.push(0);
            }
            Rule::Scan { .. } => rule_ids.push(1),
            Rule::Unmerge { target, .. } => {
                lexeme_used.push(i as u32);
                lexeme_ids.push(lexicon.node_to_id(target) as u32);
                rule_ids.push(2);
            }
            Rule::UnmergeFromMover { .. } => rule_ids.push(3),
            Rule::Unmove { target, .. } => {
                lexeme_used.push(i as u32);
                lexeme_ids.push(lexicon.node_to_id(target) as u32);
                rule_ids.push(4);
            }
            Rule::UnmoveFromMover { .. } => rule_ids.push(5),
        }
    }
    let rule_ids =
        Tensor::<B, 1, Int>::from_data(Data::from(rule_ids.as_slice()).convert(), lexicon.device())
            .unsqueeze_dim(1);
    let lexeme_used = Tensor::<B, 1, Int>::from_data(
        Data::from(lexeme_used.as_slice()).convert(),
        lexicon.device(),
    );
    let lexeme_ids: Tensor<B, 2, _> = Tensor::<B, 1, Int>::from_data(
        Data::from(lexeme_ids.as_slice()).convert(),
        lexicon.device(),
    )
    .unsqueeze_dim(1);

    let arange = Tensor::arange(0..(rules.len() as i64), &rule_logits.device());
    rule_logits
        .select(0, arange)
        .gather(1, rule_ids)
        .squeeze(1)
        .sum_dim(0)
        + lexeme_logits
            .select(0, lexeme_used)
            .gather(1, lexeme_ids)
            .squeeze(1)
            .sum_dim(0)
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
        if let Some(x) = &n {
            self.position = x.n_steps();
        }
        n
    }

    fn choose(&mut self) {
        if self.parse_buffer.is_empty() {
            self.next_parse = self.upcoming_parses.pop();
            return;
        }

        if self.parse_buffer.len() == 1 {
            self.next_parse = self.parse_buffer.pop();
            return;
        }
        let rule_logits = self
            .rule_logits
            .clone()
            .slice([self.position..self.position + 1, 0..6])
            .to_data()
            .convert::<f64>()
            .value;
        let lexeme_logits = self
            .rule_logits
            .clone()
            .slice([
                self.position..self.position + 1,
                0..self.lexicon.n_lexemes(),
            ])
            .to_data()
            .convert::<f64>()
            .value;

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
        if let Some(min_log_prob) = self.config.parsing_config.min_log_prob {
            if beam.log_prob() < min_log_prob {
                return;
            }
        }

        if let Some(max_steps) = self.config.parsing_config.max_steps {
            if beam.n_steps() > max_steps {
                return;
            }
        }
        if beam.pushable(&self.config.parsing_config) && (!beam.burnt() || !self.valid_only) {
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
        rule_logits: Tensor<B, 2>,
        lexeme_logits: Tensor<B, 2>,
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
            position: 0,
            global_steps: 0,
            lexicon,
            valid_only,
            config,
            rule_logits,
            lexeme_logits,
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
    type Item = (CompletedParse, Tensor<B, 1>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut beam) = self.parses.pop() {
            if beam.burnt() {
                if !self.valid_only {
                    let (parse, history, valid, rules) = beam.into_completed_parse();
                    let rule_prob = rules_to_prob(
                        &rules,
                        self.parses.rule_logits.clone(),
                        self.parses.lexeme_logits.clone(),
                        self.lexicon,
                    );
                    return Some((
                        CompletedParse::new(parse, history, valid, self.lexicon),
                        rule_prob,
                    ));
                } else {
                    continue;
                }
            }

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
                let (parse, history, valid, rules) = beam.into_completed_parse();
                let rule_prob = rules_to_prob(
                    &rules,
                    self.parses.rule_logits.clone(),
                    self.parses.lexeme_logits.clone(),
                    self.lexicon,
                );
                if self.valid_only {
                    if valid {
                        return Some((
                            CompletedParse::new(parse, history, valid, self.lexicon),
                            rule_prob,
                        ));
                    }
                } else {
                    return Some((
                        CompletedParse::new(parse, history, valid, self.lexicon),
                        rule_prob,
                    ));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
    use logprob::LogProb;

    use super::*;
    use crate::{lexicon::Lexiconable, tests::get_a_n_lexicon, ParsingConfig};

    #[test]
    fn check_rule_loss() -> anyhow::Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let config = NeuralConfig {
            n_strings_per_grammar: 20,
            padding_length: 10,
            temperature: 1.0,
            compatible_weight: 0.99,
            parsing_config: ParsingConfig::new(
                LogProb::new(-200.0).unwrap(),
                LogProb::from_raw_prob(0.5).unwrap(),
                50,
                200,
            ),
        };

        let (_, lexicon) = get_a_n_lexicon(&config, 1.0, &mut rng)?;
        let mut rules = vec![];
        for cat in [0, 1] {
            for (_, j) in lexicon
                .find_category(&cat)?
                .iter()
                .chain(lexicon.find_licensee(&cat)?.iter())
            {
                rules.push(Rule::Start(*j));
                rules.push(Rule::Unmerge {
                    child: *j,
                    target: *j,
                    parent: 0,
                    child_id: 0,
                    complement_id: 0,
                });
                rules.push(Rule::Scan {
                    node: *j,
                    parent: 0,
                });
                rules.push(Rule::Unmerge {
                    child: *j,
                    target: *j,
                    parent: 0,
                    child_id: 0,
                    complement_id: 0,
                });
                rules.push(Rule::UnmergeFromMover {
                    child: *j,
                    storage: 0,
                    parent: 0,
                    child_id: 0,
                    stored_id: 0,
                });
                rules.push(Rule::UnmoveFromMover {
                    parent: 0,
                    storage: 0,
                    child_id: 0,
                    stored_id: 0,
                });
                rules.push(Rule::Unmove {
                    target: *j,
                    child_id: 0,
                    stored_id: 0,
                    parent: 0,
                });
            }
        }
        let rule_logits = Tensor::<Autodiff<NdArray>, 2>::random(
            [rules.len(), 6],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let lexeme_logits = Tensor::<Autodiff<NdArray>, 2>::random(
            [rules.len(), 2],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let rule_prob_slow = rules
            .iter()
            .enumerate()
            .map(|(i, r)| {
                rule_to_prob(
                    &rule_logits
                        .clone()
                        .slice([i..i + 1, 0..6])
                        .to_data()
                        .convert()
                        .value,
                    &lexeme_logits
                        .clone()
                        .slice([i..i + 1, 0..2])
                        .to_data()
                        .convert()
                        .value,
                    r,
                    &lexicon,
                )
            })
            .sum::<f64>() as f32;

        let a: f32 = rules_to_prob(&rules, rule_logits, lexeme_logits, &lexicon)
            .into_scalar()
            .elem();
        approx::assert_relative_eq!(a, rule_prob_slow);
        Ok(())
    }
}
