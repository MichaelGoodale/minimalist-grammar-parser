use burn::tensor::{backend::Backend, ElementConversion, Tensor};

use crate::neural_lexicon::NeuralLexicon;

use super::*;

#[derive(Debug, Clone)]
pub struct NeuralBeam<'a, B: Backend> {
    log_probability: B::FloatElem,
    lexicon: &'a NeuralLexicon<B>,
    pub queue: BinaryHeap<Reverse<ParseMoment>>,
    generated_sentences: Vec<Tensor<B, 1>>,
    rules: ThinVec<Rule>,
    top_id: usize,
    steps: usize,
    record_rules: bool,
}

impl<'a, B: Backend> NeuralBeam<'a, B>
where
    NeuralLexicon<B>: Lexiconable<(usize, usize), usize>,
{
    pub fn new(
        lexicon: &'a NeuralLexicon<B>,
        initial_category: usize,
        record_rules: bool,
    ) -> Result<NeuralBeam<'a, B>> {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
        let category_index = lexicon.find_category(&initial_category)?;

        queue.push(Reverse(ParseMoment::new(
            FutureTree {
                node: category_index,
                index: GornIndex::default(),
                id: 0,
            },
            thin_vec![],
        )));

        Ok(NeuralBeam {
            log_probability: Tensor::<B, 1>::ones([1], lexicon.device()).into_scalar(),
            queue,
            lexicon,
            generated_sentences: vec![],
            rules: if record_rules {
                thin_vec![Rule::Start(category_index)]
            } else {
                thin_vec![]
            },
            top_id: 0,
            steps: 0,
            record_rules,
        })
    }

    pub fn yield_good_parse(self) -> Option<Tensor<B, 2>> {
        if self.queue.is_empty() && !self.generated_sentences.is_empty() {
            let sentence = self.generated_sentences;
            Some(Tensor::stack(sentence, 0) + self.log_probability)
        } else {
            None
        }
    }
}

impl<B: Backend> PartialEq for NeuralBeam<'_, B> {
    fn eq(&self, other: &Self) -> bool {
        self.steps == other.steps
            && self.top_id == other.top_id
            && self.rules == other.rules
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<B: Backend> PartialOrd for NeuralBeam<'_, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: Backend> Eq for NeuralBeam<'_, B> {}

impl<B: Backend> Ord for NeuralBeam<'_, B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let a: f32 = self.log_probability.elem();
        let b: f32 = other.log_probability.elem();
        a.partial_cmp(&b).unwrap()
    }
}

impl<B: Backend> Beam<(usize, usize)> for NeuralBeam<'_, B>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem>,
{
    type Probability = B::FloatElem;

    fn log_probability(&self) -> &Self::Probability {
        &self.log_probability
    }

    fn log_probability_mut(&mut self) -> &mut Self::Probability {
        &mut self.log_probability
    }

    fn pop_moment(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    fn push_moment(&mut self, x: ParseMoment) {
        self.queue.push(Reverse(x))
    }

    fn push_rule(&mut self, x: Rule) {
        self.rules.push(x)
    }

    fn record_rules(&self) -> bool {
        self.record_rules
    }

    fn scan(
        v: &mut ParseHeap<(usize, usize), Self>,
        moment: &ParseMoment,
        mut beam: Self,
        s: &Option<(usize, usize)>,
        child_node: NodeIndex,
        child_prob: Self::Probability,
    ) {
        beam.queue.shrink_to_fit();
        if let Some((pos, lex)) = s {
            beam.generated_sentences
                .push(beam.lexicon.lemma_at_position(*pos, *lex))
        }
        beam.log_probability = beam.log_probability + child_prob;
        if beam.record_rules() {
            beam.rules.push(Rule::Scan {
                node: child_node,
                parent: moment.tree.id,
            });
        }
        beam.steps += 1;
        v.push(beam);
    }

    fn inc(&mut self) {
        self.steps += 1;
    }

    fn n_steps(&self) -> usize {
        self.steps
    }

    fn top_id(&self) -> usize {
        self.top_id
    }

    fn top_id_mut(&mut self) -> &mut usize {
        &mut self.top_id
    }
    fn pushable(&self, config: &ParsingConfig) -> bool {
        self.n_steps() < config.max_steps
    }
}
