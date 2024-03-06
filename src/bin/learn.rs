use burn::{
    module::Module,
    tensor::{
        backend::{AutodiffBackend, Backend},
        ops::IntTensor,
        Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep},
};
use minimalist_grammar_parser::{get_neural_outputs, NeuralConfig};
use rand::{thread_rng, Rng};

#[derive(Module, Debug)]
struct GrammarModel<B: Backend> {
    types: Tensor<B, 3>,
    lemmas: Tensor<B, 3>,
    categories: Tensor<B, 3>,
    weights: Tensor<B, 2>,
}

impl<B: Backend> GrammarModel<B>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem> + Into<f32>,
{
    fn forward(
        &self,
        targets: Tensor<B, 2, Int>,
        neural_config: &NeuralConfig,
        rng: &mut impl Rng,
    ) -> Tensor<B, 1> {
        get_neural_outputs(
            self.lemmas.clone(),
            self.types.clone(),
            self.categories.clone(),
            self.weights.clone(),
            targets,
            neural_config,
            rng,
        )
    }
}

struct TargetStrings<'a, B: Backend> {
    targets: Tensor<B, 2, Int>,
    neural_config: &'a NeuralConfig,
}

impl<'a, B: AutodiffBackend> TrainStep<TargetStrings<'a, B>, Tensor<B, 1>> for GrammarModel<B>
where
    B::FloatElem: std::ops::Add<B::FloatElem, Output = B::FloatElem> + Into<f32>,
{
    fn step(&self, item: TargetStrings<B>) -> burn::train::TrainOutput<Tensor<B, 1>> {
        let loss = self.forward(item.targets, item.neural_config, &mut rand::thread_rng());
        TrainOutput::new(self, loss.backward(), loss)
    }
}

fn main() {}
