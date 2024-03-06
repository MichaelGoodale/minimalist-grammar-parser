use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
struct GrammarModel<B: Backend> {
    types: Tensor<B, 3>,
    lemmas: Tensor<B, 3>,
    categories: Tensor<B, 3>,
    weights: Tensor<B, 2>,
}

impl<B: Backend> GrammarModel<B> {
    fn forward(&self, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        todo!();
    }
}

fn main() {}
