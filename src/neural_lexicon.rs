use burn::tensor::{backend::Backend, Tensor};

struct NeuralLexicon<B: Backend> {
    types: Tensor<B, 3>,  //(lexeme, lexeme_pos, type_distribution)
    lemmas: Tensor<B, 3>, //(lexeme, lexeme_pos, lemma_distribution)
}
