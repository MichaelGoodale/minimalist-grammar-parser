use burn::tensor::{backend::Backend, Tensor};

use crate::lexicon::{Feature, FeatureOrLemma};

use super::neural_lexicon::NeuralFeature;

pub const LICENSOR_POS: usize = 0;
pub const LEFT_SELECTOR_POS: usize = 1;
pub const RIGHT_SELECTOR_POS: usize = 2;

pub fn log_sum_exp_dim<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let max = tensor.clone().max_dim(dim);
    (tensor - max.clone()).exp().sum_dim(dim).log() + max
}

pub fn to_feature(pos: usize, category: usize) -> NeuralFeature {
    match pos {
        LICENSOR_POS => FeatureOrLemma::Feature(Feature::Licensor(category)),
        LEFT_SELECTOR_POS => {
            FeatureOrLemma::Feature(Feature::Selector(category, crate::Direction::Left))
        }
        RIGHT_SELECTOR_POS => {
            FeatureOrLemma::Feature(Feature::Selector(category, crate::Direction::Right))
        }
        _ => panic!("Invalid position!"),
    }
}
