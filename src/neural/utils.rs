use burn::tensor::{backend::Backend, Tensor};

use crate::lexicon::{Feature, FeatureOrLemma};

use super::neural_lexicon::NeuralFeature;

pub fn log_add<B: Backend>(x: Tensor<B, 1>, y: Tensor<B, 1>) -> Tensor<B, 1> {
    let max = x.clone().max_pair(y.clone());
    max.clone() + ((x - max.clone()).exp() + (y - max).exp()).log()
}

pub const LEMMA_POS: usize = 0;
pub const CATEGORY_POS: usize = 1;
pub const LICENSOR_POS: usize = 2;
pub const LICENSEE_POS: usize = 3;
pub const LEFT_SELECTOR_POS: usize = 4;
pub const RIGHT_SELECTOR_POS: usize = 5;

pub const POSITIONS: [usize; 6] = [
    LEMMA_POS,
    CATEGORY_POS,
    LICENSOR_POS,
    LICENSEE_POS,
    LEFT_SELECTOR_POS,
    RIGHT_SELECTOR_POS,
];

pub fn log_sum_exp_dim<B: Backend, const D: usize, const D2: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D2> {
    let max = tensor.clone().max_dim(dim);
    ((tensor - max.clone()).exp().sum_dim(dim).log() + max).squeeze(dim)
}

pub fn to_feature(pos: usize, category: usize) -> NeuralFeature {
    match pos {
        CATEGORY_POS => FeatureOrLemma::Feature(Feature::Category(category)),
        LICENSEE_POS => FeatureOrLemma::Feature(Feature::Licensee(category)),
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
