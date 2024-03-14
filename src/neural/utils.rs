use burn::tensor::{backend::Backend, Tensor};

pub fn log_add<B: Backend>(x: Tensor<B, 1>, y: Tensor<B, 1>) -> Tensor<B, 1> {
    let max = x.clone().max_pair(y.clone());
    max.clone() + ((x - max.clone()).exp() + (y - max).exp()).log()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TensorPosition(pub usize);

pub const LEMMA_POS: TensorPosition = TensorPosition(0);
pub const CATEGORY_POS: TensorPosition = TensorPosition(1);
pub const LICENSOR_POS: TensorPosition = TensorPosition(2);
pub const LICENSEE_POS: TensorPosition = TensorPosition(3);
pub const LEFT_SELECTOR_POS: TensorPosition = TensorPosition(4);
pub const RIGHT_SELECTOR_POS: TensorPosition = TensorPosition(5);

pub const POSITIONS: [TensorPosition; 6] = [
    LEMMA_POS,
    CATEGORY_POS,
    LICENSOR_POS,
    LICENSEE_POS,
    LEFT_SELECTOR_POS,
    RIGHT_SELECTOR_POS,
];
