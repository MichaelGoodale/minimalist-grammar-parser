use self::neural_beam::{StringPath, StringProbHistory};

pub mod loss;
pub mod neural_beam;
pub mod neural_lexicon;
pub mod parameterization;
pub mod pathfinder;
mod utils;

#[derive(Debug, Clone)]
pub struct CompletedParse {
    parse: StringPath,
    history: StringProbHistory,
    valid: bool,
}

impl CompletedParse {
    pub fn len(&self) -> usize {
        self.parse.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parse.is_empty()
    }
}

pub const N_TYPES: usize = 3;
