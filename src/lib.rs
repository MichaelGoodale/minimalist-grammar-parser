#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone)]
struct GornIndex {
    index: Vec<Direction>,
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

mod grammars;
pub mod lexicon;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
