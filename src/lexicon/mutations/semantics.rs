use ahash::HashMap;
use simple_semantics::{
    Entity,
    lambda::{self, RootedLambdaPool, types::LambdaType},
};

use crate::lexicon::SemanticLexicon;

use super::*;

#[derive(Debug, Clone)]
pub enum TypeConstraintData<C> {
    None,
    Featural(HashMap<C, Vec<LambdaType>>),
}

#[derive(Debug, Clone)]
pub struct LearntSemanticLexicon<T: Eq, C: Eq> {
    lexicon: SemanticLexicon<T, C>,
    typing: TypeConstraintData<C>,
}

impl<T, C> LearntSemanticLexicon<T, C>
where
    T: Eq + Debug + Clone,
    C: Eq + Debug + Clone + FreshCategory,
{
    pub fn new_random(lemmas: &[T], base_category: &C, rng: &mut impl Rng) -> Self {
        let lexicon = Lexicon::random(base_category, lemmas, None, rng);

        let mut semantic_entries = HashMap::default();
        let actors = [Entity::Actor(0), Entity::Actor(1)];
        for leaf in lexicon.leaves.iter() {
            let lambda_type = LambdaType::random(rng);
            semantic_entries.insert(
                *leaf,
                RootedLambdaPool::random_expr(lambda_type, &actors, None, rng),
            );
        }

        LearntSemanticLexicon {
            lexicon: SemanticLexicon::new(lexicon, semantic_entries),
            typing: TypeConstraintData::None,
        }
    }

    pub fn lexicon(&self) -> &SemanticLexicon<T, C> {
        &self.lexicon
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn random_semantic_lexicon() -> anyhow::Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..100 {
            let lex = LearntSemanticLexicon::new_random(&["the", "dog", "runs"], &0, &mut rng);
            println!("{}", lex.lexicon);
            println!("______________________________________________________");
        }
        panic!();
        Ok(())
    }
}
