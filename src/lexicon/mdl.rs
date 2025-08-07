//! Module which defines helper functions to calculate the MDL of MGs according to Ermolaeva, 2021
//!
//! - Ermolaeva, M. (2021). Learning Syntax via Decomposition [The University of Chicago]. https://doi.org/10.6082/uchicago.3015

use super::{Feature, FeatureOrLemma, Lexicon, LexiconError};
use ahash::AHashSet;
use std::hash::Hash;

///Defines the cost of a type of lemma
pub trait SymbolCost: Sized {
    ///This is the length of a member where each sub-unit has [``n_phonemes``] possible encodings.
    /// # Example
    /// Let Phon = $\{a,b,c\}$ so, `n_phonemes` should be 3 (passed to [``mdl_score``].
    /// The string, abcabc should have symbol_cost 6.
    fn symbol_cost(x: &Option<Self>) -> u16;
}

impl SymbolCost for &str {
    fn symbol_cost(x: &Option<Self>) -> u16 {
        match x {
            Some(x) => x.len().try_into().unwrap(),
            None => 0,
        }
    }
}

impl SymbolCost for String {
    fn symbol_cost(x: &Option<Self>) -> u16 {
        match x {
            Some(x) => x.len().try_into().unwrap(),
            None => 0,
        }
    }
}

impl SymbolCost for char {
    fn symbol_cost(x: &Option<Self>) -> u16 {
        match x {
            Some(_) => 1,
            None => 0,
        }
    }
}

impl SymbolCost for u8 {
    fn symbol_cost(x: &Option<Self>) -> u16 {
        match x {
            Some(_) => 1,
            None => 0,
        }
    }
}

///Number of types of features, e.g. the space of possible features in [``Feature``] enum.
///Here it is six to account for left and right attachment.
const MG_TYPES: u16 = 6;

impl<T: Eq + std::fmt::Debug + Clone + SymbolCost, Category: Eq + std::fmt::Debug + Clone + Hash>
    Lexicon<T, Category>
{
    /// Returns the MDL score of a lexicon assuming the number of phonemes is fixed.
    pub fn mdl_score_fixed_category_size(
        &self,
        n_phonemes: u16,
        n_categories: u16,
    ) -> Result<f64, LexiconError> {
        self.mdl_inner(n_phonemes, Some(n_categories))
    }

    ///Returns the MDL Score of the lexicon
    ///
    /// # Arguments
    /// * `n_phonemes` - The size of required to encode a symbol of the phonology. e.g in English orthography, it would be 26.
    pub fn mdl_score(&self, n_phonemes: u16) -> Result<f64, LexiconError> {
        self.mdl_inner(n_phonemes, None)
    }

    fn mdl_inner(&self, n_phonemes: u16, n_categories: Option<u16>) -> Result<f64, LexiconError> {
        let mut category_symbols = AHashSet::new();
        let mut lexemes: Vec<(f64, f64)> = Vec::with_capacity(self.leaves.len());

        for leaf in self.leaves.iter() {
            if let FeatureOrLemma::Lemma(lemma) = &self.graph[leaf.0] {
                let n_phonemes = T::symbol_cost(lemma);

                let mut nx = leaf.0;
                let mut n_features = 0;
                while let Some(parent) = self.parent_of(nx) {
                    if parent == self.root {
                        break;
                    } else if let FeatureOrLemma::Feature(f) = &self.graph[parent] {
                        category_symbols.insert(match f {
                            Feature::Category(c)
                            | Feature::Licensor(c)
                            | Feature::Licensee(c)
                            | Feature::Affix(c, _)
                            | Feature::Selector(c, _) => c,
                        });
                        n_features += 1;
                    } else if let FeatureOrLemma::Complement(c, _d) = &self.graph[parent] {
                        category_symbols.insert(c);
                        n_features += 1;
                    }
                    nx = parent;
                }
                lexemes.push((n_phonemes.into(), n_features.into()));
            } else {
                return Err(LexiconError::MissingLexeme(*leaf));
            }
        }
        let n_categories: u16 =
            n_categories.unwrap_or_else(|| category_symbols.len().try_into().unwrap());

        let bits_per_feature: f64 = (MG_TYPES * n_categories).into();
        let bits_per_feature = bits_per_feature.ln();
        let bits_per_phoneme: f64 = (Into::<f64>::into(n_phonemes)).ln();

        Ok(lexemes
            .into_iter()
            .map(|(n_phonemes, n_categories)| {
                n_phonemes * bits_per_phoneme + bits_per_feature * n_categories
            })
            .sum())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mdl_score_test() -> anyhow::Result<()> {
        let ga: &str = "mary::d -k
laughs::=d +k t
laughed::=d +k t
jumps::=d +k t
jumped::=d +k t";
        let gb: &str = "mary::d -k
laugh::=d v
jump::=d v
s::=v +k t
ed::=v +k t";
        for (g, n_categories, string_size, feature_size) in
            [(ga, 3, 28_f64, 14_f64), (gb, 4, 16_f64, 12_f64)]
        {
            let lex = Lexicon::from_string(g)?;
            for alphabet_size in [26, 32, 37] {
                let bits_per_symbol: f64 = (MG_TYPES * n_categories).into();
                let bits_per_phoneme: f64 = alphabet_size.into();
                assert_relative_eq!(
                    lex.mdl_score(alphabet_size)?,
                    string_size * bits_per_phoneme.ln() + feature_size * bits_per_symbol.ln()
                );
            }
        }
        Ok(())
    }
}
