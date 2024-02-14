use std::any::Any;

use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use anyhow::{bail, Context, Result};
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use logprob::LogProb;
use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef};

struct TensorPosition(usize);

impl TryInto<TensorPosition> for &FeatureOrLemma<(), usize> {
    fn try_into(self) -> Result<TensorPosition> {
        Ok(TensorPosition(match self {
            FeatureOrLemma::Lemma(_) => 0,
            FeatureOrLemma::Feature(Feature::Category(_)) => 1,
            FeatureOrLemma::Feature(Feature::Licensor(_)) => 2,
            FeatureOrLemma::Feature(Feature::Licensee(_)) => 3,
            FeatureOrLemma::Feature(Feature::Selector(_, crate::Direction::Left)) => 4,
            FeatureOrLemma::Feature(Feature::Selector(_, crate::Direction::Right)) => 5,
            _ => bail!("No conversion for root"),
        }))
    }

    type Error = anyhow::Error;
}

#[derive(Debug)]
struct NeuralLexicon<B: Backend> {
    types: Tensor<B, 3>,      //(lexeme, lexeme_pos, type_distribution)
    lemmas: Tensor<B, 3>,     //(lexeme, lexeme_pos, lemma_distribution)
    categories: Tensor<B, 3>, //(lexeme, lexeme_pos, category_distribution)
    graph: DiGraph<((usize, usize), FeatureOrLemma<(), usize>), ()>,
    root: NodeIndex,
}

impl<B: Backend> NeuralLexicon<B> {
    fn new(
        types: Tensor<B, 3>,
        lemmas: Tensor<B, 3>,
        categories: Tensor<B, 3>,
    ) -> NeuralLexicon<B> {
        //TODO Implement function which uses types/lemmas/categories and gumbel softmax to make a
        //specific grammar

        todo!();
    }
}

impl<B: Backend> Lexiconable<(), usize> for NeuralLexicon<B>
where
    B::FloatElem: TryInto<f64>,
    <<B as Backend>::FloatElem as TryInto<f64>>::Error: std::fmt::Debug,
{
    fn n_children(&self, nx: petgraph::prelude::NodeIndex) -> usize {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .count()
    }

    fn children_of(
        &self,
        nx: petgraph::prelude::NodeIndex,
    ) -> impl Iterator<Item = petgraph::prelude::NodeIndex> + '_ {
        self.graph
            .edges_directed(nx, petgraph::Direction::Outgoing)
            .map(|e| e.target())
    }

    fn get(
        &self,
        nx: petgraph::prelude::NodeIndex,
    ) -> Option<(
        &crate::lexicon::FeatureOrLemma<(), usize>,
        logprob::LogProb<f64>,
    )> {
        if let Some(((lexeme, position), feature)) = self.graph.node_weight(nx) {
            let type_position: TensorPosition = feature.try_into().unwrap();
            let p: Tensor<B, 1> = self
                .types
                .clone()
                .slice([
                    *lexeme..lexeme + 1,
                    *position..position + 1,
                    type_position.0..type_position.0 + 1,
                ])
                .squeeze::<2>(0)
                .squeeze::<1>(0);
            let p: Tensor<B, 1> = {
                match feature {
                    FeatureOrLemma::Feature(f) => match f {
                        Feature::Category(c)
                        | Feature::Selector(c, _)
                        | Feature::Licensor(c)
                        | Feature::Licensee(c) => self
                            .categories
                            .clone()
                            .slice([*lexeme..lexeme + 1, *position..position + 1, *c..c + 1])
                            .squeeze::<2>(0)
                            .squeeze::<1>(0),
                    },
                    _ => Tensor::ones_like(&p),
                }
            } * p;
            let p: f64 = p.into_scalar().try_into().unwrap();
            Some((feature, LogProb::new(p).unwrap()))
        } else {
            None
        }
    }

    fn find_licensee(&self, category: &usize) -> anyhow::Result<petgraph::prelude::NodeIndex> {
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                (_, FeatureOrLemma::Feature(Feature::Licensee(c))) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }

    fn find_category(&self, category: &usize) -> anyhow::Result<petgraph::prelude::NodeIndex> {
        self.graph
            .neighbors_directed(self.root, petgraph::Direction::Outgoing)
            .find(|i| match &self.graph[*i] {
                (_, FeatureOrLemma::Feature(Feature::Licensee(c))) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }
}
