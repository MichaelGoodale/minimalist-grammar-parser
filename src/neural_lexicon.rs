use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use anyhow::{bail, Context, Result};
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef};

struct TensorPosition(usize);

impl TryInto<TensorPosition> for &FeatureOrLemma<usize, usize> {
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
    types: Tensor<B, 3>,  //(lexeme, lexeme_pos, type_distribution)
    lemmas: Tensor<B, 3>, //(lexeme, lexeme_pos, lemma_distribution)
    graph: DiGraph<((usize, usize), FeatureOrLemma<usize, usize>), ()>,
    root: NodeIndex,
}

impl<B: Backend> Lexiconable<usize, usize> for NeuralLexicon<B> {
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
        &crate::lexicon::FeatureOrLemma<usize, usize>,
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
            Some((feature, p))
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
