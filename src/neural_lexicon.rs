use crate::lexicon::{Feature, FeatureOrLemma, Lexiconable};
use anyhow::Context;
use burn::tensor::{activation::log_softmax, backend::Backend, Tensor};
use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TensorPosition(usize);

const LEMMA_POS: TensorPosition = TensorPosition(0);
const CATEGORY_POS: TensorPosition = TensorPosition(1);
const LICENSOR_POS: TensorPosition = TensorPosition(2);
const LICENSEE_POS: TensorPosition = TensorPosition(3);
const LEFT_SELECTOR_POS: TensorPosition = TensorPosition(4);
const RIGHT_SELECTOR_POS: TensorPosition = TensorPosition(5);
pub const N_TYPES: usize = 6;

type NeuralGraph<B> = DiGraph<(Option<Tensor<B, 1>>, FeatureOrLemma<usize, usize>), ()>;

#[derive(Debug)]
pub struct NeuralLexicon<B: Backend> {
    lemmas: Tensor<B, 3>, //(lexeme, lexeme_pos, lemma_distribution)
    graph: NeuralGraph<B>,
    root: NodeIndex,
    device: B::Device,
}

fn to_feature(pos: TensorPosition, category: usize) -> FeatureOrLemma<usize, usize> {
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

fn get_prob_of_type_category<B: Backend>(
    types: &Tensor<B, 3>,
    categories: &Tensor<B, 3>,
    category: usize,
    type_position: TensorPosition,
    n_lexemes: usize,
    pos: usize,
) -> Tensor<B, 1> {
    let p_of_type: Tensor<B, 1> = types
        .clone()
        .slice([
            0..n_lexemes,
            pos..pos + 1,
            type_position.0..type_position.0 + 1,
        ])
        .reshape([-1]);

    let p_of_category: Tensor<B, 1> = categories
        .clone()
        .slice([0..n_lexemes, pos..pos + 1, category..category + 1])
        .reshape([-1]);

    log_sum_exp(p_of_type + p_of_category)
}

fn log_sum_exp<B: Backend>(a: Tensor<B, 1>) -> Tensor<B, 1> {
    let max: Tensor<B, 1> = a.clone().max();
    println!("{} {}", a, max);
    (a - max.clone()).exp().sum().log() + max
}

fn get_prob_of_lemma<B: Backend>(
    types: &Tensor<B, 3>,
    lemmas: &Tensor<B, 3>,
    n_lexemes: usize,
    pronounced_lemma: bool,
    pos: usize,
) -> Tensor<B, 1> {
    let mut p_of_type: Tensor<B, 1> = types
        .clone()
        .slice([0..n_lexemes, pos..pos + 1, LEMMA_POS.0..LEMMA_POS.0 + 1])
        .reshape([-1]);

    if !pronounced_lemma {
        let p_of_unpronounced_lemma: Tensor<B, 1> = lemmas
            .clone()
            .slice([0..n_lexemes, pos..pos + 1, 0..1])
            .reshape([-1]);
        p_of_type = p_of_type + p_of_unpronounced_lemma;
    }
    log_sum_exp(p_of_type)
}

fn add_children<B: Backend>(
    graph: &mut NeuralGraph<B>,
    parent: NodeIndex,
    pos: usize,
    categories: &Tensor<B, 3>,
    lemmas: &Tensor<B, 3>,
    types: &Tensor<B, 3>,
    n_lexemes: usize,
) -> Vec<NodeIndex> {
    for pronounced in [true, false] {
        let lemma: NodeIndex = graph.add_node((
            Some(get_prob_of_lemma(types, lemmas, n_lexemes, pronounced, pos)),
            FeatureOrLemma::Lemma(match pronounced {
                true => Some(pos),
                false => None,
            }),
        ));
        graph.add_edge(parent, lemma, ());
    }

    let mut children = vec![];
    for node_type in [LEFT_SELECTOR_POS, RIGHT_SELECTOR_POS, LICENSOR_POS] {
        for category in 0..categories.shape().dims[2] {
            let descendent: NodeIndex = graph.add_node((
                Some(get_prob_of_type_category(
                    types, categories, category, node_type, n_lexemes, pos,
                )),
                to_feature(node_type, category),
            ));
            graph.add_edge(parent, descendent, ());
            children.push(descendent);
        }
    }
    children
}

fn add_til_category<B: Backend>(
    graph: &mut NeuralGraph<B>,
    parent: NodeIndex,
    pos: usize,
    categories: &Tensor<B, 3>,
    types: &Tensor<B, 3>,
    n_lexemes: usize,
) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    let mut licensee_children = vec![];
    let mut category_children = vec![];
    for node_type in [LICENSEE_POS, CATEGORY_POS] {
        for category in 0..categories.shape().dims[2] {
            let descendent: NodeIndex = graph.add_node((
                Some(get_prob_of_type_category(
                    types, categories, category, node_type, n_lexemes, pos,
                )),
                to_feature(node_type, category),
            ));
            graph.add_edge(parent, descendent, ());
            match node_type {
                LICENSEE_POS => licensee_children.push(descendent),
                CATEGORY_POS => category_children.push(descendent),
                _ => panic!("This is impossible to reach because of the preceeding array"),
            }
        }
    }
    (category_children, licensee_children)
}

impl<B: Backend> NeuralLexicon<B> {
    pub fn new(
        types: Tensor<B, 3>,          //(lexeme, lexeme_pos, type_distribution)
        lexeme_weights: Tensor<B, 1>, // (lexeme)
        lemmas: Tensor<B, 3>,         //(lexeme, lexeme_pos, lemma_distribution)
        categories: Tensor<B, 3>,     //(lexeme, lexeme_pos, categories_position)
    ) -> NeuralLexicon<B> {
        //TODO Implement function which uses types/lemmas/categories and gumbel softmax to make a
        //specific grammar
        //
        let n_lexemes = types.shape().dims[0];
        let n_positions = types.shape().dims[1];

        let mut graph = DiGraph::new();
        let root = graph.add_node((None, FeatureOrLemma::Root));
        let lexeme_weights = log_softmax(lexeme_weights, 0).reshape([n_lexemes, 1, 1]);
        let types = log_softmax(types, 2) + lexeme_weights.clone();
        let categories = log_softmax(categories, 2) + lexeme_weights.clone();
        let lemmas = log_softmax(lemmas, 2) + lexeme_weights;

        let mut position = 0;

        let (category_children, mut licensee_children) =
            add_til_category(&mut graph, root, position, &categories, &types, n_lexemes);

        let mut children = category_children;
        let mut new_children = vec![];

        let mut new_licensee_children = vec![];

        while position < n_positions - 1 {
            position += 1;

            for child in children.drain(0..) {
                new_children.extend(add_children(
                    &mut graph,
                    child,
                    position,
                    &categories,
                    &lemmas,
                    &types,
                    n_lexemes,
                ));
            }
            children.extend(new_children.drain(0..));

            for child in licensee_children.drain(0..) {
                let (category_children, licensee_children) =
                    add_til_category(&mut graph, child, position, &categories, &types, n_lexemes);
                new_licensee_children.extend(licensee_children.into_iter());
                children.extend(category_children.into_iter());
            }
            licensee_children.extend(new_licensee_children.drain(0..));
        }
        let device = lemmas.device();

        NeuralLexicon {
            lemmas,
            graph,
            root,
            device,
        }
    }

    pub fn lemma_at_position(&self, pos: usize) -> Tensor<B, 1> {
        let shape = self.lemmas.shape().dims;
        self.lemmas
            .clone()
            .slice([0..shape[0], pos..pos + 1, 0..shape[2]])
            .reshape([-1])
    }
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<B: Backend> Lexiconable<usize, usize> for NeuralLexicon<B> {
    type Probability = Tensor<B, 1>;

    fn one(&self) -> Self::Probability {
        Tensor::<B, 1>::ones([1], &self.device)
    }

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
        Self::Probability,
    )> {
        if let Some((Some(p), feature)) = self.graph.node_weight(nx) {
            Some((feature, p.clone()))
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
                (_, FeatureOrLemma::Feature(Feature::Category(c))) => c == category,
                _ => false,
            })
            .with_context(|| format!("{category:?} is not a valid category in the lexicon!"))
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::Tensor;

    use super::{NeuralLexicon, N_TYPES};

    #[test]
    fn make_new_lexicon() -> anyhow::Result<()> {
        let lemmas = Tensor::<NdArray, 3>::random(
            [3, 5, 3],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let types = Tensor::<NdArray, 3>::random(
            [3, 5, N_TYPES],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let categories = Tensor::<NdArray, 3>::random(
            [3, 5, 2],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );
        let lexeme_weights = Tensor::<NdArray, 1>::random(
            [3],
            burn::tensor::Distribution::Default,
            &NdArrayDevice::default(),
        );

        NeuralLexicon::new(types, lexeme_weights, lemmas, categories);
        Ok(())
    }
}
