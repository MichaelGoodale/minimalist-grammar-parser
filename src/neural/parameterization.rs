use super::{utils::clamp_prob, N_TYPES};
use ahash::HashMap;
use anyhow::bail;
use burn::tensor::{
    activation::{log_sigmoid, log_softmax},
    backend::Backend,
    Data, Device, Shape, Tensor,
};
use logprob::LogProb;
use rand::Rng;
use rand_distr::{Distribution, Gumbel};

fn gumbel_vector<B: Backend, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
    rng: &mut impl Rng,
) -> Tensor<B, D> {
    let n = shape.dims.iter().product();
    let g = Gumbel::<f64>::new(0.0, 1.0).unwrap();
    let v = g.sample_iter(rng).take(n).collect::<Vec<_>>();
    let data = Data::from(v.as_slice());
    Tensor::from_data(data.convert(), device).reshape(shape)
}

fn gumbel_activation<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    inverse_temperature: f64,
    rng: &mut impl Rng,
) -> Tensor<B, D> {
    let shape = tensor.shape();
    let device = tensor.device();

    log_softmax(
        (log_softmax(tensor, dim) + gumbel_vector(shape, &device, rng)) * inverse_temperature,
        dim,
    )
}

fn activation_function<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    inverse_temperature: f64,
    gumbel: bool,
    rng: &mut impl Rng,
) -> Tensor<B, D> {
    if gumbel {
        gumbel_activation(tensor, dim, inverse_temperature, rng)
    } else {
        log_softmax(tensor, dim)
    }
}

#[derive(Debug, Clone)]
pub struct GrammarParameterization<B: Backend> {
    types: Tensor<B, 3>,                //(lexeme, lexeme_pos, type_distribution)
    type_categories: Tensor<B, 3>,      //(lexeme, lexeme_pos, categories_position)
    licensee_categories: Tensor<B, 3>,  //(lexeme, lexeme_pos, categories_position)
    lemmas: Tensor<B, 2>,               //(lexeme, lemma_distribution)
    categories: Tensor<B, 2>,           //(lexeme, categories)
    included_features: Tensor<B, 2>,    //(lexeme, n_features)
    included_licensees: Tensor<B, 2>,   //(lexeme, n_licensees)
    unnormalized_weights: Tensor<B, 1>, //(lexeme)
    include_lemma: Tensor<B, 1>,        // (lexeme)
    dont_include_lemma: Tensor<B, 1>,   // (lexeme)
    lemma_lookups: HashMap<(usize, usize), LogProb<f64>>,
    n_lexemes: usize,
    n_features: usize,
    n_licensees: usize,
    n_categories: usize,
    n_lemmas: usize,
    pad_vector: Tensor<B, 1>, //(n_lemmas)
    end_vector: Tensor<B, 1>, //(n_lemmas)
    silent_probabilities: Tensor<B, 2>,
}

impl<B: Backend> GrammarParameterization<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        types: Tensor<B, 3>,                // (lexeme, n_features, types)
        type_categories: Tensor<B, 3>,      // (lexeme, n_features, N_TYPES)
        licensee_categories: Tensor<B, 3>,  // (lexeme, n_licensee, categories)
        included_features: Tensor<B, 2>,    // (lexeme, n_features)
        included_licensees: Tensor<B, 2>,   // (lexeme, n_licensees)
        lemmas: Tensor<B, 2>,               // (lexeme, n_lemmas)
        silent_probabilities: Tensor<B, 2>, //(lexeme, 2)
        categories: Tensor<B, 2>,           // (lexeme, n_categories)
        weights: Tensor<B, 1>,              // (lexeme)
        include_lemma: Tensor<B, 1>,
        pad_vector: Tensor<B, 1>,
        end_vector: Tensor<B, 1>,
        temperature: f64,
        gumbel: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<GrammarParameterization<B>> {
        let [n_lexemes, n_features, n_categories] = type_categories.shape().dims;
        let n_lemmas = lemmas.shape().dims[1];
        let n_licensees = licensee_categories.shape().dims[1];

        if n_lemmas <= 1 {
            bail!(format!("The model needs at least 2 lemmas, not {n_lemmas} since idx=0 is the silent lemma."))
        }
        if types.shape().dims[2] != N_TYPES {
            bail!(format!(
                "Tensor types must be of shape [n_lexemes, n_features, {N_TYPES}]"
            ));
        }

        let inverse_temperature = 1.0 / temperature;

        let included_features =
            activation_function(included_features, 1, inverse_temperature, gumbel, rng);
        let included_licensees =
            activation_function(included_licensees, 1, inverse_temperature, gumbel, rng);
        let dont_include_lemma = log_sigmoid(-include_lemma.clone());
        let include_lemma = log_sigmoid(include_lemma);

        let types = activation_function(types, 2, inverse_temperature, gumbel, rng);
        let type_categories =
            activation_function(type_categories, 2, inverse_temperature, gumbel, rng);
        let lemmas = activation_function(lemmas, 1, inverse_temperature, gumbel, rng);
        let unnormalized_weights = weights.clone();
        let licensee_categories =
            activation_function(licensee_categories, 2, inverse_temperature, gumbel, rng);
        let categories = activation_function(categories, 1, inverse_temperature, gumbel, rng);

        let pad_vector = log_softmax(pad_vector, 0);
        let end_vector = log_softmax(end_vector, 0);

        let silent_probabilities =
            gumbel_activation(silent_probabilities, 1, inverse_temperature, rng);

        let mut lemma_lookups = HashMap::default();
        for lexeme_i in 0..n_lexemes {
            let v = lemmas
                .clone()
                .slice([lexeme_i..lexeme_i + 1, 0..n_lemmas])
                .to_data()
                .convert::<f64>()
                .value;
            for (lemma_i, v) in v.into_iter().enumerate() {
                lemma_lookups.insert((lexeme_i, lemma_i), LogProb::new(clamp_prob(v)?).unwrap());
            }
        }

        Ok(GrammarParameterization {
            types,
            type_categories,
            lemmas,
            licensee_categories,
            included_features,
            included_licensees,
            include_lemma,
            dont_include_lemma,
            unnormalized_weights,
            lemma_lookups,
            n_lexemes,
            n_features,
            n_licensees,
            n_lemmas,
            n_categories,
            categories,
            pad_vector,
            end_vector,
            silent_probabilities,
        })
    }

    pub fn pad_vector(&self) -> &Tensor<B, 1> {
        &self.pad_vector
    }

    pub fn end_vector(&self) -> &Tensor<B, 1> {
        &self.end_vector
    }

    pub fn licensee_categories(&self) -> &Tensor<B, 3> {
        &self.licensee_categories
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_licensees(&self) -> usize {
        self.n_licensees
    }

    pub fn type_categories(&self) -> &Tensor<B, 3> {
        &self.type_categories
    }

    pub fn types(&self) -> &Tensor<B, 3> {
        &self.types
    }

    pub fn categories(&self) -> &Tensor<B, 2> {
        &self.categories
    }
    pub fn silent_probabilities(&self) -> &Tensor<B, 2> {
        &self.silent_probabilities
    }

    pub fn lemma_lookups(&self) -> &HashMap<(usize, usize), LogProb<f64>> {
        &self.lemma_lookups
    }

    pub fn include_lemma(&self) -> &Tensor<B, 1> {
        &self.include_lemma
    }

    pub fn dont_include_lemma(&self) -> &Tensor<B, 1> {
        &self.dont_include_lemma
    }

    pub fn included_licensees(&self) -> &Tensor<B, 2> {
        &self.included_licensees
    }
    pub fn included_features(&self) -> &Tensor<B, 2> {
        &self.included_features
    }

    pub fn unnormalized_weights(&self) -> &Tensor<B, 1> {
        &self.unnormalized_weights
    }

    pub fn device(&self) -> Device<B> {
        self.types.device()
    }

    pub fn n_lemmas(&self) -> usize {
        self.n_lemmas
    }

    pub fn n_lexemes(&self) -> usize {
        self.n_lexemes
    }

    pub fn n_categories(&self) -> usize {
        self.n_categories
    }

    pub fn lemmas(&self) -> &Tensor<B, 2> {
        &self.lemmas
    }
}
