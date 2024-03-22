use super::neural_beam::{StringPath, StringProbHistory};
use super::neural_lexicon::{
    GrammarParameterization, NeuralFeature, NeuralLexicon, NeuralProbabilityRecord,
};
use super::utils::log_sum_exp_dim;
use crate::{NeuralGenerator, ParsingConfig};
use burn::tensor::Int;
use burn::tensor::{backend::Backend, Tensor};
use moka::sync::Cache;

pub struct NeuralConfig {
    pub n_grammars: usize,
    pub n_strings_per_grammar: usize,
    pub n_strings_to_sample: usize,
    pub padding_length: usize,
    pub negative_weight: Option<f64>,
    pub temperature: f64,
    pub parsing_config: ParsingConfig,
}

///Not technically correct as it treats the number of categories as independent from the expected
///depth.
//fn expected_mdl_score<B: Backend>(
//    types: Tensor<B, 3>,
//    categories: Tensor<B, 3>,
//    lemma_inclusion: Tensor<B, 3>,
//) -> B::FloatElem {
//    todo!();
//}

fn retrieve_strings<B: Backend>(
    lexicon: &NeuralLexicon<B>,
    neural_config: &NeuralConfig,
) -> (Vec<StringPath>, Vec<StringProbHistory>) {
    let mut grammar_strings: Vec<_> = vec![];
    let mut string_paths: Vec<_> = vec![];

    for (s, h) in NeuralGenerator::new(lexicon, &neural_config.parsing_config)
        .filter(|(s, _h)| s.len() < neural_config.padding_length)
        .take(neural_config.n_strings_per_grammar)
    {
        string_paths.push(h);
        grammar_strings.push(s);
    }
    (grammar_strings, string_paths)
}

fn string_path_to_tensor<B: Backend>(
    strings: &[StringPath],
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
) -> Tensor<B, 3> {
    let mut s_tensor: Tensor<B, 3> = g
        .pad_vector()
        .clone()
        .unsqueeze_dim::<2>(0)
        .repeat(0, neural_config.padding_length)
        .unsqueeze_dim(0)
        .repeat(0, neural_config.n_strings_per_grammar);

    for (s_i, s) in strings.iter().enumerate() {
        for (w_i, lexeme) in s.iter().enumerate() {
            let values = g
                .lemmas()
                .clone()
                .slice([*lexeme..lexeme + 1, 0..g.n_lemmas()])
                .unsqueeze_dim(0);
            s_tensor = s_tensor.slice_assign([s_i..s_i + 1, w_i..w_i + 1, 0..g.n_lemmas()], values)
        }
        s_tensor = s_tensor.slice_assign(
            [s_i..s_i + 1, s.len()..s.len() + 1, 0..g.n_lemmas()],
            g.end_vector().clone().unsqueeze_dims(&[0, 1]),
        )
    }
    s_tensor
}

const EPSILON: f64 = -10.0;
const EPSILON_INV: f64 = -0.000045400960370489214;

fn get_string_prob<B: Backend>(
    string_paths: &[StringProbHistory],
    lexicon: &NeuralLexicon<B>,
    neural_config: &NeuralConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let move_p: f64 = neural_config.parsing_config.move_prob.into_inner();
    let merge_p: f64 = neural_config
        .parsing_config
        .move_prob
        .opposite_prob()
        .into_inner();

    let n_strings: f64 = (string_paths.len() as f64).ln();
    let n_fakes: f64 = ((neural_config.n_strings_per_grammar - string_paths.len()) as f64).ln();
    let mut string_path_tensor = Tensor::<B, 1>::full(
        [neural_config.n_strings_per_grammar],
        EPSILON_INV - n_strings,
        device,
    );

    if neural_config.n_strings_per_grammar > string_paths.len() {
        string_path_tensor = string_path_tensor.slice_assign(
            [string_paths.len()..neural_config.n_strings_per_grammar],
            Tensor::<B, 1>::full(
                [neural_config.n_strings_per_grammar - string_paths.len()],
                EPSILON - n_fakes,
                device,
            ),
        );
    }

    for (i, string_path) in string_paths.iter().enumerate() {
        let mut p: Tensor<B, 1> = Tensor::zeros([1], device);
        for (prob_type, count) in string_path.iter() {
            match prob_type {
                NeuralProbabilityRecord::OneProb => (),
                NeuralProbabilityRecord::MoveRuleProb => p = p.add_scalar(move_p * (*count as f64)),
                NeuralProbabilityRecord::MergeRuleProb => {
                    p = p.add_scalar(merge_p * (*count as f64))
                }
                NeuralProbabilityRecord::Feature(lexeme) => {
                    let feature = NeuralProbabilityRecord::Feature(*lexeme);
                    p = p.add(
                        lexicon
                            .get_weight(&feature)
                            .unwrap()
                            .clone()
                            .mul_scalar(*count),
                    );
                }
                NeuralProbabilityRecord::EdgeAndFeature((n, e)) => {
                    let e = NeuralProbabilityRecord::Edge(*e);
                    let n = NeuralProbabilityRecord::Feature(*n);
                    p = p.add(
                        (lexicon.get_weight(&n).unwrap().clone()
                            + lexicon.get_weight(&e).unwrap().clone())
                        .mul_scalar(*count),
                    );
                }
                NeuralProbabilityRecord::Edge(_) => {
                    panic!("This should never be in a parse path")
                }
            }
        }
        string_path_tensor = string_path_tensor.slice_assign([i..i + 1], p);
    }
    string_path_tensor
}

pub type NeuralGrammarCache =
    Cache<Vec<Vec<NeuralFeature>>, (Vec<StringPath>, Vec<StringProbHistory>)>;

pub fn get_grammar<B: Backend>(
    g: &GrammarParameterization<B>,
    neural_config: &NeuralConfig,
) -> (Tensor<B, 3>, Tensor<B, 1>) {
    let lexicon = NeuralLexicon::new_superimposed(g);
    let (strings, string_probs) = retrieve_strings(&lexicon, neural_config);

    //(1, n_grammar_strings)
    let string_probs = get_string_prob(&string_probs, &lexicon, neural_config, &g.device());

    //(n_grammar_strings, padding_length, n_lemmas)
    (
        string_path_to_tensor(&strings, g, neural_config),
        string_probs,
    )
}

pub fn get_neural_outputs<B: Backend>(
    g: &GrammarParameterization<B>,
    targets: Tensor<B, 2, Int>,
    neural_config: &NeuralConfig,
) -> Tensor<B, 1> {
    let n_targets = targets.shape().dims[0];

    let lexicon = NeuralLexicon::new_superimposed(g);

    let (strings, string_probs) = retrieve_strings(&lexicon, neural_config);

    //(n_grammar_strings, padding_length, n_lemmas)
    let grammar = string_path_to_tensor(&strings, g, neural_config);

    //(n_strings_per_grammar);
    let string_probs = if strings.is_empty() {
        Tensor::full(
            [neural_config.n_strings_per_grammar],
            EPSILON - (neural_config.n_strings_per_grammar as f64).ln(),
            &g.device(),
        )
    } else {
        get_string_prob(&string_probs, &lexicon, neural_config, &targets.device())
    };

    //(n_targets, n_grammar_strings, padding_length, n_lemmas)
    let grammar: Tensor<B, 4> = grammar.unsqueeze_dim(0).repeat(0, n_targets);

    //(n_targets, n_strings_per_grammar, padding_length, 1)
    let targets: Tensor<B, 4, Int> = targets
        .unsqueeze_dims(&[1, 3])
        .repeat(1, neural_config.n_strings_per_grammar);

    //Probability of generating every target for each string.
    //(n_targets, n_grammar_strings)
    let loss: Tensor<B, 2> = grammar
        .gather(3, targets)
        .squeeze::<3>(3)
        .sum_dim(2)
        .squeeze(2)
        + string_probs.unsqueeze_dim(0);

    //Probability of generating each of the targets
    let loss: Tensor<B, 1> = log_sum_exp_dim(loss, 1);
    -loss.sum()
}
