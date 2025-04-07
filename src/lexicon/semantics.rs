use std::collections::HashMap;

use super::*;

use simple_semantics::lambda::LambdaExprRef;
use simple_semantics::lambda::LambdaPool;
use simple_semantics::lambda::RootedLambdaPool;
use simple_semantics::language::Expr;
use simple_semantics::lot_parser;
use simple_semantics::LabelledScenarios;

#[derive(Debug, Clone)]
pub struct SemanticLexicon<T: Eq, Category: Eq> {
    lexicon: Lexicon<T, Category>,
    semantic_entries: HashMap<NodeIndex, RootedLambdaPool<Expr>>,
}

fn semantic_grammar_parser<'src>() -> impl Parser<
    'src,
    &'src str,
    SemanticLexicon<&'src str, &'src str>,
    extra::Full<Rich<'src, char>, extra::SimpleState<LabelledScenarios>, ()>,
> {
    entry_parser::<extra::Full<Rich<char>, extra::SimpleState<LabelledScenarios>, ()>>()
        .then_ignore(just("::").padded())
        .then(lot_parser().labelled("LOT expression"))
        .separated_by(newline())
        .collect::<Vec<_>>()
        .map(|vec| {
            let (lexical_entries, interpretations): (Vec<_>, Vec<_>) = vec.into_iter().unzip();

            //  Assumes that the leaves iterator goes in order of lexical_entries
            let lexicon = Lexicon::new(lexical_entries);
            let mut semantic_entries = HashMap::default();
            for ((leaf, _), entry) in lexicon.leaves.iter().zip(interpretations.into_iter()) {
                semantic_entries.insert(*leaf, entry);
            }

            SemanticLexicon {
                lexicon,
                semantic_entries,
            }
        })
        .then_ignore(end())
}

impl<'src> SemanticLexicon<&'src str, &'src str> {
    fn parse(s: &'src str) -> anyhow::Result<(Self, LabelledScenarios)> {
        let mut state = extra::SimpleState(LabelledScenarios::default());

        Ok((
            semantic_grammar_parser()
                .parse_with_state(s, &mut state)
                .into_result()
                .map_err(|x| {
                    anyhow::Error::msg(
                        x.into_iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    )
                })?,
            state.0,
        ))
    }
}

impl<T: Eq, C: Eq> SemanticLexicon<T, C> {
    pub fn interpretation(&self, nx: NodeIndex) -> &RootedLambdaPool<Expr> {
        self.semantic_entries
            .get(&nx)
            .expect("There is no lemma of that node index!")
    }
}

#[cfg(test)]
mod test {
    use logprob::LogProb;

    use super::SemanticLexicon;
    use crate::{Parser, ParsingConfig};

    #[test]
    fn trivial_montague() -> anyhow::Result<()> {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda <e,<e,t>> x ((lambda <e,t> y (some(e, all_e, AgentOf(e, x) & PatientOf(e,y) & p_likes(e)))))";
        let (semantic, _scenario) = SemanticLexicon::parse(lexicon)?;

        let (_, _, rules) =
            Parser::new(&semantic.lexicon, "v", &["john", "likes", "mary"], &config)?
                .next()
                .unwrap();
        let interpretation = rules.to_interpretation(&semantic)?;
        let interpretation = interpretation.into_pool()?;
        assert_eq!(
            "some(x0,all_e,((AgentOf(x0,a1))&(PatientOf(x0,a0)))&(p0(x0)))",
            interpretation.to_string()
        );
        Ok(())
    }

    #[test]
    fn moving_montague() -> anyhow::Result<()> {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda <e,<e,t>> x ((lambda <e,t> y (some(e, all_e, AgentOf(e, x) & PatientOf(e,y) & p_likes(e)))))";
        let lexicon = format!(
            "{}\n::=v c::lambda <t,t> x (x)\n::v= +wh c::lambda <t,t> x (x)\nknows::c= =d v::lambda <<e,t>,<e,t>> P (lambda <e,t> x (P(x)))\nwho::d -wh::lambda <<e,t>,<e,t>> P (P)",
            lexicon
        );
        let (semantic, _scenario) = SemanticLexicon::parse(&lexicon)?;

        let (_, _, rules) = Parser::new(
            &semantic.lexicon,
            "c",
            &["john", "knows", "who", "likes", "mary"],
            &config,
        )?
        .next()
        .unwrap();
        dbg!(&rules);
        let interpretation = rules.to_interpretation(&semantic)?;
        let interpretation = interpretation.into_pool()?;
        assert_eq!(
            "some(x0,all_e,((AgentOf(x0,a1))&(PatientOf(x0,a0)))&(p0(x0)))",
            interpretation.to_string()
        );
        Ok(())
    }
}
