use std::collections::HashMap;
use std::fmt::Debug;

use crate::ParsingConfig;

use super::*;

use itertools::Itertools;
use simple_semantics::LabelledScenarios;
use simple_semantics::LanguageExpression;
use simple_semantics::lambda::RootedLambdaPool;
use simple_semantics::language::Expr;
use simple_semantics::lot_parser;

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
    pub fn parse(s: &'src str) -> anyhow::Result<(Self, LabelledScenarios)> {
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

impl<T: Eq + Clone + Debug, C: Eq + Clone + Debug> SemanticLexicon<T, C> {
    pub fn interpretation(&self, nx: NodeIndex) -> &RootedLambdaPool<Expr> {
        self.semantic_entries
            .get(&nx)
            .expect("There is no lemma of that node index!")
    }

    pub fn lexicon(&self) -> &Lexicon<T, C> {
        &self.lexicon
    }

    pub fn parse_and_interpret<'a>(
        &'a self,
        initial_category: C,
        sentence: &'a [T],
        config: &'a ParsingConfig,
    ) -> anyhow::Result<
        impl Iterator<
            Item = (
                LogProb<f64>,
                &'a [T],
                impl Iterator<Item = LanguageExpression>,
            ),
        >,
    > {
        Ok(
            crate::Parser::new(&self.lexicon, initial_category, sentence, config)?.map(
                move |(p, s, r)| {
                    (
                        p,
                        s,
                        r.to_interpretation(self)
                            .filter_map(|(pool, _)| pool.into_pool().ok())
                            .collect_vec()
                            .into_iter(),
                    )
                },
            ),
        )
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use logprob::LogProb;

    use super::SemanticLexicon;
    use crate::{Generator, Parser, ParsingConfig};

    #[test]
    fn trivial_montague() -> anyhow::Result<()> {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda e x (lambda e y (some(e, all_e, AgentOf(e, x) & PatientOf(e,y) & p_likes(e))))";
        let (semantic, _scenario) = SemanticLexicon::parse(lexicon)?;

        let (_, _, rules) =
            Parser::new(&semantic.lexicon, "v", &["john", "likes", "mary"], &config)?
                .next()
                .unwrap();
        let (interpretation, mut history) = rules.to_interpretation(&semantic).next().unwrap();
        let interpretation = interpretation.into_pool()?;
        assert_eq!(
            "some(x0,all_e,((AgentOf(x0,a1))&(PatientOf(x0,a0)))&(p0(x0)))",
            interpretation.to_string()
        );

        let latex = rules.to_semantic_latex(&semantic, &history);
        println!("{latex}");
        assert_eq!(
            latex,
            "\\begin{forest}\n[{\\semder{v}{FA} }\n\t[{\\lex{john}{\\cancel{d}}{LexicalEntry} } ]\n\t[{\\semder{\\cancel{=d} v}{FA} }\n\t\t[{\\lex{likes}{\\cancel{d=} =d v}{LexicalEntry} } ]\n\t\t[{\\lex{mary}{\\cancel{d}}{LexicalEntry} } ] ] ]\n\\end{forest}"
        );

        //TODO: Not passing reliably because node2rule is not bijective in the pretty code
        history = history.into_rich(&semantic, &rules);
        let latex = rules.to_semantic_latex(&semantic, &history);
        println!("{latex}");
        assert_eq!(latex, "");
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
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda e x (lambda e y (some(e, all_e, AgentOf(e, x) & PatientOf(e,y) & p_likes(e))))";
        let lexicon = format!(
            "{}\n::=v c::lambda t phi (phi)\n::v= +wh c::lambda t phi (phi)\nknows::c= =d v::lambda <e,t> P (lambda e x (P(x)))\nwho::d -wh::lambda <e,t> P (P)",
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
        let (interpretation, history) = rules.to_interpretation(&semantic).next().unwrap();
        let interpretation = interpretation.into_pool()?;
        assert_eq!(
            interpretation.to_string(),
            "some(x0,all_e,((AgentOf(x0,a1))&(PatientOf(x0,a0)))&(p0(x0)))",
        );

        let latex = rules.to_semantic_latex(&semantic, &history);

        println!("{}", latex);
        assert_eq!(
            latex,
            "\\begin{forest}\n[{\\semder{c}{FA} }\n\t[{\\semder{\\cancel{v}}{FA} }\n\t\t[{\\lex{john}{\\cancel{d}}{LexicalEntry} } ]\n\t\t[{\\semder{\\cancel{=d} v}{FA} }\n\t\t\t[{\\lex{knows}{\\cancel{c=} =d v}{LexicalEntry} } ]\n\t\t\t[{\\semder{\\cancel{c}}{ApplyFromStorage} }\n\t\t\t\t[{$t0$},name=node1 ]\n\t\t\t\t[{\\semder[{\\mover{\\cancel{-wh}}{0}}]{\\cancel{+wh} c}{FA} }\n\t\t\t\t\t[{\\lex{$\\epsilon$}{\\cancel{v=} +wh c}{LexicalEntry} } ]\n\t\t\t\t\t[{\\semder[{\\mover{-wh}{0}}]{\\cancel{v}}{Store} }\n\t\t\t\t\t\t[{\\lex{who}{\\cancel{d} -wh}{LexicalEntry} },name=node2 ]\n\t\t\t\t\t\t[{\\semder{\\cancel{=d} v}{FA} }\n\t\t\t\t\t\t\t[{\\lex{likes}{\\cancel{d=} =d v}{LexicalEntry} } ]\n\t\t\t\t\t\t\t[{\\lex{mary}{\\cancel{d}}{LexicalEntry} } ] ] ] ] ] ] ]\n\t[{\\lex{$\\epsilon$}{\\cancel{=v} c}{LexicalEntry} } ] ]\n\\draw[densely dotted,->] (node2) to[out=west,in=south west] (node1);\n\\end{forest}"
        );
        Ok(())
    }

    #[test]
    fn qr_test() -> anyhow::Result<()> {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lexical = [
            "everyone::d -k -q::lambda <e,t> P (every(x, all_a, P(x)))",
            "someone::d -k -q::lambda <e,t> P (some(x, all_a, P(x)))",
            "likes::d= V::lambda e x (lambda e y (some(e, all_e, AgentOf(e, y)&p_likes(e)&PatientOf(e, x))))",
            "::v= +k +q t::lambda t x (x)",
            "::V= +k d= +q v::lambda <e,t> p (p)",
        ];

        let lexicon = lexical.join("\n");
        let (lex, _scenarios) = SemanticLexicon::parse(&lexicon)?;

        let mut v = vec![];
        for (_, s, rules) in Generator::new(&lex.lexicon, "t", &config)?.take(10) {
            let mut s = s.join(" ");
            for interpretation in rules
                .to_interpretation(&lex)
                .map(|(pool, _)| pool.into_pool().unwrap().to_string())
                .unique()
            {
                s.push('\n');
                s.push_str(&interpretation);
            }
            println!("{}", s);
            v.push(s);
        }
        assert_eq!(
            vec![
                "someone someone likes\nsome(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nsome(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
                "someone everyone likes\nsome(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nevery(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
                "everyone everyone likes\nevery(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nevery(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
                "everyone someone likes\nevery(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nsome(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))"
            ],
            v
        );
        println!("sov good");

        let lexical = [
            "everyone::d -k -q::lambda <e,t> P (every(x, all_a, P(x)))",
            "someone::d -k -q::lambda <e,t> P (some(x, all_a, P(x)))",
            "likes::d= V -v::lambda e x (lambda e y (some(e, all_e, AgentOf(e, y)&p_likes(e)&PatientOf(e, x))))",
            "::v= +v +k +q t::lambda t x (x)",
            "::V= +k d= +q v::lambda <e,t> p (p)",
        ];

        let lexicon = lexical.join("\n");
        let (lex, _scenarios) = SemanticLexicon::parse(&lexicon)?;

        let mut v = vec![];
        for (_, s, rules) in Generator::new(&lex.lexicon, "t", &config)?.take(10) {
            let mut s = s.join(" ");
            for interpretation in rules
                .to_interpretation(&lex)
                .map(|(pool, _)| pool.into_pool().unwrap().to_string())
                .unique()
            {
                s.push('\n');
                s.push_str(&interpretation);
            }
            println!("{}", s);
            v.push(s);
        }
        assert_eq!(
            vec![
                "everyone likes someone\nevery(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nsome(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
                "everyone likes everyone\nevery(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nevery(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
                "someone likes everyone\nsome(x0,all_a,every(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nevery(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
                "someone likes someone\nsome(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x0))&(p0(x2)))&(PatientOf(x2,x1)))))\nsome(x0,all_a,some(x1,all_a,some(x2,all_e,((AgentOf(x2,x1))&(p0(x2)))&(PatientOf(x2,x0)))))",
            ],
            v
        );
        Ok(())
    }
}
