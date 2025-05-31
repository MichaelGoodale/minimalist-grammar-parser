use ahash::HashMap;
use simple_semantics::language::UnprocessedParseTree;
use std::fmt::Debug;

use crate::ParsingConfig;

use super::*;

use itertools::Itertools;
use simple_semantics::LabelledScenarios;
use simple_semantics::LanguageExpression;
use simple_semantics::lambda::RootedLambdaPool;
use simple_semantics::language::{Expr, lot_parser};

#[derive(Debug, Clone)]
pub struct SemanticLexicon<T: Eq, Category: Eq> {
    lexicon: Lexicon<T, Category>,
    semantic_entries: HashMap<NodeIndex, RootedLambdaPool<Expr>>,
}

impl<T: Eq, C: Eq> SemanticLexicon<T, C> {
    pub fn new(
        lexicon: Lexicon<T, C>,
        semantic_entries: HashMap<NodeIndex, RootedLambdaPool<Expr>>,
    ) -> Self {
        SemanticLexicon {
            lexicon,
            semantic_entries,
        }
    }
}

#[allow(clippy::type_complexity)]
fn semantic_grammar_parser<'src>() -> impl Parser<
    'src,
    &'src str,
    anyhow::Result<(
        Lexicon<&'src str, &'src str>,
        Vec<(NodeIndex, UnprocessedParseTree<'src>)>,
    )>,
    extra::Err<Rich<'src, char>>,
> {
    entry_parser()
        .then_ignore(just("::").padded())
        .then(lot_parser().labelled("LOT expression"))
        .separated_by(newline())
        .collect::<Vec<_>>()
        .map(|vec| {
            let (lexical_entries, interpretations): (Vec<_>, Vec<_>) = vec.into_iter().unzip();

            //  Assumes that the leaves iterator goes in order of lexical_entries
            let lexicon = Lexicon::new(lexical_entries);
            let semantic_entries = lexicon
                .leaves
                .iter()
                .copied()
                .zip(interpretations)
                .collect();

            Ok((lexicon, semantic_entries))
        })
        .then_ignore(end())
}

impl<'src> SemanticLexicon<&'src str, &'src str> {
    pub fn parse(s: &'src str) -> anyhow::Result<(Self, LabelledScenarios)> {
        let mut labels = LabelledScenarios::default();

        let (lexicon, semantic_entries) = semantic_grammar_parser()
            .parse(s)
            .into_result()
            .map_err(|x| {
                anyhow::Error::msg(
                    x.into_iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })??;

        let semantic_lexicon = SemanticLexicon {
            lexicon,
            semantic_entries: semantic_entries
                .into_iter()
                .map(|(k, v)| v.to_pool(&mut labels).map(|x| (k, x)))
                .collect::<Result<_>>()?,
        };
        Ok((semantic_lexicon, labels))
    }

    pub fn parse_with_labels(s: &'src str, labels: &mut LabelledScenarios) -> anyhow::Result<Self> {
        let (lexicon, semantic_entries) = semantic_grammar_parser()
            .parse(s)
            .into_result()
            .map_err(|x| {
                anyhow::Error::msg(
                    x.into_iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })??;
        let semantic_lexicon = SemanticLexicon {
            lexicon,
            semantic_entries: semantic_entries
                .into_iter()
                .map(|(k, v)| v.to_pool(labels).map(|x| (k, x)))
                .collect::<Result<_>>()?,
        };
        Ok(semantic_lexicon)
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

    pub fn lexicon_mut(&mut self) -> &mut Lexicon<T, C> {
        &mut self.lexicon
    }

    pub fn interpretations(&self) -> &HashMap<NodeIndex, RootedLambdaPool<Expr>> {
        &self.semantic_entries
    }

    pub fn interpretations_mut(&mut self) -> &mut HashMap<NodeIndex, RootedLambdaPool<Expr>> {
        &mut self.semantic_entries
    }

    pub fn lexicon_and_interpretations_mut(
        &mut self,
    ) -> (
        &mut Lexicon<T, C>,
        &mut HashMap<NodeIndex, RootedLambdaPool<Expr>>,
    ) {
        (&mut self.lexicon, &mut self.semantic_entries)
    }

    pub fn parse_and_interpret<'a>(
        &'a self,
        category: C,
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
        Ok(self
            .lexicon
            .parse(sentence, category, config)?
            .map(move |(p, s, r)| {
                (
                    p,
                    s,
                    r.to_interpretation(self)
                        .filter_map(|(pool, _)| pool.into_pool().ok())
                        .collect_vec()
                        .into_iter(),
                )
            }))
    }
}

impl<T, C> Display for SemanticLexicon<T, C>
where
    T: Eq + Display + std::fmt::Debug + Clone,
    C: Eq + Display + std::fmt::Debug + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.lexicon
                .lexemes()
                .unwrap()
                .iter()
                .zip(self.lexicon.leaves.iter())
                .map(|(l, n)| format!("{l}::{}", self.semantic_entries[n]))
                .join("\n")
        )
    }
}
#[cfg(test)]
mod test {
    use itertools::Itertools;
    use logprob::LogProb;

    use super::SemanticLexicon;
    use crate::ParsingConfig;

    #[test]
    fn trivial_montague() -> anyhow::Result<()> {
        let config: ParsingConfig = ParsingConfig::new(
            LogProb::new(-256.0).unwrap(),
            LogProb::from_raw_prob(0.5).unwrap(),
            100,
            1000,
        );
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda a x (lambda a y (some_e(e, all_e, AgentOf(e, x) & PatientOf(e,y) & pe_likes(e))))";

        //Scenarios are not reliably assigning values!
        let (semantic, _scenario) = SemanticLexicon::parse(lexicon)?;
        let (_, _, rules) = semantic
            .lexicon
            .parse(&["john", "likes", "mary"], "v", &config)?
            .next()
            .unwrap();
        let (interpretation, mut history) = rules.to_interpretation(&semantic).next().unwrap();
        let interpretation = interpretation.into_pool()?;

        #[cfg(feature = "pretty")]
        {
            let latex = rules.to_semantic_latex(&semantic, &history);
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}\n[{\\rulesemder{v}{FA} }\n\t[{\\plainlex{john}{\\cancel{d}} } ]\n\t[{\\rulesemder{\\cancel{=d} v}{FA} }\n\t\t[{\\plainlex{likes}{\\cancel{d=} =d v} } ]\n\t\t[{\\plainlex{mary}{\\cancel{d}} } ] ] ]\n\\end{forest}"
            );

            history = history.into_rich(&semantic, &rules);
            let latex = rules.to_semantic_latex(&semantic, &history);
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}\n[{\\semder{v}{\\semanticRule[FA]{some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,a0)) \\& pe0(x)))}} }\n\t[{\\semlex{john}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a0}} } ]\n\t[{\\semder{\\cancel{=d} v}{\\semanticRule[FA]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,x\\_l)) \\& pe0(x))))}} }\n\t\t[{\\semlex{likes}{\\cancel{d=} =d v}{\\semanticRule[LexicalEntry]{{$\\lambda_{a}$}x\\_l ({$\\lambda_{a}$}y\\_l (some\\_e(x,all\\_e,((AgentOf(x,x\\_l) \\& PatientOf(x,y\\_l)) \\& pe0(x)))))}} } ]\n\t\t[{\\semlex{mary}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a1}} } ] ] ]\n\\end{forest}"
            );
        }
        assert_eq!(
            "some_e(x,all_e,((AgentOf(x,a1) & PatientOf(x,a0)) & pe0(x)))",
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
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda a x (lambda a y (some_e(e, all_e, AgentOf(e, x) & PatientOf(e,y) & pe_likes(e))))";
        let lexicon = format!(
            "{}\n::=v c::lambda t phi (phi)\n::v= +wh c::lambda t phi (phi)\nknows::c= =d v::lambda <a,t> P (lambda a x (P(x)))\nwho::d -wh::lambda <a,t> P (P)",
            lexicon
        );
        let (semantic, _scenario) = SemanticLexicon::parse(&lexicon)?;

        let (_, _, rules) = semantic
            .lexicon
            .parse(&["john", "knows", "who", "likes", "mary"], "c", &config)?
            .next()
            .unwrap();
        dbg!(&rules);
        let (interpretation, mut history) = rules.to_interpretation(&semantic).next().unwrap();
        let interpretation = interpretation.into_pool()?;
        assert_eq!(
            interpretation.to_string(),
            "some_e(x,all_e,((AgentOf(x,a1) & PatientOf(x,a0)) & pe0(x)))",
        );
        #[cfg(feature = "pretty")]
        {
            let latex = rules.to_semantic_latex(&semantic, &history);

            println!("{}", latex);
            assert_eq!(
                latex,
                "\\begin{forest}\n[{\\rulesemder{c}{FA} }\n\t[{\\rulesemder{\\cancel{v}}{FA} }\n\t\t[{\\plainlex{john}{\\cancel{d}} } ]\n\t\t[{\\rulesemder{\\cancel{=d} v}{FA} }\n\t\t\t[{\\plainlex{knows}{\\cancel{c=} =d v} } ]\n\t\t\t[{\\rulesemder{\\cancel{c}}{ApplyFromStorage} }\n\t\t\t\t[{$t0$},name=node1 ]\n\t\t\t\t[{\\rulesemder[{\\mover{\\cancel{-wh}}{0}}]{\\cancel{+wh} c}{FA} }\n\t\t\t\t\t[{\\plainlex{$\\epsilon$}{\\cancel{v=} +wh c} } ]\n\t\t\t\t\t[{\\rulesemder[{\\mover{-wh}{0}}]{\\cancel{v}}{Store} }\n\t\t\t\t\t\t[{\\plainlex{who}{\\cancel{d} -wh} },name=node2 ]\n\t\t\t\t\t\t[{\\rulesemder{\\cancel{=d} v}{FA} }\n\t\t\t\t\t\t\t[{\\plainlex{likes}{\\cancel{d=} =d v} } ]\n\t\t\t\t\t\t\t[{\\plainlex{mary}{\\cancel{d}} } ] ] ] ] ] ] ]\n\t[{\\plainlex{$\\epsilon$}{\\cancel{=v} c} } ] ]\n\\draw[densely dotted,->] (node2) to[out=west,in=south west] (node1);\n\\end{forest}"
            );

            history = history.into_rich(&semantic, &rules);
            let latex = rules.to_semantic_latex(&semantic, &history);
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}\n[{\\semder{c}{\\semanticRule[FA]{some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,a0)) \\& pe0(x)))}} }\n\t[{\\semder{\\cancel{v}}{\\semanticRule[FA]{some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,a0)) \\& pe0(x)))}} }\n\t\t[{\\semlex{john}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a0}} } ]\n\t\t[{\\semder{\\cancel{=d} v}{\\semanticRule[FA]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,x\\_l)) \\& pe0(x))))}} }\n\t\t\t[{\\semlex{knows}{\\cancel{c=} =d v}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}x\\_l ({$\\lambda_{a}$}y\\_l ((x\\_l)(y\\_l)))}} } ]\n\t\t\t[{\\semder{\\cancel{c}}{\\semanticRule[ApplyFromStorage]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,x\\_l)) \\& pe0(x))))}} }\n\t\t\t\t[{$t0$},name=node1 ]\n\t\t\t\t[{\\semder[{\\mover{\\cancel{-wh}}{0}}]{\\cancel{+wh} c}{\\semanticRule[FA]{some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,0\\_f)) \\& pe0(x)))}} }\n\t\t\t\t\t[{\\semlex{$\\epsilon$}{\\cancel{v=} +wh c}{\\semanticRule[LexicalEntry]{{$\\lambda_{t}$}x\\_l (x\\_l)}} } ]\n\t\t\t\t\t[{\\semder[{\\mover{-wh}{0}}]{\\cancel{v}}{\\semanticRule[Store]{some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,0\\_f)) \\& pe0(x)))}} }\n\t\t\t\t\t\t[{\\semlex{who}{\\cancel{d} -wh}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}x\\_l (x\\_l)}} },name=node2 ]\n\t\t\t\t\t\t[{\\semder{\\cancel{=d} v}{\\semanticRule[FA]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,a1) \\& PatientOf(x,x\\_l)) \\& pe0(x))))}} }\n\t\t\t\t\t\t\t[{\\semlex{likes}{\\cancel{d=} =d v}{\\semanticRule[LexicalEntry]{{$\\lambda_{a}$}x\\_l ({$\\lambda_{a}$}y\\_l (some\\_e(x,all\\_e,((AgentOf(x,x\\_l) \\& PatientOf(x,y\\_l)) \\& pe0(x)))))}} } ]\n\t\t\t\t\t\t\t[{\\semlex{mary}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a1}} } ] ] ] ] ] ] ]\n\t[{\\semlex{$\\epsilon$}{\\cancel{=v} c}{\\semanticRule[LexicalEntry]{{$\\lambda_{t}$}x\\_l (x\\_l)}} } ] ]\n\\draw[densely dotted,->] (node2) to[out=west,in=south west] (node1);\n\\end{forest}"
            );
        }
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
            "everyone::d -k -q::lambda <a,t> P (every(x, all_a, P(x)))",
            "someone::d -k -q::lambda <a,t> P (some(x, all_a, P(x)))",
            "likes::d= V::lambda a x (lambda a y (some_e(e, all_e, AgentOf(e, y)&pe_likes(e)&PatientOf(e, x))))",
            "::v= +k +q t::lambda t x (x)",
            "::V= +k d= +q v::lambda <a,t> p (p)",
        ];

        let lexicon = lexical.join("\n");
        let (lex, _scenarios) = SemanticLexicon::parse(&lexicon)?;

        let mut v = vec![];
        for (_, s, rules) in lex.lexicon.generate("t", &config)?.take(10) {
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
                "someone someone likes\nsome(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nsome(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
                "someone everyone likes\nsome(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nevery(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
                "everyone everyone likes\nevery(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nevery(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
                "everyone someone likes\nevery(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nsome(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))"
            ],
            v
        );
        println!("sov good");

        let lexical = [
            "everyone::d -k -q::lambda <a,t> P (every(x, all_a, P(x)))",
            "someone::d -k -q::lambda <a,t> P (some(x, all_a, P(x)))",
            "likes::d= V -v::lambda a x (lambda a y (some_e(e, all_e, AgentOf(e, y)&pe_likes(e)&PatientOf(e, x))))",
            "::v= +v +k +q t::lambda t x (x)",
            "::V= +k d= +q v::lambda <a,t> p (p)",
        ];

        let lexicon = lexical.join("\n");
        let (lex, _scenarios) = SemanticLexicon::parse(&lexicon)?;

        let mut v = vec![];
        for (_, s, rules) in lex.lexicon.generate("t", &config)?.take(10) {
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
                "everyone likes someone\nevery(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nsome(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
                "everyone likes everyone\nevery(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nevery(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
                "someone likes everyone\nsome(x,all_a,every(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nevery(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
                "someone likes someone\nsome(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,x) & pe0(z)) & PatientOf(z,y)))))\nsome(x,all_a,some(y,all_a,some_e(z,all_e,((AgentOf(z,y) & pe0(z)) & PatientOf(z,x)))))",
            ],
            v
        );

        #[cfg(feature = "pretty")]
        {
            let (_, _, rules) = lex
                .lexicon
                .parse(&["everyone", "likes", "someone"], "t", &config)?
                .next()
                .unwrap();

            let (_, mut history) = rules.to_interpretation(&lex).next().unwrap();
            let latex = rules.to_semantic_latex(&lex, &history);

            println!("{}", latex);
            assert_eq!(
                latex,
                "\\begin{forest}\n[{\\rulesemder{t}{ApplyFromStorage} }\n\t[{$t0$},name=node0 ]\n\t[{\\rulesemder[{\\mover{\\cancel{-q}}{0}}]{\\cancel{+q} t}{UpdateTrace} }\n\t\t[{$t1$},name=node1 ]\n\t\t[{\\rulesemder[{\\mover[{-q}]{\\cancel{-k}}{1}}]{\\cancel{+k} +q t}{Id} }\n\t\t\t[{$t3$},name=node2 ]\n\t\t\t[{\\rulesemder[{\\mover{\\cancel{-v}}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{+v} +k +q t}{FA} }\n\t\t\t\t[{\\plainlex{$\\epsilon$}{\\cancel{v=} +v +k +q t} } ]\n\t\t\t\t[{\\rulesemder[{\\mover{-v}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{v}}{ApplyFromStorage} }\n\t\t\t\t\t[{$t2$},name=node3 ]\n\t\t\t\t\t[{\\rulesemder[{\\mover{\\cancel{-q}}{2}, \\mover{-v}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{+q} v}{Store} }\n\t\t\t\t\t\t[{\\rulesemder[{\\mover{-q}{2}, \\mover{-v}{3}}]{\\cancel{d=} +q v}{UpdateTrace} }\n\t\t\t\t\t\t\t[{$t4$},name=node5 ]\n\t\t\t\t\t\t\t[{\\rulesemder[{\\mover[{-q}]{\\cancel{-k}}{4}, \\mover{-v}{3}}]{\\cancel{+k} d= +q v}{FA} }\n\t\t\t\t\t\t\t\t[{\\plainlex{$\\epsilon$}{\\cancel{V=} +k d= +q v} } ]\n\t\t\t\t\t\t\t\t[{\\rulesemder[{\\mover[{-q}]{-k}{4}}]{\\cancel{V} -v}{Store} },name=node8\n\t\t\t\t\t\t\t\t\t[{\\plainlex{likes}{\\cancel{d=} V -v} } ]\n\t\t\t\t\t\t\t\t\t[{\\plainlex{someone}{\\cancel{d} -k -q} },name=node6 ] ] ] ]\n\t\t\t\t\t\t[{\\plainlex{everyone}{\\cancel{d} -k -q} },name=node4 ] ] ] ] ] ] ]\n\\draw[densely dotted,->] (node1) to[out=west,in=south west] (node0);\n\\draw[densely dotted,->] (node4) to[out=west,in=south west] (node1);\n\\draw[densely dotted,->] (node5) to[out=west,in=south west] (node3);\n\\draw[densely dotted,->] (node6) to[out=west,in=south west] (node5);\n\\draw[densely dotted,->] (node8) to[out=west,in=south west] (node2);\n\\end{forest}"
            );

            history = history.into_rich(&lex, &rules);
            let latex = rules.to_semantic_latex(&lex, &history);
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}\n[{\\semder{t}{\\semanticRule[ApplyFromStorage]{every(x,all\\_a,some(y,all\\_a,some\\_e(z,all\\_e,((AgentOf(z,x) \\& pe0(z)) \\& PatientOf(z,y)))))}} }\n\t[{$t0$},name=node0 ]\n\t[{\\semder[{\\mover{\\cancel{-q}}{0}}]{\\cancel{+q} t}{\\semanticRule[UpdateTrace]{some(x,all\\_a,some\\_e(y,all\\_e,((AgentOf(y,0\\_f) \\& pe0(y)) \\& PatientOf(y,x))))}} }\n\t\t[{$t1$},name=node1 ]\n\t\t[{\\semder[{\\mover[{-q}]{\\cancel{-k}}{1}}]{\\cancel{+k} +q t}{\\semanticRule[Id]{some(x,all\\_a,some\\_e(y,all\\_e,((AgentOf(y,1\\_f) \\& pe0(y)) \\& PatientOf(y,x))))}} }\n\t\t\t[{$t3$},name=node2 ]\n\t\t\t[{\\semder[{\\mover{\\cancel{-v}}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{+v} +k +q t}{\\semanticRule[FA]{some(x,all\\_a,some\\_e(y,all\\_e,((AgentOf(y,1\\_f) \\& pe0(y)) \\& PatientOf(y,x))))}} }\n\t\t\t\t[{\\semlex{$\\epsilon$}{\\cancel{v=} +v +k +q t}{\\semanticRule[LexicalEntry]{{$\\lambda_{t}$}x\\_l (x\\_l)}} } ]\n\t\t\t\t[{\\semder[{\\mover{-v}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{v}}{\\semanticRule[ApplyFromStorage]{some(x,all\\_a,some\\_e(y,all\\_e,((AgentOf(y,1\\_f) \\& pe0(y)) \\& PatientOf(y,x))))}} }\n\t\t\t\t\t[{$t2$},name=node3 ]\n\t\t\t\t\t[{\\semder[{\\mover{\\cancel{-q}}{2}, \\mover{-v}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{+q} v}{\\semanticRule[Store]{some\\_e(x,all\\_e,((AgentOf(x,1\\_f) \\& pe0(x)) \\& PatientOf(x,2\\_f)))}} }\n\t\t\t\t\t\t[{\\semder[{\\mover{-q}{2}, \\mover{-v}{3}}]{\\cancel{d=} +q v}{\\semanticRule[UpdateTrace]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,x\\_l) \\& pe0(x)) \\& PatientOf(x,2\\_f))))}} }\n\t\t\t\t\t\t\t[{$t4$},name=node5 ]\n\t\t\t\t\t\t\t[{\\semder[{\\mover[{-q}]{\\cancel{-k}}{4}, \\mover{-v}{3}}]{\\cancel{+k} d= +q v}{\\semanticRule[FA]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,x\\_l) \\& pe0(x)) \\& PatientOf(x,4\\_f))))}} }\n\t\t\t\t\t\t\t\t[{\\semlex{$\\epsilon$}{\\cancel{V=} +k d= +q v}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}x\\_l (x\\_l)}} } ]\n\t\t\t\t\t\t\t\t[{\\semder[{\\mover[{-q}]{-k}{4}}]{\\cancel{V} -v}{\\semanticRule[Store]{{$\\lambda_{a}$}x\\_l (some\\_e(x,all\\_e,((AgentOf(x,x\\_l) \\& pe0(x)) \\& PatientOf(x,4\\_f))))}} },name=node8\n\t\t\t\t\t\t\t\t\t[{\\semlex{likes}{\\cancel{d=} V -v}{\\semanticRule[LexicalEntry]{{$\\lambda_{a}$}x\\_l ({$\\lambda_{a}$}y\\_l (some\\_e(x,all\\_e,((AgentOf(x,y\\_l) \\& pe0(x)) \\& PatientOf(x,x\\_l)))))}} } ]\n\t\t\t\t\t\t\t\t\t[{\\semlex{someone}{\\cancel{d} -k -q}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}x\\_l (some(x,all\\_a,(x\\_l)(x)))}} },name=node6 ] ] ] ]\n\t\t\t\t\t\t[{\\semlex{everyone}{\\cancel{d} -k -q}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}x\\_l (every(x,all\\_a,(x\\_l)(x)))}} },name=node4 ] ] ] ] ] ] ]\n\\draw[densely dotted,->] (node1) to[out=west,in=south west] (node0);\n\\draw[densely dotted,->] (node4) to[out=west,in=south west] (node1);\n\\draw[densely dotted,->] (node5) to[out=west,in=south west] (node3);\n\\draw[densely dotted,->] (node6) to[out=west,in=south west] (node5);\n\\draw[densely dotted,->] (node8) to[out=west,in=south west] (node2);\n\\end{forest}"
            );
        }
        Ok(())
    }

    #[test]
    fn obscure_error_with_rich() -> anyhow::Result<()> {
        let grammar = "ε::0= =2 +1 0::lambda a x (pa0(x))
ran::2::lambda t x_l (a1)
John::0 -1::a1";

        let (lexicon, _) = SemanticLexicon::parse(grammar)?;
        for (_, _, r) in lexicon
            .lexicon
            .parse(&["John", "ran"], "0", &ParsingConfig::default())?
        {
            for (pool, h) in r.to_interpretation(&lexicon) {
                pool.into_pool()?;
                h.into_rich(&lexicon, &r);
            }
        }
        Ok(())
    }
}
