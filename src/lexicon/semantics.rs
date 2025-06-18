use ahash::HashMap;
use simple_semantics::language::LambdaParseError;
use std::fmt::Debug;

use crate::ParsingConfig;

use super::*;

use itertools::Itertools;
use simple_semantics::LanguageExpression;
use simple_semantics::lambda::RootedLambdaPool;
use simple_semantics::language::Expr;

#[derive(Debug, Clone)]
pub struct SemanticLexicon<'src, T: Eq, Category: Eq> {
    lexicon: Lexicon<T, Category>,
    semantic_entries: HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>>,
}

impl<'src, T: Eq, C: Eq> SemanticLexicon<'src, T, C> {
    pub fn new(
        lexicon: Lexicon<T, C>,
        semantic_entries: HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>>,
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
    (
        Lexicon<&'src str, &'src str>,
        Vec<(
            NodeIndex,
            Result<RootedLambdaPool<'src, Expr<'src>>, LambdaParseError>,
        )>,
    ),
    extra::Err<Rich<'src, char>>,
> {
    entry_parser()
        .then_ignore(just("::").padded())
        .then(
            any()
                .and_is(newline().not())
                .repeated()
                .to_slice()
                .map(RootedLambdaPool::parse),
        )
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

            (lexicon, semantic_entries)
        })
        .then_ignore(end())
}

impl<'src> SemanticLexicon<'src, &'src str, &'src str> {
    pub fn parse(s: &'src str) -> Result<Self, LambdaParseError> {
        let (lexicon, semantic_entries) = semantic_grammar_parser().parse(s).into_result()?;

        let semantic_lexicon = SemanticLexicon {
            lexicon,
            semantic_entries: semantic_entries
                .into_iter()
                .map(|(k, v)| v.map(|v| (k, v)))
                .collect::<Result<_, _>>()?,
        };
        Ok(semantic_lexicon)
    }
}

impl<'src, T: Eq + Clone + Debug, C: Eq + Clone + Debug> SemanticLexicon<'src, T, C> {
    pub fn interpretation(&self, nx: NodeIndex) -> &RootedLambdaPool<'src, Expr<'src>> {
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

    pub fn interpretations(&self) -> &HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>> {
        &self.semantic_entries
    }

    pub fn interpretations_mut(
        &mut self,
    ) -> &mut HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>> {
        &mut self.semantic_entries
    }

    pub fn lexicon_and_interpretations_mut(
        &mut self,
    ) -> (
        &mut Lexicon<T, C>,
        &mut HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>>,
    ) {
        (&mut self.lexicon, &mut self.semantic_entries)
    }

    #[allow(clippy::type_complexity)]
    pub fn parse_and_interpret<'a, 'b: 'a>(
        &'a self,
        category: C,
        sentence: &'b [T],
        config: &'b ParsingConfig,
    ) -> Result<
        impl Iterator<
            Item = (
                LogProb<f64>,
                &'a [T],
                impl Iterator<Item = LanguageExpression<'src>>,
            ),
        >,
        ParsingError<C>,
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

impl<T, C> Display for SemanticLexicon<'_, T, C>
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
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda a x (lambda a y (some_e(e, all_e, AgentOf(y, e) & PatientOf(x, e) & pe_likes(e))))";

        //Scenarios are not reliably assigning values!
        let semantic = SemanticLexicon::parse(lexicon)?;
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
                "\\begin{forest}\n[{\\semder{v}{\\semanticRule[FA]{some\\_e(x, all\\_e, AgentOf(a\\_j, x) \\& PatientOf(a\\_m, x) \\& pe\\_likes(x))}} }\n\t[{\\semlex{john}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a\\_j}} } ]\n\t[{\\semder{\\cancel{=d} v}{\\semanticRule[FA]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& PatientOf(a\\_m, y) \\& pe\\_likes(y))}} }\n\t\t[{\\semlex{likes}{\\cancel{d=} =d v}{\\semanticRule[LexicalEntry]{{$\\lambda_{a}$}x {$\\lambda_{a}$}y some\\_e(z, all\\_e, AgentOf(y, z) \\& PatientOf(x, z) \\& pe\\_likes(z))}} } ]\n\t\t[{\\semlex{mary}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a\\_m}} } ] ] ]\n\\end{forest}"
            );
        }
        assert_eq!(
            "some_e(x, all_e, AgentOf(a_j, x) & PatientOf(a_m, x) & pe_likes(x))",
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
        let lexicon = "john::d::a_j\nmary::d::a_m\nlikes::d= =d v::lambda a x (lambda a y (some_e(e, all_e, AgentOf(x, e) & PatientOf(y, e) & pe_likes(e))))";
        let lexicon = format!(
            "{}\n::=v c::lambda t phi (phi)\n::v= +wh c::lambda t phi (phi)\nknows::c= =d v::lambda <a,t> P (lambda a x (P(x)))\nwho::d -wh::lambda <a,t> P (P)",
            lexicon
        );

        let s = lexicon.as_str();
        let semantic = SemanticLexicon::parse(s)?;

        let (_, _, rules) = semantic
            .lexicon()
            .parse(&["john", "knows", "who", "likes", "mary"], "c", &config)
            .map_err(|x| x.inner_into::<String>())?
            .next()
            .unwrap();
        dbg!(&rules);
        let (interpretation, mut history) = rules.to_interpretation(&semantic).next().unwrap();
        let interpretation = interpretation.into_pool()?;
        assert_eq!(
            interpretation.to_string(),
            "some_e(x, all_e, AgentOf(a_m, x) & PatientOf(a_j, x) & pe_likes(x))"
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
                "\\begin{forest}\n[{\\semder{c}{\\semanticRule[FA]{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(a\\_j, x) \\& pe\\_likes(x))}} }\n\t[{\\semder{\\cancel{v}}{\\semanticRule[FA]{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(a\\_j, x) \\& pe\\_likes(x))}} }\n\t\t[{\\semlex{john}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a\\_j}} } ]\n\t\t[{\\semder{\\cancel{=d} v}{\\semanticRule[FA]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(a\\_m, y) \\& PatientOf(x, y) \\& pe\\_likes(y))}} }\n\t\t\t[{\\semlex{knows}{\\cancel{c=} =d v}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P {$\\lambda_{a}$}x P(x)}} } ]\n\t\t\t[{\\semder{\\cancel{c}}{\\semanticRule[ApplyFromStorage]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(a\\_m, y) \\& PatientOf(x, y) \\& pe\\_likes(y))}} }\n\t\t\t\t[{$t0$},name=node1 ]\n\t\t\t\t[{\\semder[{\\mover{\\cancel{-wh}}{0}}]{\\cancel{+wh} c}{\\semanticRule[FA]{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(0\\#a, x) \\& pe\\_likes(x))}} }\n\t\t\t\t\t[{\\semlex{$\\epsilon$}{\\cancel{v=} +wh c}{\\semanticRule[LexicalEntry]{{$\\lambda_{t}$}phi phi}} } ]\n\t\t\t\t\t[{\\semder[{\\mover{-wh}{0}}]{\\cancel{v}}{\\semanticRule[Store]{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(0\\#a, x) \\& pe\\_likes(x))}} }\n\t\t\t\t\t\t[{\\semlex{who}{\\cancel{d} -wh}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P P}} },name=node2 ]\n\t\t\t\t\t\t[{\\semder{\\cancel{=d} v}{\\semanticRule[FA]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(a\\_m, y) \\& PatientOf(x, y) \\& pe\\_likes(y))}} }\n\t\t\t\t\t\t\t[{\\semlex{likes}{\\cancel{d=} =d v}{\\semanticRule[LexicalEntry]{{$\\lambda_{a}$}x {$\\lambda_{a}$}y some\\_e(z, all\\_e, AgentOf(x, z) \\& PatientOf(y, z) \\& pe\\_likes(z))}} } ]\n\t\t\t\t\t\t\t[{\\semlex{mary}{\\cancel{d}}{\\semanticRule[LexicalEntry]{a\\_m}} } ] ] ] ] ] ] ]\n\t[{\\semlex{$\\epsilon$}{\\cancel{=v} c}{\\semanticRule[LexicalEntry]{{$\\lambda_{t}$}phi phi}} } ] ]\n\\draw[densely dotted,->] (node2) to[out=west,in=south west] (node1);\n\\end{forest}"
            );

            let typst = rules.to_json(semantic.lexicon());
            println!("{}", typst);
            let typst = rules.to_semantic_json(&semantic, &history);
            println!("{}", typst);
            assert_eq!(
                typst,
                "[{\"Node\":{\"features\":[\"c\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(a_j, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Actor\":\"j\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"v\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(a_j, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Actor\":\"j\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"d\"],\"movement\":[],\"lemma\":\"john\",\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"a_j\",\"tokens\":[{\"Actor\":\"j\"}],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"=d\",\"v\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"lambda a x some_e(y, all_e, AgentOf(a_m, y) & PatientOf(x, y) & pe_likes(y))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Lambda\":\"x\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"y\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Lambda\":\"x\"}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"c=\",\"=d\",\"v\"],\"movement\":[],\"lemma\":\"knows\",\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda <a,t> P lambda a x P(x)\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Lambda\":\"P\"}}},{\"Lambda\":{\"t\":\"a\",\"var\":{\"Lambda\":\"x\"}}},{\"Var\":{\"Lambda\":\"P\"}},\"OpenDelim\",{\"Var\":{\"Lambda\":\"x\"}},\"CloseDelim\"],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"c\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"ApplyFromStorage\",\"state\":{\"expr\":\"lambda a x some_e(y, all_e, AgentOf(a_m, y) & PatientOf(x, y) & pe_likes(y))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Lambda\":\"x\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"y\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Lambda\":\"x\"}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Trace\":{\"trace\":0,\"new_trace\":null,\"semantics\":{\"rule\":\"Trace\",\"state\":null}}},[{\"Node\":{\"features\":[\"+wh\",\"c\"],\"movement\":[{\"trace_id\":0,\"canceled\":true,\"features\":[\"-wh\"]}],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(0#a, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Free\":{\"label\":\"0\",\"t\":\"a\",\"anon\":true}}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{\"0\":{\"expr\":\"lambda <a,t> P P\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Lambda\":\"P\"}}},{\"Var\":{\"Lambda\":\"P\"}}],\"type\":\"a\"}}}}}},{\"Leaf\":{\"features\":[\"v=\",\"+wh\",\"c\"],\"movement\":[],\"lemma\":null,\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda t phi phi\",\"tokens\":[{\"Lambda\":{\"t\":\"t\",\"var\":{\"Lambda\":\"phi\"}}},{\"Var\":{\"Lambda\":\"phi\"}}],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"v\"],\"movement\":[{\"trace_id\":0,\"canceled\":false,\"features\":[\"-wh\"]}],\"trace\":null,\"semantics\":{\"rule\":\"Store\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(0#a, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Free\":{\"label\":\"0\",\"t\":\"a\",\"anon\":true}}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{\"0\":{\"expr\":\"lambda <a,t> P P\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Lambda\":\"P\"}}},{\"Var\":{\"Lambda\":\"P\"}}],\"type\":\"a\"}}}}}},{\"Leaf\":{\"features\":[\"d\",\"-wh\"],\"movement\":[],\"lemma\":\"who\",\"trace\":0,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda <a,t> P P\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Lambda\":\"P\"}}},{\"Var\":{\"Lambda\":\"P\"}}],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"=d\",\"v\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"lambda a x some_e(y, all_e, AgentOf(a_m, y) & PatientOf(x, y) & pe_likes(y))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Lambda\":\"x\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"y\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Lambda\":\"x\"}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"y\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"d=\",\"=d\",\"v\"],\"movement\":[],\"lemma\":\"likes\",\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda a x lambda a y some_e(z, all_e, AgentOf(x, z) & PatientOf(y, z) & pe_likes(z))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Lambda\":\"x\"}}},{\"Lambda\":{\"t\":\"a\",\"var\":{\"Lambda\":\"y\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Quantifier\":\"z\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Var\":{\"Lambda\":\"x\"}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"z\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Lambda\":\"y\"}},\"ArgSep\",{\"Var\":{\"Quantifier\":\"z\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Quantifier\":\"z\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"d\"],\"movement\":[],\"lemma\":\"mary\",\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"a_m\",\"tokens\":[{\"Actor\":\"m\"}],\"movers\":{}}}}}]]]]]],{\"Leaf\":{\"features\":[\"=v\",\"c\"],\"movement\":[],\"lemma\":null,\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda t phi phi\",\"tokens\":[{\"Lambda\":{\"t\":\"t\",\"var\":{\"Lambda\":\"phi\"}}},{\"Var\":{\"Lambda\":\"phi\"}}],\"movers\":{}}}}}]"
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
            "likes::d= V::lambda a x (lambda a y (some_e(e, all_e, AgentOf(y, e)&pe_likes(e)&PatientOf(x, e))))",
            "::v= +k +q t::lambda t x (x)",
            "::V= +k d= +q v::lambda <a,t> p (p)",
        ];

        let lexicon = lexical.join("\n");
        let lex = SemanticLexicon::parse(&lexicon).unwrap();

        let mut v = vec![];
        for (_, s, rules) in lex
            .lexicon
            .generate("t", &config)
            .map_err(|e| e.inner_into::<String>())?
            .take(10)
        {
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
                "someone someone likes\nsome(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nsome(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))",
                "someone everyone likes\nsome(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nevery(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))",
                "everyone everyone likes\nevery(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nevery(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))",
                "everyone someone likes\nevery(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nsome(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))"
            ],
            v
        );
        println!("sov good");

        let lexical = [
            "everyone::d -k -q::lambda <a,t> P (every(x, all_a, P(x)))",
            "someone::d -k -q::lambda <a,t> P (some(x, all_a, P(x)))",
            "likes::d= V -v::lambda a x (lambda a y (some_e(e, all_e, AgentOf(y, e)&pe_likes(e)&PatientOf(x, e))))",
            "::v= +v +k +q t::lambda t x (x)",
            "::V= +k d= +q v::lambda <a,t> p (p)",
        ];

        let lexicon = lexical.join("\n");
        let lex = SemanticLexicon::parse(&lexicon)?;

        let mut v = vec![];
        for (_, s, rules) in lex
            .lexicon
            .generate("t", &config)
            .map_err(|e| e.inner_into::<String>())?
            .take(10)
        {
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
                "everyone likes someone\nevery(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nsome(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))",
                "everyone likes everyone\nevery(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nevery(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))",
                "someone likes everyone\nsome(x, all_a, every(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nevery(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))",
                "someone likes someone\nsome(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(x, z) & pe_likes(z) & PatientOf(y, z))))\nsome(x, all_a, some(y, all_a, some_e(z, all_e, AgentOf(y, z) & pe_likes(z) & PatientOf(x, z))))"
            ],
            v
        );

        #[cfg(feature = "pretty")]
        {
            let (_, _, rules) = lex
                .lexicon
                .parse(&["everyone", "likes", "someone"], "t", &config)
                .map_err(|e| e.inner_into::<String>())?
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
                "\\begin{forest}\n[{\\semder{t}{\\semanticRule[ApplyFromStorage]{every(x, all\\_a, some(y, all\\_a, some\\_e(z, all\\_e, AgentOf(x, z) \\& pe\\_likes(z) \\& PatientOf(y, z))))}} }\n\t[{$t0$},name=node0 ]\n\t[{\\semder[{\\mover{\\cancel{-q}}{0}}]{\\cancel{+q} t}{\\semanticRule[UpdateTrace]{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(0\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} }\n\t\t[{$t1$},name=node1 ]\n\t\t[{\\semder[{\\mover[{-q}]{\\cancel{-k}}{1}}]{\\cancel{+k} +q t}{\\semanticRule[Id]{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(1\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} }\n\t\t\t[{$t3$},name=node2 ]\n\t\t\t[{\\semder[{\\mover{\\cancel{-v}}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{+v} +k +q t}{\\semanticRule[FA]{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(1\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} }\n\t\t\t\t[{\\semlex{$\\epsilon$}{\\cancel{v=} +v +k +q t}{\\semanticRule[LexicalEntry]{{$\\lambda_{t}$}phi phi}} } ]\n\t\t\t\t[{\\semder[{\\mover{-v}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{v}}{\\semanticRule[ApplyFromStorage]{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(1\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} }\n\t\t\t\t\t[{$t2$},name=node3 ]\n\t\t\t\t\t[{\\semder[{\\mover{\\cancel{-q}}{2}, \\mover{-v}{3}, \\mover[{-q}]{-k}{1}}]{\\cancel{+q} v}{\\semanticRule[Store]{some\\_e(x, all\\_e, AgentOf(1\\#a, x) \\& pe\\_likes(x) \\& PatientOf(2\\#a, x))}} }\n\t\t\t\t\t\t[{\\semder[{\\mover{-q}{2}, \\mover{-v}{3}}]{\\cancel{d=} +q v}{\\semanticRule[UpdateTrace]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& pe\\_likes(y) \\& PatientOf(2\\#a, y))}} }\n\t\t\t\t\t\t\t[{$t4$},name=node5 ]\n\t\t\t\t\t\t\t[{\\semder[{\\mover[{-q}]{\\cancel{-k}}{4}, \\mover{-v}{3}}]{\\cancel{+k} d= +q v}{\\semanticRule[FA]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& pe\\_likes(y) \\& PatientOf(4\\#a, y))}} }\n\t\t\t\t\t\t\t\t[{\\semlex{$\\epsilon$}{\\cancel{V=} +k d= +q v}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P P}} } ]\n\t\t\t\t\t\t\t\t[{\\semder[{\\mover[{-q}]{-k}{4}}]{\\cancel{V} -v}{\\semanticRule[Store]{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& pe\\_likes(y) \\& PatientOf(4\\#a, y))}} },name=node8\n\t\t\t\t\t\t\t\t\t[{\\semlex{likes}{\\cancel{d=} V -v}{\\semanticRule[LexicalEntry]{{$\\lambda_{a}$}x {$\\lambda_{a}$}y some\\_e(z, all\\_e, AgentOf(y, z) \\& pe\\_likes(z) \\& PatientOf(x, z))}} } ]\n\t\t\t\t\t\t\t\t\t[{\\semlex{someone}{\\cancel{d} -k -q}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P some(x, all\\_a, P(x))}} },name=node6 ] ] ] ]\n\t\t\t\t\t\t[{\\semlex{everyone}{\\cancel{d} -k -q}{\\semanticRule[LexicalEntry]{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P every(x, all\\_a, P(x))}} },name=node4 ] ] ] ] ] ] ]\n\\draw[densely dotted,->] (node1) to[out=west,in=south west] (node0);\n\\draw[densely dotted,->] (node4) to[out=west,in=south west] (node1);\n\\draw[densely dotted,->] (node5) to[out=west,in=south west] (node3);\n\\draw[densely dotted,->] (node6) to[out=west,in=south west] (node5);\n\\draw[densely dotted,->] (node8) to[out=west,in=south west] (node2);\n\\end{forest}"
            );
        }
        Ok(())
    }

    #[test]
    fn obscure_error_with_rich() -> anyhow::Result<()> {
        let grammar = "ε::0= =2 +1 0::lambda a x (pa_0(x))
ran::2::lambda t x (a_1)
John::0 -1::a_1";

        let lexicon = SemanticLexicon::parse(grammar)?;
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
