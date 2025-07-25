use ahash::HashMap;
use simple_semantics::language::LambdaParseError;
use std::fmt::Debug;

use crate::{ParsingConfig, PhonContent};

use super::*;

use itertools::Itertools;
use simple_semantics::LanguageExpression;
use simple_semantics::lambda::RootedLambdaPool;
use simple_semantics::language::Expr;

///A lexicon that is paired with semantic interpretations for its leaf nodes.
///
///Each leaf must have a semantic interpretation defined as a [`RootedLambdaPool<Expr>`].
#[derive(Debug, Clone)]
pub struct SemanticLexicon<'src, T: Eq, Category: Eq> {
    lexicon: Lexicon<T, Category>,
    semantic_entries: HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>>,
}

impl<'src, T: Eq, C: Eq> SemanticLexicon<'src, T, C> {
    ///Create a new [`SemanticLexicon`] by combining a [`Lexicon`] and a [`HashMap`] of leaf nodes
    ///and semantic interpretations ([`RootedLambdaPool`])
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
    ///Create a new semantic lexicon by parsing a string.
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
    ///Get the interpretation of a leaf node. Panics if the node has no semantic interpretation.
    pub fn interpretation(&self, nx: NodeIndex) -> &RootedLambdaPool<'src, Expr<'src>> {
        self.semantic_entries
            .get(&nx)
            .expect("There is no lemma of that node index!")
    }

    ///Get a reference to the underlying [`Lexicon`]
    pub fn lexicon(&self) -> &Lexicon<T, C> {
        &self.lexicon
    }

    ///Get a mutable reference to the underlying [`Lexicon`]
    pub fn lexicon_mut(&mut self) -> &mut Lexicon<T, C> {
        &mut self.lexicon
    }

    ///Get a reference to the underlying [`HashMap`] of lexical entries.
    pub fn interpretations(&self) -> &HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>> {
        &self.semantic_entries
    }

    ///Get a mutable reference to the underlying [`HashMap`] of lexical entries.
    pub fn interpretations_mut(
        &mut self,
    ) -> &mut HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>> {
        &mut self.semantic_entries
    }

    ///Get a mutable reference to both the underlying [`Lexicon`] and [`HashMap`] of lexical
    ///entries.
    pub fn lexicon_and_interpretations_mut(
        &mut self,
    ) -> (
        &mut Lexicon<T, C>,
        &mut HashMap<NodeIndex, RootedLambdaPool<'src, Expr<'src>>>,
    ) {
        (&mut self.lexicon, &mut self.semantic_entries)
    }

    ///Remaps the lexicon to a new category or lemma type
    pub fn remap_lexicon<T2: Eq, C2: Eq>(
        self,
        lemma_map: impl Fn(&T) -> T2,
        category_map: impl Fn(&C) -> C2,
    ) -> SemanticLexicon<'src, T2, C2> {
        let SemanticLexicon {
            lexicon,
            semantic_entries,
        } = self;

        let lexicon = lexicon.remap_lexicon(lemma_map, category_map);

        SemanticLexicon {
            lexicon,
            semantic_entries,
        }
    }

    ///Parse a sentence and return all its parses and their interpretations as nested iterators.
    #[allow(clippy::type_complexity)]
    pub fn parse_and_interpret<'a, 'b: 'a>(
        &'a self,
        category: C,
        sentence: &'b [PhonContent<T>],
        config: &'b ParsingConfig,
    ) -> Result<
        impl Iterator<
            Item = (
                LogProb<f64>,
                &'a [PhonContent<T>],
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
    use crate::{ParsingConfig, PhonContent};

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
            .parse(&PhonContent::from(["john", "likes", "mary"]), "v", &config)?
            .next()
            .unwrap();
        let (interpretation, mut history) = rules.to_interpretation(&semantic).next().unwrap();
        let interpretation = interpretation.into_pool()?;

        #[cfg(feature = "pretty")]
        {
            let latex = rules.to_semantic_tree(&semantic, &history).to_latex();
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}[\\semder{v}{\\textsc{FA}} [\\lex{d}{john}{\\textsc{LexicalEntry}}] [\\semder{=d v}{\\textsc{FA}} [\\lex{d= =d v}{likes}{\\textsc{LexicalEntry}}] [\\lex{d}{mary}{\\textsc{LexicalEntry}}]]]\\end{forest}"
            );

            history = history.into_rich(&semantic, &rules);
            let latex = rules.to_semantic_tree(&semantic, &history).to_latex();
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}[\\semder{v}{\\texttt{some\\_e(x, all\\_e, AgentOf(a\\_j, x) \\& PatientOf(a\\_m, x) \\& pe\\_likes(x))}} [\\lex{d}{john}{\\texttt{a\\_j}}] [\\semder{=d v}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& PatientOf(a\\_m, y) \\& pe\\_likes(y))}} [\\lex{d= =d v}{likes}{\\texttt{{$\\lambda_{a}$}x {$\\lambda_{a}$}y some\\_e(z, all\\_e, AgentOf(y, z) \\& PatientOf(x, z) \\& pe\\_likes(z))}}] [\\lex{d}{mary}{\\texttt{a\\_m}}]]]\\end{forest}"
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
            "{lexicon}\n::=v c::lambda t phi (phi)\n::v= +wh c::lambda t phi (phi)\nknows::c= =d v::lambda <a,t> P (lambda a x (P(x)))\nwho::d -wh::lambda <a,t> P (P)",
        );

        let s = lexicon.as_str();
        let semantic = SemanticLexicon::parse(s)?;

        let (_, _, rules) = semantic
            .lexicon()
            .parse(
                &PhonContent::from(["john", "knows", "who", "likes", "mary"]),
                "c",
                &config,
            )
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
            let latex = rules.to_semantic_tree(&semantic, &history).to_latex();

            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}[\\semder{c}{\\textsc{FA}} [\\semder{v}{\\textsc{FA}} [\\lex{d}{john}{\\textsc{LexicalEntry}}] [\\semder{=d v}{\\textsc{FA}} [\\lex{c= =d v}{knows}{\\textsc{LexicalEntry}}] [\\semder{c}{\\textsc{ApplyFromStorage}} [$t$, name=tracet0] [\\semder{+wh c}{\\textsc{FA}} [\\lex{v= +wh c}{$\\epsilon$}{\\textsc{LexicalEntry}}] [\\semder{v}{\\textsc{Store}} [\\lex{d -wh}{who}{\\textsc{LexicalEntry}}] [\\semder{=d v}{\\textsc{FA}} [\\lex{d= =d v}{likes}{\\textsc{LexicalEntry}}] [\\lex{d}{mary}{\\textsc{LexicalEntry}}]]]]]]] [\\lex{=v c}{$\\epsilon$}{\\textsc{LexicalEntry}}]]\\end{forest}"
            );

            history = history.into_rich(&semantic, &rules);
            let latex = rules.to_semantic_tree(&semantic, &history).to_latex();
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}[\\semder{c}{\\texttt{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(a\\_j, x) \\& pe\\_likes(x))}} [\\semder{v}{\\texttt{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(a\\_j, x) \\& pe\\_likes(x))}} [\\lex{d}{john}{\\texttt{a\\_j}}] [\\semder{=d v}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(a\\_m, y) \\& PatientOf(x, y) \\& pe\\_likes(y))}} [\\lex{c= =d v}{knows}{\\texttt{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P {$\\lambda_{a}$}x P(x)}}] [\\semder{c}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(a\\_m, y) \\& PatientOf(x, y) \\& pe\\_likes(y))}} [$t$, name=tracet0] [\\semder{+wh c}{\\texttt{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(0\\#a, x) \\& pe\\_likes(x))}} [\\lex{v= +wh c}{$\\epsilon$}{\\texttt{{$\\lambda_{t}$}phi phi}}] [\\semder{v}{\\texttt{some\\_e(x, all\\_e, AgentOf(a\\_m, x) \\& PatientOf(0\\#a, x) \\& pe\\_likes(x))}} [\\lex{d -wh}{who}{\\texttt{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P P}}] [\\semder{=d v}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(a\\_m, y) \\& PatientOf(x, y) \\& pe\\_likes(y))}} [\\lex{d= =d v}{likes}{\\texttt{{$\\lambda_{a}$}x {$\\lambda_{a}$}y some\\_e(z, all\\_e, AgentOf(x, z) \\& PatientOf(y, z) \\& pe\\_likes(z))}}] [\\lex{d}{mary}{\\texttt{a\\_m}}]]]]]]] [\\lex{=v c}{$\\epsilon$}{\\texttt{{$\\lambda_{t}$}phi phi}}]]\\end{forest}"
            );

            let typst = serde_json::to_string(&rules.to_tree(semantic.lexicon()))?;
            println!("{typst}");
            let typst = serde_json::to_string(&rules.to_semantic_tree(&semantic, &history))?;
            println!("{typst}");
            assert_eq!(
                typst,
                "[{\"Node\":{\"features\":[\"c\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(a_j, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Actor\":\"j\"},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"v\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(a_j, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Actor\":\"j\"},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"d\"],\"movement\":[],\"lemma\":{\"Single\":\"john\"},\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"a_j\",\"tokens\":[{\"Actor\":\"j\"}],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"=d\",\"v\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"lambda a x some_e(y, all_e, AgentOf(a_m, y) & PatientOf(x, y) & pe_likes(y))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Bound\":\"x\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"y\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"ArgSep\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"c=\",\"=d\",\"v\"],\"movement\":[],\"lemma\":{\"Single\":\"knows\"},\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda <a,t> P lambda a x P(x)\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Bound\":\"P\"}}},{\"Lambda\":{\"t\":\"a\",\"var\":{\"Bound\":\"x\"}}},{\"Var\":{\"Bound\":\"P\"}},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\"],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"c\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"ApplyFromStorage\",\"state\":{\"expr\":\"lambda a x some_e(y, all_e, AgentOf(a_m, y) & PatientOf(x, y) & pe_likes(y))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Bound\":\"x\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"y\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"ArgSep\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Trace\":{\"trace\":0,\"new_trace\":null,\"semantics\":{\"rule\":\"Trace\",\"state\":null}}},[{\"Node\":{\"features\":[\"+wh\",\"c\"],\"movement\":[{\"trace_id\":0,\"canceled\":true,\"features\":[\"-wh\"]}],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(0#a, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Free\":{\"label\":\"0\",\"t\":\"a\",\"anon\":true}}},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{\"0\":{\"expr\":\"lambda <a,t> P P\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Bound\":\"P\"}}},{\"Var\":{\"Bound\":\"P\"}}],\"type\":\"a\"}}}}}},{\"Leaf\":{\"features\":[\"v=\",\"+wh\",\"c\"],\"movement\":[],\"lemma\":{\"Single\":null},\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda t phi phi\",\"tokens\":[{\"Lambda\":{\"t\":\"t\",\"var\":{\"Bound\":\"phi\"}}},{\"Var\":{\"Bound\":\"phi\"}}],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"v\"],\"movement\":[{\"trace_id\":0,\"canceled\":false,\"features\":[\"-wh\"]}],\"trace\":null,\"semantics\":{\"rule\":\"Store\",\"state\":{\"expr\":\"some_e(x, all_e, AgentOf(a_m, x) & PatientOf(0#a, x) & pe_likes(x))\",\"tokens\":[{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"x\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Free\":{\"label\":\"0\",\"t\":\"a\",\"anon\":true}}},\"ArgSep\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{\"0\":{\"expr\":\"lambda <a,t> P P\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Bound\":\"P\"}}},{\"Var\":{\"Bound\":\"P\"}}],\"type\":\"a\"}}}}}},{\"Leaf\":{\"features\":[\"d\",\"-wh\"],\"movement\":[],\"lemma\":{\"Single\":\"who\"},\"trace\":0,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda <a,t> P P\",\"tokens\":[{\"Lambda\":{\"t\":\"<a,t>\",\"var\":{\"Bound\":\"P\"}}},{\"Var\":{\"Bound\":\"P\"}}],\"movers\":{}}}}},[{\"Node\":{\"features\":[\"=d\",\"v\"],\"movement\":[],\"trace\":null,\"semantics\":{\"rule\":\"FunctionalApplication\",\"state\":{\"expr\":\"lambda a x some_e(y, all_e, AgentOf(a_m, y) & PatientOf(x, y) & pe_likes(y))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Bound\":\"x\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"y\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Actor\":\"m\"},\"ArgSep\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"ArgSep\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"y\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"d=\",\"=d\",\"v\"],\"movement\":[],\"lemma\":{\"Single\":\"likes\"},\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda a x lambda a y some_e(z, all_e, AgentOf(x, z) & PatientOf(y, z) & pe_likes(z))\",\"tokens\":[{\"Lambda\":{\"t\":\"a\",\"var\":{\"Bound\":\"x\"}}},{\"Lambda\":{\"t\":\"a\",\"var\":{\"Bound\":\"y\"}}},{\"Quantifier\":{\"q\":\"some\",\"var\":{\"Bound\":\"z\"},\"t\":\"e\"}},\"OpenDelim\",{\"Const\":\"all_e\"},\"ArgSep\",{\"Func\":\"AgentOf\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"x\"}},\"ArgSep\",{\"Var\":{\"Bound\":\"z\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"PatientOf\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"y\"}},\"ArgSep\",{\"Var\":{\"Bound\":\"z\"}},\"CloseDelim\",{\"Func\":\"&\"},{\"Func\":\"likes\"},\"OpenDelim\",{\"Var\":{\"Bound\":\"z\"}},\"CloseDelim\",\"CloseDelim\"],\"movers\":{}}}}},{\"Leaf\":{\"features\":[\"d\"],\"movement\":[],\"lemma\":{\"Single\":\"mary\"},\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"a_m\",\"tokens\":[{\"Actor\":\"m\"}],\"movers\":{}}}}}]]]]]],{\"Leaf\":{\"features\":[\"=v\",\"c\"],\"movement\":[],\"lemma\":{\"Single\":null},\"trace\":null,\"semantics\":{\"rule\":\"Scan\",\"state\":{\"expr\":\"lambda t phi phi\",\"tokens\":[{\"Lambda\":{\"t\":\"t\",\"var\":{\"Bound\":\"phi\"}}},{\"Var\":{\"Bound\":\"phi\"}}],\"movers\":{}}}}}]"
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
            let mut s = PhonContent::try_flatten(s)?.join(" ");
            for interpretation in rules
                .to_interpretation(&lex)
                .map(|(pool, _)| pool.into_pool().unwrap().to_string())
                .unique()
            {
                s.push('\n');
                s.push_str(&interpretation);
            }
            println!("{s}");
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
            let mut s = PhonContent::try_flatten(s)?.join(" ");
            for interpretation in rules
                .to_interpretation(&lex)
                .map(|(pool, _)| pool.into_pool().unwrap().to_string())
                .unique()
            {
                s.push('\n');
                s.push_str(&interpretation);
            }
            println!("{s}");
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
                .parse(
                    &PhonContent::from(["everyone", "likes", "someone"]),
                    "t",
                    &config,
                )
                .map_err(|e| e.inner_into::<String>())?
                .next()
                .unwrap();

            let (_, mut history) = rules.to_interpretation(&lex).next().unwrap();
            let latex = rules.to_semantic_tree(&lex, &history).to_latex();

            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}[\\semder{t}{\\textsc{ApplyFromStorage}} [$t$, name=tracet0] [\\semder{+q t}{\\textsc{UpdateTrace}} [$t$, name=tracet1] [\\semder{+k +q t}{\\textsc{Id}} [$t$, name=tracet3] [\\semder{+v +k +q t}{\\textsc{FA}} [\\lex{v= +v +k +q t}{$\\epsilon$}{\\textsc{LexicalEntry}}] [\\semder{v}{\\textsc{ApplyFromStorage}} [$t$, name=tracet2] [\\semder{+q v}{\\textsc{Store}} [\\semder{d= +q v}{\\textsc{UpdateTrace}} [$t$, name=tracet4] [\\semder{+k d= +q v}{\\textsc{FA}} [\\lex{V= +k d= +q v}{$\\epsilon$}{\\textsc{LexicalEntry}}] [\\semder{V -v}{\\textsc{Store}} [\\lex{d= V -v}{likes}{\\textsc{LexicalEntry}}] [\\lex{d -k -q}{someone}{\\textsc{LexicalEntry}}]]]] [\\lex{d -k -q}{everyone}{\\textsc{LexicalEntry}}]]]]]]]\\end{forest}"
            );

            history = history.into_rich(&lex, &rules);
            let latex = rules.to_semantic_tree(&lex, &history).to_latex();
            println!("{latex}");
            assert_eq!(
                latex,
                "\\begin{forest}[\\semder{t}{\\texttt{every(x, all\\_a, some(y, all\\_a, some\\_e(z, all\\_e, AgentOf(x, z) \\& pe\\_likes(z) \\& PatientOf(y, z))))}} [$t$, name=tracet0] [\\semder{+q t}{\\texttt{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(0\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} [$t$, name=tracet1] [\\semder{+k +q t}{\\texttt{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(1\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} [$t$, name=tracet3] [\\semder{+v +k +q t}{\\texttt{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(1\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} [\\lex{v= +v +k +q t}{$\\epsilon$}{\\texttt{{$\\lambda_{t}$}phi phi}}] [\\semder{v}{\\texttt{some(x, all\\_a, some\\_e(y, all\\_e, AgentOf(1\\#a, y) \\& pe\\_likes(y) \\& PatientOf(x, y)))}} [$t$, name=tracet2] [\\semder{+q v}{\\texttt{some\\_e(x, all\\_e, AgentOf(1\\#a, x) \\& pe\\_likes(x) \\& PatientOf(2\\#a, x))}} [\\semder{d= +q v}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& pe\\_likes(y) \\& PatientOf(2\\#a, y))}} [$t$, name=tracet4] [\\semder{+k d= +q v}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& pe\\_likes(y) \\& PatientOf(4\\#a, y))}} [\\lex{V= +k d= +q v}{$\\epsilon$}{\\texttt{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P P}}] [\\semder{V -v}{\\texttt{{$\\lambda_{a}$}x some\\_e(y, all\\_e, AgentOf(x, y) \\& pe\\_likes(y) \\& PatientOf(4\\#a, y))}} [\\lex{d= V -v}{likes}{\\texttt{{$\\lambda_{a}$}x {$\\lambda_{a}$}y some\\_e(z, all\\_e, AgentOf(y, z) \\& pe\\_likes(z) \\& PatientOf(x, z))}}] [\\lex{d -k -q}{someone}{\\texttt{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P some(x, all\\_a, P(x))}}]]]] [\\lex{d -k -q}{everyone}{\\texttt{{$\\lambda_{\\left\\langle a,t\\right\\rangle }$}P every(x, all\\_a, P(x))}}]]]]]]]\\end{forest}"
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
        for (_, _, r) in lexicon.lexicon.parse(
            &PhonContent::from(["John", "ran"]),
            "0",
            &ParsingConfig::default(),
        )? {
            for (pool, h) in r.to_interpretation(&lexicon) {
                pool.into_pool()?;
                h.into_rich(&lexicon, &r);
            }
        }
        Ok(())
    }
}
