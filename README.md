# Minimalist Grammar Parser

This repo defines a number of structs and methods to parse and generate Minimalist Grammars
(MGs) from Stabler (1997). Specifically, it implements a variety of the MG algorithm adapted
from Stabler (2011) and Stabler (2013)

## Examples of usage

The following code generates 4 sentences from the $a^nb^n$ language.

```rust
use minimalist_grammar_parser::Lexicon;
use minimalist_grammar_parser::ParsingConfig;
let lexicon = Lexicon::from_string("a::s= b= s\nb::b\n::s")?;
let v = lexicon
    .generate("s", &ParsingConfig::default())?
    .take(4)
    .map(|(_prob, s, _rules)| {
        s.into_iter()
            .map(|word| word.try_inner().unwrap())
            .collect::<Vec<_>>()
            .join("")
    })
    .collect::<Vec<_>>();
assert_eq!(v, vec!["", "ab", "aabb", "aaabbb"]);
```

## References

- Stabler, E. (1997). Derivational minimalism. In C. Retoré (Ed.), Logical Aspects of Computational Linguistics (pp. 68–95). Springer. <https://doi.org/10.1007/BFb0052152>
- Stabler, E. (2011). Top-Down Recognizers for MCFGs and MGs. In F. Keller & D. Reitter (Eds.), Proceedings of the 2nd Workshop on Cognitive Modeling and Computational Linguistics (pp. 39–48). Association for Computational Linguistics. <https://aclanthology.org/W11-0605>
- Stabler, E. (2013). Two Models of Minimalist, Incremental Syntactic Analysis. Topics in Cognitive Science, 5(3), 611–633. <https://doi.org/10.1111/tops.12031>
