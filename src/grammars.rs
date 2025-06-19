//! This module defines a few example grammars that can be useful in testing or otherwise.

///The grammar from Stabler (2013)
///
/// - Stabler, E. (2013). Two Models of Minimalist, Incremental Syntactic Analysis. Topics in Cognitive Science, 5(3), 611–633. <https://doi.org/10.1111/tops.12031>
pub const STABLER2011: &str = "::V= C
::V= +W C
knows::C= =D V
says::C= =D V
prefers::D= =D V
drinks::D= =D V
king::N
wine::N
beer::N
queen::N
the::N= D
which::N= D -W";

///A version of [`STABLER2011`] without embedded clauses.
pub const SIMPLESTABLER2011: &str = "::V= C
::V= +W C
likes::D= =D V
queen::N
king::N
the::N= D
which::N= D -W";

///The copy language (any string of a's and b's repeated twice).
pub const COPY_LANGUAGE: &str = "::=T +r +l T
::T -r -l
a::=A +l T -l
a::=T +r A -r
b::=B +l T -l
b::=T +r B -r";

///The same copy language as above but showing the placement of the unprounced words
pub const ALT_COPY_LANGUAGE: &str = "E::=T +r +l T
S::T -r -l
a::=A +l T -l
a::=T +r A -r
b::=B +l T -l
b::=T +r B -r";

///The [Dyck language](https://en.wikipedia.org/wiki/Dyck_language) or the language of balanced
///parentheses.
pub const DYCK_LANGUAGE: &str = "(::S= R= S\n)::S= R\n::S";
