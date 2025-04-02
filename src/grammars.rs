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

pub const SIMPLESTABLER2011: &str = "::V= C
::V= +W C
likes::D= =D V
queen::N
king::N
the::N= D
which::N= D -W";

pub const COPY_LANGUAGE: &str = "::=T +r +l T
::T -r -l
a::=A +l T -l
a::=T +r A -r
b::=B +l T -l
b::=T +r B -r";

pub const ALT_COPY_LANGUAGE: &str = "E::=T +r +l T
S::T -r -l
a::=A +l T -l
a::=T +r A -r
b::=B +l T -l
b::=T +r B -r";
