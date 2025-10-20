use super::trees::{FutureTree, GornIndex, ParseMoment};
use crate::{Direction, ParsingConfig, lexicon::Lexicon, parsing::RuleIndex};
use anyhow::Result;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use thin_vec::thin_vec;

#[test]
fn infix_order() -> Result<()> {
    let a: GornIndex = [Direction::Left].into();
    let b: GornIndex = [Direction::Left, Direction::Left, Direction::Right].into();
    let c: GornIndex = [Direction::Left, Direction::Right].into();
    let d: GornIndex = [Direction::Left, Direction::Left, Direction::Right].into();
    let e: GornIndex = [].into();
    let f: GornIndex = [Direction::Right].into();

    let mut list = [a, b, c, d, e, f];
    list.sort_by(GornIndex::infix_order);

    assert_eq!(list, [b, d, a, c, e, f]);
    Ok(())
}

#[test]
fn index_order() -> Result<()> {
    let a: GornIndex = [Direction::Left, Direction::Left, Direction::Left].into();
    let b: GornIndex = [Direction::Left, Direction::Left, Direction::Right].into();
    assert!(a < b);
    assert!(!a.is_ancestor_of(b));
    assert!(!b.is_ancestor_of(a));

    let b: GornIndex = [Direction::Left, Direction::Left].into();
    assert!(b.is_ancestor_of(a));
    assert!(!a.is_ancestor_of(b));

    assert!(b < a);
    let b: GornIndex = [Direction::Right].into();
    assert!(a < b);
    assert!(!a.is_ancestor_of(b));
    assert!(!b.is_ancestor_of(a));

    let a = ParseMoment::new(
        FutureTree {
            node: NodeIndex::new(0),
            id: RuleIndex::one(),
            index: [Direction::Left, Direction::Right].into(),
        },
        thin_vec![FutureTree {
            node: NodeIndex::new(0),
            id: RuleIndex::one(),
            index: [Direction::Right].into(),
        },],
        None,
    );

    assert_eq!(a.least_index(), &[Direction::Left, Direction::Right].into());

    let mut raw_directions: Vec<Vec<_>> = [
        vec![Direction::Left],
        vec![Direction::Left, Direction::Right],
        vec![Direction::Right, Direction::Left],
        vec![Direction::Left, Direction::Right, Direction::Left],
        vec![Direction::Left, Direction::Right, Direction::Right],
        vec![Direction::Left, Direction::Right, Direction::Right],
        vec![
            Direction::Left,
            Direction::Left,
            Direction::Left,
            Direction::Left,
        ],
        vec![
            Direction::Left,
            Direction::Left,
            Direction::Right,
            Direction::Left,
            Direction::Left,
        ],
        vec![
            Direction::Left,
            Direction::Left,
            Direction::Left,
            Direction::Left,
            Direction::Left,
        ],
        vec![Direction::Left, Direction::Right, Direction::Right],
        vec![Direction::Right, Direction::Left],
    ]
    .to_vec();
    raw_directions.sort();
    let mut gorn_indices: Vec<GornIndex> = raw_directions
        .clone()
        .into_iter()
        .map(|x| x.into())
        .collect();
    gorn_indices.sort();
    let gorn_indices: Vec<Vec<Direction>> = gorn_indices.into_iter().map(|x| x.into()).collect();
    assert_eq!(gorn_indices, raw_directions);
    Ok(())
}

#[test]
fn smc() -> anyhow::Result<()> {
    let lex = Lexicon::from_string("a::d= +w +w c\nb::d -w -w")?;
    lex.generate(
        "c",
        &ParsingConfig::new(
            LogProb::new(-128.0)?,
            LogProb::from_raw_prob(0.5)?,
            1000,
            100,
        ),
    )?
    .next()
    .unwrap();

    let lex = Lexicon::from_string("a::d= d= +w +w c\nb::d -w")?;
    assert!(
        lex.generate(
            "c",
            &ParsingConfig::new(
                LogProb::new(-128.0)?,
                LogProb::from_raw_prob(0.5)?,
                1000,
                100,
            ),
        )?
        .next()
        .is_none()
    );

    Ok(())
}

#[test]
fn specificer_island_constraints() -> anyhow::Result<()> {
    let lex = Lexicon::from_string("a::d= +w c\nb::d -w")?;
    lex.generate(
        "c",
        &ParsingConfig::new(
            LogProb::new(-128.0)?,
            LogProb::from_raw_prob(0.5)?,
            1000,
            100,
        ),
    )?
    .next()
    .unwrap();

    //We can extract from the specifier if the -w feature is on top.
    let lex = Lexicon::from_string("a::b= =c +w c\nb::b\nc::c -w")?;
    lex.generate(
        "c",
        &ParsingConfig::new(
            LogProb::new(-128.0)?,
            LogProb::from_raw_prob(0.5)?,
            1000,
            100,
        ),
    )?
    .next()
    .unwrap();

    //We can't extract from the specifier if the -w feature is a sub constiuent.
    let lex = Lexicon::from_string("a::b= =c +w c\nb::b\n::z= c\nc::z -w")?;
    assert!(
        lex.generate(
            "c",
            &ParsingConfig::new(
                LogProb::new(-128.0)?,
                LogProb::from_raw_prob(0.5)?,
                1000,
                100,
            ),
        )?
        .next()
        .is_none()
    );

    Ok(())
}
