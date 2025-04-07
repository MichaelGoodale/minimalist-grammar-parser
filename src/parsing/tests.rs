use super::trees::{FutureTree, GornIndex, ParseMoment};
use crate::{parsing::RuleIndex, Direction};
use anyhow::Result;
use petgraph::graph::NodeIndex;
use thin_vec::thin_vec;

#[test]
fn index_order() -> Result<()> {
    let a: GornIndex = [Direction::Left, Direction::Left, Direction::Left].into();
    let b: GornIndex = [Direction::Left, Direction::Left, Direction::Right].into();
    assert!(a < b);

    let b: GornIndex = [Direction::Left, Direction::Left].into();

    assert!(b < a);
    let b: GornIndex = [Direction::Right].into();
    assert!(a < b);

    let a = ParseMoment::new(
        FutureTree {
            node: NodeIndex::new(0),
            id: RuleIndex(0),
            index: [Direction::Left, Direction::Right].into(),
        },
        thin_vec![FutureTree {
            node: NodeIndex::new(0),
            id: RuleIndex(0),
            index: [Direction::Right].into(),
        }],
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
