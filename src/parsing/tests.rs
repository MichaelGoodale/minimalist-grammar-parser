use super::trees::{FutureTree, GornIndex, ParseMoment};
use crate::Direction;
use anyhow::Result;
use petgraph::graph::NodeIndex;

#[test]
fn index_order() -> Result<()> {
    let a = GornIndex {
        index: vec![Direction::Left, Direction::Left, Direction::Left],
    };
    let b = GornIndex {
        index: vec![Direction::Left, Direction::Left, Direction::Right],
    };
    assert!(a < b);

    let b = GornIndex {
        index: vec![Direction::Left, Direction::Left],
    };

    assert!(b < a);
    let b = GornIndex {
        index: vec![Direction::Right],
    };
    assert!(a < b);

    let a = ParseMoment {
        tree: FutureTree {
            node: NodeIndex::new(0),
            id: 0,
            index: GornIndex {
                index: vec![Direction::Left, Direction::Right],
            },
        },
        movers: vec![FutureTree {
            node: NodeIndex::new(0),
            id: 0,
            index: GornIndex {
                index: vec![Direction::Right],
            },
        }],
    };

    assert_eq!(
        a.least_index(),
        &GornIndex {
            index: vec![Direction::Left, Direction::Right]
        }
    );

    Ok(())
}
