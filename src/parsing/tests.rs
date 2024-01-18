use super::trees::{FutureTree, GornIndex, ParseMoment};
use crate::Direction;
use anyhow::Result;
use petgraph::graph::NodeIndex;
use tinyvec::tiny_vec;

#[test]
fn index_order() -> Result<()> {
    let a = GornIndex {
        index: tiny_vec!(Direction::Left, Direction::Left, Direction::Left),
    };
    let b = GornIndex {
        index: tiny_vec!(Direction::Left, Direction::Left, Direction::Right),
    };
    assert!(a < b);

    let b = GornIndex {
        index: tiny_vec!(Direction::Left, Direction::Left),
    };

    assert!(b < a);
    let b = GornIndex {
        index: tiny_vec!([Direction; 5] => Direction::Right),
    };
    assert!(a < b);

    let a = ParseMoment {
        tree: FutureTree {
            node: NodeIndex::new(0),
            id: 0,
            index: GornIndex {
                index: tiny_vec!(Direction::Left, Direction::Right),
            },
        },
        movers: vec![FutureTree {
            node: NodeIndex::new(0),
            id: 0,
            index: GornIndex {
                index: tiny_vec!([Direction; 5] => Direction::Right),
            },
        }],
    };

    assert_eq!(
        a.least_index(),
        &GornIndex {
            index: tiny_vec!(Direction::Left, Direction::Right)
        }
    );

    Ok(())
}
