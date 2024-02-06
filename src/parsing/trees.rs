use std::borrow::Borrow;

use petgraph::graph::NodeIndex;

use crate::Direction;
use bitvec::prelude::*;
use thin_vec::ThinVec;

type IndexArray = BitArray<u64, Lsb0>;

pub const MAX_STEPS: usize = 63;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
pub struct GornIndex {
    index: BitArray<u64, Lsb0>,
    size: usize,
}

impl<T> From<T> for GornIndex
where
    T: Borrow<[Direction]>,
{
    fn from(value: T) -> Self {
        let value = value.borrow();
        let size = value.len();
        let mut index = IndexArray::default();
        value
            .iter()
            .enumerate()
            .for_each(|(i, &x)| index.set(i, x.into()));
        GornIndex { size, index }
    }
}
impl From<GornIndex> for Vec<Direction> {
    fn from(value: GornIndex) -> Self {
        value
            .index
            .into_iter()
            .take(value.size)
            .map(|x| x.into())
            .collect()
    }
}

impl GornIndex {
    #[inline]
    pub fn clone_push(&self, d: Direction) -> Self {
        let mut v = *self;
        v.size += 1;
        v.index.set(v.size, d.into());
        v
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FutureTree {
    pub node: NodeIndex,
    pub index: GornIndex,
    pub id: usize,
}

impl PartialOrd for FutureTree {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FutureTree {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.index.cmp(&other.index) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.node.cmp(&other.node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseMoment {
    pub tree: FutureTree,
    pub movers: ThinVec<FutureTree>,
}

impl ParseMoment {
    pub fn least_index(&self) -> &GornIndex {
        let mut least = &self.tree.index;
        for m in self.movers.iter() {
            if m.index < *least {
                least = &m.index
            }
        }
        least
    }

    pub fn new(tree: FutureTree, movers: ThinVec<FutureTree>) -> Self {
        ParseMoment { tree, movers }
    }
    pub fn no_movers(&self) -> bool {
        self.movers.is_empty()
    }
}

impl PartialOrd for ParseMoment {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ParseMoment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.least_index().cmp(other.least_index())
    }
}
