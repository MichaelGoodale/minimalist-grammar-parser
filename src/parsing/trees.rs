use std::borrow::Borrow;

use petgraph::graph::NodeIndex;

use crate::Direction;
use bitvec::prelude::*;
use thin_vec::ThinVec;

use super::RuleIndex;

pub const N_CHUNKS: u32 = 128 / usize::BITS;
type IndexArray = BitArray<[usize; N_CHUNKS as usize], Lsb0>;

///A compile time limitation on the maximum number of steps in a derivation (set to 128).
pub const MAX_STEPS: usize = (usize::BITS * N_CHUNKS) as usize;

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GornIndex {
    index: IndexArray,
    size: usize,
}

impl std::fmt::Debug for GornIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.index[..self.size]
                    .iter()
                    .map(|x| {
                        let x: bool = *x.as_ref();
                        let x: u8 = x.into();
                        x
                    })
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PartialOrd for GornIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GornIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index[..self.size].cmp(&other.index[..other.size])
    }
}

impl Iterator for GornIterator {
    type Item = Direction;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.gorn.size {
            let d = self.gorn.index[self.pos];
            self.pos += 1;
            Some(d.into())
        } else {
            None
        }
    }
}

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GornIterator {
    pos: usize,
    gorn: GornIndex,
}

impl IntoIterator for GornIndex {
    type Item = Direction;

    type IntoIter = GornIterator;

    fn into_iter(self) -> Self::IntoIter {
        GornIterator { pos: 0, gorn: self }
    }
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
        v.index.set(v.size, d.into());
        v.size += 1;
        v
    }

    pub fn new(d: Direction) -> Self {
        let mut v = GornIndex {
            size: 0,
            index: IndexArray::default(),
        };
        v.index.set(v.size, d.into());
        v.size += 1;
        v
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FutureTree {
    pub node: NodeIndex,
    pub index: GornIndex,
    pub id: RuleIndex,
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
pub(crate) struct ParseMoment {
    pub tree: FutureTree,
    pub stolen_head: Option<StolenHead>,
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

    pub fn new(
        tree: FutureTree,
        movers: ThinVec<FutureTree>,
        stolen_head: Option<StolenHead>,
    ) -> Self {
        ParseMoment {
            tree,
            movers,
            stolen_head,
        }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StolenHead {
    Stealer(usize),
    StolenHead(usize, GornIndex),
}

impl StolenHead {
    pub(crate) fn new_stolen(pos: usize, direction: Direction) -> Self {
        StolenHead::StolenHead(pos, GornIndex::new(direction))
    }
}
