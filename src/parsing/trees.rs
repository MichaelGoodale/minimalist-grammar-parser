use petgraph::graph::NodeIndex;

use crate::Direction;

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
pub struct GornIndex {
    pub index: Vec<Direction>,
}

impl GornIndex {
    pub fn clone_push(&self, d: Direction) -> Self {
        let mut v = self.clone();
        v.index.push(d);
        v
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutureTree {
    pub node: NodeIndex,
    pub index: GornIndex,
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
    pub movers: Vec<FutureTree>,
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
