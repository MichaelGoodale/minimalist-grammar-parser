//! Module to define the core parsing algorithm used to generate or parse strings from MGs
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::lexicon::{Feature, FeatureOrLemma, LexemeId, Lexicon, ParsingError};
use crate::parsing::rules::StolenInfo;
use crate::parsing::trees::StolenHead;
use crate::{Direction, ParseHeap, ParsingConfig};
use beam::Scanner;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use thin_vec::{ThinVec, thin_vec};
pub use trees::GornIndex;
use trees::{FutureTree, ParseMoment};

use rules::Rule;
pub use rules::RulePool;
pub(crate) use rules::{PartialRulePool, RuleHolder, RuleIndex};

#[derive(Debug, Clone)]
pub(crate) struct BeamWrapper<T, B: Scanner<T>> {
    log_prob: LogProb<f64>,
    queue: BinaryHeap<Reverse<ParseMoment>>,
    heads: Vec<PossibleHeads>,
    rules: PartialRulePool,
    n_consec_empties: usize,
    pub beam: B,
    phantom: PhantomData<T>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum PossibleHeads {
    Possibles(PossibleTree),
    FoundTree(HeadTree, RuleIndex),
    #[default]
    PlaceHolder,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PossibleTree {
    heads: Vec<LexemeId>,
    edges: BTreeMap<usize, Vec<(Direction, usize)>>,
}

impl PossibleTree {
    ///Goes through possible trees and keeps only that follow the GornIndex (which needn't go to
    ///the bottom), it also only keeps nodes at the destination are children of [`node`].
    ///
    ///Returns false if the tree is now empty.
    fn filter_node<T: Eq, C: Eq>(
        &mut self,
        node: NodeIndex,
        d: GornIndex,
        lexicon: &Lexicon<T, C>,
    ) -> bool {
        let mut heads: Vec<_> = self
            .edges
            .get(&0)
            .unwrap()
            .iter()
            .map(|(_, x)| *x)
            .collect();

        let mut old_heads = vec![];
        for dir in d {
            let mut new_heads = vec![];
            for head in heads.iter() {
                if let Some(x) = self.edges.get_mut(head) {
                    x.retain(|(d, _)| d == &dir);
                    new_heads.extend(x.iter().map(|(_, x)| *x));
                }
            }
            if new_heads.is_empty() {
                return false;
            }
            old_heads = heads.clone();
            heads = new_heads;
        }

        heads.retain(|x| {
            if node == lexicon.parent_of(self.heads.get(x - 1).unwrap().0).unwrap() {
                true
            } else {
                old_heads.iter().for_each(|k| {
                    self.edges.remove(k);
                });
                false
            }
        });
        !heads.is_empty()
    }

    pub(crate) fn add_head(&mut self, head: LexemeId) -> usize {
        self.heads.push(head);
        self.heads.len() //1-indexed so that 0 is always the root (which is not directly
        //represented)
    }

    pub(crate) fn add_edge(&mut self, source: usize, end: usize, dir: Direction) {
        self.edges.entry(source).or_default().push((dir, end));
    }

    pub(crate) fn new() -> Self {
        PossibleTree {
            heads: vec![],
            edges: BTreeMap::default(),
        }
    }

    fn to_trees<'a>(&'a self, child_node: LexemeId) -> PossibleTreeIterator<'a> {
        PossibleTreeIterator {
            stack: self
                .edges
                .get(&0)
                .unwrap()
                .iter()
                .filter_map(|(_, x)| {
                    let head = *self.heads.get(x - 1).unwrap();
                    if child_node == head {
                        Some((
                            *x,
                            HeadTree {
                                heads: vec![(head, None)],
                            },
                        ))
                    } else {
                        None
                    }
                })
                .collect(),
            trees: self,
            buffer: vec![],
        }
    }
}

struct PossibleTreeIterator<'a> {
    stack: Vec<(usize, HeadTree)>,
    trees: &'a PossibleTree,
    buffer: Vec<HeadTree>,
}
impl Iterator for PossibleTreeIterator<'_> {
    type Item = HeadTree;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.buffer.pop() {
            return Some(b);
        }

        while let Some((pos, head)) = self.stack.pop() {
            match self.trees.edges.get(&pos) {
                Some(children) => {
                    for ((dir, pos), mut head) in children
                        .iter()
                        .zip(itertools::repeat_n(head, children.len()))
                    {
                        head.add_new_child(*self.trees.heads.get(pos - 1).unwrap(), *dir);
                        self.stack.push((*pos, head));
                    }
                }
                None => self.buffer.push(head),
            }
        }

        self.buffer.pop()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HeadTree {
    heads: Vec<(LexemeId, Option<Direction>)>,
}

impl HeadTree {
    fn add_new_child(&mut self, head: LexemeId, dir: Direction) {
        self.heads.last_mut().unwrap().1 = Some(dir);
        self.heads.push((head, None));
    }
}
impl HeadTree {
    fn to_filled_head<T: Eq + Clone, C: Eq>(&self, lex: &Lexicon<T, C>) -> Vec<T> {
        let mut v = Vec::with_capacity(self.heads.len());
        let mut pos = 0;
        let mut dir = Direction::Left;
        for (h, d) in self.heads.iter() {
            if let Some(h) = lex.leaf_to_lemma(*h).unwrap().as_ref().cloned() {
                match dir {
                    Direction::Left => v.insert(pos, h),
                    Direction::Right => {
                        pos += 1;
                        v.insert(pos, h);
                    }
                }
                if let Some(d) = d {
                    dir = *d;
                } else {
                    break;
                }
            }
        }
        v
    }

    pub fn find_node(&self, index: GornIndex) -> Option<LexemeId> {
        dbg!(index, self);
        let mut pos = 0;
        for (d, (_, other_d)) in index.into_iter().zip(&self.heads) {
            if let Some(other_d) = other_d
                && other_d == &d
            {
                pos += 1;
            } else {
                println!("NONE");
                return None;
            }
        }

        self.heads.get(pos).map(|(x, _)| *x)
        /*
        let mut pos = self;
        let mut node = self.head;
        for d in index {
            if let Some((child, child_d)) = &pos.child
                && *child_d == d
            {
                node = child.head;
                pos = child;
            } else {
                return None;
            }
        }
        Some(node)*/
    }
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> PartialEq for BeamWrapper<T, B> {
    fn eq(&self, other: &Self) -> bool {
        self.beam == other.beam
            && self.log_prob == other.log_prob
            && self.queue.clone().into_sorted_vec() == other.queue.clone().into_sorted_vec()
    }
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> Eq for BeamWrapper<T, B> {}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> PartialOrd for BeamWrapper<T, B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> Ord for BeamWrapper<T, B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.log_prob.cmp(&other.log_prob)
    }
}

impl<T: Clone + Eq + std::fmt::Debug, B: Scanner<T> + Eq + Clone> BeamWrapper<T, B> {
    fn check_node(&self, index: GornIndex, node: LexemeId, headtree: &HeadTree) -> bool {
        if let Some(found_node) = headtree.find_node(index) {
            found_node == node
        } else {
            false
        }
    }

    fn scan<Category: Eq + std::fmt::Debug + Clone>(
        mut self,
        v: &mut ParseHeap<T, B>,
        moment: &ParseMoment,
        s: &Option<T>,
        child_node: LexemeId,
        child_prob: LogProb<f64>,
        lexicon: &Lexicon<T, Category>,
    ) {
        match s {
            Some(_) => self.n_consec_empties = 0,
            None => self.n_consec_empties += 1,
        }
        let rule = match moment.stolen_head {
            Some(StolenHead::Stealer(pos)) => {
                let trees = std::mem::take(&mut self.heads[pos]);

                let PossibleHeads::Possibles(possible_trees) = trees else {
                    panic!()
                };

                for (filled_head_tree, id_tree) in possible_trees
                    .to_trees(child_node)
                    .map(|x| (x.to_filled_head(lexicon), x))
                {
                    let mut new_beam = self.clone();
                    if new_beam.beam.multiscan(filled_head_tree) {
                        new_beam.heads[pos] = PossibleHeads::FoundTree(id_tree, moment.tree.id);
                        new_beam.log_prob += child_prob;
                        new_beam.rules.push_rule(
                            v.rules_mut(),
                            Rule::Scan {
                                lexeme: child_node,
                                stolen: StolenInfo::Stealer,
                            },
                            moment.tree.id,
                        );
                        v.push(new_beam);
                    }
                }
                None
            }
            Some(StolenHead::StolenHead(pos, index)) => {
                let PossibleHeads::FoundTree(headtree, rule_id) = &self.heads[pos] else {
                    panic!()
                };

                if self.check_node(index, child_node, headtree) {
                    Some(Rule::Scan {
                        lexeme: child_node,
                        stolen: StolenInfo::Stolen(*rule_id, index),
                    })
                } else {
                    None
                }
            }
            None => {
                if self.beam.scan(s) {
                    Some(Rule::Scan {
                        lexeme: child_node,
                        stolen: StolenInfo::Normal,
                    })
                } else {
                    None
                }
            }
        };

        if let Some(rule) = rule {
            self.log_prob += child_prob;
            self.rules.push_rule(v.rules_mut(), rule, moment.tree.id);
            v.push(self);
        }
    }

    fn new_future_tree(&mut self, node: NodeIndex, index: GornIndex) -> (FutureTree, RuleIndex) {
        let id = self.rules.fresh();
        (FutureTree { node, index, id }, id)
    }

    fn push_moment(
        &mut self,
        node: NodeIndex,
        index: GornIndex,
        movers: ThinVec<FutureTree>,
        stolen_head: Option<StolenHead>,
    ) -> RuleIndex {
        let (tree, id) = self.new_future_tree(node, index);
        self.queue.push(Reverse(ParseMoment {
            tree,
            movers,
            stolen_head,
        }));
        id
    }

    pub(super) fn new(beam: B, category_index: NodeIndex) -> Self {
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::with_capacity(5);
        queue.push(Reverse(ParseMoment::new(
            FutureTree {
                node: category_index,
                index: GornIndex::default(),
                id: RuleIndex::one(),
            },
            thin_vec![],
            None,
        )));
        BeamWrapper {
            beam,
            heads: vec![],
            n_consec_empties: 0,
            queue,
            log_prob: LogProb::prob_of_one(),
            rules: PartialRulePool::default(),
            phantom: PhantomData,
        }
    }

    pub fn log_prob(&self) -> LogProb<f64> {
        self.log_prob
    }

    pub fn n_steps(&self) -> usize {
        self.rules.n_steps()
    }

    pub fn n_consecutive_empty(&self) -> usize {
        self.n_consec_empties
    }

    pub fn pop_moment(&mut self) -> Option<ParseMoment> {
        if let Some(Reverse(x)) = self.queue.pop() {
            Some(x)
        } else {
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

fn clone_push<T: Clone>(v: &[T], x: T) -> ThinVec<T> {
    let mut v: ThinVec<T> = ThinVec::from(v);
    v.push(x);
    v
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmerge_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Clone + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    beam: &BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    dir: Direction,
    child_prob: LogProb<f64>,
) -> bool {
    let mut new_beam = false;
    for mover in moment.movers.iter() {
        for stored_child_node in lexicon.children_of(mover.node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Category(stored)) if stored == cat => {
                    let mut beam = beam.clone();
                    let movers = moment
                        .movers
                        .iter()
                        .filter(|&v| v != mover)
                        .cloned()
                        .collect();

                    let (stored_movers, child_movers) = if lexicon.is_complement(child_node) {
                        (movers, thin_vec![])
                    } else {
                        (thin_vec![], movers)
                    };

                    let child_id = beam.push_moment(
                        child_node,
                        moment.tree.index,
                        child_movers,
                        moment.stolen_head,
                    );
                    let stored_id =
                        beam.push_moment(stored_child_node, mover.index, stored_movers, None);

                    beam.log_prob += stored_prob + child_prob;

                    let trace_id = beam.rules.fresh_trace();

                    beam.rules.push_rule(
                        v.rules_mut(),
                        Rule::UnmergeFromMover {
                            child: child_node,
                            child_id,
                            stored_id,
                            trace_id,
                            destination_id: mover.id,
                            dir,
                        },
                        moment.tree.id,
                    );
                    v.push(beam);
                    new_beam = true;
                }
                _ => (),
            }
        }
    }
    new_beam
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmerge<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    dir: Direction,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    head_info: HeadMovement,
) -> Result<(), ParsingError<Category>> {
    let complement = lexicon.find_category(cat)?;

    //This enforces the SpICmg constraint (it could be loosened by determining how to divide the
    //subsets of movers)
    let (complement_movers, child_movers) = if lexicon.is_complement(child_node) {
        (moment.movers.clone(), thin_vec![])
    } else {
        (thin_vec![], moment.movers.clone())
    };

    let (child_head, comp_head) = match head_info {
        HeadMovement::HeadMovement { stealer, stolen } => (Some(stealer), Some(stolen)),
        HeadMovement::Inherit => (moment.stolen_head, None),
    };

    let complement_id = beam.push_moment(
        complement,
        moment.tree.index.clone_push(dir),
        complement_movers,
        comp_head,
    );
    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(dir.flip()),
        child_movers,
        child_head,
    );

    beam.log_prob += child_prob;
    beam.rules.push_rule(
        v.rules_mut(),
        Rule::Unmerge {
            child: child_node,
            child_id,
            complement_id,
            dir,
            affix: head_info != HeadMovement::Inherit, //This means we're in an affix merge
        },
        moment.tree.id,
    );
    v.push(beam);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove_from_mover<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Clone + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    beam: &BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    already_mover_of_this_cat: bool,
) -> bool {
    let mut new_beam_found = false;

    for mover in moment
        .movers
        .iter()
        .filter(|x| lexicon.get_feature_category(x.node) == Some(cat) || !already_mover_of_this_cat)
    //This checks for the SMC. This is because we can't add a new moved -x if we've already got
    //a -x in our movers. Unless of course, that -x is achieved by getting rid of a -x that is
    //already there. E.g. if the movers are [<-w, -w>, <-x, -w>] we can turn the <-w, -w> to -w
    //since we still will only have one -w in the movers. We can't do <-x, -w> since that would
    //lead to two -w in the movers at the same time, violating the SMC.
    {
        for stored_child_node in lexicon.children_of(mover.node) {
            let (stored, stored_prob) = lexicon.get(stored_child_node).unwrap();
            match stored {
                FeatureOrLemma::Feature(Feature::Licensee(s)) if cat == s => {
                    let mut beam = beam.clone();
                    let (stored_tree, stored_id) =
                        beam.new_future_tree(stored_child_node, mover.index);

                    let movers = moment
                        .movers
                        .iter()
                        .filter(|&v| v != mover)
                        .cloned()
                        .chain(std::iter::once(stored_tree))
                        .collect();

                    let child_id =
                        beam.push_moment(child_node, moment.tree.index, movers, moment.stolen_head);
                    beam.log_prob += stored_prob + child_prob;

                    let trace_id = beam.rules.fresh_trace();
                    beam.rules.push_rule(
                        v.rules_mut(),
                        Rule::UnmoveFromMover {
                            child_id,
                            stored_id,
                            trace_id,
                            destination_id: mover.id,
                        },
                        moment.tree.id,
                    );
                    v.push(beam);
                    new_beam_found = true;
                }
                _ => (),
            }
        }
    }
    new_beam_found
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
) -> Result<(), ParsingError<Category>> {
    let stored = lexicon.find_licensee(cat)?;
    let (stored, stored_id) =
        beam.new_future_tree(stored, moment.tree.index.clone_push(Direction::Left));

    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(Direction::Right),
        clone_push(&moment.movers, stored),
        moment.stolen_head,
    );

    beam.log_prob += child_prob;
    beam.rules.push_rule(
        v.rules_mut(),
        Rule::Unmove {
            child_id,
            stored_id,
        },
        moment.tree.id,
    );
    v.push(beam);
    Ok(())
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum HeadMovement {
    HeadMovement {
        stealer: StolenHead,
        stolen: StolenHead,
    },
    Inherit,
}

#[inline]
fn set_beam_head<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone,
>(
    moment: &ParseMoment,
    dir: Direction,
    lexicon: &Lexicon<U, Category>,
    child_node: NodeIndex,
    beam: &mut BeamWrapper<T, B>,
) -> Result<Option<HeadMovement>, ParsingError<Category>> {
    match moment.stolen_head {
        Some(StolenHead::Stealer(_)) => panic!(
            "Only possible if there are multiple head movements to one lemma which is not yet supported"
        ),
        Some(StolenHead::StolenHead(pos, index)) => {
            match &mut beam.heads[pos] {
                PossibleHeads::Possibles(head_trees) => {
                    if !head_trees.filter_node(child_node, index, lexicon) {
                        return Ok(None);
                    }
                }
                PossibleHeads::FoundTree(heads, _) => {
                    let Some(node) = heads.find_node(index) else {
                        //Wrong shape of head movement
                        return Ok(None);
                    };
                    if child_node != lexicon.parent_of(node.0).unwrap() {
                        //Wrong head for the head movement
                        return Ok(None);
                    }
                }
                PossibleHeads::PlaceHolder => {
                    panic!("This enum variant is just used for swapping!")
                }
            }
            Ok(Some(HeadMovement::HeadMovement {
                stealer: StolenHead::StolenHead(pos, index),
                stolen: StolenHead::StolenHead(pos, index.clone_push(dir)),
            }))
        }
        None => {
            let head_pos = beam.heads.len();
            let possible_heads = PossibleHeads::Possibles(lexicon.possible_heads(child_node)?);
            beam.heads.push(possible_heads);
            Ok(Some(HeadMovement::HeadMovement {
                stealer: StolenHead::Stealer(head_pos),
                stolen: StolenHead::new_stolen(head_pos, dir),
            }))
        }
    }
}

pub(crate) fn expand<
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone,
    L: Borrow<Lexicon<T, Category>>,
>(
    extender: &mut ParseHeap<T, B>,
    moment: ParseMoment,
    beam: BeamWrapper<T, B>,
    lexicon: L,
    config: &ParsingConfig,
) {
    let lexicon = lexicon.borrow();
    let n_children = lexicon.n_children(moment.tree.node);
    let new_beams = itertools::repeat_n(beam, n_children);

    new_beams
        .zip(lexicon.children_of(moment.tree.node))
        .for_each(
            |(mut beam, child_node)| match lexicon.get(child_node).unwrap() {
                (FeatureOrLemma::Lemma(s), p) if moment.no_movers() => {
                    beam.scan(extender, &moment, s, LexemeId(child_node), p, lexicon)
                }
                (FeatureOrLemma::Complement(cat, dir), mut p)
                | (FeatureOrLemma::Feature(Feature::Selector(cat, dir)), mut p) => {
                    if unmerge_from_mover(
                        extender,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        *dir,
                        p + config.move_prob,
                    ) {
                        p += config.dont_move_prob
                    }
                    let _ = unmerge(
                        extender,
                        lexicon,
                        &moment,
                        beam,
                        cat,
                        *dir,
                        child_node,
                        p,
                        HeadMovement::Inherit,
                    );
                }
                (FeatureOrLemma::Feature(Feature::Licensor(cat)), mut p) => {
                    let already_mover_of_this_cat = moment.movers.iter().any(|x| {
                        lexicon
                            .get_feature_category(x.node)
                            .map(|x| x == cat)
                            .unwrap_or(false)
                    });
                    if unmove_from_mover(
                        extender,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        p + config.move_prob,
                        already_mover_of_this_cat,
                    ) {
                        p += config.dont_move_prob
                    }
                    if !already_mover_of_this_cat {
                        //This corresponds to the SMC here.
                        let _ = unmove(extender, lexicon, &moment, beam, cat, child_node, p);
                    }
                }
                (FeatureOrLemma::Feature(Feature::Affix(cat, dir)), p) => {
                    if let Ok(Some(head_info)) =
                        set_beam_head(&moment, *dir, lexicon, child_node, &mut beam)
                    {
                        //We set dir=right since headmovement is always after a right-merge, even
                        //if you can put it to the left or right of the head
                        let _ = unmerge(
                            extender,
                            lexicon,
                            &moment,
                            beam,
                            cat,
                            Direction::Right,
                            child_node,
                            p,
                            head_info,
                        );
                    }
                }
                (FeatureOrLemma::Lemma(_), _)
                | (FeatureOrLemma::Feature(Feature::Category(_)), _)
                | (FeatureOrLemma::Feature(Feature::Licensee(_)), _) => (),
                (FeatureOrLemma::Root, _) => unimplemented!("Impossible to parse the root node"),
            },
        );
}

pub mod beam;
pub mod rules;

mod trees;
pub use trees::MAX_STEPS;

#[cfg(test)]
mod tests;
