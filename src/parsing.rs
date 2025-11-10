//! Module to define the core parsing algorithm used to generate or parse strings from MGs
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
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
struct AffixedHead {
    heads: Vec<LexemeId>,
    pos: usize,
    head: usize,
    tree: Option<FutureTree>,
    current_index: GornIndex,
}

impl AffixedHead {
    fn insert(&mut self, nx: LexemeId, dir: Direction) {
        match dir {
            Direction::Left => self.heads.insert(self.pos, nx),
            Direction::Right => {
                self.pos += 1;
                self.heads.insert(self.pos, nx);
            }
        }
        if self.pos <= self.head {
            self.head += 1;
        }
        self.current_index.push(dir);
    }

    fn is_unfinished_and_before(&self, index: &GornIndex) -> bool {
        let Some(tree) = self.tree else {
            return false;
        };

        tree.index.comes_before(index)
    }

    fn new(nx: LexemeId, tree: FutureTree) -> Self {
        AffixedHead {
            heads: vec![nx],
            pos: 0,
            head: 0,
            tree: Some(tree),
            current_index: GornIndex::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BeamWrapper<T, B: Scanner<T>> {
    log_prob: LogProb<f64>,
    queue: BinaryHeap<Reverse<ParseMoment>>,
    head_buffer: Vec<(LexemeId, RuleIndex)>,
    heads: Vec<AffixedHead>,
    rules: PartialRulePool,
    n_consec_empties: usize,
    pub beam: B,
    phantom: PhantomData<T>,
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
            StolenHead::Stealer(head_id) => {
                let heads = &mut self.heads[head_id];
                // We change this one so that the head is the lemma we are at, previously we didn't
                // know which it was, so at that index was an affix feature.
                heads.heads[heads.head] = child_node;

                let heads = heads
                    .heads
                    .iter()
                    .filter_map(|x| lexicon.leaf_to_lemma(*x).unwrap().as_ref())
                    .collect();

                if self.beam.multiscan(heads) {
                    //All nodes for scanning (except for StolenHeads) were held until ready to be used here.
                    for (lexeme, rule_id) in self.head_buffer.drain(..) {
                        let s = lexicon.leaf_to_lemma(lexeme).unwrap();
                        if self.beam.scan(s) {
                            let p = lexicon.log_prob(lexeme.0);
                            self.log_prob += p;
                            self.rules.push_rule(
                                v.rules_mut(),
                                Rule::Scan {
                                    lexeme,
                                    stolen: StolenInfo::Normal,
                                },
                                rule_id,
                            );
                        } else {
                            //We stop early without adding the beam back on since its a failure
                            return;
                        }
                    }
                    Some(Rule::Scan {
                        lexeme: child_node,
                        stolen: StolenInfo::Stealer,
                    })
                } else {
                    None
                }
            }
            StolenHead::StolenHead {
                rule,
                direction,
                stealer_id,
                is_done,
            } => {
                if is_done {
                    let tree = self.heads[stealer_id].tree.take().unwrap();
                    self.add_tree_to_queue(tree, thin_vec![], StolenHead::Stealer(stealer_id));
                }
                self.heads[stealer_id].insert(child_node, direction);
                Some(Rule::Scan {
                    lexeme: child_node,
                    stolen: StolenInfo::Stolen(rule, self.heads[stealer_id].current_index),
                })
            }
            StolenHead::None => {
                if s.is_some()
                    && self
                        .heads
                        .iter()
                        .any(|x| x.is_unfinished_and_before(&moment.tree.index))
                {
                    //We need to figure out stolen heads before scanning this guy, so off to the
                    //buffer he goes
                    self.head_buffer.push((child_node, moment.tree.id));
                    v.push(self);
                    return;
                } else if self.beam.scan(s) {
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
        stolen_head_info: StolenHead,
    ) -> RuleIndex {
        let (tree, id) = self.new_future_tree(node, index);
        self.add_tree_to_queue(tree, movers, stolen_head_info);
        id
    }

    fn add_tree_to_queue(
        &mut self,
        tree: FutureTree,
        movers: ThinVec<FutureTree>,
        stolen_head: StolenHead,
    ) {
        self.queue.push(Reverse(ParseMoment {
            tree,
            movers,
            stolen_head,
        }));
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
            StolenHead::None,
        )));
        BeamWrapper {
            beam,
            n_consec_empties: 0,
            queue,
            heads: vec![],
            head_buffer: vec![],
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
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Clone + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<T, Category>,
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
                    let stored_id = beam.push_moment(
                        stored_child_node,
                        mover.index,
                        stored_movers,
                        StolenHead::None,
                    );

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
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<T, Category>,
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
        HeadMovement::HeadMovement { stealer, stolen } => (stealer, stolen),
        HeadMovement::Inherit => (moment.stolen_head, StolenHead::None),
    };

    let (child_tree, child_id) =
        beam.new_future_tree(child_node, moment.tree.index.clone_push(dir.flip()));
    let (complement_tree, complement_id) =
        beam.new_future_tree(complement, moment.tree.index.clone_push(dir));

    match (child_head, comp_head) {
        (StolenHead::None, StolenHead::None) => {
            //do normal stuff
            beam.add_tree_to_queue(child_tree, child_movers, StolenHead::None);
            beam.add_tree_to_queue(complement_tree, complement_movers, StolenHead::None);
        }
        (StolenHead::Stealer(_), stole @ StolenHead::StolenHead { .. }) => {
            //First time we are stealing a head!
            #[cfg(test)]
            assert!(child_movers.is_empty());

            beam.heads
                .push(AffixedHead::new(LexemeId(child_node), child_tree)); //assumes that affixes are
            //complements, e.g. that child_node is a lexeme
            beam.add_tree_to_queue(complement_tree, complement_movers, stole.with_done());
        }
        (StolenHead::Stealer(_), StolenHead::Stealer(_)) => panic!("A"),
        (
            stole_child @ StolenHead::StolenHead { .. },
            stolen_comp @ StolenHead::StolenHead { .. },
        ) => {
            #[cfg(test)]
            assert!(child_movers.is_empty());

            //Previous head was stolen, and we're stealing this one too.
            //the complement is behaving normally.
            beam.add_tree_to_queue(child_tree, child_movers, stole_child.with_not_done());
            beam.add_tree_to_queue(complement_tree, complement_movers, stolen_comp.with_done());
        }
        (stole @ StolenHead::StolenHead { .. }, StolenHead::None) => {
            //We are finished stealing heads, the child should go up to be with its stealer while
            //the complement is behaving normally.
            #[cfg(test)]
            assert!(child_movers.is_empty());
            beam.add_tree_to_queue(child_tree, child_movers, stole.with_done());
            beam.add_tree_to_queue(complement_tree, complement_movers, StolenHead::None);
        }
        (_, _) => panic!("Should be impossible"),
    };

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
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Clone + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<T, Category>,
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
    Category: Eq + std::fmt::Debug + Clone,
    B: Scanner<T> + Eq + Clone,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<T, Category>,
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
fn set_beam_head<T: Eq + std::fmt::Debug + Clone, B: Scanner<T> + Eq + Clone>(
    moment: &ParseMoment,
    dir: Direction,
    beam: &mut BeamWrapper<T, B>,
) -> HeadMovement {
    match moment.stolen_head {
        StolenHead::Stealer(_) => panic!(
            "Only possible if there are multiple head movements to one lemma which is not yet supported"
        ),
        StolenHead::StolenHead {
            stealer_id,
            rule,
            direction: old_dir,
            is_done,
        } => HeadMovement::HeadMovement {
            stealer: StolenHead::StolenHead {
                stealer_id,
                rule,
                direction: old_dir,
                is_done: false,
            },
            stolen: StolenHead::StolenHead {
                stealer_id,
                direction: dir,
                rule,
                is_done,
            },
        },
        StolenHead::None => {
            let id = beam.heads.len();
            HeadMovement::HeadMovement {
                stealer: StolenHead::Stealer(id),
                stolen: StolenHead::new_stolen(beam.rules.peek_fresh(), dir, id),
            }
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
                (FeatureOrLemma::Lemma(s), p) if moment.should_be_scanned() => {
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
                    let head_info = set_beam_head(&moment, *dir, &mut beam);
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
