//! Module to define the core parsing algorithm used to generate or parse strings from MGs
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::lexicon::{Feature, FeatureOrLemma, Lexicon, ParsingError};
use crate::parsing::trees::StolenHead;
use crate::{Direction, ParseHeap, ParsingConfig};
use beam::Scanner;
use logprob::LogProb;
use petgraph::graph::NodeIndex;
use thin_vec::{ThinVec, thin_vec};
use trees::{FutureTree, GornIndex, ParseMoment};

use rules::Rule;
pub use rules::RulePool;
pub(crate) use rules::{PartialRulePool, RuleHolder, RuleIndex};

#[derive(Debug, Clone)]
pub(crate) struct BeamWrapper<T, B: Scanner<T>> {
    log_prob: LogProb<f64>,
    queue: BinaryHeap<Reverse<ParseMoment>>,
    heads: Vec<PossibleHeads>,
    rules: PartialRulePool,
    pub beam: B,
    phantom: PhantomData<T>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PossibleHeads {
    head: NodeIndex,
    direction: Direction,
    possibilities: Vec<HeadTree>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HeadTree {
    possible_heads: Vec<NodeIndex>,
    possible_children: Option<HeadTreeChild>,
}

impl HeadTree {
    pub(crate) fn just_heads(nodes: Vec<NodeIndex>) -> Self {
        Self {
            possible_heads: nodes,
            possible_children: None,
        }
    }
    pub(crate) fn merge_left(mut self, child: HeadTree) -> Self {
        self.possible_children = Some(HeadTreeChild::LeftChild(Box::new(child)));
        self
    }
    pub(crate) fn merge_right(mut self, child: HeadTree) -> Self {
        self.possible_children = Some(HeadTreeChild::RightChild(Box::new(child)));
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum HeadTreeChild {
    LeftChild(Box<HeadTree>),
    RightChild(Box<HeadTree>),
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

impl<T: Eq + std::fmt::Debug, B: Scanner<T> + Eq> BeamWrapper<T, B> {
    fn scan(
        mut self,
        v: &mut ParseHeap<T, B>,
        moment: &ParseMoment,
        s: &Option<T>,
        child_node: NodeIndex,
        child_prob: LogProb<f64>,
    ) {
        if self.beam.scan(s) {
            self.log_prob += child_prob;
            self.rules.push_rule(
                v.rules_mut(),
                Rule::Scan { node: child_node },
                moment.tree.id,
            );
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
        let mut queue = BinaryHeap::<Reverse<ParseMoment>>::new();
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
    child_prob: LogProb<f64>,
    config: &ParsingConfig,
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

                    beam.log_prob += stored_prob + child_prob + config.move_prob;

                    let trace_id = beam.rules.fresh_trace();
                    beam.rules.push_rule(
                        v.rules_mut(),
                        Rule::UnmergeFromMover {
                            child: child_node,
                            child_id,
                            stored_id,
                            trace_id,
                            destination_id: mover.id,
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
    B: Scanner<T> + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    rule_prob: LogProb<f64>,
) -> Result<(), ParsingError<Category>> {
    let complement = lexicon.find_category(cat)?;

    //This enforces the SpICmg constraint (it could be loosened by determining how to divide the
    //subsets of movers)
    let (complement_movers, child_movers) = if lexicon.is_complement(child_node) {
        (moment.movers.clone(), thin_vec![])
    } else {
        (thin_vec![], moment.movers.clone())
    };

    let complement_id = beam.push_moment(
        complement,
        moment.tree.index.clone_push(*dir),
        complement_movers,
        None,
    );
    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(dir.flip()),
        child_movers,
        moment.stolen_head,
    );

    beam.log_prob += child_prob + rule_prob;
    beam.rules.push_rule(
        v.rules_mut(),
        Rule::Unmerge {
            child: child_node,
            child_id,
            complement_id,
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
    config: &ParsingConfig,
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
                    beam.log_prob += stored_prob + child_prob + config.move_prob;

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
    B: Scanner<T> + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
    rule_prob: LogProb<f64>,
) -> Result<(), ParsingError<Category>> {
    let stored = lexicon.find_licensee(cat)?;
    let (stored, stored_id) =
        beam.new_future_tree(stored, moment.tree.index.clone_push(Direction::Left));

    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(Direction::Right),
        clone_push(&moment.movers, stored),
        None,
    );

    beam.log_prob += child_prob + rule_prob;
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn unmove_head<
    T: Eq + std::fmt::Debug + Clone,
    U: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone + Hash,
    B: Scanner<T> + Eq,
>(
    v: &mut ParseHeap<T, B>,
    lexicon: &Lexicon<U, Category>,
    moment: &ParseMoment,
    mut beam: BeamWrapper<T, B>,
    cat: &Category,
    dir: &Direction,
    child_node: NodeIndex,
    child_prob: LogProb<f64>,
) -> Result<(), ParsingError<Category>> {
    //Not totally clear how to do this,
    //looks like we need to enrich upcoming FutureTrees such that we know whether they have had
    //their head stolen, and crucially, we need a way to check that the stolen head corresponds to
    //the one in this affix. The best way may be to keep track of the *destination* of stolen heads
    //

    let complement = lexicon.find_category(cat)?;
    let (complement_movers, child_movers) = (moment.movers.clone(), thin_vec![]);

    let head_pos = beam.heads.len();
    let possible_heads = PossibleHeads {
        head: child_node,
        direction: *dir,
        possibilities: lexicon.possible_heads(cat)?,
    };
    beam.heads.push(possible_heads);

    let complement_id = beam.push_moment(
        complement,
        moment.tree.index.clone_push(Direction::Right),
        complement_movers,
        Some(StolenHead::StolenHead(head_pos, *dir)),
    );
    let child_id = beam.push_moment(
        child_node,
        moment.tree.index.clone_push(Direction::Left),
        child_movers,
        Some(StolenHead::Stealer(head_pos)),
    );

    beam.log_prob += child_prob;
    beam.rules.push_rule(
        v.rules_mut(),
        Rule::Unmerge {
            child: child_node,
            child_id,
            complement_id,
        },
        moment.tree.id,
    );
    v.push(beam);
    Ok(())
}

pub(crate) fn expand<
    T: Eq + std::fmt::Debug + Clone,
    Category: Eq + std::fmt::Debug + Clone + Hash,
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
            |(beam, child_node)| match lexicon.get(child_node).unwrap() {
                (FeatureOrLemma::Lemma(s), p) if moment.no_movers() => {
                    beam.scan(extender, &moment, s, child_node, p)
                }
                (FeatureOrLemma::Complement(cat, dir), p)
                | (FeatureOrLemma::Feature(Feature::Selector(cat, dir)), p) => {
                    let new_beam_found = unmerge_from_mover(
                        extender, lexicon, &moment, &beam, cat, child_node, p, config,
                    );
                    let _ = unmerge(
                        extender,
                        lexicon,
                        &moment,
                        beam,
                        cat,
                        dir,
                        child_node,
                        p,
                        if new_beam_found {
                            config.dont_move_prob
                        } else {
                            LogProb::prob_of_one()
                        },
                    );
                }
                (FeatureOrLemma::Feature(Feature::Licensor(cat)), p) => {
                    let already_mover_of_this_cat = moment.movers.iter().any(|x| {
                        lexicon
                            .get_feature_category(x.node)
                            .map(|x| x == cat)
                            .unwrap_or(false)
                    });
                    let new_beam_found = unmove_from_mover(
                        extender,
                        lexicon,
                        &moment,
                        &beam,
                        cat,
                        child_node,
                        p,
                        config,
                        already_mover_of_this_cat,
                    );
                    if !already_mover_of_this_cat {
                        //This corresponds to the SMC here.
                        let _ = unmove(
                            extender,
                            lexicon,
                            &moment,
                            beam,
                            cat,
                            child_node,
                            p,
                            if new_beam_found {
                                config.dont_move_prob
                            } else {
                                LogProb::prob_of_one()
                            },
                        );
                    }
                }
                (FeatureOrLemma::Feature(Feature::Affix(cat, dir)), p) => {
                    let _ = unmove_head(extender, lexicon, &moment, beam, cat, dir, child_node, p);
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
