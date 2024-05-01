use std::marker::PhantomData;

use crate::{
    parsing::{beam::Beam, ParseHolder},
    ParsingConfig,
};
use allocator_api2::alloc::{Allocator, Global};
use min_max_heap::MinMaxHeap;

#[derive(Debug, Clone)]
pub struct ParseHeap<'a, T, B: Beam<T>, A: Allocator = Global> {
    parse_heap: MinMaxHeap<B, A>,
    phantom: PhantomData<T>,
    global_steps: usize,
    config: &'a ParsingConfig,
}

impl<T, B: Beam<T>, A: Allocator> ParseHolder<T, B> for ParseHeap<'_, T, B, A> {
    fn add(&mut self, beam: B) {
        self.global_steps += 1;
        let mut pushable = true;
        if let Some(max_steps) = self.config.global_steps {
            pushable = self.global_steps < max_steps;
        }

        if let Some(min_log_prob) = self.config.min_log_prob {
            if beam.log_prob() < min_log_prob {
                pushable = false;
            }
        }

        if let Some(max_steps) = self.config.max_steps {
            if beam.n_steps() > max_steps {
                pushable = false;
            }
        }

        if pushable && beam.pushable(self.config) {
            if let Some(max_beams) = self.config.max_beams {
                if self.parse_heap.len() > max_beams {
                    self.parse_heap.push_pop_min(beam);
                } else {
                    self.parse_heap.push(beam);
                }
            } else {
                self.parse_heap.push(beam);
            }
        }
    }
}

impl<'a, T, B: Beam<T>, A: Allocator> ParseHeap<'a, T, B, A>
where
    B: Ord,
{
    pub fn new(parse_heap: MinMaxHeap<B, A>, config: &'a ParsingConfig) -> Self {
        ParseHeap {
            global_steps: 0,
            parse_heap,
            config,
            phantom: PhantomData,
        }
    }

    pub fn pop(&mut self) -> Option<B> {
        self.parse_heap.pop_max()
    }
}
