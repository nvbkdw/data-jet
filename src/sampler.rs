use rand::seq::SliceRandom;
use rand::thread_rng;
use std::ops::Range;

// ---------------------------------------------------------------------------
// Len trait
// ---------------------------------------------------------------------------

/// Basic trait for anything that has a length.
pub trait Len {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Sampler trait
// ---------------------------------------------------------------------------

/// Every Sampler is iterable, has a length, and can be copied.
pub trait Sampler: Len + IntoIterator<Item = usize> + Copy {
    /// Create a new sampler from the dataset length.
    fn new(data_source_len: usize) -> Self;
}

// ---------------------------------------------------------------------------
// SequentialSampler
// ---------------------------------------------------------------------------

/// Yields indices from zero to `data_source_len` in ascending order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequentialSampler {
    pub data_source_len: usize,
}

impl Sampler for SequentialSampler {
    fn new(data_source_len: usize) -> Self {
        Self { data_source_len }
    }
}

impl Len for SequentialSampler {
    fn len(&self) -> usize {
        self.data_source_len
    }
}

impl IntoIterator for SequentialSampler {
    type Item = usize;
    type IntoIter = Range<usize>;
    fn into_iter(self) -> Self::IntoIter {
        0..self.data_source_len
    }
}

// ---------------------------------------------------------------------------
// RandomSampler
// ---------------------------------------------------------------------------

/// Sampler that returns indices in random order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RandomSampler {
    data_source_len: usize,
}

impl Sampler for RandomSampler {
    fn new(data_source_len: usize) -> Self {
        Self { data_source_len }
    }
}

impl Len for RandomSampler {
    fn len(&self) -> usize {
        self.data_source_len
    }
}

impl IntoIterator for RandomSampler {
    type Item = usize;
    type IntoIter = RandomSamplerIter;
    fn into_iter(self) -> Self::IntoIter {
        let mut vec: Vec<usize> = (0..self.data_source_len).collect();
        vec.shuffle(&mut thread_rng());
        RandomSamplerIter { indexes: vec, idx: 0 }
    }
}

/// Iterator that yields indices in shuffled order.
#[derive(Debug)]
pub struct RandomSamplerIter {
    indexes: Vec<usize>,
    idx: usize,
}

impl Iterator for RandomSamplerIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.indexes.len() {
            self.idx += 1;
            Some(self.indexes[self.idx - 1])
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.indexes.len() - self.idx;
        (len, Some(len))
    }
}

impl ExactSizeIterator for RandomSamplerIter {}

// ---------------------------------------------------------------------------
// BatchSampler
// ---------------------------------------------------------------------------

/// Wraps another sampler to yield mini-batches of indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchSampler<S = SequentialSampler> {
    pub sampler: S,
    pub batch_size: usize,
    pub drop_last: bool,
}

impl<S: Sampler> Len for BatchSampler<S> {
    fn len(&self) -> usize {
        if self.drop_last {
            self.sampler.len() / self.batch_size
        } else {
            (self.sampler.len() + self.batch_size - 1) / self.batch_size
        }
    }
}

impl<S: Sampler> BatchSampler<S> {
    pub fn iter(&self) -> BatchIterator<S::IntoIter> {
        BatchIterator {
            sampler: self.sampler.into_iter(),
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }
}

impl<S: Sampler> IntoIterator for &BatchSampler<S> {
    type IntoIter = BatchIterator<<S as IntoIterator>::IntoIter>;
    type Item = Vec<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator that yields batches of indices.
#[derive(Debug)]
pub struct BatchIterator<I: Iterator<Item = usize>> {
    sampler: I,
    batch_size: usize,
    drop_last: bool,
}

impl<I: Iterator<Item = usize>> Iterator for BatchIterator<I> {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut current_idx = self.sampler.next();
        while let Some(idx) = current_idx {
            batch.push(idx);
            if batch.len() == self.batch_size {
                return Some(batch);
            }
            current_idx = self.sampler.next();
        }
        if !batch.is_empty() && !self.drop_last {
            return Some(batch);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, _) = self.sampler.size_hint();
        let lower = if self.drop_last {
            lower / self.batch_size
        } else {
            (lower + self.batch_size - 1) / self.batch_size
        };
        (lower, Some(lower))
    }
}

impl<I: Iterator<Item = usize> + ExactSizeIterator> ExactSizeIterator for BatchIterator<I> {}
