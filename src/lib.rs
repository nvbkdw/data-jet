mod collate;
mod sampler;
#[cfg(feature = "curvine")]
mod example;

use collate::Collate;
use sampler::{BatchSampler, Len, RandomSampler, Sampler, SequentialSampler};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Runtime;

// ---------------------------------------------------------------------------
// Global tokio runtime (lazy, mirrors ai-dataloader's THREAD_POOL pattern)
// ---------------------------------------------------------------------------

static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    TOKIO_RUNTIME.get_or_init(|| Runtime::new().expect("failed to create tokio runtime"))
}

// ---------------------------------------------------------------------------
// Dataset: holds feature matrix + label vector in contiguous Rust memory
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TensorDataset {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
}

impl Len for TensorDataset {
    fn len(&self) -> usize {
        self.features.len()
    }
}

// ---------------------------------------------------------------------------
// Async sample fetching
// ---------------------------------------------------------------------------

trait AsyncGetSample: Send + Sync {
    type Sample: Send + 'static;
    fn get_sample(
        &self,
        index: usize,
    ) -> impl std::future::Future<Output = Self::Sample> + Send;
}

impl AsyncGetSample for TensorDataset {
    type Sample = (Vec<f64>, f64);
    fn get_sample(
        &self,
        index: usize,
    ) -> impl std::future::Future<Output = Self::Sample> + Send {
        let sample = (self.features[index].clone(), self.labels[index]);
        async move { sample }
    }
}

async fn fetch_batch_async(dataset: Arc<TensorDataset>, indices: Vec<usize>) -> FlatBatch {
    let mut handles = Vec::with_capacity(indices.len());
    for idx in indices {
        let ds = Arc::clone(&dataset);
        handles.push(tokio::spawn(async move { ds.get_sample(idx).await }));
    }

    let mut samples = Vec::with_capacity(handles.len());
    for handle in handles {
        samples.push(handle.await.expect("sample fetch task panicked"));
    }

    FlatCollate.collate(samples)
}

// ---------------------------------------------------------------------------
// Collate: batch samples into flat vecs with shape metadata
// ---------------------------------------------------------------------------

/// Collated batch stored as flat contiguous memory for zero-copy handoff.
struct FlatBatch {
    features: Vec<f64>, // row-major [batch_size, feature_dim]
    nrows: usize,
    ncols: usize,
    labels: Vec<f64>,
}

#[derive(Clone, Debug)]
struct FlatCollate;

impl Collate<(Vec<f64>, f64)> for FlatCollate {
    type Output = FlatBatch;

    fn collate(&self, batch: Vec<(Vec<f64>, f64)>) -> Self::Output {
        let nrows = batch.len();
        let ncols = batch[0].0.len();

        let mut features = Vec::with_capacity(nrows * ncols);
        let mut labels = Vec::with_capacity(nrows);

        for (feat, label) in batch {
            features.extend(feat);
            labels.push(label);
        }

        FlatBatch {
            features,
            nrows,
            ncols,
            labels,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch iterator — yields (numpy.ndarray, numpy.ndarray) per batch
// ---------------------------------------------------------------------------

/// Pre-computed batch iterator. Each `__next__` converts one Rust-owned batch
/// into a pair of numpy arrays via zero-copy ownership transfer.
#[pyclass]
struct DataJetIterator {
    /// (features_flat, nrows, ncols, labels) per batch — GIL-free storage.
    batches: Vec<(Vec<f64>, usize, usize, Vec<f64>)>,
    pos: usize,
}

#[pymethods]
impl DataJetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Zero-copy path:
    /// 1. `PyArray1::from_vec` transfers Vec ownership to NumPy — no memcpy.
    /// 2. `.reshape` creates a view with the correct shape — no memcpy.
    fn __next__(&mut self, py: Python<'_>) -> Option<PyObject> {
        if self.pos >= self.batches.len() {
            return None;
        }

        let (features_flat, nrows, ncols, labels) =
            std::mem::take(&mut self.batches[self.pos]);
        self.pos += 1;

        // from_vec: ownership transfer, zero-copy.
        let flat_array = PyArray1::from_vec(py, features_flat);
        // reshape into 2-D: returns a new view, no data copy.
        let py_features = flat_array
            .reshape([nrows, ncols])
            .expect("reshape failed");
        let py_labels = PyArray1::from_vec(py, labels);

        Some((py_features, py_labels).into_pyobject(py).unwrap().into())
    }

    fn __len__(&self) -> usize {
        self.batches.len() - self.pos
    }
}

// ---------------------------------------------------------------------------
// Python-facing DataLoader
// ---------------------------------------------------------------------------

/// High-performance DataLoader backed by Rust's `ai-dataloader` crate.
///
/// Data is loaded and batched entirely in Rust. Batches are passed to Python
/// as numpy arrays using zero-copy ownership transfer (no memcpy).
#[pyclass(name = "RustDataLoader")]
struct RustDataLoader {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

#[pymethods]
impl RustDataLoader {
    #[new]
    #[pyo3(signature = (features, labels, batch_size=1, shuffle=false, drop_last=false))]
    fn new(
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> PyResult<Self> {
        let features: Vec<Vec<f64>> = features.extract()?;
        let labels: Vec<f64> = labels.extract()?;

        if features.is_empty() {
            return Err(PyValueError::new_err("features must not be empty"));
        }
        if features.len() != labels.len() {
            return Err(PyValueError::new_err(
                "features and labels must have the same length",
            ));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }

        Ok(Self {
            features,
            labels,
            batch_size,
            shuffle,
            drop_last,
        })
    }

    fn __len__(&self) -> usize {
        let n = self.features.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Build batch indices via ai-dataloader's samplers, fetch each batch's
    /// samples in parallel on the tokio runtime, and return a Python iterator
    /// that yields `(features, labels)` numpy arrays.
    fn __iter__(&self, py: Python<'_>) -> DataJetIterator {
        let dataset = Arc::new(TensorDataset {
            features: self.features.clone(),
            labels: self.labels.clone(),
        });
        let batch_size = self.batch_size;
        let drop_last = self.drop_last;
        let shuffle = self.shuffle;
        let dataset_len = dataset.len();

        // Release the GIL while doing Rust computation
        let stored = py.allow_threads(|| {
            get_runtime().block_on(async {
                let batch_indices: Vec<Vec<usize>> = if shuffle {
                    BatchSampler {
                        sampler: RandomSampler::new(dataset_len),
                        batch_size,
                        drop_last,
                    }
                    .iter()
                    .collect()
                } else {
                    BatchSampler {
                        sampler: SequentialSampler::new(dataset_len),
                        batch_size,
                        drop_last,
                    }
                    .iter()
                    .collect()
                };

                let mut batches = Vec::with_capacity(batch_indices.len());
                for indices in batch_indices {
                    let batch = fetch_batch_async(Arc::clone(&dataset), indices).await;
                    batches.push((batch.features, batch.nrows, batch.ncols, batch.labels));
                }
                batches
            })
        });

        DataJetIterator {
            batches: stored,
            pos: 0,
        }
    }

    #[getter]
    fn dataset_len(&self) -> usize {
        self.features.len()
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    fn shuffle(&self) -> bool {
        self.shuffle
    }

    #[getter]
    fn drop_last(&self) -> bool {
        self.drop_last
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustDataLoader>()?;
    #[cfg(feature = "curvine")]
    {
        m.add_class::<example::CurvineDataLoader>()?;
    }
    Ok(())
}
