use ai_dataloader::collate::Collate;
use ai_dataloader::indexable::DataLoader;
use ai_dataloader::{Dataset, GetSample, Len};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

impl GetSample for TensorDataset {
    type Sample = (Vec<f64>, f64);

    fn get_sample(&self, index: usize) -> Self::Sample {
        (self.features[index].clone(), self.labels[index])
    }
}

impl Dataset for TensorDataset {}

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

    /// Build the Rust DataLoader, iterate entirely in Rust, and return
    /// a Python iterator that yields `(features, labels)` numpy arrays.
    fn __iter__(&self) -> DataJetIterator {
        let dataset = TensorDataset {
            features: self.features.clone(),
            labels: self.labels.clone(),
        };

        let batches: Vec<FlatBatch> = if self.shuffle {
            let mut builder = DataLoader::builder(dataset)
                .batch_size(self.batch_size)
                .collate_fn(FlatCollate)
                .shuffle();
            if self.drop_last {
                builder = builder.drop_last();
            }
            builder.build().iter().collect()
        } else {
            let mut builder = DataLoader::builder(dataset)
                .batch_size(self.batch_size)
                .collate_fn(FlatCollate);
            if self.drop_last {
                builder = builder.drop_last();
            }
            builder.build().iter().collect()
        };

        let stored = batches
            .into_iter()
            .map(|b| (b.features, b.nrows, b.ncols, b.labels))
            .collect();

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
    Ok(())
}
