use bytes::Bytes;
use curvine_client::file::CurvineFileSystem;
use curvine_common::conf::ClusterConf;
use curvine_common::fs::{Path as CvPath, Reader};
use curvine_common::FsResult;
use numpy::ndarray::ArrayView1;
use numpy::PyArray1;
use orpc::runtime::{RpcRuntime, Runtime as OrpcRuntime};
use orpc::sys::DataSlice;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};

// ---------------------------------------------------------------------------
// FileData — the result of reading one file, either zero-copy or copied
// ---------------------------------------------------------------------------

/// Holds a fully-read file's data.  Decided at read time (async phase) so
/// multi-chunk RDMA allocations are freed immediately while single-chunk
/// files keep their buffer for true zero-copy handoff to NumPy.
enum FileData {
    /// Single chunk — backed directly by an RDMA or TCP buffer.
    /// `bytes_to_numpy` creates a NumPy view with no memcpy.
    ZeroCopy(Bytes),
    /// Multiple chunks were concatenated into a contiguous Vec during the
    /// async read phase, immediately releasing the per-chunk RDMA allocations.
    /// `PyArray1::from_vec` transfers ownership with no memcpy.
    Copied(Vec<u8>),
}

// ---------------------------------------------------------------------------
// Zero-copy helpers
// ---------------------------------------------------------------------------

/// Convert a DataSlice into `bytes::Bytes` without copying when possible.
fn dataslice_into_bytes(slice: DataSlice) -> Bytes {
    match slice {
        DataSlice::Empty => Bytes::new(),
        DataSlice::Bytes(b) => b,
        DataSlice::Buffer(b) => b.freeze(),
        other => Bytes::from(other.as_slice().to_vec()),
    }
}

/// Py-visible owner that pins a `Bytes` buffer as the base object of a
/// NumPy array.  Prevented from being GC'd as long as the array exists.
#[pyclass]
struct BytesOwner {
    data: Bytes,
}

/// Create a NumPy uint8 array backed directly by `data` — true zero-copy.
fn bytes_to_numpy<'py>(
    py: Python<'py>,
    data: Bytes,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let owner = Bound::new(py, BytesOwner { data })?;
    let (ptr, len) = {
        let b = owner.borrow();
        let s: &[u8] = b.data.as_ref();
        (s.as_ptr(), s.len())
    };
    let view = unsafe { ArrayView1::from_shape_ptr(len, ptr) };
    Ok(unsafe { PyArray1::borrow_from_array(&view, owner.into_any()) })
}

// ---------------------------------------------------------------------------
// CurvineDataset — async file reads via curvine RDMA/TCP
// ---------------------------------------------------------------------------

struct CurvineDataset {
    fs: CurvineFileSystem,
    paths: Vec<CvPath>,
}

impl CurvineDataset {
    /// Read one file.  Returns `FileData::ZeroCopy` when the entire file
    /// comes back in a single chunk (common for files ≤ one RDMA block),
    /// or `FileData::Copied` after concatenating multiple chunks (which
    /// immediately drops the per-chunk Bytes and frees RDMA allocations).
    async fn read_file(&self, index: usize) -> FsResult<FileData> {
        let path = &self.paths[index];
        let mut reader = self.fs.open(path).await?;

        // First chunk
        let first = if reader.has_remaining() {
            let chunk = reader.async_read(None).await?;
            if chunk.is_empty() {
                reader.complete().await?;
                return Ok(FileData::Copied(Vec::new()));
            }
            dataslice_into_bytes(chunk)
        } else {
            reader.complete().await?;
            return Ok(FileData::Copied(Vec::new()));
        };

        // If the file was fully consumed in one chunk, return zero-copy.
        if !reader.has_remaining() {
            reader.complete().await?;
            return Ok(FileData::ZeroCopy(first));
        }

        // Multiple chunks — concatenate now so RDMA buffers are freed early.
        let file_len = reader.len() as usize;
        let mut buf = Vec::with_capacity(file_len);
        buf.extend_from_slice(&first);
        drop(first); // free first chunk's RDMA allocation immediately

        while reader.has_remaining() {
            let chunk = reader.async_read(None).await?;
            if chunk.is_empty() {
                break;
            }
            buf.extend_from_slice(chunk.as_slice());
            // chunk (DataSlice) dropped here → RDMA allocation freed
        }
        reader.complete().await?;
        Ok(FileData::Copied(buf))
    }
}

// ---------------------------------------------------------------------------
// CurvineIterator — pre-loaded batches, yields list[numpy.ndarray[uint8]]
// ---------------------------------------------------------------------------

#[pyclass]
pub struct CurvineIterator {
    /// Pre-loaded batches. Each batch is a Vec of files.
    batches: Vec<Vec<FileData>>,
    pos: usize,
}

#[pymethods]
impl CurvineIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Pure computation — no I/O, no blocking.
    /// Single-chunk files → `bytes_to_numpy` (zero-copy, no memcpy).
    /// Multi-chunk files → `PyArray1::from_vec` (zero-copy ownership transfer).
    fn __next__(&mut self, py: Python<'_>) -> Option<PyObject> {
        if self.pos >= self.batches.len() {
            return None;
        }

        let batch = std::mem::take(&mut self.batches[self.pos]);
        self.pos += 1;

        let py_list: Vec<PyObject> = batch
            .into_iter()
            .map(|file_data| match file_data {
                FileData::ZeroCopy(bytes) => bytes_to_numpy(py, bytes)
                    .expect("failed to create numpy array")
                    .into_pyobject(py)
                    .unwrap()
                    .into(),
                FileData::Copied(vec) => PyArray1::from_vec(py, vec)
                    .into_pyobject(py)
                    .unwrap()
                    .into(),
            })
            .collect();

        Some(py_list.into_pyobject(py).unwrap().into())
    }

    fn __len__(&self) -> usize {
        self.batches.len() - self.pos
    }
}

// ---------------------------------------------------------------------------
// CurvineDataLoader — Python-facing DataLoader for curvine files
// ---------------------------------------------------------------------------

#[pyclass(name = "CurvineDataLoader")]
pub struct CurvineDataLoader {
    dataset: Arc<CurvineDataset>,
    rt: Arc<OrpcRuntime>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    prefetch_size: usize,
}

#[pymethods]
impl CurvineDataLoader {
    #[new]
    #[pyo3(signature = (config_path, file_paths, batch_size=1, shuffle=false, drop_last=false, prefetch_size=0))]
    fn new(
        config_path: &str,
        file_paths: Vec<String>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        prefetch_size: usize,
    ) -> PyResult<Self> {
        if file_paths.is_empty() {
            return Err(PyValueError::new_err("file_paths must not be empty"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }

        // 0 means "use default": 10 * batch_size
        let prefetch_size = if prefetch_size == 0 {
            10 * batch_size
        } else {
            prefetch_size
        };

        let conf = ClusterConf::from(config_path)
            .map_err(|e| PyValueError::new_err(format!("failed to load config: {}", e)))?;

        let rt = Arc::new(OrpcRuntime::default("data-jet-curvine"));

        let fs = CurvineFileSystem::with_rt(conf, Arc::clone(&rt))
            .map_err(|e| PyIOError::new_err(format!("failed to create filesystem: {}", e)))?;

        let paths: Vec<CvPath> = file_paths
            .into_iter()
            .map(|p| CvPath::new(&p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyValueError::new_err(format!("invalid path: {}", e)))?;

        let dataset = Arc::new(CurvineDataset { fs, paths });

        Ok(Self {
            dataset,
            rt,
            batch_size,
            shuffle,
            drop_last,
            prefetch_size,
        })
    }

    fn __len__(&self) -> usize {
        let n = self.dataset.paths.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Releases the GIL, reads files in parallel on the orpc runtime
    /// (up to `prefetch_size` concurrently), assembles batches, and returns
    /// an iterator whose __next__ is pure computation (no I/O, no blocking).
    fn __iter__(&self, py: Python<'_>) -> PyResult<CurvineIterator> {
        let dataset = Arc::clone(&self.dataset);
        let rt = Arc::clone(&self.rt);
        let batch_size = self.batch_size;
        let drop_last = self.drop_last;
        let shuffle = self.shuffle;
        let prefetch_size = self.prefetch_size;
        let dataset_len = dataset.paths.len();

        let result = py.allow_threads(|| {
            rt.block_on(async {
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

                // Spawn file reads with semaphore-based concurrency control.
                let total_files: usize = batch_indices.iter().map(|b| b.len()).sum();
                let sem = Arc::new(tokio::sync::Semaphore::new(prefetch_size));
                let mut handles = Vec::with_capacity(total_files);
                let mut file_to_batch: Vec<(usize, usize)> =
                    Vec::with_capacity(total_files);

                for (batch_idx, indices) in batch_indices.iter().enumerate() {
                    for (pos_in_batch, &file_idx) in indices.iter().enumerate() {
                        file_to_batch.push((batch_idx, pos_in_batch));
                        let ds = Arc::clone(&dataset);
                        let permit = Arc::clone(&sem).acquire_owned().await.unwrap();
                        handles.push(rt.spawn(async move {
                            let result = ds.read_file(file_idx)
                                .await
                                .map_err(|e| format!("failed to read file {}: {}", file_idx, e));
                            drop(permit);
                            result
                        }));
                    }
                }

                // Collect results into batch structure.
                let mut batches: Vec<Vec<Option<FileData>>> = batch_indices
                    .iter()
                    .map(|b| (0..b.len()).map(|_| None).collect())
                    .collect();

                for (i, handle) in handles.into_iter().enumerate() {
                    let file_data = handle
                        .await
                        .map_err(|e| format!("task join error: {}", e))??;
                    let (batch_idx, pos) = file_to_batch[i];
                    batches[batch_idx][pos] = Some(file_data);
                }

                let batches: Vec<Vec<FileData>> = batches
                    .into_iter()
                    .map(|batch| batch.into_iter().map(|opt| opt.unwrap()).collect())
                    .collect();

                Ok::<_, String>(batches)
            })
        });

        let batches =
            result.map_err(|e| PyRuntimeError::new_err(format!("batch fetch failed: {}", e)))?;

        Ok(CurvineIterator { batches, pos: 0 })
    }

    #[getter]
    fn dataset_len(&self) -> usize {
        self.dataset.paths.len()
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

    #[getter]
    fn prefetch_size(&self) -> usize {
        self.prefetch_size
    }
}
