use bytes::Bytes;
use curvine_client::file::{CurvineFileSystem, FsReader};
use curvine_common::conf::ClusterConf;
use curvine_common::fs::{Path as CvPath, Reader};
use curvine_common::state::FileBlocks;
use curvine_common::FsResult;
use numpy::ndarray::ArrayView1;
use numpy::PyArray1;
use orpc::runtime::{RpcRuntime, Runtime as OrpcRuntime};
use orpc::sys::DataSlice;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;

use orpc::common::{LogConf, Logger};

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
    /// Original string paths — used as cache keys for block location dedup.
    path_strings: Vec<String>,
}

impl CurvineDataset {
    /// Pre-fetch block locations for all dataset paths, deduplicating by
    /// path string so repeated files only incur one metadata RPC.
    async fn prefetch_block_locations(&self) -> Result<Vec<FileBlocks>, String> {
        let mut cache: HashMap<&str, FileBlocks> = HashMap::new();
        let mut result = Vec::with_capacity(self.paths.len());

        for (i, path) in self.paths.iter().enumerate() {
            let key = self.path_strings[i].as_str();
            let blocks = if let Some(cached) = cache.get(key) {
                cached.clone()
            } else {
                let blocks = self
                    .fs
                    .get_block_locations(path)
                    .await
                    .map_err(|e| format!("failed to get block locations for {}: {}", key, e))?;
                cache.insert(key, blocks.clone());
                blocks
            };
            result.push(blocks);
        }

        Ok(result)
    }

    /// Read one file using pre-fetched block locations.
    /// Skips complete() — worker cleans up stale state via connection
    /// lifecycle, saving one RPC round-trip per file.
    async fn read_file(
        &self,
        index: usize,
        file_blocks: FileBlocks,
        _permit: OwnedSemaphorePermit,
    ) -> FsResult<FileData> {
        let path = &self.paths[index];
        let mut reader = FsReader::new(path.clone(), self.fs.fs_context(), file_blocks)?;

        // First chunk
        let first = if reader.has_remaining() {
            let chunk = reader.async_read(None).await?;
            if chunk.is_empty() {
                return Ok(FileData::Copied(Vec::new()));
            }
            dataslice_into_bytes(chunk)
        } else {
            return Ok(FileData::Copied(Vec::new()));
        };

        // If the file was fully consumed in one chunk, return zero-copy.
        if !reader.has_remaining() {
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
    #[pyo3(signature = (config_path, file_paths, batch_size=1, shuffle=false, drop_last=false, prefetch_size=0, log_level=None, io_threads=0, worker_threads=0))]
    fn new(
        config_path: &str,
        file_paths: Vec<String>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        prefetch_size: usize,
        log_level: Option<&str>,
        io_threads: usize,
        worker_threads: usize,
    ) -> PyResult<Self> {
        if file_paths.is_empty() {
            return Err(PyValueError::new_err("file_paths must not be empty"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }

        // Initialize tracing logger (once; subsequent calls are no-ops).
        if let Some(level) = log_level {
            Logger::init(LogConf {
                level: level.to_uppercase(),
                ..LogConf::default()
            });
        }

        // 0 means "use default": 10 * batch_size
        let prefetch_size = if prefetch_size == 0 {
            10 * batch_size
        } else {
            prefetch_size
        };

        let conf = ClusterConf::from(config_path)
            .map_err(|e| PyValueError::new_err(format!("failed to load config: {}", e)))?;

        let rt = Arc::new(if io_threads == 0 && worker_threads == 0 {
            OrpcRuntime::default("data-jet-curvine")
        } else {
            let default_threads = 2 * std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(2);
            let io = if io_threads == 0 { 32 } else { io_threads };
            let workers = if worker_threads == 0 { default_threads.max(4) } else { worker_threads };
            OrpcRuntime::new("data-jet-curvine", io, workers)
        });

        let fs = CurvineFileSystem::with_rt(conf, Arc::clone(&rt))
            .map_err(|e| PyIOError::new_err(format!("failed to create filesystem: {}", e)))?;

        let path_strings = file_paths.clone();

        let paths: Vec<CvPath> = file_paths
            .into_iter()
            .map(|p| CvPath::new(&p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyValueError::new_err(format!("invalid path: {}", e)))?;

        let dataset = Arc::new(CurvineDataset {
            fs,
            paths,
            path_strings,
        });

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
                // Pre-fetch block locations, deduplicating by path string.
                // For repeated files (e.g. i%100) this turns 1000 metadata
                // RPCs into 100.
                let all_blocks = dataset.prefetch_block_locations().await?;
                let all_blocks = Arc::new(all_blocks);

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
                        let blocks = all_blocks[file_idx].clone();
                        let permit = Arc::clone(&sem).acquire_owned().await.unwrap();
                        handles.push(rt.spawn(async move {
                            ds.read_file(file_idx, blocks, permit)
                                .await
                                .map_err(|e| format!("failed to read file {}: {}", file_idx, e))
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
