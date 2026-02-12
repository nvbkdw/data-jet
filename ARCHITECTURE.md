# ARCHITECTURE.md

Deep-dive into how `ai-dataloader` (the Rust crate in `rust/ai-dataloader/`) is implemented,
and how `data-jet` wraps it.

---

## 1. Crate-Level Overview

```
ai-dataloader/src/
├── lib.rs                          # Crate root: re-exports, global THREAD_POOL
├── collate.rs                      # Collate trait + NoOpCollate
├── collate/
│   ├── default_collate.rs          # DefaultCollate (delegates to sub-modules)
│   ├── default_collate/
│   │   ├── primitive.rs            # Vec<scalar> → Array1  (ndarray)
│   │   ├── tuple.rs                # Vec<(A,B,...)> → (Collate<A>, Collate<B>, ...)
│   │   ├── ndarray.rs              # Vec<Array<D>> → Array<D+1>  (stack along axis 0)
│   │   ├── sequence.rs             # Vec<Vec<T>> → Vec<Collate<T>>  (transpose + collate)
│   │   ├── map.rs                  # Vec<HashMap<K,V>> → HashMap<K, Collate<V>>
│   │   ├── array.rs                # Vec<[T; N]> → collation per position
│   │   ├── string.rs               # Vec<String> → Vec<String>  (no-op)
│   │   ├── reference.rs            # &T delegation
│   │   └── nonzero.rs              # NonZero wrappers
│   └── torch_collate.rs            # TorchCollate → tch::Tensor (optional "tch" feature)
├── indexable.rs                    # Module root for indexable (map-style) DataLoader
├── indexable/
│   ├── dataloader.rs               # DataLoader struct + SingleProcessDataLoaderIter
│   ├── dataloader/builder.rs       # Builder pattern (batch_size, shuffle, num_threads, etc.)
│   ├── dataset.rs                  # Dataset = Len + GetSample (trait composition)
│   ├── dataset/len.rs              # Len trait + impls for std collections
│   ├── dataset/get_sample.rs       # GetSample trait + impls for Vec, VecDeque
│   ├── dataset/ndarray_dataset.rs  # NdarrayDataset<A1,A2,D1,D2>
│   ├── fetch.rs                    # Fetcher trait + MapDatasetFetcher (THE THREADING POINT)
│   └── sampler.rs                  # Sampler trait
│       ├── sequential_sampler.rs   # 0..N in order
│       ├── random_sampler.rs       # shuffled permutation
│       └── batch_sampler.rs        # groups indices into batches
├── iterable.rs                     # Module root for iterable (streaming) DataLoader
└── iterable/
    ├── dataloader.rs               # Iterable DataLoader + IntoIter/Iter
    └── dataloader/builder.rs       # Builder (no threading, no sampler)
```

---

## 2. The Two DataLoader Variants

### 2.1 Indexable DataLoader (Map-Style)

**Location**: `src/indexable/dataloader.rs`

This is the primary DataLoader. It mirrors PyTorch's "map-style" DataLoader: the dataset
supports random access via `get_sample(index)`.

**Core struct** (`DataLoader<D, S, C>`):
- `D: Dataset` — the dataset (must implement `Len + GetSample`)
- `S: Sampler` — index generation strategy (default: `SequentialSampler`)
- `C: Collate<D::Sample>` — how to merge samples into a batch (default: `DefaultCollate`)

**Iteration flow** (per batch):

```
BatchSampler<S>           MapDatasetFetcher<D, C>           Collate<T>
     │                            │                             │
     ├── .next()                  │                             │
     │   returns Vec<usize>       │                             │
     │   (batch of indices)       │                             │
     │                            │                             │
     └───────────────────────────►├── .fetch(indices)           │
                                  │   calls get_sample(idx)     │
                                  │   for each idx              │
                                  │   (parallel if rayon)       │
                                  │                             │
                                  └────────────────────────────►├── .collate(Vec<Sample>)
                                                                │   returns C::Output
                                                                │   (e.g. Array2, Tensor, etc.)
```

**Key type**: `SingleProcessDataLoaderIter` — despite its name, it CAN use multiple threads
internally via rayon (see Section 3). The "single process" refers to there being no
multi-process prefetching queue (unlike PyTorch's `num_workers` which spawns child processes).

### 2.2 Iterable DataLoader (Streaming)

**Location**: `src/iterable/dataloader.rs`

For datasets that implement `IntoIterator` (streams, generators, etc.). No random access needed.

- **No sampler** — items come from the iterator in order
- **No parallel fetching** — items are consumed sequentially from the iterator
- **No rayon support** — entirely single-threaded
- **Shuffle** — only shuffles *within* each batch (not globally), using `rand::seq::SliceRandom`

This variant is simpler: each `next()` call does `.take(batch_size).collect()` then collates.

---

## 3. Threading Model (The Critical Section)

### 3.1 Where Threading Happens

Threading occurs in **exactly one place**: `src/indexable/fetch.rs`, inside `MapDatasetFetcher::fetch()`.

```rust
// With rayon feature enabled:
fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output {
    let data = THREAD_POOL
        .get()
        .expect("thread pool is initialized")
        .install(|| {
            possibly_batched_index
                .into_par_iter()                          // parallel iterator
                .map(|idx| self.dataset.get_sample(idx))  // each sample fetched in parallel
                .collect()
        });
    self.collate_fn.collate(data)  // collation is single-threaded
}

// Without rayon:
fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output {
    let data = possibly_batched_index
        .into_iter()                                      // sequential iterator
        .map(|idx| self.dataset.get_sample(idx))
        .collect();
    self.collate_fn.collate(data)
}
```

**What is parallelized**: The `get_sample()` calls within a single batch. If `batch_size=32`,
then up to 32 `get_sample()` calls execute concurrently across the thread pool.

**What is NOT parallelized**:
- Collation (`collate()`) — always runs on the calling thread after all samples are collected
- Batch-to-batch iteration — strictly sequential; the next batch starts only after the
  current one is fully fetched + collated
- Sampler index generation — single-threaded
- The iterable DataLoader — no parallelism at all

### 3.2 The Thread Pool

**Location**: `src/lib.rs`

```rust
#[cfg(feature = "rayon")]
pub static THREAD_POOL: OnceCell<ThreadPool> = OnceCell::new();
```

- **Global singleton**: One `rayon::ThreadPool` shared across all DataLoader instances in the process
- **Initialized lazily**: Created on the first call to `Builder::build()`
- **Default thread count**: `std::thread::available_parallelism()` (all available CPU cores)
- **Configurable**: `DataLoader::builder(ds).num_threads(4).build()`

**OnceCell semantics**: The `THREAD_POOL` uses `OnceCell`, which means it can only be set
**once**. The builder code attempts to reset it if the thread count changes, but `OnceCell::set()`
silently fails if already initialized. This means:

```rust
// First DataLoader: creates pool with 8 threads
DataLoader::builder(ds1).num_threads(8).build();

// Second DataLoader: tries to set pool to 4 threads — SILENTLY IGNORED
// Still uses the 8-thread pool from the first call
DataLoader::builder(ds2).num_threads(4).build();
```

This is a known limitation in the crate (the builder checks `current_num_threads() != self.num_threads`
and tries to re-set, but `OnceCell` won't allow it).

### 3.3 How Many Threads?

| Scenario | Threads Used |
|----------|-------------|
| `rayon` feature disabled (`default-features = false`) | 1 (calling thread only) |
| `rayon` feature enabled, default | `std::thread::available_parallelism()` (all CPU cores) |
| `rayon` feature enabled, `.num_threads(N)` | N (but only if first pool creation) |
| Iterable DataLoader | Always 1 (no rayon support) |

### 3.4 No Async

The crate uses **no async/await anywhere**. There are:
- No `async fn`
- No `Future` types
- No tokio/async-std/smol dependencies
- No `Poll`/`Waker` implementations

Everything is synchronous. Parallelism is achieved solely through rayon's `par_iter()`,
which is a work-stealing thread pool — not async I/O.

### 3.5 Trait Bounds That Enable Parallelism

The threading design is encoded in the trait bounds:

```rust
impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset + Sync,      // Sync: dataset can be shared across threads safely
    S: Sampler,
    C: Collate<D::Sample>,
    D::Sample: Send,         // Send: samples can be moved across thread boundaries
```

- `D: Sync` — required so multiple threads can call `get_sample()` on the same `&D` reference simultaneously
- `D::Sample: Send` — required so samples produced in worker threads can be collected into the main thread

---

## 4. Sampling Pipeline

### 4.1 Sampler Trait

```rust
pub trait Sampler: Len + IntoIterator<Item = usize> + Copy {
    fn new(data_source_len: usize) -> Self;
}
```

A Sampler is simply an iterator over `usize` indices. It must be `Copy` (constructed from
just the dataset length).

### 4.2 SequentialSampler

Yields `0, 1, 2, ..., N-1`. Implemented as `Range<usize>` internally.

### 4.3 RandomSampler

Creates a shuffled permutation `[5, 2, 9, 0, 7, ...]` using Fisher-Yates via
`rand::seq::SliceRandom::shuffle`. The full permutation is materialized upfront in a `Vec<usize>`.

Note: `replacement: bool` field exists but `replacement = true` is `todo!()` (unimplemented).

### 4.4 BatchSampler

Wraps any `Sampler` and groups its indices into fixed-size batches:

```
Sampler:      [0, 1, 2, 3, 4, 5, 6]
BatchSampler: [[0,1,2], [3,4,5], [6]]     (batch_size=3, drop_last=false)
              [[0,1,2], [3,4,5]]           (batch_size=3, drop_last=true)
```

---

## 5. Collation Pipeline

### 5.1 The Collate Trait

```rust
pub trait Collate<T> {
    type Output;
    fn collate(&self, batch: Vec<T>) -> Self::Output;
}
```

Also implemented for closures: `impl<T, F, O> Collate<T> for F where F: Fn(Vec<T>) -> O`.

### 5.2 DefaultCollate

Uses Rust's trait system to recursively collate nested types at compile time:

| Input | Output | Mechanism |
|-------|--------|-----------|
| `Vec<i32>` | `Array1<i32>` (ndarray) | `primitive.rs` macro |
| `Vec<f64>` | `Array1<f64>` (ndarray) | `primitive.rs` macro |
| `Vec<String>` | `Vec<String>` (no-op) | `string.rs` |
| `Vec<u8>` | `Vec<u8>` (no-op) | `primitive.rs` special case |
| `Vec<(A, B)>` | `(Collate<A>::Output, Collate<B>::Output)` | `tuple.rs` macro (up to 12-tuple) |
| `Vec<Array<D>>` | `Array<D+1>` (stacked along axis 0) | `ndarray.rs` |
| `Vec<Vec<T>>` | `Vec<Collate<T>::Output>` (transpose + collate) | `sequence.rs` |
| `Vec<HashMap<K,V>>` | `HashMap<K, Collate<V>::Output>` | `map.rs` |

### 5.3 NoOpCollate

Returns `Vec<T>` unchanged. Useful for custom post-processing.

### 5.4 TorchCollate (optional `tch` feature)

Collates into `tch::Tensor` for direct GPU usage. Not used in data-jet.

---

## 6. How data-jet Uses ai-dataloader

**Location**: `src/lib.rs`

data-jet uses ai-dataloader with the **rayon feature disabled** (`default-features = false`
in Cargo.toml). Instead of using ai-dataloader's `DataLoader`/`Fetcher` iteration mechanism,
data-jet implements its own **tokio-based async parallel fetch loop** while still relying on
ai-dataloader's sampler infrastructure (`BatchSampler`, `SequentialSampler`, `RandomSampler`)
and `Collate` trait.

### 6.1 Why Tokio Instead of Rayon

ai-dataloader's rayon integration parallelizes `get_sample()` calls via a work-stealing thread
pool. This is effective for CPU-bound work but cannot benefit from async I/O. data-jet replaces
this with tokio's multi-thread runtime so that future dataset implementations (file reads,
network fetches) can use real async I/O. For the current in-memory `TensorDataset`, the async
wrapper is trivial (`async move { sample }`) but the infrastructure is in place.

The GIL is released via `py.allow_threads()` before entering the tokio runtime, so Python
threads are not blocked during Rust computation.

### 6.2 The Tokio Runtime

```rust
static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();
```

- **Global singleton**: One `tokio::runtime::Runtime` shared across all DataLoader instances
- **Initialized lazily**: Created on first use via `OnceLock` (mirrors ai-dataloader's `THREAD_POOL` pattern)
- **Default config**: Multi-thread runtime using all available CPU cores
- **Bridging sync→async**: `get_runtime().block_on(async { ... })` is called from within
  `py.allow_threads()` to enter the async context from synchronous PyO3 code

### 6.3 AsyncGetSample Trait

```rust
trait AsyncGetSample: Send + Sync {
    type Sample: Send + 'static;
    fn get_sample(&self, index: usize) -> impl Future<Output = Self::Sample> + Send;
}
```

This replaces ai-dataloader's synchronous `GetSample` trait. The `Send` bound on the returned
future is required for `tokio::spawn`. Uses RPITIT (return position `impl Trait` in traits,
stable since Rust 1.75).

For `TensorDataset`, the implementation constructs the sample synchronously and wraps it in
`async move { sample }`. Future I/O-bound datasets would do real async work here.

### 6.4 Parallel Batch Fetching

```rust
async fn fetch_batch_async(dataset: Arc<TensorDataset>, indices: Vec<usize>) -> FlatBatch {
    // Spawn one tokio task per sample in the batch
    let handles: Vec<_> = indices.iter().map(|&idx| {
        let ds = Arc::clone(&dataset);
        tokio::spawn(async move { ds.get_sample(idx).await })
    }).collect();

    // Await all tasks, collecting results in order
    let samples: Vec<_> = handles.into_iter()
        .map(|h| h.await.expect("task panicked"))
        .collect();

    FlatCollate.collate(samples)
}
```

Each `get_sample()` call within a batch is dispatched as a separate tokio task. All tasks start
concurrently on the multi-thread runtime, then results are collected in index order. Collation
(`FlatCollate`) runs single-threaded after all samples are gathered — same as the rayon approach.

The dataset is wrapped in `Arc` for shared ownership across spawned tasks.

### 6.5 Custom Collate: FlatCollate

Instead of `DefaultCollate` (which produces `ndarray::Array`), data-jet uses `FlatCollate`:

```rust
struct FlatBatch {
    features: Vec<f64>,   // row-major flat buffer [batch_size * feature_dim]
    nrows: usize,
    ncols: usize,
    labels: Vec<f64>,
}
```

This avoids the ndarray v0.15/v0.16 version conflict (see CLAUDE.md) and keeps the data
as raw `Vec<f64>` for direct zero-copy handoff to NumPy.

### 6.6 The Full Iteration Flow in data-jet

```
Python: for X, y in loader:
         │
         ▼
    RustDataLoader.__iter__(py)
         │
         ├── py.allow_threads()              # release GIL
         ├── get_runtime().block_on(async {   # enter tokio runtime
         │       BatchSampler → Vec<Vec<usize>>     # index generation (ai-dataloader samplers)
         │       for each batch:
         │           tokio::spawn(get_sample(0))     # parallel sample fetch
         │           tokio::spawn(get_sample(1))     # via tokio tasks
         │           tokio::spawn(get_sample(2))     # ...
         │           await all → Vec<Sample>
         │           FlatCollate → FlatBatch
         │   })
         ▼
    DataJetIterator.__next__()              # per-batch, on each Python iteration
         │
         ├── std::mem::take()               # moves Vec out (no clone)
         ├── PyArray1::from_vec()           # zero-copy: Rust Vec → NumPy array
         ├── .reshape([n, m])              # zero-copy: view, no data copy
         └── returns (features, labels) as Python tuple
         │
         ▼
    Python: torch.from_numpy(X)            # zero-copy: NumPy → PyTorch tensor
```

Important: the entire tokio-based iteration runs eagerly inside `__iter__()`.
All batches are pre-computed in Rust (with GIL released), then yielded one-by-one
to Python. The Rust computation is fully complete before the first `__next__()` call.

---

## 7. Comparison with PyTorch's DataLoader

| Feature | PyTorch | ai-dataloader | data-jet |
|---------|---------|---------------|----------|
| Multi-process workers | Yes (`num_workers`) | No | No |
| Thread-based parallelism | No (GIL) | Yes (rayon, per-batch) | Yes (tokio, per-batch) |
| Async I/O support | No | No | Yes (tokio runtime) |
| Async prefetching | Yes (prefetch_factor) | No | No |
| Custom sampler | Yes | Yes (trait-based) | No (fixed to Sequential/Random) |
| Custom collate | Yes | Yes (trait-based) | Yes (FlatCollate hardcoded) |
| Pin memory | Yes | No | No |
| Persistent workers | Yes | N/A | N/A |
| Zero-copy to tensors | No (Python overhead) | N/A (Rust only) | Yes (Rust→NumPy→PyTorch) |
| GIL release during fetch | N/A | N/A | Yes (`py.allow_threads`) |

---

## 8. Summary of Threading Answer

1. **Does it use async?** ai-dataloader itself has zero async code. data-jet adds a tokio
   multi-thread runtime in `src/lib.rs` to dispatch `get_sample()` calls as async tasks.

2. **Does it fetch items in a batch in parallel?** Yes. In data-jet, each `get_sample()` call
   within a batch is spawned as a `tokio::spawn` task on the multi-thread runtime. All tasks
   in a batch start concurrently, then results are collected in order. Collation is always
   single-threaded. Batches are still processed sequentially (one after another).

3. **How many threads?** The tokio runtime defaults to all available CPU cores
   (`std::thread::available_parallelism()`). The runtime is a global singleton created lazily
   via `OnceLock` — shared across all DataLoader instances in the process.

4. **What about ai-dataloader's rayon?** Rayon is disabled (`default-features = false`).
   data-jet bypasses ai-dataloader's `DataLoader`/`Fetcher` iteration entirely, using only
   the sampler infrastructure (`BatchSampler`, `RandomSampler`, `SequentialSampler`) and the
   `Collate` trait. The parallel fetching is implemented directly in data-jet via tokio.

5. **GIL handling?** The Python GIL is released via `py.allow_threads()` before entering
   `get_runtime().block_on()`, so Python threads are not blocked during Rust/tokio computation.
