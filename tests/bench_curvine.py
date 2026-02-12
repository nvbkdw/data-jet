"""Benchmark: CurvineDataLoader reading files from curvine distributed cache."""

import time

import numpy as np

from data_jet import CurvineDataLoader

CONFIG_PATH = "/root/workspace/distributed-cache/curvine-cluster-bench.toml"
# FILE_PATHS = [f"/fuse-bench/{i%100}" for i in range(1000)]
FILE_PATHS = [f"/noetik-training-data-609524518243-us-east-2/dataloader_test/platform_dataset/tile_{i}.safetensors" for i in range(1000)]


def bench_curvine_dataloader(batch_size, shuffle, num_epochs=5):
    loader = CurvineDataLoader(
        config_path=CONFIG_PATH,
        file_paths=FILE_PATHS,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    print(f"batch_size={batch_size}, shuffle={shuffle}, files={len(FILE_PATHS)}, epochs={num_epochs}")
    print(f"  batches per epoch: {len(loader)}")

    total_bytes_all = 0
    total_time_all = 0.0

    for epoch in range(num_epochs):
        epoch_bytes = 0
        epoch_files = 0
        t0 = time.perf_counter()

        for batch in loader:
            for arr in batch:
                assert isinstance(arr, np.ndarray)
                assert arr.dtype == np.uint8
                epoch_bytes += arr.nbytes
                epoch_files += 1
            # garbage collect
            del batch

        elapsed = time.perf_counter() - t0
        total_bytes_all += epoch_bytes
        total_time_all += elapsed

        mb = epoch_bytes / (1024 * 1024)
        throughput = mb / elapsed if elapsed > 0 else float("inf")
        print(f"  epoch {epoch}: {epoch_files} files, {mb:.1f} MB in {elapsed:.3f}s ({throughput:.1f} MB/s)")

    total_mb = total_bytes_all / (1024 * 1024)
    avg_throughput = total_mb / total_time_all if total_time_all > 0 else float("inf")
    print(f"  total: {total_mb:.1f} MB in {total_time_all:.3f}s (avg {avg_throughput:.1f} MB/s)")
    print()


if __name__ == "__main__":
    print("=== CurvineDataLoader Benchmark ===\n")

    # bench_curvine_dataloader(batch_size=10, shuffle=False)
    bench_curvine_dataloader(batch_size=40, shuffle=True)
    # bench_curvine_dataloader(batch_size=100, shuffle=False)
