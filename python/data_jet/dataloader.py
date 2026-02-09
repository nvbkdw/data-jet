"""PyTorch-compatible DataLoader wrapping the Rust implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from data_jet._core import RustDataLoader

if TYPE_CHECKING:
    from collections.abc import Iterator


class DataLoader:
    """Drop-in replacement for ``torch.utils.data.DataLoader``.

    Data is loaded and batched in Rust. Each batch is returned as a pair of
    ``torch.Tensor`` objects created from numpy arrays via **zero-copy**
    (``torch.from_numpy`` shares memory with the underlying numpy buffer,
    which itself was produced by zero-copy ownership transfer from Rust).

    Args:
        features: 2-D array-like of shape ``(n_samples, n_features)``.
        labels: 1-D array-like of shape ``(n_samples,)``.
        batch_size: Number of samples per batch.
        shuffle: Re-shuffle data every epoch.
        drop_last: Drop the last incomplete batch.
        device: Target ``torch.device`` (default ``"cpu"``). If a CUDA device
            is specified the tensors are moved after zero-copy creation.
    """

    def __init__(
        self,
        features: np.ndarray | list,
        labels: np.ndarray | list,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D, got {features.ndim}-D"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be 1-D, got {labels.ndim}-D"
            )

        self._loader = RustDataLoader(
            features.tolist(),
            labels.tolist(),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self._device = torch.device(device)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        device = self._device
        for np_features, np_labels in self._loader:
            # torch.from_numpy shares memory — zero-copy.
            x = torch.from_numpy(np_features)
            y = torch.from_numpy(np_labels)
            if device.type != "cpu":
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            yield x, y

    @property
    def dataset_len(self) -> int:
        """Total number of samples."""
        return self._loader.dataset_len

    @property
    def batch_size(self) -> int:
        return self._loader.batch_size

    @property
    def shuffle(self) -> bool:
        return self._loader.shuffle

    @property
    def drop_last(self) -> bool:
        return self._loader.drop_last
