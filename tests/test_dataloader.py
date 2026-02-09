"""Tests for data-jet DataLoader."""

import numpy as np
import pytest
import torch

from data_jet import DataLoader


@pytest.fixture
def sample_data():
    np.random.seed(42)
    features = np.random.randn(100, 10)
    labels = np.random.randn(100)
    return features, labels


class TestDataLoader:
    def test_basic_iteration(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=32)

        batches = list(loader)
        assert len(batches) == 4  # ceil(100/32)

        # First 3 batches should be full.
        for x, y in batches[:3]:
            assert x.shape == (32, 10)
            assert y.shape == (32,)

        # Last batch has the remainder.
        x, y = batches[-1]
        assert x.shape == (4, 10)
        assert y.shape == (4,)

    def test_drop_last(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=32, drop_last=True)

        batches = list(loader)
        assert len(batches) == 3  # floor(100/32)
        for x, y in batches:
            assert x.shape == (32, 10)

    def test_shuffle(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=100, shuffle=True)

        # Collect two epochs — shuffled order should differ (with high probability).
        epoch1 = list(loader)[0][1].numpy()
        epoch2 = list(loader)[0][1].numpy()
        # Labels should contain the same values but (likely) in different order.
        np.testing.assert_array_equal(np.sort(epoch1), np.sort(epoch2))

    def test_len(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=32)
        assert len(loader) == 4

    def test_dtype(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=16)
        x, y = next(iter(loader))
        assert x.dtype == torch.float64
        assert y.dtype == torch.float64

    def test_zero_copy(self, sample_data):
        """Verify numpy arrays from Rust share memory with torch tensors."""
        features, labels = sample_data
        from data_jet._core import RustDataLoader

        rust_loader = RustDataLoader(
            features.tolist(), labels.tolist(), batch_size=16
        )
        for np_feat, np_lab in rust_loader:
            t_feat = torch.from_numpy(np_feat)
            t_lab = torch.from_numpy(np_lab)
            # from_numpy shares memory: modifying the tensor modifies the array.
            t_feat[0, 0] = -999.0
            assert np_feat[0, 0] == -999.0
            t_lab[0] = -888.0
            assert np_lab[0] == -888.0
            break  # one batch is enough

    def test_multiple_epochs(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=50)
        for _ in range(3):
            batches = list(loader)
            assert len(batches) == 2

    def test_properties(self, sample_data):
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=16, shuffle=True, drop_last=True)
        assert loader.dataset_len == 100
        assert loader.batch_size == 16
        assert loader.shuffle is True
        assert loader.drop_last is True

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            DataLoader(np.array([[]]), np.array([1.0]), batch_size=0)

        with pytest.raises(ValueError):
            DataLoader(np.random.randn(5, 3), np.random.randn(3), batch_size=2)

    def test_pytorch_training_loop(self, sample_data):
        """Smoke test: use the loader in a minimal PyTorch training loop."""
        features, labels = sample_data
        loader = DataLoader(features, labels, batch_size=32, shuffle=True)

        model = torch.nn.Linear(10, 1, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        for x, y in loader:
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
