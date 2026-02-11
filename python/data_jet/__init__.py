"""data-jet: PyTorch-compatible DataLoader powered by Rust."""

from data_jet._core import RustDataLoader
from data_jet.dataloader import DataLoader

__all__ = ["DataLoader", "RustDataLoader"]

try:
    from data_jet._core import CurvineDataLoader

    __all__ += ["CurvineDataLoader"]
except ImportError:
    pass
