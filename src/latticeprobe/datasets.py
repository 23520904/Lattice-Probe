"""
PyTorch Dataset wrappers for sharded .npz LWE datasets.

Each shard file must contain:
  a:     int16,   shape (shard_size, k, n)   — coefficients in [0, q)
  b:     int16,   shape (shard_size, n)      — coefficients in [0, q)
  label: int8,    shape (shard_size,)        — 1=LWE, 0=uniform
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from latticeprobe.params import LWEParams
from latticeprobe.representations import to_graph, to_sequence


class LWESequenceDataset(Dataset):
    """
    Loads sharded .npz files and returns (token_tensor, label) pairs for the
    transformer model.

    Args:
        shard_dir: directory containing shard_*.npz files.
        params:    LWEParams (provides k, n, q for sequence length).
    """

    def __init__(self, shard_dir: str, params: LWEParams, repr_type: str = "coeff"):
        self.params = params
        self.repr_type = repr_type
        paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not paths:
            raise FileNotFoundError(f"No shard_*.npz files found in {shard_dir}")

        # Load all shards into memory (use memmap-style lazy loading for large datasets
        # by concatenating after load; fine for scale ≤ 2^18 with typical RAM).
        a_parts, b_parts, label_parts = [], [], []
        for p in paths:
            d = np.load(p)
            a_parts.append(d["a"])
            b_parts.append(d["b"])
            label_parts.append(d["label"])

        self._a = np.concatenate(a_parts, axis=0)
        self._b = np.concatenate(b_parts, axis=0)
        self._labels = np.concatenate(label_parts, axis=0)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self._a[idx].copy()
        b = self._b[idx].copy()

        if self.repr_type == "ntt":
            from latticeprobe.ring import ntt
            a = np.stack([ntt(a_row) for a_row in a])
            b = ntt(b)
        elif self.repr_type == "dual":
            from latticeprobe.ring import ntt
            a_ntt = np.stack([ntt(a_row) for a_row in a])
            b_ntt = ntt(b)
            a = np.concatenate([a, a_ntt], axis=-1)
            b = np.concatenate([b, b_ntt], axis=-1)

        tokens = to_sequence(a, b)
        label  = torch.tensor(self._labels[idx], dtype=torch.float)
        return tokens, label


class LWEGraphDataset(Dataset):
    """
    Loads sharded .npz files and returns (torch_geometric Data, label) pairs
    for the GNN model.

    Use with torch_geometric.loader.DataLoader for correct graph batching.
    """

    def __init__(self, shard_dir: str, params: LWEParams, repr_type: str = "coeff"):
        self.params = params
        self.repr_type = repr_type
        paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not paths:
            raise FileNotFoundError(f"No shard_*.npz files found in {shard_dir}")

        a_parts, b_parts, label_parts = [], [], []
        for p in paths:
            d = np.load(p)
            a_parts.append(d["a"])
            b_parts.append(d["b"])
            label_parts.append(d["label"])

        self._a = np.concatenate(a_parts, axis=0)
        self._b = np.concatenate(b_parts, axis=0)
        self._labels = np.concatenate(label_parts, axis=0)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        a = self._a[idx].copy()
        b = self._b[idx].copy()

        if self.repr_type == "ntt":
            from latticeprobe.ring import ntt
            a = np.stack([ntt(a_row) for a_row in a])
            b = ntt(b)
        elif self.repr_type == "dual":
            from latticeprobe.ring import ntt
            a_ntt = np.stack([ntt(a_row) for a_row in a])
            b_ntt = ntt(b)
            a = np.concatenate([a, a_ntt], axis=-1)
            b = np.concatenate([b, b_ntt], axis=-1)

        graph = to_graph(a, b, self.params)
        graph.y = torch.tensor([self._labels[idx]], dtype=torch.float)
        return graph, graph.y


def save_shard(path: str, a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> None:
    """Save one dataset shard to a .npz file."""
    np.savez_compressed(path, a=a.astype(np.int16),
                        b=b.astype(np.int16), label=labels.astype(np.int8))
