"""Tests for model architectures — shape checks and forward-pass sanity."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from latticeprobe.models.gnn import LWEGNN
from latticeprobe.models.transformer import LWETransformer
from latticeprobe.params import get_params
from latticeprobe.representations import to_graph, to_sequence


@pytest.fixture
def params_512():
    return get_params("ML-KEM-512")


@pytest.fixture
def params_768():
    return get_params("ML-KEM-768")


def make_batch_tokens(params, batch_size=4):
    rng = np.random.default_rng(0)
    tokens = []
    for _ in range(batch_size):
        a = rng.integers(0, params.q, size=(params.k, params.n), dtype=np.int64)
        b = rng.integers(0, params.q, size=(params.n,), dtype=np.int64)
        tokens.append(to_sequence(a, b))
    return torch.stack(tokens)   # (B, seq_len)


def make_batch_graphs(params, batch_size=4):
    rng = np.random.default_rng(1)
    graphs = []
    for _ in range(batch_size):
        a = rng.integers(0, params.q, size=(params.k, params.n), dtype=np.int64)
        b = rng.integers(0, params.q, size=(params.n,), dtype=np.int64)
        g = to_graph(a, b, params)
        g.y = torch.tensor([1.0])
        graphs.append(g)
    return Batch.from_data_list(graphs)


# ── Transformer ───────────────────────────────────────────────────────────────

class TestTransformer:
    def test_output_shape_512(self, params_512):
        model = LWETransformer(params_512)
        x = make_batch_tokens(params_512, batch_size=4)
        logits = model(x)
        assert logits.shape == (4, 1)

    def test_output_shape_768(self, params_768):
        model = LWETransformer(params_768)
        x = make_batch_tokens(params_768, batch_size=3)
        logits = model(x)
        assert logits.shape == (3, 1)

    def test_output_dtype(self, params_512):
        model = LWETransformer(params_512)
        x = make_batch_tokens(params_512)
        logits = model(x)
        assert logits.dtype == torch.float32

    def test_no_nan(self, params_512):
        model = LWETransformer(params_512)
        x = make_batch_tokens(params_512)
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_param_count_approx(self, params_512):
        """
        d_model=512, nhead=8, 8 layers, ff=2048. Expected ~27,936,257.
        Paper claims ~51M — PAPER INCONSISTENCY (see transformer.py docstring).
        Accept ±5% of 27,936,257.
        """
        model = LWETransformer(params_512)
        n_params = sum(p.numel() for p in model.parameters())
        expected = 27_936_257
        assert abs(n_params - expected) / expected < 0.05, (
            f"Transformer param count {n_params:,} deviates >5% from expected {expected:,}. "
            "Architecture may have changed. Paper claims ~51M — PAPER INCONSISTENCY."
        )

    def test_gradients_flow(self, params_512):
        model = LWETransformer(params_512)
        x = make_batch_tokens(params_512, batch_size=2)
        labels = torch.tensor([[1.0], [0.0]])
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        grad = model.head.weight.grad
        assert grad is not None and not torch.isnan(grad).any()


# ── GNN ───────────────────────────────────────────────────────────────────────

class TestGNN:
    def test_output_shape_512(self, params_512):
        model = LWEGNN(params_512)
        batch = make_batch_graphs(params_512, batch_size=4)
        logits = model(batch)
        assert logits.shape == (4, 1)

    def test_output_dtype(self, params_512):
        model = LWEGNN(params_512)
        batch = make_batch_graphs(params_512)
        logits = model(batch)
        assert logits.dtype == torch.float32

    def test_no_nan(self, params_512):
        model = LWEGNN(params_512)
        batch = make_batch_graphs(params_512)
        logits = model(batch)
        assert not torch.isnan(logits).any()

    def test_param_count_approx(self, params_512):
        """
        SAGEConv(1→256)×1 + SAGEConv(256→256)×5 + BatchNorm×6 + Linear(256,1).
        Expected ~662,273. Paper claims 18M — PAPER INCONSISTENCY (see gnn.py docstring).
        Accept ±5% of 662,273.
        """
        model = LWEGNN(params_512)
        n_params = sum(p.numel() for p in model.parameters())
        expected = 662_273
        assert abs(n_params - expected) / expected < 0.05, (
            f"GNN param count {n_params:,} deviates >5% from expected {expected:,}. "
            "Architecture may have changed. Paper claims 18M — PAPER INCONSISTENCY."
        )

    def test_gradients_flow(self, params_512):
        model = LWEGNN(params_512)
        batch = make_batch_graphs(params_512, batch_size=2)
        logits = model(batch)
        labels = torch.tensor([[1.0], [0.0]])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        grad = model.head.weight.grad
        assert grad is not None and not torch.isnan(grad).any()

    def test_different_params(self, params_768):
        model = LWEGNN(params_768)
        batch = make_batch_graphs(params_768, batch_size=2)
        logits = model(batch)
        assert logits.shape == (2, 1)
