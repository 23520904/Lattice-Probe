"""Tests for representations.py — sequence and graph construction."""

import numpy as np
import pytest
import torch

from latticeprobe.params import get_params
from latticeprobe.representations import to_graph, to_sequence


@pytest.fixture
def sample_512():
    params = get_params("ML-KEM-512")
    rng = np.random.default_rng(0)
    a = rng.integers(0, params.q, size=(params.k, params.n), dtype=np.int64)
    b = rng.integers(0, params.q, size=(params.n,), dtype=np.int64)
    return a, b, params


@pytest.fixture
def sample_768():
    params = get_params("ML-KEM-768")
    rng = np.random.default_rng(1)
    a = rng.integers(0, params.q, size=(params.k, params.n), dtype=np.int64)
    b = rng.integers(0, params.q, size=(params.n,), dtype=np.int64)
    return a, b, params


# ── Sequence representation ───────────────────────────────────────────────────

def test_sequence_length_512(sample_512):
    a, b, params = sample_512
    tokens = to_sequence(a, b)
    assert tokens.shape == (params.n * (params.k + 1),)  # 256*3 = 768


def test_sequence_length_768(sample_768):
    a, b, params = sample_768
    tokens = to_sequence(a, b)
    assert tokens.shape == (params.n * (params.k + 1),)  # 256*4 = 1024


def test_sequence_dtype(sample_512):
    a, b, params = sample_512
    tokens = to_sequence(a, b)
    assert tokens.dtype == torch.long


def test_sequence_range(sample_512):
    a, b, params = sample_512
    tokens = to_sequence(a, b)
    assert tokens.min() >= 0
    assert tokens.max() < params.q


def test_sequence_content(sample_512):
    """First k*n tokens = flattened a, last n tokens = b."""
    a, b, params = sample_512
    tokens = to_sequence(a, b)
    np.testing.assert_array_equal(tokens[:params.k * params.n].numpy(), a.reshape(-1))
    np.testing.assert_array_equal(tokens[params.k * params.n:].numpy(), b)


# ── Graph representation ──────────────────────────────────────────────────────

def test_graph_node_count(sample_512):
    a, b, params = sample_512
    data = to_graph(a, b, params)
    expected_nodes = params.k * params.n + params.n  # k*n variable + n equation
    assert data.x.shape == (expected_nodes, 1)


def test_graph_edge_count(sample_512):
    a, b, params = sample_512
    data = to_graph(a, b, params)
    # k*n directed edges each way → 2 * k*n undirected edges
    expected_edges = 2 * params.k * params.n
    assert data.edge_index.shape == (2, expected_edges)
    assert data.edge_attr.shape == (expected_edges, 1)


def test_graph_node_features_range(sample_512):
    a, b, params = sample_512
    data = to_graph(a, b, params)
    assert data.x.min() >= 0.0
    assert data.x.max() <= 1.0


def test_graph_edge_index_valid(sample_512):
    a, b, params = sample_512
    data = to_graph(a, b, params)
    n_nodes = params.k * params.n + params.n
    assert data.edge_index.min() >= 0
    assert data.edge_index.max() < n_nodes


def test_graph_undirected(sample_512):
    """Every edge (u,v) should also have (v,u)."""
    a, b, params = sample_512
    data = to_graph(a, b, params)
    ei = data.edge_index.numpy()
    edges = set(zip(ei[0], ei[1]))
    for u, v in list(edges):
        assert (v, u) in edges, f"Edge ({u},{v}) missing reverse"
