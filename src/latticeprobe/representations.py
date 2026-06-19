"""
Data representations for LWE samples.

Sequence representation: flatten (a, b) coefficients to a 1D token tensor.
Graph representation:    bipartite graph with coefficient edges (torch_geometric).
"""

import numpy as np
import torch
from torch_geometric.data import Data

from latticeprobe.params import LWEParams


def to_sequence(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    """
    Flatten (a, b) into an integer token sequence of length n*(k+1).

    Modular embedding: tokens are integers in [0, q) as per paper §4.2.1.

    Args:
        a: shape (k, n), coefficients in [0, q)
        b: shape (n,),   coefficients in [0, q)
    Returns:
        tokens: int64 tensor of shape (k*n + n,)
    """
    flat_a = a.reshape(-1)          # (k*n,)
    tokens = np.concatenate([flat_a, b])  # (k*n + n,)
    return torch.tensor(tokens, dtype=torch.long)


def to_graph(a: np.ndarray, b: np.ndarray, params: LWEParams) -> Data:
    """
    Build a bipartite graph from one LWE sample (paper §4.2.2).
    """
    k, n_actual = a.shape
    q = params.q

    # Node features
    x_var = torch.tensor(a.reshape(-1) / q, dtype=torch.float).unsqueeze(1)  # (k*n, 1)
    x_eq  = torch.tensor(b / q,             dtype=torch.float).unsqueeze(1)  # (n, 1)
    x = torch.cat([x_var, x_eq], dim=0)                                       # (k*n+n, 1)

    # Edges: for each r in [k], c in [n_actual]: variable (r*n+c) <-> equation (k*n+c)
    r_idx = np.repeat(np.arange(k), n_actual)           # (k*n,)
    c_idx = np.tile(np.arange(n_actual), k)             # (k*n,)
    var_nodes = r_idx * n_actual + c_idx                # (k*n,)
    eq_nodes  = k * n_actual + c_idx                    # (k*n,)

    # Undirected: add both directions
    src = np.concatenate([var_nodes, eq_nodes])
    dst = np.concatenate([eq_nodes,  var_nodes])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Edge attributes: normalised coefficient value (same for both directions)
    weights = a[r_idx, c_idx] / q               # (k*n,)
    edge_attr = torch.tensor(
        np.concatenate([weights, weights]), dtype=torch.float
    ).unsqueeze(1)                               # (2*k*n, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
