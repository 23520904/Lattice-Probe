"""
GraphSAGE-based LWE distinguisher (paper §4.3).

Paper spec:
  - GraphSAGE backbone (Hamilton et al., 2017)
  - 6 message-passing layers
  - hidden dim = 256
  - Readout: global mean pool → Linear(256, 1)

  PAPER INCONSISTENCY: paper claims ~18M parameters.
  Actual count for SAGEConv(1→256)×6 + BatchNorm×6 + Linear(256,1): ~662,273.
  The paper's figure is ~27× too large for the stated architecture.

Input:  bipartite graph from representations.to_graph()
          k·n variable nodes  (normalised a-coefficients)
          n   equation nodes  (normalised b-coefficients)
          2·k·n undirected edges (edge_attr stored in graph but NOT used by SAGEConv)
Output: single binary logit per graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from latticeprobe.params import LWEParams


class LWEGNN(nn.Module):
    """
    Args:
        params:      LWEParams — used to document expected graph structure.
        hidden:      hidden dimension per layer (default 256).
        num_layers:  GraphSAGE layers (default 6).
        dropout:     dropout probability between layers (default 0.1).
    """

    def __init__(
        self,
        params: LWEParams,
        hidden: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.params = params
        self.dropout = dropout

        in_channels = 1   # single normalised coefficient feature per node

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden
            # SAGEConv: mean-aggregation GraphSAGE (paper §4.3).
            # edge_attr is present in the graph for inspection but GraphSAGE
            # does not use edge features in its standard formulation.
            self.convs.append(SAGEConv(in_ch, hidden))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.head.bias)

    def count_parameters(self) -> int:
        """Return total trainable parameter count. Expected: ~662,273."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric Batch (batched graphs from DataLoader).
        Returns:
            logits: float tensor of shape (B, 1).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)   # SAGEConv does not take edge_attr
            x = norm(x)
            x = x.relu()
            x = self.drop(x)

        x = global_mean_pool(x, batch)   # (B, hidden)
        return self.head(x)              # (B, 1)
