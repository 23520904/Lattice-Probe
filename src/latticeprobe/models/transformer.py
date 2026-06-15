"""
Transformer-based LWE distinguisher (paper §4.3).

Paper spec:
  - 8 encoder layers
  - 12 attention heads  [implemented as 8: 512 % 12 ≠ 0; head_dim = 512/8 = 64]
  - hidden dim d_model = 512
  - FFN inner dim = 2048  (4 × d_model, standard ratio)
  - ~27M parameters

Input:  integer token sequence of length k·n + n (coefficients in [0, q)).
Output: single binary logit via CLS-token head.
"""

import torch
import torch.nn as nn

from latticeprobe.params import LWEParams


class LWETransformer(nn.Module):
    """
    Args:
        params:          LWEParams — sets seq_len = n*(k+1) and vocab size q.
        d_model:         embedding / hidden dimension  (paper: 512).
        nhead:           attention heads               (paper: 12 → used 8; 512%8=0).
        num_layers:      encoder stack depth           (paper: 8).
        dim_feedforward: FFN inner dimension           (paper: 2048).
        dropout:         dropout probability           (default 0.1).
    """

    def __init__(
        self,
        params: LWEParams,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.params = params
        seq_len = params.n * (params.k + 1)
        q = params.q

        self.token_embed = nn.Embedding(q, d_model)
        self.pos_embed   = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: int64 tensor of shape (B, seq_len), tokens in [0, q).
        Returns:
            logits: float tensor of shape (B, 1), raw binary logit.
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(positions)   # (B, L, d)
        h = self.encoder(h)                                    # (B, L, d)
        h = self.norm(h[:, 0])                                 # CLS token: (B, d)
        return self.head(h)                                    # (B, 1)
