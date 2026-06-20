"""
Transformer-based LWE distinguisher (paper §4.3).

Paper spec:
  - 8 encoder layers
  - hidden dim d_model = 512
  - FFN inner dim = 2048  (4 × d_model, standard ratio)

  PAPER INCONSISTENCY: d_model=512 is not divisible by 12.
  Implementation uses nhead=8. (head_dim = 64, clean integer division.)

  PAPER INCONSISTENCY: paper claims ~51M parameters.
  Actual count for d_model=512, nhead=8, 8 layers, ff=2048: ~27.9M.
  Reaching 51M with d_model=512 would require ff_dim ≈ 5,000 — not stated.

Input:  integer token sequence of length k·n + n (coefficients in [0, q)).
        A learnable CLS token is prepended in forward(), making the encoder
        input length k·n + n + 1.  Classification reads from position 0.
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
        nhead:           attention heads               (paper: 12 → used 8; 512%12≠0).
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

        # Vocabulary embedding: tokens are integers in [0, q)
        self.token_embed = nn.Embedding(q, d_model)
        # Positional embedding covers content positions 0..seq_len-1 (for
        # coeff domain) and 0..2*seq_len-1 (for dual domain).  The CLS token
        # does not receive a positional embedding — its position is implicit.
        self.pos_embed = nn.Embedding(seq_len * 2, d_model)

        # Learnable CLS token prepended to every sequence before encoding.
        # Shape (1, 1, d_model) — broadcast over batch dimension.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

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

    def count_parameters(self) -> int:
        """Return total trainable parameter count. Expected: ~27,936,257."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: int64 tensor of shape (B, seq_len), tokens in [0, q).
        Returns:
            logits: float tensor of shape (B, 1), raw binary logit.
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)   # (1, L)
        content = self.token_embed(x) + self.pos_embed(positions)    # (B, L, d)

        # Prepend CLS token — position 0 in the encoder input is always the
        # dedicated classification token, not a content coefficient.
        cls = self.cls_token.expand(B, -1, -1)                       # (B, 1, d)
        h = torch.cat([cls, content], dim=1)                         # (B, L+1, d)

        h = self.encoder(h)                                           # (B, L+1, d)
        h = self.norm(h[:, 0])                                        # CLS output: (B, d)
        return self.head(h)                                           # (B, 1)
