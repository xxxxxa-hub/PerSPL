"""PerSPL reward model: low-rank personalized reward head.

Reward for user u on embedding x:

    r_u(x) = softmax(W[u])^T (V x) / T

where V (D, K) is a shared low-rank basis and W (M, K) is a per-user
weight matrix with row-wise softmax. ``V_ref`` is a frozen reference V
(set externally) used by ``cosine_reg_loss`` to anchor V during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerSPLRewardHead(nn.Module):
    """Personalized low-rank reward head."""

    def __init__(
        self,
        emb_dim: int,
        num_users: int,
        rank: int = 8,
        temperature: float = 100.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_users = num_users
        self.rank = rank
        self.temperature = temperature

        self.V = nn.Parameter(torch.randn(emb_dim, rank))
        self.W = nn.Parameter(torch.randn(num_users, rank))

        self.register_buffer("V_ref", torch.zeros(emb_dim, rank))

    def forward(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        projected = x @ self.V
        user_weights = F.softmax(self.W[user_ids], dim=-1)
        return (projected * user_weights).sum(dim=-1) / self.temperature

    def score_single(self, emb: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(emb, user_ids)

    def bt_loss(self, x_diff: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x_diff, user_ids)
        return -F.logsigmoid(logits).mean()

    def cosine_reg_loss(self) -> torch.Tensor:
        V_norm = F.normalize(self.V, dim=0)
        V_ref_norm = F.normalize(self.V_ref, dim=0)
        cos_sim = (V_norm * V_ref_norm).sum(dim=0)
        return (1.0 - cos_sim).mean()
