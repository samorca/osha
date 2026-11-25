from __future__ import annotations

import torch
from torch import nn


class RemediationRanker(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
        )
        self.score_head = nn.Linear(
            64,
            1,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.network(features)
        score = torch.sigmoid(
            self.score_head(
                representation,
            ).squeeze(-1)
        )
        return {
            "recommendation_score": score,
            "representation": representation,
        }
