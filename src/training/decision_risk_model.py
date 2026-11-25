from __future__ import annotations

import torch
from torch import nn


class DecisionRiskModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 16,
        risk_count: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(
            hidden_dim,
            risk_count,
        )
        self.policy_probability_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.network(features)
        policy_probability = torch.sigmoid(
            self.policy_probability_head(
                representation,
            ).squeeze(-1)
        )
        return {
            "risk_logits": self.risk_head(representation),
            "policy_probability": policy_probability,
            "representation": representation,
        }
