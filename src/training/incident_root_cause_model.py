from __future__ import annotations

import torch
from torch import nn


class IncidentRootCauseModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 16,
        severity_count: int = 5,
        root_cause_count: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.severity_head = nn.Linear(
            hidden_dim,
            severity_count,
        )
        self.root_cause_head = nn.Linear(
            hidden_dim,
            root_cause_count,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.encoder(features)
        severity_logits = self.severity_head(representation)
        root_cause_logits = self.root_cause_head(representation)
        return {
            "severity_logits": severity_logits,
            "root_cause_logits": root_cause_logits,
            "representation": representation,
        }
