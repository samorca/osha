import torch
from torch import nn


class InfrastructureEventModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_sources,
        num_components,
        num_events,
        embedding_dim=192,
        hidden_dim=256,
        dropout=0.20,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.source_head = nn.Linear(
            hidden_dim,
            num_sources,
        )

        self.component_head = nn.Linear(
            hidden_dim,
            num_components,
        )

        self.event_head = nn.Linear(
            hidden_dim,
            num_events,
        )

    def encode(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)

        mask = attention_mask.unsqueeze(-1)

        summed = (embedded * mask).sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1.0)

        pooled = summed / lengths

        representation = self.encoder(pooled)

        return representation

    def forward(self, input_ids, attention_mask):
        representation = self.encode(
            input_ids,
            attention_mask,
        )

        return {
            "representation": representation,
            "source_logits": self.source_head(
                representation
            ),
            "component_logits": self.component_head(
                representation
            ),
            "event_logits": self.event_head(
                representation
            ),
        }
