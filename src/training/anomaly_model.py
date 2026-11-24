from __future__ import annotations

"""
Neural anomaly-classification model for OSHA.

Input:
    fixed-width numeric infrastructure features

Output:
    anomaly logits
    anomaly probability
    latent representation

The model performs binary classification:

    0 = normal
    1 = anomaly

Threshold selection is deliberately separate and
belongs to evaluation / threshold analysis.
"""

from typing import (
    Any,
    Dict,
    Mapping,
)

import torch

from torch import nn


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------


class InfrastructureAnomalyModel(nn.Module):
    """
    Feed-forward anomaly model for structured
    infrastructure features.

    The network separates:

        input projection
        representation encoder
        anomaly classification head

    so the learned representation can also be
    reused by downstream intelligence systems.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        representation_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        if representation_dim < 1:
            raise ValueError("representation_dim must be >= 1")

        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = int(input_dim)

        self.hidden_dim = int(hidden_dim)

        self.representation_dim = int(representation_dim)

        self.dropout = float(dropout)

        # -----------------------------------------------------
        # Input projection
        # -----------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )

        # -----------------------------------------------------
        # Feature encoder
        # -----------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(
                self.hidden_dim,
                self.representation_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.representation_dim),
        )

        # -----------------------------------------------------
        # Binary anomaly head
        # -----------------------------------------------------

        self.anomaly_head = nn.Linear(
            self.representation_dim,
            1,
        )

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    def _validate_features(
        self,
        features: torch.Tensor,
    ) -> None:
        if not isinstance(
            features,
            torch.Tensor,
        ):
            raise TypeError("features must be a torch.Tensor")

        if features.ndim != 2:
            raise ValueError("features must have shape [batch_size, input_dim]")

        if features.shape[1] != self.input_dim:
            raise ValueError(
                "Feature dimension does "
                "not match model input_dim: "
                f"expected {self.input_dim}, "
                f"received "
                f"{features.shape[1]}"
            )

        if not torch.is_floating_point(features):
            raise TypeError("features tensor must use a floating-point dtype")

        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or infinite values")

    # ---------------------------------------------------------
    # Encode
    # ---------------------------------------------------------

    def encode(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_features(features)

        projected = self.input_projection(features)

        return self.encoder(projected)

    # ---------------------------------------------------------
    # Logits
    # ---------------------------------------------------------

    def logits(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(features)

        return self.anomaly_head(representation).squeeze(-1)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[
        str,
        torch.Tensor,
    ]:
        representation = self.encode(features)

        anomaly_logits = self.anomaly_head(representation).squeeze(-1)

        anomaly_probability = torch.sigmoid(anomaly_logits)

        return {
            "representation": (representation),
            "anomaly_logits": (anomaly_logits),
            "anomaly_probability": (anomaly_probability),
        }

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> Dict[
        str,
        torch.Tensor,
    ]:
        """
        Return probability and thresholded binary
        prediction.

        Threshold calibration itself belongs to
        analyze_anomaly_thresholds.py.
        """

        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        was_training = self.training

        self.eval()

        output = self.forward(features)

        prediction = (output["anomaly_probability"] >= threshold).to(dtype=torch.long)

        if was_training:
            self.train()

        return {
            **output,
            "anomaly_prediction": (prediction),
        }

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def model_config(
        self,
    ) -> Dict[str, Any]:
        return {
            "input_dim": (self.input_dim),
            "hidden_dim": (self.hidden_dim),
            "representation_dim": (self.representation_dim),
            "dropout": (self.dropout),
            "output_type": ("binary_logit"),
            "normal_label": 0,
            "anomaly_label": 1,
        }

    # ---------------------------------------------------------
    # Reconstruct from config
    # ---------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Mapping[
            str,
            Any,
        ],
    ) -> "InfrastructureAnomalyModel":
        required = ("input_dim",)

        missing = [name for name in required if name not in config]

        if missing:
            raise ValueError(
                "Anomaly model "
                "configuration is "
                "missing required "
                "fields: " + ", ".join(missing)
            )

        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(
                config.get(
                    "hidden_dim",
                    256,
                )
            ),
            representation_dim=int(
                config.get(
                    "representation_dim",
                    128,
                )
            ),
            dropout=float(
                config.get(
                    "dropout",
                    0.20,
                )
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())

        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return {
            **self.model_config(),
            "parameter_count": (parameter_count),
            "trainable_parameter_count": (trainable_parameter_count),
            "produces_representation": (True),
            "threshold_embedded_in_model": (False),
        }


# ---------------------------------------------------------------
# Model reconstruction helper
# ---------------------------------------------------------------


def build_anomaly_model(
    config: Mapping[
        str,
        Any,
    ],
) -> InfrastructureAnomalyModel:
    return InfrastructureAnomalyModel.from_config(config)


# ---------------------------------------------------------------
# Legacy GRU compatibility
# ---------------------------------------------------------------


class GRUAutoencoder(nn.Module):
    """
    Compatibility adapter for anomaly-evaluation
    checkpoints that predate InfrastructureAnomalyModel.
    """

    def __init__(
        self,
        *,
        number_features: int,
        hidden_size: int = 64,
        latent_size: int = 32,
        number_layers: int = 1,
    ) -> None:
        super().__init__()
        self.number_features = int(number_features)
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.number_layers = int(number_layers)

        self.encoder = nn.GRU(
            input_size=self.number_features,
            hidden_size=self.hidden_size,
            num_layers=self.number_layers,
            batch_first=True,
        )
        self.latent = nn.Linear(
            self.hidden_size,
            self.latent_size,
        )
        self.decoder = nn.GRU(
            input_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.number_layers,
            batch_first=True,
        )
        self.output = nn.Linear(
            self.hidden_size,
            self.number_features,
        )

    def forward(
        self,
        windows: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _, hidden = self.encoder(windows)
        encoded = torch.relu(
            self.latent(hidden[-1])
        )
        repeated = encoded.unsqueeze(1).repeat(
            1,
            windows.shape[1],
            1,
        )
        decoded, _ = self.decoder(repeated)
        reconstruction = self.output(decoded)
        return {
            "reconstruction": reconstruction,
            "latent": encoded,
        }


def reconstruction_scores(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(
        torch.square(original - reconstruction),
        dim=tuple(range(1, original.ndim)),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "InfrastructureAnomalyModel",
    "build_anomaly_model",
    "GRUAutoencoder",
    "reconstruction_scores",
]
