"""
OSHA Database Repository Package.

Repositories provide the persistence boundary between OSHA's application
services and SQLAlchemy ORM models.

Higher-level components should use these repositories instead of placing
database queries directly inside collectors, APIs, intelligence modules,
forecasting services or remediation workflows.

Repository responsibilities:

    BaseRepository
        Common ORM persistence operations.

    SourceRepository
        Infrastructure source discovery and lifecycle.

    ResourceRepository
        Discovered infrastructure resources.

    ResourceRelationshipRepository
        Infrastructure topology and dependency edges.

    AnomalyRepository
        Detected abnormal behaviour.

    PredictionRepository
        Forecasts and model-generated predictions.

    IncidentRepository
        Operational incident lifecycle and root-cause tracking.

    RemediationRepository
        Remediation proposal, approval, execution-state and rollback
        persistence.

    AuditRepository
        Append-oriented operational and security audit history.

Transaction boundaries remain with the database session/service layer.
Repositories may flush changes but must not independently commit normal
application transactions.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base repository
# ---------------------------------------------------------------------------

from .base import (
    BaseRepository,
    EntityNotFoundError,
    InvalidPaginationError,
    RepositoryError,
)


# ---------------------------------------------------------------------------
# Infrastructure repositories
# ---------------------------------------------------------------------------

from .source import SourceRepository
from .resource import ResourceRepository
from .relationship import ResourceRelationshipRepository


# ---------------------------------------------------------------------------
# Intelligence / operational repositories
# ---------------------------------------------------------------------------

from .anomaly import AnomalyRepository
from .prediction import PredictionRepository
from .incident import IncidentRepository
from .decision import DecisionRepository


# ---------------------------------------------------------------------------
# Remediation / audit repositories
# ---------------------------------------------------------------------------

from .remediation import RemediationRepository
from .audit import AuditRepository


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Base
    "BaseRepository",
    "EntityNotFoundError",
    "InvalidPaginationError",
    "RepositoryError",

    # Infrastructure
    "SourceRepository",
    "ResourceRepository",
    "ResourceRelationshipRepository",

    # Detection / intelligence
    "AnomalyRepository",
    "PredictionRepository",

    # Incidents
    "IncidentRepository",
    "DecisionRepository",

    # Remediation
    "RemediationRepository",

    # Audit
    "AuditRepository",
]
