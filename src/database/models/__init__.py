"""
OSHA Database Models.

This package contains the relational persistence models used by OSHA.

Importing this package registers the core ORM models with SQLAlchemy's
declarative metadata.

Model groups:

    Source
        Infrastructure environments monitored by OSHA.

    Resource
        Infrastructure objects discovered inside those environments.

    ResourceRelationship
        Topology relationships between discovered resources.

    Anomaly
        Abnormal infrastructure behaviour detected by OSHA.

    Prediction
        Forecasts and model-generated assessments.

    Incident
        Operational incidents created from correlated evidence.

    Remediation
        Proposed, approved and executed corrective actions.

    AuditEvent
        Append-oriented operational and security audit records.
"""

from __future__ import annotations

# Infrastructure sources
from .source import (
    Source,
    SourceStatus,
    SourceType,
)

# Infrastructure resources
from .resource import (
    Resource,
    ResourceLifecycleState,
    ResourceStatus,
    ResourceType,
)

# Infrastructure topology
from .relationship import (
    RelationshipDiscoveryMethod,
    RelationshipStatus,
    RelationshipType,
    ResourceRelationship,
)

# Detection
from .anomaly import (
    Anomaly,
    AnomalySeverity,
    AnomalyStatus,
    AnomalyType,
    DetectionMethod,
)

# Prediction / forecasting
from .prediction import (
    Prediction,
    PredictionOutcome,
    PredictionSeverity,
    PredictionStatus,
    PredictionType,
)

# Incident management
from .incident import (
    Incident,
    IncidentPriority,
    IncidentSeverity,
    IncidentSource,
    IncidentStatus,
    IncidentType,
    RootCauseStatus,
)

# Decisions
from .decision import Decision

# Remediation
from .remediation import (
    ApprovalMode,
    ExecutionMode,
    Remediation,
    RemediationOutcome,
    RemediationRisk,
    RemediationStatus,
    RemediationType,
)

# Audit
from .audit import (
    ActorType,
    AuditAction,
    AuditCategory,
    AuditEvent,
    AuditOutcome,
    AuditSeverity,
)


__all__ = [
    # Source
    "Source",
    "SourceStatus",
    "SourceType",

    # Resource
    "Resource",
    "ResourceLifecycleState",
    "ResourceStatus",
    "ResourceType",

    # Relationship
    "RelationshipDiscoveryMethod",
    "RelationshipStatus",
    "RelationshipType",
    "ResourceRelationship",

    # Anomaly
    "Anomaly",
    "AnomalySeverity",
    "AnomalyStatus",
    "AnomalyType",
    "DetectionMethod",

    # Prediction
    "Prediction",
    "PredictionOutcome",
    "PredictionSeverity",
    "PredictionStatus",
    "PredictionType",

    # Incident
    "Incident",
    "IncidentPriority",
    "IncidentSeverity",
    "IncidentSource",
    "IncidentStatus",
    "IncidentType",
    "RootCauseStatus",

    # Decisions
    "Decision",

    # Remediation
    "ApprovalMode",
    "ExecutionMode",
    "Remediation",
    "RemediationOutcome",
    "RemediationRisk",
    "RemediationStatus",
    "RemediationType",

    # Audit
    "ActorType",
    "AuditAction",
    "AuditCategory",
    "AuditEvent",
    "AuditOutcome",
    "AuditSeverity",
]
