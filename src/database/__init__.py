"""
OSHA Database Package.

Provides the public interface to OSHA's relational persistence subsystem.

The package contains:

    - ORM declarative foundation
    - environment-driven configuration
    - SQLAlchemy engine management
    - connection health checks
    - ORM session and transaction management
    - infrastructure and operational models

Application code should prefer importing database facilities from this
package rather than depending unnecessarily on internal implementation
modules.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# ORM foundation
# ---------------------------------------------------------------------------

from .base import (
    Base,
    ExternalIdentityMixin,
    TenantMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    VersionMixin,
    generate_uuid,
    metadata,
    utc_now,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from .config import (
    DatabaseConfig,
    get_database_config,
    load_database_config,
    reset_database_config,
)


# ---------------------------------------------------------------------------
# Connection / Engine
# ---------------------------------------------------------------------------

from .connection import (
    DatabaseConnectionError,
    check_database_connection,
    create_database_engine,
    dispose_engine,
    get_engine,
    get_pool_status,
    require_database_connection,
    reset_engine,
)


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

from .engine import (
    DatabaseEngineManager,
    database_diagnostics,
    database_engine_manager,
    database_healthcheck,
    start_database,
    stop_database,
)


# ---------------------------------------------------------------------------
# Session / transactions
# ---------------------------------------------------------------------------

from .session import (
    DatabaseSessionError,
    commit_session,
    create_session,
    create_session_factory,
    flush_session,
    get_session,
    get_session_factory,
    reset_session_factory,
    rollback_session,
    session_context,
    session_scope,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from .models import (
    ActorType,
    Anomaly,
    AnomalySeverity,
    AnomalyStatus,
    AnomalyType,
    ApprovalMode,
    AuditAction,
    AuditCategory,
    AuditEvent,
    AuditOutcome,
    AuditSeverity,
    DetectionMethod,
    Decision,
    ExecutionMode,
    Incident,
    IncidentPriority,
    IncidentSeverity,
    IncidentSource,
    IncidentStatus,
    IncidentType,
    Prediction,
    PredictionOutcome,
    PredictionSeverity,
    PredictionStatus,
    PredictionType,
    RelationshipDiscoveryMethod,
    RelationshipStatus,
    RelationshipType,
    Remediation,
    RemediationOutcome,
    RemediationRisk,
    RemediationStatus,
    RemediationType,
    Resource,
    ResourceLifecycleState,
    ResourceRelationship,
    ResourceStatus,
    ResourceType,
    RootCauseStatus,
    Source,
    SourceStatus,
    SourceType,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # ORM foundation
    "Base",
    "ExternalIdentityMixin",
    "TenantMixin",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "VersionMixin",
    "generate_uuid",
    "metadata",
    "utc_now",

    # Configuration
    "DatabaseConfig",
    "get_database_config",
    "load_database_config",
    "reset_database_config",

    # Connection
    "DatabaseConnectionError",
    "check_database_connection",
    "create_database_engine",
    "dispose_engine",
    "get_engine",
    "get_pool_status",
    "require_database_connection",
    "reset_engine",

    # Engine lifecycle
    "DatabaseEngineManager",
    "database_diagnostics",
    "database_engine_manager",
    "database_healthcheck",
    "start_database",
    "stop_database",

    # Session
    "DatabaseSessionError",
    "commit_session",
    "create_session",
    "create_session_factory",
    "flush_session",
    "get_session",
    "get_session_factory",
    "reset_session_factory",
    "rollback_session",
    "session_context",
    "session_scope",

    # Source
    "Source",
    "SourceStatus",
    "SourceType",

    # Resource
    "Resource",
    "ResourceLifecycleState",
    "ResourceStatus",
    "ResourceType",

    # Relationships
    "ResourceRelationship",
    "RelationshipDiscoveryMethod",
    "RelationshipStatus",
    "RelationshipType",

    # Anomaly
    "Anomaly",
    "AnomalySeverity",
    "AnomalyStatus",
    "AnomalyType",
    "DetectionMethod",
    "Decision",

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
