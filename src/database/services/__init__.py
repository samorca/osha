"""
OSHA Database Services.

Provides higher-level database workflows built on top of OSHA's repository
layer.

Services coordinate multiple repositories while keeping database queries out
of collectors, APIs, intelligence modules and orchestration components.

Service responsibilities:

    DiscoveryService
        Persists infrastructure discovered by Linux, Kubernetes, OpenStack,
        Ceph and other collectors.

    TopologyService
        Provides hierarchy, dependency, relationship and bounded topology
        traversal operations.

    IncidentService
        Coordinates anomalies, predictions, incidents, topology context,
        remediation records and audit history.

Transaction ownership remains with the caller through OSHA's database
session layer. Services must not independently commit normal application
transactions.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Discovery service
# ---------------------------------------------------------------------------

from .discovery import (
    DiscoveryParentNotFoundError,
    DiscoveryResourceNotFoundError,
    DiscoveryResult,
    DiscoveryService,
    DiscoveryServiceError,
    DiscoveryValidationError,
    RelationshipDiscovery,
    ResourceDiscovery,
    SourceDiscovery,
)


# ---------------------------------------------------------------------------
# Topology service
# ---------------------------------------------------------------------------

from .topology import (
    ABSOLUTE_MAX_DEPTH,
    ABSOLUTE_MAX_NODES,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    TopologyNeighbour,
    TopologyResourceNotFoundError,
    TopologyService,
    TopologyServiceError,
    TopologySnapshot,
    TopologyTraversal,
    TopologyTraversalLimitError,
)


# ---------------------------------------------------------------------------
# Incident service
# ---------------------------------------------------------------------------

from .incident import (
    IncidentContext,
    IncidentCreationResult,
    IncidentEvidenceNotFoundError,
    IncidentService,
    IncidentServiceError,
    IncidentStateError,
    IncidentTenantMismatchError,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Discovery
    "DiscoveryParentNotFoundError",
    "DiscoveryResourceNotFoundError",
    "DiscoveryResult",
    "DiscoveryService",
    "DiscoveryServiceError",
    "DiscoveryValidationError",
    "RelationshipDiscovery",
    "ResourceDiscovery",
    "SourceDiscovery",

    # Topology
    "ABSOLUTE_MAX_DEPTH",
    "ABSOLUTE_MAX_NODES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "TopologyNeighbour",
    "TopologyResourceNotFoundError",
    "TopologyService",
    "TopologyServiceError",
    "TopologySnapshot",
    "TopologyTraversal",
    "TopologyTraversalLimitError",

    # Incident
    "IncidentContext",
    "IncidentCreationResult",
    "IncidentEvidenceNotFoundError",
    "IncidentService",
    "IncidentServiceError",
    "IncidentStateError",
    "IncidentTenantMismatchError",
]
