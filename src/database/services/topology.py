"""
OSHA Infrastructure Topology Service.

Provides bounded topology traversal and infrastructure relationship queries.

The service sits above:

    ResourceRepository
    ResourceRelationshipRepository

and provides higher-level operations such as:

    - direct neighbours
    - dependencies
    - dependants
    - incoming topology
    - outgoing topology
    - bounded multi-hop traversal
    - resource hierarchy
    - topology snapshots
    - impact neighbourhoods

The service deliberately prevents unlimited graph traversal.

OSHA may monitor very large infrastructure environments, so every traversal
must have explicit depth and node limits.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy.orm import Session

from ..models.relationship import (
    RelationshipType,
    ResourceRelationship,
)
from ..models.resource import Resource
from ..repositories.relationship import (
    ResourceRelationshipRepository,
)
from ..repositories.resource import ResourceRepository


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_NODES = 500

ABSOLUTE_MAX_DEPTH = 10
ABSOLUTE_MAX_NODES = 5000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TopologyServiceError(RuntimeError):
    """Base error raised by OSHA topology operations."""


class TopologyResourceNotFoundError(TopologyServiceError):
    """Raised when the requested topology root does not exist."""


class TopologyTraversalLimitError(TopologyServiceError):
    """Raised when invalid topology traversal limits are requested."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TopologyNeighbour:
    """
    One resource directly connected to another resource.
    """

    resource: Resource
    relationship: ResourceRelationship
    direction: str


@dataclass(frozen=True, slots=True)
class TopologyTraversal:
    """
    Result of a bounded topology traversal.
    """

    root: Resource
    resources: tuple[Resource, ...]
    relationships: tuple[ResourceRelationship, ...]

    max_depth: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    """
    Infrastructure topology representation around a resource.
    """

    root: Resource

    children: tuple[Resource, ...]

    incoming: tuple[ResourceRelationship, ...]
    outgoing: tuple[ResourceRelationship, ...]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TopologyService:
    """
    Coordinate topology queries across OSHA infrastructure resources.

    The service is bound to one SQLAlchemy Session.

    It does not commit transactions.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:
        if session is None:
            raise ValueError(
                "TopologyService requires an active Session"
            )

        self.session = session

        self.resources = ResourceRepository(
            session
        )

        self.relationships = ResourceRelationshipRepository(
            session
        )

    # ------------------------------------------------------------------
    # Resource resolution
    # ------------------------------------------------------------------

    def get_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
    ) -> Resource:
        """
        Return a tenant-owned topology resource.
        """
        resource = self.resources.get(
            resource_id
        )

        if (
            resource is None
            or resource.tenant_id != tenant_id
        ):
            raise TopologyResourceNotFoundError(
                f"resource {resource_id} was not found "
                "for the specified tenant"
            )

        return resource

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def children(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        resource_type: str | None = None,
        limit: int = 100,
    ) -> tuple[Resource, ...]:
        """
        Return direct hierarchical children of a resource.
        """
        self.get_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        return tuple(
            self.resources.list_children(
                tenant_id=tenant_id,
                parent_id=resource_id,
                resource_type=resource_type,
                offset=0,
                limit=limit,
                include_deleted=False,
            )
        )

    # ------------------------------------------------------------------
    # Incoming / outgoing edges
    # ------------------------------------------------------------------

    def outgoing(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        relationship_type: str | None = None,
        limit: int = 100,
    ) -> tuple[ResourceRelationship, ...]:
        """Return topology edges originating from a resource."""
        self.get_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        return tuple(
            self.relationships.list_outgoing(
                tenant_id=tenant_id,
                resource_id=resource_id,
                relationship_type=relationship_type,
                offset=0,
                limit=limit,
                include_inactive=False,
            )
        )

    def incoming(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        relationship_type: str | None = None,
        limit: int = 100,
    ) -> tuple[ResourceRelationship, ...]:
        """Return topology edges targeting a resource."""
        self.get_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        return tuple(
            self.relationships.list_incoming(
                tenant_id=tenant_id,
                resource_id=resource_id,
                relationship_type=relationship_type,
                offset=0,
                limit=limit,
                include_inactive=False,
            )
        )

    # ------------------------------------------------------------------
    # Direct neighbours
    # ------------------------------------------------------------------

    def neighbours(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        relationship_type: str | None = None,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return one-hop resources connected to resource_id.

        Direction is relative to the requested root resource:

            outgoing
            incoming
        """
        root = self.get_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        edges = self.relationships.list_connected(
            tenant_id=tenant_id,
            resource_id=root.id,
            relationship_type=relationship_type,
            offset=0,
            limit=limit,
        )

        neighbours: list[TopologyNeighbour] = []

        seen: set[
            tuple[uuid.UUID, uuid.UUID]
        ] = set()

        for edge in edges:
            if edge.source_resource_id == root.id:
                neighbour = edge.target_resource
                direction = "outgoing"
            else:
                neighbour = edge.source_resource
                direction = "incoming"

            if neighbour is None:
                continue

            key = (
                edge.id,
                neighbour.id,
            )

            if key in seen:
                continue

            seen.add(key)

            neighbours.append(
                TopologyNeighbour(
                    resource=neighbour,
                    relationship=edge,
                    direction=direction,
                )
            )

        return tuple(neighbours)

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    def dependencies(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return direct dependencies of a resource.

        The canonical topology direction is:

            resource
                DEPENDS_ON
            dependency
        """
        neighbours = self.neighbours(
            tenant_id=tenant_id,
            resource_id=resource_id,
            relationship_type=RelationshipType.DEPENDS_ON.value,
            limit=limit,
        )

        return tuple(
            neighbour
            for neighbour in neighbours
            if neighbour.direction == "outgoing"
        )

    def dependants(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return resources that directly depend on the resource.

        Example:

            application
                DEPENDS_ON
            database

        Querying dependants of database returns application.
        """
        neighbours = self.neighbours(
            tenant_id=tenant_id,
            resource_id=resource_id,
            relationship_type=RelationshipType.DEPENDS_ON.value,
            limit=limit,
        )

        return tuple(
            neighbour
            for neighbour in neighbours
            if neighbour.direction == "incoming"
        )

    # ------------------------------------------------------------------
    # Runtime placement
    # ------------------------------------------------------------------

    def runs_on(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return resources on which the requested resource runs.
        """
        neighbours = self.neighbours(
            tenant_id=tenant_id,
            resource_id=resource_id,
            relationship_type=RelationshipType.RUNS_ON.value,
            limit=limit,
        )

        return tuple(
            neighbour
            for neighbour in neighbours
            if neighbour.direction == "outgoing"
        )

    def workloads_on(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return resources that RUN_ON the requested resource.
        """
        neighbours = self.neighbours(
            tenant_id=tenant_id,
            resource_id=resource_id,
            relationship_type=RelationshipType.RUNS_ON.value,
            limit=limit,
        )

        return tuple(
            neighbour
            for neighbour in neighbours
            if neighbour.direction == "incoming"
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def mounts(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return storage resources mounted by the requested resource.
        """
        neighbours = self.neighbours(
            tenant_id=tenant_id,
            resource_id=resource_id,
            relationship_type=RelationshipType.MOUNTS.value,
            limit=limit,
        )

        return tuple(
            neighbour
            for neighbour in neighbours
            if neighbour.direction == "outgoing"
        )

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    def communications(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 100,
    ) -> tuple[TopologyNeighbour, ...]:
        """
        Return resources directly connected through COMMUNICATES_WITH.
        """
        return self.neighbours(
            tenant_id=tenant_id,
            resource_id=resource_id,
            relationship_type=(
                RelationshipType.COMMUNICATES_WITH.value
            ),
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        limit: int = 250,
    ) -> TopologySnapshot:
        """
        Build a one-hop topology snapshot around a resource.
        """
        root = self.get_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        children = self.resources.list_children(
            tenant_id=tenant_id,
            parent_id=root.id,
            offset=0,
            limit=limit,
            include_deleted=False,
        )

        outgoing = self.relationships.list_outgoing(
            tenant_id=tenant_id,
            resource_id=root.id,
            offset=0,
            limit=limit,
            include_inactive=False,
        )

        incoming = self.relationships.list_incoming(
            tenant_id=tenant_id,
            resource_id=root.id,
            offset=0,
            limit=limit,
            include_inactive=False,
        )

        return TopologySnapshot(
            root=root,
            children=tuple(children),
            incoming=tuple(incoming),
            outgoing=tuple(outgoing),
        )

    # ------------------------------------------------------------------
    # Bounded traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        relationship_types: Iterable[str] | None = None,
        direction: str = "both",
    ) -> TopologyTraversal:
        """
        Traverse the topology graph with strict bounds.

        Parameters:
            max_depth:
                Maximum number of edges traversed from the root.

            max_nodes:
                Maximum number of resources returned.

            relationship_types:
                Optional allowed relationship types.

            direction:
                "outgoing"
                "incoming"
                "both"

        This performs breadth-first traversal.

        Unlimited traversal is deliberately unsupported.
        """
        self._validate_traversal(
            max_depth=max_depth,
            max_nodes=max_nodes,
            direction=direction,
        )

        root = self.get_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        allowed_types = (
            set(relationship_types)
            if relationship_types is not None
            else None
        )

        queue: deque[
            tuple[uuid.UUID, int]
        ] = deque(
            [(root.id, 0)]
        )

        visited_resources: set[uuid.UUID] = {
            root.id
        }

        seen_relationships: set[uuid.UUID] = set()

        discovered_resources: dict[
            uuid.UUID,
            Resource,
        ] = {
            root.id: root
        }

        discovered_relationships: dict[
            uuid.UUID,
            ResourceRelationship,
        ] = {}

        truncated = False

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            edges = self._edges_for_traversal(
                tenant_id=tenant_id,
                resource_id=current_id,
                direction=direction,
            )

            for edge in edges:
                if (
                    allowed_types is not None
                    and edge.relationship_type not in allowed_types
                ):
                    continue

                if edge.id not in seen_relationships:
                    seen_relationships.add(
                        edge.id
                    )

                    discovered_relationships[
                        edge.id
                    ] = edge

                neighbour = self._opposite_resource(
                    edge=edge,
                    resource_id=current_id,
                )

                if neighbour is None:
                    continue

                if neighbour.id in visited_resources:
                    continue

                if len(visited_resources) >= max_nodes:
                    truncated = True
                    continue

                visited_resources.add(
                    neighbour.id
                )

                discovered_resources[
                    neighbour.id
                ] = neighbour

                queue.append(
                    (
                        neighbour.id,
                        depth + 1,
                    )
                )

        return TopologyTraversal(
            root=root,
            resources=tuple(
                discovered_resources.values()
            ),
            relationships=tuple(
                discovered_relationships.values()
            ),
            max_depth=max_depth,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Impact neighbourhood
    # ------------------------------------------------------------------

    def impact_neighbourhood(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        max_depth: int = 2,
        max_nodes: int = 250,
    ) -> TopologyTraversal:
        """
        Return a bounded topology neighbourhood useful for impact analysis.

        The method does not claim that every connected resource is affected.
        It provides topology context that higher intelligence/correlation
        layers can evaluate.
        """
        return self.traverse(
            tenant_id=tenant_id,
            resource_id=resource_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            direction="both",
        )

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def _edges_for_traversal(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        direction: str,
    ) -> tuple[ResourceRelationship, ...]:
        """Load edges required for a traversal step."""
        if direction == "outgoing":
            return tuple(
                self.relationships.list_outgoing(
                    tenant_id=tenant_id,
                    resource_id=resource_id,
                    offset=0,
                    limit=1000,
                    include_inactive=False,
                )
            )

        if direction == "incoming":
            return tuple(
                self.relationships.list_incoming(
                    tenant_id=tenant_id,
                    resource_id=resource_id,
                    offset=0,
                    limit=1000,
                    include_inactive=False,
                )
            )

        return tuple(
            self.relationships.list_connected(
                tenant_id=tenant_id,
                resource_id=resource_id,
                offset=0,
                limit=1000,
            )
        )

    @staticmethod
    def _opposite_resource(
        *,
        edge: ResourceRelationship,
        resource_id: uuid.UUID,
    ) -> Resource | None:
        """
        Return the opposite endpoint of an edge relative to resource_id.
        """
        if edge.source_resource_id == resource_id:
            return edge.target_resource

        if edge.target_resource_id == resource_id:
            return edge.source_resource

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_traversal(
        *,
        max_depth: int,
        max_nodes: int,
        direction: str,
    ) -> None:
        """Validate safe topology traversal bounds."""
        if max_depth < 0:
            raise TopologyTraversalLimitError(
                "max_depth cannot be negative"
            )

        if max_depth > ABSOLUTE_MAX_DEPTH:
            raise TopologyTraversalLimitError(
                f"max_depth cannot exceed {ABSOLUTE_MAX_DEPTH}"
            )

        if max_nodes < 1:
            raise TopologyTraversalLimitError(
                "max_nodes must be greater than 0"
            )

        if max_nodes > ABSOLUTE_MAX_NODES:
            raise TopologyTraversalLimitError(
                f"max_nodes cannot exceed {ABSOLUTE_MAX_NODES}"
            )

        if direction not in {
            "outgoing",
            "incoming",
            "both",
        }:
            raise TopologyTraversalLimitError(
                "direction must be 'outgoing', 'incoming', or 'both'"
            )


__all__ = [
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
]
