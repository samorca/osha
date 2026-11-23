from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
    deque,
)
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from src.correlation.registry import (
    ResourceRegistry,
)

from src.telemetry.schema import (
    Relationship,
    ResourceIdentity,
)


def edge_key(
    edge: Relationship,
) -> Tuple[str, str, str]:
    return (
        edge.source_resource_id,
        edge.target_resource_id,
        edge.relationship_type,
    )


@dataclass(frozen=True)
class TopologyPath:
    """
    Shortest discovered topology path between
    two resources.
    """

    resource_ids: List[str]

    relationships: List[str]

    distance: int

    def to_dict(self) -> dict:
        return {
            "resource_ids": list(
                self.resource_ids
            ),
            "relationships": list(
                self.relationships
            ),
            "distance": self.distance,
        }


class TopologyGraph:
    """
    In-memory infrastructure topology graph.

    Nodes:
        normalized resource IDs

    Edges:
        explicit Relationship objects

    The graph itself does not infer causation.
    """

    def __init__(
        self,
        registry: Optional[
            ResourceRegistry
        ] = None,
    ) -> None:
        self.registry = registry

        self._edges: Dict[
            Tuple[str, str, str],
            Relationship,
        ] = {}

        self._outgoing: Dict[
            str,
            Set[
                Tuple[str, str, str]
            ],
        ] = defaultdict(set)

        self._incoming: Dict[
            str,
            Set[
                Tuple[str, str, str]
            ],
        ] = defaultdict(set)

        self._nodes: Set[str] = set()

    # ---------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------

    def add_relationship(
        self,
        relationship: Relationship,
    ) -> bool:
        """
        Add or improve an edge.

        When the same exact relationship appears
        again, retain the edge with the stronger
        confidence value.
        """

        key = edge_key(
            relationship
        )

        existing = self._edges.get(
            key
        )

        if existing is not None:
            old_confidence = (
                existing.confidence
                or 0.0
            )

            new_confidence = (
                relationship.confidence
                or 0.0
            )

            if (
                new_confidence
                <= old_confidence
            ):
                return False

        self._edges[
            key
        ] = relationship

        self._outgoing[
            relationship.
            source_resource_id
        ].add(key)

        self._incoming[
            relationship.
            target_resource_id
        ].add(key)

        self._nodes.add(
            relationship.
            source_resource_id
        )

        self._nodes.add(
            relationship.
            target_resource_id
        )

        return True

    def add_many(
        self,
        relationships: Iterable[
            Relationship
        ],
    ) -> int:
        added = 0

        for relationship in relationships:
            if self.add_relationship(
                relationship
            ):
                added += 1

        return added

    # ---------------------------------------------------------
    # Basic graph access
    # ---------------------------------------------------------

    def nodes(
        self,
    ) -> List[str]:
        nodes = set(
            self._nodes
        )

        if self.registry is not None:
            nodes.update(
                self.registry.resource_ids()
            )

        return sorted(nodes)

    def relationships(
        self,
    ) -> List[Relationship]:
        return list(
            self._edges.values()
        )

    def get_resource(
        self,
        resource_id: str,
    ) -> Optional[
        ResourceIdentity
    ]:
        if self.registry is None:
            return None

        return (
            self.registry.get_resource(
                resource_id
            )
        )

    # ---------------------------------------------------------
    # Edge navigation
    # ---------------------------------------------------------

    def outgoing(
        self,
        resource_id: str,
        relationship_type: Optional[
            str
        ] = None,
    ) -> List[Relationship]:
        edges = [
            self._edges[key]
            for key in (
                self._outgoing.get(
                    resource_id,
                    set(),
                )
            )
            if key in self._edges
        ]

        if (
            relationship_type
            is not None
        ):
            edges = [
                edge
                for edge in edges
                if (
                    edge.relationship_type
                    == relationship_type
                )
            ]

        return edges

    def incoming(
        self,
        resource_id: str,
        relationship_type: Optional[
            str
        ] = None,
    ) -> List[Relationship]:
        edges = [
            self._edges[key]
            for key in (
                self._incoming.get(
                    resource_id,
                    set(),
                )
            )
            if key in self._edges
        ]

        if (
            relationship_type
            is not None
        ):
            edges = [
                edge
                for edge in edges
                if (
                    edge.relationship_type
                    == relationship_type
                )
            ]

        return edges

    def neighbors(
        self,
        resource_id: str,
        *,
        direction: str = "both",
        relationship_types: Optional[
            Set[str]
        ] = None,
    ) -> List[str]:
        """
        Return directly connected resources.

        direction:
            in
            out
            both
        """

        output: Set[str] = set()

        if direction in {
            "out",
            "both",
        }:
            for edge in self.outgoing(
                resource_id
            ):
                if (
                    relationship_types
                    and edge.relationship_type
                    not in relationship_types
                ):
                    continue

                output.add(
                    edge.target_resource_id
                )

        if direction in {
            "in",
            "both",
        }:
            for edge in self.incoming(
                resource_id
            ):
                if (
                    relationship_types
                    and edge.relationship_type
                    not in relationship_types
                ):
                    continue

                output.add(
                    edge.source_resource_id
                )

        if direction not in {
            "in",
            "out",
            "both",
        }:
            raise ValueError(
                "direction must be "
                "'in', 'out', or 'both'"
            )

        return sorted(output)

    def degree(
        self,
        resource_id: str,
    ) -> Dict[str, int]:
        incoming = len(
            self._incoming.get(
                resource_id,
                set(),
            )
        )

        outgoing = len(
            self._outgoing.get(
                resource_id,
                set(),
            )
        )

        return {
            "incoming": incoming,
            "outgoing": outgoing,
            "total": (
                incoming
                + outgoing
            ),
        }

    # ---------------------------------------------------------
    # Path discovery
    # ---------------------------------------------------------

    def shortest_path(
        self,
        source_resource_id: str,
        target_resource_id: str,
        *,
        max_depth: int = 8,
        direction: str = "both",
        relationship_types: Optional[
            Set[str]
        ] = None,
    ) -> Optional[TopologyPath]:
        """
        Breadth-first path search.

        `both` treats resource connectivity as
        navigable in either direction while still
        preserving the actual relationship names.

        It does not imply that the relationship
        itself is semantically bidirectional.
        """

        if (
            source_resource_id
            == target_resource_id
        ):
            return TopologyPath(
                resource_ids=[
                    source_resource_id
                ],
                relationships=[],
                distance=0,
            )

        if max_depth < 1:
            return None

        if direction not in {
            "in",
            "out",
            "both",
        }:
            raise ValueError(
                "direction must be "
                "'in', 'out', or 'both'"
            )

        queue = deque(
            [
                (
                    source_resource_id,
                    [
                        source_resource_id
                    ],
                    [],
                )
            ]
        )

        visited = {
            source_resource_id
        }

        while queue:
            (
                current,
                path_nodes,
                path_edges,
            ) = queue.popleft()

            if (
                len(path_edges)
                >= max_depth
            ):
                continue

            candidates: List[
                Tuple[
                    str,
                    Relationship,
                ]
            ] = []

            if direction in {
                "out",
                "both",
            }:
                candidates.extend(
                    (
                        edge.target_resource_id,
                        edge,
                    )
                    for edge
                    in self.outgoing(
                        current
                    )
                )

            if direction in {
                "in",
                "both",
            }:
                candidates.extend(
                    (
                        edge.source_resource_id,
                        edge,
                    )
                    for edge
                    in self.incoming(
                        current
                    )
                )

            for (
                neighbor,
                edge,
            ) in candidates:
                if (
                             relationship_types
                    and edge.relationship_type
                    not in relationship_types
                ):
                    continue

                if neighbor in visited:
                    continue

                new_nodes = (
                    path_nodes
                    + [neighbor]
                )

                new_edges = (
                    path_edges
                    + [
                        edge.
                        relationship_type
                    ]
                )

                if (
                    neighbor
                    == target_resource_id
                ):
                    return TopologyPath(
                        resource_ids=(
                            new_nodes
                        ),
                        relationships=(
                            new_edges
                        ),
                        distance=len(
                            new_edges
                        ),
                    )

                visited.add(
                    neighbor
                )

                queue.append(
                    (
                        neighbor,
                        new_nodes,
                        new_edges,
                    )
                )

        return None

    def distance(
        self,
        source_resource_id: str,
        target_resource_id: str,
        *,
        max_depth: int = 8,
        direction: str = "both",
    ) -> Optional[int]:
        path = self.shortest_path(
            source_resource_id,
            target_resource_id,
            max_depth=max_depth,
            direction=direction,
        )

        if path is None:
            return None

        return path.distance

    def connected(
        self,
        left_resource_id: str,
        right_resource_id: str,
        *,
        max_depth: int = 8,
    ) -> bool:
        return (
            self.shortest_path(
                left_resource_id,
                right_resource_id,
                max_depth=max_depth,
                direction="both",
            )
            is not None
        )

    # ---------------------------------------------------------
    # Relationship queries
    # ---------------------------------------------------------

    def relationships_between(
        self,
        left_resource_id: str,
        right_resource_id: str,
    ) -> List[Relationship]:
        output = []

        for edge in self.outgoing(
            left_resource_id
        ):
            if (
                edge.target_resource_id
                == right_resource_id
            ):
                output.append(
                    edge
                )

        for edge in self.outgoing(
            right_resource_id
        ):
            if (
                edge.target_resource_id
                == left_resource_id
            ):
                output.append(
                    edge
                )

        return output

    # ---------------------------------------------------------
    # Local topology / blast radius
    # ---------------------------------------------------------

    def subgraph(
        self,
        resource_id: str,
        *,
        depth: int = 2,
        direction: str = "both",
    ) -> "TopologyGraph":
        """
        Return a graph containing resources and
        relationships within N hops.

        This gives later intelligence code a
        bounded topology context around a signal.
        """

        if depth < 0:
            raise ValueError(
                "depth must be >= 0"
            )

        if direction not in {
            "in",
            "out",
            "both",
        }:
            raise ValueError(
                "direction must be "
                "'in', 'out', or 'both'"
            )

        result = TopologyGraph(
            registry=self.registry
        )

        visited = {
            resource_id
        }

        queue = deque(
            [
                (
                    resource_id,
                    0,
                )
            ]
        )

        selected_edges: Set[
            Tuple[str, str, str]
        ] = set()

        while queue:
            (
                current,
                current_depth,
            ) = queue.popleft()

            if (
                current_depth
                >= depth
            ):
                continue

            edges: List[
                Relationship
            ] = []

            if direction in {
                "out",
                "both",
            }:
                edges.extend(
                    self.outgoing(
                        current
                    )
                )

            if direction in {
                "in",
                "both",
            }:
                edges.extend(
                    self.incoming(
                        current
                    )
                )

            for edge in edges:
                selected_edges.add(
                    edge_key(edge)
                )

                if (
                    edge.source_resource_id
                    == current
                ):
                    neighbor = (
                        edge.
                        target_resource_id
                    )
                else:
                    neighbor = (
                        edge.
                        source_resource_id
                    )

                if neighbor not in visited:
                    visited.add(
                        neighbor
                    )

                    queue.append(
                        (
                            neighbor,
                            current_depth
                            + 1,
                        )
                    )

        for key in selected_edges:
            result.add_relationship(
                self._edges[
                    key
                ]
            )

        result._nodes.update(
            visited
        )

        return result

    def affected_resources(
        self,
        resource_id: str,
        *,
        depth: int = 2,
    ) -> List[ResourceIdentity]:
        """
        Return neighboring resources within a
        bounded number of hops.

        "Affected" here means topology-near.
        It does not claim they are actually
        unhealthy.
        """

        graph = self.subgraph(
            resource_id,
            depth=depth,
            direction="both",
        )

        return [
            node
            for node in graph.nodes()
            if node != resource_id
        ]

    # ---------------------------------------------------------
    # Context for correlation/intelligence
    # ---------------------------------------------------------

    def resource_context(
        self,
        resource_id: str,
    ) -> Dict[str, Any]:
        resource = (
            self.get_resource(
                resource_id
            )
        )

        return {
            "resource_id": resource_id,
            "resource": (
                resource.to_dict()
                if resource is not None
                else None
            ),
            "degree": (
                self.degree(
                    resource_id
                )
            ),
            "incoming_relationships": [
                {
                    "source": (
                        edge.
                        source_resource_id
                    ),
                    "type": (
                        edge.
                        relationship_type
                    ),
                    "confidence": (
                        edge.confidence
                    ),
                    "evidence": (
                        edge.evidence
                    ),
                }
                for edge
                in self.incoming(
                    resource_id
                )
            ],
            "outgoing_relationships": [
                {
                    "target": (
                        edge.
                        target_resource_id
                    ),
                    "type": (
                        edge.
                        relationship_type
                    ),
                    "confidence": (
                        edge.confidence
                    ),
                    "evidence": (
                        edge.evidence
                    ),
                }
                for edge
                in self.outgoing(
                    resource_id
                )
            ],
            "neighbors": (
                self.neighbors(
                    resource_id
                )
            ),
        }

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        relationship_types = (
            Counter(
                edge.relationship_type
                for edge
                in self._edges.values()
            )
        )

        providers = Counter(
            edge.provider
            for edge
            in self._edges.values()
        )

        isolated = []

        if self.registry is not None:
            for resource_id in (
                self.registry.
                resource_ids()
            ):
                if (
                    self.degree(
                        resource_id
                    )["total"]
                    == 0
                ):
                    isolated.append(
                        resource_id
                    )

        return {
            "node_count": len(
                self.nodes()
            ),
            "edge_count": len(
                self._edges
            ),
            "relationship_types": (
                dict(
                    relationship_types
                )
            ),
            "providers": (
                dict(providers)
            ),
            "isolated_resource_count": (
                len(isolated)
            ),
            "isolated_resource_ids": (
                sorted(isolated)
            ),
        }

    # ---------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------

    def clear(
        self,
    ) -> None:
        self._edges.clear()

        self._outgoing.clear()

        self._incoming.clear()

        self._nodes.clear()

    def __len__(
        self,
    ) -> int:
        return len(
            self._edges
        )

    def __iter__(
        self,
    ) -> Iterator[
        Relationship
    ]:
        return iter(
            self._edges.values()
        )


def build_topology(
    registry: ResourceRegistry,
    relationships: Iterable[
        Relationship
    ],
) -> TopologyGraph:
    """
    Build a topology graph from an existing
    resource registry and extracted relationships.
    """

    graph = TopologyGraph(
        registry=registry
    )

    graph.add_many(
        relationships
    )

    return graph
