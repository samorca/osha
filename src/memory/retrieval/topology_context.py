from collections.abc import Iterable
from typing import Any


def topology_context(
    nodes: Iterable[Any],
    relations: Iterable[Any],
    resource_id: str,
    max_depth: int = 2,
) -> dict[str, Any]:
    if not isinstance(resource_id, str) or not resource_id:
        raise TypeError("resource_id must be a non-empty string")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool):
        raise TypeError("max_depth must be an integer")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    node_map = {node.resource_id: node for node in nodes}
    relations = list(relations)
    visited = {resource_id}
    frontier = {resource_id}
    related_relations = []

    for _ in range(max_depth):
        next_frontier = set()
        for relation in relations:
            if relation.source_id in frontier:
                neighbor = relation.target_id
                related_relations.append(relation)
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
            if relation.target_id in frontier:
                neighbor = relation.source_id
                related_relations.append(relation)
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return {
        "root_resource_id": resource_id,
        "nodes": [node_map[item] for item in visited if item in node_map],
        "relations": related_relations,
    }