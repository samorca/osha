from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any, Dict, List

from src.memory.schemas import ResourceNode, ResourceRelation


def topology_context(
    nodes: Iterable[ResourceNode],
    relations: Iterable[ResourceRelation],
    resource_id: str,
    *,
    max_depth: int = 2,
) -> Dict[str, List[Any]]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    node_map = {node.resource_id: node for node in nodes}
    relation_list = list(relations)
    depth = {resource_id: 0}
    queue = deque([resource_id])

    while queue:
        current = queue.popleft()
        current_depth = depth[current]
        if current_depth >= max_depth:
            continue
        for relation in relation_list:
            if relation.source_id != current:
                continue
            neighbor = relation.target_id
            if neighbor not in depth:
                depth[neighbor] = current_depth + 1
                queue.append(neighbor)

    included_ids = set(depth)
    selected_nodes = [
        node for node_id, node in node_map.items() if node_id in included_ids
    ]
    selected_relations = [
        relation
        for relation in relation_list
        if relation.source_id in included_ids
        and relation.target_id in included_ids
        and depth[relation.source_id] < max_depth
    ]
    return {
        "nodes": selected_nodes,
        "relations": selected_relations,
    }
