from __future__ import annotations

import json
import os
import tempfile
from abc import (
    ABC,
    abstractmethod,
)
from copy import deepcopy
from dataclasses import (
    dataclass,
    fields,
)
from pathlib import Path
from threading import RLock
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from src.memory.schema import (
    BaselineMemory,
    EpisodeMemory,
    IncidentMemory,
    MemoryObject,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    OutcomeMemory,
    PatternMemory,
    RootCauseMemory,
    VerificationStatus,
    memory_is_active,
    utc_now,
)


TMemory = TypeVar(
    "TMemory",
    bound=MemoryRecord,
)


MEMORY_CLASSES: Dict[
    str,
    Type[MemoryRecord],
] = {
    MemoryType.EPISODE.value: (
        EpisodeMemory
    ),
    MemoryType.PATTERN.value: (
        PatternMemory
    ),
    MemoryType.BASELINE.value: (
        BaselineMemory
    ),
    MemoryType.INCIDENT.value: (
        IncidentMemory
    ),
    MemoryType.ROOT_CAUSE.value: (
        RootCauseMemory
    ),
    MemoryType.OUTCOME.value: (
        OutcomeMemory
    ),
}


class MemoryStoreError(
    RuntimeError
):
    """
    Base error for memory storage failures.
    """


class MemoryNotFoundError(
    MemoryStoreError
):
    """
    Raised when a required memory record does not
    exist.
    """


class MemoryConflictError(
    MemoryStoreError
):
    """
    Raised when a create-only write conflicts
    with an existing memory record.
    """


class MemorySerializationError(
    MemoryStoreError
):
    """
    Raised when memory cannot be serialized or
    reconstructed.
    """


@dataclass(frozen=True)
class MemoryQuery:
    """
    Provider-independent memory query.

    All populated fields are combined using AND.

    `resource_ids`, `source_ids`, and `tags`
    require at least one matching value by
    default.
    """

    memory_ids: tuple[str, ...] = ()

    memory_types: tuple[str, ...] = ()

    environment_ids: tuple[str, ...] = ()

    statuses: tuple[str, ...] = ()

    verification_statuses: tuple[
        str,
        ...,
    ] = ()

    resource_ids: tuple[str, ...] = ()

    source_ids: tuple[str, ...] = ()

    tags: tuple[str, ...] = ()

    active_only: bool = False

    updated_after: Optional[str] = None

    updated_before: Optional[str] = None

    limit: Optional[int] = None

    offset: int = 0

    newest_first: bool = True

    def validate(
        self,
    ) -> None:
        if self.offset < 0:
            raise ValueError(
                "offset must be >= 0"
            )

        if (
            self.limit is not None
            and self.limit < 1
        ):
            raise ValueError(
                "limit must be >= 1"
            )


@dataclass(frozen=True)
class MemoryStoreStats:
    memory_count: int

    active_count: int

    archived_count: int

    expired_count: int

    verified_count: int

    environments: int

    memory_types: Dict[str, int]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_count": (
                self.memory_count
            ),
            "active_count": (
                self.active_count
            ),
            "archived_count": (
                self.archived_count
            ),
            "expired_count": (
                self.expired_count
            ),
            "verified_count": (
                self.verified_count
            ),
            "environments": (
                self.environments
            ),
            "memory_types": dict(
                self.memory_types
            ),
        }


# ---------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------

def memory_to_dict(
    memory: MemoryObject,
) -> Dict[str, Any]:
    """
    Convert one typed memory object into a
    persistence-safe dictionary.
    """

    memory.finalize()

    result = memory.to_dict()

    if not isinstance(
        result,
        dict,
    ):
        raise MemorySerializationError(
            "Memory serialization did not "
            "produce a dictionary"
        )

    return result


def memory_class_for_type(
    memory_type: str,
) -> Type[MemoryRecord]:
    memory_class = (
        MEMORY_CLASSES.get(
            memory_type
        )
    )

    if memory_class is None:
        raise MemorySerializationError(
            "Unsupported memory_type: "
            f"{memory_type}"
        )

    return memory_class


def memory_from_dict(
    value: Mapping[str, Any],
) -> MemoryObject:
    """
    Reconstruct a typed memory object.

    Only dataclass fields accepted by the target
    constructor are passed to it. Derived fields
    such as subclass `memory_type` are therefore
    not supplied when they use init=False.
    """

    raw_type = value.get(
        "memory_type"
    )

    if not isinstance(
        raw_type,
        str,
    ) or not raw_type.strip():
        raise MemorySerializationError(
            "Persisted memory has no "
            "memory_type"
        )

    memory_class = (
        memory_class_for_type(
            raw_type.strip()
        )
    )

    accepted_fields = {
        item.name
        for item in fields(
            memory_class
        )
        if item.init
    }

    constructor_values = {
        key: deepcopy(item)
        for key, item
        in value.items()
        if key in accepted_fields
    }

    try:
        memory = memory_class(
            **constructor_values
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise MemorySerializationError(
            "Unable to reconstruct "
            f"{raw_type} memory: {exc}"
        ) from exc

    memory.finalize()

    return memory  # type: ignore[return-value]


def clone_memory(
    memory: TMemory,
) -> TMemory:
    """
    Return an isolated typed copy.

    Stores should not expose mutable internal
    objects directly to callers.
    """

    cloned = memory_from_dict(
        memory_to_dict(memory)
    )

    return cloned  # type: ignore[return-value]


# ---------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------

def parse_timestamp(
    value: Optional[str],
) -> Optional[Any]:
    if not value:
        return None

    from datetime import (
        datetime,
        timezone,
    )

    text = value.strip()

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            text
        )

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def intersects(
    left: Sequence[str],
    right: Sequence[str],
) -> bool:
    if not right:
        return True

    return bool(
        set(left).intersection(
            right
        )
    )


def memory_matches(
    memory: MemoryObject,
    query: MemoryQuery,
) -> bool:
    """
    Evaluate one memory object against a query.
    """

    if (
        query.memory_ids
        and memory.memory_id
        not in query.memory_ids
    ):
        return False

    if (
        query.memory_types
        and memory.memory_type
        not in query.memory_types
    ):
        return False

    if (
        query.environment_ids
        and memory.environment_id
        not in query.environment_ids
    ):
        return False

    if (
        query.statuses
        and memory.status
        not in query.statuses
    ):
        return False

    if (
        query.verification_statuses
        and memory.verification_status
        not in query.
        verification_statuses
    ):
        return False

    if (
        query.active_only
        and not memory_is_active(
            memory
        )
    ):
        return False

    if not intersects(
        memory.resource_ids,
        query.resource_ids,
    ):
        return False

    if not intersects(
        memory.source_ids,
        query.source_ids,
    ):
        return False

    if not intersects(
        memory.tags,
        query.tags,
    ):
        return False

    updated = parse_timestamp(
        memory.updated_at
    )

    after = parse_timestamp(
        query.updated_after
    )

    before = parse_timestamp(
        query.updated_before
    )

    if after is not None:
        if (
            updated is None
            or updated < after
        ):
            return False

    if before is not None:
        if (
            updated is None
            or updated > before
        ):
            return False

    return True


def apply_query(
    memories: Iterable[
        MemoryObject
    ],
    query: MemoryQuery,
) -> List[MemoryObject]:
    query.validate()

    output = [
        memory
        for memory in memories
        if memory_matches(
            memory,
            query,
        )
    ]

    output.sort(
        key=lambda item: (
            parse_timestamp(
                item.updated_at
            )
            or parse_timestamp(
                item.created_at
            )
        ),
        reverse=query.newest_first,
    )

    start = query.offset

    if query.limit is None:
        return output[start:]

    return output[
        start:
        start + query.limit
    ]


# ---------------------------------------------------------------
# Abstract storage interface
# ---------------------------------------------------------------

class MemoryStore(ABC):
    """
    Persistence-independent OSHA memory store.

    Other memory modules should depend on this
    interface rather than a specific database.
    """

    @abstractmethod
    def put(
        self,
        memory: MemoryObject,
        *,
        create_only: bool = False,
    ) -> MemoryObject:
        """
        Create or replace one memory record.
        """

    @abstractmethod
    def get(
        self,
        memory_id: str,
    ) -> Optional[MemoryObject]:
        """
        Return one memory object or None.
        """

    @abstractmethod
    def query(
        self,
        query: Optional[
            MemoryQuery
        ] = None,
    ) -> List[MemoryObject]:
        """
        Return memory records matching a query.
        """

    @abstractmethod
    def delete(
        self,
        memory_id: str,
    ) -> bool:
        """
        Permanently delete one memory record.
        """

    @abstractmethod
    def count(
        self,
        query: Optional[
            MemoryQuery
        ] = None,
    ) -> int:
        """
        Count matching memory records.
        """

    @abstractmethod
    def clear(
        self,
    ) -> None:
        """
        Remove every memory record.
        """

    def require(
        self,
        memory_id: str,
    ) -> MemoryObject:
        memory = self.get(
            memory_id
        )

        if memory is None:
            raise MemoryNotFoundError(
                "Memory was not found: "
                f"{memory_id}"
            )

        return memory

    def put_many(
        self,
        memories: Iterable[
            MemoryObject
        ],
        *,
        create_only: bool = False,
    ) -> List[MemoryObject]:
        return [
            self.put(
                memory,
                create_only=(
                    create_only
                ),
            )
            for memory in memories
        ]

    def delete_many(
        self,
        memory_ids: Iterable[str],
    ) -> int:
        deleted = 0

        for memory_id in memory_ids:
            if self.delete(memory_id):
                deleted += 1

        return deleted

    def archive(
        self,
        memory_id: str,
    ) -> MemoryObject:
        memory = self.require(
            memory_id
        )

        memory.status = (
            MemoryStatus.ARCHIVED.value
        )

        memory.updated_at = utc_now()

        return self.put(memory)

    def expire(
        self,
        memory_id: str,
    ) -> MemoryObject:
        memory = self.require(
            memory_id
        )

        memory.status = (
            MemoryStatus.EXPIRED.value
        )

        memory.updated_at = utc_now()

        return self.put(memory)

    def verify(
        self,
        memory_id: str,
        *,
        status: str = (
            VerificationStatus.
            VERIFIED.value
        ),
    ) -> MemoryObject:
        allowed = {
            item.value
            for item
            in VerificationStatus
        }

        if status not in allowed:
            raise ValueError(
                "Unsupported verification "
                f"status: {status}"
            )

        memory = self.require(
            memory_id
        )

        memory.verification_status = (
            status
        )

        memory.updated_at = utc_now()

        return self.put(memory)

    def stats(
        self,
    ) -> MemoryStoreStats:
        memories = self.query(
            MemoryQuery()
        )

        type_counts: Dict[
            str,
            int,
        ] = {}

        for memory in memories:
            type_counts[
                memory.memory_type
            ] = (
                type_counts.get(
                    memory.memory_type,
                    0,
                )
                + 1
            )

        return MemoryStoreStats(
            memory_count=len(
                memories
            ),
            active_count=sum(
                1
                for memory in memories
                if memory.status
                == MemoryStatus.
                ACTIVE.value
            ),
            archived_count=sum(
                1
                for memory in memories
                if memory.status
                == MemoryStatus.
                ARCHIVED.value
            ),
            expired_count=sum(
                1
                for memory in memories
                if memory.status
                == MemoryStatus.
                EXPIRED.value
            ),
            verified_count=sum(
                1
                for memory in memories
                if (
                    memory.
                    verification_status
                    == VerificationStatus.
                    VERIFIED.value
                )
            ),
            environments=len(
                {
                    memory.environment_id
                    for memory in memories
                }
            ),
            memory_types=type_counts,
        )


# ---------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------

class InMemoryMemoryStore(
    MemoryStore
):
    """
    Thread-safe development/test store.

    This store is not durable across process
    restarts.
    """

    def __init__(
        self,
        memories: Iterable[
            MemoryObject
        ] = (),
    ) -> None:
        self._lock = RLock()

        self._memories: Dict[
            str,
            MemoryObject,
        ] = {}

        self.put_many(memories)

    def put(
        self,
        memory: MemoryObject,
        *,
        create_only: bool = False,
    ) -> MemoryObject:
        memory.finalize()

        if not memory.memory_id:
            raise MemoryStoreError(
                "Memory has no memory_id "
                "after finalize()"
            )

        with self._lock:
            if (
                create_only
                and memory.memory_id
                in self._memories
            ):
                raise MemoryConflictError(
                    "Memory already exists: "
                    f"{memory.memory_id}"
                )

            stored = clone_memory(
                memory
            )

            stored.updated_at = utc_now()

            self._memories[
                stored.memory_id
            ] = stored

            return clone_memory(
                stored
            )

    def get(
        self,
        memory_id: str,
    ) -> Optional[MemoryObject]:
        with self._lock:
            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return None

            memory.mark_accessed()

            self._memories[
                memory_id
            ] = memory

            return clone_memory(
                memory
            )

    def query(
        self,
        query: Optional[
            MemoryQuery
        ] = None,
    ) -> List[MemoryObject]:
        active_query = (
            query
            or MemoryQuery()
        )

        with self._lock:
            memories = apply_query(
                self._memories.values(),
                active_query,
            )

            return [
                clone_memory(memory)
                for memory in memories
            ]

    def delete(
        self,
        memory_id: str,
    ) -> bool:
        with self._lock:
            return (
                self._memories.pop(
                    memory_id,
                    None,
                )
                is not None
            )

    def count(
        self,
        query: Optional[
            MemoryQuery
        ] = None,
    ) -> int:
        return len(
            self.query(query)
        )

    def clear(
        self,
    ) -> None:
        with self._lock:
            self._memories.clear()

    def __len__(
        self,
    ) -> int:
        with self._lock:
            return len(
                self._memories
            )

    def __iter__(
        self,
    ) -> Iterator[MemoryObject]:
        return iter(
            self.query()
        )


# ---------------------------------------------------------------
# Local JSON snapshot implementation
# ---------------------------------------------------------------

class JSONFileMemoryStore(
    InMemoryMemoryStore
):
    """
    Local durable store backed by one JSON file.

    The complete active snapshot is rewritten
    atomically after each mutation.

    This is suitable for local development and
    small OSHA environments. It is not intended
    as the final high-volume production backend.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        create_parent: bool = True,
    ) -> None:
        self.path = Path(path)

        self.create_parent = (
            create_parent
        )

        super().__init__()

        self.load()

    def load(
        self,
    ) -> int:
        with self._lock:
            if not self.path.exists():
                return 0

            try:
                with self.path.open(
                    "r",
                    encoding="utf-8",
                ) as source:
                    payload = json.load(
                        source
                    )

            except (
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise MemoryStoreError(
                    "Unable to load memory "
                    f"store {self.path}: {exc}"
                ) from exc

            if isinstance(
                payload,
                Mapping,
            ):
                raw_memories = payload.get(
                    "memories",
                    [],
                )

            elif isinstance(
                payload,
                list,
            ):
                # Backward-compatible simple
                # list representation.
                raw_memories = payload

            else:
                raise MemorySerializationError(
                    "Memory store root must "
                    "be an object or list"
                )

            if not isinstance(
                raw_memories,
                list,
            ):
                raise MemorySerializationError(
                    "'memories' must be a list"
                )

            loaded: Dict[
                str,
                MemoryObject,
            ] = {}

            for index, raw_memory in enumerate(
                raw_memories
            ):
                if not isinstance(
                    raw_memory,
                    Mapping,
                ):
                    raise (
                        MemorySerializationError(
                            "Memory entry "
                            f"{index} is not "
                            "an object"
                        )
                    )

                memory = memory_from_dict(
                    raw_memory
                )

                if not memory.memory_id:
                    continue

                loaded[
                    memory.memory_id
                ] = memory

            self._memories = loaded

            return len(loaded)

    def flush(
        self,
    ) -> None:
        """
        Atomically rewrite the JSON snapshot.

        A temporary file is written in the same
        directory and then replaced over the
        destination.
        """

        with self._lock:
            if self.create_parent:
                self.path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            payload = {
                "schema_version": "1",
                "updated_at": utc_now(),
                "memories": [
                    memory_to_dict(
                        memory
                    )
                    for memory in (
                        self._memories.values()
                    )
                ],
            }

            temporary_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=str(
                        self.path.parent
                    ),
                    prefix=(
                        f".{self.path.name}."
                    ),
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(
                        temporary.name
                    )

                    json.dump(
                        payload,
                        temporary,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )

                    temporary.write("\n")

                    temporary.flush()

                    os.fsync(
                        temporary.fileno()
                    )

                os.replace(
                    temporary_path,
                    self.path,
                )

            except OSError as exc:
                if (
                    temporary_path
                    is not None
                    and temporary_path.exists()
                ):
                    try:
                        temporary_path.unlink()
                    except OSError:
                        pass

                raise MemoryStoreError(
                    "Unable to persist memory "
                    f"store {self.path}: {exc}"
                ) from exc

    def put(
        self,
        memory: MemoryObject,
        *,
        create_only: bool = False,
    ) -> MemoryObject:
        with self._lock:
            stored = super().put(
                memory,
                create_only=(
                    create_only
                ),
            )

            self.flush()

            return stored

    def get(
        self,
        memory_id: str,
    ) -> Optional[MemoryObject]:
        with self._lock:
            memory = super().get(
                memory_id
            )

            if memory is not None:
                # Persist last_accessed_at.
                self.flush()

            return memory

    def delete(
        self,
        memory_id: str,
    ) -> bool:
        with self._lock:
            deleted = super().delete(
                memory_id
            )

            if deleted:
                self.flush()

            return deleted

    def clear(
        self,
    ) -> None:
        with self._lock:
            super().clear()

            self.flush()


def create_memory_store(
    *,
    backend: str = "memory",
    path: Optional[
        Path | str
    ] = None,
) -> MemoryStore:
    """
    Convenience factory used by memory.pipeline.

    Supported development backends:

      memory
      json
    """

    normalized = (
        backend.strip().lower()
    )

    if normalized == "memory":
        return InMemoryMemoryStore()

    if normalized == "json":
        if path is None:
            raise ValueError(
                "path is required for "
                "the JSON memory backend"
            )

        return JSONFileMemoryStore(
            path
        )

    raise ValueError(
        "Unsupported memory backend: "
        f"{backend}"
    )
