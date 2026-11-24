from __future__ import annotations

"""
Generic repository contracts and memory
repository for OSHA persistence.
"""

from abc import (
    ABC,
    abstractmethod,
)

from copy import deepcopy

from threading import RLock

from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    TypeVar,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    environment_id,
    required_string,
)

from src.database.connection import (
    DatabaseConnection,
)

from src.database.errors import (
    RepositoryConflictError,
    RepositoryNotFoundError,
    RepositoryValidationError,
)

from src.database.schema import (
    PersistenceOperation,
    PersistenceResult,
)


# ---------------------------------------------------------------
# Generic repository type
# ---------------------------------------------------------------

T = TypeVar("T")


# ---------------------------------------------------------------
# Repository contract
# ---------------------------------------------------------------


class Repository(
    ABC,
    Generic[T],
):
    """
    Generic persistence contract.

    Domain-specific repositories may build on
    this abstraction later.
    """

    @abstractmethod
    def create(
        self,
        record_id: str,
        value: T,
        *,
        environment: str,
    ) -> PersistenceResult:
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        record_id: str,
        *,
        environment: str,
    ) -> Optional[T]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        record_id: str,
        value: T,
        *,
        environment: str,
        expected_version: Optional[int] = None,
    ) -> PersistenceResult:
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        record_id: str,
        *,
        environment: str,
        expected_version: Optional[int] = None,
    ) -> PersistenceResult:
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        *,
        environment: str,
    ) -> List[T]:
        raise NotImplementedError


# ---------------------------------------------------------------
# Memory repository
# ---------------------------------------------------------------


class MemoryRepository(
    Repository[
        Dict[
            str,
            Any,
        ]
    ]
):
    """
    Environment-isolated repository backed by
    MemoryDatabaseAdapter's process-local store.

    Internal structure:

        store
          ↓
        repositories
          ↓
        repository name
          ↓
        environment ID
          ↓
        record ID
          ↓
        {
            version
            created_at
            updated_at
            value
        }

    This is intended for development/testing,
    not durable production persistence.
    """

    def __init__(
        self,
        *,
        connection: DatabaseConnection,
        name: str,
    ) -> None:
        self.connection = connection

        self.name = required_string(
            name,
            name="repository name",
            maximum_length=256,
        )

        self._lock = RLock()

    # ---------------------------------------------------------
    # Root store
    # ---------------------------------------------------------

    def _store(
        self,
    ) -> Dict[str, Any]:
        handle = self.connection.handle

        if (
            not isinstance(
                handle,
                dict,
            )
            or "store" not in handle
        ):
            raise (
                RepositoryValidationError(
                    "MemoryRepository requires the memory database backend."
                )
            )

        root = handle["store"]

        repositories = root.setdefault(
            "repositories",
            {},
        )

        return repositories.setdefault(
            self.name,
            {},
        )

    # ---------------------------------------------------------
    # Environment-isolated store
    # ---------------------------------------------------------

    def _environment_store(
        self,
        environment: str,
    ) -> Dict[
        str,
        Dict[
            str,
            Any,
        ],
    ]:
        normalized = environment_id(environment)

        store = self._store()

        return store.setdefault(
            normalized,
            {},
        )

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_record_id(
        record_id: str,
    ) -> str:
        return required_string(
            record_id,
            name="record_id",
            maximum_length=512,
        )

    @staticmethod
    def _normalize_value(
        value: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        """
        Store only bounded JSON-safe mappings.

        Domain-specific repositories can later
        explicitly translate dataclasses/domain
        objects to dictionaries before calling
        this layer.
        """

        safe = to_json_safe(dict(value))

        if not isinstance(
            safe,
            dict,
        ):
            raise (
                RepositoryValidationError(
                    "Repository value must serialize to a mapping."
                )
            )

        return safe

    @staticmethod
    def _public_value(
        stored: Dict[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        """
        Return a deep copy so callers cannot
        mutate repository state without using
        update().
        """

        return deepcopy(stored["value"])

    # ---------------------------------------------------------
    # Create
    # ---------------------------------------------------------

    def create(
        self,
        record_id: str,
        value: Dict[
            str,
            Any,
        ],
        *,
        environment: str,
    ) -> PersistenceResult:
        record_id = self._normalize_record_id(record_id)

        normalized_value = self._normalize_value(value)

        with self._lock:
            records = self._environment_store(environment)

            if record_id in records:
                raise (
                    RepositoryConflictError(
                        f"Record already exists in repository {self.name}: {record_id}"
                    )
                )

            now = utc_now()

            records[record_id] = {
                "version": 1,
                "created_at": now,
                "updated_at": now,
                "value": deepcopy(normalized_value),
            }

        return PersistenceResult(
            operation=(PersistenceOperation.CREATE.value),
            success=True,
            repository=self.name,
            record_id=record_id,
            attributes={
                "version": 1,
            },
        ).finalize()

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        record_id: str,
        *,
        environment: str,
    ) -> Optional[
        Dict[
            str,
            Any,
        ]
    ]:
        record_id = self._normalize_record_id(record_id)

        with self._lock:
            stored = self._environment_store(environment).get(record_id)

            if stored is None:
                return None

            return self._public_value(stored)

    # ---------------------------------------------------------
    # Require
    # ---------------------------------------------------------

    def require(
        self,
        record_id: str,
        *,
        environment: str,
    ) -> Dict[str, Any]:
        value = self.get(
            record_id,
            environment=environment,
        )

        if value is None:
            raise (
                RepositoryNotFoundError(
                    f"Record not found in repository {self.name}: {record_id}"
                )
            )

        return value

    # ---------------------------------------------------------
    # Version
    # ---------------------------------------------------------

    def version(
        self,
        record_id: str,
        *,
        environment: str,
    ) -> Optional[int]:
        record_id = self._normalize_record_id(record_id)

        with self._lock:
            stored = self._environment_store(environment).get(record_id)

            if stored is None:
                return None

            return int(stored["version"])

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    def update(
        self,
        record_id: str,
        value: Dict[
            str,
            Any,
        ],
        *,
        environment: str,
        expected_version: Optional[int] = None,
    ) -> PersistenceResult:
        record_id = self._normalize_record_id(record_id)

        normalized_value = self._normalize_value(value)

        with self._lock:
            records = self._environment_store(environment)

            stored = records.get(record_id)

            if stored is None:
                raise (
                    RepositoryNotFoundError(
                        f"Record not found in repository {self.name}: {record_id}"
                    )
                )

            current_version = int(stored["version"])

            if expected_version is not None and current_version != int(
                expected_version
            ):
                raise (RepositoryConflictError("Repository version conflict."))

            new_version = current_version + 1

            stored["version"] = new_version

            stored["updated_at"] = utc_now()

            stored["value"] = deepcopy(normalized_value)

        return PersistenceResult(
            operation=(PersistenceOperation.UPDATE.value),
            success=True,
            repository=self.name,
            record_id=record_id,
            attributes={
                "version": (new_version),
                "previous_version": (current_version),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(
        self,
        record_id: str,
        *,
        environment: str,
        expected_version: Optional[int] = None,
    ) -> PersistenceResult:
        record_id = self._normalize_record_id(record_id)

        with self._lock:
            records = self._environment_store(environment)

            stored = records.get(record_id)

            if stored is None:
                raise (
                    RepositoryNotFoundError(
                        f"Record not found in repository {self.name}: {record_id}"
                    )
                )

            current_version = int(stored["version"])

            if expected_version is not None and current_version != int(
                expected_version
            ):
                raise (RepositoryConflictError("Repository version conflict."))

            del records[record_id]

        return PersistenceResult(
            operation=(PersistenceOperation.DELETE.value),
            success=True,
            repository=self.name,
            record_id=record_id,
            attributes={
                "deleted_version": (current_version),
            },
        ).finalize()

    # ---------------------------------------------------------
    # List environment records
    # ---------------------------------------------------------

    def list(
        self,
        *,
        environment: str,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        with self._lock:
            records = self._environment_store(environment)

            return [
                self._public_value(records[record_id]) for record_id in sorted(records)
            ]

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def metadata(
        self,
        record_id: str,
        *,
        environment: str,
    ) -> Optional[
        Dict[
            str,
            Any,
        ]
    ]:
        record_id = self._normalize_record_id(record_id)

        normalized_environment = environment_id(environment)

        with self._lock:
            stored = self._environment_store(normalized_environment).get(record_id)

            if stored is None:
                return None

            return {
                "record_id": (record_id),
                "repository": (self.name),
                "environment_id": (normalized_environment),
                "version": int(stored["version"]),
                "created_at": (stored["created_at"]),
                "updated_at": (stored["updated_at"]),
            }

    # ---------------------------------------------------------
    # Count
    # ---------------------------------------------------------

    def count(
        self,
        *,
        environment: str,
    ) -> int:
        with self._lock:
            return len(self._environment_store(environment))

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            store = self._store()

            return {
                "repository": (self.name),
                "backend": ("memory"),
                "environment_count": (len(store)),
                "record_count": (sum(len(records) for records in store.values())),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "Repository",
    "MemoryRepository",
]
