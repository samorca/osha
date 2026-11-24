from __future__ import annotations

"""
Ordered, checksum-verified, backend-neutral
migrations for OSHA persistence.
"""

from dataclasses import (
    dataclass,
    field,
)

from hashlib import sha256
from threading import RLock

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from src.core.serialization import (
    canonical_json,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    integer,
    required_string,
)

from src.database.config import (
    MigrationConfig,
)

from src.database.connection import (
    DatabaseConnection,
)

from src.database.errors import (
    MigrationApplyError,
    MigrationChecksumError,
    MigrationConflictError,
)

from src.database.schema import (
    MigrationRecord,
    MigrationState,
)

from src.database.transaction import (
    DatabaseTransaction,
)


# ---------------------------------------------------------------
# Migration callable
# ---------------------------------------------------------------

MigrationCallable = Callable[
    [Any],
    None,
]


# ---------------------------------------------------------------
# Migration definition
# ---------------------------------------------------------------


@dataclass(frozen=True)
class Migration:
    """
    One ordered database migration.

    apply receives the backend handle.

    Backend-specific migrations can therefore
    use their backend's adapter contract without
    exposing that logic to the rest of OSHA.
    """

    migration_id: str

    name: str

    version: int

    apply: MigrationCallable

    checksum: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "Migration":
        required_string(
            self.migration_id,
            name="migration_id",
            maximum_length=256,
        )

        required_string(
            self.name,
            name="migration name",
            maximum_length=256,
        )

        integer(
            self.version,
            name="migration version",
            minimum=1,
        )

        if not callable(self.apply):
            raise ValueError("migration apply must be callable")

        return self

    # ---------------------------------------------------------
    # Checksum
    # ---------------------------------------------------------

    def effective_checksum(
        self,
    ) -> str:
        """
        Return the explicitly supplied checksum,
        or derive a deterministic metadata
        checksum.

        For production SQL/schema migrations,
        prefer supplying a checksum calculated
        from the actual migration artifact.
        """

        self.finalize()

        if self.checksum:
            return required_string(
                self.checksum,
                name=("migration checksum"),
                maximum_length=256,
            )

        payload = canonical_json(
            {
                "migration_id": (self.migration_id),
                "name": self.name,
                "version": (self.version),
                "attributes": (self.attributes),
            }
        )

        return sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------


class MigrationRegistry:
    """
    Ordered registry of known migrations.

    Both version numbers and migration IDs must
    be unique.
    """

    def __init__(
        self,
    ) -> None:
        self._migrations: Dict[
            int,
            Migration,
        ] = {}

        self._ids: Dict[
            str,
            int,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        migration: Migration,
    ) -> Migration:
        migration.finalize()

        with self._lock:
            if migration.version in self._migrations:
                raise (
                    MigrationConflictError(
                        f"Migration version already registered: {migration.version}"
                    )
                )

            if migration.migration_id in self._ids:
                raise (
                    MigrationConflictError(
                        f"Migration ID already registered: {migration.migration_id}"
                    )
                )

            self._migrations[migration.version] = migration

            self._ids[migration.migration_id] = migration.version

        return migration

    # ---------------------------------------------------------
    # Ordered list
    # ---------------------------------------------------------

    def list(
        self,
    ) -> List[Migration]:
        with self._lock:
            return [self._migrations[version] for version in sorted(self._migrations)]

    # ---------------------------------------------------------
    # Up to target version
    # ---------------------------------------------------------

    def up_to(
        self,
        target_version: Optional[int],
    ) -> List[Migration]:
        values = self.list()

        if target_version is None:
            return values

        target_version = integer(
            target_version,
            name="target_version",
            minimum=1,
        )

        return [item for item in values if (item.version <= target_version)]


# ---------------------------------------------------------------
# Applied migration store
# ---------------------------------------------------------------


class MigrationStore:
    """
    Tracks applied migration metadata inside the
    connected backend.

    This initial implementation supports the
    mapping-style memory backend.

    A production PostgreSQL adapter should later
    provide a durable migration-store
    implementation using a dedicated migration
    table.
    """

    ROOT_KEY = "__osha_migrations__"

    def __init__(
        self,
        connection: DatabaseConnection,
    ) -> None:
        self.connection = connection

        self._lock = RLock()

    def _store(
        self,
    ) -> Dict[
        str,
        Dict[
            str,
            Any,
        ],
    ]:
        handle = self.connection.handle

        if not isinstance(
            handle,
            dict,
        ):
            raise MigrationApplyError(
                "Default MigrationStore "
                "currently requires a "
                "mapping-style backend "
                "handle."
            )

        return handle.setdefault(
            self.ROOT_KEY,
            {},
        )

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        migration_id: str,
    ) -> Optional[
        Dict[
            str,
            Any,
        ]
    ]:
        with self._lock:
            value = self._store().get(migration_id)

            return dict(value) if value is not None else None

    # ---------------------------------------------------------
    # Record applied
    # ---------------------------------------------------------

    def record_applied(
        self,
        migration: Migration,
    ) -> MigrationRecord:
        migration.finalize()

        checksum = migration.effective_checksum()

        with self._lock:
            store = self._store()

            if migration.migration_id in store:
                raise (
                    MigrationConflictError(
                        "Migration is "
                        "already recorded "
                        "as applied: "
                        f"{migration.migration_id}"
                    )
                )

            applied_at = utc_now()

            store[migration.migration_id] = {
                "migration_id": (migration.migration_id),
                "name": (migration.name),
                "version": (migration.version),
                "checksum": (checksum),
                "applied_at": (applied_at),
            }

        return MigrationRecord(
            migration_id=(migration.migration_id),
            name=migration.name,
            version=(migration.version),
            state=(MigrationState.APPLIED.value),
            checksum=checksum,
            applied_at=applied_at,
            attributes=dict(migration.attributes),
        ).finalize()

    # ---------------------------------------------------------
    # Applied migrations
    # ---------------------------------------------------------

    def applied(
        self,
    ) -> List[MigrationRecord]:
        with self._lock:
            values = list(self._store().values())

        return [
            MigrationRecord(
                migration_id=(item["migration_id"]),
                name=item["name"],
                version=int(item["version"]),
                state=(MigrationState.APPLIED.value),
                checksum=(item.get("checksum")),
                applied_at=(item.get("applied_at")),
            ).finalize()
            for item in sorted(
                values,
                key=lambda item: int(item["version"]),
            )
        ]


# ---------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------


class MigrationRunner:
    """
    Determine and apply pending migrations.

    Every migration is executed inside its own
    DatabaseTransaction.
    """

    def __init__(
        self,
        *,
        connection: DatabaseConnection,
        registry: Optional[MigrationRegistry] = None,
        config: Optional[MigrationConfig] = None,
        store: Optional[MigrationStore] = None,
    ) -> None:
        self.connection = connection

        self.registry = registry or MigrationRegistry()

        self.config = config or MigrationConfig()

        self.config.validate()

        self.store = store or MigrationStore(connection)

    # ---------------------------------------------------------
    # Pending migrations
    # ---------------------------------------------------------

    def pending(
        self,
    ) -> List[Migration]:
        if not self.config.enabled:
            return []

        pending: List[Migration] = []

        for migration in self.registry.up_to(self.config.target_version):
            applied = self.store.get(migration.migration_id)

            if applied is None:
                pending.append(migration)

                continue

            # -------------------------------------------------
            # Checksum verification
            # -------------------------------------------------

            if self.config.validate_checksums:
                expected = migration.effective_checksum()

                actual = applied.get("checksum")

                if actual != expected:
                    raise (
                        MigrationChecksumError(
                            "Applied migration "
                            "checksum does not "
                            "match registered "
                            "migration.",
                            details={
                                "migration_id": (migration.migration_id),
                                "version": (migration.version),
                            },
                        )
                    )

        return pending

    # ---------------------------------------------------------
    # Apply one migration
    # ---------------------------------------------------------

    def apply_one(
        self,
        migration: Migration,
    ) -> MigrationRecord:
        migration.finalize()

        existing = self.store.get(migration.migration_id)

        # -----------------------------------------------------
        # Already applied
        # -----------------------------------------------------

        if existing is not None:
            if (
                self.config.validate_checksums
                and existing.get("checksum") != migration.effective_checksum()
            ):
                raise (
                    MigrationChecksumError(
                        "Applied migration "
                        "checksum does not "
                        "match registered "
                        "migration.",
                        details={
                            "migration_id": (migration.migration_id),
                        },
                    )
                )

            return MigrationRecord(
                migration_id=(migration.migration_id),
                name=(migration.name),
                version=(migration.version),
                state=(MigrationState.SKIPPED.value),
                checksum=(existing.get("checksum")),
                applied_at=(existing.get("applied_at")),
                attributes={
                    "reason": ("already_applied"),
                },
            ).finalize()

        # -----------------------------------------------------
        # Transactional application
        # -----------------------------------------------------

        try:
            with DatabaseTransaction(connection=(self.connection)):
                migration.apply(self.connection.handle)

                record = self.store.record_applied(migration)

            return record

        except (
            MigrationChecksumError,
            MigrationConflictError,
        ):
            raise

        except Exception as exc:
            raise MigrationApplyError(
                "Database migration failed.",
                details={
                    "migration_id": (migration.migration_id),
                    "version": (migration.version),
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

    # ---------------------------------------------------------
    # Apply pending
    # ---------------------------------------------------------

    def apply_pending(
        self,
    ) -> List[MigrationRecord]:
        if not self.config.enabled:
            return []

        return [self.apply_one(migration) for migration in self.pending()]

    # ---------------------------------------------------------
    # Current version
    # ---------------------------------------------------------

    def current_version(
        self,
    ) -> int:
        applied = self.store.applied()

        return max(
            (item.version for item in applied),
            default=0,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "enabled": (self.config.enabled),
            "apply_on_startup": (self.config.apply_on_startup),
            "validate_checksums": (self.config.validate_checksums),
            "target_version": (self.config.target_version),
            "registered_migrations": (len(self.registry.list())),
            "applied_migrations": (len(self.store.applied())),
            "current_version": (self.current_version()),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "MigrationCallable",
    "Migration",
    "MigrationRegistry",
    "MigrationStore",
    "MigrationRunner",
]
