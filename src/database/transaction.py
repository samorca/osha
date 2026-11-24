from __future__ import annotations

"""
Backend-neutral transaction lifecycle management
for OSHA persistence.
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
    Optional,
)

from src.core.time import (
    utc_now,
)

from src.database.connection import (
    DatabaseConnection,
)

from src.database.errors import (
    TransactionCommitError,
    TransactionRollbackError,
    TransactionStateError,
    wrap_driver_error,
)

from src.database.schema import (
    TransactionRecord,
    TransactionState,
)


# ---------------------------------------------------------------
# Transaction adapter
# ---------------------------------------------------------------


class TransactionAdapter(ABC):
    """
    Backend-specific transaction operations
    behind a stable OSHA contract.

    Future implementations can wrap PostgreSQL,
    SQLite, or another backend's native
    transaction API.
    """

    @abstractmethod
    def begin(
        self,
        handle: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def commit(
        self,
        handle: Any,
        transaction_handle: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def rollback(
        self,
        handle: Any,
        transaction_handle: Any,
    ) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------
# Memory transaction adapter
# ---------------------------------------------------------------


class MemoryTransactionAdapter(TransactionAdapter):
    """
    Snapshot-based transactions for the
    development memory backend.

    begin:
        copy the process-local store

    commit:
        discard snapshot

    rollback:
        restore snapshot

    This is useful for development semantics but
    is not equivalent to a production database's
    isolation/concurrency guarantees.
    """

    def begin(
        self,
        handle: Dict[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        if not bool(
            handle.get(
                "open",
                False,
            )
        ):
            raise RuntimeError("Memory database handle is closed.")

        return {
            "snapshot": deepcopy(
                handle.get(
                    "store",
                    {},
                )
            ),
            "active": True,
        }

    def commit(
        self,
        handle: Dict[
            str,
            Any,
        ],
        transaction_handle: Dict[
            str,
            Any,
        ],
    ) -> None:
        if not transaction_handle.get(
            "active",
            False,
        ):
            raise RuntimeError("Memory transaction is not active.")

        transaction_handle["active"] = False

        transaction_handle.pop(
            "snapshot",
            None,
        )

    def rollback(
        self,
        handle: Dict[
            str,
            Any,
        ],
        transaction_handle: Dict[
            str,
            Any,
        ],
    ) -> None:
        if not transaction_handle.get(
            "active",
            False,
        ):
            raise RuntimeError("Memory transaction is not active.")

        handle["store"] = deepcopy(
            transaction_handle.get(
                "snapshot",
                {},
            )
        )

        transaction_handle["active"] = False

        transaction_handle.pop(
            "snapshot",
            None,
        )


# ---------------------------------------------------------------
# Database transaction
# ---------------------------------------------------------------


class DatabaseTransaction:
    """
    Own one transaction lifecycle against a
    ready DatabaseConnection.

    Lifecycle:

        CREATED
           ↓
        ACTIVE
        /    \
       ↓      ↓
    COMMITTED ROLLED_BACK

    Backend failures may instead produce FAILED.
    """

    def __init__(
        self,
        *,
        connection: DatabaseConnection,
        adapter: Optional[TransactionAdapter] = None,
    ) -> None:
        self.connection = connection

        # Accessing connection.handle also
        # guarantees that the connection is
        # currently READY.
        if adapter is None:
            if (
                isinstance(
                    connection.handle,
                    dict,
                )
                and connection.config.backend == "memory"
            ):
                adapter = MemoryTransactionAdapter()

            else:
                raise (
                    TransactionStateError(
                        "A transaction adapter is required for this database backend."
                    )
                )

        self.adapter = adapter

        self.record = TransactionRecord(
            connection_id=(self.connection.status.connection_id),
        ).finalize()

        self._transaction_handle: Any = None

        self._lock = RLock()

    # ---------------------------------------------------------
    # Active
    # ---------------------------------------------------------

    @property
    def active(
        self,
    ) -> bool:
        return self.record.state == TransactionState.ACTIVE.value

    # ---------------------------------------------------------
    # Begin
    # ---------------------------------------------------------

    def begin(
        self,
    ) -> TransactionRecord:
        with self._lock:
            if self.record.state != TransactionState.CREATED.value:
                raise (
                    TransactionStateError(
                        f"Transaction cannot begin from state {self.record.state}."
                    )
                )

            if not self.connection.ready:
                raise (TransactionStateError("Database connection must be ready."))

        try:
            transaction_handle = self.adapter.begin(self.connection.handle)

        except Exception as exc:
            with self._lock:
                self.record.state = TransactionState.FAILED.value

                self.record.error_type = type(exc).__name__

                self.record.completed_at = utc_now()

            raise wrap_driver_error(
                exc,
                message=("Database transaction begin failed."),
                retryable=False,
                code=("transaction_begin_failed"),
            ) from exc

        with self._lock:
            self._transaction_handle = transaction_handle

            self.record.state = TransactionState.ACTIVE.value

            self.record.started_at = utc_now()

            return self.record.finalize()

    # ---------------------------------------------------------
    # Commit
    # ---------------------------------------------------------

    def commit(
        self,
    ) -> TransactionRecord:
        with self._lock:
            if self.record.state != TransactionState.ACTIVE.value:
                raise (
                    TransactionStateError(
                        f"Transaction cannot commit from state {self.record.state}."
                    )
                )

            transaction_handle = self._transaction_handle

        try:
            self.adapter.commit(
                self.connection.handle,
                transaction_handle,
            )

        except Exception as exc:
            with self._lock:
                self.record.state = TransactionState.FAILED.value

                self.record.error_type = type(exc).__name__

                self.record.completed_at = utc_now()

            raise TransactionCommitError(
                "Database transaction commit failed.",
                details={
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

        with self._lock:
            self._transaction_handle = None

            self.record.state = TransactionState.COMMITTED.value

            self.record.completed_at = utc_now()

            return self.record.finalize()

    # ---------------------------------------------------------
    # Rollback
    # ---------------------------------------------------------

    def rollback(
        self,
        *,
        reason: Optional[str] = None,
    ) -> TransactionRecord:
        with self._lock:
            if self.record.state != TransactionState.ACTIVE.value:
                raise (
                    TransactionStateError(
                        f"Transaction cannot roll back from state {self.record.state}."
                    )
                )

            transaction_handle = self._transaction_handle

        try:
            self.adapter.rollback(
                self.connection.handle,
                transaction_handle,
            )

        except Exception as exc:
            with self._lock:
                self.record.state = TransactionState.FAILED.value

                self.record.error_type = type(exc).__name__

                self.record.completed_at = utc_now()

            raise (
                TransactionRollbackError(
                    "Database transaction rollback failed.",
                    details={
                        "error_type": (type(exc).__name__),
                        "error_module": (type(exc).__module__),
                    },
                )
            ) from exc

        with self._lock:
            self._transaction_handle = None

            self.record.state = TransactionState.ROLLED_BACK.value

            self.record.rollback_reason = reason

            self.record.completed_at = utc_now()

            return self.record.finalize()

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "DatabaseTransaction":
        self.begin()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        if not self.active:
            return False

        if exc is None:
            self.commit()

        else:
            self.rollback(reason=("Transaction scope exited with an exception."))

        # The original application exception
        # must continue outward after rollback.
        return False

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "transaction": (self.record.to_dict()),
            "active": (self.active),
            "adapter": (type(self.adapter).__name__),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "TransactionAdapter",
    "MemoryTransactionAdapter",
    "DatabaseTransaction",
]
