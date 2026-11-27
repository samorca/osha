"""
OSHA Base Repository.

Provides common persistence operations for SQLAlchemy ORM models.

Repositories form the persistence boundary between OSHA application/domain
logic and SQLAlchemy.

Responsibilities:
    - retrieve entities
    - persist entities
    - delete entities when appropriate
    - expose common query helpers
    - provide consistent pagination
    - keep direct ORM access out of higher application layers

Transactions are NOT committed automatically here.

Transaction ownership belongs to the caller through OSHA's session layer:

    with session_scope() as session:
        repository = ResourceRepository(session)
        repository.add(resource)

This allows several repository operations to participate in one atomic
database transaction.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Generic, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from ..base import Base


ModelT = TypeVar(
    "ModelT",
    bound=Base,
)


class RepositoryError(RuntimeError):
    """Base exception raised by OSHA repository operations."""


class EntityNotFoundError(RepositoryError):
    """Raised when a requested database entity does not exist."""


class InvalidPaginationError(RepositoryError):
    """Raised when invalid pagination parameters are supplied."""


class BaseRepository(Generic[ModelT]):
    """
    Generic repository for an OSHA SQLAlchemy model.

    Concrete repositories define ``model``:

        class ResourceRepository(BaseRepository[Resource]):
            model = Resource

    The repository deliberately does not commit transactions. This is
    important because a single OSHA operation may need to update several
    entities atomically.
    """

    model: type[ModelT]

    def __init__(
        self,
        session: Session,
    ) -> None:
        if session is None:
            raise ValueError(
                "repository requires an active SQLAlchemy Session"
            )

        self.session = session

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    @property
    def model_class(self) -> type[ModelT]:
        """Return the ORM model managed by the repository."""
        model = getattr(self, "model", None)

        if model is None:
            raise RepositoryError(
                f"{self.__class__.__name__} does not define a model"
            )

        return model

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        entity_id: uuid.UUID,
    ) -> ModelT | None:
        """
        Return an entity by primary key.

        Returns None when the entity does not exist.
        """
        return self.session.get(
            self.model_class,
            entity_id,
        )

    def get_required(
        self,
        entity_id: uuid.UUID,
    ) -> ModelT:
        """
        Return an entity by primary key or raise EntityNotFoundError.
        """
        entity = self.get(entity_id)

        if entity is None:
            raise EntityNotFoundError(
                f"{self.model_class.__name__} "
                f"{entity_id} was not found"
            )

        return entity

    def exists(
        self,
        entity_id: uuid.UUID,
    ) -> bool:
        """Return whether an entity exists."""
        statement = (
            select(self.model_class.id)
            .where(
                self.model_class.id == entity_id
            )
            .limit(1)
        )

        return self.session.scalar(statement) is not None

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def select_statement(self) -> Select:
        """
        Return the base SELECT statement for the repository model.

        Concrete repositories may extend this statement with domain-specific
        filters.
        """
        return select(self.model_class)

    # ------------------------------------------------------------------
    # Listing / pagination
    # ------------------------------------------------------------------

    def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[ModelT]:
        """
        Return a bounded page of entities.

        ``limit`` is intentionally bounded to protect API and worker paths
        from accidentally loading unbounded tables into memory.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            self.select_statement()
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def count(self) -> int:
        """Return the total number of entities for this model."""
        statement = select(
            func.count(self.model_class.id)
        )

        result = self.session.scalar(statement)

        return int(result or 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def add(
        self,
        entity: ModelT,
        *,
        flush: bool = False,
    ) -> ModelT:
        """
        Add an entity to the current Session.

        The transaction is not committed here.

        ``flush=True`` can be used when the caller needs generated database
        state before the surrounding transaction completes.
        """
        self._validate_entity(entity)

        self.session.add(entity)

        if flush:
            self.session.flush()

        return entity

    def add_all(
        self,
        entities: Sequence[ModelT],
        *,
        flush: bool = False,
    ) -> Sequence[ModelT]:
        """
        Add multiple entities to the current Session.

        The sequence is validated before persistence.
        """
        for entity in entities:
            self._validate_entity(entity)

        self.session.add_all(
            list(entities)
        )

        if flush:
            self.session.flush()

        return entities

    def flush(self) -> None:
        """Flush pending changes without committing the transaction."""
        self.session.flush()

    def refresh(
        self,
        entity: ModelT,
    ) -> ModelT:
        """Refresh an entity from the database."""
        self._validate_entity(entity)

        self.session.refresh(entity)

        return entity

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(
        self,
        entity: ModelT,
        *,
        flush: bool = False,
    ) -> None:
        """
        Delete an entity from the current Session.

        This should only be used by models where physical deletion is valid.

        Infrastructure history models should often prefer domain-specific
        soft-delete operations.
        """
        self._validate_entity(entity)

        self.session.delete(entity)

        if flush:
            self.session.flush()

    def delete_by_id(
        self,
        entity_id: uuid.UUID,
        *,
        flush: bool = False,
    ) -> bool:
        """
        Delete an entity by ID.

        Returns:
            True if an entity existed and was scheduled for deletion.
            False if the entity did not exist.
        """
        entity = self.get(entity_id)

        if entity is None:
            return False

        self.delete(
            entity,
            flush=flush,
        )

        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_entity(
        self,
        entity: ModelT,
    ) -> None:
        """Verify an entity is compatible with this repository."""
        if not isinstance(
            entity,
            self.model_class,
        ):
            raise TypeError(
                f"{self.__class__.__name__} manages "
                f"{self.model_class.__name__}, got "
                f"{entity.__class__.__name__}"
            )

    @staticmethod
    def _validate_pagination(
        *,
        offset: int,
        limit: int,
    ) -> None:
        """Validate safe repository pagination values."""
        if offset < 0:
            raise InvalidPaginationError(
                "offset must be greater than or equal to 0"
            )

        if limit < 1:
            raise InvalidPaginationError(
                "limit must be greater than 0"
            )

        if limit > 1000:
            raise InvalidPaginationError(
                "limit must not exceed 1000"
            )


__all__ = [
    "BaseRepository",
    "EntityNotFoundError",
    "InvalidPaginationError",
    "RepositoryError",
]
