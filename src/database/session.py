"""
OSHA Database Session Management.

Provides SQLAlchemy ORM session creation and transaction boundaries for
OSHA's relational persistence layer.

Responsibilities:
    - create ORM sessions
    - provide transactional context managers
    - commit successful transactions
    - roll back failed transactions
    - guarantee session cleanup
    - expose dependency-friendly session access

Engine lifecycle remains the responsibility of connection.py / engine.py.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, Iterator

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .connection import get_engine


logger = logging.getLogger(__name__)


class DatabaseSessionError(RuntimeError):
    """Raised when OSHA cannot complete a database session operation."""


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def create_session_factory(
    engine: Engine | None = None,
) -> sessionmaker:
    """
    Create an SQLAlchemy session factory.

    expire_on_commit=False is intentional for OSHA service-layer code so
    successfully committed ORM objects can still expose their loaded values
    after the transaction completes without immediately requiring another
    database round trip.

    autoflush=False keeps database writes explicit at service/repository
    boundaries rather than flushing unexpectedly during unrelated queries.
    """
    database_engine = engine or get_engine()

    return sessionmaker(
        bind=database_engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    """
    Return the process-wide ORM session factory.

    The factory itself is safe to reuse. Individual Session instances must
    not be shared between concurrent requests, workers or tasks.
    """
    return create_session_factory()


# ---------------------------------------------------------------------------
# Raw session creation
# ---------------------------------------------------------------------------

def create_session() -> Session:
    """
    Create a new ORM Session.

    The caller owns the returned Session and must close it.

    For most OSHA code, prefer session_scope() so commit, rollback and close
    behaviour is handled automatically.
    """
    factory = get_session_factory()
    return factory()


# ---------------------------------------------------------------------------
# Transactional session scope
# ---------------------------------------------------------------------------

@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional OSHA ORM session.

    Behaviour:

        success
            -> flush
            -> commit
            -> close

        exception
            -> rollback
            -> close
            -> re-raise original exception

    Example:

        with session_scope() as session:
            session.add(resource)

    The transaction boundary belongs here so callers do not accidentally
    forget rollback or cleanup.
    """
    session = create_session()

    try:
        yield session

        session.commit()

    except Exception:
        try:
            session.rollback()
        except SQLAlchemyError:
            logger.exception(
                "Failed to roll back OSHA database transaction"
            )

        logger.exception(
            "OSHA database transaction failed"
        )

        raise

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Explicit read-only-style scope
# ---------------------------------------------------------------------------

@contextmanager
def session_context() -> Iterator[Session]:
    """
    Provide a Session without automatically committing.

    This is useful for read operations or cases where the caller deliberately
    controls transaction behaviour.

    Any active transaction is rolled back before final cleanup if an
    exception escapes the context.

    Example:

        with session_context() as session:
            resource = session.get(Resource, resource_id)

    For write operations, session_scope() is normally preferred.
    """
    session = create_session()

    try:
        yield session

    except Exception:
        try:
            session.rollback()
        except SQLAlchemyError:
            logger.exception(
                "Failed to roll back OSHA database session"
            )

        raise

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Dependency-oriented generator
# ---------------------------------------------------------------------------

def get_session() -> Generator[Session, None, None]:
    """
    Yield a Session for frameworks using generator-based dependencies.

    This dependency manages Session lifetime only.

    Transaction decisions remain with the service/repository layer rather
    than being silently committed by a web framework dependency.

    Example:

        session = next(get_session())

    Framework integrations should normally consume this as a generator
    dependency rather than calling next() directly.
    """
    session = create_session()

    try:
        yield session

    except Exception:
        try:
            session.rollback()
        except SQLAlchemyError:
            logger.exception(
                "Failed to roll back OSHA dependency session"
            )

        raise

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Transaction helpers
# ---------------------------------------------------------------------------

def commit_session(session: Session) -> None:
    """
    Commit a Session with consistent error handling.

    On commit failure the transaction is rolled back before the exception is
    surfaced to the caller.
    """
    try:
        session.commit()

    except SQLAlchemyError as exc:
        try:
            session.rollback()
        except SQLAlchemyError:
            logger.exception(
                "Rollback failed after OSHA database commit failure"
            )

        logger.exception(
            "OSHA database commit failed"
        )

        raise DatabaseSessionError(
            "Unable to commit OSHA database transaction"
        ) from exc


def rollback_session(session: Session) -> None:
    """
    Explicitly roll back a Session.

    Raises DatabaseSessionError if SQLAlchemy reports a rollback failure.
    """
    try:
        session.rollback()

    except SQLAlchemyError as exc:
        logger.exception(
            "OSHA database rollback failed"
        )

        raise DatabaseSessionError(
            "Unable to roll back OSHA database transaction"
        ) from exc


def flush_session(session: Session) -> None:
    """
    Flush pending ORM changes without committing the transaction.

    Useful when generated identifiers or database-side values are required
    before the surrounding transaction is committed.
    """
    try:
        session.flush()

    except SQLAlchemyError as exc:
        try:
            session.rollback()
        except SQLAlchemyError:
            logger.exception(
                "Rollback failed after OSHA database flush failure"
            )

        logger.exception(
            "OSHA database flush failed"
        )

        raise DatabaseSessionError(
            "Unable to flush OSHA database transaction"
        ) from exc


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def reset_session_factory() -> None:
    """
    Clear the cached ORM session factory.

    This should be used when the process-wide database engine is deliberately
    replaced, disposed or reconfigured.

    Existing Session instances are not modified by this function.
    """
    get_session_factory.cache_clear()


__all__ = [
    "DatabaseSessionError",
    "commit_session",
    "create_session",
    "create_session_factory",
    "flush_session",
    "get_session",
    "get_session_factory",
    "reset_session_factory",
    "rollback_session",
    "session_context",
    "session_scope",
]
