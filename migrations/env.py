"""
OSHA Alembic Migration Environment.

Connects Alembic to OSHA's SQLAlchemy metadata and database configuration.

All ORM models are imported before migrations run so Base.metadata contains
the complete OSHA relational schema.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.database.base import Base
from src.database.config import get_database_config

# Import the model package so every mapped table is registered with
# Base.metadata before Alembic performs schema comparison.
import src.database.models  # noqa: F401


config = context.config


if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# Alembic uses this metadata for autogeneration.
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Return the OSHA database URL for Alembic.

    Credentials come from OSHA's environment-based database configuration
    rather than being hard-coded into alembic.ini.
    """
    database_config = get_database_config()

    return database_config.url.render_as_string(
        hide_password=False,
    )


def run_migrations_offline() -> None:
    """
    Run migrations without creating a live database connection.

    Useful when Alembic needs to generate SQL scripts rather than execute
    migrations directly.
    """
    url = get_database_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
        compare_server_default=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations against the configured OSHA relational database.
    """
    configuration = config.get_section(
        config.config_ini_section,
    ) or {}

    configuration["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=False,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
