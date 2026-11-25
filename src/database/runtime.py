from __future__ import annotations

from typing import Mapping, Optional

from src.database.config import from_environment
from src.database.pipeline import DatabasePipeline
from src.security.secrets import SecretManager


def create_database_pipeline(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> DatabasePipeline:
    config = from_environment(environ=environ)
    secrets = SecretManager()
    secrets.register_default_environment_provider(environ=environ)
    return DatabasePipeline(config=config, secrets=secrets)
