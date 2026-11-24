from __future__ import annotations

"""
Non-secret Ceph integration configuration for
OSHA.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.core.validation import (
    boolean,
    number,
    required_string,
)

from src.integrations.schema import (
    IntegrationTarget,
    ProviderKind,
)

from src.security.schema import (
    SecretReference,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = "OSHA_CEPH_"


# ---------------------------------------------------------------
# Ceph API authentication
# ---------------------------------------------------------------


@dataclass(frozen=True)
class CephAuthConfig:
    """
    Non-secret Ceph Dashboard API authentication
    metadata.

    Actual password/token values remain behind
    SecretReference / SecretManager.
    """

    username: Optional[str] = None

    credential: Optional[SecretReference] = None

    def validate(
        self,
    ) -> None:
        if self.username is not None:
            required_string(
                self.username,
                name="username",
                maximum_length=256,
            )

        if self.credential is not None:
            self.credential.finalize()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "username": (self.username),
            "credential": (self.credential.to_dict() if self.credential else None),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Ceph configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class CephConfig:
    """
    Configuration for one Ceph Dashboard REST
    API target.
    """

    environment_id: str

    target_id: str

    api_url: str

    auth: CephAuthConfig = field(default_factory=CephAuthConfig)

    verify_tls: bool = True

    ca_cert_path: Optional[str] = None

    request_timeout_seconds: float = 30.0

    # Ceph's API requires explicit endpoint API
    # versions in its Accept media type.
    api_version: str = "1.0"

    enable_health: bool = True

    enable_osd: bool = True

    enable_pools: bool = True

    enable_hosts: bool = True

    enable_monitors: bool = True

    enable_services: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        required_string(
            self.environment_id,
            name="environment_id",
            maximum_length=256,
        )

        required_string(
            self.target_id,
            name="target_id",
            maximum_length=512,
        )

        required_string(
            self.api_url,
            name="api_url",
            maximum_length=2048,
        )

        required_string(
            self.api_version,
            name="api_version",
            maximum_length=32,
        )

        number(
            self.request_timeout_seconds,
            name=("request_timeout_seconds"),
            minimum=0.001,
        )

        if self.ca_cert_path is not None:
            required_string(
                self.ca_cert_path,
                name="ca_cert_path",
                maximum_length=2048,
            )

        self.auth.validate()

    # ---------------------------------------------------------
    # Base API URL
    # ---------------------------------------------------------

    @property
    def normalized_api_url(
        self,
    ) -> str:
        """
        Normalize target to Ceph Dashboard's
        documented /api base path.
        """

        self.validate()

        value = self.api_url.rstrip("/")

        if value.endswith("/api"):
            return value

        return value + "/api"

    # ---------------------------------------------------------
    # Versioned media type
    # ---------------------------------------------------------

    @property
    def accept_header(
        self,
    ) -> str:
        self.validate()

        return f"application/vnd.ceph.api.v{self.api_version}+json"

    # ---------------------------------------------------------
    # Generic integration target
    # ---------------------------------------------------------

    def target(
        self,
    ) -> IntegrationTarget:
        self.validate()

        return IntegrationTarget(
            provider=(ProviderKind.CEPH.value),
            environment_id=(self.environment_id),
            target_id=(self.target_id),
            endpoint=(self.normalized_api_url),
            credential=(self.auth.credential),
            attributes={
                "api_version": (self.api_version),
                "verify_tls": (self.verify_tls),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Safe dictionary
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "environment_id": (self.environment_id),
            "target_id": (self.target_id),
            "api_url": (self.normalized_api_url),
            "api_version": (self.api_version),
            "verify_tls": (self.verify_tls),
            "ca_cert_path": (self.ca_cert_path),
            "request_timeout_seconds": (self.request_timeout_seconds),
            "auth": (self.auth.to_dict()),
            "resources": {
                "health": (self.enable_health),
                "osd": (self.enable_osd),
                "pools": (self.enable_pools),
                "hosts": (self.enable_hosts),
                "monitors": (self.enable_monitors),
                "services": (self.enable_services),
            },
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------


def _get(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Any = None,
) -> Any:
    return values.get(
        prefix + name,
        default,
    )


def _text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
        default,
    )

    if value is None:
        return None

    result = str(value).strip()

    return result or None


def _bool(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: bool,
) -> bool:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return boolean(
        value,
        name=prefix + name,
    )


def _float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: float,
) -> float:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return number(
        value,
        name=prefix + name,
        minimum=0.001,
    )


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------


def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> CephConfig:
    api_url = _text(
        values,
        prefix,
        "API_URL",
    )

    if not api_url:
        raise ValueError("OSHA_CEPH_API_URL is required")

    secret_key = _text(
        values,
        prefix,
        "SECRET_KEY",
    )

    secret_name = (
        _text(
            values,
            prefix,
            "SECRET_NAME",
        )
        or "ceph-credential"
    )

    secret_provider = (
        _text(
            values,
            prefix,
            "SECRET_PROVIDER",
        )
        or "environment"
    )

    credential = (
        SecretReference(
            name=secret_name,
            provider=(secret_provider),
            key=secret_key,
            version=_text(
                values,
                prefix,
                "SECRET_VERSION",
            ),
        ).finalize()
        if secret_key
        else None
    )

    config = CephConfig(
        environment_id=(
            _text(
                values,
                prefix,
                "ENVIRONMENT_ID",
                "local-dev",
            )
            or "local-dev"
        ),
        target_id=(
            _text(
                values,
                prefix,
                "TARGET_ID",
                "ceph-default",
            )
            or "ceph-default"
        ),
        api_url=api_url,
        auth=CephAuthConfig(
            username=_text(
                values,
                prefix,
                "USERNAME",
            ),
            credential=credential,
        ),
        verify_tls=_bool(
            values,
            prefix,
            "VERIFY_TLS",
            True,
        ),
        ca_cert_path=_text(
            values,
            prefix,
            "CA_CERT_PATH",
        ),
        request_timeout_seconds=(
            _float(
                values,
                prefix,
                "REQUEST_TIMEOUT_SECONDS",
                30.0,
            )
        ),
        api_version=(
            _text(
                values,
                prefix,
                "API_VERSION",
                "1.0",
            )
            or "1.0"
        ),
        enable_health=_bool(
            values,
            prefix,
            "ENABLE_HEALTH",
            True,
        ),
        enable_osd=_bool(
            values,
            prefix,
            "ENABLE_OSD",
            True,
        ),
        enable_pools=_bool(
            values,
            prefix,
            "ENABLE_POOLS",
            True,
        ),
        enable_hosts=_bool(
            values,
            prefix,
            "ENABLE_HOSTS",
            True,
        ),
        enable_monitors=_bool(
            values,
            prefix,
            "ENABLE_MONITORS",
            True,
        ),
        enable_services=_bool(
            values,
            prefix,
            "ENABLE_SERVICES",
            True,
        ),
        attributes={
            "config_source": ("mapping"),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------


def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> CephConfig:
    source = os.environ if environ is None else environ

    return from_mapping(
        source,
        prefix=prefix,
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------


def safe_config_summary(
    config: CephConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "environment_id": (config.environment_id),
        "target_id": (config.target_id),
        "api_url": (config.normalized_api_url),
        "api_version": (config.api_version),
        "verify_tls": (config.verify_tls),
        "username_configured": (config.auth.username is not None),
        "credential_reference_configured": (config.auth.credential is not None),
        "credential_provider": (
            config.auth.credential.provider if config.auth.credential else None
        ),
        "resources": {
            "health": (config.enable_health),
            "osd": (config.enable_osd),
            "pools": (config.enable_pools),
            "hosts": (config.enable_hosts),
            "monitors": (config.enable_monitors),
            "services": (config.enable_services),
        },
        "mutating_operations_enabled": (False),
        "credential_value_exposed": (False),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "CephAuthConfig",
    "CephConfig",
    "from_mapping",
    "from_environment",
    "safe_config_summary",
]
