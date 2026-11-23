from __future__ import annotations

"""
Non-secret MinIO integration configuration for
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

DEFAULT_ENV_PREFIX = (
    "OSHA_MINIO_"
)


# ---------------------------------------------------------------
# Authentication configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class MinioAuthConfig:
    """
    Non-secret MinIO/S3 authentication metadata.

    access_key is an identity value.

    The associated secret key remains behind
    SecretReference and SecretManager.
    """

    access_key: Optional[
        str
    ] = None

    credential: Optional[
        SecretReference
    ] = None

    def validate(
        self,
    ) -> None:
        if self.access_key is not None:
            required_string(
                self.access_key,
                name="access_key",
                maximum_length=512,
            )

        if self.credential is not None:
            self.credential.finalize()

        if (
            self.access_key is None
            and self.credential
            is not None
        ):
            raise ValueError(
                "MinIO secret credential "
                "requires an access_key"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "access_key_configured": (
                self.access_key
                is not None
            ),

            # Do not print the access key in safe
            # config output. Presence is enough
            # for normal diagnostics.
            "credential": (
                self.credential.to_dict()
                if self.credential
                else None
            ),

            "credential_value_exposed": (
                False
            ),
        }


# ---------------------------------------------------------------
# MinIO config
# ---------------------------------------------------------------

@dataclass(frozen=True)
class MinioConfig:
    """
    Configuration for one MinIO/S3-compatible
    object-storage target.
    """

    environment_id: str

    target_id: str

    endpoint: str

    auth: MinioAuthConfig = field(
        default_factory=MinioAuthConfig
    )

    secure: bool = True

    verify_tls: bool = True

    region: Optional[
        str
    ] = None

    request_timeout_seconds: float = 30.0

    enable_buckets: bool = True

    enable_objects: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

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
            self.endpoint,
            name="endpoint",
            maximum_length=2048,
        )

        if (
            "://" in self.endpoint
        ):
            raise ValueError(
                "MinIO endpoint must be "
                "host[:port] without an "
                "http:// or https:// prefix"
            )

        if self.region is not None:
            required_string(
                self.region,
                name="region",
                maximum_length=256,
            )

        number(
            self.request_timeout_seconds,
            name=(
                "request_timeout_seconds"
            ),
            minimum=0.001,
        )

        self.auth.validate()

    # ---------------------------------------------------------
    # Display endpoint
    # ---------------------------------------------------------

    @property
    def endpoint_url(
        self,
    ) -> str:
        self.validate()

        scheme = (
            "https"
            if self.secure
            else "http"
        )

        return (
            f"{scheme}://"
            f"{self.endpoint}"
        )

    # ---------------------------------------------------------
    # Generic target
    # ---------------------------------------------------------

    def target(
        self,
    ) -> IntegrationTarget:
        self.validate()

        return IntegrationTarget(
            provider=(
                ProviderKind.
                MINIO.value
            ),

            environment_id=(
                self.environment_id
            ),

            target_id=(
                self.target_id
            ),

            endpoint=(
                self.endpoint_url
            ),

            region=(
                self.region
            ),

            credential=(
                self.auth.credential
            ),

            attributes={
                "secure": (
                    self.secure
                ),

                "verify_tls": (
                    self.verify_tls
                ),

                "access_key_configured": (
                    self.auth.
                    access_key
                    is not None
                ),
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
            "environment_id": (
                self.environment_id
            ),

            "target_id": (
                self.target_id
            ),

            "endpoint": (
                self.endpoint_url
            ),

            "secure": (
                self.secure
            ),

            "verify_tls": (
                self.verify_tls
            ),

            "region": (
                self.region
            ),

            "request_timeout_seconds": (
                self.
                request_timeout_seconds
            ),

            "auth": (
                self.auth.to_dict()
            ),

            "resources": {
                "buckets": (
                    self.enable_buckets
                ),

                "objects": (
                    self.enable_objects
                ),
            },

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Mapping helpers
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
    default: Optional[
        str
    ] = None,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
        default,
    )

    if value is None:
        return None

    result = str(
        value
    ).strip()

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

    if (
        value is None
        or str(value).strip() == ""
    ):
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

    if (
        value is None
        or str(value).strip() == ""
    ):
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
) -> MinioConfig:
    endpoint = _text(
        values,
        prefix,
        "ENDPOINT",
    )

    if not endpoint:
        raise ValueError(
            "OSHA_MINIO_ENDPOINT "
            "is required"
        )

    access_key = _text(
        values,
        prefix,
        "ACCESS_KEY",
    )

    secret_key_reference = (
        _text(
            values,
            prefix,
            "SECRET_KEY",
        )
    )

    secret_name = (
        _text(
            values,
            prefix,
            "SECRET_NAME",
        )
        or "minio-secret-key"
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

            provider=(
                secret_provider
            ),

            key=(
                secret_key_reference
            ),

            version=_text(
                values,
                prefix,
                "SECRET_VERSION",
            ),
        ).finalize()
        if secret_key_reference
        else None
    )

    config = MinioConfig(
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
                "minio-default",
            )
            or "minio-default"
        ),

        endpoint=endpoint,

        auth=MinioAuthConfig(
            access_key=access_key,

            credential=credential,
        ),

        secure=_bool(
            values,
            prefix,
            "SECURE",
            True,
        ),

        verify_tls=_bool(
            values,
            prefix,
            "VERIFY_TLS",
            True,
        ),

        region=_text(
            values,
            prefix,
            "REGION",
        ),

        request_timeout_seconds=(
            _float(
                values,
                prefix,
                "REQUEST_TIMEOUT_SECONDS",
                30.0,
            )
        ),

        enable_buckets=_bool(
            values,
            prefix,
            "ENABLE_BUCKETS",
            True,
        ),

        enable_objects=_bool(
            values,
            prefix,
            "ENABLE_OBJECTS",
            True,
        ),

        attributes={
            "config_source": (
                "mapping"
            ),
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
) -> MinioConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    return from_mapping(
        source,
        prefix=prefix,
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: MinioConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "environment_id": (
            config.environment_id
        ),

        "target_id": (
            config.target_id
        ),

        "endpoint": (
            config.endpoint_url
        ),

        "secure": (
            config.secure
        ),

        "verify_tls": (
            config.verify_tls
        ),

        "region": (
            config.region
        ),

        "access_key_configured": (
            config.auth.access_key
            is not None
        ),

        "credential_reference_configured": (
            config.auth.credential
            is not None
        ),

        "credential_provider": (
            config.auth.
            credential.provider
            if config.auth.credential
            else None
        ),

        "resources": {
            "buckets": (
                config.enable_buckets
            ),

            "objects": (
                config.enable_objects
            ),
        },

        "mutating_operations_enabled": (
            False
        ),

        "credential_value_exposed": (
            False
        ),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",

    "MinioAuthConfig",
    "MinioConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
