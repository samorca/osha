from __future__ import annotations

"""
Non-secret Kubernetes integration configuration
for OSHA.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from pathlib import Path

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

DEFAULT_ENV_PREFIX = "OSHA_KUBERNETES_"


# ---------------------------------------------------------------
# Authentication/configuration modes
# ---------------------------------------------------------------


class KubernetesConfigMode(
    str,
    Enum,
):
    KUBECONFIG = "kubeconfig"

    IN_CLUSTER = "in_cluster"

    DIRECT = "direct"


# ---------------------------------------------------------------
# Kubernetes authentication configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class KubernetesAuthConfig:
    """
    Kubernetes client configuration source.

    No bearer token, client key, or certificate
    contents belong directly in this object.
    """

    mode: str = KubernetesConfigMode.KUBECONFIG.value

    kubeconfig_path: Optional[str] = None

    context: Optional[str] = None

    api_server: Optional[str] = None

    ca_cert_path: Optional[str] = None

    credential: Optional[SecretReference] = None

    def validate(
        self,
    ) -> None:
        if self.mode not in {item.value for item in KubernetesConfigMode}:
            raise ValueError(f"Unknown Kubernetes configuration mode: {self.mode}")

        # -----------------------------------------------------
        # Kubeconfig mode
        # -----------------------------------------------------

        if self.mode == KubernetesConfigMode.KUBECONFIG.value:
            if self.kubeconfig_path is not None:
                required_string(
                    self.kubeconfig_path,
                    name=("kubeconfig_path"),
                    maximum_length=2048,
                )

            if self.context is not None:
                required_string(
                    self.context,
                    name=("kubeconfig context"),
                    maximum_length=512,
                )

        # -----------------------------------------------------
        # In-cluster mode
        # -----------------------------------------------------

        elif self.mode == KubernetesConfigMode.IN_CLUSTER.value:
            if self.api_server is not None:
                raise ValueError(
                    "api_server should not be configured for in_cluster mode"
                )

        # -----------------------------------------------------
        # Direct API configuration
        # -----------------------------------------------------

        elif self.mode == KubernetesConfigMode.DIRECT.value:
            if not self.api_server:
                raise ValueError("direct Kubernetes configuration requires api_server")

            required_string(
                self.api_server,
                name="api_server",
                maximum_length=2048,
            )

        if self.ca_cert_path is not None:
            required_string(
                self.ca_cert_path,
                name="ca_cert_path",
                maximum_length=2048,
            )

        if self.credential is not None:
            self.credential.finalize()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "mode": self.mode,
            "kubeconfig_path": (self.kubeconfig_path),
            "context": self.context,
            "api_server": (self.api_server),
            "ca_cert_path": (self.ca_cert_path),
            "credential": (self.credential.to_dict() if self.credential else None),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Kubernetes integration configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class KubernetesConfig:
    """
    Configuration for one Kubernetes cluster
    integration.

    This config controls how OSHA connects and
    which read-side API groups it exposes.

    Mutating permissions remain governed by the
    shared integration/security/remediation
    layers.
    """

    environment_id: str

    target_id: str

    auth: KubernetesAuthConfig = field(default_factory=(KubernetesAuthConfig))

    default_namespace: str = "default"

    verify_tls: bool = True

    request_timeout_seconds: float = 30.0

    enable_core: bool = True

    enable_apps: bool = True

    enable_batch: bool = False

    enable_networking: bool = True

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
            self.default_namespace,
            name="default_namespace",
            maximum_length=253,
        )

        number(
            self.request_timeout_seconds,
            name=("request_timeout_seconds"),
            minimum=0.001,
        )

        self.auth.validate()

    # ---------------------------------------------------------
    # Generic IntegrationTarget
    # ---------------------------------------------------------

    def target(
        self,
    ) -> IntegrationTarget:
        self.validate()

        endpoint = (
            self.auth.api_server
            if (self.auth.mode == KubernetesConfigMode.DIRECT.value)
            else None
        )

        return IntegrationTarget(
            provider=(ProviderKind.KUBERNETES.value),
            environment_id=(self.environment_id),
            target_id=(self.target_id),
            endpoint=endpoint,
            namespace=(self.default_namespace),
            credential=(self.auth.credential),
            attributes={
                "configuration_mode": (self.auth.mode),
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
            "auth": (self.auth.to_dict()),
            "default_namespace": (self.default_namespace),
            "verify_tls": (self.verify_tls),
            "request_timeout_seconds": (self.request_timeout_seconds),
            "api_groups": {
                "core": (self.enable_core),
                "apps": (self.enable_apps),
                "batch": (self.enable_batch),
                "networking": (self.enable_networking),
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
) -> KubernetesConfig:
    """
    Build Kubernetes integration configuration.

    SECRET_KEY identifies a secret entry. It is
    not itself the bearer token.
    """

    mode = (
        _text(
            values,
            prefix,
            "CONFIG_MODE",
            KubernetesConfigMode.KUBECONFIG.value,
        )
        or (KubernetesConfigMode.KUBECONFIG.value)
    ).lower()

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
        or "kubernetes-credential"
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
            provider=secret_provider,
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

    kubeconfig_path = _text(
        values,
        prefix,
        "KUBECONFIG",
    )

    # Leave None if not explicitly configured.
    # The Python client can then use its normal
    # kubeconfig loading behavior.
    if kubeconfig_path:
        kubeconfig_path = str(Path(kubeconfig_path).expanduser())

    config = KubernetesConfig(
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
                "kubernetes-default",
            )
            or "kubernetes-default"
        ),
        auth=KubernetesAuthConfig(
            mode=mode,
            kubeconfig_path=(kubeconfig_path),
            context=_text(
                values,
                prefix,
                "CONTEXT",
            ),
            api_server=_text(
                values,
                prefix,
                "API_SERVER",
            ),
            ca_cert_path=_text(
                values,
                prefix,
                "CA_CERT_PATH",
            ),
            credential=credential,
        ),
        default_namespace=(
            _text(
                values,
                prefix,
                "DEFAULT_NAMESPACE",
                "default",
            )
            or "default"
        ),
        verify_tls=_bool(
            values,
            prefix,
            "VERIFY_TLS",
            True,
        ),
        request_timeout_seconds=(_float(values, prefix, "REQUEST_TIMEOUT_SECONDS", 30)),
        enable_core=_bool(
            values,
            prefix,
            "ENABLE_CORE",
            True,
        ),
        enable_apps=_bool(
            values,
            prefix,
            "ENABLE_APPS",
            True,
        ),
        enable_batch=_bool(
            values,
            prefix,
            "ENABLE_BATCH",
            False,
        ),
        enable_networking=_bool(
            values,
            prefix,
            "ENABLE_NETWORKING",
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
) -> KubernetesConfig:
    source = os.environ if environ is None else environ

    return from_mapping(
        source,
        prefix=prefix,
    )


# ---------------------------------------------------------------
# Safe configuration summary
# ---------------------------------------------------------------


def safe_config_summary(
    config: KubernetesConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "environment_id": (config.environment_id),
        "target_id": (config.target_id),
        "configuration_mode": (config.auth.mode),
        "kubeconfig_configured": (config.auth.kubeconfig_path is not None),
        "context_configured": (config.auth.context is not None),
        "api_server_configured": (config.auth.api_server is not None),
        "default_namespace": (config.default_namespace),
        "verify_tls": (config.verify_tls),
        "request_timeout_seconds": (config.request_timeout_seconds),
        "credential_reference_configured": (config.auth.credential is not None),
        "credential_provider": (
            config.auth.credential.provider if config.auth.credential else None
        ),
        "api_groups": {
            "core": (config.enable_core),
            "apps": (config.enable_apps),
            "batch": (config.enable_batch),
            "networking": (config.enable_networking),
        },
        "secret_value_exposed": (False),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "KubernetesConfigMode",
    "KubernetesAuthConfig",
    "KubernetesConfig",
    "from_mapping",
    "from_environment",
    "safe_config_summary",
]
