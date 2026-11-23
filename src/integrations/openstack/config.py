from __future__ import annotations

"""
Non-secret OpenStack integration configuration
for OSHA.
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
    "OSHA_OPENSTACK_"
)


# ---------------------------------------------------------------
# OpenStack interfaces
# ---------------------------------------------------------------

_ALLOWED_INTERFACES = {
    "public",
    "internal",
    "admin",
}


# ---------------------------------------------------------------
# OpenStack authentication configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class OpenStackAuthConfig:
    """
    Non-secret Keystone/OpenStack authentication
    metadata.

    Password/token values must NOT be stored in
    this structure.
    """

    auth_url: str

    username: Optional[
        str
    ] = None

    user_id: Optional[
        str
    ] = None

    project_name: Optional[
        str
    ] = None

    project_id: Optional[
        str
    ] = None

    user_domain_name: Optional[
        str
    ] = None

    user_domain_id: Optional[
        str
    ] = None

    project_domain_name: Optional[
        str
    ] = None

    project_domain_id: Optional[
        str
    ] = None

    credential: Optional[
        SecretReference
    ] = None

    def validate(
        self,
    ) -> None:
        required_string(
            self.auth_url,
            name="auth_url",
            maximum_length=2048,
        )

        if (
            self.username is None
            and self.user_id is None
        ):
            raise ValueError(
                "OpenStack authentication "
                "requires username or "
                "user_id"
            )

        if (
            self.project_name is None
            and self.project_id is None
        ):
            raise ValueError(
                "OpenStack authentication "
                "requires project_name or "
                "project_id"
            )

        if self.username is not None:
            required_string(
                self.username,
                name="username",
                maximum_length=256,
            )

        if self.user_id is not None:
            required_string(
                self.user_id,
                name="user_id",
                maximum_length=256,
            )

        if self.project_name is not None:
            required_string(
                self.project_name,
                name="project_name",
                maximum_length=256,
            )

        if self.project_id is not None:
            required_string(
                self.project_id,
                name="project_id",
                maximum_length=256,
            )

        if self.credential is not None:
            self.credential.finalize()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "auth_url": (
                self.auth_url
            ),

            "username": (
                self.username
            ),

            "user_id": (
                self.user_id
            ),

            "project_name": (
                self.project_name
            ),

            "project_id": (
                self.project_id
            ),

            "user_domain_name": (
                self.user_domain_name
            ),

            "user_domain_id": (
                self.user_domain_id
            ),

            "project_domain_name": (
                self.project_domain_name
            ),

            "project_domain_id": (
                self.project_domain_id
            ),

            "credential": (
                self.credential.to_dict()
                if self.credential
                else None
            ),
        }


# ---------------------------------------------------------------
# OpenStack integration configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class OpenStackConfig:
    """
    Configuration for one OpenStack cloud target.

    Credential values remain behind
    security/SecretManager.
    """

    environment_id: str

    target_id: str

    auth: OpenStackAuthConfig

    region_name: Optional[
        str
    ] = None

    interface: str = "public"

    verify_tls: bool = True

    compute_api_version: str = "2"

    identity_api_version: str = "3"

    network_api_version: str = "2"

    # These allow us to progressively expose
    # services instead of assuming every cloud
    # provides every OpenStack service.
    enable_compute: bool = True

    enable_network: bool = True

    enable_identity: bool = True

    enable_image: bool = True

    enable_volume: bool = True

    enable_object_store: bool = False

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

        self.auth.validate()

        if (
            self.interface
            not in _ALLOWED_INTERFACES
        ):
            raise ValueError(
                "OpenStack interface must "
                "be one of public, "
                "internal, or admin"
            )

        if self.region_name is not None:
            required_string(
                self.region_name,
                name="region_name",
                maximum_length=256,
            )

        required_string(
            self.compute_api_version,
            name="compute_api_version",
            maximum_length=32,
        )

        required_string(
            self.identity_api_version,
            name="identity_api_version",
            maximum_length=32,
        )

        required_string(
            self.network_api_version,
            name="network_api_version",
            maximum_length=32,
        )

    # ---------------------------------------------------------
    # Generic integration target
    # ---------------------------------------------------------

    def target(
        self,
    ) -> IntegrationTarget:
        self.validate()

        return IntegrationTarget(
            provider=(
                ProviderKind.
                OPENSTACK.value
            ),

            environment_id=(
                self.environment_id
            ),

            target_id=(
                self.target_id
            ),

            endpoint=(
                self.auth.auth_url
            ),

            region=(
                self.region_name
            ),

            credential=(
                self.auth.credential
            ),

            attributes={
                "interface": (
                    self.interface
                ),

                "verify_tls": (
                    self.verify_tls
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

            "auth": (
                self.auth.to_dict()
            ),

            "region_name": (
                self.region_name
            ),

            "interface": (
                self.interface
            ),

            "verify_tls": (
                self.verify_tls
            ),

            "compute_api_version": (
                self.compute_api_version
            ),

            "identity_api_version": (
                self.identity_api_version
            ),

            "network_api_version": (
                self.network_api_version
            ),

            "services": {
                "compute": (
                    self.enable_compute
                ),
                "network": (
                    self.enable_network
                ),
                "identity": (
                    self.enable_identity
                ),
                "image": (
                    self.enable_image
                ),
                "volume": (
                    self.enable_volume
                ),
                "object_store": (
                    self.
                    enable_object_store
                ),
            },

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Environment parsing helpers
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
) -> OpenStackConfig:
    """
    Load OpenStack integration configuration.

    OSHA_OPENSTACK_SECRET_KEY contains the name
    of the secret entry/environment variable,
    not the secret value itself.
    """

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
        or "openstack-credential"
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

    auth_url = _text(
        values,
        prefix,
        "AUTH_URL",
    )

    if auth_url is None:
        raise ValueError(
            "OSHA_OPENSTACK_AUTH_URL "
            "is required"
        )

    config = OpenStackConfig(
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
                "openstack-default",
            )
            or "openstack-default"
        ),

        auth=OpenStackAuthConfig(
            auth_url=auth_url,

            username=_text(
                values,
                prefix,
                "USERNAME",
            ),

            user_id=_text(
                values,
                prefix,
                "USER_ID",
            ),

            project_name=_text(
                values,
                prefix,
                "PROJECT_NAME",
            ),

            project_id=_text(
                values,
                prefix,
                "PROJECT_ID",
            ),

            user_domain_name=_text(
                values,
                prefix,
                "USER_DOMAIN_NAME",
                "Default",
            ),

            user_domain_id=_text(
                values,
                prefix,
                "USER_DOMAIN_ID",
            ),

            project_domain_name=_text(
                values,
                prefix,
                "PROJECT_DOMAIN_NAME",
                "Default",
            ),

            project_domain_id=_text(
                values,
                prefix,
                "PROJECT_DOMAIN_ID",
            ),

            credential=credential,
        ),

        region_name=_text(
            values,
            prefix,
            "REGION_NAME",
        ),

        interface=(
            _text(
                values,
                prefix,
                "INTERFACE",
                "public",
            )
            or "public"
        ).lower(),

        verify_tls=_bool(
            values,
            prefix,
            "VERIFY_TLS",
            True,
        ),

        compute_api_version=(
            _text(
                values,
                prefix,
                "COMPUTE_API_VERSION",
                "2",
            )
            or "2"
        ),

        identity_api_version=(
            _text(
                values,
                prefix,
                "IDENTITY_API_VERSION",
                "3",
            )
            or "3"
        ),

        network_api_version=(
            _text(
                values,
                prefix,
                "NETWORK_API_VERSION",
                "2",
            )
            or "2"
        ),

        enable_compute=_bool(
            values,
            prefix,
            "ENABLE_COMPUTE",
            True,
        ),

        enable_network=_bool(
            values,
            prefix,
            "ENABLE_NETWORK",
            True,
        ),

        enable_identity=_bool(
            values,
            prefix,
            "ENABLE_IDENTITY",
            True,
        ),

        enable_image=_bool(
            values,
            prefix,
            "ENABLE_IMAGE",
            True,
        ),

        enable_volume=_bool(
            values,
            prefix,
            "ENABLE_VOLUME",
            True,
        ),

        enable_object_store=_bool(
            values,
            prefix,
            "ENABLE_OBJECT_STORE",
            False,
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
) -> OpenStackConfig:
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
    config: OpenStackConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "environment_id": (
            config.environment_id
        ),

        "target_id": (
            config.target_id
        ),

        "auth_url": (
            config.auth.auth_url
        ),

        "region_name": (
            config.region_name
        ),

        "interface": (
            config.interface
        ),

        "verify_tls": (
            config.verify_tls
        ),

        "username_configured": (
            config.auth.username
            is not None
        ),

        "project_name_configured": (
            config.auth.project_name
            is not None
        ),

        "project_id_configured": (
            config.auth.project_id
            is not None
        ),

        "credential_reference_configured": (
            config.auth.credential
            is not None
        ),

        "credential_provider": (
            config.auth.credential.
            provider
            if config.auth.credential
            else None
        ),

        "services": {
            "compute": (
                config.enable_compute
            ),
            "network": (
                config.enable_network
            ),
            "identity": (
                config.enable_identity
            ),
            "image": (
                config.enable_image
            ),
            "volume": (
                config.enable_volume
            ),
            "object_store": (
                config.enable_object_store
            ),
        },

        "secret_value_exposed": (
            False
        ),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",

    "OpenStackAuthConfig",
    "OpenStackConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
