from __future__ import annotations

"""
OpenStack SDK client boundary for OSHA.

Provider-specific SDK objects remain inside this
module and openstack/adapter.py.
"""

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.integrations.errors import (
    IntegrationAuthenticationError,
    IntegrationCapabilityError,
    IntegrationConnectionError,
    IntegrationOperationError,
    IntegrationResourceNotFoundError,
    wrap_provider_error,
)

from src.integrations.openstack.config import (
    OpenStackConfig,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------

try:
    from openstack import (
        connection as openstack_connection,
    )

except ImportError:
    openstack_connection = None


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class OpenStackClientError(IntegrationOperationError):
    default_code = "openstack_client_error"


# ---------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------


def resource_to_dict(
    resource: Any,
) -> Dict[str, Any]:
    """
    Convert an OpenStack SDK resource into a
    bounded ordinary mapping.

    We prefer SDK-provided conversion methods and
    avoid blindly traversing arbitrary object
    internals.
    """

    if resource is None:
        return {}

    if isinstance(
        resource,
        Mapping,
    ):
        return dict(resource)

    to_dict = getattr(
        resource,
        "to_dict",
        None,
    )

    if callable(to_dict):
        try:
            value = to_dict()

        except TypeError:
            # Some SDK resource implementations
            # expose to_dict with optional flags.
            value = to_dict(original_names=False)

        if isinstance(
            value,
            Mapping,
        ):
            return dict(value)

    # Keep fallback deliberately narrow.
    result: Dict[
        str,
        Any,
    ] = {}

    for name in (
        "id",
        "name",
        "status",
    ):
        value = getattr(
            resource,
            name,
            None,
        )

        if value is not None:
            result[name] = value

    return result


def resources_to_dicts(
    resources: Iterable[Any],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    return [resource_to_dict(resource) for resource in resources]


# ---------------------------------------------------------------
# OpenStack client
# ---------------------------------------------------------------


class OpenStackClient:
    """
    Thin wrapper around openstacksdk Connection.

    The client deals with OpenStack SDK semantics.

    Generic integration lifecycle, authorization,
    mutation policy and target selection remain in
    the higher IntegrationAdapter layer.
    """

    def __init__(
        self,
        *,
        config: OpenStackConfig,
        credential: Optional[SecretValue] = None,
    ) -> None:
        self.config = config

        self.config.validate()

        self.credential = credential

        self._connection: Any = None

    # ---------------------------------------------------------
    # Connected
    # ---------------------------------------------------------

    @property
    def connected(
        self,
    ) -> bool:
        return self._connection is not None

    # ---------------------------------------------------------
    # Connection
    # ---------------------------------------------------------

    @property
    def connection(
        self,
    ) -> Any:
        if self._connection is None:
            raise (IntegrationConnectionError("OpenStack client is not connected."))

        return self._connection

    # ---------------------------------------------------------
    # Authentication mapping
    # ---------------------------------------------------------

    def _auth(
        self,
    ) -> Dict[str, Any]:
        auth = self.config.auth

        values: Dict[
            str,
            Any,
        ] = {
            "auth_url": (auth.auth_url),
        }

        if auth.username:
            values["username"] = auth.username

        if auth.user_id:
            values["user_id"] = auth.user_id

        if auth.project_name:
            values["project_name"] = auth.project_name

        if auth.project_id:
            values["project_id"] = auth.project_id

        if auth.user_domain_name:
            values["user_domain_name"] = auth.user_domain_name

        if auth.user_domain_id:
            values["user_domain_id"] = auth.user_domain_id

        if auth.project_domain_name:
            values["project_domain_name"] = auth.project_domain_name

        if auth.project_domain_id:
            values["project_domain_id"] = auth.project_domain_id

        if self.credential is not None:
            # SecretValue is deliberately
            # revealed only at the provider SDK
            # authentication boundary.
            values["password"] = self.credential.reveal()

        return values

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    def connect(
        self,
    ) -> Any:
        """
        Construct an explicit OpenStack SDK
        Connection.

        This avoids implicitly loading another
        cloud definition when OSHA is already
        supplying its own normalized config.
        """

        if self.connected:
            return self._connection

        if openstack_connection is None:
            raise (IntegrationConnectionError("openstacksdk is not installed."))

        try:
            self._connection = openstack_connection.Connection(
                region_name=(self.config.region_name),
                auth=self._auth(),
                verify=(self.config.verify_tls),
                compute_api_version=(self.config.compute_api_version),
                identity_api_version=(self.config.identity_api_version),
                network_api_version=(self.config.network_api_version),
                compute_interface=(self.config.interface),
                network_interface=(self.config.interface),
                identity_interface=(self.config.interface),
            )

            return self._connection

        except Exception as exc:
            self._connection = None

            raise wrap_provider_error(
                exc,
                provider="openstack",
                target_id=(self.config.target_id),
                message=("OpenStack connection failed."),
                retryable=True,
                code=("openstack_connection_failed"),
            ) from exc

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(
        self,
    ) -> None:
        """
        Release the local SDK connection object.

        If the SDK connection exposes close(),
        invoke it.
        """

        connection = self._connection

        if connection is None:
            return

        close = getattr(
            connection,
            "close",
            None,
        )

        try:
            if callable(close):
                close()

        except Exception as exc:
            raise wrap_provider_error(
                exc,
                provider="openstack",
                target_id=(self.config.target_id),
                message=("OpenStack client close failed."),
                retryable=False,
                code=("openstack_close_failed"),
            ) from exc

        finally:
            self._connection = None

    # ---------------------------------------------------------
    # Connectivity
    # ---------------------------------------------------------

    def ping(
        self,
    ) -> bool:
        """
        Perform a lightweight authenticated API
        request.

        Identity projects are used when identity
        support is enabled.
        """

        if not self.connected:
            return False

        try:
            if self.config.enable_identity:
                iterator = self.connection.identity.projects()

                # Force at least initial API
                # evaluation without consuming
                # the complete project list.
                next(
                    iter(iterator),
                    None,
                )

                return True

            if self.config.enable_compute:
                iterator = self.connection.compute.servers(details=False)

                next(
                    iter(iterator),
                    None,
                )

                return True

            # A configured connection object is
            # insufficient to prove remote
            # connectivity if no service probe is
            # enabled.
            return False

        except Exception:
            return False

    # ---------------------------------------------------------
    # Compute
    # ---------------------------------------------------------

    def list_servers(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_service("compute")

        try:
            return resources_to_dicts(self.connection.compute.servers(details=True))

        except Exception as exc:
            raise self._operation_error(
                exc,
                "OpenStack server listing failed.",
            ) from exc

    def get_server(
        self,
        server_id: str,
    ) -> Dict[str, Any]:
        self._require_service("compute")

        try:
            server = self.connection.compute.get_server(server_id)

        except Exception as exc:
            raise self._operation_error(
                exc,
                "OpenStack server lookup failed.",
            ) from exc

        if server is None:
            raise (IntegrationResourceNotFoundError("OpenStack server was not found."))

        return resource_to_dict(server)

    # ---------------------------------------------------------
    # Network
    # ---------------------------------------------------------

    def list_networks(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_service("network")

        try:
            return resources_to_dicts(self.connection.network.networks())

        except Exception as exc:
            raise self._operation_error(
                exc,
                "OpenStack network listing failed.",
            ) from exc

    # ---------------------------------------------------------
    # Images
    # ---------------------------------------------------------

    def list_images(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_service("image")

        try:
            return resources_to_dicts(self.connection.image.images())

        except Exception as exc:
            raise self._operation_error(
                exc,
                "OpenStack image listing failed.",
            ) from exc

    # ---------------------------------------------------------
    # Volumes
    # ---------------------------------------------------------

    def list_volumes(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_service("volume")

        try:
            return resources_to_dicts(self.connection.block_storage.volumes())

        except Exception as exc:
            raise self._operation_error(
                exc,
                "OpenStack volume listing failed.",
            ) from exc

    # ---------------------------------------------------------
    # Service guard
    # ---------------------------------------------------------

    def _require_service(
        self,
        service: str,
    ) -> None:
        enabled = {
            "compute": (self.config.enable_compute),
            "network": (self.config.enable_network),
            "identity": (self.config.enable_identity),
            "image": (self.config.enable_image),
            "volume": (self.config.enable_volume),
            "object_store": (self.config.enable_object_store),
        }

        if service not in enabled:
            raise (IntegrationCapabilityError(f"Unknown OpenStack service: {service}"))

        if not enabled[service]:
            raise (
                IntegrationCapabilityError(
                    f"OpenStack service is disabled by configuration: {service}"
                )
            )

    # ---------------------------------------------------------
    # Error translation
    # ---------------------------------------------------------

    def _operation_error(
        self,
        error: BaseException,
        message: str,
    ) -> OpenStackClientError:
        wrapped = wrap_provider_error(
            error,
            provider="openstack",
            target_id=(self.config.target_id),
            message=message,
            retryable=False,
            code=("openstack_operation_failed"),
        )

        return OpenStackClientError(
            message,
            code=wrapped.code,
            retryable=(wrapped.retryable),
            details=(wrapped.details),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connected": (self.connected),
            "target_id": (self.config.target_id),
            "region_name": (self.config.region_name),
            "interface": (self.config.interface),
            "services": {
                "compute": (self.config.enable_compute),
                "network": (self.config.enable_network),
                "identity": (self.config.enable_identity),
                "image": (self.config.enable_image),
                "volume": (self.config.enable_volume),
                "object_store": (self.config.enable_object_store),
            },
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "OpenStackClientError",
    "resource_to_dict",
    "resources_to_dicts",
    "OpenStackClient",
]
