from __future__ import annotations

"""
OSHA IntegrationAdapter implementation for
OpenStack.

This first implementation intentionally exposes
read-only infrastructure discovery capabilities.

Mutating OpenStack operations can be added later
through the remediation/security execution path.
"""

from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
)

from src.integrations.base import (
    IntegrationAdapter,
)

from src.integrations.config import (
    IntegrationsConfig,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationOperationError,
)

from src.integrations.openstack.client import (
    OpenStackClient,
)

from src.integrations.openstack.config import (
    OpenStackConfig,
)

from src.integrations.schema import (
    IntegrationCapability,
    IntegrationOperation,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Capability names
# ---------------------------------------------------------------

CAPABILITY_COMPUTE_READ = (
    "compute.read"
)

CAPABILITY_NETWORK_READ = (
    "network.read"
)

CAPABILITY_IMAGE_READ = (
    "image.read"
)

CAPABILITY_STORAGE_READ = (
    "storage.read"
)


# ---------------------------------------------------------------
# Supported actions
# ---------------------------------------------------------------

ACTION_LIST = "list"

ACTION_GET = "get"


# ---------------------------------------------------------------
# OpenStack adapter
# ---------------------------------------------------------------

class OpenStackAdapter(
    IntegrationAdapter
):
    """
    Translate provider-neutral OSHA operations
    into OpenStackClient calls.

    Current capabilities are read-only:

      compute.read
      network.read
      image.read
      storage.read

    Infrastructure mutations are deliberately
    absent from this first adapter version.
    """

    def __init__(
        self,
        *,
        openstack_config: OpenStackConfig,
        config: Optional[
            IntegrationsConfig
        ] = None,
    ) -> None:
        self.openstack_config = (
            openstack_config
        )

        self.openstack_config.validate()

        super().__init__(
            target=(
                self.openstack_config.
                target()
            ),

            config=(
                config
                or IntegrationsConfig()
            ),
        )

    # ---------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------

    def define_capabilities(
        self,
    ) -> Iterable[
        IntegrationCapability
    ]:
        """
        Advertise only services explicitly
        enabled in OpenStackConfig.
        """

        capabilities = []

        if (
            self.openstack_config.
            enable_compute
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_COMPUTE_READ
                    ),

                    resource="compute",

                    actions=[
                        ACTION_LIST,
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": (
                            "openstack"
                        ),

                        "service": (
                            "compute"
                        ),
                    },
                )
            )

        if (
            self.openstack_config.
            enable_network
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_NETWORK_READ
                    ),

                    resource="network",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": (
                            "openstack"
                        ),

                        "service": (
                            "network"
                        ),
                    },
                )
            )

        if (
            self.openstack_config.
            enable_image
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_IMAGE_READ
                    ),

                    resource="image",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": (
                            "openstack"
                        ),

                        "service": (
                            "image"
                        ),
                    },
                )
            )

        if (
            self.openstack_config.
            enable_volume
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_STORAGE_READ
                    ),

                    resource="volume",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": (
                            "openstack"
                        ),

                        "service": (
                            "block_storage"
                        ),
                    },
                )
            )

        return capabilities

    # ---------------------------------------------------------
    # Create provider client
    # ---------------------------------------------------------

    def connect_client(
        self,
        credential: Optional[
            SecretValue
        ],
    ) -> OpenStackClient:
        client = OpenStackClient(
            config=(
                self.openstack_config
            ),

            credential=credential,
        )

        client.connect()

        return client

    # ---------------------------------------------------------
    # Close provider client
    # ---------------------------------------------------------

    def close_client(
        self,
        client: OpenStackClient,
    ) -> None:
        client.close()

    # ---------------------------------------------------------
    # Connectivity/readiness
    # ---------------------------------------------------------

    def ping_client(
        self,
        client: OpenStackClient,
    ) -> bool:
        return client.ping()

    # ---------------------------------------------------------
    # Operation dispatch
    # ---------------------------------------------------------

    def execute_operation(
        self,
        client: OpenStackClient,
        operation: IntegrationOperation,
    ) -> Any:
        """
        Translate one validated generic OSHA
        operation into an OpenStackClient call.

        IntegrationAdapter.dispatch() already
        performs:

          target validation
          capability validation
          read/mutation policy checks

        before reaching this method.
        """

        operation.finalize()

        capability = (
            operation.capability
        )

        action = (
            operation.action
        )

        # -----------------------------------------------------
        # Compute
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_COMPUTE_READ
        ):
            if action == ACTION_LIST:
                return {
                    "servers": (
                        client.list_servers()
                    )
                }

            if action == ACTION_GET:
                if not operation.resource_id:
                    raise (
                        IntegrationOperationError(
                            "compute.read/get "
                            "requires "
                            "resource_id."
                        )
                    )

                return {
                    "server": (
                        client.get_server(
                            operation.
                            resource_id
                        )
                    )
                }

        # -----------------------------------------------------
        # Network
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_NETWORK_READ
        ):
            if action == ACTION_LIST:
                return {
                    "networks": (
                        client.
                        list_networks()
                    )
                }

        # -----------------------------------------------------
        # Images
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_IMAGE_READ
        ):
            if action == ACTION_LIST:
                return {
                    "images": (
                        client.list_images()
                    )
                }

        # -----------------------------------------------------
        # Storage
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_STORAGE_READ
        ):
            if action == ACTION_LIST:
                return {
                    "volumes": (
                        client.list_volumes()
                    )
                }

        # -----------------------------------------------------
        # Unsupported operation
        # -----------------------------------------------------

        raise (
            IntegrationCapabilityError(
                "Unsupported OpenStack "
                "integration operation: "
                f"{capability}/{action}"
            )
        )

    # ---------------------------------------------------------
    # Provider-specific summary
    # ---------------------------------------------------------

    def provider_summary(
        self,
    ) -> Dict[str, Any]:
        client_summary = None

        if self.ready:
            client = self.client

            if isinstance(
                client,
                OpenStackClient,
            ):
                client_summary = (
                    client.summary()
                )

        return {
            "provider": "openstack",

            "target_id": (
                self.target.target_id
            ),

            "environment_id": (
                self.target.
                environment_id
            ),

            "region_name": (
                self.openstack_config.
                region_name
            ),

            "interface": (
                self.openstack_config.
                interface
            ),

            "ready": self.ready,

            "client": (
                client_summary
            ),

            "capabilities": [
                capability.name
                for capability
                in self.capabilities()
            ],

            "mutating_capabilities_advertised": (
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
    "CAPABILITY_COMPUTE_READ",
    "CAPABILITY_NETWORK_READ",
    "CAPABILITY_IMAGE_READ",
    "CAPABILITY_STORAGE_READ",

    "ACTION_LIST",
    "ACTION_GET",

    "OpenStackAdapter",
]
