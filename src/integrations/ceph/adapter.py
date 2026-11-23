from __future__ import annotations

"""
OSHA IntegrationAdapter implementation for Ceph.

The initial Ceph integration deliberately exposes
read-only cluster observation capabilities.
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

from src.integrations.ceph.client import (
    CephClient,
)

from src.integrations.ceph.config import (
    CephConfig,
)

from src.integrations.config import (
    IntegrationsConfig,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationOperationError,
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

CAPABILITY_CLUSTER_READ = (
    "cluster.read"
)

CAPABILITY_HEALTH_READ = (
    "health.read"
)

CAPABILITY_OSD_READ = (
    "osd.read"
)

CAPABILITY_POOL_READ = (
    "pool.read"
)

CAPABILITY_HOST_READ = (
    "host.read"
)

CAPABILITY_MONITOR_READ = (
    "monitor.read"
)

CAPABILITY_SERVICE_READ = (
    "service.read"
)


# ---------------------------------------------------------------
# Actions
# ---------------------------------------------------------------

ACTION_GET = "get"

ACTION_LIST = "list"


# ---------------------------------------------------------------
# Ceph adapter
# ---------------------------------------------------------------

class CephAdapter(
    IntegrationAdapter
):
    """
    Translate OSHA IntegrationOperation objects
    into CephClient read operations.

    Current capabilities:

      cluster.read
      health.read
      osd.read
      pool.read
      host.read
      monitor.read
      service.read

    No mutating capability is advertised.
    """

    def __init__(
        self,
        *,
        ceph_config: CephConfig,
        config: Optional[
            IntegrationsConfig
        ] = None,
    ) -> None:
        self.ceph_config = (
            ceph_config
        )

        self.ceph_config.validate()

        super().__init__(
            target=(
                self.ceph_config.
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
        capabilities = [
            # -------------------------------------------------
            # Cluster summary
            # -------------------------------------------------

            IntegrationCapability(
                name=(
                    CAPABILITY_CLUSTER_READ
                ),

                resource="cluster",

                actions=[
                    ACTION_GET,
                ],

                read_only=True,

                attributes={
                    "provider": "ceph",
                    "api": (
                        "dashboard"
                    ),
                },
            )
        ]

        # -----------------------------------------------------
        # Health
        # -----------------------------------------------------

        if (
            self.ceph_config.
            enable_health
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_HEALTH_READ
                    ),

                    resource="health",

                    actions=[
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "ceph",
                        "api": (
                            "dashboard"
                        ),
                    },
                )
            )

        # -----------------------------------------------------
        # OSDs
        # -----------------------------------------------------

        if self.ceph_config.enable_osd:
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_OSD_READ
                    ),

                    resource="osd",

                    actions=[
                        ACTION_LIST,
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "ceph",
                        "api": (
                            "dashboard"
                        ),
                    },
                )
            )

        # -----------------------------------------------------
        # Pools
        # -----------------------------------------------------

        if (
            self.ceph_config.
            enable_pools
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_POOL_READ
                    ),

                    resource="pool",

                    actions=[
                        ACTION_LIST,
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "ceph",
                        "api": (
                            "dashboard"
                        ),
                    },
                )
            )

        # -----------------------------------------------------
        # Hosts
        # -----------------------------------------------------

        if (
            self.ceph_config.
            enable_hosts
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_HOST_READ
                    ),

                    resource="host",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "ceph",
                        "api": (
                            "dashboard"
                        ),
                    },
                )
            )

        # -----------------------------------------------------
        # Monitors
        # -----------------------------------------------------

        if (
            self.ceph_config.
            enable_monitors
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_MONITOR_READ
                    ),

                    resource="monitor",

                    actions=[
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "ceph",
                        "api": (
                            "dashboard"
                        ),
                    },
                )
            )

        # -----------------------------------------------------
        # Services
        # -----------------------------------------------------

        if (
            self.ceph_config.
            enable_services
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_SERVICE_READ
                    ),

                    resource="service",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "ceph",
                        "api": (
                            "dashboard"
                        ),
                    },
                )
            )

        return capabilities

    # ---------------------------------------------------------
    # Connect client
    # ---------------------------------------------------------

    def connect_client(
        self,
        credential: Optional[
            SecretValue
        ],
    ) -> CephClient:
        client = CephClient(
            config=(
                self.ceph_config
            ),

            credential=credential,
        )

        client.connect()

        return client

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close_client(
        self,
        client: CephClient,
    ) -> None:
        client.close()

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping_client(
        self,
        client: CephClient,
    ) -> bool:
        return client.ping()

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    def execute_operation(
        self,
        client: CephClient,
        operation: IntegrationOperation,
    ) -> Any:
        """
        Translate one validated OSHA operation
        into a bounded Ceph REST API read.

        IntegrationAdapter.dispatch() performs
        target/capability/security validation
        before this method executes.
        """

        operation.finalize()

        capability = (
            operation.capability
        )

        action = (
            operation.action
        )

        # -----------------------------------------------------
        # Cluster summary
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_CLUSTER_READ
            and action
            == ACTION_GET
        ):
            return {
                "cluster": (
                    client.
                    cluster_summary()
                )
            }

        # -----------------------------------------------------
        # Cluster health
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_HEALTH_READ
            and action
            == ACTION_GET
        ):
            return {
                "health": (
                    client.health()
                )
            }

        # -----------------------------------------------------
        # OSDs
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_OSD_READ
        ):
            if action == ACTION_LIST:
                return {
                    "osds": (
                        client.list_osds()
                    )
                }

            if action == ACTION_GET:
                resource_id = (
                    self._require_resource_id(
                        operation,
                        capability=(
                            capability
                        ),
                        action=action,
                    )
                )

                return {
                    "osd": (
                        client.get_osd(
                            resource_id
                        )
                    )
                }

        # ----------------------------------------------------
        # Pools
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_POOL_READ
        ):
            if action == ACTION_LIST:
                return {
                    "pools": (
                        client.
                        list_pools()
                    )
                }

            if action == ACTION_GET:
                resource_id = (
                    self._require_resource_id(
                        operation,
                        capability=(
                            capability
                        ),
                        action=action,
                    )
                )

                return {
                    "pool": (
                        client.get_pool(
                            resource_id
                        )
                    )
                }

        # -----------------------------------------------------
        # Hosts
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_HOST_READ
            and action
            == ACTION_LIST
        ):
            return {
                "hosts": (
                    client.list_hosts()
                )
            }

        # -----------------------------------------------------
        # Monitors
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_MONITOR_READ
            and action
            == ACTION_GET
        ):
            return {
                "monitors": (
                    client.monitors()
                )
            }

        # -----------------------------------------------------
        # Services
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_SERVICE_READ
            and action
            == ACTION_LIST
        ):
            return {
                "services": (
                    client.
                    list_services()
                )
            }

        # -----------------------------------------------------
        # Unsupported
        # -----------------------------------------------------

        raise (
            IntegrationCapabilityError(
                "Unsupported Ceph "
                "integration operation: "
                f"{capability}/{action}"
            )
        )

    # ---------------------------------------------------------
    # Required resource ID
    # ---------------------------------------------------------

    @staticmethod
    def _require_resource_id(
        operation: IntegrationOperation,
        *,
        capability: str,
        action: str,
    ) -> str:
        if not operation.resource_id:
            raise (
                IntegrationOperationError(
                    f"{capability}/{action} "
                    "requires resource_id."
                )
            )

        return operation.resource_id

    # ---------------------------------------------------------
    # Provider summary
    # ---------------------------------------------------------

    def provider_summary(
        self,
    ) -> Dict[str, Any]:
        client_summary = None

        if self.ready:
            client = self.client

            if isinstance(
                client,
                CephClient,
            ):
                client_summary = (
                    client.summary()
                )

        return {
            "provider": "ceph",

            "environment_id": (
                self.target.
                environment_id
            ),

            "target_id": (
                self.target.target_id
            ),

            "api_url": (
                self.ceph_config.
                normalized_api_url
            ),

            "api_version": (
                self.ceph_config.
                api_version
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
    "CAPABILITY_CLUSTER_READ",
    "CAPABILITY_HEALTH_READ",
    "CAPABILITY_OSD_READ",
    "CAPABILITY_POOL_READ",
    "CAPABILITY_HOST_READ",
    "CAPABILITY_MONITOR_READ",
    "CAPABILITY_SERVICE_READ",

    "ACTION_GET",
    "ACTION_LIST",

    "CephAdapter",
]
