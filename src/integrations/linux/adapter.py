from __future__ import annotations

"""
OSHA IntegrationAdapter implementation for Linux.

This initial adapter exposes read-only host and
system observation capabilities.

It does not provide arbitrary shell execution.
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

from src.integrations.linux.client import (
    LinuxClient,
)

from src.integrations.linux.config import (
    LinuxConfig,
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

CAPABILITY_HOST_READ = (
    "host.read"
)

CAPABILITY_CPU_READ = (
    "cpu.read"
)

CAPABILITY_MEMORY_READ = (
    "memory.read"
)

CAPABILITY_DISK_READ = (
    "disk.read"
)

CAPABILITY_NETWORK_READ = (
    "network.read"
)

CAPABILITY_PROCESS_READ = (
    "process.read"
)

CAPABILITY_SYSTEM_SNAPSHOT = (
    "system.snapshot"
)


# ---------------------------------------------------------------
# Actions
# ---------------------------------------------------------------

ACTION_GET = "get"

ACTION_LIST = "list"

ACTION_SNAPSHOT = "snapshot"


# ---------------------------------------------------------------
# Linux adapter
# ---------------------------------------------------------------

class LinuxAdapter(
    IntegrationAdapter
):
    """
    Translate OSHA IntegrationOperation objects
    into read-only LinuxClient operations.

    Current capabilities:

      host.read
      cpu.read
      memory.read
      disk.read
      network.read
      process.read
      system.snapshot
    """

    def __init__(
        self,
        *,
        linux_config: LinuxConfig,
        config: Optional[
            IntegrationsConfig
        ] = None,
    ) -> None:
        self.linux_config = (
            linux_config
        )

        self.linux_config.validate()

        super().__init__(
            target=(
                self.linux_config.
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
            IntegrationCapability(
                name=(
                    CAPABILITY_HOST_READ
                ),

                resource="host",

                actions=[
                    ACTION_GET,
                ],

                read_only=True,

                attributes={
                    "provider": "linux",
                    "source": (
                        "local_system"
                    ),
                },
            )
        ]

        # -----------------------------------------------------
        # CPU
        # -----------------------------------------------------

        if self.linux_config.include_cpu:
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_CPU_READ
                    ),

                    resource="cpu",

                    actions=[
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "linux",
                        "source": "psutil",
                    },
                )
            )

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        if (
            self.linux_config.
            include_memory
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_MEMORY_READ
                    ),

                    resource="memory",

                    actions=[
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "linux",
                        "source": "psutil",
                    },
                )
            )

        # -----------------------------------------------------
        # Disk
        # -----------------------------------------------------

        if self.linux_config.include_disks:
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_DISK_READ
                    ),

                    resource="disk",

                    actions=[
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "linux",
                        "source": "psutil",
                    },
                )
            )

        # -----------------------------------------------------
        # Network
        # -----------------------------------------------------

        if (
            self.linux_config.
            include_network
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_NETWORK_READ
                    ),

                    resource="network",

                    actions=[
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "linux",
                        "source": "psutil",
                    },
                )
            )

        # -----------------------------------------------------
        # Processes
        # -----------------------------------------------------

        if (
            self.linux_config.
            include_processes
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_PROCESS_READ
                    ),

                    resource="process",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": "linux",
                        "source": "psutil",
                    },
                )
            )

        # -----------------------------------------------------
        # Combined snapshot
        # -----------------------------------------------------

        capabilities.append(
            IntegrationCapability(
                name=(
                    CAPABILITY_SYSTEM_SNAPSHOT
                ),

                resource="system",

                actions=[
                    ACTION_SNAPSHOT,
                ],

                read_only=True,

                attributes={
                    "provider": "linux",

                    "source": (
                        "local_system"
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
    ) -> LinuxClient:
        """
        LinuxClient currently supports LOCAL
        mode only.

        credential is accepted for consistency
        with IntegrationAdapter but isn't used by
        the local client.
        """

        client = LinuxClient(
            config=(
                self.linux_config
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
        client: LinuxClient,
    ) -> None:
        client.close()

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping_client(
        self,
        client: LinuxClient,
    ) -> bool:
        return client.ping()

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    def execute_operation(
        self,
        client: LinuxClient,
        operation: IntegrationOperation,
    ) -> Any:
        """
        Translate one validated OSHA operation
        into a LinuxClient read.

        No command string is accepted by this
        method.
        """

        operation.finalize()

        capability = (
            operation.capability
        )

        action = (
            operation.action
        )

        # -----------------------------------------------------
        # Host
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_HOST_READ
            and action
            == ACTION_GET
        ):
            return {
                "host": (
                    client.host_info()
                )
            }

        # -----------------------------------------------------
        # CPU
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_CPU_READ
             and action
            == ACTION_GET
        ):
            return {
                "cpu": (
                    client.cpu_info()
                )
            }

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_MEMORY_READ
            and action
            == ACTION_GET
        ):
            return {
                "memory": (
                    client.memory_info()
                )
            }

        # -----------------------------------------------------
        # Disk
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_DISK_READ
            and action
            == ACTION_GET
        ):
            return {
                "disks": (
                    client.disk_info()
                )
            }

        # -----------------------------------------------------
        # Network
        # -----------------------------------------------------

        if (
            capability            == CAPABILITY_NETWORK_READ
            and action
            == ACTION_GET
        ):
            return {
                "network": (
                    client.network_info()
                )
            }

        # -----------------------------------------------------
        # Processes
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_PROCESS_READ
            and action
            == ACTION_LIST
        ):
            limit = (
                operation.parameters.get(
                    "limit"
                )
            )

            if limit is not None:
                try:
                    limit = int(
                        limit
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:
                    raise (
                        IntegrationOperationError(
                            "Process limit must "
                            "be an integer."
                        )
                    ) from exc

                if limit < 1:
                    raise (
                        IntegrationOperationError(
                            "Process limit must "
                            "be >= 1."
                        )
                    )

            return {
                "processes": (
                    client.processes(
                        limit=limit
                    )
                )
            }

        # -----------------------------------------------------
        # Full configured snapshot
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_SYSTEM_SNAPSHOT
            and action
            == ACTION_SNAPSHOT
        ):
            process_limit = (
                operation.parameters.get(
                    "process_limit"
                )
            )

            if process_limit is not None:
                try:
                    process_limit = int(
                        process_limit
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:
                    raise (
                        IntegrationOperationError(
                            "process_limit "
                            "must be an "
                            "integer."
                        )
                    ) from exc

                if process_limit < 1:
                    raise (
                        IntegrationOperationError(
                            "process_limit "
                            "must be >= 1."
                        )
                    )

            return {
                "snapshot": (
                    client.snapshot(
                        process_limit=(
                            process_limit
                        )
                    )
                )
            }

        # -----------------------------------------------------
        # Unsupported
        # -----------------------------------------------------

        raise (
            IntegrationCapabilityError(
                "Unsupported Linux "
                "integration operation: "
                f"{capability}/{action}"
            )
        )

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
                LinuxClient,
            ):
                client_summary = (
                    client.summary()
                )

        return {
            "provider": "linux",

            "environment_id": (
                self.target.
                environment_id
            ),

            "target_id": (
                self.target.target_id
            ),

            "mode": (
                self.linux_config.
                connection.mode
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

            "arbitrary_command_execution_enabled": (
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
    "CAPABILITY_HOST_READ",
    "CAPABILITY_CPU_READ",
    "CAPABILITY_MEMORY_READ",
    "CAPABILITY_DISK_READ",
    "CAPABILITY_NETWORK_READ",
    "CAPABILITY_PROCESS_READ",
    "CAPABILITY_SYSTEM_SNAPSHOT",

    "ACTION_GET",
    "ACTION_LIST",
    "ACTION_SNAPSHOT",

    "LinuxAdapter",
]
