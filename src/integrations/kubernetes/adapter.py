from __future__ import annotations

"""
OSHA IntegrationAdapter implementation for
Kubernetes.

This initial provider adapter deliberately
advertises read-only capabilities.
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

from src.integrations.kubernetes.client import (
    KubernetesClient,
)

from src.integrations.kubernetes.config import (
    KubernetesConfig,
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

CAPABILITY_WORKLOAD_READ = (
    "workload.read"
)

CAPABILITY_NODE_READ = (
    "node.read"
)

CAPABILITY_NAMESPACE_READ = (
    "namespace.read"
)

CAPABILITY_SERVICE_READ = (
    "service.read"
)

CAPABILITY_DEPLOYMENT_READ = (
    "deployment.read"
)

CAPABILITY_NETWORKING_READ = (
    "networking.read"
)


# ---------------------------------------------------------------
# Actions
# ---------------------------------------------------------------

ACTION_LIST = "list"

ACTION_GET = "get"


# ---------------------------------------------------------------
# Kubernetes adapter
# ---------------------------------------------------------------

class KubernetesAdapter(
    IntegrationAdapter
):
    """
    Translate OSHA IntegrationOperation objects
    into KubernetesClient read operations.

    Current read capabilities:

      workload.read
      node.read
      namespace.read
      service.read
      deployment.read
      networking.read
    """

    def __init__(
        self,
        *,
        kubernetes_config: KubernetesConfig,
        config: Optional[
            IntegrationsConfig
        ] = None,
    ) -> None:
        self.kubernetes_config = (
            kubernetes_config
        )

        self.kubernetes_config.validate()

        super().__init__(
            target=(
                self.kubernetes_config.
                target()
            ),

            config=(
                config
                or IntegrationsConfig()
            ),
        )

    # ---------------------------------------------------------
    # Advertise capabilities
    # ---------------------------------------------------------

    def define_capabilities(
        self,
    ) -> Iterable[
        IntegrationCapability
    ]:
        capabilities = []

        # -----------------------------------------------------
        # CoreV1Api-backed resources
        # -----------------------------------------------------

        if (
            self.kubernetes_config.
            enable_core
        ):
            capabilities.extend(
                [
                    IntegrationCapability(
                        name=(
                            CAPABILITY_WORKLOAD_READ
                        ),

                        resource="pod",

                        actions=[
                            ACTION_LIST,
                            ACTION_GET,
                        ],

                        read_only=True,

                        attributes={
                            "provider": (
                                "kubernetes"
                            ),
                            "api_group": (
                                "core/v1"
                            ),
                        },
                    ),

                    IntegrationCapability(
                        name=(
                            CAPABILITY_NODE_READ
                        ),

                        resource="node",

                        actions=[
                            ACTION_LIST,
                            ACTION_GET,
                        ],

                        read_only=True,

                        attributes={
                            "provider": (
                                "kubernetes"
                            ),
                            "api_group": (
                                "core/v1"
                            ),
                        },
                    ),

                    IntegrationCapability(
                        name=(
                            CAPABILITY_NAMESPACE_READ
                        ),

                        resource="namespace",

                        actions=[
                            ACTION_LIST,
                        ],

                        read_only=True,

                        attributes={
                            "provider": (
                                "kubernetes"
                            ),
                            "api_group": (
                                "core/v1"
                            ),
                        },
                    ),

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
                            "provider": (
                                "kubernetes"
                            ),
                            "api_group": (
                                "core/v1"
                            ),
                        },
                    ),
                ]
            )

        # -----------------------------------------------------
        # AppsV1Api
        # -----------------------------------------------------

        if (
            self.kubernetes_config.
            enable_apps
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_DEPLOYMENT_READ
                    ),

                    resource="deployment",

                    actions=[
                        ACTION_LIST,
                        ACTION_GET,
                    ],

                    read_only=True,

                    attributes={
                        "provider": (
                            "kubernetes"
                        ),
                        "api_group": (
                            "apps/v1"
                        ),
                    },
                )
            )

        # -----------------------------------------------------
        # NetworkingV1Api
        # -----------------------------------------------------

        if (
            self.kubernetes_config.
            enable_networking
        ):
            capabilities.append(
                IntegrationCapability(
                    name=(
                        CAPABILITY_NETWORKING_READ
                    ),

                    resource="ingress",

                    actions=[
                        ACTION_LIST,
                    ],

                    read_only=True,

                    attributes={
                        "provider": (
                            "kubernetes"
                        ),
                        "api_group": (
                            "networking.k8s.io/v1"
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
    ) -> KubernetesClient:
        client = KubernetesClient(
            config=(
                self.kubernetes_config
            ),

            credential=credential,
        )

        client.connect()

        return client

    # ---------------------------------------------------------
    # Close client
    # ---------------------------------------------------------

    def close_client(
        self,
        client: KubernetesClient,
    ) -> None:
        client.close()

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping_client(
        self,
        client: KubernetesClient,
    ) -> bool:
        return client.ping()

    # ---------------------------------------------------------
    # Execute operation
    # ---------------------------------------------------------

    def execute_operation(
        self,
        client: KubernetesClient,
        operation: IntegrationOperation,
    ) -> Any:
        """
        Translate a validated IntegrationOperation
        into one KubernetesClient call.

        IntegrationAdapter.dispatch() has already
        checked:

          provider/environment/target
          advertised capability
          supported action
          read/mutation policy
        """

        operation.finalize()

        capability = (
            operation.capability
        )

        action = (
            operation.action
        )

        # Namespace can be supplied as a bounded
        # operation parameter. If omitted,
        # resource get calls fall back to the
        # configured default namespace.
        namespace = (
            operation.parameters.get(
                "namespace"
            )
        )

        if namespace is not None:
            namespace = str(
                namespace
            ).strip()

            if not namespace:
                namespace = None

        # -----------------------------------------------------
        # Pods / workloads
        # -----------------------------------------------------

        if (
            capability
            == CAPABILITY_WORKLOAD_READ
        ):
            if action == ACTION_LIST:
                return {
                    "pods": (
                        client.list_pods(
                            namespace=(
                                namespace
                            )
                        )
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
                    "pod": (
                        client.get_pod(
                            name=resource_id,
                            namespace=(
                                namespace
                            ),
                        )
                    )
                }

        # -----------------------------------------------------
        # Nodes
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_NODE_READ
        ):
            if action == ACTION_LIST:
                return {
                    "nodes": (
                        client.list_nodes()
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
                    "node": (
                        client.get_node(
                            name=resource_id
                        )
                    )
                }

        # -----------------------------------------------------
        # Namespaces
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_NAMESPACE_READ
        ):
            if action == ACTION_LIST:
                return {
                    "namespaces": (
                        client.
                        list_namespaces()
                    )
                }

        # -----------------------------------------------------
        # Services
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_SERVICE_READ
        ):
            if action == ACTION_LIST:
                return {
                    "services": (
                        client.
                        list_services(
                            namespace=(
                                namespace
                            )
                        )
                    )
                }

        # -----------------------------------------------------
        # Deployments
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_DEPLOYMENT_READ
        ):
            if action == ACTION_LIST:
                return {
                    "deployments": (
                        client.
                        list_deployments(
                            namespace=(
                                namespace
                            )
                        )
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
                    "deployment": (
                        client.
                        get_deployment(
                            name=resource_id,
                            namespace=(
                                namespace
                            ),
                        )
                    )
                }

        # -----------------------------------------------------
        # Networking / ingress
        # -----------------------------------------------------

        elif (
            capability
            == CAPABILITY_NETWORKING_READ
        ):
            if action == ACTION_LIST:
                return {
                    "ingresses": (
                        client.
                        list_ingresses(
                            namespace=(
                                namespace
                            )
                        )
                    )
                }

        # -----------------------------------------------------
        # Unsupported
        # -----------------------------------------------------

        raise (
            IntegrationCapabilityError(
                "Unsupported Kubernetes "
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

        return (
            operation.resource_id
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
                KubernetesClient,
            ):
                client_summary = (
                    client.summary()
                )

        return {
            "provider": (
                "kubernetes"
            ),

            "environment_id": (
                self.target.
                environment_id
            ),

            "target_id": (
                self.target.target_id
            ),

            "configuration_mode": (
                self.kubernetes_config.
                auth.mode
            ),

            "default_namespace": (
                self.kubernetes_config.
                default_namespace
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
    "CAPABILITY_WORKLOAD_READ",
    "CAPABILITY_NODE_READ",
    "CAPABILITY_NAMESPACE_READ",
    "CAPABILITY_SERVICE_READ",
    "CAPABILITY_DEPLOYMENT_READ",
    "CAPABILITY_NETWORKING_READ",

    "ACTION_LIST",
    "ACTION_GET",

    "KubernetesAdapter",
]
