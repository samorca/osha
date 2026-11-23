from __future__ import annotations

"""
Kubernetes Python client boundary for OSHA.

Provider-specific Kubernetes API objects remain
inside this module and kubernetes/adapter.py.
"""

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationConnectionError,
    IntegrationOperationError,
    IntegrationResourceNotFoundError,
    wrap_provider_error,
)

from src.integrations.kubernetes.config import (
    KubernetesConfig,
    KubernetesConfigMode,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Optional Kubernetes client import
# ---------------------------------------------------------------

try:
    from kubernetes import (
        client as kubernetes_client,
    )

    from kubernetes import (
        config as kubernetes_config,
    )

except ImportError:
    kubernetes_client = None
    kubernetes_config = None


# ---------------------------------------------------------------
# Client error
# ---------------------------------------------------------------

class KubernetesClientError(
    IntegrationOperationError
):
    default_code = (
        "kubernetes_client_error"
    )


# ---------------------------------------------------------------
# Kubernetes object serialization
# ---------------------------------------------------------------

def resource_to_dict(
    resource: Any,
) -> Dict[str, Any]:
    """
    Convert a Kubernetes API resource to an
    ordinary dictionary.

    ApiClient.sanitize_for_serialization is used
    when the Kubernetes package is available.
    """

    if resource is None:
        return {}

    if isinstance(
        resource,
        dict,
    ):
        return dict(
            resource
        )

    if kubernetes_client is not None:
        try:
            value = (
                kubernetes_client.
                ApiClient().
                sanitize_for_serialization(
                    resource
                )
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        except Exception:
            pass

    # Narrow fallback.
    result: Dict[
        str,
        Any,
    ] = {}

    metadata = getattr(
        resource,
        "metadata",
        None,
    )

    if metadata is not None:
        name = getattr(
            metadata,
            "name",
            None,
        )

        namespace = getattr(
            metadata,
            "namespace",
            None,
        )

        uid = getattr(
            metadata,
            "uid",
            None,
        )

        if name is not None:
            result[
                "name"
            ] = name

        if namespace is not None:
            result[
                "namespace"
            ] = namespace

        if uid is not None:
            result[
                "uid"
            ] = uid

    return result


def resources_to_dicts(
    resources: Iterable[
        Any
    ],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    return [
        resource_to_dict(
            resource
        )
        for resource
        in resources
    ]


# ---------------------------------------------------------------
# Kubernetes client
# ---------------------------------------------------------------

class KubernetesClient:
    """
    Thin GPIO-free wrapper around the official
    Kubernetes Python client.

    It exposes only the operations that OSHA's
    KubernetesAdapter deliberately chooses to
    support.
    """

    def __init__(
        self,
        *,
        config: KubernetesConfig,
        credential: Optional[
            SecretValue
        ] = None,
    ) -> None:
        self.config = config

        self.config.validate()

        self.credential = (
            credential
        )

        self._api_client: Any = None

        self._core: Any = None

        self._apps: Any = None

        self._networking: Any = None

        self._batch: Any = None

    # ---------------------------------------------------------
    # Connected
    # ---------------------------------------------------------

    @property
    def connected(
        self,
    ) -> bool:
        return (
            self._api_client
            is not None
        )

    # ---------------------------------------------------------
    # API accessors
    # ---------------------------------------------------------

    @property
    def core(
        self,
    ) -> Any:
        self._require_connected()

        if self._core is None:
            raise (
                IntegrationCapabilityError(
                    "Kubernetes CoreV1Api "
                    "is disabled."
                )
            )

        return self._core

    @property
    def apps(
        self,
    ) -> Any:
        self._require_connected()

        if self._apps is None:
            raise (
                IntegrationCapabilityError(
                    "Kubernetes AppsV1Api "
                    "is disabled."
                )
            )

        return self._apps

    @property
    def networking(
        self,
    ) -> Any:
        self._require_connected()

        if self._networking is None:
            raise (
                IntegrationCapabilityError(
                    "Kubernetes "
                    "NetworkingV1Api is "
                    "disabled."
                )
            )

        return self._networking

    # ---------------------------------------------------------
    # Connection configuration
    # ---------------------------------------------------------

    def _load_configuration(
        self,
    ) -> Any:
        """
        Load/build one Kubernetes Configuration
        without persisting secret values into
        OSHA configuration objects.
        """

        if (
            kubernetes_client is None
            or kubernetes_config is None
        ):
            raise (
                IntegrationConnectionError(
                    "Kubernetes Python "
                    "client is not "
                    "installed."
                )
            )

        mode = (
            self.config.auth.mode
        )

        # -----------------------------------------------------
        # Kubeconfig
        # -----------------------------------------------------

        if (
            mode
            == KubernetesConfigMode.
            KUBECONFIG.value
        ):
            try:
                kubernetes_config.load_kube_config(
                        config_file=(
                            self.config.
                            auth.
                            kubeconfig_path
                        ),

                        context=(
                            self.config.
                            auth.context
                        ),
                    )

                configuration = (
                    kubernetes_client.
                    Configuration.
                    get_default_copy()
                )

            except Exception as exc:
                raise wrap_provider_error(
                    exc,
                    provider="kubernetes",
                    target_id=(
                        self.config.
                        target_id
                    ),
                    message=(
                        "Kubernetes "
                        "kubeconfig loading "
                        "failed."
                    ),
                    retryable=False,
                    code=(
                        "kubernetes_config_failed"
                    ),
                ) from exc

        # -----------------------------------------------------
        # In-cluster
        # -----------------------------------------------------

        elif (
            mode
            == KubernetesConfigMode.
            IN_CLUSTER.value
        ):
            try:
                kubernetes_config.load_incluster_config()

                configuration = (
                    kubernetes_client.
                    Configuration.
                    get_default_copy()
                )

            except Exception as exc:
                raise wrap_provider_error(
                    exc,
                    provider="kubernetes",
                    target_id=(
                        self.config.
                        target_id
                    ),
                    message=(
                        "Kubernetes "
                        "in-cluster "
                        "configuration "
                        "failed."
                    ),
                    retryable=False,
                    code=(
                        "kubernetes_incluster_"
                        "config_failed"
                    ),
                ) from exc

        # -----------------------------------------------------
        # Direct
        # -----------------------------------------------------

        elif (
            mode
            == KubernetesConfigMode.
            DIRECT.value
        ):
            configuration = (
                kubernetes_client.
                Configuration()
            )

            configuration.host = (
                self.config.auth.
                api_server
            )

            configuration.verify_ssl = (
                self.config.
                verify_tls
            )

            if (
                self.config.auth.
                ca_cert_path
            ):
                configuration.ssl_ca_cert = (
                    self.config.auth.
                    ca_cert_path
                )

            if (
                self.credential
                is not None
            ):
                # Reveal only at the provider
                # client authentication boundary.
                configuration.api_key = {
                    "authorization": (
                        "Bearer "
                        + self.
                        credential.reveal()
                    )
                }

        else:
            raise (
                IntegrationConnectionError(
                    "Unsupported Kubernetes "
                    "configuration mode: "
                    f"{mode}"
                )
            )

        # -----------------------------------------------------
        # OSHA TLS policy
        # -----------------------------------------------------

        configuration.verify_ssl = (
            self.config.verify_tls
        )

        if (
            self.config.auth.
            ca_cert_path
        ):
            configuration.ssl_ca_cert = (
                self.config.auth.
                ca_cert_path
            )

        return configuration

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    def connect(
        self,
    ) -> Any:
        if self.connected:
            return self._api_client

        configuration = (
            self._load_configuration()
        )

        try:
            self._api_client = (
                kubernetes_client.
                ApiClient(
                    configuration
                )
            )

            if (
                self.config.enable_core
            ):
                self._core = (
                    kubernetes_client.
                    CoreV1Api(
                        self._api_client
                    )
                )

            if (
                self.config.enable_apps
            ):
                self._apps = (
                    kubernetes_client.
                    AppsV1Api(
                        self._api_client
                    )
                )

            if (
                self.config.
                enable_networking
            ):
                self._networking = (
                    kubernetes_client.
                    NetworkingV1Api(
                        self._api_client
                    )
                )

            if (
                self.config.enable_batch
            ):
                self._batch = (
                    kubernetes_client.
                    BatchV1Api(
                        self._api_client
                    )
                )

            return self._api_client

        except Exception as exc:
            self._clear()

            raise wrap_provider_error(
                exc,
                provider="kubernetes",
                target_id=(
                    self.config.target_id
                ),
                message=(
                    "Kubernetes API client "
                    "creation failed."
                ),
                retryable=True,
                code=(
                    "kubernetes_connection_failed"
                ),
            ) from exc

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(
        self,
    ) -> None:
        api_client = (
            self._api_client
        )

        if api_client is None:
            return

        try:
            close = getattr(
                api_client,
                "close",
                None,
            )

            if callable(
                close
            ):
                close()

        except Exception as exc:
            raise wrap_provider_error(
                exc,
                provider="kubernetes",
                target_id=(
                    self.config.target_id
                ),
                message=(
                    "Kubernetes client "
                    "close failed."
                ),
                retryable=False,
                code=(
                    "kubernetes_close_failed"
                ),
            ) from exc

        finally:
            self._clear()

    def _clear(
        self,
    ) -> None:
        self._batch = None
        self._networking = None
        self._apps = None
        self._core = None
        self._api_client = None

    # ---------------------------------------------------------
    # Connectivity/readiness
    # ---------------------------------------------------------

    def ping(
        self,
    ) -> bool:
        """
        Make one lightweight read request against
        the API server.
        """

        if not self.connected:
            return False

        try:
            if self._core is not None:
                self._core.list_namespace(
                    limit=1,
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    ),
                )

                return True

            if self._apps is not None:
                self._apps.list_deployment_for_all_namespaces(
                    limit=1,
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    ),
                )

                return True

            return False

        except Exception:
            return False

    # ---------------------------------------------------------
    # Pods
    # ---------------------------------------------------------

    def list_pods(
        self,
        *,
        namespace: Optional[
            str
        ] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        try:
            if namespace:
                response = (
                    self.core.
                    list_namespaced_pod(
                        namespace=namespace,
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        ),
                    )
                )

            else:
                response = (
                    self.core.
                    list_pod_for_all_namespaces(
                        watch=False,
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        ),
                    )
                )

            return resources_to_dicts(
                response.items
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes pod "
                "listing failed.",
            ) from exc

    def get_pod(
        self,
        *,
        name: str,
        namespace: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        namespace = (
            namespace
            or self.config.
            default_namespace
        )

        try:
            value = (
                self.core.
                read_namespaced_pod(
                    name=name,
                    namespace=namespace,
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    ),
                )
            )

            return resource_to_dict(
                value
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes pod lookup "
                "failed.",
            ) from exc

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    def list_nodes(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        try:
            response = (
                self.core.list_node(
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    )
                )
            )

            return resources_to_dicts(
                response.items
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes node "
                "listing failed.",
            ) from exc

    def get_node(
        self,
        *,
        name: str,
    ) -> Dict[str, Any]:
        try:
            value = (
                self.core.read_node(
                    name=name,
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    ),
                )
            )

            return resource_to_dict(
                value
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes node lookup "
                "failed.",
            ) from exc

    # ---------------------------------------------------------
    # Namespaces
    # ---------------------------------------------------------

    def list_namespaces(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        try:
            response = (
                self.core.list_namespace(
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    )
                )
            )

            return resources_to_dicts(
                response.items
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes namespace "
                "listing failed.",
            ) from exc

    # ---------------------------------------------------------
    # Services
    # ---------------------------------------------------------

    def list_services(
        self,
        *,
        namespace: Optional[
            str
        ] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        try:
            if namespace:
                response = (
                    self.core.
                    list_namespaced_service(
                        namespace=namespace,
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        ),
                    )
                )

            else:
                response = (
                    self.core.
                    list_service_for_all_namespaces(
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        )
                    )
                )

            return resources_to_dicts(
                response.items
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes service "
                "listing failed.",
            ) from exc

    # ---------------------------------------------------------
    # Deployments
    # ---------------------------------------------------------

    def list_deployments(
        self,
        *,
        namespace: Optional[
            str
        ] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        try:
            if namespace:
                response = (
                    self.apps.
                    list_namespaced_deployment(
                        namespace=namespace,
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        ),
                    )
                )

            else:
                response = (
                    self.apps.
                    list_deployment_for_all_namespaces(
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        )
                    )
                )

            return resources_to_dicts(
                response.items
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes deployment "
                "listing failed.",
            ) from exc

    def get_deployment(
        self,
        *,
        name: str,
        namespace: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        namespace = (
            namespace
            or self.config.
            default_namespace
        )

        try:
            value = (
                self.apps.
                read_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    _request_timeout=(
                        self.config.
                        request_timeout_seconds
                    ),
                )
            )

            return resource_to_dict(
                value
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes deployment "
                "lookup failed.",
            ) from exc

    # ---------------------------------------------------------
    # Ingresses
    # ---------------------------------------------------------

    def list_ingresses(
        self,
        *,
        namespace: Optional[
            str
        ] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        try:
            if namespace:
                response = (
                    self.networking.
                    list_namespaced_ingress(
                        namespace=namespace,
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        ),
                    )
                )

            else:
                response = (
                    self.networking.
                    list_ingress_for_all_namespaces(
                        _request_timeout=(
                            self.config.
                            request_timeout_seconds
                        )
                    )
                )

            return resources_to_dicts(
                response.items
            )

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Kubernetes ingress "
                "listing failed.",
            ) from exc

    # ---------------------------------------------------------
    # Connection guard
    # ---------------------------------------------------------

    def _require_connected(
        self,
    ) -> None:
        if not self.connected:
            raise (
                IntegrationConnectionError(
                    "Kubernetes client "
                    "is not connected."
                )
            )

    # ---------------------------------------------------------
    # Error translation
    # ---------------------------------------------------------

    def _operation_error(
        self,
        error: BaseException,
        message: str,
    ) -> KubernetesClientError:
        wrapped = wrap_provider_error(
            error,
            provider="kubernetes",
            target_id=(
                self.config.target_id
            ),
            message=message,
            retryable=False,
            code=(
                "kubernetes_operation_failed"
            ),
        )

        return KubernetesClientError(
            message,
            code=wrapped.code,
            retryable=(
                wrapped.retryable
            ),
            details=(
                wrapped.details
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connected": (
                self.connected
            ),

            "target_id": (
                self.config.target_id
            ),

            "configuration_mode": (
                self.config.auth.mode
            ),

            "default_namespace": (
                self.config.
                default_namespace
            ),

            "api_groups": {
                "core": (
                    self._core
                    is not None
                ),

                "apps": (
                    self._apps
                    is not None
                ),

                "networking": (
                    self._networking
                    is not None
                ),

                "batch": (
                    self._batch
                    is not None
                ),
            },

            "credential_value_exposed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "KubernetesClientError",

    "resource_to_dict",
    "resources_to_dicts",

    "KubernetesClient",
]
