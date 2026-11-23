from __future__ import annotations

"""
Provider-neutral integration contracts for OSHA
infrastructure adapters.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.core.identifiers import (
    random_id,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    environment_id,
    required_string,
    unique_strings,
)

from src.security.schema import (
    SecretReference,
)


# ---------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------

INTEGRATION_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Provider kind
# ---------------------------------------------------------------

class ProviderKind(
    str,
    Enum,
):
    OPENSTACK = "openstack"

    KUBERNETES = "kubernetes"

    LINUX = "linux"

    CEPH = "ceph"

    MINIO = "minio"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Integration lifecycle state
# ---------------------------------------------------------------

class IntegrationState(
    str,
    Enum,
):
    CREATED = "created"

    CONNECTING = "connecting"

    READY = "ready"

    DEGRADED = "degraded"

    FAILED = "failed"

    CLOSING = "closing"

    CLOSED = "closed"


# ---------------------------------------------------------------
# Integration health
# ---------------------------------------------------------------

class IntegrationHealthStatus(
    str,
    Enum,
):
    UNKNOWN = "unknown"

    HEALTHY = "healthy"

    DEGRADED = "degraded"

    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------
# Operation status
# ---------------------------------------------------------------

class OperationStatus(
    str,
    Enum,
):
    PENDING = "pending"

    RUNNING = "running"

    SUCCEEDED = "succeeded"

    FAILED = "failed"

    BLOCKED = "blocked"

    CANCELLED = "cancelled"


# ---------------------------------------------------------------
# Integration target
# ---------------------------------------------------------------

@dataclass(frozen=True)
class IntegrationTarget:
    """
    Identifies one external infrastructure target.

    Examples:

      OpenStack cloud
      Kubernetes cluster
      Linux host
      Ceph cluster
      MinIO deployment

    credential is a SecretReference only.

    No password/token/private key belongs in this
    object.
    """

    provider: str

    environment_id: str

    target_id: str

    endpoint: Optional[
        str
    ] = None

    region: Optional[
        str
    ] = None

    namespace: Optional[
        str
    ] = None

    credential: Optional[
        SecretReference
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "IntegrationTarget":
        if (
            self.provider
            not in {
                item.value
                for item
                in ProviderKind
            }
        ):
            raise ValueError(
                "Unknown integration "
                f"provider: {self.provider}"
            )

        environment_id(
            self.environment_id
        )

        required_string(
            self.target_id,
            name="target_id",
            maximum_length=512,
        )

        if self.endpoint is not None:
            required_string(
                self.endpoint,
                name="endpoint",
                maximum_length=2048,
            )

        if self.region is not None:
            required_string(
                self.region,
                name="region",
                maximum_length=256,
            )

        if self.namespace is not None:
            required_string(
                self.namespace,
                name="namespace",
                maximum_length=256,
            )

        if self.credential is not None:
            self.credential.finalize()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "provider": (
                self.provider
            ),

            "environment_id": (
                self.environment_id
            ),

            "target_id": (
                self.target_id
            ),

            "endpoint": (
                self.endpoint
            ),

            "region": (
                self.region
            ),

            "namespace": (
                self.namespace
            ),

            "credential": (
                self.credential.
                to_dict()
                if self.credential
                else None
            ),

            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Integration capability
# ---------------------------------------------------------------

@dataclass(frozen=True)
class IntegrationCapability:
    """
    Describes a capability exposed by an adapter.

    Examples could include conceptual capabilities
    such as:

      compute.read
      compute.manage
      storage.read
      network.read
      workload.read

    Exact provider capability catalogs will be
    defined by provider-specific adapters.
    """

    name: str

    resource: str

    actions: List[
        str
    ] = field(
        default_factory=list
    )

    read_only: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "IntegrationCapability":
        required_string(
            self.name,
            name="capability name",
            maximum_length=256,
        )

        required_string(
            self.resource,
            name="capability resource",
            maximum_length=256,
        )

        object.__setattr__(
            self,
            "actions",
            unique_strings(
                self.actions
            ),
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "name": self.name,

            "resource": (
                self.resource
            ),

            "actions": list(
                self.actions
            ),

            "read_only": (
                self.read_only
            ),

            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Integration status
# ---------------------------------------------------------------

@dataclass
class IntegrationStatus:
    """
    Connection/lifecycle status for one
    IntegrationTarget.
    """

    target: IntegrationTarget

    state: str = (
        IntegrationState.CREATED.value
    )

    integration_id: Optional[
        str
    ] = None

    connected_at: Optional[
        str
    ] = None

    closed_at: Optional[
        str
    ] = None

    last_error_type: Optional[
        str
    ] = None

    capabilities: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "IntegrationStatus":
        self.target.finalize()

        if (
            self.state
            not in {
                item.value
                for item
                in IntegrationState
            }
        ):
            raise ValueError(
                "Unknown integration state: "
                f"{self.state}"
            )

        self.capabilities = (
            unique_strings(
                self.capabilities
            )
        )

        if not self.integration_id:
            self.integration_id = (
                random_id(
                    "integration:connection"
                )
            )

        self.updated_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                INTEGRATION_SCHEMA_VERSION
            ),

            "integration_id": (
                self.integration_id
            ),

            "target": (
                self.target.to_dict()
            ),

            "state": self.state,

            "connected_at": (
                self.connected_at
            ),

            "closed_at": (
                self.closed_at
            ),

            "last_error_type": (
                self.last_error_type
            ),

            "capabilities": (
                list(
                    self.capabilities
                )
            ),

            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),

            "updated_at": (
                self.updated_at
            ),
        }


# ---------------------------------------------------------------
# Integration operation
# ---------------------------------------------------------------

@dataclass
class IntegrationOperation:
    """
    Provider-neutral request to an infrastructure
    integration.

    This describes the requested operation.

    Authorization and remediation safety checks
    must happen before mutating operations reach
    the adapter.
    """

    target: IntegrationTarget

    capability: str

    action: str

    resource_id: Optional[
        str
    ] = None

    parameters: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    requested_at: str = field(
        default_factory=utc_now
    )

    operation_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "IntegrationOperation":
        self.target.finalize()

        required_string(
            self.capability,
            name="capability",
            maximum_length=256,
        )

        required_string(
            self.action,
            name="action",
            maximum_length=128,
        )

        if (
            self.resource_id
            is not None
        ):
            required_string(
                self.resource_id,
                name="resource_id",
                maximum_length=1024,
            )

        self.reference_ids = (
            unique_strings(
                self.reference_ids
            )
        )

        if not self.operation_id:
            self.operation_id = (
                random_id(
                    "integration:operation"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                INTEGRATION_SCHEMA_VERSION
            ),

            "operation_id": (
                self.operation_id
            ),

            "target": (
                self.target.to_dict()
            ),

            "capability": (
                self.capability
            ),

            "action": self.action,

            "resource_id": (
                self.resource_id
            ),

            "parameters": (
                to_json_safe(
                    self.parameters
                )
            ),

            "reference_ids": (
                list(
                    self.reference_ids
                )
            ),

            "requested_at": (
                self.requested_at
            ),
        }


# ---------------------------------------------------------------
# Integration result
# ---------------------------------------------------------------

@dataclass
class IntegrationResult:
    """
    Bounded result produced by an integration
    operation.

    Raw provider exceptions should not be placed
    in this structure.
    """

    operation_id: str

    status: str

    value: Any = None

    error_type: Optional[
        str
    ] = None

    retryable: bool = False

    reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    completed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "IntegrationResult":
        required_string(
            self.operation_id,
            name="operation_id",
            maximum_length=512,
        )

        if (
            self.status
            not in {
                item.value
                for item
                in OperationStatus
            }
        ):
            raise ValueError(
                "Unknown integration "
                "operation status: "
                f"{self.status}"
            )

        self.reference_ids = (
            unique_strings(
                self.reference_ids
            )
        )

        return self

    @property
    def success(
        self,
    ) -> bool:
        return (
            self.status
            == OperationStatus.
            SUCCEEDED.value
        )

    def to_dict(
        self,
        *,
        include_value: bool = True,
    ) -> Dict[str, Any]:
        self.finalize()

        result = {
            "schema_version": (
                INTEGRATION_SCHEMA_VERSION
            ),

            "operation_id": (
                self.operation_id
            ),

            "status": self.status,

            "success": (
                self.success
            ),

            "error_type": (
                self.error_type
            ),

            "retryable": (
                self.retryable
            ),

            "reference_ids": (
                list(
                    self.reference_ids
                )
            ),

            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),

            "completed_at": (
                self.completed_at
            ),
        }

        if include_value:
            result[
                "value"
            ] = to_json_safe(
                self.value
            )

        return result


# ---------------------------------------------------------------
# Integration health
# ---------------------------------------------------------------

@dataclass
class IntegrationHealth:
    """
    Operational health of an OSHA integration
    connection to its configured target.

    This reports whether the adapter can
    communicate with the target.

    Broader infrastructure health still belongs
    to collectors/telemetry/intelligence.
    """

    target: IntegrationTarget

    status: str

    integration_id: Optional[
        str
    ] = None

    latency_seconds: Optional[
        float
    ] = None

    message: Optional[
        str
    ] = None

    error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    observed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "IntegrationHealth":
        self.target.finalize()

        if (
            self.status
            not in {
                item.value
                for item
                in IntegrationHealthStatus
            }
        ):
            raise ValueError(
                "Unknown integration "
                f"health status: {self.status}"
            )

        if (
            self.latency_seconds
            is not None
            and self.latency_seconds
            < 0
        ):
            raise ValueError(
                "latency_seconds "
                "must be >= 0"
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                INTEGRATION_SCHEMA_VERSION
            ),

            "target": (
                self.target.to_dict()
            ),

            "status": self.status,

            "integration_id": (
                self.integration_id
            ),

            "latency_seconds": (
                self.latency_seconds
            ),

            "message": self.message,

            "error_type": (
                self.error_type
            ),

            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),

            "observed_at": (
                self.observed_at
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "INTEGRATION_SCHEMA_VERSION",

    "ProviderKind",
    "IntegrationState",
    "IntegrationHealthStatus",
    "OperationStatus",

    "IntegrationTarget",
    "IntegrationCapability",
    "IntegrationStatus",

    "IntegrationOperation",
    "IntegrationResult",

    "IntegrationHealth",
]
