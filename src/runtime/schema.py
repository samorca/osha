from __future__ import annotations

import hashlib
import json
import math
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)


RUNTIME_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------

class RuntimeState(
    str,
    Enum,
):
    CREATED = "created"

    STARTING = "starting"

    RUNNING = "running"

    DEGRADED = "degraded"

    STOPPING = "stopping"

    STOPPED = "stopped"

    FAILED = "failed"


class ServiceState(
    str,
    Enum,
):
    REGISTERED = "registered"

    STARTING = "starting"

    RUNNING = "running"

    DEGRADED = "degraded"

    STOPPING = "stopping"

    STOPPED = "stopped"

    FAILED = "failed"


class ReadinessStatus(
    str,
    Enum,
):
    UNKNOWN = "unknown"

    READY = "ready"

    NOT_READY = "not_ready"

    DEGRADED = "degraded"


class ShutdownReason(
    str,
    Enum,
):
    NONE = "none"

    REQUESTED = "requested"

    SIGNAL = "signal"

    FAILURE = "failure"

    DEPENDENCY_FAILURE = (
        "dependency_failure"
    )

    MAINTENANCE = "maintenance"


# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    if depth > maximum_depth:
        return (
            "[MAX_DEPTH_REACHED]"
        )

    if (
        value is None
        or isinstance(
            value,
            (
                bool,
                int,
                str,
            ),
        )
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        datetime,
    ):
        if value.tzinfo is None:
            value = value.replace(
                tzinfo=timezone.utc
            )

        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
            )
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return json_safe(
            asdict(value),
            depth=depth + 1,
            maximum_depth=(
                maximum_depth
            ),
        )

    return str(value)


def stable_hash(
    value: Mapping[
        str,
        Any,
    ],
) -> str:
    encoded = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ---------------------------------------------------------------
# Runtime evidence
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeEvidence:
    """
    Evidence about the application runtime.

    This should not be used to imply external
    infrastructure health.
    """

    evidence_type: str

    description: str

    source_id: Optional[
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

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "evidence_type": (
                self.evidence_type
            ),
            "description": (
                self.description
            ),
            "source_id": (
                self.source_id
            ),
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "observed_at": (
                self.observed_at
            ),
        }


# ---------------------------------------------------------------
# Service status
# ---------------------------------------------------------------

@dataclass
class ServiceStatus:
    """
    Runtime state for one registered OSHA
    application service/component.
    """

    name: str

    state: str = (
        ServiceState.
        REGISTERED.value
    )

    readiness: str = (
        ReadinessStatus.
        UNKNOWN.value
    )

    required: bool = True

    dependencies: List[
        str
    ] = field(
        default_factory=list
    )

    started_at: Optional[
        str
    ] = None

    stopped_at: Optional[
        str
    ] = None

    last_error: Optional[
        str
    ] = None

    restart_count: int = 0

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "ServiceStatus":
        self.name = (
            self.name.strip()
        )

        if not self.name:
            raise ValueError(
                "Service name is required"
            )

        if (
            self.state
            not in {
                item.value
                for item
                in ServiceState
            }
        ):
            raise ValueError(
                "Unknown service state: "
                f"{self.state}"
            )

        if (
            self.readiness
            not in {
                item.value
                for item
                in ReadinessStatus
            }
        ):
            raise ValueError(
                "Unknown readiness "
                f"status: {self.readiness}"
            )

        self.dependencies = sorted(
            set(
                self.dependencies
            )
        )

        self.restart_count = max(
            0,
            int(
                self.restart_count
            ),
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Runtime context
# ---------------------------------------------------------------

@dataclass
class RuntimeContext:
    """
    Identity and scope for one OSHA application
    runtime instance.
    """

    environment_id: str

    instance_name: str = "osha"

    dry_run: bool = True

    process_id: Optional[
        int
    ] = None

    host_id: Optional[
        str
    ] = None

    started_by: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    context_id: Optional[
        str
    ] = None

    def build_context_id(
        self,
    ) -> str:
        return (
            "runtime:context:"
            + stable_hash(
                {
                    "environment_id": (
                        self.environment_id
                    ),
                    "instance_name": (
                        self.instance_name
                    ),
                    "host_id": (
                        self.host_id
                    ),
                    "created_at": (
                        self.created_at
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "RuntimeContext":
        self.environment_id = (
            self.environment_id.strip()
        )

        self.instance_name = (
            self.instance_name.strip()
        )

        if not self.environment_id:
            raise ValueError(
                "environment_id is required"
            )

        if not self.instance_name:
            raise ValueError(
                "instance_name is required"
            )

        if (
            self.process_id
            is not None
        ):
            self.process_id = max(
                0,
                int(
                    self.process_id
                ),
            )

        if not self.context_id:
            self.context_id = (
                self.build_context_id()
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Runtime status
# ---------------------------------------------------------------

@dataclass
class RuntimeStatus:
    """
    Complete lifecycle status for one OSHA
    runtime instance.
    """

    context: RuntimeContext

    state: str = (
        RuntimeState.CREATED.value
    )

    readiness: str = (
        ReadinessStatus.
        UNKNOWN.value
    )

    services: List[
        ServiceStatus
    ] = field(
        default_factory=list
    )

    shutdown_reason: str = (
        ShutdownReason.NONE.value
    )

    error: Optional[
        str
    ] = None

    evidence: List[
        RuntimeEvidence
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    started_at: Optional[
        str
    ] = None

    stopped_at: Optional[
        str
    ] = None

    updated_at: str = field(
        default_factory=utc_now
    )

    runtime_id: Optional[
        str
    ] = None

    schema_version: str = (
        RUNTIME_SCHEMA_VERSION
    )

    def build_runtime_id(
        self,
    ) -> str:
        self.context.finalize()

        return (
            "runtime:instance:"
            + stable_hash(
                {
                    "context_id": (
                        self.context.
                        context_id
                    ),
                    "created_at": (
                        self.context.
                        created_at
                    ),
                }
            )
        )

    def service(
        self,
        name: str,
    ) -> Optional[
        ServiceStatus
    ]:
        for service in self.services:
            if (
                service.name
                == name
            ):
                return service

        return None

    def finalize(
        self,
    ) -> "RuntimeStatus":
        self.context.finalize()

        if (
            self.state
            not in {
                item.value
                for item
                in RuntimeState
            }
        ):
            raise ValueError(
                "Unknown runtime state: "
                f"{self.state}"
            )

        if (
            self.readiness
            not in {
                item.value
                for item
                in ReadinessStatus
            }
        ):
            raise ValueError(
                "Unknown readiness "
                f"status: {self.readiness}"
            )

        if (
            self.shutdown_reason
            not in {
                item.value
                for item
                in ShutdownReason
            }
        ):
            raise ValueError(
                "Unknown shutdown reason: "
                f"{self.shutdown_reason}"
            )

        seen = set()

        normalized: List[
            ServiceStatus
        ] = []

        for service in self.services:
            service.finalize()

            if service.name in seen:
                raise ValueError(
                    "Duplicate runtime "
                    f"service: {service.name}"
                )

            seen.add(
                service.name
            )

            normalized.append(
                service
            )

        self.services = (
            normalized
        )

        if not self.runtime_id:
            self.runtime_id = (
                self.build_runtime_id()
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
                self.schema_version
            ),
            "runtime_id": (
                self.runtime_id
            ),
            "context": (
                self.context.to_dict()
            ),
            "state": self.state,
            "readiness": (
                self.readiness
            ),
            "services": [
                item.to_dict()
                for item
                in self.services
            ],
            "shutdown_reason": (
                self.shutdown_reason
            ),
            "error": self.error,
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "started_at": (
                self.started_at
            ),
            "stopped_at": (
                self.stopped_at
            ),
            "updated_at": (
                self.updated_at
            ),
        }


# ---------------------------------------------------------------
# Top-level runtime result
# ---------------------------------------------------------------

@dataclass
class RuntimeResult:
    """
    Aggregate runtime information across one or
    more OSHA runtime instances.
    """

    runtimes: List[
        RuntimeStatus
    ] = field(
        default_factory=list
    )

    generated_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    schema_version: str = (
        RUNTIME_SCHEMA_VERSION
    )

    def finalize(
        self,
    ) -> "RuntimeResult":
        self.runtimes = [
            item.finalize()
            for item
            in self.runtimes
        ]

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        states: Dict[
            str,
            int,
        ] = {}

        readiness: Dict[
            str,
            int,
        ] = {}

        for runtime in self.runtimes:
            states[
                runtime.state
            ] = (
                states.get(
                    runtime.state,
                    0,
                )
                + 1
            )

            readiness[
                runtime.readiness
            ] = (
                readiness.get(
                    runtime.readiness,
                    0,
                )
                + 1
            )

        return {
            "schema_version": (
                self.schema_version
            ),
            "generated_at": (
                self.generated_at
            ),
            "runtime_count": len(
                self.runtimes
            ),
            "states": states,
            "readiness": readiness,
            "runtimes": [
                item.to_dict()
                for item
                in self.runtimes
            ],
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
        }
