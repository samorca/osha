from __future__ import annotations

import hashlib
import json
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


SCHEMA_VERSION = "1"


class Provider(str, Enum):
    """
    Infrastructure/provider families currently
    understood by OSHA.

    OTHER allows future collectors to enter the
    normalization pipeline before a dedicated
    enum value is added.
    """

    LINUX = "linux"
    KUBERNETES = "kubernetes"
    OPENSTACK = "openstack"

    CEPH = "ceph"
    MINIO = "minio"

    TERRAFORM = "terraform"
    PROMETHEUS = "prometheus"

    POSTGRESQL = "postgresql"
    REDIS = "redis"
    KAFKA = "kafka"

    GOOGLE_CLOUD = "google_cloud"
    IBM_CLOUD = "ibm_cloud"

    OTHER = "other"


class SignalType(str, Enum):
    """
    High-level signal categories used by OSHA.

    Provider-specific collectors may expose
    hundreds of original signal names. The
    normalization layer maps them into this
    smaller provider-neutral vocabulary.
    """

    LOG = "log"
    METRIC = "metric"
    EVENT = "event"

    RESOURCE_STATE = "resource_state"

    STATE_CHANGE = "state_change"

    ALERT = "alert"

    CONFIGURATION = "configuration"

    SECURITY = "security"

    INVENTORY = "inventory"

    HEALTH = "health"

    RELATIONSHIP = "relationship"

    COLLECTOR_STATUS = "collector_status"

    UNKNOWN = "unknown"


class ResourceType(str, Enum):
    """
    Common infrastructure resource categories.

    The specific provider type remains available
    separately in native_type.
    """

    CLOUD = "cloud"
    REGION = "region"
    ZONE = "zone"

    HOST = "host"
    VM = "virtual_machine"

    CLUSTER = "cluster"
    NODE = "node"

    NAMESPACE = "namespace"
    WORKLOAD = "workload"
    POD = "pod"
    CONTAINER = "container"

    PROCESS = "process"
    SERVICE = "service"

    NETWORK = "network"
    SUBNET = "subnet"
    ROUTER = "router"
    PORT = "port"
    IP_ADDRESS = "ip_address"

    STORAGE = "storage"
    VOLUME = "volume"
    SNAPSHOT = "snapshot"
    FILESYSTEM = "filesystem"
    BLOCK_DEVICE = "block_device"

    IMAGE = "image"

    DATABASE = "database"
    QUEUE = "queue"

    USER = "user"
    GROUP = "group"
    PROJECT = "project"

    APPLICATION = "application"

    COLLECTOR = "collector"

    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """
    Provider-neutral relationship vocabulary.

    Relationships should be asserted only when
    the source data gives enough evidence.
    """

    CONTAINS = "contains"

    PART_OF = "part_of"

    RUNS_ON = "runs_on"

    HOSTS = "hosts"

    CONNECTED_TO = "connected_to"

    ATTACHED_TO = "attached_to"

    USES = "uses"

    DEPENDS_ON = "depends_on"

    MANAGED_BY = "managed_by"

    PROVISIONED_BY = "provisioned_by"

    MEMBER_OF = "member_of"

    ROUTES_TO = "routes_to"

    RELATED_TO = "related_to"


class Severity(str, Enum):
    """
    Common OSHA severity vocabulary.

    The original/native severity should still be
    preserved in signal attributes when useful.
    """

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(
    value: Any,
) -> str:
    """
    Convert a timestamp into a stable ISO-8601
    representation.

    Missing/unparseable timestamps are not
    guessed from provider data. When there is no
    source timestamp, OSHA records the time of
    normalization.
    """

    if value is None:
        return utc_now()

    if isinstance(
        value,
        datetime,
    ):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.isoformat()

    text = str(value).strip()

    if not text:
        return utc_now()

    return text


def normalize_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    """
    Convert arbitrary normalized values into
    JSON-safe structures.

    Secret redaction belongs in redaction.py;
    this function exists only for safe JSON
    serialization.
    """

    if depth > maximum_depth:
        return "[MAX_DEPTH_REACHED]"

    if value is None:
        return None

    if isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
        ),
    ):
        return value

    if isinstance(
        value,
        datetime,
    ):
        return normalize_timestamp(value)

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(maximum_depth),
            )
            for key, item in value.items()
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
                maximum_depth=(maximum_depth),
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
            maximum_depth=(maximum_depth),
        )

    return str(value)


def stable_hash(
    value: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ResourceIdentity:
    """
    Provider-neutral identity for one
    infrastructure resource.

    provider_id:
        Original provider identifier.

    canonical_id:
        OSHA's cross-source identifier after
        identity resolution.

    native_type:
        Original resource type reported by the
        collector.
    """

    resource_type: str

    provider: str

    provider_id: str

    canonical_id: Optional[str] = None

    native_type: Optional[str] = None

    name: Optional[str] = None

    environment_id: str = "default"

    region: Optional[str] = None

    zone: Optional[str] = None

    cluster_id: Optional[str] = None

    namespace: Optional[str] = None

    host: Optional[str] = None

    service: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def effective_id(
        self,
    ) -> str:
        """
        Return canonical ID when identity
        resolution has occurred.

        Otherwise return a deterministic
        provider-scoped identity.
        """

        if self.canonical_id:
            return self.canonical_id

        components = [
            self.provider,
            self.environment_id,
            self.resource_type,
            self.provider_id,
        ]

        return "::".join(str(component) for component in components)

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


@dataclass
class Signal:
    """
    Provider-neutral observation attached to a
    resource.

    One signal represents one observation such
    as:

      log
      metric measurement
      event
      alert
      health observation
      resource state
      configuration state
    """

    signal_type: str

    timestamp: str

    provider: str

    resource_id: str

    environment_id: str = "default"

    signal_name: Optional[str] = None

    severity: str = Severity.UNKNOWN.value

    value: Any = None

    unit: Optional[str] = None

    message: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    source_record_id: Optional[str] = None

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


@dataclass
class Relationship:
    """
    Explicit relationship between two
    normalized resources.
    """

    source_resource_id: str

    target_resource_id: str

    relationship_type: str

    provider: str

    environment_id: str = "default"

    timestamp: str = field(default_factory=utc_now)

    confidence: Optional[float] = None

    evidence: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


@dataclass
class TelemetryRecord:
    """
    Canonical record emitted by the OSHA
    telemetry normalization layer.

    It intentionally keeps the normalized
    resource, signal, relationships, and native
    provider data separate.
    """

    resource: ResourceIdentity

    signal: Signal

    relationships: List[Relationship] = field(default_factory=list)

    native: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    normalized_at: str = field(default_factory=utc_now)

    record_id: Optional[str] = None

    def build_record_id(
        self,
    ) -> str:
        """
        Generate a deterministic ID from the
        observation itself.

        normalized_at is excluded so that
        re-normalizing the same source record
        does not automatically create a new
        identity.
        """

        body = {
            "schema_version": (self.schema_version),
            "resource": (self.resource.to_dict()),
            "signal": (self.signal.to_dict()),
            "relationships": [
                relationship.to_dict() for relationship in self.relationships
            ],
            "native": (json_safe(self.native)),
        }

        return stable_hash(body)

    def finalize(
        self,
    ) -> "TelemetryRecord":
        if not self.record_id:
            self.record_id = self.build_record_id()

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "record_id": (self.record_id),
            "normalized_at": (self.normalized_at),
            "resource": (self.resource.to_dict()),
            "signal": (self.signal.to_dict()),
            "relationships": [
                relationship.to_dict() for relationship in self.relationships
            ],
            "native": (json_safe(self.native)),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )


def build_resource(
    *,
    provider: str,
    resource_type: str,
    provider_id: str,
    environment_id: str = "default",
    canonical_id: Optional[str] = None,
    native_type: Optional[str] = None,
    name: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    cluster_id: Optional[str] = None,
    namespace: Optional[str] = None,
    host: Optional[str] = None,
    service: Optional[str] = None,
    attributes: Optional[Mapping[str, Any]] = None,
) -> ResourceIdentity:
    """
    Convenience constructor used by provider
    normalizers.
    """

    return ResourceIdentity(
        resource_type=(str(resource_type)),
        provider=str(provider),
        provider_id=str(provider_id),
        canonical_id=(normalize_string(canonical_id)),
        native_type=(normalize_string(native_type)),
        name=normalize_string(name),
        environment_id=(str(environment_id or "default")),
        region=normalize_string(region),
        zone=normalize_string(zone),
        cluster_id=(normalize_string(cluster_id)),
        namespace=(normalize_string(namespace)),
        host=normalize_string(host),
        service=normalize_string(service),
        attributes=(json_safe(dict(attributes or {}))),
    )


def build_signal(
    *,
    provider: str,
    resource_id: str,
    signal_type: str,
    timestamp: Any = None,
    environment_id: str = "default",
    signal_name: Optional[str] = None,
    severity: str = (Severity.UNKNOWN.value),
    value: Any = None,
    unit: Optional[str] = None,
    message: Optional[str] = None,
    attributes: Optional[Mapping[str, Any]] = None,
    source_record_id: Optional[str] = None,
) -> Signal:
    """
    Convenience constructor used by
    normalize.py and signal.py.
    """

    return Signal(
        signal_type=str(signal_type),
        timestamp=(normalize_timestamp(timestamp)),
        provider=str(provider),
        resource_id=str(resource_id),
        environment_id=(str(environment_id or "default")),
        signal_name=(normalize_string(signal_name)),
        severity=str(severity or Severity.UNKNOWN.value),
        value=json_safe(value),
        unit=normalize_string(unit),
        message=(normalize_string(message)),
        attributes=(json_safe(dict(attributes or {}))),
        source_record_id=(normalize_string(source_record_id)),
    )


def build_record(
    *,
    resource: ResourceIdentity,
    signal: Signal,
    relationships: Optional[List[Relationship]] = None,
    native: Optional[Mapping[str, Any]] = None,
) -> TelemetryRecord:
    """
    Construct and finalize one canonical OSHA
    telemetry record.
    """

    return TelemetryRecord(
        resource=resource,
        signal=signal,
        relationships=(relationships or []),
        native=(json_safe(dict(native or {}))),
    ).finalize()
