from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from src.telemetry.schema import (
    ResourceIdentity,
    Signal,
    TelemetryRecord,
)


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    """
    Parse ISO-8601 timestamps for registry
    ordering/filtering.

    Invalid timestamps return None rather than
    causing the registry to fail.
    """

    text = optional_string(value)

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def resource_key(
    resource: ResourceIdentity,
) -> str:
    """
    Return the normalized effective identifier.

    canonical_id is automatically preferred by
    ResourceIdentity.effective_id().
    """

    return resource.effective_id()


@dataclass(frozen=True)
class RegistryStats:
    resources: int

    records: int

    environments: int

    providers: int

    resource_types: int

    canonical_resources: int

    def to_dict(self) -> dict:
        return {
            "resources": (self.resources),
            "records": self.records,
            "environments": (self.environments),
            "providers": (self.providers),
            "resource_types": (self.resource_types),
            "canonical_resources": (self.canonical_resources),
        }


class ResourceRegistry:
    """
    Indexed registry for canonical OSHA telemetry.

    It stores normalized resources separately
    from individual telemetry observations.

    This class does not perform persistence.
    A database-backed registry can later use the
    same conceptual interface.
    """

    def __init__(
        self,
    ) -> None:
        # Primary resource index.
        self._resources: Dict[
            str,
            ResourceIdentity,
        ] = {}

        # Primary telemetry-record index.
        self._records: Dict[
            str,
            TelemetryRecord,
        ] = {}

        # Resource -> telemetry record IDs.
        self._records_by_resource: Dict[
            str,
            List[str],
        ] = defaultdict(list)

        # Environment -> resource IDs.
        self._resources_by_environment: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Provider -> resource IDs.
        self._resources_by_provider: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Common resource type -> resource IDs.
        self._resources_by_type: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Host -> resource IDs.
        self._resources_by_host: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Service -> resource IDs.
        self._resources_by_service: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Cluster -> resource IDs.
        self._resources_by_cluster: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Namespace -> resource IDs.
        self._resources_by_namespace: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Canonical ID -> provider-scoped resource
        # IDs that resolve to it.
        self._canonical_members: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Signal type -> record IDs.
        self._records_by_signal_type: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Signal severity -> record IDs.
        self._records_by_severity: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Provider -> record IDs.
        self._records_by_provider: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Environment -> record IDs.
        self._records_by_environment: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

    # ---------------------------------------------------------
    # Resource registration
    # ---------------------------------------------------------

    def register_resource(
        self,
        resource: ResourceIdentity,
    ) -> str:
        """
        Register or refresh one normalized
        resource.

        Returns its effective registry key.
        """

        key = resource_key(resource)

        self._resources[key] = resource

        environment = resource.environment_id or "default"

        self._resources_by_environment[environment].add(key)

        self._resources_by_provider[resource.provider].add(key)

        self._resources_by_type[resource.resource_type].add(key)

        host = optional_string(resource.host)

        if host:
            self._resources_by_host[host.lower()].add(key)

        service = optional_string(resource.service)

        if service:
            self._resources_by_service[service.lower()].add(key)

        cluster = optional_string(resource.cluster_id)

        if cluster:
            self._resources_by_cluster[cluster].add(key)

        namespace = optional_string(resource.namespace)

        if namespace:
            namespace_key = f"{environment}:{namespace}"

            self._resources_by_namespace[namespace_key].add(key)

        if resource.canonical_id:
            self._canonical_members[resource.canonical_id].add(key)

        return key

    # ---------------------------------------------------------
    # Telemetry registration
    # ---------------------------------------------------------

    def register(
        self,
        telemetry: TelemetryRecord,
    ) -> str:
        """
        Register one canonical TelemetryRecord.

        Duplicate record IDs replace the stored
        object rather than creating duplicate
        registry entries.
        """

        telemetry.finalize()

        if not telemetry.record_id:
            raise ValueError("Telemetry record has no record_id after finalize()")

        record_id = telemetry.record_id

        resource_id = self.register_resource(telemetry.resource)

        already_present = record_id in self._records

        self._records[record_id] = telemetry

        if (
            not already_present
            and record_id not in self._records_by_resource[resource_id]
        ):
            self._records_by_resource[resource_id].append(record_id)

        signal = telemetry.signal

        self._records_by_signal_type[signal.signal_type].add(record_id)

        self._records_by_severity[signal.severity].add(record_id)

        self._records_by_provider[telemetry.resource.provider].add(record_id)

        self._records_by_environment[telemetry.resource.environment_id].add(record_id)

        return record_id

    def register_many(
        self,
        records: Iterable[TelemetryRecord],
    ) -> int:
        count = 0

        for telemetry in records:
            self.register(telemetry)

            count += 1

        return count

    # ---------------------------------------------------------
    # Direct lookups
    # ---------------------------------------------------------

    def get_resource(
        self,
        resource_id: str,
    ) -> Optional[ResourceIdentity]:
        return self._resources.get(resource_id)

    def get_record(
        self,
        record_id: str,
    ) -> Optional[TelemetryRecord]:
        return self._records.get(record_id)

    def has_resource(
        self,
        resource_id: str,
    ) -> bool:
        return resource_id in self._resources

    def has_record(
        self,
        record_id: str,
    ) -> bool:
        return record_id in self._records

    # ---------------------------------------------------------
    # Resource queries
    # ---------------------------------------------------------

    def resources(
        self,
    ) -> List[ResourceIdentity]:
        return list(self._resources.values())

    def resource_ids(
        self,
    ) -> List[str]:
        return sorted(self._resources.keys())

    def resources_for_provider(
        self,
        provider: str,
    ) -> List[ResourceIdentity]:
        keys = self._resources_by_provider.get(
            provider,
            set(),
        )

        return [self._resources[key] for key in sorted(keys) if key in self._resources]

    def resources_for_type(
        self,
        resource_type: str,
    ) -> List[ResourceIdentity]:
        keys = self._resources_by_type.get(
            resource_type,
            set(),
        )

        return [self._resources[key] for key in sorted(keys) if key in self._resources]

    def resources_for_environment(
        self,
        environment_id: str,
    ) -> List[ResourceIdentity]:
        keys = self._resources_by_environment.get(
            environment_id,
            set(),
        )

        return [self._resources[key] for key in sorted(keys) if key in self._resources]

    def resources_for_host(
        self,
        host: str,
    ) -> List[ResourceIdentity]:
        key = host.strip().lower()

        resource_ids = self._resources_by_host.get(
            key,
            set(),
        )

        return [
            self._resources[resource_id]
            for resource_id in sorted(resource_ids)
            if resource_id in self._resources
        ]

    def resources_for_service(
        self,
        service: str,
    ) -> List[ResourceIdentity]:
        key = service.strip().lower()

        resource_ids = self._resources_by_service.get(
            key,
            set(),
        )

        return [
            self._resources[resource_id]
            for resource_id in sorted(resource_ids)
            if resource_id in self._resources
        ]

    def resources_for_cluster(
        self,
        cluster_id: str,
    ) -> List[ResourceIdentity]:
        resource_ids = self._resources_by_cluster.get(
            cluster_id,
            set(),
        )

        return [
            self._resources[resource_id]
            for resource_id in sorted(resource_ids)
            if resource_id in self._resources
        ]

    def resources_for_namespace(
        self,
        namespace: str,
        *,
        environment_id: str = ("default"),
    ) -> List[ResourceIdentity]:
        key = f"{environment_id}:{namespace}"

        resource_ids = self._resources_by_namespace.get(
            key,
            set(),
        )

        return [
            self._resources[resource_id]
            for resource_id in sorted(resource_ids)
            if resource_id in self._resources
        ]

    def canonical_members(
        self,
        canonical_id: str,
    ) -> List[ResourceIdentity]:
        resource_ids = self._canonical_members.get(
            canonical_id,
            set(),
        )

        return [
            self._resources[resource_id]
            for resource_id in sorted(resource_ids)
            if resource_id in self._resources
        ]

    # ---------------------------------------------------------
    # Record queries
    # ---------------------------------------------------------

    def records(
        self,
    ) -> List[TelemetryRecord]:
        return list(self._records.values())

    def record_ids(
        self,
    ) -> List[str]:
        return sorted(self._records.keys())

    def records_for_resource(
        self,
        resource_id: str,
        *,
        chronological: bool = True,
    ) -> List[TelemetryRecord]:
        record_ids = self._records_by_resource.get(
            resource_id,
            [],
        )

        records = [
            self._records[record_id]
            for record_id in record_ids
            if record_id in self._records
        ]

        if chronological:
            records.sort(
                key=lambda item: (
                    parse_timestamp(item.signal.timestamp)
                    or datetime.min.replace(tzinfo=timezone.utc)
                )
            )

        return records

    def records_for_provider(
        self,
        provider: str,
    ) -> List[TelemetryRecord]:
        ids = self._records_by_provider.get(
            provider,
            set(),
        )

        return [
            self._records[record_id]
            for record_id in sorted(ids)
            if record_id in self._records
        ]

    def records_for_environment(
        self,
        environment_id: str,
    ) -> List[TelemetryRecord]:
        ids = self._records_by_environment.get(
            environment_id,
            set(),
        )

        return [
            self._records[record_id]
            for record_id in sorted(ids)
            if record_id in self._records
        ]

    def records_for_signal_type(
        self,
        signal_type: str,
    ) -> List[TelemetryRecord]:
        ids = self._records_by_signal_type.get(
            signal_type,
            set(),
        )

        return [
            self._records[record_id]
            for record_id in sorted(ids)
            if record_id in self._records
        ]

    def records_for_severity(
        self,
        severity: str,
    ) -> List[TelemetryRecord]:
        ids = self._records_by_severity.get(
            severity,
            set(),
        )

        return [
            self._records[record_id]
            for record_id in sorted(ids)
            if record_id in self._records
        ]

    # ---------------------------------------------------------
    # Temporal queries
    # ---------------------------------------------------------

    def records_between(
        self,
        start: datetime,
        end: datetime,
        *,
        environment_id: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> List[TelemetryRecord]:
        """
        Return observations within an inclusive
        UTC-aware time range.

        Invalid record timestamps are ignored.
        """

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        if end < start:
            raise ValueError("end must be greater than or equal to start")

        if resource_id:
            candidates = self.records_for_resource(
                resource_id,
                chronological=False,
            )

        elif environment_id:
            candidates = self.records_for_environment(environment_id)

        else:
            candidates = self.records()

        output = []

        for telemetry in candidates:
            if environment_id and telemetry.resource.environment_id != environment_id:
                continue

            timestamp = parse_timestamp(telemetry.signal.timestamp)

            if timestamp is None:
                continue

            if start <= timestamp <= end:
                output.append(telemetry)

        output.sort(
            key=lambda item: (
                parse_timestamp(item.signal.timestamp)
                or datetime.min.replace(tzinfo=timezone.utc)
            )
        )

        return output

    # ---------------------------------------------------------
    # Composite lookup helpers
    # ---------------------------------------------------------

    def find_resources(
        self,
        *,
        provider: Optional[str] = None,
        resource_type: Optional[str] = None,
        environment_id: Optional[str] = None,
        host: Optional[str] = None,
        service: Optional[str] = None,
        cluster_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[ResourceIdentity]:
        """
        Filter resources without requiring every
        caller to understand registry indexes.
        """

        candidates = self.resources()

        output = []

        for resource in candidates:
            if provider is not None and resource.provider != provider:
                continue

            if resource_type is not None and resource.resource_type != resource_type:
                continue

            if environment_id is not None and resource.environment_id != environment_id:
                continue

            if host is not None:
                expected = host.strip().lower()

                actual = (resource.host or "").strip().lower()

                if actual != expected:
                    continue

            if service is not None:
                expected = service.strip().lower()

                actual = (resource.service or "").strip().lower()

                if actual != expected:
                    continue

            if cluster_id is not None and resource.cluster_id != cluster_id:
                continue

            if namespace is not None and resource.namespace != namespace:
                continue

            output.append(resource)

        return output

    # ---------------------------------------------------------
    # Statistics and maintenance
    # ---------------------------------------------------------

    def stats(
        self,
    ) -> RegistryStats:
        canonical_ids = {
            resource.canonical_id
            for resource in self._resources.values()
            if resource.canonical_id
        }

        environments = {
            resource.environment_id for resource in self._resources.values()
        }

        providers = {resource.provider for resource in self._resources.values()}

        resource_types = {
            resource.resource_type for resource in self._resources.values()
        }

        return RegistryStats(
            resources=len(self._resources),
            records=len(self._records),
            environments=len(environments),
            providers=len(providers),
            resource_types=len(resource_types),
            canonical_resources=len(canonical_ids),
        )

    def clear_records(
        self,
    ) -> None:
        """
        Remove telemetry observations while
        retaining the resource inventory.
        """

        self._records.clear()

        self._records_by_resource.clear()

        self._records_by_signal_type.clear()

        self._records_by_severity.clear()

        self._records_by_provider.clear()

        self._records_by_environment.clear()

    def clear(
        self,
    ) -> None:
        """
        Remove all resources and observations.
        """

        self._resources.clear()

        self._records.clear()

        self._records_by_resource.clear()

        self._resources_by_environment.clear()

        self._resources_by_provider.clear()

        self._resources_by_type.clear()

        self._resources_by_host.clear()

        self._resources_by_service.clear()

        self._resources_by_cluster.clear()

        self._resources_by_namespace.clear()

        self._canonical_members.clear()

        self._records_by_signal_type.clear()

        self._records_by_severity.clear()

        self._records_by_provider.clear()

        self._records_by_environment.clear()

    def __len__(
        self,
    ) -> int:
        return len(self._records)

    def __iter__(
        self,
    ) -> Iterator[TelemetryRecord]:
        return iter(self._records.values())
