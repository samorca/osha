from __future__ import annotations

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.telemetry.redaction import (
    DEFAULT_POLICY,
    RedactionPolicy,
    redact_attributes,
    redact_text,
)

from src.telemetry.schema import (
    Provider,
    ResourceIdentity,
    Severity,
    Signal,
    SignalType,
    build_signal,
    normalize_string,
)


# ---------------------------------------------------------------
# Common signal mappings
# ---------------------------------------------------------------

SIGNAL_TYPE_ALIASES = {
    # Logs
    "log": SignalType.LOG.value,
    "journal": SignalType.LOG.value,
    "journal_log": SignalType.LOG.value,
    # Metrics
    "metric": SignalType.METRIC.value,
    "measurement": SignalType.METRIC.value,
    "metric_measurement": SignalType.METRIC.value,
    # Events
    "event": SignalType.EVENT.value,
    "openstack_event": SignalType.EVENT.value,
    "kubernetes_event": SignalType.EVENT.value,
    # Resource state
    "resource_state": SignalType.RESOURCE_STATE.value,
    "service_state": SignalType.RESOURCE_STATE.value,
    "process_state": SignalType.RESOURCE_STATE.value,
    "container_state": SignalType.RESOURCE_STATE.value,
    "filesystem_state": SignalType.RESOURCE_STATE.value,
    "network_state": SignalType.RESOURCE_STATE.value,
    "network_interface": SignalType.RESOURCE_STATE.value,
    "route_state": SignalType.RESOURCE_STATE.value,
    "dns_state": SignalType.RESOURCE_STATE.value,
    "swap_state": SignalType.RESOURCE_STATE.value,
    "package_state": SignalType.RESOURCE_STATE.value,
    "compute_instance": SignalType.RESOURCE_STATE.value,
    "compute_service": SignalType.RESOURCE_STATE.value,
    "compute_hypervisor": SignalType.RESOURCE_STATE.value,
    "network": SignalType.RESOURCE_STATE.value,
    "subnet": SignalType.RESOURCE_STATE.value,
    "port": SignalType.RESOURCE_STATE.value,
    "router": SignalType.RESOURCE_STATE.value,
    "floating_ip": SignalType.RESOURCE_STATE.value,
    "network_agent": SignalType.RESOURCE_STATE.value,
    "volume": SignalType.RESOURCE_STATE.value,
    "snapshot": SignalType.RESOURCE_STATE.value,
    "backup": SignalType.RESOURCE_STATE.value,
    "image": SignalType.RESOURCE_STATE.value,
    "project": SignalType.RESOURCE_STATE.value,
    "domain": SignalType.RESOURCE_STATE.value,
    "region": SignalType.RESOURCE_STATE.value,
    "role": SignalType.RESOURCE_STATE.value,
    "user": SignalType.RESOURCE_STATE.value,
    "group": SignalType.RESOURCE_STATE.value,
    "node_state": SignalType.RESOURCE_STATE.value,
    "pod_state": SignalType.RESOURCE_STATE.value,
    "deployment_state": SignalType.RESOURCE_STATE.value,
    # State change
    "state_change": SignalType.STATE_CHANGE.value,
    # Alerts
    "alert": SignalType.ALERT.value,
    "alarm": SignalType.ALERT.value,
    # Configuration
    "configuration": SignalType.CONFIGURATION.value,
    "configuration_state": SignalType.CONFIGURATION.value,
    # Security
    "security": SignalType.SECURITY.value,
    "security_posture": SignalType.SECURITY.value,
    "security_identity": SignalType.SECURITY.value,
    # Inventory
    "inventory": SignalType.INVENTORY.value,
    "block_device": SignalType.INVENTORY.value,
    "container_runtime": SignalType.INVENTORY.value,
    "container_definition": SignalType.INVENTORY.value,
    "package_inventory": SignalType.INVENTORY.value,
    "openstack_service": SignalType.INVENTORY.value,
    "compute_flavor": SignalType.INVENTORY.value,
    "compute_aggregate": SignalType.INVENTORY.value,
    "availability_zone": SignalType.INVENTORY.value,
    "server_group": SignalType.INVENTORY.value,
    "keypair_metadata": SignalType.INVENTORY.value,
    "security_group": SignalType.INVENTORY.value,
    "volume_type": SignalType.INVENTORY.value,
    "metric_resource": SignalType.INVENTORY.value,
    "metric_definition": SignalType.INVENTORY.value,
    # Health
    "health": SignalType.HEALTH.value,
    # Relationships
    "relationship": SignalType.RELATIONSHIP.value,
    # Collector state
    "collector_status": SignalType.COLLECTOR_STATUS.value,
    "collector_error": SignalType.COLLECTOR_STATUS.value,
    "collection_summary": SignalType.COLLECTOR_STATUS.value,
    "service_summary": SignalType.COLLECTOR_STATUS.value,
    "compute_summary": SignalType.COLLECTOR_STATUS.value,
    "network_summary": SignalType.COLLECTOR_STATUS.value,
    "storage_summary": SignalType.COLLECTOR_STATUS.value,
    "image_summary": SignalType.COLLECTOR_STATUS.value,
    "identity_summary": SignalType.COLLECTOR_STATUS.value,
    "telemetry_summary": SignalType.COLLECTOR_STATUS.value,
    "event_summary": SignalType.COLLECTOR_STATUS.value,
    "log_summary": SignalType.COLLECTOR_STATUS.value,
    "log_source_state": SignalType.COLLECTOR_STATUS.value,
}


SEVERITY_ALIASES = {
    # Trace/debug
    "trace": Severity.TRACE.value,
    "verbose": Severity.DEBUG.value,
    "debug": Severity.DEBUG.value,
    # Informational
    "info": Severity.INFO.value,
    "information": Severity.INFO.value,
    "informational": Severity.INFO.value,
    "notice": Severity.NOTICE.value,
    "normal": Severity.INFO.value,
    "ok": Severity.INFO.value,
    # Warning
    "warn": Severity.WARNING.value,
    "warning": Severity.WARNING.value,
    "minor": Severity.WARNING.value,
    # Error
    "err": Severity.ERROR.value,
    "error": Severity.ERROR.value,
    "failed": Severity.ERROR.value,
    "failure": Severity.ERROR.value,
    "major": Severity.ERROR.value,
    # Critical
    "crit": Severity.CRITICAL.value,
    "critical": Severity.CRITICAL.value,
    "fatal": Severity.CRITICAL.value,
    "alert": Severity.ALERT.value,
    "emerg": Severity.EMERGENCY.value,
    "emergency": Severity.EMERGENCY.value,
    "unknown": Severity.UNKNOWN.value,
}


JOURNAL_PRIORITY_SEVERITY = {
    "0": Severity.EMERGENCY.value,
    "1": Severity.ALERT.value,
    "2": Severity.CRITICAL.value,
    "3": Severity.ERROR.value,
    "4": Severity.WARNING.value,
    "5": Severity.NOTICE.value,
    "6": Severity.INFO.value,
    "7": Severity.DEBUG.value,
}


# ---------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------


def optional_string(
    value: Any,
) -> Optional[str]:
    return normalize_string(value)


def first_value(
    payload: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key not in payload:
            continue

        value = payload.get(key)

        if value is None:
            continue

        if isinstance(value, str):
            if not value.strip():
                continue

        return value

    return None


def normalized_name(
    value: Any,
) -> Optional[str]:
    text = optional_string(value)

    if not text:
        return None

    return text.lower().replace("-", "_").replace(" ", "_")


def map_signal_type(
    native_signal_type: Any,
) -> str:
    normalized = normalized_name(native_signal_type)

    if not normalized:
        return SignalType.UNKNOWN.value

    mapped = SIGNAL_TYPE_ALIASES.get(normalized)

    if mapped:
        return mapped

    # Preserve broad suffix meaning without
    # inventing a provider-specific mapping.
    if normalized.endswith("_event"):
        return SignalType.EVENT.value

    if normalized.endswith("_metric"):
        return SignalType.METRIC.value

    if normalized.endswith("_alert"):
        return SignalType.ALERT.value

    if normalized.endswith("_alarm"):
        return SignalType.ALERT.value

    if normalized.endswith("_state"):
        return SignalType.RESOURCE_STATE.value

    if normalized.endswith("_summary"):
        return SignalType.COLLECTOR_STATUS.value

    return SignalType.UNKNOWN.value


def normalize_severity(
    value: Any,
    *,
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Normalize provider/native severity values.

    For journald records, PRIORITY values 0-7 are
    also recognized.
    """

    text = optional_string(value)

    if text:
        normalized = normalized_name(text)

        if normalized in SEVERITY_ALIASES:
            return SEVERITY_ALIASES[normalized]

        if text in JOURNAL_PRIORITY_SEVERITY:
            return JOURNAL_PRIORITY_SEVERITY[text]

    if payload:
        priority = optional_string(payload.get("priority"))

        if priority in JOURNAL_PRIORITY_SEVERITY:
            return JOURNAL_PRIORITY_SEVERITY[priority]

        healthy = payload.get("healthy")

        if healthy is False:
            return Severity.ERROR.value

        status = normalized_name(payload.get("status"))

        state = normalized_name(payload.get("state"))

        if status in {
            "error",
            "failed",
            "failure",
            "down",
            "unhealthy",
        }:
            return Severity.ERROR.value

        if state in {
            "error",
            "failed",
            "failure",
            "down",
            "unhealthy",
        }:
            return Severity.ERROR.value

        if status in {
            "warning",
            "degraded",
            "pending",
            "unknown",
        }:
            return Severity.WARNING.value

        if state in {
            "warning",
            "degraded",
            "pending",
            "unknown",
        }:
            return Severity.WARNING.value

    return Severity.UNKNOWN.value


def source_record_id(
    record_data: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        first_value(
            record_data,
            "record_id",
            "id",
            "message_id",
        )
    )


def signal_timestamp(
    record_data: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Any:
    return first_value(
        record_data,
        "timestamp",
        "observed_at",
        "collected_at",
    ) or first_value(
        payload,
        "timestamp",
        "generated",
        "updated_at",
        "created_at",
        "state_timestamp",
        "heartbeat_timestamp",
    )


def extract_message(
    native_signal_type: Optional[str],
    payload: Mapping[str, Any],
) -> Optional[str]:
    message = first_value(
        payload,
        "message",
        "reason",
        "description",
        "state_reason",
        "error",
    )

    if message is None:
        return None

    text = optional_string(message)

    if text is None:
        return None

    return redact_text(text)


def signal_name(
    native_signal_type: Optional[str],
    payload: Mapping[str, Any],
) -> Optional[str]:
    """
    Determine a stable signal name.

    Dynamic identifiers should remain in
    attributes and not be embedded into signal
    names.
    """

    candidates = (
        payload.get("event_type"),
        payload.get("metric_name"),
        payload.get("alarm_name"),
        payload.get("signal_name"),
        payload.get("collector"),
        native_signal_type,
    )

    for candidate in candidates:
        text = optional_string(candidate)

        if text:
            return text

    return None


def metric_value(
    payload: Mapping[str, Any],
) -> Any:
    return first_value(
        payload,
        "value",
        "metric_value",
        "measurement",
    )


def metric_unit(
    payload: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        first_value(
            payload,
            "unit",
            "metric_unit",
        )
    )


def state_value(
    payload: Mapping[str, Any],
) -> Any:
    value = first_value(
        payload,
        "status",
        "state",
        "active_state",
        "phase",
        "healthy",
        "available",
        "enabled",
    )

    if value is not None:
        return value

    return None


def collector_status_value(
    payload: Mapping[str, Any],
) -> Any:
    if "available" in payload:
        return payload.get("available")

    if "successful" in payload:
        return payload.get("successful")

    if "collector_error_count" in payload:
        return {"collector_error_count": (payload.get("collector_error_count"))}

    return state_value(payload)


def signal_value(
    common_type: str,
    payload: Mapping[str, Any],
) -> Any:
    if common_type == SignalType.METRIC.value:
        return metric_value(payload)

    if common_type in {
        SignalType.RESOURCE_STATE.value,
        SignalType.STATE_CHANGE.value,
        SignalType.HEALTH.value,
        SignalType.SECURITY.value,
    }:
        return state_value(payload)

    if common_type == SignalType.ALERT.value:
        return first_value(
            payload,
            "state",
            "status",
            "severity",
            "threshold",
        )

    if common_type == SignalType.COLLECTOR_STATUS.value:
        return collector_status_value(payload)

    return None


def base_attributes(
    record_data: Mapping[str, Any],
    payload: Mapping[str, Any],
    native_signal_type: Optional[str],
) -> Dict[str, Any]:
    """
    Preserve useful context without copying the
    complete native payload into Signal.

    The full redacted provider record is retained
    later in TelemetryRecord.native.
    """

    attributes = {
        "native_signal_type": (native_signal_type),
        "record_type": (record_data.get("record_type")),
        "service": (record_data.get("service")),
        "source": (record_data.get("source")),
        "region": (record_data.get("region")),
        "host_name": (record_data.get("host_name")),
    }

    selected_payload_keys = (
        "status",
        "state",
        "active_state",
        "sub_state",
        "phase",
        "healthy",
        "available",
        "enabled",
        "event_type",
        "service_family",
        "action",
        "reason",
        "priority",
        "severity",
        "metric_id",
        "metric_name",
        "granularity",
        "alarm_id",
        "project_id",
        "namespace",
        "node_name",
        "pod_name",
        "container_name",
        "pid",
        "service_name",
    )

    for key in selected_payload_keys:
        if key in payload:
            attributes[key] = payload.get(key)

    return {key: value for key, value in attributes.items() if value is not None}


# ---------------------------------------------------------------
# Provider-specific signal enrichment
# ---------------------------------------------------------------


def linux_signal_attributes(
    native_signal_type: Optional[str],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    attributes: Dict[
        str,
        Any,
    ] = {}

    if native_signal_type == "metric":
        for key in (
            "load_1m",
            "load_5m",
            "load_15m",
            "uptime_seconds",
            "memory_total_bytes",
            "memory_available_bytes",
            "memory_used_bytes",
            "memory_usage_percent",
            "swap_total_bytes",
            "swap_free_bytes",
        ):
            if key in payload:
                attributes[key] = payload.get(key)

    elif native_signal_type == "process_state":
        for key in (
            "pid",
            "parent_pid",
            "user",
            "executable",
            "threads",
            "resident_memory_bytes",
            "virtual_memory_bytes",
        ):
            if key in payload:
                attributes[key] = payload.get(key)

    elif native_signal_type == "service_state":
        for key in (
            "service_name",
            "active_state",
            "sub_state",
            "unit_file_state",
            "main_pid",
            "restart_count",
        ):
            if key in payload:
                attributes[key] = payload.get(key)

    elif native_signal_type == "listening_socket":
        for key in (
            "protocol",
            "local_address",
            "local_port",
            "exposure",
            "process_name",
            "pid",
        ):
            if key in payload:
                attributes[key] = payload.get(key)

    return attributes


def kubernetes_signal_attributes(
    native_signal_type: Optional[str],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    attributes: Dict[
        str,
        Any,
    ] = {}

    for key in (
        "kind",
        "namespace",
        "name",
        "uid",
        "reason",
        "type",
        "phase",
        "node_name",
        "pod_name",
        "container_name",
        "restart_count",
    ):
        if key in payload:
            attributes[key] = payload.get(key)

    return attributes


def openstack_signal_attributes(
    native_signal_type: Optional[str],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    attributes: Dict[
        str,
        Any,
    ] = {}

    for key in (
        "event_type",
        "message_id",
        "service_family",
        "action",
        "metric_id",
        "metric_name",
        "granularity",
        "alarm_id",
        "severity",
        "state",
        "state_reason",
        "status",
        "healthy",
        "project_id",
        "host",
        "availability_zone",
    ):
        if key in payload:
            attributes[key] = payload.get(key)

    return attributes


def provider_attributes(
    provider: str,
    native_signal_type: Optional[str],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if provider == Provider.LINUX.value:
        return linux_signal_attributes(
            native_signal_type,
            payload,
        )

    if provider == Provider.KUBERNETES.value:
        return kubernetes_signal_attributes(
            native_signal_type,
            payload,
        )

    if provider == Provider.OPENSTACK.value:
        return openstack_signal_attributes(
            native_signal_type,
            payload,
        )

    return {}


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------


def signal_from_record(
    record_data: Mapping[str, Any],
    resource: ResourceIdentity,
    *,
    policy: RedactionPolicy = (DEFAULT_POLICY),
) -> Signal:
    """
    Convert one provider-native collector record
    into OSHA's common Signal model.

    resource.py should create ResourceIdentity
    before this function is called.
    """

    provider = optional_string(record_data.get("provider")) or Provider.OTHER.value

    raw_payload = record_data.get("payload") or {}

    if isinstance(
        raw_payload,
        Mapping,
    ):
        payload: Mapping[
            str,
            Any,
        ] = raw_payload
    else:
        payload = {}

    native_signal_type = optional_string(record_data.get("signal_type"))

    common_signal_type = map_signal_type(native_signal_type)

    native_severity = first_value(
        payload,
        "severity",
        "severity_text",
        "level",
        "priority",
        "type",
    )

    severity = normalize_severity(
        native_severity,
        payload=payload,
    )

    attributes = base_attributes(
        record_data,
        payload,
        native_signal_type,
    )

    attributes.update(
        provider_attributes(
            provider,
            native_signal_type,
            payload,
        )
    )

    safe_attributes = redact_attributes(
        attributes,
        policy=policy,
    )

    message = extract_message(
        native_signal_type,
        payload,
    )

    return build_signal(
        provider=provider,
        resource_id=(resource.effective_id()),
        signal_type=(common_signal_type),
        timestamp=signal_timestamp(
            record_data,
            payload,
        ),
        environment_id=(resource.environment_id),
        signal_name=signal_name(
            native_signal_type,
            payload,
        ),
        severity=severity,
        value=signal_value(
            common_signal_type,
            payload,
        ),
        unit=metric_unit(payload),
        message=message,
        attributes=(safe_attributes),
        source_record_id=(source_record_id(record_data)),
    )


def signal_is_error(
    signal: Signal,
) -> bool:
    return signal.severity in {
        Severity.ERROR.value,
        Severity.CRITICAL.value,
        Severity.ALERT.value,
        Severity.EMERGENCY.value,
    }


def signal_is_warning(
    signal: Signal,
) -> bool:
    return signal.severity == Severity.WARNING.value


def signal_is_metric(
    signal: Signal,
) -> bool:
    return signal.signal_type == SignalType.METRIC.value


def signal_is_event(
    signal: Signal,
) -> bool:
    return signal.signal_type == SignalType.EVENT.value


def signal_is_log(
    signal: Signal,
) -> bool:
    return signal.signal_type == SignalType.LOG.value


def signal_is_state(
    signal: Signal,
) -> bool:
    return signal.signal_type in {
        SignalType.RESOURCE_STATE.value,
        SignalType.STATE_CHANGE.value,
        SignalType.HEALTH.value,
    }


def signal_context(
    signal: Signal,
) -> Dict[str, Any]:
    """
    Small indexing representation for the future
    correlation engine.
    """

    return {
        "resource_id": (signal.resource_id),
        "provider": (signal.provider),
        "environment_id": (signal.environment_id),
        "signal_type": (signal.signal_type),
        "signal_name": (signal.signal_name),
        "timestamp": (signal.timestamp),
        "severity": (signal.severity),
    }
