from __future__ import annotations

import hashlib
import json
import re
from datetime import (
    date,
    datetime,
    timezone,
)
from typing import Any, Mapping


SCHEMA_VERSION = "1"

RECORD_TYPE = (
    "openstack_telemetry"
)

PROVIDER = "openstack"


# Keys matching these patterns should never be
# persisted verbatim by the collector.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)"
    r"(password|passwd|secret|token|"
    r"api[_-]?key|private[_-]?key|"
    r"credential|credentials|"
    r"application[_-]?credential|"
    r"auth[_-]?token|access[_-]?key|"
    r"secret[_-]?key)"
)


# Useful when values rather than keys contain
# recognizable secret material.
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN "
    r"(?:RSA |EC |OPENSSH )?"
    r"PRIVATE KEY-----",
    re.IGNORECASE,
)


BEARER_PATTERN = re.compile(
    r"(?i)"
    r"\bbearer\s+"
    r"[A-Za-z0-9._~+/=-]+"
)


def utc_now() -> str:
    """
    Return a timezone-aware UTC timestamp.

    Stored timestamps use ISO 8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def stringify_timestamp(
    value: Any,
) -> str | None:
    """
    Convert common timestamp representations
    into serializable strings.

    This function does not invent a timestamp
    when the source value is missing.
    """

    if value is None:
        return None

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
        date,
    ):
        return value.isoformat()

    text = str(value).strip()

    return text if text else None


def sensitive_key(
    key: Any,
) -> bool:
    """
    Return True when a dictionary/object key
    appears to represent authentication or
    secret material.
    """

    if key is None:
        return False

    return bool(
        SENSITIVE_KEY_PATTERN.search(
            str(key)
        )
    )


def redact_string(
    value: str,
) -> str:
    """
    Redact obvious private-key and bearer-token
    material from arbitrary strings.

    General OpenStack configuration secrets
    should primarily be removed by key name.
    """

    if PRIVATE_KEY_PATTERN.search(
        value
    ):
        return "[REDACTED_PRIVATE_KEY]"

    value = BEARER_PATTERN.sub(
        "Bearer [REDACTED]",
        value,
    )

    return value


def sanitize(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    """
    Recursively convert a value into JSON-safe
    structures while redacting secret-looking
    fields.

    This is intentionally defensive because
    OpenStack API/resource values can contain
    nested mappings and lists.
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
        ),
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return redact_string(
            value
        )

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return stringify_timestamp(
            value
        )

    if isinstance(
        value,
        bytes,
    ):
        return (
            value.decode(
                "utf-8",
                errors="replace",
            )
        )

    if isinstance(
        value,
        Mapping,
    ):
        output = {}

        for key, item in (
            value.items()
        ):
            key_text = str(key)

            if sensitive_key(
                key_text
            ):
                output[
                    key_text
                ] = "[REDACTED]"
                continue

            output[
                key_text
            ] = sanitize(
                item,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
            )

        return output

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
            sanitize(
                item,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
            )
            for item in value
        ]

    # OpenStack SDK Resource objects generally
    # expose to_dict().
    to_dict = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(to_dict):
        try:
            converted = to_dict(
                computed=False
            )
        except TypeError:
            try:
                converted = to_dict()
            except Exception:
                converted = None
        except Exception:
            converted = None

        if converted is not None:
            return sanitize(
                converted,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
            )

    return redact_string(
        str(value)
    )


def as_dict(
    resource: Any,
) -> dict:
    """
    Convert an OpenStack SDK resource or mapping
    into a sanitized dictionary.
    """

    if resource is None:
        return {}

    if isinstance(
        resource,
        Mapping,
    ):
        value = sanitize(
            resource
        )

        return (
            value
            if isinstance(
                value,
                dict,
            )
            else {}
        )

    to_dict = getattr(
        resource,
        "to_dict",
        None,
    )

    if callable(to_dict):
        try:
            value = to_dict(
                computed=False
            )
        except TypeError:
            value = to_dict()
        except Exception:
            value = None

        if value is not None:
            result = sanitize(
                value
            )

            if isinstance(
                result,
                dict,
            ):
                return result

    # Provide a minimal representation for
    # unusual objects without exposing object
    # internals or credentials.
    result = {
        "id": sanitize(
            getattr(
                resource,
                "id",
                None,
            )
        ),
        "name": sanitize(
            getattr(
                resource,
                "name",
                None,
            )
        ),
        "value": sanitize(
            str(resource)
        ),
    }

    return {
        key: value
        for key, value
        in result.items()
        if value is not None
    }


def normalized_identifier(
    value: Any,
) -> str | None:
    """
    Convert resource identifiers into a stable
    optional string representation.
    """

    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def derive_resource_id(
    payload: Mapping[str, Any],
) -> str | None:
    """
    Attempt to identify a generic resource ID
    without relying on a particular OpenStack
    service.

    Individual collector modules should provide
    an explicit resource_id whenever possible.
    """

    candidate_keys = (
        "id",
        "uuid",
        "resource_id",
        "message_id",
        "project_id",
        "server_id",
        "volume_id",
        "network_id",
        "router_id",
        "port_id",
        "image_id",
        "name",
    )

    for key in candidate_keys:
        candidate = payload.get(
            key
        )

        identifier = (
            normalized_identifier(
                candidate
            )
        )

        if identifier:
            return identifier

    return None


def build_record_id(
    record_body: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Generate a deterministic SHA-256 identifier.

    collected_at is deliberately excluded so
    identical observations can be detected as
    duplicates later.
    """

    identity = {
        key: value
        for key, value
        in record_body.items()
        if key
        not in {
            "record_id",
            "collected_at",
        }
    }

    encoded = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
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


def record(
    service: str,
    signal_type: str,
    payload: Mapping[str, Any],
    *,
    cloud_id: str,
    region: str | None = None,
    resource_id: str | None = None,
    timestamp: Any = None,
    environment_id: str = "default",
    source: str | None = None,
) -> dict:
    """
    Create one canonical OSHA OpenStack
    telemetry record.

    All OpenStack collector modules should use
    this function rather than constructing JSON
    structures independently.
    """

    safe_payload = sanitize(
        dict(payload)
    )

    if not isinstance(
        safe_payload,
        dict,
    ):
        safe_payload = {}

    resolved_resource_id = (
        normalized_identifier(
            resource_id
        )
        or derive_resource_id(
            safe_payload
        )
    )

    observed_at = (
        stringify_timestamp(
            timestamp
        )
        or utc_now()
    )

    body = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "record_type": (
            RECORD_TYPE
        ),
        "provider": PROVIDER,
        "service": str(
            service
        ),
        "signal_type": str(
            signal_type
        ),
        "timestamp": (
            observed_at
        ),
        "collected_at": (
            utc_now()
        ),
        "environment_id": (
            str(
                environment_id
                or "default"
            )
        ),
        "cloud_id": str(
            cloud_id
            or "openstack-default"
        ),
        "region": (
            normalized_identifier(
                region
            )
        ),
        "resource_id": (
            resolved_resource_id
        ),
        "source": (
            normalized_identifier(
                source
            )
        ),
        "payload": safe_payload,
    }

    body["record_id"] = (
        build_record_id(
            body
        )
    )

    return body


def error(
    service: str,
    exc: Exception,
    *,
    cloud_id: str,
    region: str | None = None,
    environment_id: str = "default",
    operation: str | None = None,
    resource_id: str | None = None,
) -> dict:
    """
    Convert a collector exception into telemetry
    rather than allowing one service failure to
    stop the entire OpenStack collector.

    Exception class and message are preserved,
    but still pass through sanitization.
    """

    payload = {
        "collector": service,
        "operation": (
            operation
        ),
        "error_type": (
            type(exc).__name__
        ),
        "message": str(exc),
    }

    return record(
        service,
        "collector_error",
        payload,
        cloud_id=cloud_id,
        region=region,
        resource_id=resource_id,
        environment_id=(
            environment_id
        ),
        source="collector",
    )


def unavailable(
    service: str,
    reason: str,
    *,
    cloud_id: str,
    region: str | None = None,
    environment_id: str = "default",
) -> dict:
    """
    Record that an optional OpenStack capability
    is unavailable without treating it as a
    Python exception.
    """

    return record(
        service,
        "collector_status",
        {
            "available": False,
            "reason": reason,
        },
        cloud_id=cloud_id,
        region=region,
        environment_id=(
            environment_id
        ),
        source="collector",
    )


def available(
    service: str,
    *,
    cloud_id: str,
    region: str | None = None,
    environment_id: str = "default",
    details: Mapping[
        str,
        Any,
    ]
    | None = None,
) -> dict:
    """
    Record successful discovery of an OpenStack
    capability.
    """

    payload = {
        "available": True,
    }

    if details:
        payload.update(
            sanitize(
                dict(details)
            )
        )

    return record(
        service,
        "collector_status",
        payload,
        cloud_id=cloud_id,
        region=region,
        environment_id=(
            environment_id
        ),
        source="collector",
    )
