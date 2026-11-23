from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.collectors.openstack.discovery import (
    OpenStackContext,
)

from src.collectors.openstack.schema import (
    record,
    sanitize,
)


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def openstack_cli_available() -> bool:
    return (
        shutil.which(
            "openstack"
        )
        is not None
    )


def safe_environment() -> Dict[str, str]:
    """
    Pass the current environment to
    python-openstackclient.

    Environment variables are never copied into
    OSHA telemetry records.
    """

    return dict(os.environ)


def run_openstack_json(
    arguments: List[str],
    *,
    timeout: int = 60,
) -> tuple[
    bool,
    Any,
    str,
]:
    """
    Execute a read-only OpenStack CLI operation
    and request machine-readable JSON.

    Full authentication environment variables
    are deliberately never returned.
    """

    if not openstack_cli_available():
        return (
            False,
            None,
            (
                "openstack CLI is not "
                "installed"
            ),
        )

    command = [
        "openstack",
        *arguments,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
            env=safe_environment(),
        )

    except subprocess.TimeoutExpired:
        return (
            False,
            None,
            (
                "OpenStack command "
                "timed out"
            ),
        )

    except OSError as exc:
        return (
         False,
            None,
            (
                "Unable to execute "
                f"OpenStack CLI: {exc}"
            ),
        )

    if result.returncode != 0:
        message = (
            result.stderr.strip()
            or "OpenStack event query failed"
        )

        return (
            False,
            None,
            message[:1000],
        )

    output = result.stdout.strip()

    if not output:
        return (
            True,
            [],
            "",
        )

    try:
        parsed = json.loads(
            output
        )
    except json.JSONDecodeError:
        return (
            False,
            None,
            (
                "OpenStack event query did not "
                "return valid JSON"
            ),
        )

    return (
        True,
        parsed,
        "",
    )


def normalize_rows(
    value: Any,
) -> List[dict]:
    """
    Normalize JSON returned by OpenStack CLI.
    """

    if isinstance(
        value,
        list,
    ):
        return [
            sanitize(
                dict(item)
            )
            for item in value
            if isinstance(
                item,
                Mapping,
            )
        ]

    if isinstance(
        value,
        Mapping,
    ):
        return [
            sanitize(
                dict(value)
            )
        ]

    return []


def lookup(
    payload: Mapping[str, Any],
    *names: str,
) -> Any:
    """
    Case-insensitive lookup for CLI columns.
    """

    lowered = {
        str(key).lower(): value
        for key, value
        in payload.items()
    }

    for name in names:
        if name in payload:
            return payload[name]

        value = lowered.get(
            name.lower()
        )

        if value is not None:
            return value

    return None


def event_message_id(
    event: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        lookup(
            event,
            "message_id",
            "message id",
            "id",
        )
    )


def event_type(
    event: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        lookup(
            event,
            "event_type",
            "event type",
            "type",
        )
    )


def event_timestamp(
    event: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        lookup(
            event,
            "generated",
            "timestamp",
            "generated_at",
            "generated at",
        )
    )


def determine_service(
    event_name: Optional[str],
) -> Optional[str]:
    """
    Derive a broad OpenStack service family from
    common event-type prefixes.

    This is OSHA's classification logic, not an
    OpenStack API field.
    """

    if not event_name:
        return None

    value = event_name.lower()

    prefixes = {
        "compute.": "compute",
        "instance.": "compute",
        "server.": "compute",

        "network.": "network",
        "port.": "network",
        "subnet.": "network",
        "router.": "network",
        "floatingip.": "network",

        "volume.": (
            "block_storage"
        ),
        "snapshot.": (
            "block_storage"
        ),
        "backup.": (
            "block_storage"
        ),

        "image.": "image",

        "identity.": "identity",
        "user.": "identity",
        "project.": "identity",
        "role.": "identity",

        "orchestration.": (
            "orchestration"
        ),
        "stack.": (
            "orchestration"
        ),
    }

    for prefix, service in (
        prefixes.items()
    ):
        if value.startswith(
            prefix
        ):
            return service

    return None


def determine_action(
    name: Optional[str],
) -> Optional[str]:
    """
    Extract a broad lifecycle action from an
    event type.

    This is deliberately conservative.
    """

    if not name:
        return None

    tokens = [
        token
        for token in (
            name.lower()
            .replace(
                "-",
                ".",
            )
            .split(".")
        )
        if token
    ]

    known_actions = (
        "create",
        "delete",
        "update",
        "resize",
        "rebuild",
        "reboot",
        "start",
        "stop",
        "pause",
        "unpause",
        "suspend",
        "resume",
        "attach",
        "detach",
        "migrate",
        "evacuate",
        "restore",
        "snapshot",
        "backup",
        "associate",
        "disassociate",
    )

    for action in (
        known_actions
    ):
        if action in tokens:
            return action

    return None


def collect_event_list(
    ctx: OpenStackContext,
) -> tuple[
    List[dict],
    List[dict],
]:
    """
    Query Ceilometer event inventory.

    Returns:
        OSHA records
        normalized event-list rows
    """

    records: List[dict] = []

    success, data, message = (
        run_openstack_json(
            [
                "event",
                "list",
                "-f",
                "json",
            ]
        )
    )

    if not success:
        records.append(
            record(
                "events",
                "collector_status",
                {
                    "component": (
                        "ceilometer"
                    ),
                    "capability": (
                        "events"
                    ),
                    "available": False,
                    "reason": message,
                },
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                source="ceilometer",
            )
        )

        return (
            records,
            [],
        )

    rows = normalize_rows(
        data
    )

    records.append(
        record(
            "events",
            "collector_status",
            {
                "component": (
                    "ceilometer"
                ),
                "capability": (
                    "events"
                ),
                "available": True,
            },
            cloud_id=(
                ctx.cloud_id
            ),
            region=ctx.region,
            environment_id=(
                ctx.environment_id
            ),
            source="ceilometer",
        )
    )

    return (
        records,
        rows,
    )


def show_event(
    message_id: str,
) -> Optional[dict]:
    """
    Attempt to retrieve the full structured
    representation of a single event.

    Raw event payloads are removed below even if
    exposed by the CLI/API.
    """

    success, data, _ = (
        run_openstack_json(
            [
                "event",
                "show",
                message_id,
                "-f",
                "json",
            ]
        )
    )

    if not success:
        return None

    if not isinstance(
        data,
        Mapping,
    ):
        return None

    return sanitize(
        dict(data)
    )


def strip_raw_payload(
    event: Mapping[str, Any],
) -> dict:
    """
    Remove raw notification/event message
    contents.

    Ceilometer can store raw notification data
    for auditing. OSHA does not ingest those raw
    payloads by default.
    """

    output = {}

    sensitive_raw_names = {
        "raw",
        "raw_event",
        "raw event",
        "notification",
        "notification_payload",
        "notification payload",
    }

    for key, value in (
        event.items()
    ):
        key_text = str(
            key
        )

        if (
            key_text.lower()
            in sensitive_raw_names
        ):
            continue

        output[
            key_text
        ] = value

    return sanitize(
        output
    )


def extract_traits(
    payload: Mapping[str, Any],
) -> Any:
    """
    Retrieve structured event traits when the
    CLI exposes them.
    """

    traits = lookup(
        payload,
        "traits",
    )

    if traits is None:
        return []

    return sanitize(traits)


def collect_events(
    ctx: OpenStackContext,
    *,
    maximum_event_details: int = 500,
) -> List[dict]:
    """
    Collect structured Ceilometer events.

    List results are always used.

    A bounded number of events can additionally
    be expanded with event show to obtain traits.
    """

    records, rows = (
        collect_event_list(
            ctx
        )
    )

    if not rows:
        return records

    for index, row in enumerate(
        rows
    ):
        message_id = (
            event_message_id(
                row
            )
        )

        base_type = event_type(
            row
        )

        generated = (
            event_timestamp(
                row
            )
        )

        detail = None

        if (
            message_id
            and index
            < maximum_event_details
        ):
            detail = show_event(
                message_id
            )

        combined = dict(row)

        if isinstance(
            detail,
            Mapping,
        ):
            combined.update(
                detail
            )

        combined = (
            strip_raw_payload(
                combined
            )
        )

        final_type = (
            event_type(
                combined
            )
            or base_type
        )

        final_timestamp = (
            event_timestamp(
                combined
            )
            or generated
        )

        service_family = (
            determine_service(
                final_type
            )
        )

        action = (
            determine_action(
                final_type
            )
        )

        traits = extract_traits(
            combined
        )

        payload = {
            "message_id": (
                message_id
            ),
            "event_type": (
                final_type
            ),
            "service_family": (
                service_family
            ),
            "action": action,
            "traits": traits,
            "event": combined,
            "raw_payload_collected": (
                False
            ),
        }

        records.append(
            record(
                "events",
                "openstack_event",
                payload,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                resource_id=(
                    message_id
                    or final_type
                    or "unknown-event"
                ),
                timestamp=(
                    final_timestamp
                ),
                source="ceilometer",
            )
        )

    return records


def event_summary(
    records: Iterable[dict],
) -> dict:
    event_types = Counter()
    services = Counter()
    actions = Counter()

    event_count = 0

    for item in records:
        if (
            item.get("signal_type")
            != "openstack_event"
        ):
            continue

        event_count += 1

        payload = (
            item.get("payload")
            or {}
        )

        event_name = (
            payload.get(
                "event_type"
            )
            or "unknown"
        )

        service = (
            payload.get(
                "service_family"
            )
            or "unknown"
        )

        action = (
            payload.get("action")
            or "unknown"
        )

        event_types[
            str(event_name)
        ] += 1

        services[
            str(service)
        ] += 1

        actions[
            str(action)
        ] += 1

    return {
        "event_count": (
            event_count
        ),
        "event_types": (
            dict(
                event_types
            )
        ),
        "service_families": (
            dict(services)
        ),
        "actions": (
            dict(actions)
        ),
    }


def collect(
    ctx: OpenStackContext,
    *,
    maximum_event_details: int = 500,
) -> List[dict]:
    """
    Collect OpenStack operational events.

    Source:
        Ceilometer event API through
        python-openstackclient's OpenStack CLI.

    Read-only:
        event list
        event show

    Raw notification payloads are not persisted.
    """

    if not openstack_cli_available():
        return [
            record(
                "events",
                "collector_status",
                {
                    "component": (
                        "openstack_cli"
                    ),
                    "available": False,
                    "reason": (
                        "openstack CLI is not "
                        "installed"
                    ),
                },
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                source="collector",
            )
        ]

    records = collect_events(
        ctx,
        maximum_event_details=(
            maximum_event_details
        ),
    )

    records.append(
        record(
            "events",
            "event_summary",
            event_summary(
                records
            ),
            cloud_id=(
                ctx.cloud_id
            ),
            region=ctx.region,
            environment_id=(
                ctx.environment_id
            ),
            resource_id=(
                ctx.cloud_id
            ),
            source="osha",
        )
    )

    return records
