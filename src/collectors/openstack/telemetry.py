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
    error,
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
    Return the current process environment for
    the OpenStack CLI subprocess.

    Credentials may be present in OS_* variables
    because they are required by openstackclient.

    The environment is passed to the subprocess
    but is NEVER persisted in OSHA telemetry.
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
    Execute a read-only openstack CLI command
    requesting JSON output.

    Returns:

        success
        parsed JSON
        error message

    stdout/stderr containing secrets are not
    written directly to telemetry records.
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
                f"Unable to execute "
                f"openstack CLI: {exc}"
            ),
        )

    if result.returncode != 0:
        # Do not persist complete stderr because
        # authentication tooling may print
        # sensitive configuration information.
        message = (
            result.stderr.strip()
            or "OpenStack command failed"
        )

        # Store only a bounded message.
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
                "OpenStack command did not "
                "return valid JSON"
            ),
        )

    return (
        True,
        parsed,
        "",
    )


def normalize_json_rows(
    value: Any,
) -> List[dict]:
    """
    Convert common OpenStack CLI JSON output
    forms into a list of dictionaries.
    """

    if isinstance(
        value,
        list,
    ):
        return [
            sanitize(item)
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


def find_value(
    payload: Mapping[str, Any],
    *names: str,
) -> Any:
    """
    Look up a value while tolerating CLI column
    capitalization differences.
    """

    lowered = {
        str(key).lower(): value
        for key, value
        in payload.items()
    }

    for name in names:
        if name in payload:
            return payload[name]

        candidate = lowered.get(
            name.lower()
        )

        if candidate is not None:
            return candidate

    return None


def resource_id_from_row(
    row: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        find_value(
            row,
            "id",
            "resource_id",
            "resource id",
            "uuid",
        )
    )


def metric_id_from_row(
    row: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        find_value(
            row,
            "id",
            "metric_id",
            "metric id",
        )
    )


def alarm_id_from_row(
    row: Mapping[str, Any],
) -> Optional[str]:
    return optional_string(
        find_value(
            row,
            "alarm_id",
            "alarm id",
            "id",
        )
    )


def collect_metric_resources(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect Gnocchi metric-resource inventory.

    No measurements are requested here. This
    gives OSHA the resource/metric topology first.
    """

    records: List[dict] = []

    success, data, message = (
        run_openstack_json(
            [
                "metric",
                "resource",
                "list",
                "-f",
                "json",
            ]
        )
    )

    if not success:
        records.append(
            record(
                "telemetry",
                "collector_status",
                {
                    "component": (
                        "gnocchi"
                    ),
                    "capability": (
                        "metric_resources"
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
                source="gnocchi",
            )
        )

        return records

    rows = normalize_json_rows(
        data
    )

    records.append(
        record(
            "telemetry",
            "collector_status",
            {
                "component": "gnocchi",
                "capability": (
                    "metric_resources"
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
            source="gnocchi",
        )
    )

    for row in rows:
        resource_id = (
            resource_id_from_row(
                row
            )
        )

        records.append(
            record(
                "telemetry",
                "metric_resource",
                row,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                resource_id=(
                    resource_id
                ),
                source="gnocchi",
            )
        )

    return records


def collect_metrics(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect Gnocchi metric definitions.

    A metric definition links a metric to its
    identifier and associated resource metadata.
    Actual measurements are collected separately.
    """

    records: List[dict] = []

    success, data, message = (
        run_openstack_json(
            [
                "metric",
                "list",
                "-f",
                "json",
            ]
        )
    )

    if not success:
        records.append(
            record(
                "telemetry",
                "collector_status",
                {
                    "component": (
                        "gnocchi"
                    ),
                    "capability": (
                        "metrics"
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
                source="gnocchi",
            )
        )

        return records

    rows = normalize_json_rows(
        data
    )

    for row in rows:
        metric_id = (
            metric_id_from_row(
                row
            )
        )

        records.append(
            record(
                "telemetry",
                "metric_definition",
                row,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                resource_id=(
                    metric_id
                ),
                source="gnocchi",
            )
        )

    return records


def extract_metric_names(
    metric_row: Mapping[
        str,
        Any,
    ],
) -> List[str]:
    """
    Extract candidate metric names from a
    Gnocchi metric-list row.

    Different client/plugin versions may expose
    columns differently, so this is defensive.
    """

    names: List[str] = []

    for key in (
        "name",
        "metric",
        "metric_name",
        "metric name",
    ):
        value = find_value(
            metric_row,
            key,
        )

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ):
            if value.strip():
                names.append(
                    value.strip()
                )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                text = optional_string(
                    item
                )

                if text:
                    names.append(
                        text
                    )

    return list(
        dict.fromkeys(
            names
        )
    )


def collect_measurements(
    ctx: OpenStackContext,
    metric_records: List[dict],
    *,
    maximum_metrics: int = 100,
    granularity: Optional[int] = None,
) -> List[dict]:
    """
    Collect measurements for a bounded number of
    discovered metrics.

    Bounding the number prevents first-time
    discovery from unexpectedly querying every
    historical metric in a large cloud.
    """

    records: List[dict] = []

    candidates = []

    for item in metric_records:
        if (
            item.get("signal_type")
            != "metric_definition"
        ):
            continue

        payload = (
            item.get("payload")
            or {}
        )

        metric_id = (
            metric_id_from_row(
                payload
            )
        )

        names = (
         (
                payload
            )
        )

        if metric_id:
            candidates.append(
                (
                    metric_id,
                    (
                        names[0]
                        if names
                        else None
                    ),
                )
            )

    candidates = candidates[
        :maximum_metrics
    ]

    for (
        metric_id,
        metric_name,
    ) in candidates:
        command = [
            "metric",
            "measures",
            "show",
            metric_id,
            "-f",
            "json",
        ]

        if granularity:
            command.extend(
                [
                    "--granularity",
                    str(granularity),
                ]
            )

        success, data, _ = (
            run_openstack_json(
                command
            )
        )

        if not success:
            # Do not produce one collector error
            # for every inaccessible metric.
            continue

        rows = (
            data
            if isinstance(
                data,
                list,
            )
            else []
        )

        for measurement in rows:
            timestamp = None
            measurement_granularity = None
            value = None

            if isinstance(
                measurement,
                list,
            ):
                if (
                    len(measurement)
                    >= 3
                ):
                    timestamp = (
                        measurement[0]
                    )

                    measurement_granularity = (
                        measurement[1]
                    )

                    value = (
                        measurement[2]
                    )

            elif isinstance(
                measurement,
                Mapping,
            ):
                timestamp = find_value(
                    measurement,
                    "timestamp",
                )

                measurement_granularity = (
                    find_value(
                        measurement,
                        "granularity",
                    )
                )

                value = find_value(
                    measurement,
                    "value",
                )

            if timestamp is None:
                continue

            payload = {
                "metric_id": (
                    metric_id
                ),
                "metric_name": (
                    metric_name
                ),
                "granularity": (
                    measurement_granularity
                ),
                "value": value,
            }

            records.append(
                record(
                    "telemetry",
                    "metric",
                    payload,
                    cloud_id=(
                        ctx.cloud_id
                    ),
                    region=ctx.region,
                    environment_id=(
                        ctx.environment_id
                    ),
                    resource_id=(
                        metric_id
                    ),
                    timestamp=(
                        timestamp
                    ),
                    source="gnocchi",
                )
            )

    return records


def collect_alarms(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect Aodh alarm definitions/state.

    This is read-only. OSHA does not create,
    change, acknowledge, or delete alarms.
    """

    records: List[dict] = []

    success, data, message = (
        run_openstack_json(
            [
                "alarm",
                "list",
                "-f",
                "json",
            ]
        )
    )

    if not success:
        records.append(
            record(
                "telemetry",
                "collector_status",
                {
                    "component": "aodh",
                    "capability": "alarms",
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
                source="aodh",
            )
        )

        return records

    rows = normalize_json_rows(
        data
    )

    records.append(
        record(
            "telemetry",
            "collector_status",
            {
                "component": "aodh",
                "capability": "alarms",
                "available": True,
            },
            cloud_id=(
                ctx.cloud_id
            ),
            region=ctx.region,
            environment_id=(
                ctx.environment_id
            ),
            source="aodh",
        )
    )

    for row in rows:
        alarm_id = alarm_id_from_row(
            row
        )

        state = optional_string(
            find_value(
                row,
                "state",
            )
        )

        enabled = find_value(
            row,
            "enabled",
        )

        severity = optional_string(
            find_value(
                row,
                "severity",
            )
        )

        state_reason = (
            optional_string(
                find_value(
                    row,
                    "state_reason",
                    "state reason",
                )
            )
        )

        payload = {
            **row,
            "alarm_id": alarm_id,
            "state": state,
            "enabled": enabled,
            "severity": severity,
            "state_reason": (
                state_reason
            ),
        }

        records.append(
            record(
                "telemetry",
                "alarm",
                sanitize(payload),
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                resource_id=(
                    alarm_id
                ),
                source="aodh",
            )
        )

    return records


def telemetry_summary(
    records: List[dict],
) -> dict:
    counts = Counter()

    alarm_states = Counter()
    alarm_severities = Counter()

    metric_values = 0

    gnocchi_available = None
    aodh_available = None

    for item in records:
        signal = str(
            item.get(
                "signal_type",
                "unknown",
            )
        )

        counts[signal] += 1

        payload = (
            item.get("payload")
            or {}
        )

        if signal == "metric":
            metric_values += 1

        elif signal == "alarm":
            state = (
                payload.get("state")
                or "unknown"
            )

            severity = (
                payload.get(
                    "severity"
                )
                or "unknown"
            )

            alarm_states[
                str(state)
            ] += 1

            alarm_severities[
                str(severity)
            ] += 1

        elif (
            signal
            == "collector_status"
        ):
            component = (
                payload.get(
                    "component"
                )
            )

            available_value = (
                payload.get(
                    "available"
                )
            )

            if (
                component
                == "gnocchi"
            ):
                gnocchi_available = (
                    bool(
                        available_value
                    )
                )

            elif component == "aodh":
                aodh_available = (
                    bool(
                        available_value
                    )
                )

    return {
        "record_counts": (
            dict(counts)
        ),
        "metric_measurement_count": (
            metric_values
        ),
        "alarm_states": (
            dict(
                alarm_states
            )
        ),
        "alarm_severities": (
            dict(
                alarm_severities
            )
        ),
        "components": {
            "gnocchi_available": (
                gnocchi_available
            ),
            "aodh_available": (
                aodh_available
            ),
        },
    }


def collect(
    ctx: OpenStackContext,
    *,
    include_measurements: bool = True,
    maximum_metrics: int = 100,
    granularity: Optional[int] = None,
) -> List[dict]:
    """
    Collect optional OpenStack operational
    telemetry.

    Sources:
      Gnocchi metric resources
      Gnocchi metric definitions
      Gnocchi metric measurements
      Aodh alarms

    Ceilometer is the data-collection layer in a
    typical telemetry architecture. Events are
    handled separately by events.py.
    """

    records: List[dict] = []

    if not openstack_cli_available():
        records.append(
            record(
                "telemetry",
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
        )

        return records

    resource_records = (
        collect_metric_resources(
            ctx
        )
    )

    records.extend(
        resource_records
    )

    metric_records = (
        collect_metrics(
            ctx
        )
    )

    records.extend(
        metric_records
    )

    if include_measurements:
        records.extend(
            collect_measurements(
                ctx,
                metric_records,
                maximum_metrics=(
                    maximum_metrics
                ),
                granularity=(
                    granularity
                ),
            )
        )

    records.extend(
        collect_alarms(
            ctx
        )
    )

    records.append(
        record(
            "telemetry",
            "telemetry_summary",
            telemetry_summary(
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
