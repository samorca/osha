from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from src.collectors.openstack.discovery import (
    OpenStackContext,
    service_available,
)

from src.collectors.openstack.schema import (
    as_dict,
    error,
    record,
    sanitize,
    unavailable,
)


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def normalize_endpoint(
    endpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Convert one Keystone catalog endpoint to a
    safe OSHA representation.

    Authentication credentials and tokens do not
    belong in endpoint records.
    """

    payload = {
        "id": optional_string(
            endpoint.get("id")
        ),
        "interface": (
            optional_string(
                endpoint.get(
                    "interface"
                )
            )
        ),
        "region": (
            optional_string(
                endpoint.get(
                    "region"
                )
            )
        ),
        "region_id": (
            optional_string(
                endpoint.get(
                    "region_id"
                )
            )
        ),
        "url": optional_string(
            endpoint.get("url")
        ),
        "enabled": endpoint.get(
            "enabled"
        ),
    }

    return sanitize(
        payload
    )


def service_catalog_items(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Retrieve the service catalog associated with
    the authenticated Keystone token.

    This describes service discovery from the
    caller's perspective.
    """

    records: List[dict] = []

    try:
        auth = (
            ctx.conn.session.auth
        )

        access = auth.get_access(
            ctx.conn.session
        )

        catalog_object = (
            access.service_catalog
        )

        catalog = getattr(
            catalog_object,
            "catalog",
            [],
        )

        if not isinstance(
            catalog,
            list,
        ):
            catalog = []

        for service in catalog:
            if not isinstance(
                service,
                Mapping,
            ):
                continue

            service_id = (
                optional_string(
                    service.get("id")
                )
            )

            service_name = (
                optional_string(
                    service.get(
                        "name"
                    )
                )
            )

            service_type = (
                optional_string(
                    service.get(
                        "type"
                    )
                )
            )

            raw_endpoints = (
                service.get(
                    "endpoints"
                )
                or []
            )

            endpoints = []

            if isinstance(
                raw_endpoints,
                list,
            ):
                for endpoint in (
                    raw_endpoints
                ):
                    if not isinstance(
                        endpoint,
                        Mapping,
                    ):
                        continue

                    endpoints.append(
                        normalize_endpoint(
                            endpoint
                        )
                    )

            regions = sorted(
                {
                    str(
                        endpoint[
                            "region"
                        ]
                    )
                    for endpoint
                    in endpoints
                    if endpoint.get(
                        "region"
                    )
                }
            )

            interfaces = sorted(
                {
                    str(
                        endpoint[
                            "interface"
                        ]
                    )
                    for endpoint
                    in endpoints
                    if endpoint.get(
                        "interface"
                    )
                }
            )

            payload = {
                "service_id": (
                    service_id
                ),
                "service_name": (
                    service_name
                ),
                "service_type": (
                    service_type
                ),
                "endpoint_count": (
                    len(endpoints)
                ),
                "regions": regions,
                "interfaces": (
                    interfaces
                ),
                "endpoints": endpoints,
            }

            resource_id = (
                service_id
                or service_type
                or service_name
                or "unknown-service"
            )

            records.append(
                record(
                    "service_catalog",
                    "openstack_service",
                    payload,
                    cloud_id=(
                        ctx.cloud_id
                    ),
                    region=(
                        ctx.region
                    ),
                    environment_id=(
                        ctx.environment_id
                    ),
                    resource_id=(
                        resource_id
                    ),
                    source="keystone",
                )
            )

    except Exception as exc:
        records.append(
            error(
                "service_catalog",
                exc,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                operation=(
                    "read_service_catalog"
                ),
            )
        )

    return records


def compute_service_payload(
    service: Any,
) -> Dict[str, Any]:
    """
    Convert an openstacksdk Nova Service
    resource into an OSHA representation.
    """

    raw = as_dict(
        service
    )

    service_id = (
        raw.get("id")
        or getattr(
            service,
            "id",
            None,
        )
    )

    host = (
        raw.get("host")
        or getattr(
            service,
            "host",
            None,
        )
    )

    binary = (
        raw.get("binary")
        or raw.get("name")
        or getattr(
            service,
                "binary",
            None,
        )
        or getattr(
            service,
            "name",
            None,
        )
    )

    state = (
        raw.get("state")
        or getattr(
            service,
            "state",
          )
    )

    status = (
        raw.get("status")
        or getattr(
            service,
            "status",
            None,
        )
    )

    availability_zone = (
        raw.get(
            "availability_zone"
        )
        or raw.get("zone")
        or getattr(
            service,
            "availability_zone",
            None,
        )
    )

    disabled_reason = (
        raw.get(
            "disabled_reason"
        )
        or getattr(
            service,
            "disabled_reason",
            None,
        )
    )

    forced_down = raw.get(
        "is_forced_down"
    )

    if forced_down is None:
        forced_down = getattr(
            service,
            "is_forced_down",
            None,
        )

    updated_at = (
        raw.get("updated_at")
        or getattr(
            service,
            "updated_at",
            None,
        )
    )

    state_text = (
        optional_string(
            state
        )
    )

    status_text = (
        optional_string(
            status
        )
    )

    normalized_state = (
        state_text.lower()
        if state_text
        else None
    )

    normalized_status = (
        status_text.lower()
        if status_text
        else None
    )

    healthy: Optional[bool] = (
        None
    )

    if (
        normalized_state == "up"
        and normalized_status
        in {
            None,
            "enabled",
        }
        and forced_down
        is not True
    ):
        healthy = True

    elif (
        normalized_state == "down"
        or normalized_status
        == "disabled"
        or forced_down
        is True
    ):
        healthy = False

    payload = {
        "service_id": (
            optional_string(
                service_id
            )
        ),
        "host": (
            optional_string(
                host
            )
        ),
        "binary": (
            optional_string(
                binary
            )
        ),
        "state": (
            state_text
        ),
        "status": (
            status_text
        ),
        "healthy": healthy,
        "availability_zone": (
            optional_string(
                availability_zone
            )
        ),
        "disabled_reason": (
            optional_string(
                disabled_reason
            )
        ),
        "is_forced_down": (
            forced_down
        ),
        "updated_at": (
            str(updated_at)
            if updated_at
            is not None
            else None
        ),
    }

    return sanitize(
        payload
    )


def compute_service_id(
    payload: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Construct a stable identifier when Nova's
    service ID is not available.
    """

    service_id = (
        optional_string(
            payload.get(
                "service_id"
            )
        )
    )

    if service_id:
        return service_id

    host = optional_string(
        payload.get("host")
    )

    binary = optional_string(
        payload.get(
            "binary"
        )
    )

    pieces = [
        piece
        for piece in (
            host,
            binary,
        )
        if piece
    ]

    if pieces:
        return ":".join(
            pieces
        )

    return (
        "unknown-compute-service"
    )


def nova_service_items(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect Nova service state.

    This is read-only. It does not enable,
    disable, start, stop, or force-down Nova
    services.
    """

    records: List[dict] = []

    if not service_available(
        ctx,
        "compute",
    ):
        records.append(
            unavailable(
                "compute",
                (
                    "Compute service endpoint "
                    "is unavailable"
                ),
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
            )
        )

        return records

    try:
        for service in (
            ctx.conn.compute.services()
        ):
            payload = (
                compute_service_payload(
                    service
                )
            )

            resource_id = (
                compute_service_id(
                    payload
                )
            )

            records.append(
                record(
                    "compute",
                    "compute_service",
                    payload,
                    cloud_id=(
                        ctx.cloud_id
                    ),
                    region=(
                        ctx.region
                    ),
                    environment_id=(
                        ctx.environment_id
                    ),
                    resource_id=(
                        resource_id
                    ),
                    timestamp=(
                        payload.get(
                            "updated_at"
                        )
                    ),
                    source="nova",
                )
            )

    except Exception as exc:
        records.append(
            error(
                "compute",
                exc,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                operation=(
                    "list_compute_services"
                ),
            )
        )

    return records


def service_summary(
    records: List[dict],
) -> dict:
    """
    Build an informational summary from records
    already collected by this module.

    This summary never replaces the individual
    records.
    """

    catalog_services = 0
    compute_services = 0
    healthy_compute = 0
    unhealthy_compute = 0
    unknown_compute = 0
    errors = 0

    service_types = set()
    regions = set()

    for item in records:
        signal_type = item.get(
            "signal_type"
        )

        payload = (
            item.get("payload")
            or {}
        )

        if (
            signal_type
            == "openstack_service"
        ):
            catalog_services += 1

            service_type = (
                payload.get(
                    "service_type"
                )
            )

            if service_type:
                service_types.add(
                    str(service_type)
                )

            for region in (
                payload.get(
                    "regions"
                )
                or []
            ):
                regions.add(
                    str(region)
                )

        elif (
            signal_type
            == "compute_service"
        ):
            compute_services += 1

            healthy = (
                payload.get(
                    "healthy"
                )
            )

            if healthy is True:
                healthy_compute += 1
            elif healthy is False:
                unhealthy_compute += 1
            else:
                unknown_compute += 1

        elif (
            signal_type
            == "collector_error"
        ):
            errors += 1

    return {
        "catalog_service_count": (
            catalog_services
        ),
        "service_types": sorted(
            service_types
        ),
        "regions": sorted(
            regions
        ),
        "compute_service_count": (
            compute_services
        ),
        "healthy_compute_services": (
            healthy_compute
        ),
        "unhealthy_compute_services": (
            unhealthy_compute
        ),
        "unknown_compute_services": (
            unknown_compute
        ),
        "collector_error_count": (
            errors
        ),
    }


def collect(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect OSHA's OpenStack service inventory.

    Sources:
      Keystone service catalog
      Nova service-state API
    """

    records: List[dict] = []

    records.extend(
        service_catalog_items(
            ctx
        )
    )

    records.extend(
        nova_service_items(
            ctx
        )
    )

    summary = service_summary(
        records
    )

    records.append(
        record(
            "services",
            "service_summary",
            summary,
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
