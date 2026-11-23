from __future__ import annotations

from collections import Counter
from typing import (
    Any,
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


def image_identifier(
    payload: Mapping[str, Any],
) -> str:
    """
    Obtain a stable identifier for a Glance
    image.
    """

    image_id = optional_string(
        payload.get("image_id")
    )

    if image_id:
        return image_id

    name = optional_string(
        payload.get("name")
    )

    if name:
        return name

    return "unknown-image"


def normalize_properties(
    value: Any,
) -> dict:
    """
    Normalize Glance custom properties.

    Secret-like property names are redacted by
    the shared schema sanitizer.
    """

    if not isinstance(
        value,
        Mapping,
    ):
        return {}

    sanitized = sanitize(
        dict(value)
    )

    return (
        sanitized
        if isinstance(
            sanitized,
            dict,
        )
        else {}
    )


def normalize_image(
    image: Any,
) -> dict:
    """
    Convert a Glance Image resource into OSHA's
    stable image representation.

    Image bytes are never downloaded by this
    collector.
    """

    raw = as_dict(image)

    image_id = (
        raw.get("id")
    )

    name = (
        raw.get("name")
    )

    status = optional_string(
        raw.get("status")
    )

    normalized_status = (
        status.lower()
        if status
        else None
    )

    healthy: Optional[bool] = (
        None
    )

    if normalized_status in {
        "active",
        "queued",
        "saving",
        "importing",
        "uploading",
    }:
        healthy = True

    elif normalized_status in {
        "killed",
        "deleted",
        "deactivated",
    }:
        healthy = False

    properties = (
        normalize_properties(
            raw.get(
                "properties"
            )
            or {}
        )
    )

    tags = raw.get(
        "tags"
    )

    if not isinstance(
        tags,
        list,
    ):
        tags = []

    owner = (
        raw.get("owner")
        or raw.get(
            "owner_id"
        )
    )

    image_size = (
        raw.get("size")
    )

    virtual_size = (
        raw.get(
            "virtual_size"
        )
    )

    hidden = raw.get(
        "is_hidden"
    )

    if hidden is None:
        hidden = raw.get(
            "hidden"
        )

    protected = raw.get(
        "is_protected"
    )

    if protected is None:
        protected = raw.get(
            "protected"
        )

    # We intentionally do not include:
    #
    #   direct_url
    #   file
    #   locations
    #
    # Those fields can reveal backend/internal
    # image-store addresses and aren't required
    # for normal OSHA resource understanding.

    payload = {
        "image_id": (
            optional_string(
                image_id
            )
        ),
        "name": (
            optional_string(name)
        ),
        "status": status,
        "healthy": healthy,
        "visibility": (
            optional_string(
                raw.get(
                    "visibility"
                )
            )
        ),
        "owner_id": (
            optional_string(
                owner
            )
        ),
        "disk_format": (
            optional_string(
                raw.get(
                    "disk_format"
                )
            )
        ),
        "container_format": (
            optional_string(
                raw.get(
                    "container_format"
                )
            )
        ),
        "size_bytes": (
            image_size
        ),
        "virtual_size_bytes": (
            virtual_size
        ),
        "minimum_disk_gb": (
            raw.get(
                "min_disk"
            )
        ),
        "minimum_ram_mb": (
            raw.get(
                "min_ram"
            )
        ),
        "checksum": (
            optional_string(
                raw.get(
                    "checksum"
                )
            )
        ),
        "hash_algorithm": (
            optional_string(
                raw.get(
                    "hash_algo"
                )
            )
        ),
        "hash_value": (
            optionalstring(
                raw.get(
                    "hash_value"
                )
            )
        ),
        "hidden": hidden,
        "protected": (
            protected
        ),
        "tags": sanitize(
            tags
        ),
        "properties": (
            properties
        ),
        "created_at": (
            raw.get(
                "created_at"
            )
        ),
        "updated_at": (
            raw.get(
                "updated_at"
            )
        ),
    }

    return sanitize(payload)


def collect_images(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect image metadata from Glance.

    Only metadata is requested. Image data is
    neither downloaded nor stored by OSHA.
    """

    records: List[dict] = []

    try:
        for image in (
            ctx.conn.image.images()
        ):
            payload = normalize_image(
                image
            )

            resource_id = (
                image_identifier(
                    payload
                )
            )

            timestamp = (
                payload.get(
                    "updated_at"
                )
                or payload.get(
                    "created_at"
                )
            )

            records.append(
                record(
                    "image",
                    "image",
                    payload,
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
                    timestamp=timestamp,
                    source="glance",
                )
            )

    except Exception as exc:
        records.append(
            error(
                "image",
                exc,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                operation=(
                    "list_images"
                ),
            )
        )

    return records


def image_summary(
    records: List[dict],
) -> dict:
    """
    Summarize already-collected image records.
    """

    statuses = Counter()
    visibilities = Counter()
    disk_formats = Counter()

    total_size_bytes = 0
    total_virtual_size_bytes = 0

    protected_count = 0
    hidden_count = 0
    unhealthy_count = 0

    valid_images = 0

    for item in records:
        if (
            item.get("signal_type")
            != "image"
        ):
            continue

        valid_images += 1

        payload = (
            item.get("payload")
            or {}
        )

        status = (
            payload.get("status")
            or "unknown"
        )

        visibility = (
            payload.get(
                "visibility"
            )
            or "unknown"
        )

        disk_format = (
            payload.get(
                "disk_format"
            )
            or "unknown"
        )

        statuses[
            str(status)
        ] += 1

        visibilities[
            str(visibility)
        ] += 1

        disk_formats[
            str(disk_format)
        ] += 1

        try:
            total_size_bytes += int(
                payload.get(
                    "size_bytes"
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

        try:
            total_virtual_size_bytes += (
                int(
                    payload.get(
                        "virtual_size_bytes"
                    )
                    or 0
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

        if (
            payload.get(
                "protected"
            )
            is True
        ):
            protected_count += 1

        if (
            payload.get(
                "hidden"
            )
            is True
        ):
            hidden_count += 1

        if (
            payload.get(
                "healthy"
            )
            is False
        ):
            unhealthy_count += 1

    return {
        "image_count": (
            valid_images
        ),
        "statuses": (
            dict(statuses)
        ),
        "visibility": (
            dict(
                visibilities
            )
        ),
        "disk_formats": (
            dict(
                disk_formats
            )
        ),
        "total_size_bytes": (
            total_size_bytes
        ),
        "total_virtual_size_bytes": (
            total_virtual_size_bytes
        ),
        "protected_image_count": (
            protected_count
        ),
        "hidden_image_count": (
            hidden_count
        ),
        "unhealthy_image_count": (
            unhealthy_count
        ),
    }


def collect(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect Glance image inventory.

    Read-only:
      list image metadata
      produce OSHA summary

    It does not:
      download images
      upload images
      delete images
      change visibility
      change protection state
    """

    if not service_available(
        ctx,
        "image",
    ):
        return [
            unavailable(
                "image",
                (
                    "Image service endpoint "
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
        ]

    records = collect_images(
        ctx
    )

    records.append(
        record(
            "image",
            "image_summary",
            image_summary(
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
