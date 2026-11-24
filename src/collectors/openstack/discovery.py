from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openstack

from src.collectors.openstack.schema import (
    available,
    error,
    record,
    sanitize,
    unavailable,
)


@dataclass(frozen=True)
class OpenStackContext:
    """
    Shared read-only context passed to all OSHA
    OpenStack collectors.

    The Connection object owns authentication
    and service access. Credentials themselves
    are never added to OSHA telemetry records.
    """

    conn: Any
    cloud_id: str
    region: Optional[str]
    environment_id: str
    config_source: str


def normalize_optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def detect_configuration_source(
    cloud: Optional[str],
) -> str:
    """
    Describe how OSHA expects openstacksdk to
    obtain its configuration.

    This reports configuration provenance only.
    It never reads credential contents.
    """

    if cloud:
        return "named_cloud"

    if os.getenv("OS_CLOUD"):
        return "os_cloud"

    if os.getenv("OS_AUTH_URL"):
        return "environment"

    if os.getenv("OS_CLIENT_CONFIG_FILE"):
        return "explicit_clouds_yaml"

    candidate_paths = [
        Path.cwd() / "clouds.yaml",
        Path.home() / ".config" / "openstack" / "clouds.yaml",
        Path("/etc/openstack/clouds.yaml"),
    ]

    for path in candidate_paths:
        if path.is_file():
            return "clouds_yaml"

    return "auto"


def configured_cloud_name(
    cloud: Optional[str],
    conn: Any,
) -> str:
    """
    Determine a stable non-secret identity for
    the connected OpenStack cloud.
    """

    explicit = normalize_optional_string(cloud)

    if explicit:
        return explicit

    env_cloud = normalize_optional_string(os.getenv("OS_CLOUD"))

    if env_cloud:
        return env_cloud

    config = getattr(
        conn,
        "config",
        None,
    )

    for attribute in (
        "name",
        "cloud",
    ):
        value = normalize_optional_string(
            getattr(
                config,
                attribute,
                None,
            )
        )

        if value:
            return value

    # OpenStack SDK uses an environment-backed
    # configuration when ordinary OS_* variables
    # are supplied.
    if os.getenv("OS_AUTH_URL"):
        return "envvars"

    return "openstack-default"


def configured_region(
    conn: Any,
    requested_region: Optional[str],
) -> Optional[str]:
    requested = normalize_optional_string(requested_region)

    if requested:
        return requested

    config = getattr(
        conn,
        "config",
        None,
    )

    config_region = normalize_optional_string(
        getattr(
            config,
            "region_name",
            None,
        )
    )

    if config_region:
        return config_region

    return normalize_optional_string(os.getenv("OS_REGION_NAME"))


def connect(
    cloud: Optional[str] = None,
    region: Optional[str] = None,
    *,
    verify: bool | str = True,
    environment_id: str = "default",
    timeout: int = 30,
    connect_retries: int = 3,
) -> OpenStackContext:
    """
    Establish and validate an OpenStack SDK
    connection.

    If cloud is provided, it is treated as a
    named cloud configuration.

    If cloud is omitted, openstacksdk performs
    its normal configuration discovery, which
    can include OS_* environment variables and
    standard clouds.yaml locations.

    This function performs authentication but
    does not mutate OpenStack resources.
    """

    kwargs: Dict[str, Any] = {
        "verify": verify,
        "timeout": timeout,
        "connect_retries": (connect_retries),
        "connect_retry_delay": 2,
    }

    if cloud:
        kwargs["cloud"] = cloud

    if region:
        kwargs["region_name"] = region

    conn = openstack.connect(**kwargs)

    # Force authentication immediately.
    # This prevents an invalid configuration
    # from surviving until the first collector
    # attempts an API request.
    conn.authorize()

    cloud_id = configured_cloud_name(
        cloud,
        conn,
    )

    region_name = configured_region(
        conn,
        region,
    )

    config_source = detect_configuration_source(cloud)

    return OpenStackContext(
        conn=conn,
        cloud_id=cloud_id,
        region=region_name,
        environment_id=(environment_id or "default"),
        config_source=(config_source),
    )


def service_available(
    ctx: OpenStackContext,
    service_type: str,
) -> bool:
    """
    Check whether the cloud service catalog
    contains a usable endpoint for a service.

    Failure is represented as False rather than
    allowing discovery to terminate.
    """

    try:
        return bool(ctx.conn.has_service(service_type))
    except Exception:
        return False


def discover_services(
    ctx: OpenStackContext,
) -> Dict[str, bool]:
    """
    Probe the major service types OSHA currently
    understands.

    This is capability discovery, not a health
    check of every backend daemon.
    """

    service_types = (
        "identity",
        "compute",
        "network",
        "image",
        "block-storage",
        "object-store",
        "orchestration",
        "placement",
        "baremetal",
        "load-balancer",
        "shared-file-system",
    )

    result = {}

    for service_type in service_types:
        result[service_type] = service_available(
            ctx,
            service_type,
        )

    return result


def get_authenticated_project(
    ctx: OpenStackContext,
) -> Dict[str, Optional[str]]:
    """
    Return non-secret identity information about
    the authenticated project/user context.

    Passwords, tokens and application
    credentials are deliberately excluded.
    """

    result: Dict[
        str,
        Optional[str],
    ] = {
        "project_id": None,
        "project_name": None,
        "user_id": None,
        "user_name": None,
    }

    try:
        auth = ctx.conn.session.auth

        access = auth.get_access(ctx.conn.session)

        project_id = getattr(
            access,
            "project_id",
            None,
        )

        user_id = getattr(
            access,
            "user_id",
            None,
        )

        result["project_id"] = normalize_optional_string(project_id)

        result["user_id"] = normalize_optional_string(user_id)

        token = getattr(
            access,
            "_data",
            None,
        )

        # _data is never persisted wholesale.
        # Only explicitly non-secret names are
        # extracted when present.
        if isinstance(
            token,
            dict,
        ):
            token_data = token.get("token") or {}

            if isinstance(
                token_data,
                dict,
            ):
                project = token_data.get("project") or {}

                user = token_data.get("user") or {}

                if isinstance(
                    project,
                    dict,
                ):
                    result["project_name"] = normalize_optional_string(
                        project.get("name")
                    )

                if isinstance(
                    user,
                    dict,
                ):
                    result["user_name"] = normalize_optional_string(user.get("name"))

    except Exception:
        # Authentication is already validated
        # in connect(). Failure here means only
        # that optional identity metadata could
        # not be extracted.
        pass

    return result


def discovery_record(
    ctx: OpenStackContext,
) -> dict:
    """
    Produce the primary record describing the
    connected OpenStack environment.
    """

    service_map = discover_services(ctx)

    identity = get_authenticated_project(ctx)

    available_services = sorted(
        service for service, enabled in service_map.items() if enabled
    )

    unavailable_services = sorted(
        service for service, enabled in service_map.items() if not enabled
    )

    payload = {
        "cloud_id": (ctx.cloud_id),
        "region": ctx.region,
        "environment_id": (ctx.environment_id),
        "configuration_source": (ctx.config_source),
        "authenticated_identity": (identity),
        "service_availability": (service_map),
        "available_services": (available_services),
        "unavailable_services": (unavailable_services),
    }

    return record(
        "discovery",
        "cloud_state",
        payload,
        cloud_id=ctx.cloud_id,
        region=ctx.region,
        environment_id=(ctx.environment_id),
        resource_id=(ctx.cloud_id),
        source="openstacksdk",
    )


def capability_records(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Produce one collector-status record for each
    service type.

    This allows the dashboard to distinguish:

        collector operational
        +
        OpenStack service unavailable

    from:

        collector failure
    """

    records: List[dict] = []

    services = discover_services(ctx)

    for service_type, enabled in services.items():
        if enabled:
            records.append(
                available(
                    service_type,
                    cloud_id=(ctx.cloud_id),
                    region=ctx.region,
                    environment_id=(ctx.environment_id),
                    details={
                        "service_type": (service_type),
                    },
                )
            )
        else:
            records.append(
                unavailable(
                    service_type,
                    ("No usable service endpoint was discovered"),
                    cloud_id=(ctx.cloud_id),
                    region=ctx.region,
                    environment_id=(ctx.environment_id),
                )
            )

    return records


def collect(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect OpenStack environment discovery
    records.

    No OpenStack resources are modified.
    """

    records = [discovery_record(ctx)]

    records.extend(capability_records(ctx))

    return records


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=("OSHA OpenStack discovery test"))

    parser.add_argument(
        "--cloud",
        default=None,
    )

    parser.add_argument(
        "--region",
        default=None,
    )

    parser.add_argument(
        "--environment",
        default="default",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help=("Disable TLS verification. Development use only."),
    )

    args = parser.parse_args()

    try:
        context = connect(
            cloud=args.cloud,
            region=args.region,
            environment_id=(args.environment),
            verify=(not args.insecure),
        )

        for item in collect(context):
            print(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,
                )
            )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": (type(exc).__name__),
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )

        raise
