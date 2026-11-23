from __future__ import annotations

from collections import Counter
from typing import (
    Any,
    Callable,
    Iterable,
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


def resource_identifier(
    payload: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    for key in (
        "project_id",
        "domain_id",
        "region_id",
        "role_id",
        "user_id",
        "group_id",
        "id",
        "name",
    ):
        value = optional_string(
            payload.get(key)
        )

        if value:
            return value

    return fallback


def normalize_project(
    project: Any,
) -> dict:
    """
    Normalize a Keystone project.

    Projects group/isolate OpenStack resources
    and identity objects.
    """

    raw = as_dict(project)

    enabled = raw.get(
        "is_enabled"
    )

    if enabled is None:
        enabled = raw.get(
            "enabled"
        )

    return sanitize(
        {
            "project_id": (
                raw.get("id")
            ),
            "name": (
                raw.get("name")
            ),
            "description": (
                raw.get(
                    "description"
                )
            ),
            "domain_id": (
                raw.get(
                    "domain_id"
                )
            ),
            "parent_id": (
                raw.get(
                    "parent_id"
                )
            ),
            "enabled": enabled,
            "is_domain": (
                raw.get(
                    "is_domain"
                )
            ),
            "tags": (
                raw.get(
                    "tags"
                )
                or []
            ),
        }
    )


def normalize_domain(
    domain: Any,
) -> dict:
    raw = as_dict(domain)

    enabled = raw.get(
        "is_enabled"
    )

    if enabled is None:
        enabled = raw.get(
            "enabled"
        )

    return sanitize(
        {
            "domain_id": (
                raw.get("id")
            ),
            "name": (
                raw.get("name")
            ),
            "description": (
                raw.get(
                    "description"
                )
            ),
            "enabled": enabled,
        }
    )


def normalize_region(
    region: Any,
) -> dict:
    raw = as_dict(region)

    region_id = (
        raw.get("id")
        or raw.get(
            "region_id"
        )
    )

    return sanitize(
        {
            "region_id": (
                region_id
            ),
            "description": (
                raw.get(
                    "description"
                )
            ),
            "parent_region_id": (
                raw.get(
                    "parent_region_id"
                )
            ),
        }
    )


def normalize_role(
    role: Any,
) -> dict:
    raw = as_dict(role)

    return sanitize(
        {
            "role_id": (
                raw.get("id")
            ),
            "name": (
                raw.get("name")
            ),
            "domain_id": (
                raw.get(
                    "domain_id"
                )
            ),
            "description": (
                raw.get(
                    "description"
                )
            ),
        }
    )


def normalize_user(
    user: Any,
) -> dict:
    """
    Normalize explicitly requested user metadata.

    Passwords, tokens, credentials and other
    authentication material are never included.
    """

    raw = as_dict(user)

    enabled = raw.get(
        "is_enabled"
    )

    if enabled is None:
        enabled = raw.get(
            "enabled"
        )

    # Do not pass the complete raw object through
    # to the output. Explicit field selection is
    # used for identity records.
    return sanitize(
        {
            "user_id": (
                raw.get("id")
            ),
            "name": (
                raw.get("name")
            ),
            "domain_id": (
                raw.get(
                    "domain_id"
                )
            ),
            "default_project_id": (
                raw.get(
                    "default_project_id"
                )
            ),
            "enabled": enabled,
            "description": (
                raw.get(
                    "description"
                )
            ),
        }
    )


def normalize_group(
    group: Any,
) -> dict:
    raw = as_dict(group)

    return sanitize(
        {
            "group_id": (
                raw.get("id")
            ),
            "name": (
                raw.get("name")
            ),
            "description": (
                raw.get(
                    "description"
                )
            ),
            "domain_id": (
                raw.get(
                    "domain_id"
                )
            ),
        }
    )


def collect_resource_list(
    ctx: OpenStackContext,
    *,
    signal_type: str,
    operation: str,
    loader: Callable[
        [],
        Iterable[Any],
    ],
    normalizer: Callable[
        [Any],
        dict,
    ],
) -> List[dict]:
    """
    Execute an isolated read-only Keystone list
    operation.

    Permissions for one Identity resource do not
    prevent other Identity resources from being
    collected.
    """

    records: List[dict] = []

    try:
        for resource in loader():
            payload = normalizer(
                resource
            )

            identifier = (
                resource_identifier(
                    payload,
                    fallback=(
                        f"unknown-"
                        f"{signal_type}"
                    ),
                )
            )

            records.append(
                record(
                    "identity",
                    signal_type,
                    payload,
                    cloud_id=(
                        ctx.cloud_id
                    ),
                    region=ctx.region,
                    environment_id=(
                        ctx.environment_id
                    ),
                    resource_id=(
                        identifier
                    ),
                    source="keystone",
                )
            )

    except Exception as exc:
        records.append(
            error(
                "identity",
                exc,
                cloud_id=(
                    ctx.cloud_id
                ),
                region=ctx.region,
                environment_id=(
                    ctx.environment_id
                ),
                operation=operation,
            )
        )

    return records


def authenticated_context(
    ctx: OpenStackContext,
) -> dict:
    """
    Create a non-secret record describing the
    identity under which OSHA authenticated.

    This does not persist the authentication
    token or credentials.
    """

    payload = {
        "project_id": None,
        "user_id": None,
        "project_name": None,
        "user_name": None,
        "domain_id": None,
    }

    try:
        auth = (
            ctx.conn.session.auth
        )

        access = auth.get_access(
            ctx.conn.session
        )

        payload[
            "project_id"
        ] = optional_string(
            getattr(
                access,
                "project_id",
                None,
            )
        )

        payload[
            "user_id"
        ] = optional_string(
            getattr(
                access,
                "user_id",
                None,
            )
        )

        payload[
            "domain_id"
        ] = optional_string(
            getattr(
                access,
                "domain_id",
                None,
            )
        )

        # Only explicitly selected non-secret
        # names are extracted from token data.
        access_data = getattr(
            access,
            "_data",
            None,
        )

        if isinstance(
            access_data,
            Mapping,
        ):
            token = (
                access_data.get(
                    "token"
                )
                or {}
            )

            if isinstance(
                token,
                Mapping,
            ):
                project = (
                    token.get(
                        "project"
                    )
                    or {}
                )

                user = (
                    token.get("user")
                    or {}
                )

                if isinstance(
                    project,
                    Mapping,
                ):
                    payload[
                        "project_name"
                    ] = (
                        optional_string(
                            project.get(
                                "name"
                            )
                        )
                    )

                if isinstance(
                    user,
                    Mapping,
                ):
                    payload[
                        "user_name"
                    ] = (
                        optional_string(
                            user.get(
                                "name"
                            )
                        )
                    )

    except Exception:
        # Connection authentication has already
        # occurred in discovery.py. This optional
        # metadata extraction should therefore
        # not fail Identity collection.
        pass

    return sanitize(payload)


def identity_summary(
    records: List[dict],
) -> dict:
    counts = Counter()

    enabled_projects = 0
    disabled_projects = 0

    enabled_domains = 0
    disabled_domains = 0

    for item in records:
        signal_type = str(
            item.get(
                "signal_type",
                "unknown",
            )
        )

        counts[
            signal_type
        ] += 1

        payload = (
            item.get("payload")
            or {}
        )

        if signal_type == "project":
            enabled = (
                payload.get(
                    "enabled"
                )
            )

            if enabled is True:
                enabled_projects += 1
            elif enabled is False:
                disabled_projects += 1

        elif signal_type == "domain":
            enabled = (
                payload.get(
                    "enabled"
                )
            )

            if enabled is True:
                enabled_domains += 1
            elif enabled is False:
                disabled_domains += 1

    return {
        "record_counts": (
            dict(counts)
        ),
        "projects": {
            "enabled": (
                enabled_projects
            ),
            "disabled": (
                disabled_projects
            ),
        },
        "domains": {
            "enabled": (
                enabled_domains
            ),
            "disabled": (
                disabled_domains
            ),
        },
    }


def collect(
    ctx: OpenStackContext,
    include_users: bool = False,
    include_groups: bool = False,
) -> List[dict]:
    """
    Collect read-only Keystone identity inventory.

    Default:
      authenticated context
      projects
      domains
      regions
      roles

    Optional:
      users
      groups

    Never collected:
      credentials
      passwords
      authentication tokens
      application credentials
    """

    if not service_available(
        ctx,
        "identity",
    ):
        return [
            unavailable(
                "identity",
                (
                    "Identity service endpoint "
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

    records: List[dict] = []

    records.append(
        record(
            "identity",
            "authenticated_identity",
            authenticated_context(
                ctx
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
            source="keystone",
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type="project",
            operation=(
                "list_projects"
            ),
            loader=lambda: (
                ctx.conn.identity.
                projects()
            ),
            normalizer=(
                normalize_project
            ),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type="domain",
            operation=(
                "list_domains"
            ),
            loader=lambda: (
                ctx.conn.identity.
                domains()
            ),
            normalizer=(
                normalize_domain
            ),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type="region",
            operation=(
                "list_regions"
            ),
            loader=lambda: (
                ctx.conn.identity.
                regions()
            ),
            normalizer=(
                normalize_region
            ),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type="role",
            operation=(
                "list_roles"
            ),
            loader=lambda: (
                ctx.conn.identity.
                roles()
            ),
            normalizer=(
                normalize_role
            ),
        )
    )

    if include_users:
        records.extend(
            collect_resource_list(
                ctx,
                signal_type="user",
                operation=(
                    "list_users"
                ),
                loader=lambda: (
                    ctx.conn.identity.
                    users()
                ),
                normalizer=(
                    normalize_user
                ),
            )
        )

    if include_groups:
        records.extend(
            collect_resource_list(
                ctx,
                signal_type="group",
                operation=(
                    "list_groups"
                ),
                loader=lambda: (
                    ctx.conn.identity.
                    groups()
                ),
                normalizer=(
                    normalize_group
                ),
            )
        )

    records.append(
        record(
            "identity",
            "identity_summary",
            identity_summary(
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
