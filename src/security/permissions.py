from __future__ import annotations

"""
Permission catalog and permission matching for
OSHA security.
"""

from dataclasses import (
    dataclass,
)

from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.core.validation import (
    environment_id,
    required_string,
)

from src.security.schema import (
    Permission,
)


# ---------------------------------------------------------------
# Wildcard
# ---------------------------------------------------------------

WILDCARD = "*"


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------

class PermissionErrorBase(
    RuntimeError
):
    pass


class PermissionConflictError(
    PermissionErrorBase
):
    pass


class PermissionNotFoundError(
    PermissionErrorBase
):
    pass


# ---------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------

def normalize_resource(
    value: Any,
) -> str:
    """
    Normalize permission resource names.

    Examples:

      remediation
      orchestrator
      worker
      telemetry
    """

    return required_string(
        value,
        name=(
            "permission resource"
        ),
        maximum_length=256,
    ).lower()


def normalize_action(
    value: Any,
) -> str:
    """
    Normalize permission action names.

    Examples:

      read
      create
      execute
      approve
    """

    return required_string(
        value,
        name=(
            "permission action"
        ),
        maximum_length=128,
    ).lower()


def normalize_scope(
    value: Optional[
        Any
    ],
) -> Optional[str]:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    return (
        text
        or None
    )


def normalize_environment(
    value: Optional[
        Any
    ],
) -> Optional[str]:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return environment_id(
        text
    )


def normalize_permission(
    value: Permission,
) -> Permission:
    value.finalize()

    return Permission(
        resource=(
            normalize_resource(
                value.resource
            )
        ),
        action=(
            normalize_action(
                value.action
            )
        ),
        scope=(
            normalize_scope(
                value.scope
            )
        ),
        environment_id=(
            normalize_environment(
                value.environment_id
            )
        ),
        attributes=dict(
            value.attributes
        ),
    ).finalize()


# ---------------------------------------------------------------
# Permission constructor
# ---------------------------------------------------------------

def permission(
    resource: str,
    action: str,
    *,
    scope: Optional[
        str
    ] = None,
    environment: Optional[
        str
    ] = None,
    attributes: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None,
) -> Permission:
    """
    Convenience constructor for normalized
    Permission objects.
    """

    return normalize_permission(
        Permission(
            resource=resource,
            action=action,
            scope=scope,
            environment_id=(
                environment
            ),
            attributes=dict(
                attributes
                or {}
            ),
        )
    )


# ---------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------

def _field_matches(
    granted: Optional[
        str
    ],
    requested: Optional[
        str
    ],
) -> bool:
    """
    Exact match with explicit wildcard support.

    Important:

      None is not treated as wildcard.

    A grant must explicitly contain "*" if it
    should match any value.
    """

    if granted == WILDCARD:
        return True

    return (
        granted
        == requested
    )


# ---------------------------------------------------------------
# Permission matching
# ---------------------------------------------------------------

def permission_matches(
    granted: Permission,
    requested: Permission,
) -> bool:
    """
    Determine whether one granted permission
    satisfies a requested permission.

    Matching rules:

      resource:
          exact or "*"

      action:
          exact or "*"

      environment:
          exact or "*"

      scope:
          exact or "*"

    There is deliberately no implicit wildcard
    for missing values.
    """

    granted = (
        normalize_permission(
            granted
        )
    )

    requested = (
        normalize_permission(
            requested
        )
    )

    if (
        granted.resource
        not in {
            WILDCARD,
            requested.resource,
        }
    ):
        return False

    if (
        granted.action
        not in {
            WILDCARD,
            requested.action,
        }
    ):
        return False

    if not _field_matches(
        granted.environment_id,
        requested.environment_id,
    ):
        return False

    if not _field_matches(
        granted.scope,
        requested.scope,
    ):
        return False

    return True


# ---------------------------------------------------------------
# Permission grant
# ---------------------------------------------------------------

@dataclass(frozen=True)
class PermissionGrant:
    """
    Associates a normalized permission with a
    principal.

    This describes a grant. The authorization
    engine may still apply policy constraints.
    """

    principal_id: str

    permission: Permission

    def key(
        self,
    ) -> str:
        return (
            f"{self.principal_id}|"
            f"{self.permission.key()}"
        )


# ---------------------------------------------------------------
# Permission catalog
# ---------------------------------------------------------------

class PermissionCatalog:
    """
    Thread-safe in-memory catalog containing:

      known permission definitions
      principal permission grants

    Having a grant does not bypass policies,
    approvals, remediation guardrails, or other
    downstream security checks.
    """

    def __init__(
        self,
    ) -> None:
        self._permissions: Dict[
            str,
            Permission,
        ] = {}

        self._grants: Dict[
            str,
            Dict[
                str,
                Permission,
            ],
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Permission definition registration
    # ---------------------------------------------------------

    def register(
        self,
        value: Permission,
    ) -> Permission:
        value = normalize_permission(
            value
        )

        key = value.key()

        with self._lock:
            if (
                key
                in self._permissions
            ):
                raise PermissionConflictError(
                    "Permission already "
                    "registered: "
                    f"{key}"
                )

            self._permissions[
                key
            ] = value

        return value

    def upsert(
        self,
        value: Permission,
    ) -> Permission:
        value = normalize_permission(
            value
        )

        with self._lock:
            self._permissions[
                value.key()
            ] = value

        return value

    # ---------------------------------------------------------
    # Definition lookup
    # ---------------------------------------------------------

    def get(
        self,
        key: str,
    ) -> Optional[
        Permission
    ]:
        with self._lock:
            return self._permissions.get(
                key
            )

    def require(
        self,
        key: str,
    ) -> Permission:
        value = self.get(
            key
        )

        if value is None:
            raise PermissionNotFoundError(
                "Unknown permission: "
                f"{key}"
            )

        return value

    def list(
        self,
    ) -> List[
        Permission
    ]:
        with self._lock:
            return [
                self._permissions[
                    key
                ]
                for key
                in sorted(
                    self._permissions
                )
            ]

    # ---------------------------------------------------------
    # Grant
    # ---------------------------------------------------------

    def grant(
        self,
        principal_id: str,
        value: Permission,
    ) -> PermissionGrant:
        """
        Grant one permission to a principal.

        The permission does not have to be added
        to the definition registry first because
        grants may come from an external identity
        or policy integration later.
        """

        principal_id = (
            required_string(
                principal_id,
                name="principal_id",
                maximum_length=512,
            )
        )

        value = normalize_permission(
            value
        )

        with self._lock:
            grants = (
                self._grants.
                setdefault(
                    principal_id,
                    {},
                )
            )

            grants[
                value.key()
            ] = value

        return PermissionGrant(
            principal_id=(
                principal_id
            ),
            permission=value,
        )

    # ---------------------------------------------------------
    # Revoke
    # ---------------------------------------------------------

    def revoke(
        self,
        principal_id: str,
        value: Permission,
    ) -> bool:
        principal_id = (
            required_string(
                principal_id,
                name="principal_id",
                maximum_length=512,
            )
        )

        value = normalize_permission(
            value
        )

        with self._lock:
            grants = (
                self._grants.get(
                    principal_id
                )
            )

            if not grants:
                return False

            removed = (
                grants.pop(
                    value.key(),
                    None,
                )
                is not None
            )

            if not grants:
                self._grants.pop(
                    principal_id,
                    None,
                )

            return removed

    # ---------------------------------------------------------
    # Principal grants
    # ---------------------------------------------------------

    def grants_for(
        self,
        principal_id: str,
    ) -> List[
        Permission
    ]:
        principal_id = (
            required_string(
                principal_id,
                name="principal_id",
                maximum_length=512,
            )
        )

        with self._lock:
            grants = (
                self._grants.get(
                    principal_id,
                    {},
                )
            )

            return [
                grants[
                    key
                ]
                for key
                in sorted(
                    grants
                )
            ]

    # ---------------------------------------------------------
    # Basic grant check
    # ---------------------------------------------------------

    def check(
        self,
        principal_id: str,
        requested: Permission,
    ) -> bool:
        """
        Check direct permission grants.

        This is NOT the final authorization
        decision.
        """

        requested = (
            normalize_permission(
                requested
            )
        )

        return any(
            permission_matches(
                granted,
                requested,
            )
            for granted
            in self.grants_for(
                principal_id
            )
        )

    # ---------------------------------------------------------
    # Matching grants
    # ---------------------------------------------------------

    def matching_grants(
        self,
        principal_id: str,
        requested: Permission,
    ) -> List[
        Permission
    ]:
        requested = (
            normalize_permission(
                requested
            )
        )

        return [
            granted
            for granted
            in self.grants_for(
                principal_id
            )
            if permission_matches(
                granted,
                requested,
            )
        ]

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_permissions": (
                    len(
                        self._permissions
                    )
                ),

                "principals_with_grants": (
                    len(
                        self._grants
                    )
                ),

                "grant_count": (
                    sum(
                        len(
                            items
                        )
                        for items
                        in self._grants.
                        values()
                    )
                ),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "WILDCARD",

    "PermissionErrorBase",
    "PermissionConflictError",
    "PermissionNotFoundError",

    "normalize_resource",
    "normalize_action",
    "normalize_scope",
    "normalize_environment",

    "normalize_permission",
    "permission",

    "permission_matches",

    "PermissionGrant",
    "PermissionCatalog",
]
