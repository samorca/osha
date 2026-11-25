from __future__ import annotations

"""
Principal construction, normalization, and
in-memory identity lookup for OSHA.
"""

from dataclasses import (
    replace,
)

from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.core.serialization import (
    DEFAULT_SENSITIVE_KEYS,
    to_json_safe,
)

from src.core.validation import (
    required_string,
    unique_strings,
)

from src.security.schema import (
    PrincipalType,
    SecurityPrincipal,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class IdentityError(RuntimeError):
    pass


class IdentityNotFoundError(IdentityError):
    pass


class IdentityConflictError(IdentityError):
    pass


# ---------------------------------------------------------------
# Principal ID normalization
# ---------------------------------------------------------------


def normalize_principal_id(
    value: Any,
) -> str:
    """
    Normalize an opaque principal identifier.

    This deliberately does not assume the ID is
    an email address, UUID, username, Entra ID,
    LDAP DN, or another provider-specific form.
    """

    return required_string(
        value,
        name="principal_id",
        maximum_length=512,
    )


# ---------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------


def normalize_role(
    value: Any,
) -> str:
    """
    OSHA role names use normalized lowercase
    strings.
    """

    return required_string(
        value,
        name="role",
        maximum_length=128,
    ).lower()


def normalize_roles(
    values: Iterable[Any],
) -> List[str]:
    return unique_strings(
        (normalize_role(value) for value in values),
        name="roles",
        allow_empty=False,
    )


# ---------------------------------------------------------------
# Group normalization
# ---------------------------------------------------------------


def normalize_group(
    value: Any,
) -> str:
    return required_string(
        value,
        name="group",
        maximum_length=256,
    )


def normalize_groups(
    values: Iterable[Any],
) -> List[str]:
    return unique_strings(
        (normalize_group(value) for value in values),
        name="groups",
        allow_empty=False,
    )


# ---------------------------------------------------------------
# Generic principal constructor
# ---------------------------------------------------------------


def principal(
    *,
    principal_id: str,
    principal_type: str,
    display_name: Optional[str] = None,
    roles: Iterable[str] = (),
    groups: Iterable[str] = (),
    attributes: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> SecurityPrincipal:
    """
    Construct one normalized SecurityPrincipal.

    Attributes are sanitized using core
    serialization before being attached.
    """

    if principal_type not in {item.value for item in PrincipalType}:
        raise IdentityError(f"Unknown principal type: {principal_type}")

    return SecurityPrincipal(
        principal_id=(normalize_principal_id(principal_id)),
        principal_type=(principal_type),
        display_name=(
            required_string(
                display_name,
                name="display_name",
                maximum_length=256,
            )
            if display_name is not None
            else None
        ),
        roles=normalize_roles(roles),
        groups=normalize_groups(groups),
        attributes=(
            to_json_safe(
                dict(attributes or {}),
                redact_sensitive=True,
                sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
            )
        ),
    ).finalize()


# ---------------------------------------------------------------
# Typed constructors
# ---------------------------------------------------------------


def user_principal(
    principal_id: str,
    **kwargs: Any,
) -> SecurityPrincipal:
    return principal(
        principal_id=principal_id,
        principal_type=(PrincipalType.USER.value),
        **kwargs,
    )


def service_principal(
    principal_id: str,
    **kwargs: Any,
) -> SecurityPrincipal:
    return principal(
        principal_id=principal_id,
        principal_type=(PrincipalType.SERVICE.value),
        **kwargs,
    )


def worker_principal(
    principal_id: str,
    **kwargs: Any,
) -> SecurityPrincipal:
    return principal(
        principal_id=principal_id,
        principal_type=(PrincipalType.WORKER.value),
        **kwargs,
    )


def system_principal(
    principal_id: str = ("osha-system"),
    **kwargs: Any,
) -> SecurityPrincipal:
    return principal(
        principal_id=principal_id,
        principal_type=(PrincipalType.SYSTEM.value),
        **kwargs,
    )


# ---------------------------------------------------------------
# Role checks
# ---------------------------------------------------------------


def has_role(
    value: SecurityPrincipal,
    role: str,
) -> bool:
    value.finalize()

    return normalize_role(role) in set(value.roles)


def has_any_role(
    value: SecurityPrincipal,
    roles: Iterable[str],
) -> bool:
    value.finalize()

    requested = set(normalize_roles(roles))

    return bool(requested.intersection(value.roles))


def has_all_roles(
    value: SecurityPrincipal,
    roles: Iterable[str],
) -> bool:
    value.finalize()

    requested = set(normalize_roles(roles))

    return requested.issubset(set(value.roles))


# ---------------------------------------------------------------
# Group checks
# ---------------------------------------------------------------


def in_group(
    value: SecurityPrincipal,
    group: str,
) -> bool:
    value.finalize()

    return normalize_group(group) in set(value.groups)


# ---------------------------------------------------------------
# Identity registry
# ---------------------------------------------------------------


class IdentityRegistry:
    """
    Thread-safe in-memory registry of normalized
    OSHA principals.

    This registry does not authenticate them.
    """

    def __init__(
        self,
    ) -> None:
        self._principals: Dict[
            str,
            SecurityPrincipal,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        value: SecurityPrincipal,
    ) -> SecurityPrincipal:
        value.finalize()

        with self._lock:
            if value.principal_id in self._principals:
                raise IdentityConflictError(
                    f"Principal already registered: {value.principal_id}"
                )

            self._principals[value.principal_id] = value

        return value

    # ---------------------------------------------------------
    # Upsert
    # ---------------------------------------------------------

    def upsert(
        self,
        value: SecurityPrincipal,
    ) -> SecurityPrincipal:
        value.finalize()

        with self._lock:
            self._principals[value.principal_id] = value

        return value

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(
        self,
        principal_id: str,
    ) -> Optional[SecurityPrincipal]:
        normalized = normalize_principal_id(principal_id)

        with self._lock:
            return self._principals.get(normalized)

    def require(
        self,
        principal_id: str,
    ) -> SecurityPrincipal:
        value = self.get(principal_id)

        if value is None:
            raise IdentityNotFoundError(f"Unknown principal: {principal_id}")

        return value

    # ---------------------------------------------------------
    # Remove
    # ---------------------------------------------------------

    def remove(
        self,
        principal_id: str,
    ) -> SecurityPrincipal:
        normalized = normalize_principal_id(principal_id)

        with self._lock:
            value = self._principals.pop(
                normalized,
                None,
            )

        if value is None:
            raise IdentityNotFoundError(f"Unknown principal: {principal_id}")

        return value

    # ---------------------------------------------------------
    # List/filter
    # ---------------------------------------------------------

    def list(
        self,
        *,
        principal_type: Optional[str] = None,
        role: Optional[str] = None,
        group: Optional[str] = None,
    ) -> List[SecurityPrincipal]:
        if principal_type is not None and principal_type not in {
            item.value for item in PrincipalType
        }:
            raise IdentityError(f"Unknown principal type: {principal_type}")

        normalized_role = normalize_role(role) if role is not None else None

        normalized_group = normalize_group(group) if group is not None else None

        with self._lock:
            values = list(self._principals.values())

        output: List[SecurityPrincipal] = []

        for value in values:
            if principal_type is not None and value.principal_type != principal_type:
                continue

            if normalized_role is not None and normalized_role not in value.roles:
                continue

            if normalized_group is not None and normalized_group not in value.groups:
                continue

            output.append(value)

        return sorted(
            output,
            key=lambda item: item.principal_id,
        )

    # ---------------------------------------------------------
    # Role update
    # ---------------------------------------------------------

    def update_roles(
        self,
        principal_id: str,
        *,
        roles: Iterable[str],
    ) -> SecurityPrincipal:
        current = self.require(principal_id)

        updated = replace(
            current,
            roles=normalize_roles(roles),
        ).finalize()

        return self.upsert(updated)

    # ---------------------------------------------------------
    # Group update
    # ---------------------------------------------------------

    def update_groups(
        self,
        principal_id: str,
        *,
        groups: Iterable[str],
    ) -> SecurityPrincipal:
        current = self.require(principal_id)

        updated = replace(
            current,
            groups=normalize_groups(groups),
        ).finalize()

        return self.upsert(updated)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            by_type: Dict[
                str,
                int,
            ] = {}

            for value in self._principals.values():
                by_type[value.principal_type] = (
                    by_type.get(
                        value.principal_type,
                        0,
                    )
                    + 1
                )

            return {
                "principal_count": (len(self._principals)),
                "principal_types": (by_type),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "IdentityError",
    "IdentityNotFoundError",
    "IdentityConflictError",
    "normalize_principal_id",
    "normalize_role",
    "normalize_group",
    "normalize_roles",
    "normalize_groups",
    "principal",
    "user_principal",
    "service_principal",
    "worker_principal",
    "system_principal",
    "has_role",
    "has_any_role",
    "has_all_roles",
    "in_group",
    "IdentityRegistry",
]
