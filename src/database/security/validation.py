"""
OSHA Database Security Validation.

Provides reusable validation functions for data entering the database layer.

Validation goals:

    - reject malformed identifiers
    - validate tenant IDs
    - validate infrastructure identifiers
    - validate IP addresses and hostnames
    - validate ports
    - validate bounded strings
    - validate enum-like values
    - validate confidence / probability values
    - validate pagination
    - validate JSON-style metadata
    - limit nested payload complexity
    - reject dangerous control characters where inappropriate

IMPORTANT:

Validation is defence-in-depth.

It does not replace:

    - SQLAlchemy parameter binding
    - PostgreSQL permissions
    - PostgreSQL Row-Level Security
    - application authorisation
    - secret-management controls
    - audit logging
"""

from __future__ import annotations

import enum
import ipaddress
import math
import re
import uuid
from collections.abc import (
    Collection,
    Mapping,
    Sequence,
)
from datetime import datetime
from typing import Any, TypeVar


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_NAME_LENGTH = 512
MAX_IDENTIFIER_LENGTH = 1024
MAX_DESCRIPTION_LENGTH = 16_384

MAX_AUDIT_MESSAGE_LENGTH = 16_384

MAX_JSON_DEPTH = 12
MAX_JSON_ITEMS = 10_000
MAX_JSON_STRING_LENGTH = 65_536

MAX_PAGE_SIZE = 1000

MIN_PORT = 1
MAX_PORT = 65_535


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_HOST_LABEL_RE = re.compile(
    r"^?:[A-Za-z0-9-]{0,61}[A-Za-z0-9]?$"
)

_SAFE_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$"
)

_CONTROL_CHARACTER_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SecurityValidationError(ValueError):
    """Base error for OSHA database security validation failures."""


class RequiredValueError(SecurityValidationError):
    """Raised when a required value is absent."""


class InvalidUUIDError(SecurityValidationError):
    """Raised when a UUID value is invalid."""


class InvalidStringError(SecurityValidationError):
    """Raised when textual input is invalid."""


class InvalidIdentifierError(SecurityValidationError):
    """Raised when an infrastructure identifier is invalid."""


class InvalidIPAddressError(SecurityValidationError):
    """Raised when an IP address is invalid."""


class InvalidHostnameError(SecurityValidationError):
    """Raised when a hostname is invalid."""


class InvalidPortError(SecurityValidationError):
    """Raised when a network port is invalid."""


class InvalidChoiceError(SecurityValidationError):
    """Raised when a value is outside an allowed set."""


class InvalidRangeError(SecurityValidationError):
    """Raised when a numeric value is outside its accepted range."""


class InvalidPaginationError(SecurityValidationError):
    """Raised when pagination values are invalid."""


class InvalidMetadataError(SecurityValidationError):
    """Raised when structured metadata violates safety limits."""


EnumT = TypeVar(
    "EnumT",
    bound=enum.Enum,
)


# ---------------------------------------------------------------------------
# Generic text
# ---------------------------------------------------------------------------

def validate_required_string(
    value: str,
    *,
    field_name: str,
    max_length: int = MAX_NAME_LENGTH,
    strip: bool = True,
    allow_newlines: bool = False,
) -> str:
    """
    Validate required textual input.

    The returned value is normalised when strip=True.
    """
    if not isinstance(value, str):
        raise InvalidStringError(
            f"{field_name} must be a string"
        )

    result = value.strip() if strip else value

    if not result:
        raise RequiredValueError(
            f"{field_name} cannot be empty"
        )

    if len(result) > max_length:
        raise InvalidStringError(
            f"{field_name} exceeds maximum length "
            f"of {max_length}"
        )

    if _CONTROL_CHARACTER_RE.search(result):
        raise InvalidStringError(
            f"{field_name} contains unsupported control characters"
        )

    if not allow_newlines and (
        "\n" in result
        or "\r" in result
    ):
        raise InvalidStringError(
            f"{field_name} cannot contain newlines"
        )

    return result


def validate_optional_string(
    value: str | None,
    *,
    field_name: str,
    max_length: int = MAX_NAME_LENGTH,
    strip: bool = True,
    allow_newlines: bool = False,
) -> str | None:
    """Validate optional textual input."""
    if value is None:
        return None

    return validate_required_string(
        value,
        field_name=field_name,
        max_length=max_length,
        strip=strip,
        allow_newlines=allow_newlines,
    )


def validate_description(
    value: str | None,
    *,
    field_name: str = "description",
) -> str | None:
    """
    Validate longer human-readable text.

    Newlines are permitted, while unsupported control characters and
    excessive length are rejected.
    """
    return validate_optional_string(
        value,
        field_name=field_name,
        max_length=MAX_DESCRIPTION_LENGTH,
        allow_newlines=True,
    )


# ---------------------------------------------------------------------------
# UUID
# ---------------------------------------------------------------------------

def validate_uuid(
    value: uuid.UUID | str,
    *,
    field_name: str = "id",
) -> uuid.UUID:
    """Validate and normalise a UUID."""
    if isinstance(value, uuid.UUID):
        return value

    if not isinstance(value, str):
        raise InvalidUUIDError(
            f"{field_name} must be a UUID or UUID string"
        )

    candidate = value.strip()

    if not candidate:
        raise InvalidUUIDError(
            f"{field_name} cannot be empty"
        )

    try:
        return uuid.UUID(candidate)

    except (ValueError, AttributeError) as exc:
        raise InvalidUUIDError(
            f"{field_name} is not a valid UUID"
        ) from exc


def validate_optional_uuid(
    value: uuid.UUID | str | None,
    *,
    field_name: str = "id",
) -> uuid.UUID | None:
    """Validate an optional UUID."""
    if value is None:
        return None

    return validate_uuid(
        value,
        field_name=field_name,
    )


def validate_tenant_id(
    value: uuid.UUID | str,
) -> uuid.UUID:
    """Validate OSHA tenant identity."""
    return validate_uuid(
        value,
        field_name="tenant_id",
    )


# ---------------------------------------------------------------------------
# Infrastructure identifiers
# ---------------------------------------------------------------------------

def validate_identifier(
    value: str,
    *,
    field_name: str = "identifier",
    max_length: int = MAX_IDENTIFIER_LENGTH,
) -> str:
    """
    Validate a structured infrastructure identifier.

    This validator is suitable for identifiers such as canonical source
    names, resource keys and externally supplied structured IDs.

    It should not be used for free-form descriptions.
    """
    result = validate_required_string(
        value,
        field_name=field_name,
        max_length=max_length,
    )

    if not _SAFE_IDENTIFIER_RE.fullmatch(result):
        raise InvalidIdentifierError(
            f"{field_name} contains unsupported characters"
        )

    return result


# ---------------------------------------------------------------------------
# Hostname
# ---------------------------------------------------------------------------

def validate_hostname(
    value: str,
    *,
    field_name: str = "hostname",
) -> str:
    """Validate a DNS-style hostname."""
    hostname = validate_required_string(
        value,
        field_name=field_name,
        max_length=253,
    )

    hostname = hostname.rstrip(".")

    if not hostname:
        raise InvalidHostnameError(
            f"{field_name} cannot be empty"
        )

    labels = hostname.split(".")

    if any(
        not _HOST_LABEL_RE.fullmatch(label)
        for label in labels
    ):
        raise InvalidHostnameError(
            f"{field_name} is not a valid hostname"
        )

    return hostname.lower()


def validate_optional_hostname(
    value: str | None,
) -> str | None:
    """Validate an optional hostname."""
    if value is None:
        return None

    return validate_hostname(value)


# ---------------------------------------------------------------------------
# IP address
# ---------------------------------------------------------------------------

def validate_ip_address(
    value: str,
    *,
    field_name: str = "ip_address",
) -> str:
    """Validate and normalise an IPv4 or IPv6 address."""
    candidate = validate_required_string(
        value,
        field_name=field_name,
        max_length=64,
    )

    try:
        parsed = ipaddress.ip_address(
            candidate
        )

    except ValueError as exc:
        raise InvalidIPAddressError(
            f"{field_name} is not a valid IP address"
        ) from exc

    return str(parsed)


def validate_optional_ip_address(
    value: str | None,
) -> str | None:
    """Validate an optional IPv4/IPv6 address."""
    if value is None:
        return None

    return validate_ip_address(value)


# ---------------------------------------------------------------------------
# Network port
# ---------------------------------------------------------------------------

def validate_port(
    value: int,
    *,
    field_name: str = "port",
) -> int:
    """Validate TCP/UDP-style port range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidPortError(
            f"{field_name} must be an integer"
        )

    if not MIN_PORT <= value <= MAX_PORT:
        raise InvalidPortError(
            f"{field_name} must be between "
            f"{MIN_PORT} and {MAX_PORT}"
        )

    return value


def validate_optional_port(
    value: int | None,
) -> int | None:
    """Validate an optional port."""
    if value is None:
        return None

    return validate_port(value)


# ---------------------------------------------------------------------------
# Choices / enum values
# ---------------------------------------------------------------------------

def validate_choice(
    value: str,
    *,
    allowed: Collection[str],
    field_name: str,
) -> str:
    """Require exact membership in an allowed set."""
    candidate = validate_required_string(
        value,
        field_name=field_name,
        max_length=255,
    )

    if candidate not in allowed:
        raise InvalidChoiceError(
            f"{field_name} has an unsupported value"
        )

    return candidate


def validate_enum_value(
    value: str,
    *,
    enum_type: type[EnumT],
    field_name: str,
) -> str:
    """Validate a string against a string-valued Enum."""
    allowed = {
        str(member.value)
        for member in enum_type
    }

    return validate_choice(
        value,
        allowed=allowed,
        field_name=field_name,
    )


# ---------------------------------------------------------------------------
# Numeric values
# ---------------------------------------------------------------------------

def validate_finite_number(
    value: float | int,
    *,
    field_name: str,
) -> float:
    """Validate a finite numeric value."""
    if isinstance(value, bool):
        raise InvalidRangeError(
            f"{field_name} must be numeric"
        )

    try:
        numeric = float(value)

    except (TypeError, ValueError) as exc:
        raise InvalidRangeError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(numeric):
        raise InvalidRangeError(
            f"{field_name} must be finite"
        )

    return numeric


def validate_range(
    value: float | int,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a finite numeric value against range boundaries."""
    numeric = validate_finite_number(
        value,
        field_name=field_name,
    )

    if minimum is not None and numeric < minimum:
        raise InvalidRangeError(
            f"{field_name} must be >= {minimum}"
        )

    if maximum is not None and numeric > maximum:
        raise InvalidRangeError(
            f"{field_name} must be <= {maximum}"
        )

    return numeric


def validate_probability(
    value: float | int,
    *,
    field_name: str = "probability",
) -> float:
    """Validate a probability/confidence style value."""
    return validate_range(
        value,
        field_name=field_name,
        minimum=0.0,
        maximum=1.0,
    )


def validate_optional_probability(
    value: float | int | None,
    *,
    field_name: str = "probability",
) -> float | None:
    """Validate optional probability/confidence."""
    if value is None:
        return None

    return validate_probability(
        value,
        field_name=field_name,
    )


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def validate_pagination(
    *,
    offset: int = 0,
    limit: int = 100,
    max_limit: int = MAX_PAGE_SIZE,
) -> tuple[int, int]:
    """Validate bounded repository pagination."""
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
    ):
        raise InvalidPaginationError(
            "offset must be a non-negative integer"
        )

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
    ):
        raise InvalidPaginationError(
            "limit must be a positive integer"
        )

    if limit > max_limit:
        raise InvalidPaginationError(
            f"limit cannot exceed {max_limit}"
        )

    return offset, limit


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

def validate_time_window(
    *,
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime]:
    """Validate a chronological time window."""
    if not isinstance(start, datetime):
        raise SecurityValidationError(
            "start must be a datetime"
        )

    if not isinstance(end, datetime):
        raise SecurityValidationError(
            "end must be a datetime"
        )

    if end < start:
        raise SecurityValidationError(
            "end cannot be earlier than start"
        )

    return start, end


# ---------------------------------------------------------------------------
# Structured metadata
# ---------------------------------------------------------------------------

def validate_metadata(
    value: Mapping[str, Any] | None,
    *,
    field_name: str = "metadata",
    max_depth: int = MAX_JSON_DEPTH,
    max_items: int = MAX_JSON_ITEMS,
    max_string_length: int = MAX_JSON_STRING_LENGTH,
) -> dict[str, Any]:
    """
    Validate JSON-compatible metadata with structural limits.

    Supported values:

        None
        bool
        int
        float
        str
        list / tuple
        dict with string keys

    Arbitrary Python objects are rejected.
    """
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise InvalidMetadataError(
            f"{field_name} must be a mapping"
        )

    item_counter = [0]

    validated = _validate_json_value(
        dict(value),
        field_name=field_name,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string_length=max_string_length,
        item_counter=item_counter,
    )

    if not isinstance(validated, dict):
        raise InvalidMetadataError(
            f"{field_name} must resolve to an object"
        )

    return validated


def _validate_json_value(
    value: Any,
    *,
    field_name: str,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string_length: int,
    item_counter: list[int],
) -> Any:
    """Recursively validate JSON-compatible values."""
    if depth > max_depth:
        raise InvalidMetadataError(
            f"{field_name} exceeds maximum nesting depth "
            f"of {max_depth}"
        )

    item_counter[0] += 1

    if item_counter[0] > max_items:
        raise InvalidMetadataError(
            f"{field_name} exceeds maximum item count "
            f"of {max_items}"
        )

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidMetadataError(
                f"{field_name} contains a non-finite number"
            )

        return value

    if isinstance(value, str):
        if len(value) > max_string_length:
            raise InvalidMetadataError(
                f"{field_name} contains a string longer than "
                f"{max_string_length}"
            )

        if _CONTROL_CHARACTER_RE.search(value):
            raise InvalidMetadataError(
                f"{field_name} contains unsupported control characters"
            )

        return value

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}

        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise InvalidMetadataError(
                    f"{field_name} object keys must be strings"
                )

            clean_key = validate_required_string(
                key,
                field_name=f"{field_name} key",
                max_length=512,
            )

            output[clean_key] = _validate_json_value(
                nested_value,
                field_name=field_name,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                item_counter=item_counter,
            )

        return output

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _validate_json_value(
                item,
                field_name=field_name,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                item_counter=item_counter,
            )
            for item in value
        ]

    raise InvalidMetadataError(
        f"{field_name} contains unsupported value type "
        f"{type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Audit-safe text
# ---------------------------------------------------------------------------

def validate_audit_message(
    value: str | None,
    *,
    field_name: str = "audit_message",
) -> str | None:
    """Validate human-readable audit text."""
    if value is None:
        return None

    return validate_optional_string(
        value,
        field_name=field_name,
        max_length=MAX_AUDIT_MESSAGE_LENGTH,
        allow_newlines=True,
    )


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def require_same_tenant(
    *,
    expected_tenant_id: uuid.UUID | str,
    actual_tenant_id: uuid.UUID | str,
) -> uuid.UUID:
    """
    Validate two tenant identities and require them to match.

    This helper provides application-level defence in depth. PostgreSQL
    Row-Level Security should independently enforce tenant isolation.
    """
    expected = validate_tenant_id(
        expected_tenant_id
    )

    actual = validate_tenant_id(
        actual_tenant_id
    )

    if expected != actual:
        raise SecurityValidationError(
            "cross-tenant database operation rejected"
        )

    return expected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "MAX_AUDIT_MESSAGE_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_IDENTIFIER_LENGTH",
    "MAX_JSON_DEPTH",
    "MAX_JSON_ITEMS",
    "MAX_JSON_STRING_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_PAGE_SIZE",
    "MAX_PORT",
    "MIN_PORT",

    # Exceptions
    "InvalidChoiceError",
    "InvalidHostnameError",
    "InvalidIPAddressError",
    "InvalidIdentifierError",
    "InvalidMetadataError",
    "InvalidPaginationError",
    "InvalidPortError",
    "InvalidRangeError",
    "InvalidStringError",
    "InvalidUUIDError",
    "RequiredValueError",
    "SecurityValidationError",

    # UUID / tenant
    "require_same_tenant",
    "validate_optional_uuid",
    "validate_tenant_id",
    "validate_uuid",

    # Strings
    "validate_audit_message",
    "validate_description",
    "validate_identifier",
    "validate_optional_string",
    "validate_required_string",

    # Network
    "validate_hostname",
    "validate_ip_address",
    "validate_optional_hostname",
    "validate_optional_ip_address",
    "validate_optional_port",
    "validate_port",

    # Choices
    "validate_choice",
    "validate_enum_value",

    # Numeric
    "validate_finite_number",
    "validate_optional_probability",
    "validate_probability",
    "validate_range",

    # Query limits
    "validate_pagination",

    # Time
    "validate_time_window",

    # Structured metadata
    "validate_metadata",
]
