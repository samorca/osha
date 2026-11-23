from __future__ import annotations

"""
Dependency-free validation primitives shared by
OSHA subsystems.
"""

import math
import re

from enum import Enum
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from src.core.exceptions import (
    IdentifierError,
    ValidationError,
)

from src.core.identifiers import (
    validate_identifier,
)


# ---------------------------------------------------------------
# Generic types
# ---------------------------------------------------------------

E = TypeVar(
    "E",
    bound=Enum,
)


# ---------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------

_SAFE_NAME_RE = re.compile(
    r"^[A-Za-z0-9]"
    r"[A-Za-z0-9._-]*$"
)

_ENVIRONMENT_ID_RE = re.compile(
    r"^[A-Za-z0-9]"
    r"[A-Za-z0-9._:/-]*$"
)


# ---------------------------------------------------------------
# Required string
# ---------------------------------------------------------------

def required_string(
    value: Any,
    *,
    name: str = "value",
    minimum_length: int = 1,
    maximum_length: Optional[
        int
    ] = None,
    strip: bool = True,
) -> str:
    """
    Validate and normalize a required string.
    """

    if value is None:
        raise ValidationError(
            f"{name} is required."
        )

    if minimum_length < 0:
        raise ValidationError(
            "minimum_length must be >= 0."
        )

    if (
        maximum_length is not None
        and maximum_length
        < minimum_length
    ):
        raise ValidationError(
            "maximum_length cannot be "
            "less than minimum_length."
        )

    result = str(
        value
    )

    if strip:
        result = result.strip()

    if (
        len(result)
        < minimum_length
    ):
        raise ValidationError(
            f"{name} must contain at "
            f"least {minimum_length} "
            "character(s)."
        )

    if (
        maximum_length is not None
        and len(result)
        > maximum_length
    ):
        raise ValidationError(
            f"{name} must contain at "
            f"most {maximum_length} "
            "character(s)."
        )

    return result


# ---------------------------------------------------------------
# Optional string
# ---------------------------------------------------------------

def optional_string(
    value: Any,
    *,
    name: str = "value",
    maximum_length: Optional[
        int
    ] = None,
) -> Optional[str]:
    """
    Normalize an optional string.

    None and blank strings become None.
    """

    if value is None:
        return None

    result = str(
        value
    ).strip()

    if not result:
        return None

    return required_string(
        result,
        name=name,
        maximum_length=(
            maximum_length
        ),
    )


# ---------------------------------------------------------------
# Integer
# ---------------------------------------------------------------

def integer(
    value: Any,
    *,
    name: str = "value",
    minimum: Optional[
        int
    ] = None,
    maximum: Optional[
        int
    ] = None,
) -> int:
    """
    Convert and validate an integer.

    bool is explicitly rejected even though bool
    is a Python subclass of int.
    """

    if isinstance(
        value,
        bool,
    ):
        raise ValidationError(
            f"{name} must be an integer."
        )

    try:
        result = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationError(
            f"{name} must be an integer."
        ) from exc

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValidationError(
            f"{name} must be >= "
            f"{minimum}."
        )

    if (
        maximum is not None
        and result > maximum
    ):
        raise ValidationError(
            f"{name} must be <= "
            f"{maximum}."
        )

    return result


# ---------------------------------------------------------------
# Number
# ---------------------------------------------------------------

def number(
    value: Any,
    *,
    name: str = "value",
    minimum: Optional[
        float
    ] = None,
    maximum: Optional[
        float
    ] = None,
    finite: bool = True,
) -> float:
    """
    Convert and validate a floating-point value.
    """

    if isinstance(
        value,
        bool,
    ):
        raise ValidationError(
            f"{name} must be numeric."
        )

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationError(
            f"{name} must be numeric."
        ) from exc

    if (
        finite
        and not math.isfinite(
            result
        )
    ):
        raise ValidationError(
            f"{name} must be finite."
        )

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValidationError(
            f"{name} must be >= "
            f"{minimum}."
        )

    if (
        maximum is not None
        and result > maximum
    ):
        raise ValidationError(
            f"{name} must be <= "
            f"{maximum}."
        )

    return result


# ---------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------

def boolean(
    value: Any,
    *,
    name: str = "value",
) -> bool:
    """
    Validate common boolean representations.
    """

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        normalized = (
            value.strip().lower()
        )

        if normalized in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False

    raise ValidationError(
        f"{name} must be a boolean."
    )


# ---------------------------------------------------------------
# Allowed value
# ---------------------------------------------------------------

def one_of(
    value: Any,
    allowed: Iterable[Any],
    *,
    name: str = "value",
) -> Any:
    """
    Require value to exist in the supplied
    collection.
    """

    values = tuple(
        allowed
    )

    if value not in values:
        raise ValidationError(
            f"{name} must be one of "
            f"{list(values)}."
        )

    return value


# ---------------------------------------------------------------
# Enum value
# ---------------------------------------------------------------

def enum_value(
    value: Any,
    enum_type: Type[E],
    *,
    name: str = "value",
) -> str:
    """
    Accept either an enum instance or its
    underlying string value.
    """

    if isinstance(
        value,
        enum_type,
    ):
        return str(
            value.value
        )

    allowed = {
        str(
            item.value
        )
        for item in enum_type
    }

    result = str(
        value
    )

    if result not in allowed:
        raise ValidationError(
            f"{name} must be one of "
            f"{sorted(allowed)}."
        )

    return result


# ---------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------

def mapping(
    value: Any,
    *,
    name: str = "value",
    allow_none: bool = False,
) -> Mapping[
    Any,
    Any,
]:
    if (
        value is None
        and allow_none
    ):
        return {}

    if not isinstance(
        value,
        Mapping,
    ):
        raise ValidationError(
            f"{name} must be a mapping."
        )

    return value


# ---------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------

def sequence(
    value: Any,
    *,
    name: str = "value",
    allow_string: bool = False,
    minimum_items: int = 0,
    maximum_items: Optional[
        int
    ] = None,
) -> Sequence[Any]:
    """
    Validate a sequence.

    Strings/bytes are rejected by default because
    callers usually mean collections of values.
    """

    if (
        isinstance(
            value,
            (
                str,
                bytes,
            ),
        )
        and not allow_string
    ):
        raise ValidationError(
            f"{name} must be a sequence, "
            "not a string."
        )

    if not isinstance(
        value,
        Sequence,
    ):
        raise ValidationError(
            f"{name} must be a sequence."
        )

    if (
        len(value)
        < minimum_items
    ):
        raise ValidationError(
            f"{name} must contain at least "
            f"{minimum_items} item(s)."
        )

    if (
        maximum_items is not None
        and len(value)
        > maximum_items
    ):
        raise ValidationError(
            f"{name} must contain at most "
            f"{maximum_items} item(s)."
        )

    return value


# ---------------------------------------------------------------
# Unique strings
# ---------------------------------------------------------------

def unique_strings(
    values: Iterable[Any],
    *,
    name: str = "values",
    allow_empty: bool = True,
) -> list[str]:
    """
    Normalize a collection of strings while
    preserving first-seen order.
    """

    result: list[
        str
    ] = []

    seen = set()

    for item in values:
        text = str(
            item
        ).strip()

        if not text:
            if allow_empty:
                continue

            raise ValidationError(
                f"{name} cannot contain "
                "empty values."
            )

        if text not in seen:
            seen.add(
                text
            )

            result.append(
                text
            )

    return result


# ---------------------------------------------------------------
# Safe simple name
# ---------------------------------------------------------------

def safe_name(
    value: Any,
    *,
    name: str = "name",
    maximum_length: int = 128,
) -> str:
    """
    Validate a simple application name.

    Accepted characters:

        A-Z a-z 0-9 . _ -
    """

    result = required_string(
        value,
        name=name,
        maximum_length=(
            maximum_length
        ),
    )

    if not _SAFE_NAME_RE.fullmatch(
        result
    ):
        raise ValidationError(
            f"{name} may contain only "
            "letters, numbers, dots, "
            "underscores, and hyphens."
        )

    return result


# -----------------------------------------------------------------------------------------------------------------

def environment_id(
    value: Any,
    *,
    name: str = "environment_id",
) -> str:
    """
    Validate a general OSHA environment ID.

    This permits simple hierarchical characters
    such as ':' and '/' without interpreting
    their semantics.
    """

    result = required_string(
        value,
        name=name,
        maximum_length=256,
    )

    if not _ENVIRONMENT_ID_RE.fullmatch(
        result
    ):
        raise ValidationError(
            f"{name} contains unsupported "
            "characters."
        )

    return result


# ---------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------

def identifier(
    value: Any,
    *,
    name: str = "identifier",
    expected_prefix: Optional[
        str
    ] = None,
) -> str:
    """
    Validate a namespaced OSHA identifier.
    """

    try:
        return validate_identifier(
            str(value),
            expected_prefix=(
                expected_prefix
            ),
        )

    except IdentifierError as exc:
        raise ValidationError(
            f"{name} is invalid.",
            details={
                "expected_prefix": (
                    expected_prefix
                ),
            },
        ) from exc


# ---------------------------------------------------------------
# Presence helper
# ---------------------------------------------------------------

def _present(
    value: Any,
) -> bool:
    """
    Determine whether a field counts as supplied.

    False and zero are intentionally considered
    present values.
    """

    if value is None:
        return False

    if isinstance(
        value,
        str,
    ):
        return bool(
            value.strip()
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            dict,
            set,
            frozenset,
        ),
    ):
        return bool(
            value
        )

    return True


# ---------------------------------------------------------------
# At least one field
# ---------------------------------------------------------------

def require_any(
    fields: Mapping[
        str,
        Any,
    ],
    *,
    names: Iterable[str],
) -> None:
    """
    Require at least one named field to be
    present.
    """

    name_list = list(
        names
    )

    selected = [
        name
        for name in name_list
        if _present(
            fields.get(
                name
            )
        )
    ]

    if not selected:
        raise ValidationError(
            "At least one of "
            f"{name_list} is required."
        )


# ---------------------------------------------------------------
# Exactly one field
# ---------------------------------------------------------------

def require_exactly_one(
    fields: Mapping[
        str,
        Any,
    ],
    *,
    names: Iterable[str],
) -> str:
    """
    Require exactly one of the named fields.

    Returns the selected field name.
    """

    name_list = list(
        names
    )

    selected = [
        name
        for name in name_list
        if _present(
            fields.get(
                name
            )
        )
    ]

    if len(selected) != 1:
        raise ValidationError(
            "Exactly one of "
            f"{name_list} is required."
        )

    return selected[
        0
    ]


# ---------------------------------------------------------------
# Mutually exclusive fields
# ---------------------------------------------------------------

def mutually_exclusive(
    fields: Mapping[
        str,
        Any,
    ],
    *,
    names: Iterable[str],
) -> None:
    """
    Allow zero or one named field, but reject two
    or more.
    """

    name_list = list(
        names
    )

    selected = [
        name
        for name in name_list
        if _present(
            fields.get(
                name
            )
        )
    ]

    if len(selected) > 1:
        raise ValidationError(
            "Fields are mutually "
            f"exclusive: {selected}."
        )


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "required_string",
    "optional_string",

    "integer",
    "number",
    "boolean",

    "one_of",
    "enum_value",

    "mapping",
    "sequence",
    "unique_strings",

    "safe_name",
    "environment_id",
    "identifier",

    "require_any",
    "require_exactly_one",
    "mutually_exclusive",
]
