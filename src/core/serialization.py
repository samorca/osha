from __future__ import annotations

"""
Bounded JSON-safe serialization helpers for
OSHA.
"""

import json
import math

from dataclasses import (
    fields,
    is_dataclass,
)

from datetime import (
    date,
    datetime,
)

from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Set,
)

from src.core.constants import (
    DEFAULT_MAX_SERIALIZATION_DEPTH,
    MAX_DEPTH_REACHED_MARKER,
)

from src.core.exceptions import (
    SerializationError,
)

from src.core.time import (
    ensure_utc,
)


# ---------------------------------------------------------------
# Serialization markers
# ---------------------------------------------------------------

DEFAULT_REDACTED_MARKER = "[REDACTED]"

DEFAULT_CYCLE_MARKER = "[CYCLE]"

DEFAULT_UNSUPPORTED_MARKER = "[UNSUPPORTED]"


# ---------------------------------------------------------------
# Sensitive field names
# ---------------------------------------------------------------

DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "private_key",
        "client_secret",
        "credential",
        "credentials",
    }
)


# ---------------------------------------------------------------
# Sensitive-key helpers
# ---------------------------------------------------------------


def _normalized_keys(
    keys: Optional[Iterable[str]],
) -> Set[str]:
    values = DEFAULT_SENSITIVE_KEYS if keys is None else keys

    return {str(item).strip().lower() for item in values if str(item).strip()}


def _is_sensitive(
    key: Any,
    *,
    sensitive_keys: Set[str],
) -> bool:
    return str(key).strip().lower() in sensitive_keys


# ---------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------


def to_json_safe(
    value: Any,
    *,
    maximum_depth: int = (DEFAULT_MAX_SERIALIZATION_DEPTH),
    redact_sensitive: bool = True,
    sensitive_keys: Optional[Iterable[str]] = None,
    redacted_marker: str = (DEFAULT_REDACTED_MARKER),
    unsupported_marker: str = (DEFAULT_UNSUPPORTED_MARKER),
    _depth: int = 0,
    _seen: Optional[Set[int]] = None,
) -> Any:
    """
    Convert known values into bounded,
    JSON-compatible structures.

    Supported values include:

      None
      strings
      booleans
      integers/floats
      Enums
      datetime/date
      pathlib.Path
      mappings
      dataclass instances
      lists/tuples
      sets/frozensets

    Arbitrary object __dict__ values are
    deliberately NOT traversed.

    Unknown objects become [UNSUPPORTED]
    rather than exposing internal attributes.
    """

    if maximum_depth < 0:
        raise SerializationError("maximum_depth must be >= 0.")

    if _depth > maximum_depth:
        return MAX_DEPTH_REACHED_MARKER

    # ---------------------------------------------------------
    # Simple values
    # ---------------------------------------------------------

    if value is None or isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    # ---------------------------------------------------------
    # Floating point
    # ---------------------------------------------------------

    if isinstance(
        value,
        float,
    ):
        # JSON NaN/Infinity behavior varies
        # between consumers. Keep output strict.
        return value if math.isfinite(value) else None

    # ---------------------------------------------------------
    # Enum
    # ---------------------------------------------------------

    if isinstance(
        value,
        Enum,
    ):
        return to_json_safe(
            value.value,
            maximum_depth=(maximum_depth),
            redact_sensitive=(redact_sensitive),
            sensitive_keys=(sensitive_keys),
            redacted_marker=(redacted_marker),
            unsupported_marker=(unsupported_marker),
            _depth=_depth + 1,
            _seen=_seen,
        )

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------

    if isinstance(
        value,
        datetime,
    ):
        return ensure_utc(value).isoformat()

    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    # ---------------------------------------------------------
    # Path
    # ---------------------------------------------------------

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    # ---------------------------------------------------------
    # Cycle tracking
    # ---------------------------------------------------------

    seen = set() if _seen is None else _seen

    track_identity = isinstance(
        value,
        (
            Mapping,
            list,
            tuple,
            set,
            frozenset,
        ),
    ) or is_dataclass(value)

    identity = id(value)

    if track_identity:
        if identity in seen:
            return DEFAULT_CYCLE_MARKER

        seen.add(identity)

    keys = _normalized_keys(sensitive_keys)

    try:
        # -----------------------------------------------------
        # Mapping
        # -----------------------------------------------------

        if isinstance(
            value,
            Mapping,
        ):
            output: Dict[
                str,
                Any,
            ] = {}

            for key, item in value.items():
                key_text = str(key)

                if redact_sensitive and _is_sensitive(
                    key_text,
                    sensitive_keys=keys,
                ):
                    output[key_text] = redacted_marker

                    continue

                output[key_text] = to_json_safe(
                    item,
                    maximum_depth=(maximum_depth),
                    redact_sensitive=(redact_sensitive),
                    sensitive_keys=keys,
                    redacted_marker=(redacted_marker),
                    unsupported_marker=(unsupported_marker),
                    _depth=_depth + 1,
                    _seen=seen,
                )

            return output

        # -----------------------------------------------------
        # Dataclass
        # -----------------------------------------------------

        if is_dataclass(value) and not isinstance(
            value,
            type,
        ):
            output: Dict[
                str,
                Any,
            ] = {}

            for definition in fields(value):
                key = definition.name

                if redact_sensitive and _is_sensitive(
                    key,
                    sensitive_keys=keys,
                ):
                    output[key] = redacted_marker

                    continue

                output[key] = to_json_safe(
                    getattr(
                        value,
                        key,
                    ),
                    maximum_depth=(maximum_depth),
                    redact_sensitive=(redact_sensitive),
                    sensitive_keys=keys,
                    redacted_marker=(redacted_marker),
                    unsupported_marker=(unsupported_marker),
                    _depth=_depth + 1,
                    _seen=seen,
                )

            return output

        # -----------------------------------------------------
        # List / tuple
        # -----------------------------------------------------

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return [
                to_json_safe(
                    item,
                    maximum_depth=(maximum_depth),
                    redact_sensitive=(redact_sensitive),
                    sensitive_keys=keys,
                    redacted_marker=(redacted_marker),
                    unsupported_marker=(unsupported_marker),
                    _depth=_depth + 1,
                    _seen=seen,
                )
                for item in value
            ]

        # -----------------------------------------------------
        # Set
        # -----------------------------------------------------

        if isinstance(
            value,
            (
                set,
                frozenset,
            ),
        ):
            serialized = [
                to_json_safe(
                    item,
                    maximum_depth=(maximum_depth),
                    redact_sensitive=(redact_sensitive),
                    sensitive_keys=keys,
                    redacted_marker=(redacted_marker),
                    unsupported_marker=(unsupported_marker),
                    _depth=_depth + 1,
                    _seen=seen,
                )
                for item in value
            ]

            # Sets have no reliable iteration
            # order. Sort by canonical JSON so
            # the serialized output is stable.
            return sorted(
                serialized,
                key=lambda item: canonical_json(
                    item,
                    redact_sensitive=False,
                ),
            )

        # -----------------------------------------------------
        # Unknown object
        # -----------------------------------------------------

        return unsupported_marker

    finally:
        if track_identity:
            seen.discard(identity)


# ---------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------


def canonical_json(
    value: Any,
    *,
    redact_sensitive: bool = True,
    sensitive_keys: Optional[Iterable[str]] = None,
) -> str:
    """
    Return deterministic compact JSON.

    Useful for:

      hashing inputs
      stable comparisons
      deterministic metadata
    """

    safe = to_json_safe(
        value,
        redact_sensitive=(redact_sensitive),
        sensitive_keys=(sensitive_keys),
    )

    try:
        return json.dumps(
            safe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SerializationError(
            "Value could not be serialized to canonical JSON."
        ) from exc


# ---------------------------------------------------------------
# Pretty JSON
# ---------------------------------------------------------------


def pretty_json(
    value: Any,
    *,
    indent: int = 2,
    redact_sensitive: bool = True,
    sensitive_keys: Optional[Iterable[str]] = None,
) -> str:
    """
    Return readable JSON suitable for bounded
    diagnostic output.
    """

    if indent < 0:
        raise SerializationError("indent must be >= 0.")

    safe = to_json_safe(
        value,
        redact_sensitive=(redact_sensitive),
        sensitive_keys=(sensitive_keys),
    )

    try:
        return json.dumps(
            safe,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SerializationError("Value could not be serialized to JSON.") from exc


# ---------------------------------------------------------------
# Explicit mapping redaction
# ---------------------------------------------------------------


def redact_mapping(
    value: Mapping[
        str,
        Any,
    ],
    *,
    sensitive_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Convert a mapping to a JSON-safe redacted
    dictionary.
    """

    safe = to_json_safe(
        value,
        redact_sensitive=True,
        sensitive_keys=(sensitive_keys),
    )

    if not isinstance(
        safe,
        dict,
    ):
        raise SerializationError("Expected a mapping serialization result.")

    return safe


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_REDACTED_MARKER",
    "DEFAULT_CYCLE_MARKER",
    "DEFAULT_UNSUPPORTED_MARKER",
    "DEFAULT_SENSITIVE_KEYS",
    "to_json_safe",
    "canonical_json",
    "pretty_json",
    "redact_mapping",
]
