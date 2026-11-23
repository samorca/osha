from __future__ import annotations

"""
UTC time and duration helpers shared across
OSHA subsystems.
"""

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Optional,
    Union,
)

from src.core.exceptions import (
    ValidationError,
)


# ---------------------------------------------------------------
# Types
# ---------------------------------------------------------------

DatetimeLike = Union[
    str,
    datetime,
]


# ---------------------------------------------------------------
# Current UTC time
# ---------------------------------------------------------------

def utc_datetime(
) -> datetime:
    """
    Return the current timezone-aware UTC
    datetime.
    """

    return datetime.now(
        timezone.utc
    )


def utc_now(
    *,
    timespec: str = "microseconds",
) -> str:
    """
    Return the current UTC timestamp in ISO-8601
    form.
    """

    return (
        utc_datetime().
        isoformat(
            timespec=timespec
        )
    )


# ---------------------------------------------------------------
# UTC normalization
# ---------------------------------------------------------------

def ensure_utc(
    value: datetime,
) -> datetime:
    """
    Normalize a datetime to timezone-aware UTC.

    Naive datetime values are interpreted as UTC.

    If a caller receives a naive LOCAL datetime,
    it should attach the correct timezone before
    calling this helper.
    """

    if not isinstance(
        value,
        datetime,
    ):
        raise ValidationError(
            "value must be a datetime "
            "instance."
        )

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


# ---------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------

def parse_timestamp(
    value: DatetimeLike,
) -> datetime:
    """
    Parse a datetime or ISO-8601 string and
    normalize it to UTC.

    Both:

        2026-11-20T10:00:00+00:00

    and:

        2026-11-20T10:00:00Z

    are supported.
    """

    if isinstance(
        value,
        datetime,
    ):
        return ensure_utc(
            value
        )

    text = str(
        value
    ).strip()

    if not text:
        raise ValidationError(
            "timestamp is required."
        )

    if text.endswith(
        (
            "Z",
            "z",
        )
    ):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = (
            datetime.fromisoformat(
                text
            )
        )

    except ValueError as exc:
        raise ValidationError(
            "timestamp must be valid "
            "ISO-8601."
        ) from exc

    return ensure_utc(
        parsed
    )


# ---------------------------------------------------------------
# Optional timestamp
# ---------------------------------------------------------------

def optional_timestamp(
    value: Optional[
        DatetimeLike
    ],
) -> Optional[
    datetime
]:
    """
    Convert optional timestamp input.

    None and an empty string become None.
    """

    if value is None:
        return None

    if (
        isinstance(
            value,
            str,
        )
        and not value.strip()
    ):
        return None

    return parse_timestamp(
        value
    )


# ---------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------

def format_timestamp(
    value: DatetimeLike,
    *,
    timespec: str = "microseconds",
) -> str:
    """
    Normalize and return an ISO-8601 UTC
    timestamp.
    """

    return (
        parse_timestamp(
            value
        ).isoformat(
            timespec=timespec
        )
    )


# ---------------------------------------------------------------
# Duration calculation
# ---------------------------------------------------------------

def duration_seconds(
    start: DatetimeLike,
    end: DatetimeLike,
) -> float:
    """
    Return seconds from start to end.

    Negative durations are deliberately
    preserved so callers can detect reversed
    timestamps.
    """

    return (
        parse_timestamp(
            end
        )
        - parse_timestamp(
            start
        )
    ).total_seconds()


# ---------------------------------------------------------------
# Add duration
# ---------------------------------------------------------------

def add_seconds(
    value: DatetimeLike,
    seconds: float,
) -> datetime:
    """
    Add or subtract seconds from a timestamp.

    Negative seconds move backward in time.
    """

    return (
        parse_timestamp(
            value
        )
        + timedelta(
            seconds=float(
                seconds
            )
        )
    )


# ---------------------------------------------------------------
# Future timestamp
# ---------------------------------------------------------------

def timestamp_after(
    seconds: float,
    *,
    now: Optional[
        DatetimeLike
    ] = None,
) -> str:
    """
    Return an ISO-8601 timestamp N seconds after
    a supplied/current UTC time.

    Useful later for:

        worker leases
        retries
        expirations
        scheduling
    """

    base = (
        utc_datetime()
        if now is None
        else parse_timestamp(
            now
        )
    )

    return (
        base
        + timedelta(
            seconds=float(
                seconds
            )
        )
    ).isoformat()


# ---------------------------------------------------------------
# Deadline reached
# ---------------------------------------------------------------

def has_reached(
    target: DatetimeLike,
    *,
    now: Optional[
        DatetimeLike
    ] = None,
) -> bool:
    """
    Return True when current/supplied time is at
    or beyond target.
    """

    current = (
        utc_datetime()
        if now is None
        else parse_timestamp(
            now
        )
    )

    return (
        current
        >= parse_timestamp(
            target
        )
    )


# ---------------------------------------------------------------
# Expiration
# ---------------------------------------------------------------

def is_expired(
    expires_at: Optional[
        DatetimeLike
    ],
    *,
    now: Optional[
        DatetimeLike
    ] = None,
) -> bool:
    """
    Return False when no expiration was provided.

    Otherwise return whether expiration has been
    reached.
    """

    if expires_at is None:
        return False

    if (
        isinstance(
            expires_at,
            str,
        )
        and not expires_at.strip()
    ):
        return False

    return has_reached(
        expires_at,
        now=now,
    )


# ---------------------------------------------------------------
# Remaining duration
# ---------------------------------------------------------------

def remaining_seconds(
    target: DatetimeLike,
    *,
    now: Optional[
        DatetimeLike
    ] = None,
    floor_zero: bool = True,
) -> float:
    """
    Return time remaining until target.

    By default expired targets return 0 rather
    than a negative value.
    """

    current = (
        utc_datetime()
        if now is None
        else parse_timestamp(
            now
        )
    )

    remaining = (
        parse_timestamp(
            target
        )
        - current
    ).total_seconds()

    if floor_zero:
        return max(
            0.0,
            remaining,
        )

    return remaining


# ---------------------------------------------------------------
# Duration validation
# ---------------------------------------------------------------

def bounded_seconds(
    value: float,
    *,
    minimum: float = 0.0,
    maximum: Optional[
        float
    ] = None,
    name: str = "duration",
) -> float:
    """
    Validate and optionally clamp a duration.

    Values below minimum are invalid.

    Values above maximum are clamped when a
    maximum is supplied.
    """

    try:
        seconds = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationError(
            f"{name} must be numeric."
        ) from exc

    if seconds < minimum:
        raise ValidationError(
            f"{name} must be >= "
            f"{minimum} seconds."
        )

    if maximum is not None:
        if maximum < minimum:
            raise ValidationError(
                "maximum duration cannot "
                "be less than minimum "
                "duration."
            )

        seconds = min(
            seconds,
            maximum,
        )

    return seconds


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "DatetimeLike",

    "utc_datetime",
    "utc_now",

    "ensure_utc",

    "parse_timestamp",
    "optional_timestamp",
    "format_timestamp",

    "duration_seconds",
    "add_seconds",
    "timestamp_after",

    "has_reached",
    "is_expired",
    "remaining_seconds",

    "bounded_seconds",
]
