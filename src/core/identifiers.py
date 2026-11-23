from __future__ import annotations

import hashlib
import json
import re
import secrets
from enum import Enum
from typing import (
    Any,
    Optional,
)

from src.core.constants import (
    APPLICATION_NAMESPACE,

    ID_PREFIX_ORCHESTRATOR_CONTEXT,
    ID_PREFIX_ORCHESTRATOR_RUN,
    ID_PREFIX_ORCHESTRATOR_SCHEDULE,

    ID_PREFIX_RUNTIME_CONTEXT,
    ID_PREFIX_RUNTIME_INSTANCE,

    ID_PREFIX_WORKER_JOB,
    ID_PREFIX_WORKER_LEASE,
)

from src.core.exceptions import (
    IdentifierError,
)


# ---------------------------------------------------------------
# Validation patterns
# ---------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^[a-z][a-z0-9_-]*"
    r"(?::[a-z][a-z0-9_-]*)*$"
)

_TOKEN_RE = re.compile(
    r"^[a-zA-Z0-9]"
    r"[a-zA-Z0-9._-]*$"
)

_HEX_RE = re.compile(
    r"^[0-9a-f]+$"
)


# ---------------------------------------------------------------
# Standard identifier kinds
# ---------------------------------------------------------------

class IdentifierKind(
    str,
    Enum,
):
    RUNTIME_CONTEXT = (
        ID_PREFIX_RUNTIME_CONTEXT
    )

    RUNTIME_INSTANCE = (
        ID_PREFIX_RUNTIME_INSTANCE
    )

    ORCHESTRATOR_CONTEXT = (
        ID_PREFIX_ORCHESTRATOR_CONTEXT
    )

    ORCHESTRATOR_RUN = (
        ID_PREFIX_ORCHESTRATOR_RUN
    )

    ORCHESTRATOR_SCHEDULE = (
        ID_PREFIX_ORCHESTRATOR_SCHEDULE
    )

    WORKER_JOB = (
        ID_PREFIX_WORKER_JOB
    )

    WORKER_LEASE = (
        ID_PREFIX_WORKER_LEASE
    )


# ---------------------------------------------------------------
# Canonical serialization for hashing
# ---------------------------------------------------------------

def canonical_json(
    value: Any,
) -> str:
    """
    Convert a value to deterministic JSON for
    stable identifier hashing.

    Mapping keys are sorted and unnecessary
    whitespace is removed.
    """

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise IdentifierError(
            "Value cannot be "
            "canonicalized for "
            "identifier hashing."
        ) from exc


# ---------------------------------------------------------------
# Stable hash
# ---------------------------------------------------------------

def stable_hash(
    value: Any,
    *,
    algorithm: str = "sha256",
) -> str:
    """
    Produce a deterministic hexadecimal digest.

    SHA-256 is the default because these hashes
    are used primarily for stable identifiers,
    provenance and deduplication keys.
    """

    encoded = canonical_json(
        value
    ).encode(
        "utf-8"
    )

    try:
        digest = hashlib.new(
            algorithm
        )

    except ValueError as exc:
        raise IdentifierError(
            "Unsupported identifier "
            "hash algorithm: "
            f"{algorithm}"
        ) from exc

    digest.update(
        encoded
    )

    return digest.hexdigest()


# ---------------------------------------------------------------
# Prefix validation
# ---------------------------------------------------------------

def validate_prefix(
    prefix: str,
) -> str:
    normalized = str(
        prefix
    ).strip().lower()

    if not normalized:
        raise IdentifierError(
            "Identifier prefix is required."
        )

    if not _PREFIX_RE.fullmatch(
        normalized
    ):
        raise IdentifierError(
            "Invalid identifier prefix: "
            f"{prefix}"
        )

    return normalized


# ---------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------

def validate_token(
    token: str,
) -> str:
    normalized = str(
        token
    ).strip()

    if not normalized:
        raise IdentifierError(
            "Identifier token is required."
        )

    if not _TOKEN_RE.fullmatch(
        normalized
    ):
        raise IdentifierError(
            "Identifier token contains "
            "unsupported characters."
        )

    return normalized


# ---------------------------------------------------------------
# Prefix + token
# ---------------------------------------------------------------

def prefixed_id(
    prefix: str,
    token: str,
) -> str:
    """
    Construct:

        prefix:token
    """

    return (
        f"{validate_prefix(prefix)}:"
        f"{validate_token(token)}"
    )


# ---------------------------------------------------------------
# Deterministic identifier
# ---------------------------------------------------------------

def deterministic_id(
    prefix: str,
    value: Any,
    *,
    algorithm: str = "sha256",
    digest_length: Optional[
        int
    ] = None,
) -> str:
    """
    Create a deterministic ID from content.

    Example shape:

        worker:job:<digest>

    Repeating the same canonical input produces
    the same identifier.
    """

    digest = stable_hash(
        value,
        algorithm=algorithm,
    )

    if digest_length is not None:
        if digest_length < 8:
            raise IdentifierError(
                "digest_length must "
                "be >= 8 when truncating "
                "a hash."
            )

        if (
            digest_length
            > len(digest)
        ):
            raise IdentifierError(
                "digest_length exceeds "
                "the hash digest length."
            )

        digest = digest[
            :digest_length
        ]

    return prefixed_id(
        prefix,
        digest,
    )


# ---------------------------------------------------------------
# Random identifier
# ---------------------------------------------------------------

def random_id(
    prefix: str,
    *,
    entropy_bytes: int = 16,
) -> str:
    """
    Create a non-deterministic instance ID.

    Use this when identity should represent a
    newly created occurrence rather than content.
    """

    if entropy_bytes < 8:
        raise IdentifierError(
            "entropy_bytes must be >= 8."
        )

    return prefixed_id(
        prefix,
        secrets.token_hex(
            entropy_bytes
        ),
    )


# ---------------------------------------------------------------
# Parse identifier
# ---------------------------------------------------------------

def parse_identifier(
    identifier: str,
) -> tuple[
    str,
    str,
]:
    """
    Split an identifier at the final colon.

    Example:

        worker:job:abc123

    becomes:

        ("worker:job", "abc123")
    """

    text = str(
        identifier
    ).strip()

    if ":" not in text:
        raise IdentifierError(
            "Identifier must contain "
            "a namespace prefix and token."
        )

    prefix, token = text.rsplit(
        ":",
        1,
    )

    return (
        validate_prefix(
            prefix
        ),
        validate_token(
            token
        ),
    )


# ---------------------------------------------------------------
# Validate complete identifier
# ---------------------------------------------------------------

def validate_identifier(
    identifier: str,
    *,
    expected_prefix: Optional[
        str
    ] = None,
    require_hex_token: bool = False,
) -> str:
    """
    Validate an OSHA identifier and optionally
    enforce its namespace.
    """

    prefix, token = (
        parse_identifier(
            identifier
        )
    )

    if (
        expected_prefix
        is not None
        and prefix
        != validate_prefix(
            expected_prefix
        )
    ):
        raise IdentifierError(
            f"Identifier prefix {prefix} "
            "does not match expected "
            f"prefix {expected_prefix}."
        )

    if (
        require_hex_token
        and not _HEX_RE.fullmatch(
            token.lower()
        )
    ):
        raise IdentifierError(
            "Identifier token must be "
            "hexadecimal."
        )

    return (
        f"{prefix}:{token}"
    )


# ---------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------

def is_identifier(
    identifier: Any,
    *,
    expected_prefix: Optional[
        str
    ] = None,
) -> bool:
    """
    Boolean validation helper.
    """

    try:
        validate_identifier(
            str(identifier),
            expected_prefix=(
                expected_prefix
            ),
        )

        return True

    except (
        IdentifierError,
        TypeError,
        ValueError,
    ):
        return False


# ---------------------------------------------------------------
# Identifier-kind helper
# ---------------------------------------------------------------

def id_for(
    kind: IdentifierKind | str,
    value: Any,
    *,
    digest_length: Optional[
        int
    ] = None,
) -> str:
    """
    Create a deterministic identifier using a
    standard IdentifierKind or explicit prefix.
    """

    prefix = (
        kind.value
        if isinstance(
            kind,
            IdentifierKind,
        )
        else str(kind)
    )

    return deterministic_id(
        prefix,
        value,
        digest_length=(
            digest_length
        ),
    )


# ---------------------------------------------------------------
# Application namespace
# ---------------------------------------------------------------

def application_id(
    token: str,
) -> str:
    """
    Construct a general OSHA namespace ID:

        osha:<token>
    """

    return prefixed_id(
        APPLICATION_NAMESPACE,
        token,
    )


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "IdentifierKind",

    "canonical_json",
    "stable_hash",

    "validate_prefix",
    "validate_token",

    "prefixed_id",
    "deterministic_id",
    "random_id",

    "parse_identifier",
    "validate_identifier",
    "is_identifier",

    "id_for",
    "application_id",
]
