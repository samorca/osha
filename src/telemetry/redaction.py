from __future__ import annotations

import hashlib
import re
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
)


REDACTED = "[REDACTED]"

DROPPED = "[DROPPED]"

PRIVATE_KEY_REDACTED = (
    "[REDACTED_PRIVATE_KEY]"
)


# Keys whose values should not survive the
# telemetry boundary.
SENSITIVE_KEY_PATTERN = re.compile(
    r"""(?ix)
    (
        password
        | passwd
        | passphrase
        | secret
        | client_secret
        | api[_-]?key
        | access[_-]?key
        | secret[_-]?key
        | private[_-]?key
        | auth[_-]?token
        | refresh[_-]?token
        | access[_-]?token
        | identity[_-]?token
        | session[_-]?token
        | bearer[_-]?token
        | credential
        | credentials
        | application[_-]?credential
        | connection[_-]?string
        | database[_-]?url
        | database[_-]?password
        | kubeconfig
    )
    """
)


# Headers that commonly contain authentication
# or session material.
SENSITIVE_HEADER_PATTERN = re.compile(
    r"""(?ix)
    ^
    (
        authorization
        | proxy-authorization
        | x-auth-token
        | x-subject-token
        | x-api-key
        | cookie
        | set-cookie
    )
    $
    """
)


PRIVATE_KEY_PATTERN = re.compile(
    r"""
    -----BEGIN
    [^\n-]*
    PRIVATE[ ]KEY-----
    .*?
    -----END
    [^\n-]*
    PRIVATE[ ]KEY-----
    """,
    re.IGNORECASE
    | re.DOTALL
    | re.VERBOSE,
)


BEARER_PATTERN = re.compile(
    r"""(?ix)
    \bBearer
    \s+
    [A-Za-z0-9._~+/=-]+
    """
)


BASIC_AUTH_PATTERN = re.compile(
    r"""(?ix)
    \bBasic
    \s+
    [A-Za-z0-9+/=]+
    """
)


# URI userinfo:
#
# scheme://username:password@host
#
# The complete username/password section is
# removed rather than attempting to retain the
# username.
URL_USERINFO_PATTERN = re.compile(
    r"""(?ix)
    (
        [a-z][a-z0-9+.-]*://
    )
    [^/@\s]+
    @
    """
)


QUERY_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (
        [?\&]
        (
            password
            | passwd
            | secret
            | token
            | api[_-]?key
            | access[_-]?key
            | client[_-]?secret
        )
        =
    )
    [^&\s]*
    """
)


KEY_VALUE_SECRET_PATTERN = re.compile(
    r"""(?ix)
    \b
    (
        password
        | passwd
        | passphrase
        | secret
        | token
        | api[_-]?key
        | access[_-]?key
        | secret[_-]?key
        | client[_-]?secret
    )
    \s*
    (
        =
        |
        :
    )
    \s*
    (
        "[^"]*"
        |
        '[^']*'
        |
        [^\s,;]+
    )
    """
)


JWT_PATTERN = re.compile(
    r"""(?x)
    \b
    eyJ[A-Za-z0-9_-]+
    \.
    [A-Za-z0-9_-]+
    \.
    [A-Za-z0-9_-]+
    \b
    """
)


@dataclass(frozen=True)
class RedactionPolicy:
    """
    OSHA telemetry redaction policy.

    maximum_depth:
        Stops pathological recursive/nested
        telemetry.

    maximum_string_length:
        Prevents huge payloads from entering
        normalization accidentally.

    hash_keys:
        Keys whose values should remain
        correlatable but not readable.

    drop_keys:
        Keys that should be removed entirely.

    redact_unknown_secrets:
        Enables generic secret-name detection.
    """

    maximum_depth: int = 20

    maximum_string_length: int = (
        262_144
    )

    hash_keys: tuple[str, ...] = ()

    drop_keys: tuple[str, ...] = (
        "raw",
        "raw_event",
        "raw_payload",
        "notification_payload",
    )

    redact_unknown_secrets: bool = True


DEFAULT_POLICY = RedactionPolicy()


def normalize_key(
    key: Any,
) -> str:
    return str(key).strip()


def normalized_key_name(
    key: Any,
) -> str:
    """
    Normalize key spelling for direct policy
    comparisons.

    Examples:

        X-Auth-Token
        x_auth_token

    normalize to comparable forms.
    """

    return (
        normalize_key(key)
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


def is_sensitive_key(
    key: Any,
) -> bool:
    text = normalized_key_name(
        key
    )

    if SENSITIVE_HEADER_PATTERN.match(
        normalize_key(key).lower()
    ):
        return True

    return bool(
        SENSITIVE_KEY_PATTERN.search(
            text
        )
    )


def stable_hash(
    value: Any,
    *,
    salt: str = "",
) -> str:
    """
    Generate a stable one-way SHA-256 digest.

    A deployment-specific non-secret salt may be
    supplied when OSHA needs pseudonymous
    identifiers.

    Do not use this helper for password storage.
    """

    encoded = (
        f"{salt}{value}"
        .encode(
            "utf-8",
            errors="replace",
        )
    )

    return (
        "sha256:"
        + hashlib.sha256(
            encoded
        ).hexdigest()
    )


def redact_text(
    value: str,
    *,
    maximum_length: int = (
        DEFAULT_POLICY.
        maximum_string_length
    ),
) -> str:
    """
    Redact common secret material from
    unstructured strings such as logs.

    Pattern matching reduces accidental leakage,
    but it cannot prove that arbitrary text is
    non-sensitive.
    """

    text = value

    text = PRIVATE_KEY_PATTERN.sub(
        PRIVATE_KEY_REDACTED,
        text,
    )

    text = BEARER_PATTERN.sub(
        "Bearer [REDACTED]",
        text,
    )

    text = BASIC_AUTH_PATTERN.sub(
        "Basic [REDACTED]",
        text,
    )

    text = JWT_PATTERN.sub(
        "[REDACTED_JWT]",
        text,
    )

    text = URL_USERINFO_PATTERN.sub(
        r"\1[REDACTED]@",
        text,
    )

    text = QUERY_SECRET_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}"
            f"{REDACTED}"
        ),
        text,
    )

    text = KEY_VALUE_SECRET_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}"
            f"{match.group(2)}"
            f"{REDACTED}"
        ),
        text,
    )

    if len(text) > maximum_length:
        text = (
            text[:maximum_length]
            + "[TRUNCATED]"
        )

    return text


def should_drop_key(
    key: Any,
    policy: RedactionPolicy,
) -> bool:
    normalized = normalized_key_name(
        key
    )

    configured = {
        normalized_key_name(item)
        for item in policy.drop_keys
    }

    return normalized in configured


def should_hash_key(
    key: Any,
    policy: RedactionPolicy,
) -> bool:
    normalized = normalized_key_name(
        key
    )

    configured = {
        normalized_key_name(item)
        for item in policy.hash_keys
    }

    return normalized in configured


def redact_mapping(
    value: Mapping[Any, Any],
    *,
    policy: RedactionPolicy,
    depth: int,
    hash_salt: str,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    for raw_key, raw_value in (
        value.items()
    ):
        key = normalize_key(
            raw_key
        )

        if should_drop_key(
            key,
            policy,
        ):
            continue

        if should_hash_key(
            key,
            policy,
        ):
            output[key] = stable_hash(
                raw_value,
                salt=hash_salt,
            )

            continue

        if (
            policy.redact_unknown_secrets
            and is_sensitive_key(key)
        ):
            output[key] = REDACTED
            continue

        output[key] = redact(
            raw_value,
            policy=policy,
            depth=depth + 1,
            hash_salt=hash_salt,
        )

    return output


def redact(
    value: Any,
    *,
    policy: RedactionPolicy = (
        DEFAULT_POLICY
    ),
    depth: int = 0,
    hash_salt: str = "",
) -> Any:
    """
    Recursively sanitize arbitrary telemetry.

    This is the primary redaction entry point.
    """

    if depth > policy.maximum_depth:
        return "[MAX_DEPTH_REACHED]"

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return redact_text(
            value,
            maximum_length=(
                policy.
                maximum_string_length
            ),
        )

    if isinstance(
        value,
        bytes,
    ):
        decoded = value.decode(
            "utf-8",
            errors="replace",
        )

        return redact_text(
            decoded,
            maximum_length=(
                policy.
                maximum_string_length
            ),
        )

    if isinstance(
        value,
        Mapping,
    ):
        return redact_mapping(
            value,
            policy=policy,
            depth=depth,
            hash_salt=hash_salt,
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            redact(
                item,
                policy=policy,
                depth=depth + 1,
                hash_salt=hash_salt,
            )
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        from dataclasses import (
            asdict,
        )

        return redact(
            asdict(value),
            policy=policy,
            depth=depth + 1,
            hash_salt=hash_salt,
        )

    return redact_text(
        str(value),
        maximum_length=(
            policy.
            maximum_string_length
        ),
    )


def redact_attributes(
    attributes: Optional[
        Mapping[str, Any]
    ],
    *,
    policy: RedactionPolicy = (
        DEFAULT_POLICY
    ),
    hash_salt: str = "",
) -> Dict[str, Any]:
    """
    Convenience function for Resource/Signal
    attribute dictionaries.
    """

    if not attributes:
        return {}

    result = redact(
        attributes,
        policy=policy,
        hash_salt=hash_salt,
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {}


def redact_native_record(
    record: Mapping[str, Any],
    *,
    policy: RedactionPolicy = (
        DEFAULT_POLICY
    ),
    hash_salt: str = "",
) -> Dict[str, Any]:
    """
    Sanitize a provider-native collector record
    before normalize.py keeps it under the
    TelemetryRecord.native field.
    """

    result = redact(
        record,
        policy=policy,
        hash_salt=hash_salt,
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {}


def contains_redaction(
    value: Any,
) -> bool:
    """
    Return whether a sanitized structure contains
    one of OSHA's redaction markers.

    Useful for diagnostics and validation.
    """

    markers = {
        REDACTED,
        DROPPED,
        PRIVATE_KEY_REDACTED,
        "[REDACTED_JWT]",
    }

    if isinstance(
        value,
        str,
    ):
        return any(
            marker in value
            for marker in markers
        )

    if isinstance(
        value,
        Mapping,
    ):
        return any(
            contains_redaction(item)
            for item in value.values()
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return any(
            contains_redaction(item)
            for item in value
        )

    return False


def build_policy(
    *,
    hash_keys: Iterable[str] = (),
    drop_keys: Iterable[str] = (
        DEFAULT_POLICY.drop_keys
    ),
    maximum_depth: int = (
        DEFAULT_POLICY.maximum_depth
    ),
    maximum_string_length: int = (
        DEFAULT_POLICY.
        maximum_string_length
    ),
    redact_unknown_secrets: bool = (
        True
    ),
) -> RedactionPolicy:
    """
    Create an immutable policy for a specific
    OSHA deployment or normalization pipeline.
    """

    return RedactionPolicy(
        maximum_depth=(
            maximum_depth
        ),
        maximum_string_length=(
            maximum_string_length
        ),
        hash_keys=tuple(
            hash_keys
        ),
        drop_keys=tuple(
            drop_keys
        ),
        redact_unknown_secrets=(
            redact_unknown_secrets
        ),
    )
