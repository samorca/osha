"""
Workload identity model for OSHA.

This module defines the canonical identity representation used by OSHA
workloads.

Examples of workloads include:

- Linux collectors
- Kubernetes collectors
- OpenStack collectors
- MinIO collectors
- training workers
- inference workers
- correlation services
- intelligence services
- decision services
- remediation workers
- API services
- audit services

Purpose
-------
Cryptography can establish integrity and authenticity, but OSHA still needs
a canonical way to describe *who* performed an operation.

This module provides that identity representation.

It intentionally does NOT:

- issue X.509 certificates
- generate private keys
- authenticate operating-system processes
- authenticate Kubernetes service accounts
- make authorization decisions

Those responsibilities belong to certificate, authentication, and policy
layers.

Security model
--------------
1. Identity is represented by immutable structured data.
2. Workload IDs are globally namespaced within an OSHA trust domain.
3. Environment and workload type are explicit.
4. Cryptographic key references may be bound to an identity.
5. Identity records contain no private keys or credentials.
6. Serialization is deterministic and suitable for hashing/signing.
7. Authentication status is kept separate from identity description.
"""

from __future__ import annotations

import re
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping
from urllib.parse import urlparse

from ..crypto.hashing import (
    Digest,
    HashAlgorithm,
    canonical_json_bytes,
    hash_json,
)
from ..crypto.key_provider import KeyReference


WORKLOAD_IDENTITY_FORMAT: Final[str] = "osha-workload-identity"
WORKLOAD_IDENTITY_VERSION: Final[int] = 1

DEFAULT_TRUST_DOMAIN: Final[str] = "osha.local"

MAX_ID_LENGTH: Final[int] = 256
MAX_NAME_LENGTH: Final[int] = 128
MAX_NAMESPACE_LENGTH: Final[int] = 128
MAX_TRUST_DOMAIN_LENGTH: Final[int] = 253
MAX_LABELS: Final[int] = 64
MAX_LABEL_KEY_LENGTH: Final[int] = 64
MAX_LABEL_VALUE_LENGTH: Final[int] = 256

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9]"
    r"[a-z0-9._-]*"
    r"[a-z0-9]$"
    r"|^[a-z0-9]$"
)

_TRUST_DOMAIN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9]"
    r"(?:[a-z0-9.-]*[a-z0-9])?$"
)


class WorkloadIdentityError(Exception):
    """Base exception for workload identity failures."""


class InvalidWorkloadIdentityError(
    WorkloadIdentityError,
    ValueError,
):
    """A workload identity is malformed or unsupported."""


class WorkloadIdentityMismatchError(
    WorkloadIdentityError,
):
    """Two workload identities do not represent the same identity."""


class WorkloadType(str, Enum):
    """Known OSHA workload classes."""

    COLLECTOR = "collector"
    INGESTION = "ingestion"
    TRAINING = "training"
    INFERENCE = "inference"
    CORRELATION = "correlation"
    INTELLIGENCE = "intelligence"
    MEMORY = "memory"
    DECISION = "decision"
    REMEDIATION = "remediation"
    SELF_HEALING = "self_healing"
    ORCHESTRATOR = "orchestrator"
    API = "api"
    WORKER = "worker"
    AUDIT = "audit"
    PROVENANCE = "provenance"
    OBSERVABILITY = "observability"
    DATABASE = "database"
    INTEGRATION = "integration"


class DeploymentEnvironment(str, Enum):
    """Deployment environment associated with an OSHA workload."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class IdentityAuthenticationState(str, Enum):
    """
    Authentication state associated with identity evidence.

    Identity description and authentication are deliberately distinct.

    A WorkloadIdentity object can describe a workload without proving that
    the current caller actually possesses that identity.
    """

    UNVERIFIED = "unverified"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class WorkloadIdentity:
    """
    Immutable OSHA workload identity.

    Canonical identity URI:

        osha://<trust-domain>/<environment>/<type>/<namespace>/<name>

    Example:

        osha://osha.local/production/remediation/system/executor

    This URI identifies the workload but contains no credential or secret.
    """

    trust_domain: str
    environment: DeploymentEnvironment
    workload_type: WorkloadType
    namespace: str
    name: str

    signing_key: KeyReference | None = None
    encryption_key: KeyReference | None = None

    labels: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        trust_domain = _normalize_trust_domain(
            self.trust_domain
        )

        namespace = _normalize_name(
            self.namespace,
            field="namespace",
            max_length=MAX_NAMESPACE_LENGTH,
        )

        name = _normalize_name(
            self.name,
            field="name",
            max_length=MAX_NAME_LENGTH,
        )

        if not isinstance(
            self.environment,
            DeploymentEnvironment,
        ):
            raise InvalidWorkloadIdentityError(
                "environment must be DeploymentEnvironment."
            )

        if not isinstance(
            self.workload_type,
            WorkloadType,
        ):
            raise InvalidWorkloadIdentityError(
                "workload_type must be WorkloadType."
            )

        if (
            self.signing_key is not None
            and not isinstance(
                self.signing_key,
                KeyReference,
            )
        ):
            raise InvalidWorkloadIdentityError(
                "signing_key must be KeyReference or None."
            )

        if (
            self.encryption_key is not None
            and not isinstance(
                self.encryption_key,
                KeyReference,
            )
        ):
            raise InvalidWorkloadIdentityError(
                "encryption_key must be KeyReference or None."
            )

        labels = _normalize_labels(
            self.labels
        )

        object.__setattr__(
            self,
            "trust_domain",
            trust_domain,
        )

        object.__setattr__(
            self,
            "namespace",
            namespace,
        )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "labels",
            labels,
        )

    @property
    def workload_id(self) -> str:
        """Return the canonical OSHA workload URI."""

        return (
            f"osha://{self.trust_domain}/"
            f"{self.environment.value}/"
            f"{self.workload_type.value}/"
            f"{self.namespace}/"
            f"{self.name}"
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Return the canonical JSON-compatible identity representation.
        """

        return {
            "encryption_key": (
                None
                if self.encryption_key is None
                else {
                    "key_id": self.encryption_key.key_id,
                    "version": self.encryption_key.version,
                }
            ),
            "environment": self.environment.value,
            "format": WORKLOAD_IDENTITY_FORMAT,
            "format_version": WORKLOAD_IDENTITY_VERSION,
            "labels": dict(
                sorted(
                    (self.labels or {}).items()
                )
            ),
            "name": self.name,
            "namespace": self.namespace,
            "signing_key": (
                None
                if self.signing_key is None
                else {
                    "key_id": self.signing_key.key_id,
                    "version": self.signing_key.version,
                }
            ),
            "trust_domain": self.trust_domain,
            "workload_id": self.workload_id,
            "workload_type": self.workload_type.value,
        }

    def canonical_bytes(self) -> bytes:
        """
        Return deterministic serialized identity bytes.

        These bytes can later be signed or hashed.
        """

        return canonical_json_bytes(
            self.as_dict()
        )

    def digest(
        self,
        algorithm: HashAlgorithm | str = (
            HashAlgorithm.SHA256
        ),
    ) -> Digest:
        """Return a cryptographic digest of this identity."""

        return hash_json(
            self.as_dict(),
            algorithm=algorithm,
        )

    def matches(
        self,
        other: "WorkloadIdentity",
    ) -> bool:
        """
        Compare workload identity.

        Labels and key references are deliberately not part of identity
        equivalence because they may change during rotation or deployment.

        The stable identity consists of:

        - trust domain
        - environment
        - workload type
        - namespace
        - name
        """

        if not isinstance(
            other,
            WorkloadIdentity,
        ):
            return False

        return (
            self.trust_domain
            == other.trust_domain
            and self.environment
            is other.environment
            and self.workload_type
            is other.workload_type
            and self.namespace
            == other.namespace
            and self.name
            == other.name
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "WorkloadIdentity":
        """Parse and strictly validate an identity representation."""

        if not isinstance(value, Mapping):
            raise InvalidWorkloadIdentityError(
                "Workload identity must be a mapping."
            )

        expected_fields = {
            "encryption_key",
            "environment",
            "format",
            "format_version",
            "labels",
            "name",
            "namespace",
            "signing_key",
            "trust_domain",
            "workload_id",
            "workload_type",
        }

        actual_fields = set(value.keys())

        missing = expected_fields - actual_fields
        unknown = actual_fields - expected_fields

        if missing:
            raise InvalidWorkloadIdentityError(
                "Workload identity is missing fields: "
                + ", ".join(sorted(missing))
                + "."
            )

        if unknown:
            raise InvalidWorkloadIdentityError(
                "Workload identity contains unknown fields: "
                + ", ".join(
                    sorted(str(item) for item in unknown)
                )
                + "."
            )

        if value["format"] != WORKLOAD_IDENTITY_FORMAT:
            raise InvalidWorkloadIdentityError(
                f"Unsupported workload identity format: "
                f"{value['format']!r}."
            )

        if (
            value["format_version"]
            != WORKLOAD_IDENTITY_VERSION
        ):
            raise InvalidWorkloadIdentityError(
                f"Unsupported workload identity version: "
                f"{value['format_version']!r}."
            )

        try:
            environment = DeploymentEnvironment(
                value["environment"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidWorkloadIdentityError(
                f"Unsupported environment: "
                f"{value['environment']!r}."
            ) from exc

        try:
            workload_type = WorkloadType(
                value["workload_type"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidWorkloadIdentityError(
                f"Unsupported workload type: "
                f"{value['workload_type']!r}."
            ) from exc

        signing_key = _parse_key_reference(
            value["signing_key"],
            field="signing_key",
        )

        encryption_key = _parse_key_reference(
            value["encryption_key"],
            field="encryption_key",
        )

        identity = cls(
            trust_domain=value["trust_domain"],
            environment=environment,
            workload_type=workload_type,
            namespace=value["namespace"],
            name=value["name"],
            signing_key=signing_key,
            encryption_key=encryption_key,
            labels=value["labels"],
        )

        supplied_workload_id = value[
            "workload_id"
        ]

        if not isinstance(
            supplied_workload_id,
            str,
        ):
            raise InvalidWorkloadIdentityError(
                "workload_id must be a string."
            )

        if supplied_workload_id != (
            identity.workload_id
        ):
            raise InvalidWorkloadIdentityError(
                "workload_id does not match the "
                "canonical identity fields."
            )

        return identity


@dataclass(frozen=True, slots=True)
class AuthenticatedWorkload:
    """
    Workload identity plus authentication state.

    Do not create VERIFIED instances merely because an identity string was
    provided by a caller.

    VERIFIED should only be assigned after a trusted authentication mechanism
    validates cryptographic identity evidence.
    """

    identity: WorkloadIdentity
    state: IdentityAuthenticationState
    authenticated_at_ns: int | None = None
    authentication_method: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.identity,
            WorkloadIdentity,
        ):
            raise InvalidWorkloadIdentityError(
                "identity must be WorkloadIdentity."
            )

        if not isinstance(
            self.state,
            IdentityAuthenticationState,
        ):
            raise InvalidWorkloadIdentityError(
                "state must be IdentityAuthenticationState."
            )

        if (
            self.state
            is IdentityAuthenticationState.VERIFIED
        ):
            if (
                isinstance(
                    self.authenticated_at_ns,
                    bool,
                )
                or not isinstance(
                    self.authenticated_at_ns,
                    int,
                )
                or self.authenticated_at_ns <= 0
            ):
                raise InvalidWorkloadIdentityError(
                    "Verified identities require a positive "
                    "authenticated_at_ns timestamp."
                )

            if (
                not isinstance(
                    self.authentication_method,
                    str,
                )
                or not self.authentication_method
            ):
                raise InvalidWorkloadIdentityError(
                    "Verified identities require an "
                    "authentication_method."
                )

        else:
            if self.authenticated_at_ns is not None:
                raise InvalidWorkloadIdentityError(
                    "Unverified identities cannot have an "
                    "authentication timestamp."
                )

            if self.authentication_method is not None:
                raise InvalidWorkloadIdentityError(
                    "Unverified identities cannot claim an "
                    "authentication method."
                )

    @property
    def verified(self) -> bool:
        return (
            self.state
            is IdentityAuthenticationState.VERIFIED
        )

    def require_verified(
        self,
    ) -> WorkloadIdentity:
        """Return identity only if authentication was verified."""

        if not self.verified:
            raise WorkloadIdentityError(
                f"Workload {self.identity.workload_id!r} "
                "has not been authenticated."
            )

        return self.identity

    @classmethod
    def unverified(
        cls,
        identity: WorkloadIdentity,
    ) -> "AuthenticatedWorkload":
        """Create an explicitly unverified workload."""

        return cls(
            identity=identity,
            state=(
                IdentityAuthenticationState.UNVERIFIED
            ),
        )

    @classmethod
    def verified_identity(
        cls,
        identity: WorkloadIdentity,
        *,
        authentication_method: str,
        authenticated_at_ns: int | None = None,
    ) -> "AuthenticatedWorkload":
        """
        Construct verified identity after external authentication.

        This method does not itself perform authentication.
        """

        return cls(
            identity=identity,
            state=IdentityAuthenticationState.VERIFIED,
            authenticated_at_ns=(
                time.time_ns()
                if authenticated_at_ns is None
                else authenticated_at_ns
            ),
            authentication_method=authentication_method,
        )


def parse_workload_id(
    workload_id: str,
) -> tuple[
    str,
    DeploymentEnvironment,
    WorkloadType,
    str,
    str,
]:
    """
    Parse an OSHA workload URI.

    Returns:

        trust_domain,
        environment,
        workload_type,
        namespace,
        name

    Parsing an identity URI does NOT authenticate it.
    """

    if not isinstance(workload_id, str):
        raise InvalidWorkloadIdentityError(
            "workload_id must be a string."
        )

    if len(workload_id) > MAX_ID_LENGTH:
        raise InvalidWorkloadIdentityError(
            "workload_id exceeds maximum length."
        )

    parsed = urlparse(workload_id)

    if parsed.scheme != "osha":
        raise InvalidWorkloadIdentityError(
            "Workload ID must use the osha URI scheme."
        )

    if not parsed.netloc:
        raise InvalidWorkloadIdentityError(
            "Workload ID does not contain a trust domain."
        )

    if (
        parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise InvalidWorkloadIdentityError(
            "Workload ID cannot contain parameters, "
            "query strings, or fragments."
        )

    components = [
        component
        for component in parsed.path.split("/")
        if component
    ]

    if len(components) != 4:
        raise InvalidWorkloadIdentityError(
            "Workload ID path must contain exactly "
            "environment/type/namespace/name."
        )

    environment_text = components[0]
    workload_type_text = components[1]
    namespace = components[2]
    name = components[3]

    try:
        environment = DeploymentEnvironment(
            environment_text
        )

    except ValueError as exc:
        raise InvalidWorkloadIdentityError(
            f"Unsupported environment: "
            f"{environment_text!r}."
        ) from exc

    try:
        workload_type = WorkloadType(
            workload_type_text
        )

    except ValueError as exc:
        raise InvalidWorkloadIdentityError(
            f"Unsupported workload type: "
            f"{workload_type_text!r}."
        ) from exc

    trust_domain = _normalize_trust_domain(
        parsed.netloc
    )

    namespace = _normalize_name(
        namespace,
        field="namespace",
        max_length=MAX_NAMESPACE_LENGTH,
    )

    name = _normalize_name(
        name,
        field="name",
        max_length=MAX_NAME_LENGTH,
    )

    return (
        trust_domain,
        environment,
        workload_type,
        namespace,
        name,
    )


def workload_identity_from_id(
    workload_id: str,
    *,
    signing_key: KeyReference | None = None,
    encryption_key: KeyReference | None = None,
    labels: Mapping[str, str] | None = None,
) -> WorkloadIdentity:
    """
    Construct WorkloadIdentity from its canonical URI.

    This operation parses identity information only.
    It does NOT authenticate the workload.
    """

    (
        trust_domain,
        environment,
        workload_type,
        namespace,
        name,
    ) = parse_workload_id(
        workload_id
    )

    return WorkloadIdentity(
        trust_domain=trust_domain,
        environment=environment,
        workload_type=workload_type,
        namespace=namespace,
        name=name,
        signing_key=signing_key,
        encryption_key=encryption_key,
        labels=labels,
    )


def _parse_key_reference(
    value: Any,
    *,
    field: str,
) -> KeyReference | None:

    if value is None:
        return None

    if not isinstance(value, Mapping):
        raise InvalidWorkloadIdentityError(
            f"{field} must be a mapping or None."
        )

    expected = {
        "key_id",
        "version",
    }

    actual = set(value.keys())

    if actual != expected:
        raise InvalidWorkloadIdentityError(
            f"{field} must contain exactly "
            "'key_id' and 'version'."
        )

    try:
        return KeyReference(
            key_id=value["key_id"],
            version=value["version"],
        )

    except Exception as exc:
        raise InvalidWorkloadIdentityError(
            f"{field} contains an invalid key reference."
        ) from exc


def _normalize_trust_domain(
    value: str,
) -> str:

    if not isinstance(value, str):
        raise InvalidWorkloadIdentityError(
            "trust_domain must be a string."
        )

    normalized = value.strip().lower()

    if not normalized:
        raise InvalidWorkloadIdentityError(
            "trust_domain cannot be empty."
        )

    if len(normalized) > (
        MAX_TRUST_DOMAIN_LENGTH
    ):
        raise InvalidWorkloadIdentityError(
            "trust_domain exceeds maximum length."
        )

    if (
        _TRUST_DOMAIN_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise InvalidWorkloadIdentityError(
            "trust_domain contains invalid characters."
        )

    if ".." in normalized:
        raise InvalidWorkloadIdentityError(
            "trust_domain cannot contain empty labels."
        )

    return normalized


def _normalize_name(
    value: str,
    *,
    field: str,
    max_length: int,
) -> str:

    if not isinstance(value, str):
        raise InvalidWorkloadIdentityError(
            f"{field} must be a string."
        )

    normalized = value.strip().lower()

    if not normalized:
        raise InvalidWorkloadIdentityError(
            f"{field} cannot be empty."
        )

    if len(normalized) > max_length:
        raise InvalidWorkloadIdentityError(
            f"{field} exceeds maximum length "
            f"of {max_length}."
        )

    if (
        _NAME_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise InvalidWorkloadIdentityError(
            f"{field} contains invalid characters."
        )

    return normalized


def _normalize_labels(
    labels: Mapping[str, str] | None,
) -> Mapping[str, str]:

    if labels is None:
        return {}

    if not isinstance(labels, Mapping):
        raise InvalidWorkloadIdentityError(
            "labels must be a mapping or None."
        )

    if len(labels) > MAX_LABELS:
        raise InvalidWorkloadIdentityError(
            f"Identity cannot contain more than "
            f"{MAX_LABELS} labels."
        )

    normalized: dict[str, str] = {}

    for key, value in labels.items():
        normalized_key = _normalize_name(
            key,
            field="label key",
            max_length=MAX_LABEL_KEY_LENGTH,
        )

        if not isinstance(value, str):
            raise InvalidWorkloadIdentityError(
                "Label values must be strings."
            )

        normalized_value = value.strip()

        if len(normalized_value) > (
            MAX_LABEL_VALUE_LENGTH
        ):
            raise InvalidWorkloadIdentityError(
                "Label value exceeds maximum length."
            )

        normalized[
            normalized_key
        ] = normalized_value

    return normalized


__all__ = [
    "AuthenticatedWorkload",
    "DEFAULT_TRUST_DOMAIN",
    "DeploymentEnvironment",
    "IdentityAuthenticationState",
    "InvalidWorkloadIdentityError",
    "WORKLOAD_IDENTITY_FORMAT",
    "WORKLOAD_IDENTITY_VERSION",
    "WorkloadIdentity",
    "WorkloadIdentityError",
    "WorkloadIdentityMismatchError",
    "WorkloadType",
    "parse_workload_id",
    "workload_identity_from_id",
]
