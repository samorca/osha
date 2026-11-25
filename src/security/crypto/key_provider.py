"""
OSHA cryptographic key-provider abstraction.

This module defines how OSHA identifies, retrieves, generates, registers,
rotates, disables, and destroys cryptographic key material.

IMPORTANT
---------
This module does NOT implement encryption or digital signatures.

Its responsibility is key management.

Higher-level modules such as:

    encryption.py
    signatures.py
    model_signing.py
    dataset_signing.py
    signed_events.py

must obtain cryptographic keys through the interfaces defined here instead
of reading secrets directly from files, environment variables, databases,
or source code.

Production deployments should provide an implementation backed by an
appropriate secret-management/KMS/HSM system.

The in-memory provider included here is intended for development, testing,
and isolated local operation. It must not be treated as persistent
production key storage.
"""

from __future__ import annotations

import secrets
import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


DEFAULT_SYMMETRIC_KEY_SIZE: Final[int] = 32
MINIMUM_SYMMETRIC_KEY_SIZE: Final[int] = 32

MAX_KEY_ID_LENGTH: Final[int] = 128
MAX_VERSION_LENGTH: Final[int] = 64


class KeyProviderError(Exception):
    """Base exception for OSHA key-provider failures."""


class KeyNotFoundError(KeyProviderError, KeyError):
    """Requested key does not exist."""


class KeyAlreadyExistsError(KeyProviderError):
    """A key with the same identity already exists."""


class KeyDisabledError(KeyProviderError):
    """Requested key exists but is disabled."""


class KeyDestroyedError(KeyProviderError):
    """Requested key has been destroyed."""


class InvalidKeyError(KeyProviderError, ValueError):
    """Provided key material or metadata is invalid."""


class KeyPolicyError(KeyProviderError):
    """Requested key operation violates key policy."""


class KeyPurpose(str, Enum):
    """
    Approved high-level purposes for OSHA keys.

    Keeping purposes explicit helps prevent accidental reuse of one
    cryptographic key for unrelated operations.
    """

    ENCRYPTION = "encryption"
    SIGNING = "signing"
    AUTHENTICATION = "authentication"
    KEY_ENCRYPTION = "key_encryption"
    AUDIT = "audit"
    REMEDIATION = "remediation"
    DATASET_SIGNING = "dataset_signing"
    MODEL_SIGNING = "model_signing"


class KeyState(str, Enum):
    """Lifecycle state of a cryptographic key."""

    ACTIVE = "active"
    DISABLED = "disabled"
    DESTROYED = "destroyed"


class KeyType(str, Enum):
    """
    Logical key type.

    SYMMETRIC contains secret bytes.

    Future asymmetric KMS/HSM implementations can expose public/private
    signing keys without requiring callers to change the provider API.
    """

    SYMMETRIC = "symmetric"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    ASYMMETRIC_PUBLIC = "asymmetric_public"


@dataclass(frozen=True, slots=True)
class KeyReference:
    """
    Non-secret identifier for a cryptographic key.

    A KeyReference is safe to place in manifests and audit events because
    it identifies a key but does not contain the key material itself.
    """

    key_id: str
    version: str

    def __post_init__(self) -> None:
        _validate_identifier(
            self.key_id,
            field="key_id",
            max_length=MAX_KEY_ID_LENGTH,
        )

        _validate_identifier(
            self.version,
            field="version",
            max_length=MAX_VERSION_LENGTH,
        )

    def __str__(self) -> str:
        return f"{self.key_id}:{self.version}"


@dataclass(frozen=True, slots=True)
class KeyMetadata:
    """
    Non-secret metadata describing a cryptographic key.
    """

    reference: KeyReference
    purpose: KeyPurpose
    key_type: KeyType
    state: KeyState
    created_at_ns: int
    rotated_from: KeyReference | None = None
    labels: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.created_at_ns <= 0:
            raise InvalidKeyError(
                "created_at_ns must be a positive integer."
            )

        if self.labels is None:
            object.__setattr__(
                self,
                "labels",
                MappingProxyType({}),
            )
            return

        normalized: dict[str, str] = {}

        for key, value in self.labels.items():
            if not isinstance(key, str) or not key:
                raise InvalidKeyError(
                    "Key metadata label names must be non-empty strings."
                )

            if not isinstance(value, str):
                raise InvalidKeyError(
                    "Key metadata label values must be strings."
                )

            normalized[key] = value

        object.__setattr__(
            self,
            "labels",
            MappingProxyType(normalized),
        )


@dataclass(frozen=True, slots=True, repr=False)
class KeyMaterial:
    """
    Secret cryptographic key bytes plus non-secret metadata.

    repr() intentionally hides the key bytes.

    Python cannot guarantee immediate memory zeroization for immutable
    bytes objects. Callers should therefore keep KeyMaterial objects
    short-lived and never log or persist their secret field.
    """

    metadata: KeyMetadata
    secret: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.secret, bytes):
            raise InvalidKeyError(
                "Key secret must be immutable bytes."
            )

        if not self.secret:
            raise InvalidKeyError(
                "Key secret cannot be empty."
            )

    def __repr__(self) -> str:
        return (
            "KeyMaterial("
            f"reference={self.metadata.reference!s}, "
            f"purpose={self.metadata.purpose.value!r}, "
            f"key_type={self.metadata.key_type.value!r}, "
            f"state={self.metadata.state.value!r}, "
            "secret=<redacted>"
            ")"
        )


class KeyProvider(ABC):
    """
    Abstract cryptographic key provider.

    Production providers may use:

    - a dedicated KMS
    - an HSM
    - a secrets-management service
    - another approved protected key store

    Application code should depend on this interface rather than a
    provider-specific SDK.
    """

    @abstractmethod
    def get_key(
        self,
        reference: KeyReference,
        *,
        expected_purpose: KeyPurpose | None = None,
    ) -> KeyMaterial:
        """
        Retrieve active key material.

        Implementations must fail if the key is disabled or destroyed.
        """

    @abstractmethod
    def get_metadata(
        self,
        reference: KeyReference,
    ) -> KeyMetadata:
        """Retrieve non-secret key metadata."""

    @abstractmethod
    def generate_symmetric_key(
        self,
        *,
        key_id: str,
        purpose: KeyPurpose,
        size_bytes: int = DEFAULT_SYMMETRIC_KEY_SIZE,
        labels: Mapping[str, str] | None = None,
    ) -> KeyReference:
        """Generate and register a new symmetric key."""

    @abstractmethod
    def rotate_symmetric_key(
        self,
        reference: KeyReference,
    ) -> KeyReference:
        """
        Create a new version of an existing symmetric key.

        The previous version remains identifiable for controlled access
        to historical protected data.
        """

    @abstractmethod
    def disable_key(
        self,
        reference: KeyReference,
    ) -> None:
        """Prevent new use of a key."""

    @abstractmethod
    def destroy_key(
        self,
        reference: KeyReference,
    ) -> None:
        """
        Mark a key as destroyed and remove accessible secret material.

        External KMS/HSM implementations should map this operation to
        their approved destruction workflow.
        """


class InMemoryKeyProvider(KeyProvider):
    """
    Thread-safe volatile key provider.

    SECURITY BOUNDARY
    -----------------
    This implementation exists for:

    - unit testing
    - integration testing
    - local development
    - ephemeral development environments

    It is NOT persistent production key storage.

    Secret key material exists in process memory and disappears when the
    process terminates.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._keys: dict[
            KeyReference,
            KeyMaterial,
        ] = {}

        self._latest_versions: dict[str, int] = {}

    def get_key(
        self,
        reference: KeyReference,
        *,
        expected_purpose: KeyPurpose | None = None,
    ) -> KeyMaterial:

        with self._lock:
            material = self._lookup(reference)

            state = material.metadata.state

            if state is KeyState.DESTROYED:
                raise KeyDestroyedError(
                    f"Key {reference} has been destroyed."
                )

            if state is KeyState.DISABLED:
                raise KeyDisabledError(
                    f"Key {reference} is disabled."
                )

            if expected_purpose is not None:
                if material.metadata.purpose is not expected_purpose:
                    raise KeyPolicyError(
                        f"Key {reference} has purpose "
                        f"{material.metadata.purpose.value!r}, "
                        f"not {expected_purpose.value!r}."
                    )

            return material

    def get_metadata(
        self,
        reference: KeyReference,
    ) -> KeyMetadata:

        with self._lock:
            return self._lookup(reference).metadata

    def generate_symmetric_key(
        self,
        *,
        key_id: str,
        purpose: KeyPurpose,
        size_bytes: int = DEFAULT_SYMMETRIC_KEY_SIZE,
        labels: Mapping[str, str] | None = None,
    ) -> KeyReference:

        _validate_identifier(
            key_id,
            field="key_id",
            max_length=MAX_KEY_ID_LENGTH,
        )

        if not isinstance(purpose, KeyPurpose):
            raise InvalidKeyError(
                "purpose must be a KeyPurpose."
            )

        _validate_key_size(size_bytes)

        with self._lock:
            version_number = (
                self._latest_versions.get(key_id, 0) + 1
            )

            reference = KeyReference(
                key_id=key_id,
                version=str(version_number),
            )

            if reference in self._keys:
                raise KeyAlreadyExistsError(
                    f"Key {reference} already exists."
                )

            secret = secrets.token_bytes(size_bytes)

            metadata = KeyMetadata(
                reference=reference,
                purpose=purpose,
                key_type=KeyType.SYMMETRIC,
                state=KeyState.ACTIVE,
                created_at_ns=time.time_ns(),
                labels=labels,
            )

            self._keys[reference] = KeyMaterial(
                metadata=metadata,
                secret=secret,
            )

            self._latest_versions[key_id] = version_number

            return reference

    def rotate_symmetric_key(
        self,
        reference: KeyReference,
    ) -> KeyReference:

        with self._lock:
            current = self._lookup(reference)

            if current.metadata.state is KeyState.DESTROYED:
                raise KeyDestroyedError(
                    f"Cannot rotate destroyed key {reference}."
                )

            if current.metadata.key_type is not KeyType.SYMMETRIC:
                raise KeyPolicyError(
                    "Only symmetric keys can be rotated by "
                    "rotate_symmetric_key()."
                )

            version_number = (
                self._latest_versions.get(
                    reference.key_id,
                    0,
                )
                + 1
            )

            new_reference = KeyReference(
                key_id=reference.key_id,
                version=str(version_number),
            )

            new_secret = secrets.token_bytes(
                len(current.secret)
            )

            metadata = KeyMetadata(
                reference=new_reference,
                purpose=current.metadata.purpose,
                key_type=KeyType.SYMMETRIC,
                state=KeyState.ACTIVE,
                created_at_ns=time.time_ns(),
                rotated_from=reference,
                labels=current.metadata.labels,
            )

            self._keys[new_reference] = KeyMaterial(
                metadata=metadata,
                secret=new_secret,
            )

            self._latest_versions[
                reference.key_id
            ] = version_number

            return new_reference

    def disable_key(
        self,
        reference: KeyReference,
    ) -> None:

        with self._lock:
            current = self._lookup(reference)

            if current.metadata.state is KeyState.DESTROYED:
                raise KeyDestroyedError(
                    f"Key {reference} is already destroyed."
                )

            if current.metadata.state is KeyState.DISABLED:
                return

            metadata = replace(
                current.metadata,
                state=KeyState.DISABLED,
            )

            self._keys[reference] = KeyMaterial(
                metadata=metadata,
                secret=current.secret,
            )

    def destroy_key(
        self,
        reference: KeyReference,
    ) -> None:

        with self._lock:
            current = self._lookup(reference)

            if current.metadata.state is KeyState.DESTROYED:
                return

            metadata = replace(
                current.metadata,
                state=KeyState.DESTROYED,
            )

            # The original immutable bytes object cannot be guaranteed to
            # be zeroized by Python. We remove our provider's accessible
            # reference to the secret and replace it with an internal
            # sentinel byte solely so KeyMaterial remains structurally
            # valid. get_key() rejects DESTROYED entries before returning.
            self._keys[reference] = KeyMaterial(
                metadata=metadata,
                secret=b"\x00",
            )

    def _lookup(
        self,
        reference: KeyReference,
    ) -> KeyMaterial:

        if not isinstance(reference, KeyReference):
            raise InvalidKeyError(
                "reference must be a KeyReference."
            )

        try:
            return self._keys[reference]

        except KeyError as exc:
            raise KeyNotFoundError(
                f"Key {reference} does not exist."
            ) from exc


def _validate_identifier(
    value: str,
    *,
    field: str,
    max_length: int,
) -> None:

    if not isinstance(value, str):
        raise InvalidKeyError(
            f"{field} must be a string."
        )

    if not value:
        raise InvalidKeyError(
            f"{field} cannot be empty."
        )

    if len(value) > max_length:
        raise InvalidKeyError(
            f"{field} exceeds maximum length "
            f"of {max_length} characters."
        )

    if value != value.strip():
        raise InvalidKeyError(
            f"{field} cannot start or end with whitespace."
        )

    for character in value:
        if not (
            character.isalnum()
            or character in "-_."
        ):
            raise InvalidKeyError(
                f"{field} contains prohibited character "
                f"{character!r}."
            )


def _validate_key_size(
    size_bytes: int,
) -> None:

    if isinstance(size_bytes, bool) or not isinstance(
        size_bytes,
        int,
    ):
        raise InvalidKeyError(
            "Key size must be an integer."
        )

    if size_bytes < MINIMUM_SYMMETRIC_KEY_SIZE:
        raise InvalidKeyError(
            "Symmetric keys must contain at least "
            f"{MINIMUM_SYMMETRIC_KEY_SIZE} bytes."
        )


__all__ = [
    "DEFAULT_SYMMETRIC_KEY_SIZE",
    "InMemoryKeyProvider",
    "InvalidKeyError",
    "KeyAlreadyExistsError",
    "KeyDestroyedError",
    "KeyDisabledError",
    "KeyMaterial",
    "KeyMetadata",
    "KeyNotFoundError",
    "KeyPolicyError",
    "KeyProvider",
    "KeyProviderError",
    "KeyPurpose",
    "KeyReference",
    "KeyState",
    "KeyType",
]
