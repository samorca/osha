"""
X.509 workload certificate support for OSHA.

This module binds OSHA WorkloadIdentity objects to X.509 certificates.

Responsibilities
----------------
- Generate development/test certificate authorities.
- Issue short-lived workload certificates.
- Encode the OSHA workload ID as a URI Subject Alternative Name (SAN).
- Parse identity from certificates.
- Validate certificate lifetime.
- Validate certificate identity binding.
- Validate direct certificate signatures against a trusted CA certificate.
- Validate expected workload identity.
- Verify that a private key matches a workload certificate.

Important boundaries
--------------------
This is an OSHA-native X.509 profile, not a claim of SPIFFE compliance.

Production deployments should normally use an approved PKI/workload identity
system rather than application-managed CA private keys.

CA private keys created by this module are intended for development/testing
unless an external production key-management design explicitly approves
otherwise.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import secrets

from dataclasses import dataclass
from enum import Enum
from typing import Final, Iterable

try:
    from cryptography import x509
    from cryptography.exceptions import (
        InvalidSignature,
        UnsupportedAlgorithm,
    )
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import (
        ec,
        ed25519,
        ed448,
        padding,
        rsa,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    from cryptography.x509.oid import (
        ExtendedKeyUsageOID,
        NameOID,
    )
except ImportError as exc:
    raise RuntimeError(
        "The 'cryptography' package is required by "
        "src.security.identity.certificates."
    ) from exc

from .workload_identity import (
    AuthenticatedWorkload,
    IdentityAuthenticationState,
    InvalidWorkloadIdentityError,
    WorkloadIdentity,
    workload_identity_from_id,
)


DEFAULT_CA_VALIDITY_DAYS: Final[int] = 30
DEFAULT_WORKLOAD_VALIDITY_MINUTES: Final[int] = 60
DEFAULT_CLOCK_SKEW_SECONDS: Final[int] = 60

MAX_CA_VALIDITY_DAYS: Final[int] = 365
MAX_WORKLOAD_VALIDITY_MINUTES: Final[int] = 24 * 60

OSHA_CERTIFICATE_PROFILE: Final[str] = "osha-x509-workload-v1"


class CertificateError(Exception):
    """Base exception for OSHA certificate failures."""


class CertificateIssuanceError(CertificateError):
    """Certificate generation or issuance failed."""


class CertificateValidationError(CertificateError):
    """Certificate validation failed."""


class CertificateExpiredError(CertificateValidationError):
    """Certificate has expired."""


class CertificateNotYetValidError(CertificateValidationError):
    """Certificate is not valid yet."""


class CertificateIdentityError(CertificateValidationError):
    """Certificate does not contain the expected workload identity."""


class CertificateTrustError(CertificateValidationError):
    """Certificate could not be validated against the trusted issuer."""


class CertificateKeyMismatchError(CertificateValidationError):
    """Private key does not match certificate public key."""


class CertificateType(str, Enum):
    """OSHA certificate classes."""

    ROOT_CA = "root_ca"
    WORKLOAD = "workload"


@dataclass(frozen=True, slots=True)
class CertificateAuthority:
    """
    Development/test certificate authority.

    private_key contains secret material and must never be logged or persisted
    without appropriate cryptographic protection.

    Production deployments should replace this object with an approved CA/KMS
    integration.
    """

    certificate: x509.Certificate
    private_key: object

    def certificate_pem(self) -> bytes:
        return self.certificate.public_bytes(
            Encoding.PEM
        )

    def private_key_pem(self) -> bytes:
        """
        Export an unencrypted private key.

        DEVELOPMENT/TEST ONLY.

        Production CA private keys should not normally be exported from their
        secure key-management boundary.
        """

        return self.private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )


@dataclass(frozen=True, slots=True)
class WorkloadCertificate:
    """Issued workload certificate and its corresponding private key."""

    identity: WorkloadIdentity
    certificate: x509.Certificate
    private_key: object

    def certificate_pem(self) -> bytes:
        return self.certificate.public_bytes(
            Encoding.PEM
        )

    def private_key_pem(self) -> bytes:
        """
        Export unencrypted workload private-key bytes.

        Avoid this in production unless an external integration explicitly
        requires export and protects the result.
        """

        return self.private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )


@dataclass(frozen=True, slots=True)
class ValidatedCertificate:
    """Successfully validated workload certificate."""

    certificate: x509.Certificate
    identity: WorkloadIdentity
    authenticated_workload: AuthenticatedWorkload
    issuer_certificate: x509.Certificate

    @property
    def serial_number(self) -> int:
        return self.certificate.serial_number


def create_development_ca(
    *,
    common_name: str = "OSHA Development Root CA",
    validity_days: int = DEFAULT_CA_VALIDITY_DAYS,
) -> CertificateAuthority:
    """
    Create an Ed25519 self-signed CA.

    This helper is intended for development and testing.

    Do not use an application-generated development CA as the production
    OSHA trust root.
    """

    _validate_nonempty_string(
        common_name,
        field="common_name",
    )

    if (
        isinstance(validity_days, bool)
        or not isinstance(validity_days, int)
        or not 1 <= validity_days <= MAX_CA_VALIDITY_DAYS
    ):
        raise CertificateIssuanceError(
            f"validity_days must be between 1 and "
            f"{MAX_CA_VALIDITY_DAYS}."
        )

    private_key = ed25519.Ed25519PrivateKey.generate()

    now = _utc_now()

    subject = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                common_name,
            )
        ]
    )

    try:
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(
                now - dt.timedelta(
                    seconds=DEFAULT_CLOCK_SKEW_SECONDS
                )
            )
            .not_valid_after(
                now + dt.timedelta(
                    days=validity_days
                )
            )
            .add_extension(
                x509.BasicConstraints(
                    ca=True,
                    path_length=0,
                ),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(
                    private_key.public_key()
                ),
                critical=False,
            )
            .sign(
                private_key=private_key,
                algorithm=None,
            )
        )
    except Exception as exc:
        raise CertificateIssuanceError(
            "Unable to create development certificate authority."
        ) from exc

    return CertificateAuthority(
        certificate=certificate,
        private_key=private_key,
    )


def issue_workload_certificate(
    identity: WorkloadIdentity,
    issuer: CertificateAuthority,
    *,
    validity_minutes: int = DEFAULT_WORKLOAD_VALIDITY_MINUTES,
    dns_names: Iterable[str] | None = None,
    ip_addresses: Iterable[str] | None = None,
) -> WorkloadCertificate:
    """
    Issue a short-lived Ed25519 workload certificate.

    The canonical OSHA workload ID is encoded as exactly one URI SAN.

    Optional DNS/IP SANs may also be included for TLS endpoint validation.
    """

    if not isinstance(identity, WorkloadIdentity):
        raise CertificateIssuanceError(
            "identity must be WorkloadIdentity."
        )

    if not isinstance(issuer, CertificateAuthority):
        raise CertificateIssuanceError(
            "issuer must be CertificateAuthority."
        )

    if (
        isinstance(validity_minutes, bool)
        or not isinstance(validity_minutes, int)
        or not 1 <= validity_minutes <= (
            MAX_WORKLOAD_VALIDITY_MINUTES
        )
    ):
        raise CertificateIssuanceError(
            f"validity_minutes must be between 1 and "
            f"{MAX_WORKLOAD_VALIDITY_MINUTES}."
        )

    _validate_ca_certificate(
        issuer.certificate
    )

    workload_private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    now = _utc_now()

    subject = x509.Name(
        [
            x509.NameAttribute(
                NameOID.ORGANIZATION_NAME,
                "OSHA",
            ),
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                identity.name,
            ),
        ]
    )

    san_entries: list[x509.GeneralName] = [
        x509.UniformResourceIdentifier(
            identity.workload_id
        )
    ]

    if dns_names is not None:
        for name in dns_names:
            _validate_nonempty_string(
                name,
                field="dns_name",
            )

            san_entries.append(
                x509.DNSName(name)
            )

    if ip_addresses is not None:
        for address in ip_addresses:
            try:
                parsed_ip = ipaddress.ip_address(
                    address
                )
            except ValueError as exc:
                raise CertificateIssuanceError(
                    f"Invalid IP address: {address!r}."
                ) from exc

            san_entries.append(
                x509.IPAddress(parsed_ip)
            )

    try:
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(
                issuer.certificate.subject
            )
            .public_key(
                workload_private_key.public_key()
            )
            .serial_number(
                x509.random_serial_number()
            )
            .not_valid_before(
                now - dt.timedelta(
                    seconds=DEFAULT_CLOCK_SKEW_SECONDS
                )
            )
            .not_valid_after(
                now + dt.timedelta(
                    minutes=validity_minutes
                )
            )
            .add_extension(
                x509.BasicConstraints(
                    ca=False,
                    path_length=None,
                ),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage(
                    [
                        ExtendedKeyUsageOID.SERVER_AUTH,
                        ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.SubjectAlternativeName(
                    san_entries
                ),
                critical=False,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(
                    workload_private_key.public_key()
                ),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(
                    issuer.private_key.public_key()
                ),
                critical=False,
            )
            .sign(
                private_key=issuer.private_key,
                algorithm=None,
            )
        )
    except Exception as exc:
        raise CertificateIssuanceError(
            "Unable to issue workload certificate."
        ) from exc

    return WorkloadCertificate(
        identity=identity,
        certificate=certificate,
        private_key=workload_private_key,
    )


def validate_workload_certificate(
    certificate: x509.Certificate,
    *,
    trusted_issuer: x509.Certificate,
    expected_identity: WorkloadIdentity | None = None,
    now: dt.datetime | None = None,
) -> ValidatedCertificate:
    """
    Validate an OSHA workload certificate against a directly trusted issuer.

    Checks performed:

    - supported certificate object
    - trusted issuer is a CA
    - issuer/subject relationship
    - certificate signature
    - certificate validity period
    - leaf basic constraints
    - digital-signature key usage
    - client/server authentication EKU
    - exactly one URI SAN
    - URI contains a valid OSHA WorkloadIdentity
    - optional exact expected identity match

    This function validates a direct leaf -> trusted CA relationship. It is
    not a general-purpose arbitrary X.509 path-building engine.
    """

    if not isinstance(
        certificate,
        x509.Certificate,
    ):
        raise CertificateValidationError(
            "certificate must be an X.509 Certificate."
        )

    if not isinstance(
        trusted_issuer,
        x509.Certificate,
    ):
        raise CertificateTrustError(
            "trusted_issuer must be an X.509 Certificate."
        )

    _validate_ca_certificate(
        trusted_issuer
    )

    current_time = (
        _utc_now()
        if now is None
        else _normalize_datetime(now)
    )

    not_before = _certificate_not_before(
        certificate
    )

    not_after = _certificate_not_after(
        certificate
    )

    if current_time < not_before:
        raise CertificateNotYetValidError(
            "Workload certificate is not valid yet."
        )

    if current_time > not_after:
        raise CertificateExpiredError(
            "Workload certificate has expired."
        )

    if certificate.issuer != trusted_issuer.subject:
        raise CertificateTrustError(
            "Certificate issuer does not match the "
            "trusted CA subject."
        )

    _verify_certificate_signature(
        certificate,
        trusted_issuer,
    )

    try:
        basic_constraints = (
            certificate.extensions.get_extension_for_class(
                x509.BasicConstraints
            ).value
        )
    except x509.ExtensionNotFound as exc:
        raise CertificateValidationError(
            "Workload certificate is missing BasicConstraints."
        ) from exc

    if basic_constraints.ca:
        raise CertificateValidationError(
            "Workload certificate cannot be a CA certificate."
        )

    try:
        key_usage = (
            certificate.extensions.get_extension_for_class(
                x509.KeyUsage
            ).value
        )
    except x509.ExtensionNotFound as exc:
        raise CertificateValidationError(
            "Workload certificate is missing KeyUsage."
        ) from exc

    if not key_usage.digital_signature:
        raise CertificateValidationError(
            "Workload certificate does not permit "
            "digital signatures."
        )

    if (
        key_usage.key_cert_sign
        or key_usage.crl_sign
    ):
        raise CertificateValidationError(
            "Workload certificate contains CA key usages."
        )

    try:
        extended_usage = (
            certificate.extensions.get_extension_for_class(
                x509.ExtendedKeyUsage
            ).value
        )
    except x509.ExtensionNotFound as exc:
        raise CertificateValidationError(
            "Workload certificate is missing ExtendedKeyUsage."
        ) from exc

    if (
        ExtendedKeyUsageOID.CLIENT_AUTH
        not in extended_usage
        or ExtendedKeyUsageOID.SERVER_AUTH
        not in extended_usage
    ):
        raise CertificateValidationError(
            "Workload certificate must support both "
            "client and server authentication."
        )

    identity = identity_from_certificate(
        certificate
    )

    if (
        expected_identity is not None
        and not identity.matches(
            expected_identity
        )
    ):
        raise CertificateIdentityError(
            "Certificate workload identity does not match "
            "the expected workload."
        )

    authenticated = (
        AuthenticatedWorkload.verified_identity(
            identity,
            authentication_method=OSHA_CERTIFICATE_PROFILE,
        )
    )

    if (
        authenticated.state
        is not IdentityAuthenticationState.VERIFIED
    ):
        raise CertificateValidationError(
            "Certificate validation did not produce "
            "a verified workload."
        )

    return ValidatedCertificate(
        certificate=certificate,
        identity=identity,
        authenticated_workload=authenticated,
        issuer_certificate=trusted_issuer,
    )


def identity_from_certificate(
    certificate: x509.Certificate,
) -> WorkloadIdentity:
    """
    Extract the OSHA workload identity from the certificate URI SAN.

    Exactly one URI SAN is required.
    """

    if not isinstance(
        certificate,
        x509.Certificate,
    ):
        raise CertificateIdentityError(
            "certificate must be an X.509 Certificate."
        )

    try:
        san = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
    except x509.ExtensionNotFound as exc:
        raise CertificateIdentityError(
            "Certificate does not contain SubjectAlternativeName."
        ) from exc

    uri_names = san.get_values_for_type(
        x509.UniformResourceIdentifier
    )

    if len(uri_names) != 1:
        raise CertificateIdentityError(
            "Workload certificate must contain exactly "
            "one URI SAN."
        )

    workload_id = uri_names[0]

    try:
        return workload_identity_from_id(
            workload_id
        )
    except InvalidWorkloadIdentityError as exc:
        raise CertificateIdentityError(
            "Certificate URI SAN does not contain a valid "
            "OSHA workload identity."
        ) from exc


def verify_private_key_matches_certificate(
    private_key: object,
    certificate: x509.Certificate,
) -> None:
    """
    Require a private key to match a certificate public key.

    No private-key bytes are logged or returned.
    """

    if not isinstance(
        certificate,
        x509.Certificate,
    ):
        raise CertificateValidationError(
            "certificate must be an X.509 Certificate."
        )

    if not hasattr(
        private_key,
        "public_key",
    ):
        raise CertificateKeyMismatchError(
            "private_key does not expose a public key."
        )

    try:
        private_public_bytes = (
            private_key.public_key().public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo,
            )
        )

        certificate_public_bytes = (
            certificate.public_key().public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo,
            )
        )
    except Exception as exc:
        raise CertificateKeyMismatchError(
            "Unable to compare certificate and private key."
        ) from exc

    if private_public_bytes != certificate_public_bytes:
        raise CertificateKeyMismatchError(
            "Private key does not match certificate public key."
        )


def load_certificate_pem(
    data: bytes,
) -> x509.Certificate:
    """Load one PEM encoded X.509 certificate."""

    if not isinstance(
        data,
        bytes,
    ):
        raise CertificateValidationError(
            "Certificate PEM data must be bytes."
        )

    try:
        return x509.load_pem_x509_certificate(
            data
        )
    except ValueError as exc:
        raise CertificateValidationError(
            "Invalid PEM X.509 certificate."
        ) from exc


def certificate_fingerprint_sha256(
    certificate: x509.Certificate,
) -> str:
    """Return certificate SHA-256 fingerprint as lowercase hex."""

    if not isinstance(
        certificate,
        x509.Certificate,
    ):
        raise CertificateValidationError(
            "certificate must be an X.509 Certificate."
        )

    return certificate.fingerprint(
        hashes.SHA256()
    ).hex()


def _validate_ca_certificate(
    certificate: x509.Certificate,
) -> None:
    if not isinstance(
        certificate,
        x509.Certificate,
    ):
        raise CertificateTrustError(
            "CA certificate must be an X.509 Certificate."
        )

    try:
        constraints = (
            certificate.extensions.get_extension_for_class(
                x509.BasicConstraints
            ).value
        )
    except x509.ExtensionNotFound as exc:
        raise CertificateTrustError(
            "CA certificate is missing BasicConstraints."
        ) from exc

    if not constraints.ca:
        raise CertificateTrustError(
            "Trusted issuer is not marked as a CA."
        )

    try:
        key_usage = (
            certificate.extensions.get_extension_for_class(
                x509.KeyUsage
            ).value
        )
    except x509.ExtensionNotFound as exc:
        raise CertificateTrustError(
            "CA certificate is missing KeyUsage."
        ) from exc

    if not key_usage.key_cert_sign:
        raise CertificateTrustError(
            "CA certificate does not permit certificate signing."
        )


def _verify_certificate_signature(
    certificate: x509.Certificate,
    issuer: x509.Certificate,
) -> None:
    """
    Verify the certificate signature using the direct issuer public key.

    Supports the common public-key types handled by cryptography.
    """

    public_key = issuer.public_key()

    try:
        if isinstance(
            public_key,
            ed25519.Ed25519PublicKey,
        ):
            public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
            )

        elif isinstance(
            public_key,
            ed448.Ed448PublicKey,
        ):
            public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
            )

        elif isinstance(
            public_key,
            rsa.RSAPublicKey,
        ):
            parameters = (
                certificate.signature_algorithm_parameters
            )

            if parameters is None:
                raise CertificateTrustError(
                    "RSA certificate does not expose "
                    "signature parameters."
                )

            public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                parameters,
                certificate.signature_hash_algorithm,
            )

        elif isinstance(
            public_key,
            ec.EllipticCurvePublicKey,
        ):
            parameters = (
                certificate.signature_algorithm_parameters
            )

            if parameters is None:
                parameters = ec.ECDSA(
                    certificate.signature_hash_algorithm
                )

            public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                parameters,
            )

        else:
            raise CertificateTrustError(
                "Unsupported CA public-key type."
            )

    except InvalidSignature as exc:
        raise CertificateTrustError(
            "Certificate signature verification failed."
        ) from exc

    except UnsupportedAlgorithm as exc:
        raise CertificateTrustError(
            "Certificate uses an unsupported signature algorithm."
        ) from exc


def _certificate_not_before(
    certificate: x509.Certificate,
) -> dt.datetime:
    value = getattr(
        certificate,
        "not_valid_before_utc",
        None,
    )

    if value is not None:
        return value

    return certificate.not_valid_before.replace(
        tzinfo=dt.timezone.utc
    )


def _certificate_not_after(
    certificate: x509.Certificate,
) -> dt.datetime:
    value = getattr(
        certificate,
        "not_valid_after_utc",
        None,
    )

    if value is not None:
        return value

    return certificate.not_valid_after.replace(
        tzinfo=dt.timezone.utc
    )


def _utc_now() -> dt.datetime:
    return dt.datetime.now(
        dt.timezone.utc
    )


def _normalize_datetime(
    value: dt.datetime,
) -> dt.datetime:
    if not isinstance(
        value,
        dt.datetime,
    ):
        raise CertificateValidationError(
            "now must be a datetime."
        )

    if value.tzinfo is None:
        raise CertificateValidationError(
            "now must be timezone-aware."
        )

    return value.astimezone(
        dt.timezone.utc
    )


def _validate_nonempty_string(
    value: str,
    *,
    field: str,
) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise CertificateError(
            f"{field} must be a non-empty string."
        )


__all__ = [
    "CertificateAuthority",
    "CertificateError",
    "CertificateExpiredError",
    "CertificateIdentityError",
    "CertificateIssuanceError",
    "CertificateKeyMismatchError",
    "CertificateNotYetValidError",
    "CertificateTrustError",
    "CertificateType",
    "CertificateValidationError",
    "DEFAULT_CA_VALIDITY_DAYS",
    "DEFAULT_WORKLOAD_VALIDITY_MINUTES",
    "OSHA_CERTIFICATE_PROFILE",
    "ValidatedCertificate",
    "WorkloadCertificate",
    "certificate_fingerprint_sha256",
    "create_development_ca",
    "identity_from_certificate",
    "issue_workload_certificate",
    "load_certificate_pem",
    "validate_workload_certificate",
    "verify_private_key_matches_certificate",
]
