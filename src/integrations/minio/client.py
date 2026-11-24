from __future__ import annotations

"""
Read-only MinIO/S3-compatible client boundary
for OSHA.

This initial implementation exposes discovery
and metadata operations only.

No bucket/object mutation is implemented here.
"""

from datetime import (
    date,
    datetime,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationConnectionError,
    IntegrationOperationError,
    IntegrationResourceNotFoundError,
    IntegrationUnavailableError,
    wrap_provider_error,
)

from src.integrations.minio.config import (
    MinioConfig,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------

try:
    from minio import (
        Minio,
    )

    try:
        from minio.error import (
            S3Error,
        )

    except ImportError:
        S3Error = Exception

except ImportError:
    Minio = None
    S3Error = Exception


try:
    import urllib3

except ImportError:
    urllib3 = None


# ---------------------------------------------------------------
# Client error
# ---------------------------------------------------------------


class MinioClientError(IntegrationOperationError):
    default_code = "minio_client_error"


# ---------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------


def _safe_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    return value


def bucket_to_dict(
    bucket: Any,
) -> Dict[str, Any]:
    """
    Convert a MinIO Bucket object into bounded
    metadata.
    """

    if bucket is None:
        return {}

    if isinstance(
        bucket,
        Mapping,
    ):
        return {str(key): (_safe_scalar(value)) for key, value in bucket.items()}

    value: Dict[
        str,
        Any,
    ] = {}

    for attribute in (
        "name",
        "creation_date",
    ):
        item = getattr(
            bucket,
            attribute,
            None,
        )

        if item is not None:
            value[attribute] = _safe_scalar(item)

    return value


def object_to_dict(
    obj: Any,
) -> Dict[str, Any]:
    """
    Convert object listing/stat metadata without
    reading object contents.
    """

    if obj is None:
        return {}

    if isinstance(
        obj,
        Mapping,
    ):
        return {str(key): (_safe_scalar(value)) for key, value in obj.items()}

    value: Dict[
        str,
        Any,
    ] = {}

    # Keep this bounded to normal object metadata.
    for attribute in (
        "object_name",
        "size",
        "etag",
        "last_modified",
        "content_type",
        "version_id",
        "is_latest",
        "delete_marker",
        "is_dir",
    ):
        item = getattr(
            obj,
            attribute,
            None,
        )

        if item is not None:
            value[attribute] = _safe_scalar(item)

    metadata = getattr(
        obj,
        "metadata",
        None,
    )

    if isinstance(
        metadata,
        Mapping,
    ):
        # Metadata can contain application-level
        # values. Preserve it as ordinary data;
        # normal OSHA serialization/redaction
        # controls still apply upstream.
        value["metadata"] = dict(metadata)

    return value


def objects_to_dicts(
    objects: Iterable[Any],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    return [object_to_dict(item) for item in objects]


# ---------------------------------------------------------------
# MinIO client
# ---------------------------------------------------------------


class MinioClient:
    """
    Thin provider wrapper around minio.Minio.

    Connection/authentication construction stays
    here.

    Generic target validation and operation
    authorization remain in IntegrationAdapter.
    """

    def __init__(
        self,
        *,
        config: MinioConfig,
        credential: Optional[SecretValue] = None,
    ) -> None:
        self.config = config

        self.config.validate()

        self.credential = credential

        self._client: Any = None

        self._http_client: Any = None

    # ---------------------------------------------------------
    # Connected
    # ---------------------------------------------------------

    @property
    def connected(
        self,
    ) -> bool:
        return self._client is not None

    # ---------------------------------------------------------
    # Provider client
    # ---------------------------------------------------------

    @property
    def client(
        self,
    ) -> Any:
        if not self.connected:
            raise (IntegrationConnectionError("MinIO client is not connected."))

        return self._client

    # ---------------------------------------------------------
    # Custom HTTP client
    # ---------------------------------------------------------

    def _build_http_client(
        self,
    ) -> Any:
        """
        Build a bounded urllib3 pool so OSHA's
        configured timeout and certificate policy
        are explicit.
        """

        if urllib3 is None:
            return None

        certificate_requirement = (
            "CERT_REQUIRED" if self.config.verify_tls else "CERT_NONE"
        )

        return urllib3.PoolManager(
            timeout=urllib3.Timeout(
                connect=(self.config.request_timeout_seconds),
                read=(self.config.request_timeout_seconds),
            ),
            cert_reqs=(certificate_requirement),
            retries=False,
        )

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    def connect(
        self,
    ) -> "MinioClient":
        if self.connected:
            return self

        if Minio is None:
            raise (IntegrationConnectionError("MinIO Python SDK is not installed."))

        access_key = self.config.auth.access_key

        secret_key = None

        if self.credential is not None:
            # Reveal only at provider SDK
            # construction.
            secret_key = self.credential.reveal()

        if access_key is not None and secret_key is None:
            raise (
                IntegrationConnectionError(
                    "MinIO access key is "
                    "configured but the "
                    "secret credential was "
                    "not resolved."
                )
            )

        if access_key is None and secret_key is not None:
            raise (
                IntegrationConnectionError(
                    "MinIO secret credential cannot be used without an access key."
                )
            )

        try:
            self._http_client = self._build_http_client()

            arguments: Dict[
                str,
                Any,
            ] = {
                "endpoint": (self.config.endpoint),
                "access_key": (access_key),
                "secret_key": (secret_key),
                "secure": (self.config.secure),
                "region": (self.config.region),
            }

            if self._http_client is not None:
                arguments["http_client"] = self._http_client

            self._client = Minio(**arguments)

            return self

        except Exception as exc:
            self._clear()

            raise wrap_provider_error(
                exc,
                provider="minio",
                target_id=(self.config.target_id),
                message=("MinIO client creation failed."),
                retryable=False,
                code=("minio_connection_failed"),
            ) from exc

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(
        self,
    ) -> None:
        """
        Release locally-held provider objects.

        Minio does not require OSHA to expose a
        remote close operation here.
        """

        self._clear()

    def _clear(
        self,
    ) -> None:
        self._client = None

        http_client = self._http_client

        self._http_client = None

        if http_client is not None:
            clear = getattr(
                http_client,
                "clear",
                None,
            )

            if callable(clear):
                try:
                    clear()

                except Exception:
                    pass

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping(
        self,
    ) -> bool:
        """
        Use a read-only bucket listing as an
        authenticated connectivity probe.
        """

        if not self.connected:
            return False

        try:
            self.client.list_buckets()

            return True

        except Exception:
            return False

    # ---------------------------------------------------------
    # Buckets
    # ---------------------------------------------------------

    def list_buckets(
        self,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_feature(
            self.config.enable_buckets,
            "buckets",
        )

        try:
            return [bucket_to_dict(bucket) for bucket in self.client.list_buckets()]

        except Exception as exc:
            raise self._operation_error(
                exc,
                "MinIO bucket listing failed.",
            ) from exc

    def bucket_exists(
        self,
        bucket_name: str,
    ) -> bool:
        self._require_feature(
            self.config.enable_buckets,
            "buckets",
        )

        bucket_name = self._identifier(
            bucket_name,
            "bucket_name",
        )

        try:
            return bool(self.client.bucket_exists(bucket_name))

        except Exception as exc:
            raise self._operation_error(
                exc,
                "MinIO bucket existence check failed.",
            ) from exc

    # ---------------------------------------------------------
    # Objects
    # ---------------------------------------------------------

    def list_objects(
        self,
        *,
        bucket_name: str,
        prefix: Optional[str] = None,
        recursive: bool = False,
        limit: Optional[int] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_feature(
            self.config.enable_objects,
            "objects",
        )

        bucket_name = self._identifier(
            bucket_name,
            "bucket_name",
        )

        if limit is not None and limit < 1:
            raise ValueError("object limit must be >= 1")

        try:
            iterator = self.client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=recursive,
            )

            output: List[
                Dict[
                    str,
                    Any,
                ]
            ] = []

            for item in iterator:
                output.append(object_to_dict(item))

                if limit is not None and len(output) >= limit:
                    break

            return output

        except Exception as exc:
            raise self._operation_error(
                exc,
                "MinIO object listing failed.",
            ) from exc

    def stat_object(
        self,
        *,
        bucket_name: str,
        object_name: str,
        version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._require_feature(
            self.config.enable_objects,
            "objects",
        )

        bucket_name = self._identifier(
            bucket_name,
            "bucket_name",
        )

        object_name = self._object_name(object_name)

        try:
            value = self.client.stat_object(
                bucket_name,
                object_name,
                version_id=(version_id),
            )

            return object_to_dict(value)

        except S3Error as exc:
            code = getattr(
                exc,
                "code",
                None,
            )

            if code in {
                "NoSuchKey",
                "NoSuchObject",
                "NoSuchBucket",
            }:
                raise (
                    IntegrationResourceNotFoundError(
                        "MinIO object or bucket was not found."
                    )
                ) from exc

            raise self._operation_error(
                exc,
                "MinIO object metadata lookup failed.",
            ) from exc

        except Exception as exc:
            raise self._operation_error(
                exc,
                "MinIO object metadata lookup failed.",
            ) from exc

    # ---------------------------------------------------------
    # Guards
    # ---------------------------------------------------------

    def _require_feature(
        self,
        enabled: bool,
        feature: str,
    ) -> None:
        if not self.connected:
            raise (IntegrationConnectionError("MinIO client is not connected."))

        if not enabled:
            raise (
                IntegrationCapabilityError(
                    f"MinIO integration feature is disabled: {feature}"
                )
            )

    # ---------------------------------------------------------
    #     # Bucket identifier
    # ---------------------------------------------------------

    @staticmethod
    def _identifier(
        value: Any,
        name: str,
    ) -> str:
        text = str(value).strip()

        if not text:
            raise (IntegrationOperationError(f"{name} is required."))

        # Bucket identifiers should not be allowed
        # to masquerade as arbitrary URLs/paths.
        if any(
            character in text
            for character in (
                "/",
                "\\",
                "?",
                "#",
            )
        ):
            raise (
                IntegrationOperationError(f"{name} contains unsupported characters.")
            )

        return text

    # ---------------------------------------------------------
    # Object name
    # ---------------------------------------------------------

    @staticmethod
    def _object_name(
        value: Any,
    ) -> str:
        """
        Object names may legitimately contain
        slash characters because object stores
        use key prefixes.

        We still reject an empty key.
        """

        text = str(value)

        if not text:
            raise (IntegrationOperationError("object_name is required."))

        return text

    # ---------------------------------------------------------
    # Provider error translation
    # ---------------------------------------------------------

    def _operation_error(
        self,
        error: BaseException,
        message: str,
    ) -> MinioClientError:
        wrapped = wrap_provider_error(
            error,
            provider="minio",
            target_id=(self.config.target_id),
            message=message,
            retryable=False,
            code=("minio_operation_failed"),
        )

        return MinioClientError(
            message,
            code=wrapped.code,
            retryable=(wrapped.retryable),
            details=(wrapped.details),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connected": (self.connected),
            "target_id": (self.config.target_id),
            "endpoint": (self.config.endpoint_url),
            "secure": (self.config.secure),
            "verify_tls": (self.config.verify_tls),
            "region": (self.config.region),
            "resources": {
                "buckets": (self.config.enable_buckets),
                "objects": (self.config.enable_objects),
            },
            "mutating_operations_enabled": (False),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "MinioClientError",
    "bucket_to_dict",
    "object_to_dict",
    "objects_to_dicts",
    "MinioClient",
]
