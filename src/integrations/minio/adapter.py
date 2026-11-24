from __future__ import annotations

"""
OSHA IntegrationAdapter implementation for
MinIO/S3-compatible object storage.

This first implementation deliberately exposes
read-only bucket and object metadata operations.
"""

from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
)

from src.integrations.base import (
    IntegrationAdapter,
)

from src.integrations.config import (
    IntegrationsConfig,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationOperationError,
)

from src.integrations.minio.client import (
    MinioClient,
)

from src.integrations.minio.config import (
    MinioConfig,
)

from src.integrations.schema import (
    IntegrationCapability,
    IntegrationOperation,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Capability names
# ---------------------------------------------------------------

CAPABILITY_BUCKET_READ = "bucket.read"

CAPABILITY_OBJECT_READ = "object.read"


# ---------------------------------------------------------------
# Actions
# ---------------------------------------------------------------

ACTION_LIST = "list"

ACTION_EXISTS = "exists"

ACTION_STAT = "stat"


# ---------------------------------------------------------------
# MinIO adapter
# ---------------------------------------------------------------


class MinioAdapter(IntegrationAdapter):
    """
    Translate OSHA IntegrationOperation objects
    into bounded MinioClient read operations.

    Current capabilities:

      bucket.read
          list
          exists

      object.read
          list
          stat

    No write capability is advertised.
    """

    def __init__(
        self,
        *,
        minio_config: MinioConfig,
        config: Optional[IntegrationsConfig] = None,
    ) -> None:
        self.minio_config = minio_config

        self.minio_config.validate()

        super().__init__(
            target=(self.minio_config.target()),
            config=(config or IntegrationsConfig()),
        )

    # ---------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------

    def define_capabilities(
        self,
    ) -> Iterable[IntegrationCapability]:
        capabilities = []

        # -----------------------------------------------------
        # Buckets
        # -----------------------------------------------------

        if self.minio_config.enable_buckets:
            capabilities.append(
                IntegrationCapability(
                    name=(CAPABILITY_BUCKET_READ),
                    resource="bucket",
                    actions=[
                        ACTION_LIST,
                        ACTION_EXISTS,
                    ],
                    read_only=True,
                    attributes={
                        "provider": "minio",
                        "protocol": ("s3-compatible"),
                    },
                )
            )

        # -----------------------------------------------------
        # Objects
        # -----------------------------------------------------

        if self.minio_config.enable_objects:
            capabilities.append(
                IntegrationCapability(
                    name=(CAPABILITY_OBJECT_READ),
                    resource="object",
                    actions=[
                        ACTION_LIST,
                        ACTION_STAT,
                    ],
                    read_only=True,
                    attributes={
                        "provider": "minio",
                        "protocol": ("s3-compatible"),
                        "object_content_read": (False),
                    },
                )
            )

        return capabilities

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    def connect_client(
        self,
        credential: Optional[SecretValue],
    ) -> MinioClient:
        client = MinioClient(
            config=(self.minio_config),
            credential=credential,
        )

        client.connect()

        return client

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close_client(
        self,
        client: MinioClient,
    ) -> None:
        client.close()

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping_client(
        self,
        client: MinioClient,
    ) -> bool:
        return client.ping()

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    def execute_operation(
        self,
        client: MinioClient,
        operation: IntegrationOperation,
    ) -> Any:
        """
        Translate one validated IntegrationOperation
        into a bounded MinIO metadata/read call.

        No object body retrieval or mutation is
        permitted here.
        """

        operation.finalize()

        capability = operation.capability

        action = operation.action

        # -----------------------------------------------------
        # Bucket list
        # -----------------------------------------------------

        if capability == CAPABILITY_BUCKET_READ and action == ACTION_LIST:
            return {"buckets": (client.list_buckets())}

        # -----------------------------------------------------
        # Bucket existence
        # -----------------------------------------------------

        if capability == CAPABILITY_BUCKET_READ and action == ACTION_EXISTS:
            bucket_name = self._bucket_name(operation)

            return {
                "bucket": (bucket_name),
                "exists": (client.bucket_exists(bucket_name)),
            }

        # -----------------------------------------------------
        # Object listing
        # -----------------------------------------------------

        if capability == CAPABILITY_OBJECT_READ and action == ACTION_LIST:
            bucket_name = self._bucket_name(operation)

            prefix = operation.parameters.get("prefix")

            if prefix is not None:
                prefix = str(prefix)

            recursive = operation.parameters.get(
                "recursive",
                False,
            )

            recursive = self._boolean_parameter(
                recursive,
                name="recursive",
            )

            limit = operation.parameters.get("limit")

            if limit is not None:
                limit = self._positive_integer(
                    limit,
                    name="limit",
                )

            return {
                "bucket": (bucket_name),
                "prefix": (prefix),
                "objects": (
                    client.list_objects(
                        bucket_name=(bucket_name),
                        prefix=prefix,
                        recursive=(recursive),
                        limit=limit,
                    )
                ),
            }

        # -----------------------------------------------------
        # Object metadata
        # -----------------------------------------------------

        if capability == CAPABILITY_OBJECT_READ and action == ACTION_STAT:
            bucket_name = self._bucket_name(operation)

            object_name = self._object_name(operation)

            version_id = operation.parameters.get("version_id")

            if version_id is not None:
                version_id = str(version_id).strip()

                if not version_id:
                    version_id = None

            return {
                "bucket": (bucket_name),
                "object_name": (object_name),
                "object": (
                    client.stat_object(
                        bucket_name=(bucket_name),
                        object_name=(object_name),
                        version_id=(version_id),
                    )
                ),
            }

        # -----------------------------------------------------
        # Unsupported
        # -----------------------------------------------------

        raise (
            IntegrationCapabilityError(
                f"Unsupported MinIO integration operation: {capability}/{action}"
            )
        )

    # ---------------------------------------------------------
    # Bucket parameter
    # ---------------------------------------------------------

    @staticmethod
    def _bucket_name(
        operation: IntegrationOperation,
    ) -> str:
        """
        A bucket can be supplied as resource_id
        or parameters['bucket'].

        resource_id takes precedence.
        """

        value = operation.resource_id or operation.parameters.get("bucket")

        if value is None:
            raise (IntegrationOperationError("MinIO operation requires a bucket."))

        value = str(value).strip()

        if not value:
            raise (IntegrationOperationError("MinIO bucket cannot be empty."))

        return value

    # ---------------------------------------------------------
    # Object parameter
    # ---------------------------------------------------------

    @staticmethod
    def _object_name(
        operation: IntegrationOperation,
    ) -> str:
        value = operation.parameters.get("object_name")

        if value is None:
            raise (IntegrationOperationError("MinIO object stat requires object_name."))

        value = str(value)

        if not value:
            raise (IntegrationOperationError("MinIO object_name cannot be empty."))

        return value

    # ---------------------------------------------------------
    # Positive integer parameter
    # ---------------------------------------------------------

    @staticmethod
    def _positive_integer(
        value: Any,
        *,
        name: str,
    ) -> int:
        try:
            result = int(value)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise (IntegrationOperationError(f"{name} must be an integer.")) from exc

        if result < 1:
            raise (IntegrationOperationError(f"{name} must be >= 1."))

        return result

    # ---------------------------------------------------------
    # Boolean parameter
    # ---------------------------------------------------------

    @staticmethod
    def _boolean_parameter(
        value: Any,
        *,
        name: str,
    ) -> bool:
        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):
            normalized = value.strip().lower()

            if normalized in {
                "true",
                "1",
                "yes",
                "on",
            }:
                return True

            if normalized in {
                "false",
                "0",
                "no",
                "off",
            }:
                return False

        raise (IntegrationOperationError(f"{name} must be a boolean."))

    # ---------------------------------------------------------
    # Provider summary
    # ---------------------------------------------------------

    def provider_summary(
        self,
    ) -> Dict[str, Any]:
        client_summary = None

        if self.ready:
            client = self.client

            if isinstance(
                client,
                MinioClient,
            ):
                client_summary = client.summary()

        return {
            "provider": "minio",
            "environment_id": (self.target.environment_id),
            "target_id": (self.target.target_id),
            "endpoint": (self.minio_config.endpoint_url),
            "secure": (self.minio_config.secure),
            "ready": (self.ready),
            "client": (client_summary),
            "capabilities": [capability.name for capability in self.capabilities()],
            "object_content_read_enabled": (False),
            "mutating_capabilities_advertised": (False),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "CAPABILITY_BUCKET_READ",
    "CAPABILITY_OBJECT_READ",
    "ACTION_LIST",
    "ACTION_EXISTS",
    "ACTION_STAT",
    "MinioAdapter",
]
