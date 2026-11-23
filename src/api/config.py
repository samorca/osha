from __future__ import annotations

"""
Production API configuration for OSHA.

This module centralizes public HTTP configuration
without constructing the FastAPI application.

Configuration areas:

    service identity
    bind/network settings
    API versioning
    documentation exposure
    CORS
    trusted hosts
    HTTPS policy
    request limits
    timeout policy
    concurrency
    inference controls
    accelerator controls
    model-management controls
    incident controls
    decision controls
    remediation safety gates
    observability
    environment loading

Environment prefix:

    OSHA_API_

Examples:

    OSHA_API_ENVIRONMENT=production

    OSHA_API_HOST=0.0.0.0

    OSHA_API_PORT=8000

    OSHA_API_CORS_ORIGINS=https://console.example.com

    OSHA_API_TRUSTED_HOSTS=api.example.com

    OSHA_API_ENABLE_REMEDIATION=false

This module does not:

    create FastAPI
    register routes
    configure logging handlers
    execute inference
    execute remediation
    mutate infrastructure
"""

import json
import os

from dataclasses import (
    dataclass,
    field,
    replace,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
)

from urllib.parse import (
    urlparse,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

API_CONFIG_VERSION = "1"

DEFAULT_ENV_PREFIX = "OSHA_API_"

DEFAULT_SERVICE_NAME = "osha-api"

DEFAULT_API_PREFIX = "/api"

DEFAULT_API_VERSION = "v1"

DEFAULT_HOST = "127.0.0.1"

DEFAULT_PORT = 8000


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class APIEnvironment(
    str,
    Enum,
):
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(
    str,
    Enum,
):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _environment_value(
    environment: Mapping[
        str,
        str,
    ],
    *,
    prefix: str,
    name: str,
) -> Optional[str]:
    value = environment.get(
        prefix + name
    )

    if value is None:
        return None

    text = str(
        value
    ).strip()

    return (
        text
        if text
        else None
    )


def parse_bool(
    value: Any,
    *,
    name: str,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }:
        return False

    raise ValueError(
        f"{name} must be a boolean"
    )


def parse_int(
    value: Any,
    *,
    name: str,
    minimum: Optional[
        int
    ] = None,
    maximum: Optional[
        int
    ] = None,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be an integer"
        )

    try:
        result = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be an integer"
        ) from exc

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    if (
        maximum is not None
        and result > maximum
    ):
        raise ValueError(
            f"{name} must be <= {maximum}"
        )

    return result


def parse_float(
    value: Any,
    *,
    name: str,
    minimum: Optional[
        float
    ] = None,
    maximum: Optional[
        float
    ] = None,
) -> float:
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    if (
        maximum is not None
        and result > maximum
    ):
        raise ValueError(
            f"{name} must be <= {maximum}"
        )

    return result


def parse_csv(
    value: Any,
) -> Tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(
        value,
        str,
    ):
        source = value.split(
            ","
        )

    elif isinstance(
        value,
        Iterable,
    ):
        source = value

    else:
        source = (
            value,
        )

    output = []
    seen = set()

    for item in source:
        text = str(
            item
        ).strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(
            text
        )

        output.append(
            text
        )

    return tuple(
        output
    )


def parse_json_mapping(
    value: Any,
    *,
    name: str,
) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(
        value,
        Mapping,
    ):
        return dict(
            value
        )

    try:
        parsed = json.loads(
            str(
                value
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{name} must contain "
            "valid JSON"
        ) from exc

    if not isinstance(
        parsed,
        Mapping,
    ):
        raise ValueError(
            f"{name} must contain "
            "a JSON object"
        )

    return dict(
        parsed
    )


def normalize_path_prefix(
    value: str,
    *,
    name: str,
) -> str:
    text = str(
        value
    ).strip()

    if not text:
        raise ValueError(
            f"{name} cannot be empty"
        )

    if not text.startswith(
        "/"
    ):
        text = (
            "/"
            + text
        )

    if (
        len(text) > 1
        and text.endswith(
            "/"
        )
    ):
        text = text[
            :-1
        ]

    if "//" in text:
        raise ValueError(
            f"{name} cannot contain //"
        )

    return text


def validate_origin(
    value: str,
) -> str:
    origin = str(
        value
    ).strip()

    if origin == "*":
        return origin

    parsed = urlparse(
        origin
    )

    if (
        parsed.scheme
        not in {
            "http",
            "https",
        }
        or not parsed.netloc
    ):
        raise ValueError(
            "CORS origin must be '*' "
            "or an absolute http/https "
            "origin"
        )

    if (
        parsed.path
        not in {
            "",
            "/",
        }
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "CORS origin must not "
            "contain a path, query, "
            "or fragment"
        )

    return origin.rstrip(
        "/"
    )


def validate_host_pattern(
    value: str,
) -> str:
    host = str(
        value
    ).strip()

    if not host:
        raise ValueError(
            "trusted host cannot be empty"
        )

    if any(
        character.isspace()
        for character in host
    ):
        raise ValueError(
            "trusted host cannot "
            "contain whitespace"
        )

    if "://" in host:
        raise ValueError(
            "trusted host must be a "
            "hostname pattern, not a URL"
        )

    if "/" in host:
        raise ValueError(
            "trusted host cannot "
            "contain a path"
        )

    return host


# ------------------------------------------------------------------
# Server config
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APIServerConfig:
    host: str = DEFAULT_HOST

    port: int = DEFAULT_PORT

    workers: int = 1

    backlog: int = 2048

    keep_alive_seconds: int = 5

    graceful_shutdown_seconds: int = 30

    proxy_headers: bool = False

    forwarded_allow_ips: Tuple[
        str,
        ...,
    ] = ()

    root_path: str = ""

    def validate(
        self,
    ) -> None:
        if not str(
            self.host
        ).strip():
            raise ValueError(
                "API host cannot be empty"
            )

        parse_int(
            self.port,
            name="port",
            minimum=1,
            maximum=65535,
        )

        parse_int(
            self.workers,
            name="workers",
            minimum=1,
        )

        parse_int(
            self.backlog,
            name="backlog",
            minimum=1,
        )

        parse_int(
            self.keep_alive_seconds,
            name="keep_alive_seconds",
            minimum=1,
        )

        parse_int(
            self.graceful_shutdown_seconds,
            name=(
                "graceful_shutdown_seconds"
            ),
            minimum=1,
        )

        if self.root_path:
            normalize_path_prefix(
                self.root_path,
                name="root_path",
            )

        if (
            self.forwarded_allow_ips
            and not self.proxy_headers
        ):
            raise ValueError(
                "forwarded_allow_ips requires "
                "proxy_headers=true"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "host": self.host,

            "port": self.port,

            "workers": self.workers,

            "backlog": self.backlog,

            "keep_alive_seconds": (
                self.keep_alive_seconds
            ),

            "graceful_shutdown_seconds": (
                self.
                graceful_shutdown_seconds
            ),

            "proxy_headers": (
                self.proxy_headers
            ),

            "forwarded_allow_ips": list(
                self.forwarded_allow_ips
            ),

            "root_path": (
                self.root_path
            ),
        }


# ------------------------------------------------------------------
# Documentation config
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APIDocumentationConfig:
    enabled: bool = True

    docs_path: str = "/docs"

    redoc_path: str = "/redoc"

    openapi_path: str = "/openapi.json"

    include_internal_routes: bool = False

    def validate(
        self,
    ) -> None:
        if not self.enabled:
            return

        paths = (
            self.docs_path,
            self.redoc_path,
            self.openapi_path,
        )

        normalized = []

        for path in paths:
            normalized.append(
                normalize_path_prefix(
                    path,
                    name=(
                        "documentation path"
                    ),
                )
            )

        if len(
            set(
                normalized
            )
        ) != len(
            normalized
        ):
            raise ValueError(
                "documentation paths "
                "must be unique"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": self.enabled,

            "docs_path": (
                self.docs_path
                if self.enabled
                else None
            ),

            "redoc_path": (
                self.redoc_path
                if self.enabled
                else None
            ),

            "openapi_path": (
                self.openapi_path
                if self.enabled
                else None
            ),

            "include_internal_routes": (
                self.
                include_internal_routes
            ),
        }


# ------------------------------------------------------------------
# CORS config
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APICORSConfig:
    enabled: bool = False

    allowed_origins: Tuple[
        str,
        ...,
    ] = ()

    allowed_methods: Tuple[
        str,
        ...,
    ] = (
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    )

    allowed_headers: Tuple[
        str,
        ...,
    ] = (
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Request-ID",
        "Idempotency-Key",
    )

    exposed_headers: Tuple[
        str,
        ...,
    ] = (
        "X-Request-ID",
    )

    allow_credentials: bool = False

    max_age_seconds: int = 600

    def validate(
        self,
    ) -> None:
        parse_int(
            self.max_age_seconds,
            name="cors max_age_seconds",
            minimum=0,
        )

        origins = tuple(
            validate_origin(
                origin
            )
            for origin
            in self.allowed_origins
        )

        if (
            self.enabled
            and not origins
        ):
            raise ValueError(
                "CORS is enabled but no "
                "allowed origins are configured"
            )

        if (
            "*"
            in origins
            and self.allow_credentials
        ):
            raise ValueError(
                "Wildcard CORS origin cannot "
                "be used with credentials"
            )

        if not self.allowed_methods:
            raise ValueError(
                "CORS allowed_methods "
                "cannot be empty"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": self.enabled,

            "allowed_origins": list(
                self.allowed_origins
            ),

            "allowed_methods": list(
                self.allowed_methods
            ),

            "allowed_headers": list(
                self.allowed_headers
            ),

            "exposed_headers": list(
                self.exposed_headers
            ),

            "allow_credentials": (
                self.allow_credentials
            ),

            "max_age_seconds": (
                self.max_age_seconds
            ),
        }


# ------------------------------------------------------------------
# Host / transport security
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APITransportSecurityConfig:
    trusted_hosts: Tuple[
        str,
        ...,
    ] = (
        "localhost",
        "127.0.0.1",
        "::1",
    )

    enforce_https: bool = False

    hsts_enabled: bool = False

    hsts_max_age_seconds: int = 31_536_000

    hsts_include_subdomains: bool = True

    hsts_preload: bool = False

    def validate(
        self,
    ) -> None:
        if not self.trusted_hosts:
            raise ValueError(
                "At least one trusted host "
                "must be configured"
            )

        for host in self.trusted_hosts:
            validate_host_pattern(
                host
            )

        parse_int(
            self.hsts_max_age_seconds,
            name=(
                "hsts_max_age_seconds"
            ),
            minimum=0,
        )

        if (
            self.hsts_enabled
            and not self.enforce_https
        ):
            raise ValueError(
                "HSTS requires HTTPS "
                "enforcement"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "trusted_hosts": list(
                self.trusted_hosts
            ),

            "enforce_https": (
                self.enforce_https
            ),

            "hsts_enabled": (
                self.hsts_enabled
            ),

            "hsts_max_age_seconds": (
                self.hsts_max_age_seconds
            ),

            "hsts_include_subdomains": (
                self.
                hsts_include_subdomains
            ),

            "hsts_preload": (
                self.hsts_preload
            ),
        }


# ------------------------------------------------------------------
# Request/resource limits
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APIRequestLimitsConfig:
    maximum_body_bytes: int = (
        2
        * 1024**2
    )

    maximum_json_depth: int = 32

    maximum_query_parameters: int = 100

    maximum_header_count: int = 100

    maximum_header_bytes: int = 65_536

    maximum_path_length: int = 2048

    maximum_batch_inference_items: int = 256

    maximum_concurrent_requests: int = 256

    maximum_concurrent_inference_requests: int = 32

    maximum_concurrent_model_operations: int = 4

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "maximum_body_bytes",
                self.maximum_body_bytes,
            ),
            (
                "maximum_json_depth",
                self.maximum_json_depth,
            ),
            (
                "maximum_query_parameters",
                self.
                maximum_query_parameters,
            ),
            (
                "maximum_header_count",
                self.maximum_header_count,
            ),
            (
                "maximum_header_bytes",
                self.maximum_header_bytes,
            ),
            (
                "maximum_path_length",
                self.maximum_path_length,
            ),
            (
                "maximum_batch_inference_items",
                self.
                maximum_batch_inference_items,
            ),
            (
                "maximum_concurrent_requests",
                self.
                maximum_concurrent_requests,
            ),
            (
                "maximum_concurrent_inference_requests",
                self.
                maximum_concurrent_inference_requests,
            ),
            (
                "maximum_concurrent_model_operations",
                self.
                maximum_concurrent_model_operations,
            ),
        ):
            parse_int(
                value,
                name=name,
                minimum=1,
            )

        if (
            self.
            maximum_concurrent_inference_requests
            > self.maximum_concurrent_requests
        ):
            raise ValueError(
                "Inference concurrency cannot "
                "exceed total request concurrency"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "maximum_body_bytes": (
                self.maximum_body_bytes
            ),

            "maximum_json_depth": (
                self.maximum_json_depth
            ),

            "maximum_query_parameters": (
                self.
                maximum_query_parameters
            ),

            "maximum_header_count": (
                self.maximum_header_count
            ),

            "maximum_header_bytes": (
                self.maximum_header_bytes
            ),

            "maximum_path_length": (
                self.maximum_path_length
            ),

            "maximum_batch_inference_items": (
                self.
                maximum_batch_inference_items
            ),

            "maximum_concurrent_requests": (
                self.
                maximum_concurrent_requests
            ),

            "maximum_concurrent_inference_requests": (
                self.
                maximum_concurrent_inference_requests
            ),

            "maximum_concurrent_model_operations": (
                self.
                maximum_concurrent_model_operations
            ),
        }


# ------------------------------------------------------------------
# Timeout config
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APITimeoutConfig:
    request_seconds: float = 30.0

    inference_seconds: float = 30.0

    model_operation_seconds: float = 120.0

    accelerator_operation_seconds: float = 30.0

    health_seconds: float = 10.0

    decision_seconds: float = 30.0

    remediation_plan_seconds: float = 30.0

    remediation_execution_seconds: float = 300.0

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "request_seconds",
                self.request_seconds,
            ),
            (
                "inference_seconds",
                self.inference_seconds,
            ),
            (
                "model_operation_seconds",
                self.model_operation_seconds,
            ),
            (
                "accelerator_operation_seconds",
                self.
                accelerator_operation_seconds,
            ),
            (
                "health_seconds",
                self.health_seconds,
            ),
            (
                "decision_seconds",
                self.decision_seconds,
            ),
            (
                "remediation_plan_seconds",
                self.
                remediation_plan_seconds,
            ),
            (
                "remediation_execution_seconds",
                self.
                remediation_execution_seconds,
            ),
        ):
            parse_float(
                value,
                name=name,
                minimum=0.001,
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "request_seconds": (
                self.request_seconds
            ),

            "inference_seconds": (
                self.inference_seconds
            ),

            "model_operation_seconds": (
                self.model_operation_seconds
            ),

            "accelerator_operation_seconds": (
                self.
                accelerator_operation_seconds
            ),

            "health_seconds": (
                self.health_seconds
            ),

            "decision_seconds": (
                self.decision_seconds
            ),

            "remediation_plan_seconds": (
                self.remediation_plan_seconds
            ),

            "remediation_execution_seconds": (
                self.
                remediation_execution_seconds
            ),
        }


# ------------------------------------------------------------------
# Feature configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APIFeatureConfig:
    health_enabled: bool = True

    inference_enabled: bool = True

    accelerator_read_enabled: bool = True

    accelerator_optimization_enabled: bool = True

    model_read_enabled: bool = True

    model_load_operations_enabled: bool = True

    incident_read_enabled: bool = True

    incident_create_enabled: bool = False

    decision_read_enabled: bool = True

    decision_evaluation_enabled: bool = False

    remediation_read_enabled: bool = True

    remediation_plan_creation_enabled: bool = False

    remediation_approval_enabled: bool = False

    remediation_execution_enabled: bool = False

    remediation_live_execution_enabled: bool = False

    def validate(
        self,
    ) -> None:
        if (
            self.remediation_live_execution_enabled
            and not self.
            remediation_execution_enabled
        ):
            raise ValueError(
                "Live remediation requires "
                "remediation execution to "
                "be enabled"
            )

        if (
            self.remediation_execution_enabled
            and not self.
            remediation_plan_creation_enabled
        ):
            raise ValueError(
                "Remediation execution requires "
                "plan creation to be enabled"
            )

        if (
            self.remediation_approval_enabled
            and not self.
            remediation_plan_creation_enabled
        ):
            raise ValueError(
                "Remediation approval requires "
                "plan creation to be enabled"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "health_enabled": (
                self.health_enabled
            ),

            "inference_enabled": (
                self.inference_enabled
            ),

            "accelerator_read_enabled": (
                self.
                accelerator_read_enabled
            ),

            "accelerator_optimization_enabled": (
                self.
                accelerator_optimization_enabled
            ),

            "model_read_enabled": (
                self.model_read_enabled
            ),

            "model_load_operations_enabled": (
                self.
                model_load_operations_enabled
            ),

            "incident_read_enabled": (
                self.incident_read_enabled
            ),

            "incident_create_enabled": (
                self.incident_create_enabled
            ),

            "decision_read_enabled": (
                self.decision_read_enabled
            ),

            "decision_evaluation_enabled": (
                self.decision_evaluation_enabled
            ),

            "remediation_read_enabled": (
                self.remediation_read_enabled
            ),

            "remediation_plan_creation_enabled": (
                self.
                remediation_plan_creation_enabled
            ),

            "remediation_approval_enabled": (
                self.
                remediation_approval_enabled
            ),

            "remediation_execution_enabled": (
                self.
                remediation_execution_enabled
            ),

            "remediation_live_execution_enabled": (
                self.
                remediation_live_execution_enabled
            ),
        }


# ------------------------------------------------------------------
# Observability
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APIObservabilityConfig:
    log_level: str = (
        LogLevel.INFO.value
    )

    access_log_enabled: bool = True

    request_id_enabled: bool = True

    request_id_header: str = (
        "X-Request-ID"
    )

    server_timing_enabled: bool = False

    metrics_enabled: bool = True

    metrics_path: str = "/metrics"

    include_health_access_logs: bool = False

    include_request_body_in_logs: bool = False

    include_response_body_in_logs: bool = False

    def validate(
        self,
    ) -> None:
        if (
            self.log_level
            not in {
                item.value
                for item
                in LogLevel
            }
        ):
            raise ValueError(
                "Unsupported API log level: "
                f"{self.log_level}"
            )

        if not str(
            self.request_id_header
        ).strip():
            raise ValueError(
                "request_id_header cannot "
                "be empty"
            )

        normalize_path_prefix(
            self.metrics_path,
            name="metrics_path",
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "log_level": (
                self.log_level
            ),

            "access_log_enabled": (
                self.access_log_enabled
            ),

            "request_id_enabled": (
                self.request_id_enabled
            ),

            "request_id_header": (
                self.request_id_header
            ),

            "server_timing_enabled": (
                self.server_timing_enabled
            ),

            "metrics_enabled": (
                self.metrics_enabled
            ),

            "metrics_path": (
                self.metrics_path
            ),

            "include_health_access_logs": (
                self.
                include_health_access_logs
            ),

            "include_request_body_in_logs": (
                self.
                include_request_body_in_logs
            ),

            "include_response_body_in_logs": (
                self.
                include_response_body_in_logs
            ),
        }


# ------------------------------------------------------------------
# Top-level API configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APIConfig:
    enabled: bool = True

    service_name: str = (
        DEFAULT_SERVICE_NAME
    )

    environment: str = (
        APIEnvironment.DEVELOPMENT.value
    )

    api_prefix: str = (
        DEFAULT_API_PREFIX
    )

    api_version: str = (
        DEFAULT_API_VERSION
    )

    server: APIServerConfig = field(
        default_factory=APIServerConfig
    )

    documentation: APIDocumentationConfig = field(
        default_factory=(
            APIDocumentationConfig
        )
    )

    cors: APICORSConfig = field(
        default_factory=APICORSConfig
    )

    transport_security: APITransportSecurityConfig = field(
        default_factory=(
            APITransportSecurityConfig
        )
    )

    limits: APIRequestLimitsConfig = field(
        default_factory=(
            APIRequestLimitsConfig
        )
    )

    timeouts: APITimeoutConfig = field(
        default_factory=APITimeoutConfig
    )

    features: APIFeatureConfig = field(
        default_factory=APIFeatureConfig
    )

    observability: APIObservabilityConfig = field(
        default_factory=(
            APIObservabilityConfig
        )
    )

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if not str(
            self.service_name
        ).strip():
            raise ValueError(
                "service_name cannot be empty"
            )

        if (
            self.environment
            not in {
                item.value
                for item
                in APIEnvironment
            }
        ):
            raise ValueError(
                "Unsupported API environment: "
                f"{self.environment}"
            )

        api_prefix = normalize_path_prefix(
            self.api_prefix,
            name="api_prefix",
        )

        version = str(
            self.api_version
        ).strip()

        if not version:
            raise ValueError(
                "api_version cannot be empty"
            )

        if "/" in version:
            raise ValueError(
                "api_version cannot contain /"
            )

        self.server.validate()
        self.documentation.validate()
        self.cors.validate()
        self.transport_security.validate()
        self.limits.validate()
        self.timeouts.validate()
        self.features.validate()
        self.observability.validate()

        if (
            self.environment
            == APIEnvironment.PRODUCTION.value
        ):
            if (
                self.documentation.enabled
                and self.documentation.
                include_internal_routes
            ):
                raise ValueError(
                    "Production documentation "
                    "cannot expose internal routes"
                )

            if (
                "*"
                in self.transport_security.
                trusted_hosts
            ):
                raise ValueError(
                    "Production trusted hosts "
                    "cannot contain '*'"
                )

            if (
                self.cors.enabled
                and "*"
                in self.cors.allowed_origins
            ):
                raise ValueError(
                    "Production CORS origins "
                    "must be explicit"
                )

            if (
                self.observability.
                include_request_body_in_logs
            ):
                raise ValueError(
                    "Production request-body "
                    "logging must remain disabled"
                )

            if (
                self.observability.
                include_response_body_in_logs
            ):
                raise ValueError(
                    "Production response-body "
                    "logging must remain disabled"
                )

        if (
            self.features.
            remediation_live_execution_enabled
            and self.environment
            != APIEnvironment.PRODUCTION.value
        ):
            # Live mutation outside production may
            # still be useful in dedicated test
            # environments, but that should be an
            # explicit code/config change rather
            # than an accidental environment flag.
            raise ValueError(
                "Live remediation execution "
                "requires production environment"
            )

        _ = api_prefix

    @property
    def versioned_prefix(
        self,
    ) -> str:
        return (
            normalize_path_prefix(
                self.api_prefix,
                name="api_prefix",
            )
            + "/"
            + self.api_version.strip()
        )

    @property
    def production(
        self,
    ) -> bool:
        return (
            self.environment
            == APIEnvironment.PRODUCTION.value
        )

    @property
    def development(
        self,
    ) -> bool:
        return (
            self.environment
            == APIEnvironment.DEVELOPMENT.value
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "config_version": (
                API_CONFIG_VERSION
            ),

            "enabled": self.enabled,

            "service_name": (
                self.service_name
            ),

            "environment": (
                self.environment
            ),

            "api_prefix": (
                self.api_prefix
            ),

            "api_version": (
                self.api_version
            ),

            "versioned_prefix": (
                self.versioned_prefix
            ),

            "server": (
                self.server.to_dict()
            ),

            "documentation": (
                self.documentation.
                to_dict()
            ),

            "cors": (
                self.cors.to_dict()
            ),

            "transport_security": (
                self.transport_security.
                to_dict()
            ),

            "limits": (
                self.limits.to_dict()
            ),

            "timeouts": (
                self.timeouts.to_dict()
            ),

            "features": (
                self.features.to_dict()
            ),

            "observability": (
                self.observability.to_dict()
            ),

            "attributes": dict(
                self.attributes
            ),
        }


# ------------------------------------------------------------------
# Mapping loader
# ------------------------------------------------------------------

def from_mapping(
    mapping: Mapping[
        str,
        Any,
    ],
    *,
    base: Optional[
        APIConfig
    ] = None,
) -> APIConfig:
    """
    Build APIConfig from a nested mapping.

    This intentionally accepts only known
    configuration sections.
    """

    source = dict(
        mapping
    )

    config = (
        base
        or APIConfig()
    )

    known = {
        "enabled",
        "service_name",
        "environment",
        "api_prefix",
        "api_version",
        "server",
        "documentation",
        "cors",
        "transport_security",
        "limits",
        "timeouts",
        "features",
        "observability",
        "attributes",
    }

    unknown = (
        set(
            source
        )
        - known
    )

    if unknown:
        raise ValueError(
            "Unknown API configuration "
            "keys: "
            + ", ".join(
                sorted(
                    unknown
                )
            )
        )

    def section(
        name: str,
    ) -> Dict[str, Any]:
        value = source.get(
            name
        )

        if value is None:
            return {}

        if not isinstance(
            value,
            Mapping,
        ):
            raise ValueError(
                f"{name} must be a mapping"
            )

        return dict(
            value
        )

    server = replace(
        config.server,
        **section(
            "server"
        ),
    )

    documentation = replace(
        config.documentation,
        **section(
            "documentation"
        ),
    )

    cors = replace(
        config.cors,
        **section(
            "cors"
        ),
    )

    transport = replace(
        config.transport_security,
        **section(
            "transport_security"
        ),
    )

    limits = replace(
        config.limits,
        **section(
            "limits"
        ),
    )

    timeouts = replace(
        config.timeouts,
        **section(
            "timeouts"
        ),
    )

    features = replace(
        config.features,
        **section(
            "features"
        ),
    )

    observability = replace(
        config.observability,
        **section(
            "observability"
        ),
    )

    result = replace(
        config,

        enabled=source.get(
            "enabled",
            config.enabled,
        ),

        service_name=source.get(
            "service_name",
            config.service_name,
        ),

        environment=source.get(
            "environment",
            config.environment,
        ),

        api_prefix=source.get(
            "api_prefix",
            config.api_prefix,
        ),

        api_version=source.get(
            "api_version",
            config.api_version,
        ),

        server=server,

        documentation=documentation,

        cors=cors,

        transport_security=(
            transport
        ),

        limits=limits,

        timeouts=timeouts,

        features=features,

        observability=(
            observability
        ),

        attributes=dict(
            source.get(
                "attributes",
                config.attributes,
            )
        ),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Environment loader
# ------------------------------------------------------------------

def from_environment(
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
    environment: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    base: Optional[
        APIConfig
    ] = None,
) -> APIConfig:
    """
    Load production-facing API settings from
    environment variables.

    Only settings intentionally exposed as
    environment configuration are loaded here.
    """

    env = (
        dict(
            os.environ
        )
        if environment is None
        else dict(
            environment
        )
    )

    config = (
        base
        or APIConfig()
    )

    top: Dict[
        str,
        Any,
    ] = {}

    raw = _environment_value(
        env,
        prefix=prefix,
        name="ENABLED",
    )

    if raw is not None:
        top[
            "enabled"
        ] = parse_bool(
            raw,
            name="ENABLED",
        )

    for env_name, key in (
        (
            "SERVICE_NAME",
            "service_name",
        ),
        (
            "ENVIRONMENT",
            "environment",
        ),
        (
            "API_PREFIX",
            "api_prefix",
        ),
        (
            "API_VERSION",
            "api_version",
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            top[
                key
            ] = raw

    # ------------------------------------------------------
    # Server
    # ------------------------------------------------------

    server: Dict[
        str,
        Any,
    ] = {}

    for env_name, key in (
        (
            "HOST",
            "host",
        ),
        (
            "ROOT_PATH",
            "root_path",
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            server[
                key
            ] = raw

    for env_name, key, minimum, maximum in (
        (
            "PORT",
            "port",
            1,
            65535,
        ),
        (
            "WORKERS",
            "workers",
            1,
            None,
        ),
        (
            "BACKLOG",
            "backlog",
            1,
            None,
        ),
        (
            "KEEP_ALIVE_SECONDS",
            "keep_alive_seconds",
            1,
            None,
        ),
        (
            "GRACEFUL_SHUTDOWN_SECONDS",
            "graceful_shutdown_seconds",
            1,
            None,
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            server[
                key
            ] = parse_int(
                raw,
                name=env_name,
                minimum=minimum,
                maximum=maximum,
            )

    raw = _environment_value(
        env,
        prefix=prefix,
        name="PROXY_HEADERS",
    )

    if raw is not None:
        server[
            "proxy_headers"
        ] = parse_bool(
            raw,
            name="PROXY_HEADERS",
        )

    raw = _environment_value(
        env,
        prefix=prefix,
        name="FORWARDED_ALLOW_IPS",
    )

    if raw is not None:
        server[
            "forwarded_allow_ips"
        ] = parse_csv(
            raw
        )

    # ------------------------------------------------------
    # Documentation
    # ------------------------------------------------------

    documentation: Dict[
        str,
        Any,
    ] = {}

    raw = _environment_value(
        env,
        prefix=prefix,
        name="DOCS_ENABLED",
    )

    if raw is not None:
        documentation[
            "enabled"
        ] = parse_bool(
            raw,
            name="DOCS_ENABLED",
        )

    # ------------------------------------------------------
    # CORS
    # ------------------------------------------------------

    cors: Dict[
        str,
        Any,
    ] = {}

    raw = _environment_value(
        env,
        prefix=prefix,
        name="CORS_ENABLED",
    )

    if raw is not None:
        cors[
            "enabled"
        ] = parse_bool(
            raw,
            name="CORS_ENABLED",
        )

    raw = _environment_value(
        env,
        prefix=prefix,
        name="CORS_ORIGINS",
    )

    if raw is not None:
        cors[
            "allowed_origins"
        ] = tuple(
            validate_origin(
                origin
            )
            for origin
            in parse_csv(
                raw
            )
        )

    raw = _environment_value(
        env,
        prefix=prefix,
        name="CORS_ALLOW_CREDENTIALS",
    )

    if raw is not None:
        cors[
            "allow_credentials"
        ] = parse_bool(
            raw,
            name=(
                "CORS_ALLOW_CREDENTIALS"
            ),
        )

    # ------------------------------------------------------
    # Trusted hosts / HTTPS
    # ------------------------------------------------------

    transport: Dict[
        str,
        Any,
    ] = {}

    raw = _environment_value(
        env,
        prefix=prefix,
        name="TRUSTED_HOSTS",
    )

    if raw is not None:
        transport[
            "trusted_hosts"
        ] = tuple(
            validate_host_pattern(
                host
            )
            for host
            in parse_csv(
                raw
            )
        )

    for env_name, key in (
        (
            "ENFORCE_HTTPS",
            "enforce_https",
        ),
        (
            "HSTS_ENABLED",
            "hsts_enabled",
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            transport[
                key
            ] = parse_bool(
                raw,
                name=env_name,
            )

    # ------------------------------------------------------
    # Limits
    # ------------------------------------------------------

    limits: Dict[
        str,
        Any,
    ] = {}

    for env_name, key in (
        (
            "MAX_BODY_BYTES",
            "maximum_body_bytes",
        ),
        (
            "MAX_BATCH_INFERENCE_ITEMS",
            "maximum_batch_inference_items",
        ),
        (
            "MAX_CONCURRENT_REQUESTS",
            "maximum_concurrent_requests",
        ),
        (
            "MAX_CONCURRENT_INFERENCE_REQUESTS",
            (
                "maximum_concurrent_"
                "inference_requests"
            ),
        ),
        (
            "MAX_CONCURRENT_MODEL_OPERATIONS",
            (
                "maximum_concurrent_"
                "model_operations"
            ),
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            limits[
                key
            ] = parse_int(
                raw,
                name=env_name,
                minimum=1,
            )

    # ------------------------------------------------------
    # Timeouts
    # ------------------------------------------------------

    timeouts: Dict[
        str,
        Any,
    ] = {}

    for env_name, key in (
        (
            "REQUEST_TIMEOUT_SECONDS",
            "request_seconds",
        ),
        (
            "INFERENCE_TIMEOUT_SECONDS",
            "inference_seconds",
        ),
        (
            "MODEL_OPERATION_TIMEOUT_SECONDS",
            "model_operation_seconds",
        ),
        (
            "REMEDIATION_EXECUTION_TIMEOUT_SECONDS",
            "remediation_execution_seconds",
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            timeouts[
                key
            ] = parse_float(
                raw,
                name=env_name,
                minimum=0.001,
            )

    # ------------------------------------------------------
    # Feature gates
    # ------------------------------------------------------

    features: Dict[
        str,
        Any,
    ] = {}

    feature_environment_names = {
        "ENABLE_INFERENCE": (
            "inference_enabled"
        ),

        "ENABLE_ACCELERATOR_OPTIMIZATION": (
            "accelerator_optimization_enabled"
        ),

        "ENABLE_MODEL_LOAD_OPERATIONS": (
            "model_load_operations_enabled"
        ),

        "ENABLE_INCIDENT_CREATE": (
            "incident_create_enabled"
        ),

        "ENABLE_DECISION_EVALUATION": (
            "decision_evaluation_enabled"
        ),

        "ENABLE_REMEDIATION_PLAN_CREATION": (
            "remediation_plan_creation_enabled"
        ),

        "ENABLE_REMEDIATION_APPROVAL": (
            "remediation_approval_enabled"
        ),

        "ENABLE_REMEDIATION_EXECUTION": (
            "remediation_execution_enabled"
        ),

        "ENABLE_LIVE_REMEDIATION": (
            "remediation_live_execution_enabled"
        ),
    }

    for env_name, key in (
        feature_environment_names.
        items()
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            features[
                key
            ] = parse_bool(
                raw,
                name=env_name,
            )

    # ------------------------------------------------------
    # Observability
    # ------------------------------------------------------

    observability: Dict[
        str,
        Any,
    ] = {}

    raw = _environment_value(
        env,
        prefix=prefix,
        name="LOG_LEVEL",
    )

    if raw is not None:
        observability[
            "log_level"
        ] = raw.lower()

    for env_name, key in (
        (
            "ACCESS_LOG_ENABLED",
            "access_log_enabled",
        ),
        (
            "METRICS_ENABLED",
            "metrics_enabled",
        ),
        (
            "SERVER_TIMING_ENABLED",
            "server_timing_enabled",
        ),
    ):
        raw = _environment_value(
            env,
            prefix=prefix,
            name=env_name,
        )

        if raw is not None:
            observability[
                key
            ] = parse_bool(
                raw,
                name=env_name,
            )

    mapping: Dict[
        str,
        Any,
    ] = {
        **top,
    }

    for key, value in (
        (
            "server",
            server,
        ),
        (
            "documentation",
            documentation,
        ),
        (
            "cors",
            cors,
        ),
        (
            "transport_security",
            transport,
        ),
        (
            "limits",
            limits,
        ),
        (
            "timeouts",
            timeouts,
        ),
        (
            "features",
            features,
        ),
        (
            "observability",
            observability,
        ),
    ):
        if value:
            mapping[
                key
            ] = value

    return from_mapping(
        mapping,
        base=config,
    )


# ------------------------------------------------------------------
# Production helper
# ------------------------------------------------------------------

def production_config(
    *,
    trusted_hosts: Iterable[
        str
    ],
    cors_origins: Iterable[
        str
    ] = (),
    enable_docs: bool = False,
) -> APIConfig:
    """
    Create a restrictive production baseline.

    Live remediation remains disabled.
    """

    hosts = tuple(
        validate_host_pattern(
            host
        )
        for host
        in trusted_hosts
    )

    origins = tuple(
        validate_origin(
            origin
        )
        for origin
        in cors_origins
    )

    config = APIConfig(
        environment=(
            APIEnvironment.PRODUCTION.value
        ),

        server=APIServerConfig(
            host="0.0.0.0",
            port=DEFAULT_PORT,
            workers=1,
        ),

        documentation=(
            APIDocumentationConfig(
                enabled=enable_docs,
                include_internal_routes=False,
            )
        ),

        cors=APICORSConfig(
            enabled=bool(
                origins
            ),
            allowed_origins=origins,
            allow_credentials=False,
        ),

        transport_security=(
            APITransportSecurityConfig(
                trusted_hosts=hosts,
                enforce_https=False,
                hsts_enabled=False,
            )
        ),

        features=APIFeatureConfig(
            health_enabled=True,
            inference_enabled=True,
            accelerator_read_enabled=True,
            accelerator_optimization_enabled=True,
            model_read_enabled=True,
            model_load_operations_enabled=True,
            incident_read_enabled=True,
            incident_create_enabled=False,
            decision_read_enabled=True,
            decision_evaluation_enabled=False,
            remediation_read_enabled=True,
            remediation_plan_creation_enabled=False,
            remediation_approval_enabled=False,
            remediation_execution_enabled=False,
            remediation_live_execution_enabled=False,
        ),

        observability=(
            APIObservabilityConfig(
                log_level=(
                    LogLevel.INFO.value
                ),
                access_log_enabled=True,
                metrics_enabled=True,
                include_request_body_in_logs=False,
                include_response_body_in_logs=False,
            )
        ),
    )

    config.validate()

    return config


# ------------------------------------------------------------------
# Safe summary
# ------------------------------------------------------------------

def safe_config_summary(
    config: APIConfig,
) -> Dict[str, Any]:
    """
    Return operational API configuration without
    exposing arbitrary user-supplied attributes.
    """

    config.validate()

    return {
        "config_version": (
            API_CONFIG_VERSION
        ),

        "enabled": config.enabled,

        "service_name": (
            config.service_name
        ),

        "environment": (
            config.environment
        ),

        "versioned_prefix": (
            config.versioned_prefix
        ),

        "server": {
            "host": config.server.host,

            "port": config.server.port,

            "workers": (
                config.server.workers
            ),

            "proxy_headers": (
                config.server.
                proxy_headers
            ),
        },

        "documentation": (
            config.documentation.
            to_dict()
        ),

        "cors": {
            "enabled": (
                config.cors.enabled
            ),

            "origin_count": len(
                config.cors.
                allowed_origins
            ),

            "allow_credentials": (
                config.cors.
                allow_credentials
            ),
        },

        "transport_security": {
            "trusted_host_count": len(
                config.
                transport_security.
                trusted_hosts
            ),

            "enforce_https": (
                config.
                transport_security.
                enforce_https
            ),

            "hsts_enabled": (
                config.
                transport_security.
                hsts_enabled
            ),
        },

        "limits": (
            config.limits.to_dict()
        ),

        "timeouts": (
            config.timeouts.to_dict()
        ),

        "features": (
            config.features.to_dict()
        ),

        "observability": (
            config.observability.
            to_dict()
        ),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "API_CONFIG_VERSION",
    "DEFAULT_ENV_PREFIX",

    "DEFAULT_SERVICE_NAME",
    "DEFAULT_API_PREFIX",
    "DEFAULT_API_VERSION",
    "DEFAULT_HOST",
    "DEFAULT_PORT",

    "APIEnvironment",
    "LogLevel",

    "parse_bool",
    "parse_int",
    "parse_float",
    "parse_csv",
    "parse_json_mapping",

    "normalize_path_prefix",
    "validate_origin",
    "validate_host_pattern",

    "APIServerConfig",
    "APIDocumentationConfig",
    "APICORSConfig",
    "APITransportSecurityConfig",
    "APIRequestLimitsConfig",
    "APITimeoutConfig",
    "APIFeatureConfig",
    "APIObservabilityConfig",

    "APIConfig",

    "from_mapping",
    "from_environment",

    "production_config",

    "safe_config_summary",
]
