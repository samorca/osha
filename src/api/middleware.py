from __future__ import annotations

"""
Production HTTP middleware for OSHA API.

Responsibilities:

    request ID generation and propagation
    request timing
    request-size enforcement
    query/header/path limits
    process-local concurrency control
    request timeout enforcement
    safe structured access logging
    response security headers
    optional HSTS
    CORS installation
    trusted-host enforcement
    optional HTTPS redirection

Security principles:

    never log request bodies
    never log response bodies
    never log Authorization
    never log Cookie
    never trust arbitrary request IDs
    never trust proxy headers here
    never silently ignore configured limits

This module contains HTTP infrastructure only.
It does not execute OSHA business logic.
"""

import asyncio
import logging
import re
import time

from contextvars import ContextVar

from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from uuid import uuid4

from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from starlette.middleware.httpsredirect import (
    HTTPSRedirectMiddleware,
)

from starlette.middleware.trustedhost import (
    TrustedHostMiddleware,
)

from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)

from src.api.config import (
    APIConfig,
)

from src.api.errors import (
    APIException,
    ConcurrencyLimitError,
    RequestTimeoutAPIError,
    RequestTooLargeError,
    error_response,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(
    __name__
)


# ------------------------------------------------------------------
# Public header names
# ------------------------------------------------------------------

DEFAULT_REQUEST_ID_HEADER = (
    "X-Request-ID"
)

PROCESS_TIME_HEADER = (
    "X-Process-Time"
)

SERVER_TIMING_HEADER = (
    "Server-Timing"
)

RETRY_AFTER_HEADER = (
    "Retry-After"
)


# ------------------------------------------------------------------
# Internal types
# ------------------------------------------------------------------

HeaderList = List[
    Tuple[
        bytes,
        bytes,
    ]
]


# ------------------------------------------------------------------
# Request context variables
# ------------------------------------------------------------------

_request_id_context: ContextVar[
    Optional[str]
] = ContextVar(
    "osha_api_request_id",
    default=None,
)

_request_started_context: ContextVar[
    Optional[float]
] = ContextVar(
    "osha_api_request_started",
    default=None,
)


# ------------------------------------------------------------------
# Request ID validation
# ------------------------------------------------------------------

_REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:/-]{1,128}$"
)


# ------------------------------------------------------------------
# Request context
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RequestContext:
    request_id: str

    started_monotonic: float

    method: str

    path: str

    client_host: Optional[str] = None

    @property
    def duration_ms(
        self,
    ) -> float:
        return max(
            0.0,
            (
                time.perf_counter()
                - self.started_monotonic
            )
            * 1000.0,
        )


# ------------------------------------------------------------------
# Current-request helpers
# ------------------------------------------------------------------

def current_request_id() -> Optional[str]:
    return _request_id_context.get()


def current_request_started() -> Optional[float]:
    return _request_started_context.get()


def current_request_duration_ms() -> Optional[float]:
    started = current_request_started()

    if started is None:
        return None

    return max(
        0.0,
        (
            time.perf_counter()
            - started
        )
        * 1000.0,
    )


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def is_http_scope(
    scope: Scope,
) -> bool:
    return (
        scope.get(
            "type"
        )
        == "http"
    )


def request_method(
    scope: Scope,
) -> str:
    return str(
        scope.get(
            "method",
            "UNKNOWN",
        )
    ).upper()


def request_path(
    scope: Scope,
) -> str:
    return str(
        scope.get(
            "path",
            "/",
        )
    )


def client_host(
    scope: Scope,
) -> Optional[str]:
    value = scope.get(
        "client"
    )

    if not value:
        return None

    try:
        return str(
            value[0]
        )

    except (
        IndexError,
        TypeError,
    ):
        return None


def scope_headers(
    scope: Scope,
) -> HeaderList:
    return list(
        scope.get(
            "headers",
            (),
        )
    )


def decode_header_value(
    value: bytes,
) -> str:
    return value.decode(
        "latin-1",
        errors="replace",
    )


def encode_header_value(
    value: str,
) -> bytes:
    return str(
        value
    ).encode(
        "latin-1",
        errors="ignore",
    )


def normalized_header_name(
    value: str,
) -> bytes:
    return str(
        value
    ).strip().lower().encode(
        "latin-1"
    )


def get_header(
    scope: Scope,
    name: str,
) -> Optional[str]:
    target = normalized_header_name(
        name
    )

    for key, value in scope_headers(
        scope
    ):
        if key.lower() == target:
            return decode_header_value(
                value
            )

    return None


def set_response_header(
    headers: HeaderList,
    name: str,
    value: str,
) -> HeaderList:
    target = normalized_header_name(
        name
    )

    result = [
        (
            key,
            current_value,
        )
        for key, current_value
        in headers
        if key.lower() != target
    ]

    result.append(
        (
            target,
            encode_header_value(
                value
            ),
        )
    )

    return result


def remove_response_header(
    headers: HeaderList,
    name: str,
) -> HeaderList:
    target = normalized_header_name(
        name
    )

    return [
        (
            key,
            value,
        )
        for key, value
        in headers
        if key.lower() != target
    ]


def response_header_value(
    headers: Sequence[
        Tuple[
            bytes,
            bytes,
        ]
    ],
    name: str,
) -> Optional[str]:
    target = normalized_header_name(
        name
    )

    for key, value in headers:
        if key.lower() == target:
            return decode_header_value(
                value
            )

    return None


def set_scope_state(
    scope: Scope,
    name: str,
    value: Any,
) -> None:
    state = scope.setdefault(
        "state",
        {}
    )

    state[
        name
    ] = value


# ------------------------------------------------------------------
# Request ID
# ------------------------------------------------------------------

def valid_request_id(
    value: Optional[str],
) -> bool:
    if value is None:
        return False

    text = str(
        value
    ).strip()

    return bool(
        _REQUEST_ID_PATTERN.fullmatch(
            text
        )
    )


def resolve_request_id(
    scope: Scope,
    *,
    header_name: str,
) -> str:
    supplied = get_header(
        scope,
        header_name,
    )

    if valid_request_id(
        supplied
    ):
        return str(
            supplied
        ).strip()

    return str(
        uuid4()
    )


# ------------------------------------------------------------------
# Scope context helpers
# ------------------------------------------------------------------

def get_scope_request_id(
    scope: Scope,
) -> Optional[str]:
    state = scope.get(
        "state",
        {},
    )

    value = state.get(
        "request_id"
    )

    if value is not None:
        return str(
            value
        )

    return current_request_id()


def get_scope_duration_ms(
    scope: Scope,
) -> Optional[float]:
    state = scope.get(
        "state",
        {},
    )

    started = state.get(
        "request_started_monotonic"
    )

    if started is None:
        return current_request_duration_ms()

    try:
        started_float = float(
            started
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    return max(
        0.0,
        (
            time.perf_counter()
            - started_float
        )
        * 1000.0,
    )


# ------------------------------------------------------------------
# API exception response
# ------------------------------------------------------------------

async def send_api_exception(
    *,
    error: BaseException,
    scope: Scope,
    send: Send,
    request_id: Optional[str],
    request_id_header: str = (
        DEFAULT_REQUEST_ID_HEADER
    ),
    duration_ms: Optional[
        float
    ] = None,
) -> None:
    """
    Send one standardized JSON API error directly
    from ASGI middleware.
    """

    (
        status_code,
        response,
        extra_headers,
    ) = error_response(
        error,
        request_id=request_id,
        duration_ms=duration_ms,
    )

    body = response.model_dump_json(
        exclude_none=True,
    ).encode(
        "utf-8"
    )

    headers: HeaderList = [
        (
            b"content-type",
            b"application/json; charset=utf-8",
        ),
        (
            b"content-length",
            str(
                len(
                    body
                )
            ).encode(
                "ascii"
            ),
        ),
        (
            b"cache-control",
            b"no-store",
        ),
    ]

    for name, value in (
        extra_headers.items()
    ):
        headers = set_response_header(
            headers,
            name,
            value,
        )

    if request_id:
        headers = set_response_header(
            headers,
            request_id_header,
            request_id,
        )

    await send(
        {
            "type": (
                "http.response.start"
            ),

            "status": (
                status_code
            ),

            "headers": headers,
        }
    )

    await send(
        {
            "type": (
                "http.response.body"
            ),

            "body": body,

            "more_body": False,
        }
    )


# ------------------------------------------------------------------
# Request context middleware
# ------------------------------------------------------------------

class RequestContextMiddleware:
    """
    Establish request identity and timing.

    Exposes:

        request.state.request_id
        request.state.request_context
        current_request_id()
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        header_name: str = (
            DEFAULT_REQUEST_ID_HEADER
        ),
        server_timing_enabled: bool = False,
    ) -> None:
        self.app = app

        self.header_name = (
            str(
                header_name
            ).strip()
            or DEFAULT_REQUEST_ID_HEADER
        )

        self.server_timing_enabled = bool(
            server_timing_enabled
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not is_http_scope(
            scope
        ):
            await self.app(
                scope,
                receive,
                send,
            )

            return

        started = time.perf_counter()

        request_id = resolve_request_id(
            scope,
            header_name=(
                self.header_name
            ),
        )

        context = RequestContext(
            request_id=request_id,

            started_monotonic=started,

            method=request_method(
                scope
            ),

            path=request_path(
                scope
            ),

            client_host=client_host(
                scope
            ),
        )

        set_scope_state(
            scope,
            "request_id",
            request_id,
        )

        set_scope_state(
            scope,
            "request_started_monotonic",
            started,
        )

        set_scope_state(
            scope,
            "request_context",
            context,
        )

        request_id_token = (
            _request_id_context.set(
                request_id
            )
        )

        started_token = (
            _request_started_context.set(
                started
            )
        )

        async def send_with_context(
            message: Message,
        ) -> None:
            if (
                message.get(
                    "type"
                )
                == "http.response.start"
            ):
                headers = list(
                    message.get(
                        "headers",
                        (),
                    )
                )

                duration_ms = max(
                    0.0,
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000.0,
                )

                headers = set_response_header(
                    headers,
                    self.header_name,
                    request_id,
                )

                headers = set_response_header(
                    headers,
                    PROCESS_TIME_HEADER,
                    (
                        f"{duration_ms / 1000.0:.6f}"
                    ),
                )

                if self.server_timing_enabled:
                    headers = set_response_header(
                        headers,
                        SERVER_TIMING_HEADER,
                        (
                            "app;dur="
                            f"{duration_ms:.3f}"
                        ),
                    )

                message = {
                    **message,
                    "headers": headers,
                }

            await send(
                message
            )

        try:
            await self.app(
                scope,
                receive,
                send_with_context,
            )

        finally:
            _request_id_context.reset(
                request_id_token
            )

            _request_started_context.reset(
                started_token
            )


# ------------------------------------------------------------------
# Request limits middleware
# ------------------------------------------------------------------

class RequestLimitsMiddleware:
    """
    Enforce HTTP envelope limits.

    Content-Length is checked first where present.

    Streamed body bytes are counted independently,
    preventing clients from bypassing body limits
    by omitting Content-Length.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        maximum_body_bytes: int,
        maximum_query_parameters: int,
        maximum_header_count: int,
        maximum_header_bytes: int,
        maximum_path_length: int,
        request_id_header: str = (
            DEFAULT_REQUEST_ID_HEADER
        ),
    ) -> None:
        self.app = app

        self.maximum_body_bytes = int(
            maximum_body_bytes
        )

        self.maximum_query_parameters = int(
            maximum_query_parameters
        )

        self.maximum_header_count = int(
            maximum_header_count
        )

        self.maximum_header_bytes = int(
            maximum_header_bytes
        )

        self.maximum_path_length = int(
            maximum_path_length
        )

        self.request_id_header = (
            request_id_header
        )

        for name, value in (
            (
                "maximum_body_bytes",
                self.maximum_body_bytes,
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
        ):
            if value < 1:
                raise ValueError(
                    f"{name} must be >= 1"
                )

    def _validate_envelope(
        self,
        scope: Scope,
    ) -> None:
        path = request_path(
            scope
        )

        path_bytes = len(
            path.encode(
                "utf-8"
            )
        )

        if (
            path_bytes
            > self.maximum_path_length
        ):
            raise RequestTooLargeError(
                "The request path is too long.",
                details={
                    "limit": (
                        self.maximum_path_length
                    ),

                    "current": (
                        path_bytes
                    ),
                },
            )

        headers = scope_headers(
            scope
        )

        header_count = len(
            headers
        )

        if (
            header_count
            > self.maximum_header_count
        ):
            raise RequestTooLargeError(
                "The request contains too "
                "many headers.",
                details={
                    "limit": (
                        self.maximum_header_count
                    ),

                    "current": (
                        header_count
                    ),
                },
            )

        header_bytes = sum(
            len(
                key
            )
            + len(
                value
            )
            + 4
            for key, value
            in headers
        )

        if (
            header_bytes
            > self.maximum_header_bytes
        ):
            raise RequestTooLargeError(
                "The request headers are "
                "too large.",
                details={
                    "limit": (
                        self.maximum_header_bytes
                    ),

                    "current": (
                        header_bytes
                    ),
                },
            )

        query_string = scope.get(
            "query_string",
            b"",
        )

        if query_string:
            query_count = (
                query_string.count(
                    b"&"
                )
                + 1
            )

        else:
            query_count = 0

        if (
            query_count
            > self.maximum_query_parameters
        ):
            raise RequestTooLargeError(
                "The request contains too "
                "many query parameters.",
                details={
                    "limit": (
                        self.
                        maximum_query_parameters
                    ),

                    "current": (
                        query_count
                    ),
                },
            )

        content_length = get_header(
            scope,
            "content-length",
        )

        if content_length is None:
            return

        try:
            content_length_value = int(
                content_length
            )

        except ValueError as exc:
            raise RequestTooLargeError(
                "The Content-Length header "
                "is invalid."
            ) from exc

        if content_length_value < 0:
            raise RequestTooLargeError(
                "The Content-Length header "
                "is invalid."
            )

        if (
            content_length_value
            > self.maximum_body_bytes
        ):
            raise RequestTooLargeError(
                details={
                    "limit": (
                        self.maximum_body_bytes
                    ),

                    "current": (
                        content_length_value
                    ),
                },
            )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not is_http_scope(
            scope
        ):
            await self.app(
                scope,
                receive,
                send,
            )

            return

        try:
            self._validate_envelope(
                scope
            )

        except APIException as exc:
            await send_api_exception(
                error=exc,

                scope=scope,

                send=send,

                request_id=(
                    get_scope_request_id(
                        scope
                    )
                ),

                request_id_header=(
                    self.request_id_header
                ),

                duration_ms=(
                    get_scope_duration_ms(
                        scope
                    )
                ),
            )

            return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes

            message = await receive()

            if (
                message.get(
                    "type"
                )
                == "http.request"
            ):
                body = message.get(
                    "body",
                    b"",
                )

                received_bytes += len(
                    body
                )

                if (
                    received_bytes
                    > self.maximum_body_bytes
                ):
                    raise RequestTooLargeError(
                        details={
                            "limit": (
                                self.maximum_body_bytes
                            ),

                            "current": (
                                received_bytes
                            ),
                        },
                    )

            return message

        try:
            await self.app(
                scope,
                limited_receive,
                send,
            )

        except RequestTooLargeError as exc:
            await send_api_exception(
                error=exc,

                scope=scope,

                send=send,

                request_id=(
                    get_scope_request_id(
                        scope
                    )
                ),

                request_id_header=(
                    self.request_id_header
                ),

                duration_ms=(
                    get_scope_duration_ms(
                        scope
                    )
                ),
            )


# ------------------------------------------------------------------
# Concurrency limit
# ------------------------------------------------------------------

class ConcurrencyLimitMiddleware:
    """
    Limit simultaneous requests per worker.

    This is deliberately process-local.

    A multi-worker deployment needs an external
    admission/rate-limiting layer for global
    limits.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        maximum_concurrent_requests: int,
        retry_after_seconds: int = 1,
        request_id_header: str = (
            DEFAULT_REQUEST_ID_HEADER
        ),
    ) -> None:
        self.app = app

        self.maximum = int(
            maximum_concurrent_requests
        )

        if self.maximum < 1:
            raise ValueError(
                "maximum_concurrent_requests "
                "must be >= 1"
            )

        self.retry_after_seconds = max(
            1,
            int(
                retry_after_seconds
            ),
        )

        self.request_id_header = (
            request_id_header
        )

        self._active = 0

        self._lock = asyncio.Lock()

    async def _try_acquire(
        self,
    ) -> bool:
        async with self._lock:
            if (
                self._active
                >= self.maximum
            ):
                return False

            self._active += 1

            return True

    async def _release(
        self,
    ) -> None:
        async with self._lock:
            self._active = max(
                0,
                self._active - 1,
            )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not is_http_scope(
            scope
        ):
            await self.app(
                scope,
                receive,
                send,
            )

            return

        acquired = await self._try_acquire()

        if not acquired:
            error = ConcurrencyLimitError(
                details={
                    "limit": self.maximum,

                    "retry_after_seconds": (
                        self.
                        retry_after_seconds
                    ),
                },

                headers={
                    RETRY_AFTER_HEADER: str(
                        self.
                        retry_after_seconds
                    ),
                },
            )

            await send_api_exception(
                error=error,

                scope=scope,

                send=send,

                request_id=(
                    get_scope_request_id(
                        scope
                    )
                ),

                request_id_header=(
                    self.request_id_header
                ),

                duration_ms=(
                    get_scope_duration_ms(
                        scope
                    )
                ),
            )

            return

        try:
            await self.app(
                scope,
                receive,
                send,
            )

        finally:
            await self._release()


# ------------------------------------------------------------------
# Timeout middleware
# ------------------------------------------------------------------

class RequestTimeoutMiddleware:
    """
    Enforce a global request timeout.

    Service-specific inference/model/remediation
    timeouts still exist and remain more specific.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        timeout_seconds: float,
        request_id_header: str = (
            DEFAULT_REQUEST_ID_HEADER
        ),
    ) -> None:
        self.app = app

        self.timeout_seconds = float(
            timeout_seconds
        )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be > 0"
            )

        self.request_id_header = (
            request_id_header
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not is_http_scope(
            scope
        ):
            await self.app(
                scope,
                receive,
                send,
            )

            return

        response_started = False

        async def tracked_send(
            message: Message,
        ) -> None:
            nonlocal response_started

            if (
                message.get(
                    "type"
                )
                == "http.response.start"
            ):
                response_started = True

            await send(
                message
            )

        try:
            async with asyncio.timeout(
                self.timeout_seconds
            ):
                await self.app(
                    scope,
                    receive,
                    tracked_send,
                )

        except TimeoutError:
            if response_started:
                raise

            error = RequestTimeoutAPIError(
                details={
                    "operation": (
                        "http_request"
                    ),
                },
            )

            await send_api_exception(
                error=error,

                scope=scope,

                send=send,

                request_id=(
                    get_scope_request_id(
                        scope
                    )
                ),

                request_id_header=(
                    self.request_id_header
                ),

                duration_ms=(
                    get_scope_duration_ms(
                        scope
                    )
                ),
            )


# ------------------------------------------------------------------
# Security headers
# ------------------------------------------------------------------

class SecurityHeadersMiddleware:
    """
    Add conservative security headers.

    HSTS is emitted only for HTTPS requests and
    only when explicitly enabled.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        hsts_enabled: bool = False,
        hsts_max_age_seconds: int = 31_536_000,
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False,
    ) -> None:
        self.app = app

        self.hsts_enabled = bool(
            hsts_enabled
        )

        self.hsts_max_age_seconds = int(
            hsts_max_age_seconds
        )

        if self.hsts_max_age_seconds < 0:
            raise ValueError(
                "hsts_max_age_seconds "
                "must be >= 0"
            )

        self.hsts_include_subdomains = bool(
            hsts_include_subdomains
        )

        self.hsts_preload = bool(
            hsts_preload
        )

    def _hsts_value(
        self,
    ) -> str:
        values = [
            (
                "max-age="
                + str(
                    self.
                    hsts_max_age_seconds
                )
            )
        ]

        if self.hsts_include_subdomains:
            values.append(
                "includeSubDomains"
            )

        if self.hsts_preload:
            values.append(
                "preload"
            )

        return "; ".join(
            values
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not is_http_scope(
            scope
        ):
            await self.app(
                scope,
                receive,
                send,
            )

            return

        async def secure_send(
            message: Message,
        ) -> None:
            if (
                message.get(
                    "type"
                )
                == "http.response.start"
            ):
                headers = list(
                    message.get(
                        "headers",
                        (),
                    )
                )

                # Do not advertise application
                # server implementation.
                headers = (
                    remove_response_header(
                        headers,
                        "server",
                    )
                )

                security_headers = {
                    "X-Content-Type-Options": (
                        "nosniff"
                    ),

                    "X-Frame-Options": (
                        "DENY"
                    ),

                    "Referrer-Policy": (
                        "no-referrer"
                    ),

                    "Permissions-Policy": (
                        "camera=(), "
                        "microphone=(), "
                        "geolocation=(), "
                        "payment=(), "
                        "usb=()"
                    ),

                    "Cross-Origin-Opener-Policy": (
                        "same-origin"
                    ),

                    "Cross-Origin-Resource-Policy": (
                        "same-origin"
                    ),
                }

                for name, value in (
                    security_headers.items()
                ):
                    if (
                        response_header_value(
                            headers,
                            name,
                        )
                        is None
                    ):
                        headers = (
                            set_response_header(
                                headers,
                                name,
                                value,
                            )
                        )

                path = request_path(
                    scope
                )

                status_code = int(
                    message.get(
                        "status",
                        200,
                    )
                )

                # Health and failure responses
                # should not be cached.
                if (
                    path.startswith(
                        "/health"
                    )
                    or status_code >= 400
                ):
                    headers = (
                        set_response_header(
                            headers,
                            "Cache-Control",
                            "no-store",
                        )
                    )

                if (
                    self.hsts_enabled
                    and scope.get(
                        "scheme"
                    )
                    == "https"
                ):
                    headers = (
                        set_response_header(
                            headers,
                            (
                                "Strict-Transport-"
                                "Security"
                            ),
                            self._hsts_value(),
                        )
                    )

                message = {
                    **message,
                    "headers": headers,
                }

            await send(
                message
            )

        await self.app(
            scope,
            receive,
            secure_send,
        )


# ------------------------------------------------------------------
# Access logging
# ------------------------------------------------------------------

class AccessLogMiddleware:
    """
    Structured request completion logs.

    Explicitly excluded:

        request body
        response body
        query values
        Authorization header
        cookies
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool = True,
        include_health_requests: bool = False,
    ) -> None:
        self.app = app

        self.enabled = bool(
            enabled
        )

        self.include_health_requests = bool(
            include_health_requests
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            not is_http_scope(
                scope
            )
            or not self.enabled
        ):
            await self.app(
                scope,
                receive,
                send,
            )

            return

        started = time.perf_counter()

        status_code = 500
        response_bytes = 0
        error_type = None

        async def logging_send(
            message: Message,
        ) -> None:
            nonlocal status_code
            nonlocal response_bytes

            if (
                message.get(
                    "type"
                )
                == "http.response.start"
            ):
                status_code = int(
                    message.get(
                        "status",
                        500,
                    )
                )

            elif (
                message.get(
                    "type"
                )
                == "http.response.body"
            ):
                response_bytes += len(
                    message.get(
                        "body",
                        b"",
                    )
                )

            await send(
                message
            )

        try:
            await self.app(
                scope,
                receive,
                logging_send,
            )

        except Exception as exc:
            error_type = type(
                exc
            ).__name__

            raise

        finally:
            path = request_path(
                scope
            )

            health_request = (
                path == "/health"
                or path.startswith(
                    "/health/"
                )
            )

            if (
                not health_request
                or self.
                include_health_requests
            ):
                duration_ms = max(
                    0.0,
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000.0,
                )

                fields: Dict[
                    str,
                    Any,
                ] = {
                    "request_id": (
                        get_scope_request_id(
                            scope
                        )
                    ),

                    "method": (
                        request_method(
                            scope
                        )
                    ),

                    "path": path,

                    "status_code": (
                        status_code
                    ),

                    "duration_ms": round(
                        duration_ms,
                        3,
                    ),

                    "response_bytes": (
                        response_bytes
                    ),

                    "client_host": (
                        client_host(
                            scope
                        )
                    ),

                    "http_version": (
                        scope.get(
                            "http_version"
                        )
                    ),

                    "error_type": (
                        error_type
                    ),
                }

                if status_code >= 500:
                    level = logging.ERROR

                elif status_code >= 400:
                    level = logging.WARNING

                else:
                    level = logging.INFO

                logger.log(
                    level,
                    "API request completed",
                    extra={
                        "http_request": (
                            fields
                        ),
                    },
                )


# ------------------------------------------------------------------
# Middleware installation
# ------------------------------------------------------------------

def install_middleware(
    app: FastAPI,
    config: APIConfig,
) -> None:
    """
    Install the OSHA API middleware stack.

    FastAPI/Starlette wraps middleware according
    to registration order, so infrastructure
    middleware is registered deliberately here in
    one location.
    """

    config.validate()

    request_id_header = (
        config.
        observability.
        request_id_header
    )

    # ------------------------------------------------------
    # Request timeout
    # ------------------------------------------------------

    app.add_middleware(
        RequestTimeoutMiddleware,

        timeout_seconds=(
            config.
            timeouts.
            request_seconds
        ),

        request_id_header=(
            request_id_header
        ),
    )

    # ------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------

    app.add_middleware(
        ConcurrencyLimitMiddleware,

        maximum_concurrent_requests=(
            config.
            limits.
            maximum_concurrent_requests
        ),

        retry_after_seconds=1,

        request_id_header=(
            request_id_header
        ),
    )

    # ------------------------------------------------------
    # Request envelope limits
    # ------------------------------------------------------

    app.add_middleware(
        RequestLimitsMiddleware,

        maximum_body_bytes=(
            config.
            limits.
            maximum_body_bytes
        ),

        maximum_query_parameters=(
            config.
            limits.
            maximum_query_parameters
        ),

        maximum_header_count=(
            config.
            limits.
            maximum_header_count
        ),

        maximum_header_bytes=(
            config.
            limits.
            maximum_header_bytes
        ),

        maximum_path_length=(
            config.
            limits.
            maximum_path_length
        ),

        request_id_header=(
            request_id_header
        ),
    )

    # ------------------------------------------------------
    # Security response headers
    # ------------------------------------------------------

    app.add_middleware(
        SecurityHeadersMiddleware,

        hsts_enabled=(
            config.
            transport_security.
            hsts_enabled
        ),

        hsts_max_age_seconds=(
            config.
            transport_security.
            hsts_max_age_seconds
        ),

        hsts_include_subdomains=(
            config.
            transport_security.
            hsts_include_subdomains
        ),

        hsts_preload=(
            config.
            transport_security.
            hsts_preload
        ),
    )

    # ------------------------------------------------------
    # Access logging
    # ------------------------------------------------------

    app.add_middleware(
        AccessLogMiddleware,

        enabled=(
            config.
            observability.
            access_log_enabled
        ),

        include_health_requests=(
            config.
            observability.
            include_health_access_logs
        ),
    )

    # ------------------------------------------------------
    # Request identity/timing
    # ------------------------------------------------------

    app.add_middleware(
        RequestContextMiddleware,

        header_name=(
            request_id_header
        ),

        server_timing_enabled=(
            config.
            observability.
            server_timing_enabled
        ),
    )

    # ------------------------------------------------------
    # CORS
    # ------------------------------------------------------

    if config.cors.enabled:
        app.add_middleware(
            CORSMiddleware,

            allow_origins=list(
                config.
                cors.
                allowed_origins
            ),

            allow_credentials=(
                config.
                cors.
                allow_credentials
            ),

            allow_methods=list(
                config.
                cors.
                allowed_methods
            ),

            allow_headers=list(
                config.
                cors.
                allowed_headers
            ),

            expose_headers=list(
                config.
                cors.
                exposed_headers
            ),

            max_age=(
                config.
                cors.
                max_age_seconds
            ),
        )

    # ------------------------------------------------------
    # HTTPS
    # ------------------------------------------------------

    if (
        config.
        transport_security.
        enforce_https
    ):
        app.add_middleware(
            HTTPSRedirectMiddleware
        )

    # ------------------------------------------------------
    # Trusted host validation
    # ------------------------------------------------------

    app.add_middleware(
        TrustedHostMiddleware,

        allowed_hosts=list(
            config.
            transport_security.
            trusted_hosts
        ),
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "DEFAULT_REQUEST_ID_HEADER",
    "PROCESS_TIME_HEADER",
    "SERVER_TIMING_HEADER",
    "RETRY_AFTER_HEADER",

    "RequestContext",

    "current_request_id",
    "current_request_started",
    "current_request_duration_ms",

    "is_http_scope",

    "request_method",
    "request_path",
    "client_host",

    "scope_headers",
    "get_header",

    "set_response_header",
    "remove_response_header",
    "response_header_value",

    "valid_request_id",
    "resolve_request_id",

    "get_scope_request_id",
    "get_scope_duration_ms",

    "send_api_exception",

    "RequestContextMiddleware",
    "RequestLimitsMiddleware",
    "ConcurrencyLimitMiddleware",
    "RequestTimeoutMiddleware",
    "SecurityHeadersMiddleware",
    "AccessLogMiddleware",

    "install_middleware",
]
