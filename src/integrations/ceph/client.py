from __future__ import annotations

"""
Read-only Ceph Dashboard REST API client for OSHA.

The first implementation exposes observation
operations only.

No Ceph cluster mutation is implemented here.
"""

import json
import ssl

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from urllib.error import (
    HTTPError,
    URLError,
)

from urllib.request import (
    Request,
    build_opener,
    HTTPSHandler,
)

from src.integrations.ceph.config import (
    CephConfig,
)

from src.integrations.errors import (
    IntegrationAuthenticationError,
    IntegrationCapabilityError,
    IntegrationConnectionError,
    IntegrationOperationError,
    IntegrationResourceNotFoundError,
    IntegrationUnavailableError,
    wrap_provider_error,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Client error
# ---------------------------------------------------------------

class CephClientError(
    IntegrationOperationError
):
    default_code = (
        "ceph_client_error"
    )


# ---------------------------------------------------------------
# Ceph client
# ---------------------------------------------------------------

class CephClient:
    """
    Minimal provider client for Ceph Dashboard's
    REST API.

    Responsibilities:

      authentication
      JWT retention in-process
      versioned Ceph API GET requests
      bounded provider error translation

    The token is never returned by summary().
    """

    def __init__(
        self,
        *,
        config: CephConfig,
        credential: Optional[
            SecretValue
        ] = None,
    ) -> None:
        self.config = config

        self.config.validate()

        self.credential = (
            credential
        )

        self._token: Optional[
            str
        ] = None

        self._opener: Any = None

        self._connected = False

    # ---------------------------------------------------------
    # Connected
    # ---------------------------------------------------------

    @property
    def connected(
        self,
    ) -> bool:
        return self._connected

    # ---------------------------------------------------------
    # TLS context
    # ---------------------------------------------------------

    def _ssl_context(
        self,
    ) -> ssl.SSLContext:
        if not self.config.verify_tls:
            return (
                ssl.
                _create_unverified_context()
            )

        if self.config.ca_cert_path:
            return ssl.create_default_context(
                cafile=(
                    self.config.
                    ca_cert_path
                )
            )

        return (
            ssl.create_default_context()
        )

    # ---------------------------------------------------------
    # HTTP opener
    # ---------------------------------------------------------

    def _build_opener(
        self,
    ) -> Any:
        return build_opener(
            HTTPSHandler(
                context=(
                    self._ssl_context()
                )
            )
        )

    # ---------------------------------------------------------
    # Connect/authenticate
    # ---------------------------------------------------------

    def connect(
        self,
    ) -> "CephClient":
        if self.connected:
            return self

        if not self.config.auth.username:
            raise (
                IntegrationAuthenticationError(
                    "Ceph API username is "
                    "not configured."
                )
            )

        if self.credential is None:
            raise (
                IntegrationAuthenticationError(
                    "Ceph API credential "
                    "was not resolved."
                )
            )

        self._opener = (
            self._build_opener()
        )

        # -----------------------------------------------------
        # Authentication payload
        # -----------------------------------------------------

        payload = json.dumps(
            {
                "username": (
                    self.config.
                    auth.username
                ),

                # Reveal only at the provider
                # authentication boundary.
                "password": (
                    self.credential.
                    reveal()
                ),
            }
        ).encode(
            "utf-8"
        )

        request = Request(
            (
                self.config.
                normalized_api_url
                + "/auth"
            ),

            data=payload,

            method="POST",

            headers={
                "Accept": (
                    self.config.
                    accept_header
                ),

                "Content-Type": (
                    "application/json"
                ),
            },
        )

        try:
            with self._opener.open(
                request,
                timeout=(
                    self.config.
                    request_timeout_seconds
                ),
            ) as response:
                body = response.read()

        except HTTPError as exc:
            self._clear()

            raise (
                IntegrationAuthenticationError(
                    "Ceph API "
                    "authentication failed.",
                    details={
                        "provider": "ceph",
                        "target_id": (
                            self.config.
                            target_id
                        ),
                        "status_code": (
                            exc.code
                        ),
                    },
                )
            ) from exc

        except (
            URLError,
            TimeoutError,
            OSError,
        ) as exc:
            self._clear()

            raise (
                IntegrationUnavailableError(
                    "Ceph API is "
                    "unavailable.",
                    details={
                        "provider": "ceph",
                        "target_id": (
                            self.config.
                            target_id
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                    },
                )
            ) from exc

        except Exception as exc:
            self._clear()

            raise wrap_provider_error(
                exc,
                provider="ceph",
                target_id=(
                    self.config.target_id
                ),
                message=(
                    "Ceph API connection "
                    "failed."
                ),
                retryable=True,
                code=(
                    "ceph_connection_failed"
                ),
            ) from exc

        # -----------------------------------------------------
        # Parse authentication response
        # -----------------------------------------------------

        try:
            result = json.loads(
                body.decode(
                    "utf-8"
                )
            )

        except Exception as exc:
            self._clear()

            raise (
                IntegrationAuthenticationError(
                    "Ceph API returned an "
                    "invalid authentication "
                    "response."
                )
            ) from exc

        token = (
            result.get(
                "token"
            )
            if isinstance(
                result,
                Mapping,
            )
            else None
        )

        if not token:
            self._clear()

            raise (
                IntegrationAuthenticationError(
                    "Ceph API did not "
                    "return an "
                    "authentication token."
                )
            )

        self._token = str(
            token
        )

        self._connected = True

        return self

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(
        self,
    ) -> None:
        self._clear()

    def _clear(
        self,
    ) -> None:
        self._connected = False
        self._token = None
        self._opener = None

    # ---------------------------------------------------------
    # Request headers
    # ---------------------------------------------------------

    def _headers(
        self,
    ) -> Dict[str, str]:
        self._require_connected()

        return {
            "Accept": (
                self.config.
                accept_header
            ),

            "Authorization": (
                "Bearer "
                + str(
                    self._token
                )
            ),
        }

    # ---------------------------------------------------------
    # Generic GET
    # ---------------------------------------------------------

    def _get(
        self,
        path: str,
    ) -> Any:
        """
        Perform one authenticated Ceph API GET.

        path is internal to the provider client
        and is not accepted directly from an
        arbitrary integration operation.
        """

        self._require_connected()

        normalized = path.strip()

        if not normalized.startswith(
            "/"
        ):
            normalized = (
                "/"
                + normalized
            )

        request = Request(
            (
                self.config.
                normalized_api_url
                + normalized
            ),

            method="GET",

            headers=(
                self._headers()
            ),
        )

        try:
            with self._opener.open(
                request,
                timeout=(
                    self.config.
                    request_timeout_seconds
                ),
            ) as response:
                body = response.read()

        except HTTPError as exc:
            if exc.code == 404:
                raise (
                    IntegrationResourceNotFoundError(
                        "Ceph API resource "
                        "was not found."
                    )
                ) from exc

            if exc.code in {
                401,
                403,
            }:
                raise (
                    IntegrationAuthenticationError(
                        "Ceph API rejected "
                        "the request.",
                        details={
                            "status_code": (
                                exc.code
                            ),
                            "target_id": (
                                self.config.
                                target_id
                            ),
                        },
                    )
                ) from exc

            raise (
                CephClientError(
                    "Ceph API request "
                    "failed.",
                    details={
                        "status_code": (
                            exc.code
                        ),
                        "path": (
                            normalized
                        ),
                    },
                )
            ) from exc

        except (
            URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise (
                IntegrationUnavailableError(
                    "Ceph API request "
                    "could not reach the "
                     "target.",
                    etails={
                        "target_id": (
                            self.config.
                            target_id
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                    },
                )
            ) from exc

        try:
            if not body:
                return None

            return json.loads(
                body.decode(
                    "utf-8"
                )
            )

        except Exception as exc:
            raise (
                CephClientError(
                    "Ceph API returned "
                    "invalid JSON."
                )
            ) from exc

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping(
        self,
    ) -> bool:
        """
        Perform a small authenticated API read.
        """

        if not self.connected:
            return False

        try:
            self._get(
                "/summary"
            )

            return True

        except Exception:
            return False

    # ---------------------------------------------------------
    # Cluster summary
    # ---------------------------------------------------------

    def cluster_summary(
        self,
    ) -> Dict[str, Any]:
        value = self._get(
            "/summary"
        )

        return self._mapping(
            value,
            "Ceph cluster summary",
        )

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health(
        self,
    ) -> Any:
        self._require_feature(
            self.config.enable_health,
            "health",
        )

        return self._get(
            "/health/full"
        )

    # ---------------------------------------------------------
    # OSDs
    # ---------------------------------------------------------

    def list_osds(
        self,
    ) -> List[Any]:
        self._require_feature(
            self.config.enable_osd,
            "osd",
        )

        return self._list(
            self._get(
                "/osd"
            ),
            "Ceph OSD list",
        )

    def get_osd(
        self,
        osd_id: str,
    ) -> Any:
        self._require_feature(
            self.config.enable_osd,
            "osd",
        )

        osd_id = self._identifier(
            osd_id,
            "osd_id",
        )

        return self._get(
            "/osd/"
            + osd_id
        )

    # ---------------------------------------------------------
    # Pools
    # ---------------------------------------------------------

    def list_pools(
        self,
    ) -> List[Any]:
        self._require_feature(
            self.config.enable_pools,
            "pools",
        )

        return self._list(
            self._get(
                "/pool"
            ),
            "Ceph pool list",
        )

    def get_pool(
        self,
        pool_name: str,
    ) -> Any:
        self._require_feature(
            self.config.enable_pools,
            "pools",
        )

        pool_name = self._identifier(
            pool_name,
            "pool_name",
        )

        return self._get(
            "/pool/"
            + pool_name
        )

    # ---------------------------------------------------------
    # Hosts
    # ---------------------------------------------------------

    def list_hosts(
        self,
    ) -> List[Any]:
        self._require_feature(
            self.config.enable_hosts,
            "hosts",
        )

        return self._list(
            self._get(
                "/host"
            ),
            "Ceph host list",
        )

    # ---------------------------------------------------------
    # Monitors
    # ---------------------------------------------------------

    def monitors(
        self,
    ) -> Any:
        self._require_feature(
            self.config.
            enable_monitors,
            "monitors",
        )

        return self._get(
            "/monitor"
        )

    # ---------------------------------------------------------
    # Services
    # ---------------------------------------------------------

    def list_services(
        self,
    ) -> List[Any]:
        self._require_feature(
            self.config.
            enable_services,
            "services",
        )

        return self._list(
            self._get(
                "/service"
            ),
            "Ceph service list",
        )

    # ---------------------------------------------------------
    # Feature guard
    # ---------------------------------------------------------

    def _require_feature(
        self,
        enabled: bool,
        feature: str,
    ) -> None:
        self._require_connected()

        if not enabled:
            raise (
                IntegrationCapabilityError(
                    "Ceph integration "
                    "feature is disabled: "
                    f"{feature}"
                )
            )

    # ---------------------------------------------------------
    # Connection guard
    # ---------------------------------------------------------

    def _require_connected(
        self,
    ) -> None:
        if (
            not self.connected
            or self._token is None
            or self._opener is None
        ):
            raise (
                IntegrationConnectionError(
                    "Ceph client is not "
                    "connected."
                )
            )

    # ---------------------------------------------------------
    # Identifier guard
    # ---------------------------------------------------------

    @staticmethod
    def _identifier(
        value: Any,
        name: str,
    ) -> str:
        """
        Keep provider resource identifiers out of
        arbitrary URL-path territory.
        """

        text = str(
            value
        ).strip()

        if not text:
            raise (
                IntegrationOperationError(
                    f"{name} is required."
                )
            )

        if any(
            character
            in text
            for character
            in (
                "/",
                "\\",
                "?",
                "#",
            )
        ):
            raise (
                IntegrationOperationError(
                    f"{name} contains "
                    "unsupported "
                    "characters."
                )
            )

        return text

    # ---------------------------------------------------------
    # Response guards
    # ---------------------------------------------------------

    @staticmethod
    def _list(
        value: Any,
        name: str,
    ) -> List[Any]:
        if isinstance(
            value,
            list,
        ):
            return value

        raise (
            CephClientError(
                f"{name} response was "
                "not a list."
            )
        )

    @staticmethod
    def _mapping(
        value: Any,
        name: str,
    ) -> Dict[str, Any]:
        if isinstance(
            value,
            Mapping,
        ):
            return dict(
                value
            )

        raise (
            CephClientError(
                f"{name} response was "
                "not an object."
            )
        )

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connected": (
                self.connected
            ),

            "target_id": (
                self.config.target_id
            ),

            "api_url": (
                self.config.
                normalized_api_url
            ),

            "api_version": (
                self.config.api_version
            ),

            "verify_tls": (
                self.config.verify_tls
            ),

            "authenticated": (
                self._token is not None
            ),

            "token_exposed": False,

            "credential_value_exposed": (
                False
            ),

            "mutating_operations_enabled": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "CephClientError",

    "CephClient",
]
