from __future__ import annotations

"""
Thread-safe registry for OSHA infrastructure
integration adapters.
"""

from threading import RLock

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from src.core.validation import (
    environment_id,
    required_string,
)

from src.integrations.base import (
    IntegrationAdapter,
)

from src.integrations.errors import (
    IntegrationConflictError,
    IntegrationRegistryError,
)

from src.integrations.schema import (
    ProviderKind,
)


# ---------------------------------------------------------------
# Registry key
# ---------------------------------------------------------------

RegistryKey = Tuple[
    str,
    str,
    str,
]


def integration_key(
    *,
    environment: str,
    provider: str,
    target_id: str,
) -> RegistryKey:
    """
    Build the canonical registry key:

        environment
        provider
        target_id
    """

    normalized_environment = environment_id(environment)

    normalized_provider = required_string(
        provider,
        name="provider",
        maximum_length=128,
    ).lower()

    if normalized_provider not in {item.value for item in ProviderKind}:
        raise (
            IntegrationRegistryError(
                f"Unknown integration provider: {normalized_provider}"
            )
        )

    normalized_target = required_string(
        target_id,
        name="target_id",
        maximum_length=512,
    )

    return (
        normalized_environment,
        normalized_provider,
        normalized_target,
    )


# ---------------------------------------------------------------
# Integration registry
# ---------------------------------------------------------------


class IntegrationRegistry:
    """
    Index integration adapters by:

        environment
        provider
        target

    Example conceptual registry:

        production
        ├── openstack
        │   └── cloud-01
        ├── kubernetes
        │   └── cluster-01
        └── ceph
            └── storage-01

        lab
        ├── kubernetes
        │   └── cluster-lab
        └── linux
            └── host-01
    """

    def __init__(
        self,
    ) -> None:
        self._adapters: Dict[
            RegistryKey,
            IntegrationAdapter,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        adapter: IntegrationAdapter,
    ) -> IntegrationAdapter:
        key = integration_key(
            environment=(adapter.target.environment_id),
            provider=(adapter.target.provider),
            target_id=(adapter.target.target_id),
        )

        with self._lock:
            if key in self._adapters:
                raise (
                    IntegrationConflictError(
                        "Integration adapter already registered.",
                        details={
                            "environment_id": (key[0]),
                            "provider": (key[1]),
                            "target_id": (key[2]),
                        },
                    )
                )

            self._adapters[key] = adapter

        return adapter

    # ---------------------------------------------------------
    # Upsert
    # ---------------------------------------------------------

    def upsert(
        self,
        adapter: IntegrationAdapter,
    ) -> IntegrationAdapter:
        key = integration_key(
            environment=(adapter.target.environment_id),
            provider=(adapter.target.provider),
            target_id=(adapter.target.target_id),
        )

        with self._lock:
            self._adapters[key] = adapter

        return adapter

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        *,
        environment: str,
        provider: str,
        target_id: str,
    ) -> Optional[IntegrationAdapter]:
        key = integration_key(
            environment=environment,
            provider=provider,
            target_id=target_id,
        )

        with self._lock:
            return self._adapters.get(key)

    # ---------------------------------------------------------
    # Require
    # ---------------------------------------------------------

    def require(
        self,
        *,
        environment: str,
        provider: str,
        target_id: str,
    ) -> IntegrationAdapter:
        adapter = self.get(
            environment=environment,
            provider=provider,
            target_id=target_id,
        )

        if adapter is None:
            raise (
                IntegrationRegistryError(
                    "Integration adapter is not registered.",
                    details={
                        "environment_id": (environment),
                        "provider": (provider),
                        "target_id": (target_id),
                    },
                )
            )

        return adapter

    # ---------------------------------------------------------
    # Remove
    # ---------------------------------------------------------

    def remove(
        self,
        *,
        environment: str,
        provider: str,
        target_id: str,
        close: bool = False,
    ) -> IntegrationAdapter:
        key = integration_key(
            environment=environment,
            provider=provider,
            target_id=target_id,
        )

        with self._lock:
            adapter = self._adapters.pop(
                key,
                None,
            )

        if adapter is None:
            raise (
                IntegrationRegistryError(
                    "Integration adapter is not registered.",
                    details={
                        "environment_id": (key[0]),
                        "provider": (key[1]),
                        "target_id": (key[2]),
                    },
                )
            )

        if close:
            adapter.close()

        return adapter

    # ---------------------------------------------------------
    # List/filter
    # ---------------------------------------------------------

    def list(
        self,
        *,
        environment: Optional[str] = None,
        provider: Optional[str] = None,
        ready_only: bool = False,
    ) -> List[IntegrationAdapter]:
        normalized_environment = (
            environment_id(environment) if environment is not None else None
        )

        normalized_provider = provider.lower().strip() if provider is not None else None

        if normalized_provider is not None and normalized_provider not in {
            item.value for item in ProviderKind
        }:
            raise (
                IntegrationRegistryError(
                    f"Unknown integration provider: {normalized_provider}"
                )
            )

        with self._lock:
            entries = list(self._adapters.items())

        output: List[IntegrationAdapter] = []

        for (
            key,
            adapter,
        ) in sorted(
            entries,
            key=lambda item: item[0],
        ):
            if normalized_environment is not None and key[0] != normalized_environment:
                continue

            if normalized_provider is not None and key[1] != normalized_provider:
                continue

            if ready_only and not adapter.ready:
                continue

            output.append(adapter)

        return output

    # ---------------------------------------------------------
    # Find adapters supporting a capability
    # ---------------------------------------------------------

    def find_capability(
        self,
        capability: str,
        *,
        environment: Optional[str] = None,
        provider: Optional[str] = None,
        action: Optional[str] = None,
        ready_only: bool = True,
    ) -> List[IntegrationAdapter]:
        """
        Find adapters advertising a particular
        capability and optionally an action.

        This discovers candidates.

        It does not independently decide which
        infrastructure target should be modified.
        """

        capability = required_string(
            capability,
            name="capability",
            maximum_length=256,
        )

        action = (
            required_string(
                action,
                name="action",
                maximum_length=128,
            )
            if action is not None
            else None
        )

        matches: List[IntegrationAdapter] = []

        for adapter in self.list(
            environment=environment,
            provider=provider,
            ready_only=ready_only,
        ):
            supported = adapter.capability(capability)

            if supported is None:
                continue

            if action is not None and action not in supported.actions:
                continue

            matches.append(adapter)

        return matches

    # ---------------------------------------------------------
    # Capability inventory
    # ---------------------------------------------------------

    def capabilities(
        self,
        *,
        environment: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[
        str,
        List[
            Dict[
                str,
                Any,
            ]
        ],
    ]:
        """
        Return capability metadata grouped by
        adapter target.
        """

        output: Dict[
            str,
            List[
                Dict[
                    str,
                    Any,
                ]
            ],
        ] = {}

        for adapter in self.list(
            environment=environment,
            provider=provider,
        ):
            key = "/".join(
                (
                    adapter.target.environment_id,
                    adapter.target.provider,
                    adapter.target.target_id,
                )
            )

            output[key] = [
                capability.to_dict() for capability in adapter.capabilities()
            ]

        return output

    # ---------------------------------------------------------
    # Close all
    # ---------------------------------------------------------

    def close_all(
        self,
    ) -> Dict[
        str,
        str,
    ]:
        """
        Attempt to close all registered adapters.

        One adapter failure does not prevent
        attempts to close the others.
        """

        results: Dict[
            str,
            str,
        ] = {}

        for adapter in self.list():
            key = "/".join(
                (
                    adapter.target.environment_id,
                    adapter.target.provider,
                    adapter.target.target_id,
                )
            )

            try:
                results[key] = adapter.close().state

            except Exception as exc:
                results[key] = "error:" + type(exc).__name__

        return results

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        adapters = self.list()

        by_provider: Dict[
            str,
            int,
        ] = {}

        by_environment: Dict[
            str,
            int,
        ] = {}

        ready = 0

        for adapter in adapters:
            provider = adapter.target.provider

            environment = adapter.target.environment_id

            by_provider[provider] = (
                by_provider.get(
                    provider,
                    0,
                )
                + 1
            )

            by_environment[environment] = (
                by_environment.get(
                    environment,
                    0,
                )
                + 1
            )

            if adapter.ready:
                ready += 1

        return {
            "adapter_count": (len(adapters)),
            "ready_adapter_count": (ready),
            "adapters_by_provider": (by_provider),
            "adapters_by_environment": (by_environment),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "RegistryKey",
    "integration_key",
    "IntegrationRegistry",
]
