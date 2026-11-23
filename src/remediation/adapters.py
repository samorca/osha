from __future__ import annotations

import time
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.remediation.schema import (
    ExecutionResult,
    ExecutionStatus,
    RemediationEvidence,
    RemediationMode,
    RemediationRequest,
    RollbackResult,
    RollbackStatus,
    utc_now,
)

from src.remediation.validator import (
    AdapterCapability,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------

class AdapterError(
    RuntimeError
):
    """
    Base adapter-layer error.
    """


class AdapterNotFoundError(
    AdapterError
):
    """
    No compatible adapter could be resolved.
    """


class AdapterExecutionDisabledError(
    AdapterError
):
    """
    Adapter was asked to perform unsupported
    real execution.
    """


# ---------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------

class RemediationAdapter(
    ABC
):
    """
    Abstract provider boundary.

    Concrete adapters will eventually translate
    generic remediation requests into provider-
    specific behavior.

    This base class does not implement any
    infrastructure command.
    """

    @property
    @abstractmethod
    def capability(
        self,
    ) -> AdapterCapability:
        pass

    def supports(
        self,
        request: RemediationRequest,
    ) -> bool:
        capability = (
            self.capability
        )

        capability.validate()

        environment = (
            request.environment_id
            .strip()
            .lower()
        )

        environments = {
            item.lower()
            for item
            in capability.
            supported_environments
        }

        if not capability.enabled:
            return False

        if (
            request.action_type
            not in capability.
            supported_action_types
        ):
            return False

        if (
            environments
            and environment
            not in environments
        ):
            return False

        if (
            request.mode
            == RemediationMode.
            EXECUTE.value
        ):
            return (
                capability.
                execution_supported
            )

        return (
            capability.
            dry_run_supported
        )

    @abstractmethod
    def execute(
        self,
        request: RemediationRequest,
    ) -> ExecutionResult:
        """
        Adapter execution boundary.

        Safe adapters may implement only dry-run.
        """

    def supports_rollback(
        self,
        request: RemediationRequest,
    ) -> bool:
        rollback_type = (
            request.
            rollback_action_type
        )

        return bool(
            rollback_type
            and rollback_type
            in self.capability.
            supported_rollback_types
        )

    def rollback(
        self,
        request: RemediationRequest,
    ) -> RollbackResult:
        """
        Default rollback implementation refuses
        execution.
        """

        return RollbackResult(
            status=(
                RollbackStatus.
                UNAVAILABLE.value
            ),
            success=False,
            action_type=(
                request.
                rollback_action_type
            ),
            adapter_name=(
                self.capability.
                adapter_name
            ),
            reason=(
                "Adapter does not implement "
                "rollback execution."
            ),
        ).finalize()


# ---------------------------------------------------------------
# Dry-run adapter
# ---------------------------------------------------------------

class DryRunAdapter(
    RemediationAdapter
):
    """
    Safe development adapter.

    It records what would be requested and
    performs no infrastructure mutation.

    Execute-mode requests are explicitly
    rejected.
    """

    def __init__(
        self,
        *,
        name: str = "dry-run",
        supported_action_types: Optional[
            Iterable[str]
        ] = None,
        supported_rollback_types: Optional[
            Iterable[str]
        ] = None,
        supported_environments: Iterable[
            str
        ] = (),
    ) -> None:
        self._capability = (
            AdapterCapability(
                adapter_name=name,
                supported_action_types=tuple(
                    supported_action_types
                    or (
                        "observe_resource",
                        "collect_diagnostics",
                        "notify_operator",
                        "restart_service",
                        "reload_service",
                        "rollback_configuration",
                        "isolate_resource",
                        "scale_capacity",
                    )
                ),
                supported_rollback_types=tuple(
                    supported_rollback_types
                    or (
                        "stop_observation",
                        "discard_diagnostic_session",
                        "withdraw_notification",
                        "restore_service_state",
                        "restore_previous_"
                        "service_configuration",
                        "restore_replaced_"
                        "configuration",
                        "remove_resource_"
                        "isolation",
                        "restore_previous_"
                        "capacity",
                    )
                ),
                supported_environments=tuple(
                    supported_environments
                ),
                execution_supported=False,
                dry_run_supported=True,
                enabled=True,
            )
        )

        self._capability.validate()

    @property
    def capability(
        self,
    ) -> AdapterCapability:
        return self._capability

    def execute(
        self,
        request: RemediationRequest,
    ) -> ExecutionResult:
        request.finalize()

        if (
            request.mode
            != RemediationMode.
            DRY_RUN.value
        ):
            raise (
                AdapterExecutionDisabledError(
                    "DryRunAdapter only "
                    "accepts dry_run "
                    "remediation requests"
                )
            )

        if not self.supports(
            request
        ):
            raise AdapterError(
                "Dry-run adapter does not "
                "support action "
                f"{request.action_type} in "
                f"{request.environment_id}"
            )

        started = utc_now()

        monotonic_started = (
            time.monotonic()
        )

        # Intentionally NO infrastructure
        # operation occurs here.

        duration = (
            time.monotonic()
            - monotonic_started
        )

        return ExecutionResult(
            status=(
                ExecutionStatus.
                DRY_RUN.value
            ),
            success=True,
            adapter_name=(
                self.capability.
                adapter_name
            ),
            operation=(
                request.action_type
            ),
            started_at=started,
            finished_at=utc_now(),
            duration_seconds=(
                duration
            ),
            exit_code=None,
            stdout=None,
            stderr=None,
            changed=False,
            evidence=[
                RemediationEvidence(
                    evidence_type=(
                        "dry_run"
                    ),
                    description=(
                        "The adapter validated "
                        "the requested operation "
                        "without changing "
                        "infrastructure."
                    ),
                    score=1.0,
                    source_id=(
                        request.request_id
                    ),
                    resource_ids=list(
                        request.
                        target_resource_ids
                    ),
                    attributes={
                        "action_type": (
                            request.
                            action_type
                        ),
                        "target_service_ids": (
                            list(
                                request.
                                target_service_ids
                            )
                        ),
                    },
                )
            ],
            attributes={
                "mode": (
                    RemediationMode.
                    DRY_RUN.value
                ),
                "request_id": (
                    request.request_id
                ),
                "parameters": dict(
                    request.parameters
                ),
            },
        ).finalize()

    def rollback(
        self,
        request: RemediationRequest,
    ) -> RollbackResult:
        """
        Dry-run rollback is capability checking
        only. No rollback operation occurs.
        """

        request.finalize()

        if (
            request.mode
            != RemediationMode.
            DRY_RUN.value
        ):
            return super().rollback(
                request
            )

        if not (
            request.
            rollback_action_type
        ):
            return RollbackResult(
                status=(
                    RollbackStatus.
                    NOT_REQUIRED.value
                ),
                success=False,
                adapter_name=(
                    self.capability.
                    adapter_name
                ),
                reason=(
                    "Dry-run request has no "
                    "rollback action."
                ),
            ).finalize()

        supported = (
            self.supports_rollback(
                request
            )
        )

        return RollbackResult(
            status=(
                RollbackStatus.
                AVAILABLE.value
                if supported
                else RollbackStatus.
                UNAVAILABLE.value
            ),
            success=False,
            action_type=(
                request.
                rollback_action_type
            ),
            adapter_name=(
                self.capability.
                adapter_name
            ),
            reason=(
                "Rollback is represented by "
                "the dry-run adapter but was "
                "not executed."
                if supported
                else (
                    "Rollback type is not "
                    "supported by the "
                    "dry-run adapter."
                )
            ),
            evidence=[
                RemediationEvidence(
                    evidence_type=(
                        "rollback_capability"
                    ),
                    description=(
                        "Dry-run rollback "
                        "capability was "
                        "evaluated without "
                        "infrastructure "
                        "mutation."
                    ),
                    score=(
                        1.0
                        if supported
                        else 0.0
                    ),
                    source_id=(
                        request.request_id
                    ),
                    resource_ids=list(
                        request.
                        target_resource_ids
                    ),
                )
            ],
        ).finalize()


# ---------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------

class AdapterRegistry:
    """
    Deterministic registry of remediation
    adapters.

    Resolution itself does not execute anything.
    """

    def __init__(
        self,
        adapters: Iterable[
            RemediationAdapter
        ] = (),
    ) -> None:
        self._adapters: Dict[
            str,
            RemediationAdapter,
        ] = {}

        for adapter in adapters:
            self.register(
                adapter
            )

    def register(
        self,
        adapter: RemediationAdapter,
    ) -> None:
        name = (
            adapter.capability.
            adapter_name.strip()
        )

        if not name:
            raise ValueError(
                "Adapter name is required"
            )

        if (
            name
            in self._adapters
        ):
            raise ValueError(
                "Duplicate remediation "
                f"adapter: {name}"
            )

        adapter.capability.validate()

        self._adapters[
            name
        ] = adapter

    def unregister(
        self,
        adapter_name: str,
    ) -> bool:
        return (
            self._adapters.pop(
                adapter_name,
                None,
            )
            is not None
        )

    def get(
        self,
        adapter_name: str,
    ) -> Optional[
        RemediationAdapter
    ]:
        return self._adapters.get(
            adapter_name
        )

    def require(
        self,
        adapter_name: str,
    ) -> RemediationAdapter:
        adapter = self.get(
            adapter_name
        )

        if adapter is None:
            raise (
                AdapterNotFoundError(
                    "Remediation adapter "
                    "not found: "
                    f"{adapter_name}"
                )
            )

        return adapter

    def capabilities(
        self,
    ) -> List[
        AdapterCapability
    ]:
        return [
            adapter.capability
            for _, adapter
            in sorted(
                self._adapters.items()
            )
        ]

    def compatible(
        self,
        request: RemediationRequest,
    ) -> List[
        RemediationAdapter
    ]:
        """
        Return adapters that declare support for
        the request's action, environment and
        execution mode.
        """

        return [
            adapter
            for _, adapter
            in sorted(
                self._adapters.items()
            )
            if adapter.supports(
                request
            )
        ]

    def resolve(
        self,
        request: RemediationRequest,
        *,
        preferred_adapter: Optional[
            str
        ] = None,
    ) -> RemediationAdapter:
        request.finalize()

        if preferred_adapter:
            adapter = self.require(
                preferred_adapter
            )

            if not adapter.supports(
                request
            ):
                raise (
                    AdapterNotFoundError(
                        "Adapter "
                        f"{preferred_adapter} "
                        "is not compatible "
                        "with request "
                        f"{request.request_id}"
                    )
                )

            return adapter

        matches = self.compatible(
            request
        )

        if not matches:
            raise (
                AdapterNotFoundError(
                    "No remediation adapter "
                    "supports action "
                    f"{request.action_type} "
                    "in mode "
                    f"{request.mode}"
                )
            )

        # Registry iteration is sorted by adapter
        # name, making default resolution
        # deterministic.
        return matches[0]

    def summary(
        self,
    ) -> Dict[str, Any]:
        capabilities = (
            self.capabilities()
        )

        return {
            "adapter_count": len(
                capabilities
            ),
            "execution_capable_count": (
                sum(
                    1
                    for item
                    in capabilities
                    if item.
                    execution_supported
                )
            ),
            "dry_run_capable_count": (
                sum(
                    1
                    for item
                    in capabilities
                    if item.
                    dry_run_supported
                )
            ),
            "adapters": [
                {
                    "adapter_name": (
                        item.adapter_name
                    ),
                    "supported_action_types": (
                        list(
                            item.
                            supported_action_types
                        )
                    ),
                    "supported_rollback_types": (
                        list(
                            item.
                            supported_rollback_types
                        )
                    ),
                    "supported_environments": (
                        list(
                            item.
                            supported_environments
                        )
                    ),
                    "execution_supported": (
                        item.
                        execution_supported
                    ),
                    "dry_run_supported": (
                        item.
                        dry_run_supported
                    ),
                    "enabled": (
                        item.enabled
                    ),
                }
                for item
                in capabilities
            ],
        }


# ---------------------------------------------------------------
# Safe default registry
# ---------------------------------------------------------------

def default_adapter_registry(
) -> AdapterRegistry:
    """
    Development registry.

    No execution-capable infrastructure adapter
    is registered.
    """

    return AdapterRegistry(
        [
            DryRunAdapter()
        ]
    )
