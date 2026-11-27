from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from src.api.dependencies import APIDependencies
from src.frameworks import FrameworkApplication


FRAMEWORK_SERVICE_NAME = "frameworks"


class FrameworkService:
    """API-facing lifecycle and request boundary for framework processing."""

    def __init__(
        self,
        application: Optional[FrameworkApplication] = None,
    ) -> None:
        self.application = application or FrameworkApplication(process_role="api")

    @property
    def ready(self) -> bool:
        return self.application.ready

    def start(self) -> dict[str, Any]:
        return self.application.start()

    def close(self) -> dict[str, Any]:
        return self.application.close()

    def status(self) -> dict[str, Any]:
        return self.application.status()

    def process_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        environment_id: Optional[str] = None,
        provider_hint: Optional[str] = None,
        requested_by: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        from src.orchestrator.schema import TriggerType

        return self.application.process_records(
            records,
            environment_id=environment_id,
            provider_hint=provider_hint,
            trigger_type=TriggerType.API.value,
            requested_by=requested_by,
            correlation_id=correlation_id,
        )


def register_framework_service(dependencies: APIDependencies) -> FrameworkService:
    service = FrameworkService()
    dependencies.register_service(FRAMEWORK_SERVICE_NAME, service)
    return service


__all__ = [
    "FRAMEWORK_SERVICE_NAME",
    "FrameworkService",
    "register_framework_service",
]
