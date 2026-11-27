from __future__ import annotations

import os
import time

from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Optional, Sequence

from src.correlation.pipeline import (
    LoadIssue,
    run_pipeline as run_correlation_pipeline,
)
from src.decision.pipeline import (
    run_pipeline as run_decision_pipeline,
)
from src.decision.schema import (
    ActionCategory,
)
from src.intelligence.pipeline import (
    run_pipeline as run_intelligence_pipeline,
)
from src.integrations.config import from_environment as integrations_config_from_environment
from src.integrations.pipeline import IntegrationPipeline
from src.memory.pipeline import run_pipeline as run_memory_pipeline
from src.memory.schema import OutcomeMemory, utc_now
from src.memory.store import MemoryStore, create_memory_store
from src.observability.config import from_environment as observability_config_from_environment
from src.observability.pipeline import ObservabilityPipeline
from src.observability.schema import ObservabilityContext, Severity
from src.orchestrator.coordinator import StageExecution
from src.orchestrator.lifecycle import DEFAULT_STAGE_ORDER
from src.orchestrator.pipeline import (
    OrchestratorPipeline,
    PipelineConfig,
    PipelineInput,
    mutating_action,
)
from src.orchestrator.schema import (
    OrchestrationContext,
    StageName,
    TriggerType,
)
from src.self_healing.pipeline import (
    SelfHealingConfig,
    SelfHealingPipeline,
    self_healing_summary,
)
from src.self_healing.schema import SelfHealingResult
from src.remediation.schema import RemediationMode
from src.telemetry.normalize import normalize_records


_MAX_RECORDS_DEFAULT = 10_000
_SAFE_OBSERVATION_ACTIONS = {
    ActionCategory.OBSERVATION.value,
    ActionCategory.INVESTIGATION.value,
    ActionCategory.NOTIFICATION.value,
}


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _safe_process_role(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value.strip().lower()
    ).strip("-_")
    return normalized or "application"


class FrameworkApplication:
    """Safe composition root for OSHA's telemetry-to-decision frameworks."""

    def __init__(
        self,
        *,
        process_role: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
        integrations: Optional[IntegrationPipeline] = None,
        observability: Optional[ObservabilityPipeline] = None,
        maximum_records: Optional[int] = None,
    ) -> None:
        self.process_role = _safe_process_role(
            process_role or os.getenv("OSHA_PROCESS_ROLE", "application")
        )
        self.environment_id = os.getenv("OSHA_ENVIRONMENT_ID", "local-dev")
        self.enabled = _environment_bool("OSHA_FRAMEWORKS_ENABLED", default=True)
        self.maximum_records = maximum_records or int(
            os.getenv("OSHA_FRAMEWORK_MAX_RECORDS", str(_MAX_RECORDS_DEFAULT))
        )
        if self.maximum_records < 1:
            raise ValueError("OSHA_FRAMEWORK_MAX_RECORDS must be >= 1")

        data_directory = Path(os.getenv("OSHA_DATA_DIR", "dataset"))
        memory_path = os.getenv("OSHA_FRAMEWORK_MEMORY_STORE_PATH")
        self.memory_store = memory_store or create_memory_store(
            backend="json",
            path=memory_path
            or data_directory / "memory" / f"{self.process_role}.json",
        )
        self.integrations = integrations or IntegrationPipeline(
            config=integrations_config_from_environment()
        )
        self.observability = observability or ObservabilityPipeline(
            config=observability_config_from_environment()
        )
        self.self_healing = SelfHealingPipeline(
            config=SelfHealingConfig(
                remediation_mode=RemediationMode.DRY_RUN.value
            )
        )
        self._lock = RLock()
        self._started = False

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self._started)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._started:
                return self.status()
            if not self.enabled:
                return self.status()

            integration_result = self.integrations.start(connect_registered=False)
            self._started = True

        return {
            "ready": self.ready,
            "integrations": integration_result,
        }

    def close(self) -> dict[str, Any]:
        with self._lock:
            if not self._started:
                return {"closed": True}
            result = self.integrations.stop()
            self._started = False
        return {"closed": True, "integrations": result}

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "process_role": self.process_role,
            "environment_id": self.environment_id,
            "memory": self.memory_store.stats().to_dict(),
            "integrations": self.integrations.summary(),
            "observability": self.observability.summary(),
            "remediation_mode": RemediationMode.DRY_RUN.value,
            "infrastructure_mutations_enabled": False,
        }

    def register_integration(
        self,
        adapter: Any,
        *,
        connect: bool = False,
        credential_required: bool = False,
    ) -> Any:
        return self.integrations.register(
            adapter,
            connect=connect,
            credential_required=credential_required,
        )

    def process_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        environment_id: Optional[str] = None,
        provider_hint: Optional[str] = None,
        trigger_type: str = TriggerType.INTERNAL.value,
        requested_by: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OSHA framework processing is disabled")
        if len(records) > self.maximum_records:
            raise ValueError(
                f"record batch exceeds the {self.maximum_records}-record limit"
            )
        if not all(isinstance(record, Mapping) for record in records):
            raise TypeError("all telemetry records must be mappings")

        self.start()
        selected_environment = environment_id or self.environment_id

        with self._lock:
            return self._process_locked(
                records,
                environment_id=selected_environment,
                provider_hint=provider_hint,
                trigger_type=trigger_type,
                requested_by=requested_by,
                correlation_id=correlation_id,
            )

    def _process_locked(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        environment_id: str,
        provider_hint: Optional[str],
        trigger_type: str,
        requested_by: Optional[str],
        correlation_id: Optional[str],
    ) -> dict[str, Any]:
        started = time.monotonic()
        outputs: dict[str, Any] = {}

        def collection_stage(_: Any) -> StageExecution:
            return StageExecution(
                output={"record_count": len(records)},
                output_count=len(records),
            )

        def telemetry_stage(_: Any) -> StageExecution:
            normalized = normalize_records(
                records,
                provider_hint=provider_hint,
                environment_id=environment_id,
                source_name="framework_input",
            )
            telemetry_records = [
                result.telemetry for result in normalized if result.valid
            ]
            issues = [
                LoadIssue(
                    source=result.source_name or "framework_input",
                    line_number=result.source_line or 0,
                    error_type=(result.error or {}).get(
                        "error_type",
                        "ValidationError",
                    ),
                    message="Telemetry record was rejected during normalization.",
                )
                for result in normalized
                if not result.valid
            ]
            outputs["normalization"] = {
                "records": telemetry_records,
                "issues": issues,
                "results": normalized,
            }
            return StageExecution(
                output={
                    "valid_record_count": len(telemetry_records),
                    "rejected_record_count": len(issues),
                },
                output_count=len(telemetry_records),
            )

        def correlation_stage(_: Any) -> StageExecution:
            normalized = outputs["normalization"]
            result = run_correlation_pipeline(
                normalized["records"],
                load_issues=normalized["issues"],
            )
            outputs["correlation"] = result
            return StageExecution(
                output=result,
                output_count=len(result.result.groups),
            )

        def intelligence_stage(_: Any) -> StageExecution:
            result = run_intelligence_pipeline(outputs["correlation"])
            outputs["intelligence"] = result
            return StageExecution(
                output=result,
                output_count=len(result.result.incidents),
            )

        def memory_stage(_: Any) -> StageExecution:
            result = run_memory_pipeline(
                outputs["intelligence"],
                self.memory_store,
                retention_dry_run=True,
            )
            outputs["memory"] = result
            return StageExecution(
                output=result,
                output_count=result.store.stats().memory_count,
            )

        def decision_stage(_: Any) -> StageExecution:
            result = run_decision_pipeline(
                outputs["intelligence"],
                self.memory_store,
            )
            actions = [
                action
                for plan in result.result.plans
                for action in plan.actions
            ]
            decision_artifact = {
                "analysis": result,
                "actions": actions,
                "selected_action": actions[0] if actions else None,
                "approval_required": any(
                    action.approval.required for action in actions
                ),
            }
            outputs["decision"] = decision_artifact
            return StageExecution(
                output=decision_artifact,
                output_count=len(actions),
            )

        def remediation_stage(context: Any) -> StageExecution:
            decision = context.get(StageName.DECISION.value, {})
            actions = decision.get("actions", [])
            if not actions:
                return StageExecution(
                    skip=True,
                    reason="No decision actions were produced.",
                )

            return StageExecution(
                output={
                    "status": "dry_run",
                    "request": {"mode": RemediationMode.DRY_RUN.value},
                    "execution": {
                        "status": RemediationMode.DRY_RUN.value,
                        "changed": False,
                    },
                    "actions": [action.to_dict() for action in actions],
                    "execution_enabled": False,
                },
                output_count=len(actions),
                attributes={"dry_run": True, "execution_enabled": False},
            )

        def self_healing_stage(context: Any) -> StageExecution:
            decision = context.get(StageName.DECISION.value, {})
            analysis = decision.get("analysis")
            cycles = []
            if analysis is not None:
                decisions_by_incident = {
                    item.incident_id: item
                    for item in analysis.result.decisions
                    if item.incident_id
                }
                plans_by_incident = {
                    item.incident_id: item
                    for item in analysis.result.plans
                    if item.incident_id
                }
                for plan in analysis.result.plans:
                    for action in plan.actions:
                        if (
                            not action.eligibility.eligible
                            or action.approval.required
                            or action.category not in _SAFE_OBSERVATION_ACTIONS
                            or mutating_action(action)
                        ):
                            continue
                        incident_id = plan.incident_id
                        if not incident_id:
                            continue
                        cycles.append(
                            self.self_healing.run(
                                action,
                                decision=decisions_by_incident.get(incident_id),
                                plan=plans_by_incident.get(incident_id),
                                requested_by=requested_by,
                            )
                        )

            result = SelfHealingResult(
                sessions=[cycle.session for cycle in cycles],
                metadata={
                    "remediation_mode": RemediationMode.DRY_RUN.value,
                    "execution_enabled": False,
                    "eligible_safe_action_count": len(cycles),
                },
            ).finalize()
            outputs["self_healing"] = {
                "result": result,
                "cycles": cycles,
            }
            return StageExecution(
                output=result,
                output_count=len(result.sessions),
            )

        def outcome_stage(_: Any) -> StageExecution:
            healing = outputs.get("self_healing", {})
            cycles = healing.get("cycles", [])
            persisted = 0
            for cycle in cycles:
                payload = cycle.memory_payload
                session = cycle.session
                outcome = session.outcome
                if outcome is None or not session.incident_id:
                    continue
                self.memory_store.put(
                    OutcomeMemory(
                        environment_id=session.environment_id,
                        incident_id=session.incident_id,
                        outcome_type=outcome.status,
                        summary=outcome.summary,
                        remediation_action_ids=[session.action_id]
                        if session.action_id
                        else [],
                        recovery_verified=outcome.recovery_verified,
                        verification_source="osha_self_healing_pipeline",
                        observed_at=outcome.completed_at or utc_now(),
                        resource_ids=list(session.resource_ids),
                        attributes=payload,
                    )
                )
                persisted += 1
            result = {
                "healing": self_healing_summary(
                    healing.get("result", SelfHealingResult()).finalize()
                ),
                "outcome_memories_persisted": persisted,
            }
            outputs["outcome"] = result
            return StageExecution(output=result, output_count=persisted)

        handlers = {
            StageName.COLLECTION.value: collection_stage,
            StageName.TELEMETRY.value: telemetry_stage,
            StageName.CORRELATION.value: correlation_stage,
            StageName.INTELLIGENCE.value: intelligence_stage,
            StageName.MEMORY.value: memory_stage,
            StageName.DECISION.value: decision_stage,
            StageName.REMEDIATION.value: remediation_stage,
            StageName.SELF_HEALING.value: self_healing_stage,
            StageName.OUTCOME.value: outcome_stage,
        }
        orchestrator = OrchestratorPipeline(
            handlers=handlers,
            config=PipelineConfig(
                stage_order=DEFAULT_STAGE_ORDER,
                require_dry_run=True,
                fail_fast=True,
            ),
        )
        orchestration = orchestrator.run(
            PipelineInput(
                context=OrchestrationContext(
                    environment_id=environment_id,
                    trigger_type=trigger_type,
                    correlation_id=correlation_id,
                    requested_by=requested_by,
                    dry_run=True,
                    attributes={"input_record_count": len(records)},
                )
            )
        )

        stage_results = orchestration.run.stages
        self._record_observability(
            environment_id=environment_id,
            correlation_id=correlation_id,
            run_id=orchestration.run.run_id or "",
            stage_results=stage_results,
            elapsed_seconds=time.monotonic() - started,
        )

        normalized = outputs.get("normalization", {})
        correlation = outputs.get("correlation")
        intelligence = outputs.get("intelligence")
        memory = outputs.get("memory")
        decision = outputs.get("decision")
        healing = outputs.get("self_healing")

        failed = any(item.status == "failed" for item in stage_results)
        stopped = orchestration.run.stop_reason not in {"none", "completed"}
        status = "failed" if failed else "partial" if stopped else "completed"
        return {
            "status": status,
            "summary": {
                "status": status,
                "input_record_count": len(records),
                "normalized_record_count": len(normalized.get("records", [])),
                "rejected_record_count": len(normalized.get("issues", [])),
                "correlation_group_count": (
                    len(correlation.result.groups) if correlation else 0
                ),
                "incident_count": (
                    len(intelligence.result.incidents) if intelligence else 0
                ),
                "decision_count": (
                    len(decision["analysis"].result.decisions) if decision else 0
                ),
                "healing_session_count": (
                    len(healing["result"].sessions) if healing else 0
                ),
                "orchestration_state": orchestration.run.state,
                "stop_reason": orchestration.run.stop_reason,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            },
            "telemetry": {
                "records": [
                    item.telemetry.to_dict()
                    for item in normalized.get("results", [])
                    if item.valid and item.telemetry is not None
                ],
                "rejected": [
                    {
                        "source_line": item.source_line,
                        "source_name": item.source_name,
                        "error_type": (item.error or {}).get(
                            "error_type",
                            "ValidationError",
                        ),
                        "stage": (item.error or {}).get("stage"),
                    }
                    for item in normalized.get("results", [])
                    if not item.valid
                ],
            },
            "correlation": correlation.to_dict() if correlation else None,
            "intelligence": intelligence.to_dict() if intelligence else None,
            "memory": memory.summary() if memory else None,
            "decision": (
                decision["analysis"].to_dict() if decision is not None else None
            ),
            "self_healing": (
                {
                    "summary": self_healing_summary(healing["result"]),
                    "cycles": [cycle.to_dict() for cycle in healing["cycles"]],
                }
                if healing is not None
                else None
            ),
            "orchestration": orchestration.to_dict(),
            "integrations": self.integrations.summary(),
        }

    def _record_observability(
        self,
        *,
        environment_id: str,
        correlation_id: Optional[str],
        run_id: str,
        stage_results: Sequence[Any],
        elapsed_seconds: float,
    ) -> None:
        for stage in stage_results:
            context = ObservabilityContext(
                environment_id=environment_id,
                service="osha",
                component=stage.stage,
                orchestration_run_id=run_id or None,
                correlation_id=correlation_id,
            )
            failed = stage.status in {"failed", "blocked"}
            self.observability.events.emit(
                name="framework.stage.completed",
                context=context,
                severity=Severity.ERROR.value if failed else Severity.INFO.value,
                event_type="orchestration_stage",
                attributes={"status": stage.status},
            )
            self.observability.metrics.increment(
                name="osha_framework_stage_runs_total",
                context=context,
                labels={"stage": stage.stage, "status": stage.status},
            )
        self.observability.metrics.observe(
            name="osha_framework_run_duration_seconds",
            value=elapsed_seconds,
            context=ObservabilityContext(
                environment_id=environment_id,
                service="osha",
                component="frameworks",
                orchestration_run_id=run_id or None,
                correlation_id=correlation_id,
            ),
            unit="seconds",
        )


__all__ = ["FrameworkApplication"]
