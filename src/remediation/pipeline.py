from __future__ import annotations

import argparse
import json
import sys
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
)

from src.decision.schema import (
    ActionCandidate,
    Decision,
    DecisionPlan,
)

from src.remediation.adapters import (
    AdapterRegistry,
    default_adapter_registry,
)

from src.remediation.audit import (
    AuditConfig,
    JsonlAuditStore,
    RemediationAudit,
)

from src.remediation.executor import (
    RemediationExecutor,
)

from src.remediation.rollback import (
    RollbackManager,
)

from src.remediation.schema import (
    ExecutionStatus,
    RemediationBatchResult,
    RemediationMode,
    RemediationRequest,
    RemediationResult,
    RemediationStatus,
    RollbackStatus,
    VerificationStatus,
    utc_now,
)

from src.remediation.validator import (
    RemediationValidator,
)

from src.remediation.verify import (
    RemediationVerifier,
    VerificationObservation,
)


# ---------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------


def expiry_timestamp(
    minutes: int,
) -> str:
    if minutes < 1:
        raise ValueError("Expiry must be at least one minute")

    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def request_from_action(
    action: ActionCandidate,
    *,
    decision: Optional[Decision] = None,
    plan: Optional[DecisionPlan] = None,
    mode: str = (RemediationMode.DRY_RUN.value),
    requested_by: Optional[str] = None,
    expires_in_minutes: int = 30,
) -> RemediationRequest:
    """
    Convert an evaluated decision action into a
    remediation request.

    The default mode is dry_run.

    Creating the request does not execute the
    action.
    """

    action.finalize()

    if expires_in_minutes < 1:
        raise ValueError("expires_in_minutes must be >= 1")

    decision_id = None
    plan_id = None
    incident_id = None

    if decision is not None:
        decision.finalize()

        decision_id = decision.decision_id

        incident_id = decision.incident_id

    if plan is not None:
        plan.finalize()

        plan_id = plan.plan_id

        if incident_id is None:
            incident_id = plan.incident_id

    return RemediationRequest(
        action_id=(action.action_id or action.build_action_id()),
        action_type=(action.action_type),
        environment_id=(action.environment_id),
        mode=mode,
        decision_id=decision_id,
        plan_id=plan_id,
        incident_id=incident_id,
        target_resource_ids=list(action.target_resource_ids),
        target_service_ids=list(action.target_service_ids),
        parameters=dict(action.parameters),
        requested_by=requested_by,
        approval_status=(action.approval.status),
        approval_ids=list(action.approval.approved_by),
        risk_level=(action.risk.level),
        risk_score=(action.risk.score),
        reversibility=(action.reversibility),
        rollback_action_type=(action.rollback_action_type),
        expires_at=(expiry_timestamp(expires_in_minutes)),
        attributes={
            "action_state": (action.state),
            "eligibility_status": (action.eligibility.status),
            "eligible": (action.eligibility.eligible),
            "approval_mode": (action.approval.mode),
            "policy_ids": list(action.policy_ids),
        },
    ).finalize()


# ---------------------------------------------------------------
# Final status
# ---------------------------------------------------------------


def final_status(
    result: RemediationResult,
) -> str:
    """
    Derive the final lifecycle state.

    Dry-run completion means the controlled
    simulation completed. It does not mean
    infrastructure recovery was verified.
    """

    if not result.validation.valid:
        return RemediationStatus.BLOCKED.value

    if result.execution.status == ExecutionStatus.BLOCKED.value:
        return RemediationStatus.BLOCKED.value

    if result.execution.status in {
        ExecutionStatus.FAILED.value,
        ExecutionStatus.TIMED_OUT.value,
        ExecutionStatus.CANCELLED.value,
    }:
        if result.rollback.status == RollbackStatus.SUCCEEDED.value:
            return RemediationStatus.ROLLED_BACK.value

        return RemediationStatus.FAILED.value

    if result.request.mode == RemediationMode.DRY_RUN.value:
        return RemediationStatus.COMPLETED.value

    if result.verification.status == VerificationStatus.PASSED.value:
        return RemediationStatus.COMPLETED.value

    if result.rollback.status == RollbackStatus.SUCCEEDED.value:
        return RemediationStatus.ROLLED_BACK.value

    if result.verification.status == VerificationStatus.FAILED.value:
        return RemediationStatus.VERIFICATION_FAILED.value

    return RemediationStatus.FAILED.value


# ---------------------------------------------------------------
# Remediation pipeline
# ---------------------------------------------------------------


class RemediationPipeline:
    """
    Controlled remediation lifecycle.

    Default construction provides:

      dry-run adapter only
      execution disabled
      rollback execution disabled
      independent verification
      lifecycle auditing
    """

    def __init__(
        self,
        *,
        registry: Optional[AdapterRegistry] = None,
        validator: Optional[RemediationValidator] = None,
        executor: Optional[RemediationExecutor] = None,
        verifier: Optional[RemediationVerifier] = None,
        rollback_manager: Optional[RollbackManager] = None,
        audit: Optional[RemediationAudit] = None,
    ) -> None:
        self.registry = registry or default_adapter_registry()

        self.validator = validator or RemediationValidator(
            adapter_capabilities=(self.registry.capabilities())
        )

        self.executor = executor or RemediationExecutor(
            registry=self.registry,
            validator=self.validator,
        )

        self.verifier = verifier or RemediationVerifier()

        self.rollback_manager = rollback_manager or RollbackManager(
            registry=self.registry
        )

        self.audit = audit or RemediationAudit()

    def run(
        self,
        request: RemediationRequest,
        *,
        observations: Iterable[VerificationObservation] = (),
        health_before: Optional[float] = None,
        actor_id: Optional[str] = None,
    ) -> RemediationResult:
        """
        Process one remediation request.

        The default pipeline can only complete a
        non-mutating dry run.
        """

        request.finalize()

        result = RemediationResult(
            request=request,
            status=(RemediationStatus.CREATED.value),
            started_at=utc_now(),
            attributes={
                "execution_enabled": (False),
            },
        ).finalize()

        # -----------------------------------------------------
        # Request audit
        # -----------------------------------------------------

        record = self.audit.record_request(
            request,
            actor_id=actor_id,
            remediation_id=(result.remediation_id),
        )

        result.audit_events.append(record.event)

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        result.status = RemediationStatus.VALIDATING.value

        result.validation = self.validator.validate(request)

        record = self.audit.record_validation(
            request,
            result.validation,
            remediation_id=(result.remediation_id),
        )

        result.audit_events.append(record.event)

        if not result.validation.valid:
            result.status = RemediationStatus.BLOCKED.value

            result.completed_at = utc_now()

            record = self.audit.record_completion(result)

            result.audit_events.append(record.event)

            return result.finalize()

        # -----------------------------------------------------
        # Execution
        # -----------------------------------------------------

        result.status = RemediationStatus.READY.value

        result.execution = self.executor.execute(request)

        record = self.audit.record_execution(
            request,
            result.execution,
            remediation_id=(result.remediation_id),
        )

        result.audit_events.append(record.event)

        if result.execution.success:
            result.status = RemediationStatus.EXECUTED.value

        else:
            result.status = RemediationStatus.FAILED.value

        # -----------------------------------------------------
        # Verification
        # -----------------------------------------------------

        result.status = RemediationStatus.VERIFYING.value

        result.verification = self.verifier.verify(
            request,
            result.execution,
            observations=(observations),
            health_before=(health_before),
        )

        record = self.audit.record_verification(
            request,
            result.verification,
            remediation_id=(result.remediation_id),
        )

        result.audit_events.append(record.event)

        # -----------------------------------------------------
        # Rollback evaluation
        # -----------------------------------------------------

        result.rollback = self.rollback_manager.rollback(
            request,
            result.execution,
            result.verification,
        )

        record = self.audit.record_rollback(
            request,
            result.rollback,
            remediation_id=(result.remediation_id),
        )

        result.audit_events.append(record.event)

        # -----------------------------------------------------
        # Completion
        # -----------------------------------------------------

        result.status = final_status(result)

        result.completed_at = utc_now()

        result.attributes.update(
            {
                "dry_run": (request.mode == RemediationMode.DRY_RUN.value),
                "recovery_verified": (
                    result.verification.status == VerificationStatus.PASSED.value
                ),
                "infrastructure_changed": (result.execution.changed),
            }
        )

        record = self.audit.record_completion(result)

        result.audit_events.append(record.event)

        return result.finalize()

    def run_many(
        self,
        requests: Iterable[RemediationRequest],
    ) -> RemediationBatchResult:
        return RemediationBatchResult(
            results=[self.run(request) for request in requests],
            metadata={
                "adapter_registry": (self.registry.summary()),
                "execution_enabled": (False),
            },
        ).finalize()


# ---------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------


def write_json(
    path: Path,
    value: Mapping[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as destination:
        json.dump(
            value,
            destination,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        destination.write("\n")


def request_from_json(
    path: Path,
) -> RemediationRequest:
    """
    Load a standalone remediation request.

    This loader does not enable execution.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        data = json.load(source)

    return RemediationRequest(
        action_id=str(data["action_id"]),
        action_type=str(data["action_type"]),
        environment_id=str(data["environment_id"]),
        mode=str(
            data.get(
                "mode",
                RemediationMode.DRY_RUN.value,
            )
        ),
        decision_id=(data.get("decision_id")),
        plan_id=(data.get("plan_id")),
        incident_id=(data.get("incident_id")),
        target_resource_ids=list(
            data.get(
                "target_resource_ids",
                [],
            )
        ),
        target_service_ids=list(
            data.get(
                "target_service_ids",
                [],
            )
        ),
        parameters=dict(
            data.get(
                "parameters",
                {},
            )
        ),
        requested_by=(data.get("requested_by")),
        approval_status=(data.get("approval_status")),
        approval_ids=list(
            data.get(
                "approval_ids",
                [],
            )
        ),
        risk_level=(data.get("risk_level")),
        risk_score=float(
            data.get(
                "risk_score",
                0.0,
            )
        ),
        reversibility=(data.get("reversibility")),
        rollback_action_type=(data.get("rollback_action_type")),
        requested_at=str(
            data.get(
                "requested_at",
                utc_now(),
            )
        ),
        expires_at=(data.get("expires_at")),
        attributes=dict(
            data.get(
                "attributes",
                {},
            )
        ),
        request_id=(data.get("request_id")),
    ).finalize()


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the OSHA remediation lifecycle in safe dry-run mode.")
    )

    parser.add_argument(
        "--request",
        required=True,
        help=("Path to a JSON remediation request."),
    )

    parser.add_argument(
        "--output",
        default=("orchestration-ai-data/remediation/result.json"),
    )

    parser.add_argument(
        "--audit",
        default=("orchestration-ai-data/audit/remediation.jsonl"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    request_path = Path(args.request)

    if not request_path.is_file():
        print(
            (f"Request file does not exist: {request_path}"),
            file=sys.stderr,
        )

        return 2

    try:
        request = request_from_json(request_path)

        # The public CLI remains deliberately
        # dry-run-only.
        if request.mode != RemediationMode.DRY_RUN.value:
            print(
                ("This CLI currently accepts dry_run requests only."),
                file=sys.stderr,
            )

            return 2

        audit_config = AuditConfig()

        audit = RemediationAudit(
            store=JsonlAuditStore(
                Path(args.audit),
                config=audit_config,
            ),
            config=audit_config,
        )

        pipeline = RemediationPipeline(audit=audit)

        result = pipeline.run(request)

        write_json(
            Path(args.output),
            result.to_dict(),
        )

    except Exception as exc:
        print(
            (f"Remediation pipeline failed: {type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )

        return 1

    print("Remediation pipeline complete.")

    print(f"Mode: {result.request.mode}")

    print(f"Status: {result.status}")

    print(f"Validation: {result.validation.status}")

    print(f"Execution: {result.execution.status}")

    print(f"Infrastructure changed: {result.execution.changed}")

    print(f"Verification: {result.verification.status}")

    print(f"Rollback: {result.rollback.status}")

    print(f"Result: {args.output}")

    print(f"Audit: {args.audit}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
