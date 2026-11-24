from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.correlation.pipeline import (
    load_telemetry_files,
    run_pipeline as run_correlation_pipeline,
)

from src.decision.planner import (
    DecisionPlanner,
    PlannerConfig,
    planner_summary,
)

from src.decision.schema import (
    DecisionContext,
    DecisionEvidence,
    DecisionResult,
)

from src.intelligence.pipeline import (
    IntelligencePipelineOutput,
    run_pipeline as run_intelligence_pipeline,
)

from src.intelligence.schema import (
    IncidentCandidate,
    RootCauseStatus,
)

from src.memory.pipeline import (
    run_pipeline as run_memory_pipeline,
)

from src.memory.retrieval import (
    MemoryRetriever,
    RetrievalConfig,
)

from src.memory.schema import (
    VerificationStatus,
)

from src.memory.store import (
    MemoryStore,
    create_memory_store,
)


@dataclass
class DecisionPipelineOutput:
    """
    Complete output of one decision-planning run.

    No execution state is included because this
    layer cannot modify infrastructure.
    """

    result: DecisionResult

    elapsed_seconds: float

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "context_count": len(self.result.contexts),
            "decision_count": len(self.result.decisions),
            "unplanned_incident_count": len(self.result.unplanned_incident_ids),
            **planner_summary(self.result.plans),
        }

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "decision": (self.result.to_dict()),
        }


# ---------------------------------------------------------------
# Decision context
# ---------------------------------------------------------------


def context_from_incident(
    incident: IncidentCandidate,
    store: MemoryStore,
    *,
    retrieval_config: Optional[RetrievalConfig] = None,
) -> DecisionContext:
    """
    Convert intelligence and memory into bounded
    decision context.

    Historical memory contributes evidence but
    does not automatically verify the current
    incident or its root cause.
    """

    incident.finalize()

    retriever = MemoryRetriever(
        store,
        config=(retrieval_config or RetrievalConfig()),
    )

    matches = retriever.similar_incidents(incident)

    verified_matches = [
        item
        for item in matches
        if (item.verification_status == VerificationStatus.VERIFIED.value)
    ]

    strongest = incident.root_causes[0] if incident.root_causes else None

    # Current-incident confirmation is the
    # strongest verification source.
    root_verified = bool(
        strongest is not None and strongest.status == RootCauseStatus.CONFIRMED.value
    )

    historical_verified_causes = []

    if strongest is not None:
        historical_verified_causes = retriever.resource_history(
            strongest.resource_id,
            environment_id=(incident.environment_id),
            verified_only=True,
            limit=20,
        )

    # Historical verification is context, not
    # proof of the current incident's cause.
    historical_root_cause_match = (
        any(
            (
                getattr(
                    item,
                    "root_resource_id",
                    None,
                )
                == strongest.resource_id
                and getattr(
                    item,
                    "verification_status",
                    None,
                )
                == VerificationStatus.VERIFIED.value
            )
            for item in historical_verified_causes
        )
        if strongest is not None
        else False
    )

    outcome_ids = []

    for match in matches:
        outcome_id = match.attributes.get("outcome_memory_id")

        if outcome_id:
            outcome_ids.append(str(outcome_id))

    evidence = [
        DecisionEvidence(
            evidence_type=("incident_confidence"),
            description=(
                "Decision context includes the intelligence confidence assessment."
            ),
            score=(incident.confidence.overall),
            incident_id=(incident.incident_id),
            resource_ids=list(incident.resource_ids),
        ),
        DecisionEvidence(
            evidence_type=("incident_impact"),
            description=(
                "Decision context includes the intelligence impact assessment."
            ),
            score=(incident.impact.score),
            incident_id=(incident.incident_id),
            resource_ids=list(incident.impact.affected_resource_ids),
        ),
    ]

    if strongest is not None:
        evidence.append(
            DecisionEvidence(
                evidence_type=("root_cause"),
                description=(
                    "Decision context includes the leading root-cause hypothesis."
                ),
                score=strongest.score,
                incident_id=(incident.incident_id),
                resource_ids=[strongest.resource_id],
                attributes={
                    "candidate_id": (strongest.candidate_id),
                    "status": (strongest.status),
                    "current_incident_verified": (root_verified),
                    "historical_verified_match": (historical_root_cause_match),
                },
            )
        )

    return DecisionContext(
        environment_id=(incident.environment_id),
        incident_id=(incident.incident_id),
        incident_severity=(incident.severity),
        incident_confidence=(incident.confidence.overall),
        incident_confidence_level=(incident.confidence.level),
        impact_level=(incident.impact.level),
        impact_score=(incident.impact.score),
        user_impact_known=(incident.impact.user_impact_known),
        resource_ids=list(incident.resource_ids),
        service_ids=list(incident.impact.affected_services),
        root_cause_resource_id=(strongest.resource_id if strongest else None),
        root_cause_score=(strongest.score if strongest else 0.0),
        root_cause_status=(strongest.status if strongest else None),
        # Historical matching alone does not set
        # this field to True.
        root_cause_verified=(root_verified),
        similar_incident_ids=[
            str(item.attributes.get("incident_id"))
            for item in matches
            if item.attributes.get("incident_id")
        ],
        verified_similar_incident_ids=[
            str(item.attributes.get("incident_id"))
            for item in verified_matches
            if item.attributes.get("incident_id")
        ],
        historical_outcome_ids=(outcome_ids),
        evidence=evidence,
        limitations=list(incident.confidence.limitations),
        attributes={
            "incident_status": (incident.status),
            "similar_incident_matches": [item.to_dict() for item in matches],
            "pattern_ids": list(incident.pattern_ids),
            "anomaly_ids": list(incident.anomaly_ids),
            "historical_verified_root_cause_match": (historical_root_cause_match),
        },
    ).finalize()


# ---------------------------------------------------------------
# Decision execution
# ---------------------------------------------------------------


def run_pipeline(
    intelligence: IntelligencePipelineOutput,
    store: MemoryStore,
    *,
    planner: Optional[DecisionPlanner] = None,
    planner_config: Optional[PlannerConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
) -> DecisionPipelineOutput:
    """
    Run decision planning for all intelligence
    incidents.

    This function produces contexts, plans and
    decisions only. It cannot execute actions.
    """

    started = time.monotonic()

    active_planner = planner or DecisionPlanner(
        config=(planner_config or PlannerConfig())
    )

    contexts = []

    plans = []

    decisions = []

    unplanned = []

    planning_errors: Dict[
        str,
        str,
    ] = {}

    for incident in intelligence.result.incidents:
        try:
            context = context_from_incident(
                incident,
                store,
                retrieval_config=(retrieval_config),
            )

            plan, decision = active_planner.plan(context)

            contexts.append(context)

            plans.append(plan)

            decisions.append(decision)

        except Exception as exc:
            incident_id = incident.incident_id or "unknown"

            if incident.incident_id:
                unplanned.append(incident.incident_id)

            planning_errors[incident_id] = f"{type(exc).__name__}: {exc}"

    result = DecisionResult(
        decisions=decisions,
        plans=plans,
        contexts=contexts,
        unplanned_incident_ids=(unplanned),
        metadata={
            "execution_enabled": False,
            "incident_count": len(intelligence.result.incidents),
            "memory_record_count": (store.stats().memory_count),
            "planning_errors": (planning_errors),
        },
    ).finalize()

    return DecisionPipelineOutput(
        result=result,
        elapsed_seconds=(time.monotonic() - started),
    )


# ---------------------------------------------------------------
# Output
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


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OSHA telemetry, correlation, "
            "intelligence, memory and "
            "declarative decision planning."
        )
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=("Normalized telemetry JSONL file. Repeat for additional inputs."),
    )

    parser.add_argument(
        "--memory-backend",
        choices=(
            "memory",
            "json",
        ),
        default="json",
    )

    parser.add_argument(
        "--memory-store",
        default=("orchestration-ai-data/memory/memory.json"),
    )

    parser.add_argument(
        "--output",
        default=("orchestration-ai-data/decision/result.json"),
    )

    parser.add_argument(
        "--window-mode",
        choices=(
            "fixed",
            "sliding",
        ),
        default="sliding",
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--step-seconds",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--maximum-pairs-per-window",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--exclude-mutating-candidates",
        action="store_true",
        help=("Generate only observation, investigation and notification candidates."),
    )

    parser.add_argument(
        "--exclude-blocked-actions",
        action="store_true",
        help=("Exclude ineligible actions from the serialized decision plans."),
    )

    parser.add_argument(
        "--maximum-actions",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")

    if args.step_seconds <= 0:
        raise ValueError("--step-seconds must be > 0")

    if args.window_mode == "sliding" and args.step_seconds > args.window_seconds:
        raise ValueError("--step-seconds must not exceed --window-seconds")

    if args.maximum_pairs_per_window < 1:
        raise ValueError("--maximum-pairs-per-window must be >= 1")

    if args.maximum_actions < 1:
        raise ValueError("--maximum-actions must be >= 1")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

    except ValueError as exc:
        print(
            f"Argument error: {exc}",
            file=sys.stderr,
        )

        return 2

    input_paths = [Path(value) for value in args.input]

    missing = [path for path in input_paths if not path.is_file()]

    if missing:
        for path in missing:
            print(
                (f"Input file does not exist: {path}"),
                file=sys.stderr,
            )

        return 2

    try:
        records, load_issues = load_telemetry_files(input_paths)

        correlation = run_correlation_pipeline(
            records,
            window_mode=(args.window_mode),
            window_seconds=(args.window_seconds),
            step_seconds=(args.step_seconds),
            maximum_pairs_per_window=(args.maximum_pairs_per_window),
            load_issues=(load_issues),
        )

        intelligence = run_intelligence_pipeline(correlation)

        store = create_memory_store(
            backend=(args.memory_backend),
            path=(args.memory_store if args.memory_backend == "json" else None),
        )

        # Persist current operational history
        # before decision retrieval.
        #
        # Retention remains dry-run.
        run_memory_pipeline(
            intelligence,
            store,
            retention_dry_run=True,
        )

        output = run_pipeline(
            intelligence,
            store,
            planner_config=(
                PlannerConfig(
                    include_mutating_candidates=(not args.exclude_mutating_candidates),
                    include_ineligible_actions=(not args.exclude_blocked_actions),
                    maximum_actions_per_plan=(args.maximum_actions),
                )
            ),
        )

        write_json(
            Path(args.output),
            output.to_dict(),
        )

    except Exception as exc:
        print(
            (f"Decision pipeline failed: {type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )

        return 1

    summary = output.summary()

    print("Decision pipeline complete.")

    print(f"Contexts: {summary['context_count']}")

    print(f"Plans: {summary['plan_count']}")

    print(f"Decisions: {summary['decision_count']}")

    print(f"Actions: {summary['action_count']}")

    print(f"Eligible actions: {summary['eligible_action_count']}")

    print(f"Blocked actions: {summary['blocked_action_count']}")

    print(f"Approval required: {summary['approval_required_count']}")

    print("Execution enabled: False")

    print(f"Result: {args.output}")

    if args.fail_on_invalid and (
        correlation.rejected_record_count > 0 or correlation.load_issues
    ):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
