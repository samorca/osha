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
    Iterable,
    Mapping,
    Optional,
)

from src.correlation.pipeline import (
    PipelineOutput as CorrelationPipelineOutput,
    load_telemetry_files,
    run_pipeline as run_correlation_pipeline,
)

from src.intelligence.anomaly import (
    AnomalyConfig,
    AnomalyEngine,
    anomaly_summary,
)

from src.intelligence.confidence import (
    ConfidenceAnalyzer,
    ConfidenceConfig,
    confidence_summary,
)

from src.intelligence.explain import (
    ExplanationEngine,
    explanation_summary_stats,
    render_explanation,
)

from src.intelligence.impact import (
    ImpactAnalyzer,
    ImpactConfig,
    impact_summary,
)

from src.intelligence.incidents import (
    IncidentConfig,
    IncidentEngine,
    incident_summary,
)

from src.intelligence.patterns import (
    PatternConfig,
    PatternEngine,
    pattern_statistics,
)

from src.intelligence.root_cause import (
    RootCauseAnalyzer,
    RootCauseConfig,
    root_cause_summary,
)

from src.intelligence.schema import (
    IntelligenceResult,
)


@dataclass
class IntelligencePipelineOutput:
    """
    Complete output of one intelligence run.

    The correlation output is preserved so later
    runtime, API and memory layers can inspect the
    original registry, topology and groups.
    """

    result: IntelligenceResult

    correlation: CorrelationPipelineOutput

    elapsed_seconds: float = 0.0

    def summary(
        self,
    ) -> Dict[str, Any]:
        records = (
            self.correlation.
            registry.records()
        )

        return {
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "input_record_count": (
                self.correlation.
                input_record_count
            ),
            "valid_record_count": (
                self.correlation.
                valid_record_count
            ),
            "rejected_record_count": (
                self.correlation.
                rejected_record_count
            ),
            "correlation_group_count": len(
                self.correlation.
                result.groups
            ),
            "patterns": (
                pattern_statistics(
                    records,
                    self.result.patterns,
                ).to_dict()
            ),
            "anomalies": (
                anomaly_summary(
                    self.result.anomalies
                )
            ),
            "incidents": (
                incident_summary(
                    self.result.incidents
                )
            ),
            "root_causes": (
                root_cause_summary(
                    self.result.incidents
                )
            ),
            "impact": impact_summary(
                self.result.incidents
            ),
            "confidence": (
                confidence_summary(
                    self.result.incidents
                )
            ),
            "explanations": (
                explanation_summary_stats(
                    self.result.
                    explanations
                )
            ),
        }

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "intelligence": (
                self.result.to_dict()
            ),
            "correlation_summary": (
                self.correlation.summary()
            ),
        }


# ---------------------------------------------------------------
# Intelligence execution
# ---------------------------------------------------------------

def run_pipeline(
    correlation: CorrelationPipelineOutput,
    *,
    pattern_config: Optional[
        PatternConfig
    ] = None,
    anomaly_config: Optional[
        AnomalyConfig
    ] = None,
    incident_config: Optional[
        IncidentConfig
    ] = None,
    root_cause_config: Optional[
        RootCauseConfig
    ] = None,
    impact_config: Optional[
        ImpactConfig
    ] = None,
    confidence_config: Optional[
        ConfidenceConfig
    ] = None,
) -> IntelligencePipelineOutput:
    """
    Run the complete deterministic intelligence
    pipeline over a completed correlation run.

    Execution order:

      patterns
      anomalies
      incidents
      root causes
      impact
      confidence
      explanations
    """

    started = time.monotonic()

    records = (
        correlation.registry.records()
    )

    groups = (
        correlation.result.groups
    )

    registry = correlation.registry

    topology = correlation.topology

    # ---------------------------------------------------------
    # Patterns
    # ---------------------------------------------------------

    pattern_engine = PatternEngine(
        config=(
            pattern_config
            or PatternConfig()
        )
    )

    pattern_engine.analyze(
        records
    )

    group_patterns = (
        pattern_engine.analyze_groups(
            groups
        )
    )

    patterns = (
        pattern_engine.patterns()
    )

    # ---------------------------------------------------------
    # Anomalies
    # ---------------------------------------------------------

    anomaly_engine = AnomalyEngine(
        config=(
            anomaly_config
            or AnomalyConfig()
        )
    )

    anomalies = anomaly_engine.analyze(
        records,
        patterns=patterns,
        groups=groups,
    )

    # ---------------------------------------------------------
    # Incident candidates
    # ---------------------------------------------------------

    incident_engine = IncidentEngine(
        config=(
            incident_config
            or IncidentConfig()
        )
    )

    incidents = incident_engine.analyze(
        groups,
        anomalies,
        patterns=patterns,
    )

    # ---------------------------------------------------------
    # Root-cause hypotheses
    # ---------------------------------------------------------

    root_cause_analyzer = (
        RootCauseAnalyzer(
            registry,
            topology=topology,
            config=(
                root_cause_config
                or RootCauseConfig()
            ),
        )
    )

    incidents = (
        root_cause_analyzer.analyze(
            incidents,
            anomalies,
        )
    )

    # ---------------------------------------------------------
    # Impact
    # ---------------------------------------------------------

    impact_analyzer = (
        ImpactAnalyzer(
            registry,
            topology=topology,
            config=(
                impact_config
                or ImpactConfig()
            ),
        )
    )

    incidents = (
        impact_analyzer.analyze(
            incidents,
            anomalies,
        )
    )

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    confidence_analyzer = (
        ConfidenceAnalyzer(
            registry,
            topology=topology,
            config=(
                confidence_config
                or ConfidenceConfig()
            ),
        )
    )

    incidents = (
        confidence_analyzer.analyze(
            incidents,
            anomalies,
            groups,
        )
    )

    # ---------------------------------------------------------
    # Operator explanations
    # ---------------------------------------------------------

    explanation_engine = (
        ExplanationEngine(
            registry,
            topology=topology,
        )
    )

    explanations = (
        explanation_engine.explain(
            incidents,
            anomalies,
            patterns,
            groups,
        )
    )

    classified_group_ids = {
        group_id
        for incident in incidents
        for group_id
        in incident.
        correlation_group_ids
    }

    unclassified_group_ids = sorted(
        group.group_id
        for group in groups
        if (
            group.group_id
            and group.group_id
            not in classified_group_ids
        )
    )

    providers = {
        telemetry.resource.provider
        for telemetry in records
    }

    if len(providers) == 1:
        provider_mode = (
            f"{next(iter(providers))}"
            "-only"
        )

    elif len(providers) > 1:
        provider_mode = (
            "multi-provider"
        )

    else:
        provider_mode = "empty"

    result = IntelligenceResult(
        incidents=incidents,
        anomalies=anomalies,
        patterns=patterns,
        explanations=explanations,
        unclassified_group_ids=(
            unclassified_group_ids
        ),
        metadata={
            "provider_mode": (
                provider_mode
            ),
            "providers": sorted(
                providers
            ),
            "record_count": len(
                records
            ),
            "correlation_group_count": (
                len(groups)
            ),
            "group_pattern_count": (
                len(group_patterns)
            ),
            "topology": (
                topology.summary()
            ),
        },
    ).finalize()

    elapsed = (
        time.monotonic()
        - started
    )

    result.metadata[
        "elapsed_seconds"
    ] = round(
        elapsed,
        6,
    )

    return IntelligencePipelineOutput(
        result=result,
        correlation=correlation,
        elapsed_seconds=elapsed,
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
    ) as output:
        json.dump(
            value,
            output,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        output.write("\n")


def write_jsonl(
    path: Path,
    values: Iterable[
        Mapping[str, Any]
    ],
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as output:
        for value in values:
            output.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                    default=str,
                )
                + "\n"
            )

            count += 1

    return count


def write_explanations_text(
    path: Path,
    output: IntelligencePipelineOutput,
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as destination:
        for (
            index,
            explanation,
        ) in enumerate(
            output.result.
            explanations
        ):
            if index:
                destination.write(
                    "\n\n"
                    + "=" * 80
                    + "\n\n"
                )

            destination.write(
                render_explanation(
                    explanation
                )
            )

            destination.write("\n")

    return len(
        output.result.explanations
    )


def write_pipeline_outputs(
    output: IntelligencePipelineOutput,
    *,
    result_path: Path,
    incidents_path: Path,
    anomalies_path: Path,
    patterns_path: Path,
    explanations_path: Path,
    explanations_text_path: Path,
) -> Dict[str, int]:
    write_json(
        result_path,
        output.to_dict(),
    )

    incidents_written = write_jsonl(
        incidents_path,
        (
            incident.to_dict()
            for incident
            in output.result.incidents
        ),
    )

    anomalies_written = write_jsonl(
        anomalies_path,
        (
            anomaly.to_dict()
            for anomaly
            in output.result.anomalies
        ),
    )

    patterns_written = write_jsonl(
        patterns_path,
        (
            pattern.to_dict()
            for pattern
            in output.result.patterns
        ),
    )

    explanations_written = write_jsonl(
        explanations_path,
        (
            explanation.to_dict()
            for explanation
            in output.result.
            explanations
        ),
    )

    text_explanations_written = (
        write_explanations_text(
            explanations_text_path,
            output,
        )
    )

    return {
        "incidents": (
            incidents_written
        ),
        "anomalies": (
            anomalies_written
        ),
        "patterns": (
            patterns_written
        ),
        "explanations": (
            explanations_written
        ),
        "text_explanations": (
            text_explanations_written
        ),
    }


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OSHA correlation and "
            "intelligence over normalized "
            "telemetry JSONL."
        )
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Normalized telemetry JSONL input. "
            "Repeat for additional providers."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "orchestration-ai-data/"
            "intelligence/result.json"
        ),
    )

    parser.add_argument(
        "--incidents-output",
        default=(
            "orchestration-ai-data/"
            "intelligence/incidents.jsonl"
        ),
    )

    parser.add_argument(
        "--anomalies-output",
        default=(
            "orchestration-ai-data/"
            "intelligence/anomalies.jsonl"
        ),
    )

    parser.add_argument(
        "--patterns-output",
        default=(
            "orchestration-ai-data/"
            "intelligence/patterns.jsonl"
        ),
    )

    parser.add_argument(
        "--explanations-output",
        default=(
            "orchestration-ai-data/"
            "intelligence/"
            "explanations.jsonl"
        ),
    )

    parser.add_argument(
        "--explanations-text",
        default=(
            "orchestration-ai-data/"
            "intelligence/"
            "explanations.txt"
        ),
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
        "--minimum-pattern-occurrences",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--include-singleton-patterns",
        action="store_true",
    )

    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--incident-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--root-cause-minimum-score",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--root-cause-likely-threshold",
        type=float,
        default=0.70,
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
        raise ValueError(
            "--window-seconds must be > 0"
        )

    if args.step_seconds <= 0:
        raise ValueError(
            "--step-seconds must be > 0"
        )

    if (
        args.window_mode == "sliding"
        and args.step_seconds
        > args.window_seconds
    ):
        raise ValueError(
            "--step-seconds must not exceed "
            "--window-seconds"
        )

    if (
        args.maximum_pairs_per_window
        < 1
    ):
        raise ValueError(
            "--maximum-pairs-per-window "
            "must be >= 1"
        )

    if (
        args.minimum_pattern_occurrences
        < 1
    ):
        raise ValueError(
            "--minimum-pattern-occurrences "
            "must be >= 1"
        )

    scores = {
        "--anomaly-threshold": (
            args.anomaly_threshold
        ),
        "--incident-threshold": (
            args.incident_threshold
        ),
        "--root-cause-minimum-score": (
            args.
            root_cause_minimum_score
        ),
        "--root-cause-likely-threshold": (
            args.
            root_cause_likely_threshold
        ),
    }

    for name, value in scores.items():
        if not (
            0.0
            <= value
            <= 1.0
        ):
            raise ValueError(
                f"{name} must be "
                "between 0 and 1"
            )

    if (
        args.root_cause_minimum_score
        > args.
        root_cause_likely_threshold
    ):
        raise ValueError(
            "--root-cause-minimum-score "
            "must not exceed "
            "--root-cause-likely-threshold"
        )


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

    input_paths = [
        Path(value)
        for value in args.input
    ]

    missing = [
        path
        for path in input_paths
        if not path.is_file()
    ]

    if missing:
        for path in missing:
            print(
                (
                    "Input file does not "
                    f"exist: {path}"
                ),
                file=sys.stderr,
            )

        return 2

    records, load_issues = (
        load_telemetry_files(
            input_paths
        )
    )

    try:
        correlation = (
            run_correlation_pipeline(
                records,
                window_mode=(
                    args.window_mode
                ),
                window_seconds=(
                    args.window_seconds
                ),
                step_seconds=(
                    args.step_seconds
                ),
                maximum_pairs_per_window=(
                    args.
                    maximum_pairs_per_window
                ),
                load_issues=(
                    load_issues
                ),
            )
        )

        output = run_pipeline(
            correlation,
            pattern_config=(
                PatternConfig(
                    minimum_occurrences=(
                        args.
                        minimum_pattern_occurrences
                    ),
                    include_singletons=(
                        args.
                        include_singleton_patterns
                    ),
                )
            ),
            anomaly_config=(
                AnomalyConfig(
                    anomaly_threshold=(
                        args.
                        anomaly_threshold
                    )
                )
            ),
            incident_config=(
                IncidentConfig(
                    incident_threshold=(
                        args.
                        incident_threshold
                    )
                )
            ),
            root_cause_config=(
                RootCauseConfig(
                    minimum_score=(
                        args.
                        root_cause_minimum_score
                    ),
                    likely_threshold=(
                        args.
                        root_cause_likely_threshold
                    ),
                )
            ),
        )

    except Exception as exc:
        print(
            (
                "Intelligence pipeline failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            file=sys.stderr,
        )

        return 1

    written = write_pipeline_outputs(
        output,
        result_path=Path(
            args.output
        ),
        incidents_path=Path(
            args.incidents_output
        ),
        anomalies_path=Path(
            args.anomalies_output
        ),
        patterns_path=Path(
            args.patterns_output
        ),
        explanations_path=Path(
            args.explanations_output
        ),
        explanations_text_path=Path(
            args.explanations_text
        ),
    )

    summary = output.summary()

    print(
        "Intelligence pipeline complete."
    )

    print(
        "Valid records: "
        f"{summary['valid_record_count']}"
    )

    print(
        "Rejected records: "
        f"{summary['rejected_record_count']}"
    )

    print(
        "Correlation groups: "
        f"{summary['correlation_group_count']}"
    )

    print(
        "Patterns: "
        f"{written['patterns']}"
    )

    print(
        "Anomalies: "
        f"{written['anomalies']}"
    )

    print(
        "Incidents: "
        f"{written['incidents']}"
    )

    print(
        "Explanations: "
        f"{written['explanations']}"
    )

    print(
        f"Result: {args.output}"
    )

    print(
        "Operator explanations: "
        f"{args.explanations_text}"
    )

    if (
        args.fail_on_invalid
        and (
            correlation.
            rejected_record_count
            > 0
            or correlation.load_issues
        )
    ):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
