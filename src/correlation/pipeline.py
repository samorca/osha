from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from src.correlation.correlate import (
    CorrelationEngine,
)

from src.correlation.registry import (
    ResourceRegistry,
)

from src.correlation.relationships import (
    RelationshipExtractor,
    relationship_summary,
)

from src.correlation.schema import (
    CorrelationResult,
)

from src.correlation.scoring import (
    ScoringConfig,
)

from src.correlation.topology import (
    TopologyGraph,
    build_topology,
)

from src.telemetry.schema import (
    Relationship,
    ResourceIdentity,
    Signal,
    TelemetryRecord,
)

from src.telemetry.validation import (
    ValidationResult,
    validate_record,
    validation_summary,
)


@dataclass
class LoadIssue:
    """
    Problem encountered while loading canonical
    telemetry input.
    """

    source: str

    line_number: int

    error_type: str

    message: str

    raw_line: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "line_number": (self.line_number),
            "error_type": (self.error_type),
            "message": self.message,
            "raw_line": (self.raw_line),
        }


@dataclass
class PipelineOutput:
    """
    Complete output from one correlation run.
    """

    result: CorrelationResult

    registry: ResourceRegistry

    topology: TopologyGraph

    validation_results: List[ValidationResult] = field(default_factory=list)

    load_issues: List[LoadIssue] = field(default_factory=list)

    input_record_count: int = 0

    valid_record_count: int = 0

    rejected_record_count: int = 0

    elapsed_seconds: float = 0.0

    def summary(self) -> Dict[str, Any]:
        metadata_summary = (
            self.result.metadata.get(
                "summary",
                {},
            )
            if isinstance(
                self.result.metadata,
                Mapping,
            )
            else {}
        )

        return {
            "input_record_count": (self.input_record_count),
            "valid_record_count": (self.valid_record_count),
            "rejected_record_count": (self.rejected_record_count),
            "load_issue_count": len(self.load_issues),
            "elapsed_seconds": (
                round(
                    self.elapsed_seconds,
                    6,
                )
            ),
            "registry": (self.registry.stats().to_dict()),
            "relationships": (relationship_summary(self.topology.relationships())),
            "topology": (self.topology.summary()),
            "validation": (validation_summary(self.validation_results)),
            "correlation": (metadata_summary),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "correlation": (self.result.to_dict()),
            "load_issues": [issue.to_dict() for issue in self.load_issues],
        }


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def mapping_or_empty(
    value: Any,
) -> Dict[str, Any]:
    if isinstance(
        value,
        Mapping,
    ):
        return dict(value)

    return {}


def list_or_empty(
    value: Any,
) -> List[Any]:
    if isinstance(
        value,
        list,
    ):
        return value

    return []


# ---------------------------------------------------------------
# Canonical telemetry reconstruction
# ---------------------------------------------------------------


def resource_from_dict(
    value: Mapping[str, Any],
) -> ResourceIdentity:
    return ResourceIdentity(
        resource_type=str(
            value.get(
                "resource_type",
                "unknown",
            )
        ),
        provider=str(
            value.get(
                "provider",
                "other",
            )
        ),
        provider_id=str(
            value.get(
                "provider_id",
                "unknown-resource",
            )
        ),
        canonical_id=(optional_string(value.get("canonical_id"))),
        native_type=(optional_string(value.get("native_type"))),
        name=optional_string(value.get("name")),
        environment_id=str(
            value.get(
                "environment_id",
                "default",
            )
            or "default"
        ),
        region=optional_string(value.get("region")),
        zone=optional_string(value.get("zone")),
        cluster_id=(optional_string(value.get("cluster_id"))),
        namespace=(optional_string(value.get("namespace"))),
        host=optional_string(value.get("host")),
        service=optional_string(value.get("service")),
        attributes=(mapping_or_empty(value.get("attributes"))),
    )


def signal_from_dict(
    value: Mapping[str, Any],
) -> Signal:
    return Signal(
        signal_type=str(
            value.get(
                "signal_type",
                "unknown",
            )
        ),
        timestamp=str(
            value.get(
                "timestamp",
                "",
            )
        ),
        provider=str(
            value.get(
                "provider",
                "other",
            )
        ),
        resource_id=str(
            value.get(
                "resource_id",
                "",
            )
        ),
        environment_id=str(
            value.get(
                "environment_id",
                "default",
            )
            or "default"
        ),
        signal_name=(optional_string(value.get("signal_name"))),
        severity=str(
            value.get(
                "severity",
                "unknown",
            )
        ),
        value=value.get("value"),
        unit=optional_string(value.get("unit")),
        message=optional_string(value.get("message")),
        attributes=(mapping_or_empty(value.get("attributes"))),
        source_record_id=(optional_string(value.get("source_record_id"))),
    )


def relationship_from_dict(
    value: Mapping[str, Any],
) -> Relationship:
    confidence = value.get("confidence")

    if confidence is not None:
        try:
            confidence = float(confidence)
        except (
            TypeError,
            ValueError,
        ):
            confidence = None

    return Relationship(
        source_resource_id=str(
            value.get(
                "source_resource_id",
                "",
            )
        ),
        target_resource_id=str(
            value.get(
                "target_resource_id",
                "",
            )
        ),
        relationship_type=str(
            value.get(
                "relationship_type",
                "related_to",
            )
        ),
        provider=str(
            value.get(
                "provider",
                "other",
            )
        ),
        environment_id=str(
            value.get(
                "environment_id",
                "default",
            )
            or "default"
        ),
        timestamp=str(
            value.get(
                "timestamp",
                "",
            )
        ),
        confidence=confidence,
        evidence=(mapping_or_empty(value.get("evidence"))),
    )


def telemetry_from_dict(
    value: Mapping[str, Any],
) -> TelemetryRecord:
    """
    Reconstruct one TelemetryRecord from
    normalized JSON.
    """

    resource_data = value.get("resource")

    signal_data = value.get("signal")

    if not isinstance(
        resource_data,
        Mapping,
    ):
        raise ValueError("Telemetry record has no resource object")

    if not isinstance(
        signal_data,
        Mapping,
    ):
        raise ValueError("Telemetry record has no signal object")

    relationships = []

    for item in list_or_empty(value.get("relationships")):
        if not isinstance(
            item,
            Mapping,
        ):
            continue

        relationships.append(relationship_from_dict(item))

    telemetry = TelemetryRecord(
        resource=resource_from_dict(resource_data),
        signal=signal_from_dict(signal_data),
        relationships=relationships,
        native=(mapping_or_empty(value.get("native"))),
        schema_version=str(
            value.get(
                "schema_version",
                "1",
            )
        ),
        normalized_at=str(
            value.get(
                "normalized_at",
                "",
            )
        ),
        record_id=(optional_string(value.get("record_id"))),
    )

    return telemetry.finalize()


# ---------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------


def read_jsonl(
    path: Path,
) -> Iterator[
    tuple[
        int,
        Mapping[str, Any],
    ]
]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            value = json.loads(text)

            if not isinstance(
                value,
                Mapping,
            ):
                raise TypeError("JSONL record must be an object")

            yield (
                line_number,
                value,
            )


def load_telemetry_file(
    path: Path,
    *,
    secure_transport: Any = None,
    require_secure: bool = False,
) -> tuple[
    List[TelemetryRecord],
    List[LoadIssue],
]:
    if require_secure and secure_transport is None:
        raise ValueError(
            "require_secure needs a configured secure telemetry transport."
        )
    if secure_transport is not None:
        from src.security.integration import SecurityBoundaryError

    records: List[TelemetryRecord] = []

    issues: List[LoadIssue] = []

    try:
        source = path.open(
            "r",
            encoding="utf-8",
        )

    except OSError as exc:
        issues.append(
            LoadIssue(
                source=str(path),
                line_number=0,
                error_type=(type(exc).__name__),
                message=str(exc),
            )
        )

        return (
            records,
            issues,
        )

    with source:
        for (
            line_number,
            line,
        ) in enumerate(
            source,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            try:
                value = json.loads(text)

                if not isinstance(
                    value,
                    Mapping,
                ):
                    raise TypeError("JSONL record must be an object")

                if secure_transport is not None:
                    value = secure_transport.open(value)

                records.append(telemetry_from_dict(value))

            except Exception as exc:
                if secure_transport is not None:
                    from src.security.integration import SecurityBoundaryError

                    if isinstance(exc, SecurityBoundaryError):
                        raise
                    raise SecurityBoundaryError(
                        "Authenticated telemetry payload is invalid."
                    ) from exc
                issues.append(
                    LoadIssue(
                        source=str(path),
                        line_number=(line_number),
                        error_type=(type(exc).__name__),
                        message=str(exc),
                        raw_line=(
                            None
                            if secure_transport is not None
                            else text[:4096]
                        ),
                    )
                )

    return (
        records,
        issues,
    )


def load_telemetry_files(
    paths: Iterable[Path],
    *,
    secure_transport: Any = None,
    require_secure: bool = False,
) -> tuple[
    List[TelemetryRecord],
    List[LoadIssue],
]:
    if require_secure and secure_transport is None:
        raise ValueError(
            "require_secure needs a configured secure telemetry transport."
        )

    records: List[TelemetryRecord] = []

    issues: List[LoadIssue] = []

    seen_record_ids = set()

    for path in paths:
        loaded, file_issues = load_telemetry_file(
            path,
            secure_transport=secure_transport,
            require_secure=require_secure,
        )

        issues.extend(file_issues)

        for telemetry in loaded:
            telemetry.finalize()

            if not telemetry.record_id:
                continue

            if telemetry.record_id in seen_record_ids:
                continue

            seen_record_ids.add(telemetry.record_id)

            records.append(telemetry)

    return (
        records,
        issues,
    )


# ---------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------


def validate_records(
    records: Iterable[TelemetryRecord],
    *,
    include_invalid: bool = False,
) -> tuple[
    List[TelemetryRecord],
    List[ValidationResult],
    int,
]:
    accepted = []

    results = []

    rejected = 0

    for telemetry in records:
        result = validate_record(telemetry)

        results.append(result)

        if result.valid:
            accepted.append(telemetry)

        elif include_invalid:
            accepted.append(telemetry)

        else:
            rejected += 1

    return (
        accepted,
        results,
        rejected,
    )


def collect_embedded_relationships(
    records: Iterable[TelemetryRecord],
) -> List[Relationship]:
    """
    Preserve relationships already present in
    normalized records.
    """

    output = []

    seen = set()

    for telemetry in records:
        for relationship in telemetry.relationships:
            key = (
                relationship.source_resource_id,
                relationship.target_resource_id,
                relationship.relationship_type,
            )

            if key in seen:
                continue

            seen.add(key)

            output.append(relationship)

    return output


def run_pipeline(
    records: Iterable[TelemetryRecord],
    *,
    window_mode: str = "sliding",
    window_seconds: float = 60.0,
    step_seconds: float = 30.0,
    cross_provider_only: bool = False,
    maximum_pairs_per_window: int = (100_000),
    include_rejected_candidates: bool = (False),
    include_weak_in_groups: bool = False,
    include_invalid_records: bool = False,
    scoring_config: Optional[ScoringConfig] = None,
    load_issues: Optional[List[LoadIssue]] = None,
) -> PipelineOutput:
    """
    Execute the complete correlation pipeline.

    Steps:

      validate canonical telemetry
      build resource registry
      extract relationships
      construct topology graph
      correlate records
    """

    started = time.monotonic()

    input_records = list(records)

    (
        valid_records,
        validation_results,
        rejected_count,
    ) = validate_records(
        input_records,
        include_invalid=(include_invalid_records),
    )

    registry = ResourceRegistry()

    registry.register_many(valid_records)

    extractor = RelationshipExtractor(registry)

    extracted_relationships = extractor.extract_many(valid_records)

    embedded_relationships = collect_embedded_relationships(valid_records)

    all_relationships = [
        *embedded_relationships,
        *extracted_relationships,
    ]

    topology = build_topology(
        registry,
        all_relationships,
    )

    configuration = scoring_config or ScoringConfig()

    engine = CorrelationEngine(
        topology=topology,
        scoring_config=(configuration),
        window_mode=window_mode,
        window_seconds=(window_seconds),
        step_seconds=(step_seconds),
        cross_provider_only=(cross_provider_only),
        maximum_pairs_per_window=(maximum_pairs_per_window),
        include_rejected=(include_rejected_candidates),
        include_weak_in_groups=(include_weak_in_groups),
    )

    result = engine.correlate(valid_records)

    elapsed = time.monotonic() - started

    result.metadata["pipeline"] = {
        "input_record_count": (len(input_records)),
        "valid_record_count": (len(valid_records)),
        "rejected_record_count": (rejected_count),
        "load_issue_count": len(load_issues or []),
        "registry": (registry.stats().to_dict()),
        "topology": (topology.summary()),
        "relationships": (relationship_summary(topology.relationships())),
        "elapsed_seconds": (
            round(
                elapsed,
                6,
            )
        ),
    }

    return PipelineOutput(
        result=result,
        registry=registry,
        topology=topology,
        validation_results=(validation_results),
        load_issues=list(load_issues or []),
        input_record_count=(len(input_records)),
        valid_record_count=(len(valid_records)),
        rejected_record_count=(rejected_count),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------


def write_json(
    path: Path,
    value: Mapping[str, Any],
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


def write_candidates_jsonl(
    path: Path,
    result: CorrelationResult,
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
        for candidate in result.candidates:
            output.write(
                json.dumps(
                    candidate.to_dict(),
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


def write_groups_jsonl(
    path: Path,
    result: CorrelationResult,
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
        for group in result.groups:
            output.write(
                json.dumps(
                    group.to_dict(),
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


def write_relationships_jsonl(
    path: Path,
    topology: TopologyGraph,
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
        for relationship in topology.relationships():
            output.write(
                json.dumps(
                    relationship.to_dict(),
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


def write_load_issues_jsonl(
    path: Path,
    issues: Iterable[LoadIssue],
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
        for issue in issues:
            output.write(
                json.dumps(
                    issue.to_dict(),
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


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run OSHA correlation over normalized telemetry JSONL.")
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Normalized telemetry JSONL file. "
            "Repeat this option for multiple "
            "providers/files."
        ),
    )

    parser.add_argument(
        "--output",
        default=("dataset/correlated/result.json"),
    )

    parser.add_argument(
        "--groups-output",
        default=("dataset/correlated/groups.jsonl"),
    )

    parser.add_argument(
        "--candidates-output",
        default=("dataset/correlated/candidates.jsonl"),
    )

    parser.add_argument(
        "--relationships-output",
        default=("dataset/correlated/relationships.jsonl"),
    )

    parser.add_argument(
        "--load-errors-output",
        default=("dataset/quarantine/correlation-load-errors.jsonl"),
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
        "--cross-provider-only",
        action="store_true",
    )

    parser.add_argument(
        "--include-rejected-candidates",
        action="store_true",
    )

    parser.add_argument(
        "--include-weak-in-groups",
        action="store_true",
    )

    parser.add_argument(
        "--include-invalid-records",
        action="store_true",
        help=("Development/debug option. Invalid telemetry is normally excluded."),
    )

    parser.add_argument(
        "--correlated-threshold",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
    )

    return parser.parse_args()


def validate_cli_args(
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

    thresholds = (
        args.weak_threshold,
        args.candidate_threshold,
        args.correlated_threshold,
    )

    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Correlation thresholds must be between 0 and 1")

    if not (
        args.weak_threshold <= args.candidate_threshold <= args.correlated_threshold
    ):
        raise ValueError("Threshold order must be: weak <= candidate <= correlated")


def main() -> int:
    args = parse_args()

    try:
        validate_cli_args(args)

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

    records, load_issues = load_telemetry_files(input_paths)

    scoring_config = ScoringConfig(
        correlated_threshold=(args.correlated_threshold),
        candidate_threshold=(args.candidate_threshold),
        weak_threshold=(args.weak_threshold),
    )

    try:
        pipeline_output = run_pipeline(
            records,
            window_mode=(args.window_mode),
            window_seconds=(args.window_seconds),
            step_seconds=(args.step_seconds),
            cross_provider_only=(args.cross_provider_only),
            maximum_pairs_per_window=(args.maximum_pairs_per_window),
            include_rejected_candidates=(args.include_rejected_candidates),
            include_weak_in_groups=(args.include_weak_in_groups),
            include_invalid_records=(args.include_invalid_records),
            scoring_config=(scoring_config),
            load_issues=load_issues,
        )

    except Exception as exc:
        print(
            (f"Correlation pipeline failed: {type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )

        return 1

    output_path = Path(args.output)

    groups_path = Path(args.groups_output)

    candidates_path = Path(args.candidates_output)

    relationships_path = Path(args.relationships_output)

    errors_path = Path(args.load_errors_output)

    write_json(
        output_path,
        pipeline_output.to_dict(),
    )

    groups_written = write_groups_jsonl(
        groups_path,
        pipeline_output.result,
    )

    candidates_written = write_candidates_jsonl(
        candidates_path,
        pipeline_output.result,
    )

    relationships_written = write_relationships_jsonl(
        relationships_path,
        pipeline_output.topology,
    )

    issues_written = write_load_issues_jsonl(
        errors_path,
        pipeline_output.load_issues,
    )

    summary = pipeline_output.summary()

    print("Correlation pipeline complete.")

    print(f"Input records: {summary['input_record_count']}")

    print(f"Valid records: {summary['valid_record_count']}")

    print(f"Rejected records: {summary['rejected_record_count']}")

    print(f"Resources: {summary['registry']['resources']}")

    print(f"Topology edges: {summary['topology']['edge_count']}")

    print(f"Candidates: {candidates_written}")

    print(f"Groups: {groups_written}")

    print(f"Relationships: {relationships_written}")

    print(f"Load issues: {issues_written}")

    print(f"Result: {output_path}")

    print(f"Groups: {groups_path}")

    print(f"Candidates: {candidates_path}")

    if args.fail_on_invalid and (
        pipeline_output.rejected_record_count > 0 or pipeline_output.load_issues
    ):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
