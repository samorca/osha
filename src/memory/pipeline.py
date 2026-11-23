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

from src.intelligence.pipeline import (
    IntelligencePipelineOutput,
    run_pipeline as run_intelligence_pipeline,
)

from src.memory.baselines import (
    BaselineConfig,
    BaselineManager,
)

from src.memory.episodes import (
    EpisodeMemoryManager,
)

from src.memory.incidents import (
    IncidentMemoryManager,
)

from src.memory.patterns import (
    PatternMemoryManager,
)

from src.memory.retention import (
    RetentionManager,
    RetentionPolicy,
    RetentionRunResult,
)

from src.memory.retrieval import (
    MemoryRetriever,
    RetrievalConfig,
)

from src.memory.store import (
    MemoryStore,
    create_memory_store,
)


@dataclass
class MemoryPipelineOutput:
    """
    Complete output from one memory pipeline run.

    The memory store itself is kept outside the
    serialized result. Only store statistics and
    bounded retrieval context are emitted.
    """

    store: MemoryStore

    episode_count: int

    pattern_count: int

    baseline_count: int

    incident_count: int

    retention: RetentionRunResult

    historical_context: Dict[
        str,
        Any,
    ]

    elapsed_seconds: float

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "episodes_written": (
                self.episode_count
            ),
            "patterns_written": (
                self.pattern_count
            ),
            "baselines_touched": (
                self.baseline_count
            ),
            "incidents_written": (
                self.incident_count
            ),
            "store": (
                self.store.
                stats().to_dict()
            ),
            "retention": (
                self.retention.to_dict()
            ),
        }

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "historical_context": (
                self.historical_context
            ),
        }


# ---------------------------------------------------------------
# Intelligence-to-memory mappings
# ---------------------------------------------------------------

def incident_by_group(
    output: IntelligencePipelineOutput,
) -> Dict[str, str]:
    """
    Build:

        correlation_group_id -> incident_id

    This links only groups that intelligence
    actually promoted into incident candidates.
    """

    mapping: Dict[
        str,
        str,
    ] = {}

    for incident in (
        output.result.incidents
    ):
        incident.finalize()

        if not incident.incident_id:
            continue

        for group_id in (
            incident.
            correlation_group_ids
        ):
            mapping[
                group_id
            ] = incident.incident_id

    return mapping


def incident_ids_by_pattern(
    output: IntelligencePipelineOutput,
) -> Dict[
    str,
    list[str],
]:
    """
    Build:

        pattern_id -> incident IDs

    This association remains unverified until an
    authoritative workflow explicitly classifies
    the pattern.
    """

    mapping: Dict[
        str,
        list[str],
    ] = {}

    for incident in (
        output.result.incidents
    ):
        incident.finalize()

        if not incident.incident_id:
            continue

        for pattern_id in (
            incident.pattern_ids
        ):
            mapping.setdefault(
                pattern_id,
                [],
            ).append(
                incident.incident_id
            )

    return mapping


# ---------------------------------------------------------------
# Historical context
# ---------------------------------------------------------------

def build_historical_context(
    store: MemoryStore,
    output: IntelligencePipelineOutput,
    *,
    retrieval_config: Optional[
        RetrievalConfig
    ] = None,
) -> Dict[str, Any]:
    """
    Build bounded retrieval context after the
    current run has been written.

    This output is diagnostic context for the
    current run. Future architecture can retrieve
    history before intelligence analysis so it
    contributes to anomaly/root-cause scoring.
    """

    retriever = MemoryRetriever(
        store,
        config=(
            retrieval_config
            or RetrievalConfig()
        ),
    )

    incident_contexts = []

    for incident in (
        output.result.incidents
    ):
        incident_contexts.append(
            {
                "incident_id": (
                    incident.incident_id
                ),
                "context": (
                    retriever.context(
                        incident=incident,
                        patterns=(
                            output.result.
                            patterns
                        ),
                        resource_ids=(
                            incident.
                            resource_ids
                        ),
                    )
                ),
            }
        )

    return {
        "incidents": (
            incident_contexts
        ),
        "patterns": (
            retriever.
            pattern_context(
                output.result.patterns
            )
        ),
    }


# ---------------------------------------------------------------
# Main memory pipeline
# ---------------------------------------------------------------

def run_pipeline(
    intelligence: IntelligencePipelineOutput,
    store: MemoryStore,
    *,
    baseline_config: Optional[
        BaselineConfig
    ] = None,
    retrieval_config: Optional[
        RetrievalConfig
    ] = None,
    retention_policy: Optional[
        RetentionPolicy
    ] = None,
    retention_dry_run: bool = True,
) -> MemoryPipelineOutput:
    """
    Persist one completed intelligence run.

    Execution order:

      1. correlation episodes
      2. intelligence patterns
      3. numeric metric baselines
      4. incidents and root-cause hypotheses
      5. historical retrieval context
      6. retention evaluation

    Retention defaults to dry-run.
    """

    started = time.monotonic()

    records = (
        intelligence.correlation.
        registry.records()
    )

    groups = (
        intelligence.correlation.
        result.groups
    )

    # ---------------------------------------------------------
    # Episode memory
    # ---------------------------------------------------------

    episode_manager = (
        EpisodeMemoryManager(
            store
        )
    )

    episodes = (
        episode_manager.
        remember_many(
            groups,
            incident_by_group=(
                incident_by_group(
                    intelligence
                )
            ),
        )
    )

    # ---------------------------------------------------------
    # Pattern memory
    # ---------------------------------------------------------

    pattern_manager = (
        PatternMemoryManager(
            store
        )
    )

    pattern_memories = (
        pattern_manager.
        remember_many(
            intelligence.result.
            patterns,
            incident_ids_by_pattern=(
                incident_ids_by_pattern(
                    intelligence
                )
            ),
        )
    )

    # ---------------------------------------------------------
    # Baseline memory
    # ---------------------------------------------------------

    baseline_manager = (
        BaselineManager(
            store,
            config=(
                baseline_config
                or BaselineConfig()
            ),
        )
    )

    baselines = (
        baseline_manager.
        remember_many(
            records
        )
    )

    # ---------------------------------------------------------
    # Incident and root-cause memory
    # ---------------------------------------------------------

    incident_manager = (
        IncidentMemoryManager(
            store
        )
    )

    incident_memories = (
        incident_manager.
        remember_many(
            intelligence.result.
            incidents,
            explanations=(
                intelligence.result.
                explanations
            ),
        )
    )

    # ---------------------------------------------------------
    # Historical context
    # ---------------------------------------------------------

    context = (
        build_historical_context(
            store,
            intelligence,
            retrieval_config=(
                retrieval_config
            ),
        )
    )

    # ---------------------------------------------------------
    # Retention
    # ---------------------------------------------------------

    retention_manager = (
        RetentionManager(
            store,
            policy=(
                retention_policy
                or RetentionPolicy()
            ),
        )
    )

    retention = (
        retention_manager.run(
            dry_run=(
                retention_dry_run
            )
        )
    )

    elapsed = (
        time.monotonic()
        - started
    )

    return MemoryPipelineOutput(
        store=store,
        episode_count=len(
            episodes
        ),
        pattern_count=len(
            pattern_memories
        ),
        baseline_count=len(
            baselines
        ),
        incident_count=len(
            incident_memories
        ),
        retention=retention,
        historical_context=context,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------
# Output writer
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
            "intelligence and memory over "
            "normalized telemetry JSONL."
        )
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Normalized telemetry JSONL file. "
            "Repeat for additional inputs."
        ),
    )

    parser.add_argument(
        "--backend",
        choices=(
            "memory",
            "json",
        ),
        default="json",
    )

    parser.add_argument(
        "--store",
        default=(
            "orchestration-ai-data/"
            "memory/memory.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "orchestration-ai-data/"
            "memory/result.json"
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
        "--minimum-baseline-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--baseline-z-score",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--apply-retention",
        action="store_true",
        help=(
            "Apply archive/expiry decisions. "
            "Without this flag, retention is "
            "dry-run only."
        ),
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
        args.minimum_baseline_samples
        < 2
    ):
        raise ValueError(
            "--minimum-baseline-samples "
            "must be >= 2"
        )

    if args.baseline_z_score <= 0:
        raise ValueError(
            "--baseline-z-score "
            "must be > 0"
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

    try:
        records, load_issues = (
            load_telemetry_files(
                input_paths
            )
        )

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

        intelligence = (
            run_intelligence_pipeline(
                correlation
            )
        )

        store = create_memory_store(
            backend=args.backend,
            path=(
                args.store
                if args.backend
                == "json"
                else None
            ),
        )

        output = run_pipeline(
            intelligence,
            store,
            baseline_config=(
                BaselineConfig(
                    minimum_samples=(
                        args.
                        minimum_baseline_samples
                    ),
                    anomaly_z_score=(
                        args.
                        baseline_z_score
                    ),
                )
            ),
            retention_dry_run=(
                not args.
                apply_retention
            ),
        )

        write_json(
            Path(args.output),
            output.to_dict(),
        )

    except Exception as exc:
        print(
            (
                "Memory pipeline failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            file=sys.stderr,
        )

        return 1

    summary = output.summary()

    print(
        "Memory pipeline complete."
    )

    print(
        "Valid records: "
        f"{correlation.valid_record_count}"
    )

    print(
        "Rejected records: "
        f"{correlation.rejected_record_count}"
    )

    print(
        "Episodes written: "
        f"{summary['episodes_written']}"
    )

    print(
        "Patterns written: "
        f"{summary['patterns_written']}"
    )

    print(
        "Baselines touched: "
        f"{summary['baselines_touched']}"
    )

    print(
        "Incidents written: "
        f"{summary['incidents_written']}"
    )

    print(
        "Memory records: "
        f"{summary['store']['memory_count']}"
    )

    print(
        "Retention dry run: "
        f"{not args.apply_retention}"
    )

    print(
        f"Result: {args.output}"
    )

    if args.backend == "json":
        print(
            f"Memory store: {args.store}"
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
