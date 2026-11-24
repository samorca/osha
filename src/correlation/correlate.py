from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
    deque,
)
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.correlation.schema import (
    CorrelationCandidate,
    CorrelationDecision,
    CorrelationDirection,
    CorrelationGroup,
    CorrelationResult,
    GroupStatus,
)

from src.correlation.scoring import (
    CorrelationScorer,
    ScoringConfig,
)

from src.correlation.temporal import (
    TemporalAnalyzer,
    parse_timestamp,
)

from src.correlation.topology import (
    TopologyGraph,
)

from src.correlation.windows import (
    RecordWindow,
    fixed_windows,
    pair_candidates,
    sliding_windows,
)

from src.telemetry.schema import (
    Severity,
    SignalType,
    TelemetryRecord,
)


HistoricalScoreProvider = Callable[
    [
        TelemetryRecord,
        TelemetryRecord,
    ],
    Optional[float],
]


SUPPORTED_GROUP_DECISIONS = {
    CorrelationDecision.CORRELATED.value,
    CorrelationDecision.CANDIDATE.value,
}


def record_timestamp(
    telemetry: TelemetryRecord,
) -> Optional[datetime]:
    return parse_timestamp(telemetry.signal.timestamp)


def record_id(
    telemetry: TelemetryRecord,
) -> str:
    telemetry.finalize()

    if not telemetry.record_id:
        raise ValueError("Telemetry record has no record_id after finalize()")

    return telemetry.record_id


def candidate_pair_key(
    left: TelemetryRecord,
    right: TelemetryRecord,
) -> Tuple[str, str]:
    """
    Generate an order-independent pair key so
    overlapping windows don't create duplicate
    candidates.
    """

    left_id = record_id(left)
    right_id = record_id(right)

    return tuple(
        sorted(
            (
                left_id,
                right_id,
            )
        )
    )


def ordered_pair(
    left: TelemetryRecord,
    right: TelemetryRecord,
) -> Tuple[
    TelemetryRecord,
    TelemetryRecord,
]:
    """
    Give the earlier record the source position.

    If timestamps are equal or unavailable, use
    record ID ordering for determinism.
    """

    left_time = record_timestamp(left)

    right_time = record_timestamp(right)

    if left_time is not None and right_time is not None:
        if left_time < right_time:
            return (
                left,
                right,
            )

        if right_time < left_time:
            return (
                right,
                left,
            )

    if record_id(left) <= record_id(right):
        return (
            left,
            right,
        )

    return (
        right,
        left,
    )


def records_eligible(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    same_environment: bool = True,
) -> bool:
    """
    Apply inexpensive hard filters before
    generating evidence.
    """

    if record_id(source) == record_id(target):
        return False

    if (
        same_environment
        and source.resource.environment_id != target.resource.environment_id
    ):
        return False

    source_time = record_timestamp(source)

    target_time = record_timestamp(target)

    if source_time is None or target_time is None:
        return False

    return True


def candidate_direction(
    source: TelemetryRecord,
    target: TelemetryRecord,
    temporal: TemporalAnalyzer,
) -> str:
    relation = temporal.compare(
        source,
        target,
    )

    return relation.direction


def create_candidate(
    source: TelemetryRecord,
    target: TelemetryRecord,
    scorer: CorrelationScorer,
    *,
    historical_score: Optional[float] = None,
) -> CorrelationCandidate:
    """
    Create, enrich and classify one candidate
    pair.
    """

    source, target = ordered_pair(
        source,
        target,
    )

    source.finalize()
    target.finalize()

    evidence = scorer.evidence(
        source,
        target,
        historical_score=(historical_score),
    )

    candidate = CorrelationCandidate(
        source_record_id=(record_id(source)),
        target_record_id=(record_id(target)),
        source_resource_id=(source.resource.effective_id()),
        target_resource_id=(target.resource.effective_id()),
        source_timestamp=(source.signal.timestamp),
        target_timestamp=(target.signal.timestamp),
        environment_id=(source.resource.environment_id),
        source_provider=(source.resource.provider),
        target_provider=(target.resource.provider),
        source_signal_type=(source.signal.signal_type),
        target_signal_type=(target.signal.signal_type),
        direction=(
            candidate_direction(
                source,
                target,
                scorer.temporal,
            )
        ),
        evidence=evidence,
    )

    return scorer.candidate(candidate)


def generate_candidates_from_windows(
    windows: Iterable[RecordWindow],
    scorer: CorrelationScorer,
    *,
    historical_score_provider: Optional[HistoricalScoreProvider] = None,
    cross_provider_only: bool = False,
    maximum_pairs_per_window: int = 100_000,
    include_rejected: bool = False,
) -> List[CorrelationCandidate]:
    """
    Generate and score unique candidate pairs
    across one or more windows.

    Overlapping sliding windows may contain the
    same pair. Pairs are therefore deduplicated.
    """

    candidates: Dict[
        Tuple[str, str],
        CorrelationCandidate,
    ] = {}

    for record_window in windows:
        for left, right in pair_candidates(
            record_window,
            cross_provider_only=(cross_provider_only),
            maximum_pairs=(maximum_pairs_per_window),
        ):
            if not records_eligible(
                left,
                right,
            ):
                continue

            pair_key = candidate_pair_key(
                left,
                right,
            )

            if pair_key in candidates:
                continue

            history_score = None

            if historical_score_provider is not None:
                history_score = historical_score_provider(
                    left,
                    right,
                )

            candidate = create_candidate(
                left,
                right,
                scorer,
                historical_score=(history_score),
            )

            if (
                not include_rejected
                and candidate.decision == CorrelationDecision.REJECTED.value
            ):
                continue

            candidates[pair_key] = candidate

    output = list(candidates.values())

    output.sort(
        key=lambda item: (
            parse_timestamp(item.source_timestamp)
            or datetime.max.replace(tzinfo=timezone.utc),
            parse_timestamp(item.target_timestamp)
            or datetime.max.replace(tzinfo=timezone.utc),
            item.candidate_id or "",
        )
    )

    return output


def candidate_is_groupable(
    candidate: CorrelationCandidate,
    *,
    include_weak: bool = False,
) -> bool:
    if candidate.decision in (SUPPORTED_GROUP_DECISIONS):
        return True

    return include_weak and candidate.decision == CorrelationDecision.WEAK.value


def candidate_adjacency(
    candidates: Iterable[CorrelationCandidate],
    *,
    include_weak: bool = False,
) -> Dict[
    str,
    Set[str],
]:
    """
       Build an undirected record-to-record graph
       from groupable candidates.

       Candidate direction remains preserved on the
       candidate itself. Group connectivity is
       undirected because groups represent episodes,
       not causal graphs.
    hs.
    """

    adjacency: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    for candidate in candidates:
        if not candidate_is_groupable(
            candidate,
            include_weak=include_weak,
        ):
            continue

        source = candidate.source_record_id

        target = candidate.target_record_id

        adjacency[source].add(target)

        adjacency[target].add(source)

    return adjacency


def connected_record_components(
    candidates: Iterable[CorrelationCandidate],
    *,
    include_weak: bool = False,
) -> List[Set[str]]:
    """
    Find connected components in the candidate
    graph.

    Each component may become one correlation
    group.
    """

    adjacency = candidate_adjacency(
        candidates,
        include_weak=include_weak,
    )

    visited: Set[str] = set()

    components: List[Set[str]] = []

    for root in sorted(adjacency):
        if root in visited:
            continue

        component: Set[str] = set()

        queue = deque([root])

        visited.add(root)

        while queue:
            current = queue.popleft()

            component.add(current)

            for neighbor in sorted(
                adjacency.get(
                    current,
                    set(),
                )
            ):
                if neighbor in visited:
                    continue

                visited.add(neighbor)

                queue.append(neighbor)

        if len(component) >= 2:
            components.append(component)

    return components


def group_candidates_for_records(
    record_ids: Set[str],
    candidates: Iterable[CorrelationCandidate],
    *,
    include_weak: bool = False,
) -> List[CorrelationCandidate]:
    return [
        candidate
        for candidate in candidates
        if (
            candidate_is_groupable(
                candidate,
                include_weak=include_weak,
            )
            and candidate.source_record_id in record_ids
            and candidate.target_record_id in record_ids
        )
    ]


def timestamp_range(
    records: Sequence[TelemetryRecord],
) -> Tuple[str, str]:
    """
    Return earliest and latest valid timestamps.
    """

    valid = [
        (
            record_timestamp(telemetry),
            telemetry.signal.timestamp,
        )
        for telemetry in records
        if record_timestamp(telemetry) is not None
    ]

    if not valid:
        now = datetime.now(timezone.utc).isoformat()

        return (
            now,
            now,
        )

    valid.sort(key=lambda item: item[0])

    return (
        valid[0][1],
        valid[-1][1],
    )


def build_group(
    component: Set[str],
    records_by_id: Mapping[
        str,
        TelemetryRecord,
    ],
    candidates: Iterable[CorrelationCandidate],
    *,
    include_weak: bool = False,
) -> Optional[CorrelationGroup]:
    records = [
        records_by_id[record_identifier]
        for record_identifier in sorted(component)
        if record_identifier in records_by_id
    ]

    if len(records) < 2:
        return None

    group_candidates = group_candidates_for_records(
        component,
        candidates,
        include_weak=include_weak,
    )

    if not group_candidates:
        return None

    start, end = timestamp_range(records)

    candidate_scores = [candidate.scores.overall for candidate in group_candidates]

    strongest_score = max(
        candidate_scores,
        default=0.0,
    )

    average_score = (
        sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0
    )

    decisions = Counter(candidate.decision for candidate in group_candidates)

    group = CorrelationGroup(
        environment_id=(records[0].resource.environment_id),
        record_ids=[record_id(telemetry) for telemetry in records],
        resource_ids=[telemetry.resource.effective_id() for telemetry in records],
        candidate_ids=[
            candidate.candidate_id or candidate.build_candidate_id()
            for candidate in group_candidates
        ],
        providers=[telemetry.resource.provider for telemetry in records],
        signal_types=[telemetry.signal.signal_type for telemetry in records],
        start_timestamp=start,
        end_timestamp=end,
        strongest_score=(strongest_score),
        average_score=(average_score),
        status=GroupStatus.OPEN.value,
        attributes={
            "record_count": (len(records)),
            "candidate_count": (len(group_candidates)),
            "decision_counts": (dict(decisions)),
            "severity_counts": dict(
                Counter(telemetry.signal.severity for telemetry in records)
            ),
            "resource_type_counts": dict(
                Counter(telemetry.resource.resource_type for telemetry in records)
            ),
        },
    )

    return group.finalize()


def build_groups(
    records: Iterable[TelemetryRecord],
    candidates: Iterable[CorrelationCandidate],
    *,
    include_weak: bool = False,
) -> List[CorrelationGroup]:
    records_list = list(records)

    candidates_list = list(candidates)

    records_by_id = {record_id(telemetry): (telemetry) for telemetry in records_list}

    components = connected_record_components(
        candidates_list,
        include_weak=include_weak,
    )

    groups = []

    for component in components:
        group = build_group(
            component,
            records_by_id,
            candidates_list,
            include_weak=include_weak,
        )

        if group is not None:
            groups.append(group)

    groups.sort(
        key=lambda item: (
            parse_timestamp(item.start_timestamp)
            or datetime.max.replace(tzinfo=timezone.utc),
            item.group_id or "",
        )
    )

    return groups


def correlated_record_ids(
    groups: Iterable[CorrelationGroup],
) -> Set[str]:
    output: Set[str] = set()

    for group in groups:
        output.update(group.record_ids)

    return output


def uncorrelated_records(
    records: Iterable[TelemetryRecord],
    groups: Iterable[CorrelationGroup],
) -> List[str]:
    correlated = correlated_record_ids(groups)

    return sorted(
        record_id(telemetry)
        for telemetry in records
        if record_id(telemetry) not in correlated
    )


def correlation_summary(
    records: Sequence[TelemetryRecord],
    windows: Sequence[RecordWindow],
    candidates: Sequence[CorrelationCandidate],
    groups: Sequence[CorrelationGroup],
) -> Dict[str, Any]:
    decision_counts = Counter(candidate.decision for candidate in candidates)

    provider_counts = Counter(telemetry.resource.provider for telemetry in records)

    signal_counts = Counter(telemetry.signal.signal_type for telemetry in records)

    return {
        "record_count": len(records),
        "window_count": len(windows),
        "candidate_count": len(candidates),
        "group_count": len(groups),
        "candidate_decisions": dict(decision_counts),
        "providers": dict(provider_counts),
        "signal_types": dict(signal_counts),
        "correlated_record_count": len(correlated_record_ids(groups)),
    }


class CorrelationEngine:
    """
    Main OSHA in-memory correlation engine.

    Responsibilities:

      generate time windows
      create candidate pairs
      gather evidence
      score candidates
      assemble connected episodes

    It does not perform incident classification
    or root-cause analysis.
    """

    def __init__(
        self,
        *,
        topology: Optional[TopologyGraph] = None,
        scoring_config: Optional[ScoringConfig] = None,
        temporal_analyzer: Optional[TemporalAnalyzer] = None,
        window_mode: str = "sliding",
        window_seconds: float = 60.0,
        step_seconds: float = 30.0,
        cross_provider_only: bool = False,
        maximum_pairs_per_window: int = (100_000),
        include_rejected: bool = False,
        include_weak_in_groups: bool = (False),
        historical_score_provider: Optional[HistoricalScoreProvider] = None,
    ) -> None:
        if window_mode not in {
            "fixed",
            "sliding",
        }:
            raise ValueError("window_mode must be 'fixed' or 'sliding'")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        if step_seconds <= 0:
            raise ValueError("step_seconds must be > 0")

        if window_mode == "sliding" and step_seconds > window_seconds:
            raise ValueError("step_seconds must not exceed window_seconds")

        if maximum_pairs_per_window < 1:
            raise ValueError("maximum_pairs_per_window must be >= 1")

        self.topology = topology

        self.scoring_config = scoring_config or ScoringConfig()

        self.temporal = temporal_analyzer or TemporalAnalyzer(
            maximum_distance_seconds=(self.scoring_config.temporal_max_seconds),
            half_life_seconds=(self.scoring_config.temporal_half_life_seconds),
        )

        self.scorer = CorrelationScorer(
            config=self.scoring_config,
            topology=self.topology,
            temporal=self.temporal,
        )

        self.window_mode = window_mode

        self.window_seconds = float(window_seconds)

        self.step_seconds = float(step_seconds)

        self.cross_provider_only = cross_provider_only

        self.maximum_pairs_per_window = maximum_pairs_per_window

        self.include_rejected = include_rejected

        self.include_weak_in_groups = include_weak_in_groups

        self.historical_score_provider = historical_score_provider

    def make_windows(
        self,
        records: Iterable[TelemetryRecord],
    ) -> List[RecordWindow]:
        records_list = list(records)

        if self.window_mode == "fixed":
            return fixed_windows(
                records_list,
                window_seconds=(self.window_seconds),
            )

        return sliding_windows(
            records_list,
            window_seconds=(self.window_seconds),
            step_seconds=(self.step_seconds),
        )

    def candidates(
        self,
        windows: Iterable[RecordWindow],
    ) -> List[CorrelationCandidate]:
        return generate_candidates_from_windows(
            windows,
            self.scorer,
            historical_score_provider=(self.historical_score_provider),
            cross_provider_only=(self.cross_provider_only),
            maximum_pairs_per_window=(self.maximum_pairs_per_window),
            include_rejected=(self.include_rejected),
        )

    def correlate(
        self,
        records: Iterable[TelemetryRecord],
    ) -> CorrelationResult:
        records_list = list(records)

        for telemetry in records_list:
            telemetry.finalize()

        windows = self.make_windows(records_list)

        candidates = self.candidates(windows)

        groups = build_groups(
            records_list,
            candidates,
            include_weak=(self.include_weak_in_groups),
        )

        result = CorrelationResult(
            groups=groups,
            candidates=candidates,
            uncorrelated_record_ids=(
                uncorrelated_records(
                    records_list,
                    groups,
                )
            ),
            metadata={
                "window_mode": (self.window_mode),
                "window_seconds": (self.window_seconds),
                "step_seconds": (self.step_seconds),
                "cross_provider_only": (self.cross_provider_only),
                "maximum_pairs_per_window": (self.maximum_pairs_per_window),
                "include_rejected": (self.include_rejected),
                "include_weak_in_groups": (self.include_weak_in_groups),
                "summary": (
                    correlation_summary(
                        records_list,
                        windows,
                        candidates,
                        groups,
                    )
                ),
            },
        )

        return result.finalize()
