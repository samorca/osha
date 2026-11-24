from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.memory.schema import (
    MemoryObject,
    MemoryStatus,
    MemoryType,
    VerificationStatus,
    utc_now,
)

from src.memory.store import (
    MemoryQuery,
    MemoryStore,
)


# ---------------------------------------------------------------
# OSHA DEVELOPMENT DEFAULTS
# ---------------------------------------------------------------
#
# These values are operational defaults only.
#
# They are NOT legal, regulatory, privacy, audit
# or compliance requirements.
#
# Production deployments should override them
# through explicit organizational policy.
# ---------------------------------------------------------------

DEFAULT_RETENTION_DAYS = {
    MemoryType.EPISODE.value: 30,
    MemoryType.PATTERN.value: 180,
    MemoryType.BASELINE.value: 90,
    MemoryType.INCIDENT.value: 365,
    MemoryType.ROOT_CAUSE.value: 365,
    MemoryType.OUTCOME.value: 365,
}


DEFAULT_ARCHIVE_DAYS = {
    MemoryType.EPISODE.value: 90,
    MemoryType.PATTERN.value: 365,
    MemoryType.BASELINE.value: 180,
    MemoryType.INCIDENT.value: 730,
    MemoryType.ROOT_CAUSE.value: 730,
    MemoryType.OUTCOME.value: 730,
}


@dataclass(frozen=True)
class RetentionRule:
    """
    Lifecycle rule for one memory category.

    active_days:
        How long memory remains active.

    archive_days:
        How long archived memory can remain
        before becoming expired.

    archive_verified:
        Preserve verified memory by moving it
        into ARCHIVED rather than immediately
        expiring it.

    delete_expired:
        Whether expired memory may subsequently
        be physically deleted by the retention
        processor.

    Development defaults must be replaced when a
    deployment has explicit records-management,
    audit, security or privacy requirements.
    """

    memory_type: str

    active_days: int

    archive_days: int

    archive_verified: bool = True

    archive_unverified: bool = False

    delete_expired: bool = False

    def validate(
        self,
    ) -> None:
        if self.active_days < 1:
            raise ValueError("active_days must be >= 1")

        if self.archive_days < 0:
            raise ValueError("archive_days must be >= 0")


@dataclass
class RetentionPolicy:
    """
    Complete OSHA memory retention policy.
    """

    rules: Dict[
        str,
        RetentionRule,
    ] = field(default_factory=dict)

    keep_verified_incidents: bool = True

    keep_verified_root_causes: bool = True

    keep_verified_outcomes: bool = True

    purge_expired: bool = False

    def __post_init__(
        self,
    ) -> None:
        if not self.rules:
            self.rules.update(default_rules())

        self.validate()

    def validate(
        self,
    ) -> None:
        for rule in self.rules.values():
            rule.validate()

    def rule_for(
        self,
        memory_type: str,
    ) -> RetentionRule:
        rule = self.rules.get(memory_type)

        if rule is None:
            raise KeyError(f"No retention rule for memory type: {memory_type}")

        return rule


@dataclass(frozen=True)
class RetentionDecision:
    """
    Proposed lifecycle transition.

    evaluate_retention() does not mutate the
    memory store.
    """

    memory_id: str

    memory_type: str

    current_status: str

    action: str

    reason: str

    age_days: float

    verified: bool

    created_at: str

    reference_timestamp: str

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_id": (self.memory_id),
            "memory_type": (self.memory_type),
            "current_status": (self.current_status),
            "action": self.action,
            "reason": self.reason,
            "age_days": round(
                self.age_days,
                6,
            ),
            "verified": (self.verified),
            "created_at": (self.created_at),
            "reference_timestamp": (self.reference_timestamp),
        }


@dataclass
class RetentionRunResult:
    examined: int = 0

    kept: int = 0

    archived: int = 0

    expired: int = 0

    deleted: int = 0

    errors: int = 0

    decisions: List[RetentionDecision] = field(default_factory=list)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "examined": (self.examined),
            "kept": self.kept,
            "archived": (self.archived),
            "expired": (self.expired),
            "deleted": (self.deleted),
            "errors": self.errors,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def default_rules() -> Dict[
    str,
    RetentionRule,
]:
    """
    Build development-only default rules.

    These are application defaults, not external
    compliance requirements.
    """

    return {
        memory_type: RetentionRule(
            memory_type=memory_type,
            active_days=(DEFAULT_RETENTION_DAYS[memory_type]),
            archive_days=(DEFAULT_ARCHIVE_DAYS[memory_type]),
            archive_verified=True,
            archive_unverified=(
                memory_type
                in {
                    MemoryType.INCIDENT.value,
                    MemoryType.ROOT_CAUSE.value,
                    MemoryType.OUTCOME.value,
                }
            ),
            delete_expired=False,
        )
        for memory_type in DEFAULT_RETENTION_DAYS
    }


# ---------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------


def parse_timestamp(
    value: Optional[str],
) -> Optional[datetime]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        result = datetime.fromisoformat(text)

    except ValueError:
        return None

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result.astimezone(timezone.utc)


def normalized_now(
    value: Optional[datetime] = None,
) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def age_days(
    timestamp: Optional[str],
    *,
    now: Optional[datetime] = None,
) -> float:
    reference = parse_timestamp(timestamp)

    if reference is None:
        return 0.0

    current = normalized_now(now)

    difference = current - reference

    return max(
        0.0,
        difference.total_seconds() / 86400.0,
    )


def lifecycle_timestamp(
    memory: MemoryObject,
) -> str:
    """
    Select timestamp against which retention is
    evaluated.

    Prefer updated_at so recently refreshed
    operational knowledge remains active.

    Fall back to created_at when necessary.
    """

    if parse_timestamp(memory.updated_at) is not None:
        return memory.updated_at

    return memory.created_at


def is_verified(
    memory: MemoryObject,
) -> bool:
    return memory.verification_status == VerificationStatus.VERIFIED.value


# ---------------------------------------------------------------
# Retention protections
# ---------------------------------------------------------------


def protected_verified_memory(
    memory: MemoryObject,
    policy: RetentionPolicy,
) -> bool:
    """
    Return whether verified memory is configured
    for long-term preservation.

    Protected records may still move from active
    to archived, but this policy will not delete
    them automatically.
    """

    if not is_verified(memory):
        return False

    if memory.memory_type == MemoryType.INCIDENT.value:
        return policy.keep_verified_incidents

    if memory.memory_type == MemoryType.ROOT_CAUSE.value:
        return policy.keep_verified_root_causes

    if memory.memory_type == MemoryType.OUTCOME.value:
        return policy.keep_verified_outcomes

    return False


# ---------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------


def evaluate_retention(
    memory: MemoryObject,
    *,
    policy: Optional[RetentionPolicy] = None,
    now: Optional[datetime] = None,
) -> RetentionDecision:
    """
    Evaluate one memory object without modifying
    it.

    Possible actions:

        keep
        archive
        expire
        delete
    """

    active_policy = policy or RetentionPolicy()

    memory.finalize()

    if not memory.memory_id:
        raise ValueError("Memory has no memory_id")

    rule = active_policy.rule_for(memory.memory_type)

    reference_timestamp = lifecycle_timestamp(memory)

    current_age = age_days(
        reference_timestamp,
        now=now,
    )

    verified = is_verified(memory)

    protected = protected_verified_memory(
        memory,
        active_policy,
    )

    # Previously expired records.
    if memory.status == MemoryStatus.EXPIRED.value:
        may_delete = (
            active_policy.purge_expired and rule.delete_expired and not protected
        )

        if may_delete:
            return RetentionDecision(
                memory_id=(memory.memory_id),
                memory_type=(memory.memory_type),
                current_status=(memory.status),
                action="delete",
                reason=("Memory is expired and policy permits purge."),
                age_days=(current_age),
                verified=verified,
                created_at=(memory.created_at),
                reference_timestamp=(reference_timestamp),
            )

        return RetentionDecision(
            memory_id=(memory.memory_id),
            memory_type=(memory.memory_type),
            current_status=(memory.status),
            action="keep",
            reason=("Memory is expired but automatic deletion is not permitted."),
            age_days=current_age,
            verified=verified,
            created_at=(memory.created_at),
            reference_timestamp=(reference_timestamp),
        )

    # Archived records.
    if memory.status == MemoryStatus.ARCHIVED.value:
        if protected:
            return RetentionDecision(
                memory_id=(memory.memory_id),
                memory_type=(memory.memory_type),
                current_status=(memory.status),
                action="keep",
                reason=("Verified historical memory is protected by policy."),
                age_days=(current_age),
                verified=True,
                created_at=(memory.created_at),
                reference_timestamp=(reference_timestamp),
            )

        archive_limit = rule.active_days + rule.archive_days

        if current_age >= archive_limit:
            return RetentionDecision(
                memory_id=(memory.memory_id),
                memory_type=(memory.memory_type),
                current_status=(memory.status),
                action="expire",
                reason=("Archived memory exceeded its configured lifecycle."),
                age_days=(current_age),
                verified=verified,
                created_at=(memory.created_at),
                reference_timestamp=(reference_timestamp),
            )

        return RetentionDecision(
            memory_id=(memory.memory_id),
            memory_type=(memory.memory_type),
            current_status=(memory.status),
            action="keep",
            reason=("Archived memory remains inside its configured lifecycle."),
            age_days=current_age,
            verified=verified,
            created_at=(memory.created_at),
            reference_timestamp=(reference_timestamp),
        )

    # Active memory inside the retention window.
    if current_age < rule.active_days:
        return RetentionDecision(
            memory_id=(memory.memory_id),
            memory_type=(memory.memory_type),
            current_status=(memory.status),
            action="keep",
            reason=("Memory remains inside its active retention period."),
            age_days=current_age,
            verified=verified,
            created_at=(memory.created_at),
            reference_timestamp=(reference_timestamp),
        )

    # Active memory reached its lifecycle
    # transition threshold.
    should_archive = (
        (verified and rule.archive_verified)
        or (not verified and rule.archive_unverified)
        or protected
    )

    if should_archive:
        return RetentionDecision(
            memory_id=(memory.memory_id),
            memory_type=(memory.memory_type),
            current_status=(memory.status),
            action="archive",
            reason=("Memory exceeded its active period and policy requires archival."),
            age_days=current_age,
            verified=verified,
            created_at=(memory.created_at),
            reference_timestamp=(reference_timestamp),
        )

    return RetentionDecision(
        memory_id=(memory.memory_id),
        memory_type=(memory.memory_type),
        current_status=(memory.status),
        action="expire",
        reason=("Memory exceeded its active period and has no archival requirement."),
        age_days=current_age,
        verified=verified,
        created_at=(memory.created_at),
        reference_timestamp=(reference_timestamp),
    )


# ---------------------------------------------------------------
# Applying decisions
# ---------------------------------------------------------------


def apply_retention_decision(
    store: MemoryStore,
    decision: RetentionDecision,
) -> bool:
    """
    Apply one previously evaluated lifecycle
    action.

    Returns True when the store was changed.
    """

    if decision.action == "keep":
        return False

    if decision.action == "archive":
        store.archive(decision.memory_id)

        return True

    if decision.action == "expire":
        store.expire(decision.memory_id)

        return True

    if decision.action == "delete":
        return store.delete(decision.memory_id)

    raise ValueError(f"Unknown retention action: {decision.action}")


def run_retention(
    store: MemoryStore,
    *,
    policy: Optional[RetentionPolicy] = None,
    now: Optional[datetime] = None,
    dry_run: bool = True,
    include_decisions: bool = True,
) -> RetentionRunResult:
    """
    Evaluate and optionally apply retention over
    the complete store.

    dry_run defaults to True deliberately.

    This prevents a newly configured policy from
    immediately changing or deleting historical
    memory before its decisions can be reviewed.
    """

    active_policy = policy or RetentionPolicy()

    memories = store.query(MemoryQuery(newest_first=False))

    result = RetentionRunResult()

    for memory in memories:
        result.examined += 1

        try:
            decision = evaluate_retention(
                memory,
                policy=(active_policy),
                now=now,
            )

            if include_decisions:
                result.decisions.append(decision)

            if decision.action == "keep":
                result.kept += 1

            elif decision.action == "archive":
                result.archived += 1

            elif decision.action == "expire":
                result.expired += 1

            elif decision.action == "delete":
                result.deleted += 1

            if not dry_run:
                apply_retention_decision(
                    store,
                    decision,
                )

        except Exception:
            result.errors += 1

    return result


# ---------------------------------------------------------------
# Custom policy construction
# ---------------------------------------------------------------


def policy_from_mapping(
    value: Mapping[
        str,
        Mapping[str, Any],
    ],
    *,
    purge_expired: bool = False,
) -> RetentionPolicy:
    """
    Construct a retention policy from a mapping.

    Example structure:

        {
            "episode": {
                "active_days": 30,
                "archive_days": 60,
                "archive_verified": True,
            }
        }

    This makes it possible for a future config
    file to define retention without changing
    code.
    """

    defaults = default_rules()

    rules: Dict[
        str,
        RetentionRule,
    ] = {}

    for (
        memory_type,
        default_rule,
    ) in defaults.items():
        override = value.get(
            memory_type,
            {},
        )

        rules[memory_type] = RetentionRule(
            memory_type=(memory_type),
            active_days=int(
                override.get(
                    "active_days",
                    default_rule.active_days,
                )
            ),
            archive_days=int(
                override.get(
                    "archive_days",
                    default_rule.archive_days,
                )
            ),
            archive_verified=bool(
                override.get(
                    "archive_verified",
                    default_rule.archive_verified,
                )
            ),
            archive_unverified=bool(
                override.get(
                    "archive_unverified",
                    default_rule.archive_unverified,
                )
            ),
            delete_expired=bool(
                override.get(
                    "delete_expired",
                    default_rule.delete_expired,
                )
            ),
        )

    return RetentionPolicy(
        rules=rules,
        purge_expired=(purge_expired),
    )


# ---------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------


def retention_summary(
    decisions: Iterable[RetentionDecision],
) -> Dict[str, Any]:
    decision_list = list(decisions)

    action_counts: Dict[
        str,
        int,
    ] = {}

    type_counts: Dict[
        str,
        Dict[str, int],
    ] = {}

    for decision in decision_list:
        action_counts[decision.action] = (
            action_counts.get(
                decision.action,
                0,
            )
            + 1
        )

        type_counts.setdefault(
            decision.memory_type,
            {},
        )

        per_type = type_counts[decision.memory_type]

        per_type[decision.action] = (
            per_type.get(
                decision.action,
                0,
            )
            + 1
        )

    return {
        "decision_count": (len(decision_list)),
        "actions": (action_counts),
        "memory_types": (type_counts),
    }


class RetentionManager:
    """
    Convenience wrapper used by memory.pipeline
    and future scheduled maintenance workers.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: Optional[RetentionPolicy] = None,
    ) -> None:
        self.store = store

        self.policy = policy or RetentionPolicy()

        self.policy.validate()

    def evaluate(
        self,
        memory: MemoryObject,
        *,
        now: Optional[datetime] = None,
    ) -> RetentionDecision:
        return evaluate_retention(
            memory,
            policy=self.policy,
            now=now,
        )

    def run(
        self,
        *,
        now: Optional[datetime] = None,
        dry_run: bool = True,
    ) -> RetentionRunResult:
        return run_retention(
            self.store,
            policy=self.policy,
            now=now,
            dry_run=dry_run,
        )
