from __future__ import annotations

import heapq
from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from src.orchestrator.schema import (
    OrchestrationContext,
    TriggerType,
    stable_hash,
    utc_now,
)


# ---------------------------------------------------------------
# Schedule status
# ---------------------------------------------------------------


class ScheduleStatus(
    str,
    Enum,
):
    PENDING = "pending"

    READY = "ready"

    LEASED = "leased"

    COMPLETED = "completed"

    CANCELLED = "cancelled"

    EXPIRED = "expired"


# ---------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------


def parse_timestamp(
    value: str,
) -> datetime:
    text = str(value).strip()

    if not text:
        raise ValueError("Timestamp is required")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    result = datetime.fromisoformat(text)

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result.astimezone(timezone.utc)


# ---------------------------------------------------------------
# Scheduled run
# ---------------------------------------------------------------


@dataclass
class ScheduledRun:
    """
    Metadata describing an orchestration run that
    may become eligible for processing later.

    Scheduling this object does not execute the
    orchestration pipeline.
    """

    context: OrchestrationContext

    scheduled_for: str

    priority: int = 100

    status: str = ScheduleStatus.PENDING.value

    expires_at: Optional[str] = None

    lease_owner: Optional[str] = None

    leased_at: Optional[str] = None

    completed_at: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    schedule_id: Optional[str] = None

    created_at: str = field(default_factory=utc_now)

    def build_schedule_id(
        self,
    ) -> str:
        self.context.finalize()

        return "orchestrator:schedule:" + stable_hash(
            {
                "context_id": (self.context.context_id),
                "scheduled_for": (self.scheduled_for),
                "created_at": (self.created_at),
            }
        )

    def finalize(
        self,
    ) -> "ScheduledRun":
        self.context.finalize()

        scheduled = parse_timestamp(self.scheduled_for)

        self.scheduled_for = scheduled.isoformat()

        if self.expires_at is not None:
            expires = parse_timestamp(self.expires_at)

            self.expires_at = expires.isoformat()

            if expires <= scheduled:
                raise ValueError("expires_at must be later than scheduled_for")

        self.priority = int(self.priority)

        if self.status not in {item.value for item in ScheduleStatus}:
            raise ValueError(f"Unknown schedule status: {self.status}")

        if not self.schedule_id:
            self.schedule_id = self.build_schedule_id()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schedule_id": (self.schedule_id),
            "status": self.status,
            "scheduled_for": (self.scheduled_for),
            "priority": self.priority,
            "expires_at": (self.expires_at),
            "lease_owner": (self.lease_owner),
            "leased_at": (self.leased_at),
            "completed_at": (self.completed_at),
            "context": (self.context.to_dict()),
            "attributes": dict(self.attributes),
            "created_at": (self.created_at),
        }


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """
    Safe scheduler configuration.

    Non-dry-run contexts are rejected by default.
    """

    maximum_items: int = 10000

    allow_execute_contexts: bool = False

    def validate(
        self,
    ) -> None:
        if self.maximum_items < 1:
            raise ValueError("maximum_items must be >= 1")


class SchedulerError(RuntimeError):
    pass


# ---------------------------------------------------------------
# In-memory scheduler
# ---------------------------------------------------------------


class InMemoryScheduler:
    """
    Development scheduling/queue metadata store.

    Important:

    lease() returns work metadata.

    It does NOT call the orchestrator pipeline.
    """

    def __init__(
        self,
        *,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        self.config = config or SchedulerConfig()

        self.config.validate()

        self._items: Dict[
            str,
            ScheduledRun,
        ] = {}

        self._heap: List[
            Tuple[
                float,
                int,
                str,
            ]
        ] = []

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------

    def submit(
        self,
        item: ScheduledRun,
    ) -> ScheduledRun:
        item.finalize()

        if len(self._items) >= self.config.maximum_items:
            raise SchedulerError("Scheduler capacity has been reached")

        if item.schedule_id in self._items:
            raise SchedulerError(f"Duplicate schedule_id: {item.schedule_id}")

        # Independent scheduler boundary.
        #
        # A scheduled context cannot request the
        # live execution path unless explicitly
        # enabled in scheduler configuration.
        if not item.context.dry_run and not self.config.allow_execute_contexts:
            raise SchedulerError("Scheduler refuses non-dry-run orchestration contexts")

        timestamp = parse_timestamp(item.scheduled_for).timestamp()

        self._items[item.schedule_id] = item

        heapq.heappush(
            self._heap,
            (
                timestamp,
                item.priority,
                item.schedule_id,
            ),
        )

        return item

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(
        self,
        schedule_id: str,
    ) -> Optional[ScheduledRun]:
        return self._items.get(schedule_id)

    def list(
        self,
    ) -> List[ScheduledRun]:
        return sorted(
            self._items.values(),
            key=lambda item: (
                parse_timestamp(item.scheduled_for),
                item.priority,
                item.schedule_id or "",
            ),
        )

    # ---------------------------------------------------------
    # Ready work
    # ---------------------------------------------------------

    def ready(
        self,
        *,
        now: Optional[datetime] = None,
    ) -> List[ScheduledRun]:
        """
        Find scheduled work whose time has
        arrived.

        This method does not execute that work.
        """

        current = now or datetime.now(timezone.utc)

        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        current = current.astimezone(timezone.utc)

        output: List[ScheduledRun] = []

        for item in self.list():
            if item.status not in {
                ScheduleStatus.PENDING.value,
                ScheduleStatus.READY.value,
            }:
                continue

            if item.expires_at and current >= parse_timestamp(item.expires_at):
                item.status = ScheduleStatus.EXPIRED.value

                continue

            if current >= parse_timestamp(item.scheduled_for):
                item.status = ScheduleStatus.READY.value

                output.append(item)

        return output

    # ---------------------------------------------------------
    # Lease
    # ---------------------------------------------------------

    def lease(
        self,
        *,
        owner: str,
        now: Optional[datetime] = None,
    ) -> Optional[ScheduledRun]:
        """
        Lease the highest-priority ready item.

        Leasing transfers processing ownership.
        It does not execute the orchestration
        run.
        """

        owner = owner.strip()

        if not owner:
            raise SchedulerError("Lease owner is required")

        available = self.ready(now=now)

        if not available:
            return None

        # ready() is ordered by time then
        # priority. Lower numeric priority wins
        # within the same scheduled time.
        item = available[0]

        item.status = ScheduleStatus.LEASED.value

        item.lease_owner = owner

        lease_time = now or datetime.now(timezone.utc)

        if lease_time.tzinfo is None:
            lease_time = lease_time.replace(tzinfo=timezone.utc)

        item.leased_at = lease_time.astimezone(timezone.utc).isoformat()

        return item

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    def complete(
        self,
        schedule_id: str,
    ) -> ScheduledRun:
        item = self.get(schedule_id)

        if item is None:
            raise SchedulerError(f"Unknown schedule_id: {schedule_id}")

        if item.status != ScheduleStatus.LEASED.value:
            raise SchedulerError(f"Schedule {schedule_id} is not leased")

        item.status = ScheduleStatus.COMPLETED.value

        item.completed_at = utc_now()

        return item

    # ---------------------------------------------------------
    # Cancellation
    # ---------------------------------------------------------

    def cancel(
        self,
        schedule_id: str,
        *,
        reason: Optional[str] = None,
    ) -> ScheduledRun:
        item = self.get(schedule_id)

        if item is None:
            raise SchedulerError(f"Unknown schedule_id: {schedule_id}")

        if item.status in {
            ScheduleStatus.COMPLETED.value,
            ScheduleStatus.EXPIRED.value,
        }:
            raise SchedulerError(f"Schedule {schedule_id} is already terminal")

        item.status = ScheduleStatus.CANCELLED.value

        if reason:
            item.attributes["cancellation_reason"] = reason

        return item

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        statuses: Dict[
            str,
            int,
        ] = {}

        for item in self._items.values():
            statuses[item.status] = (
                statuses.get(
                    item.status,
                    0,
                )
                + 1
            )

        return {
            "schedule_count": len(self._items),
            "statuses": statuses,
            "allow_execute_contexts": (self.config.allow_execute_contexts),
        }


# ---------------------------------------------------------------
# Scheduled context builder
# ---------------------------------------------------------------


def scheduled_context(
    *,
    environment_id: str,
    scheduled_for: str,
    incident_id: Optional[str] = None,
    resource_ids: Iterable[str] = (),
    service_ids: Iterable[str] = (),
    requested_by: Optional[str] = None,
    dry_run: bool = True,
    priority: int = 100,
    expires_at: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> ScheduledRun:
    """
    Construct scheduling metadata for a future
    orchestration run.

    The default context is dry-run.
    """

    context = OrchestrationContext(
        environment_id=(environment_id),
        trigger_type=(TriggerType.SCHEDULE.value),
        incident_id=(incident_id),
        resource_ids=list(resource_ids),
        service_ids=list(service_ids),
        requested_by=(requested_by),
        dry_run=dry_run,
        attributes=dict(attributes or {}),
    ).finalize()

    return ScheduledRun(
        context=context,
        scheduled_for=(scheduled_for),
        priority=priority,
        expires_at=expires_at,
    ).finalize()
