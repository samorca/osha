import unittest
from datetime import datetime, timedelta, timezone

from src.memory import (
    InMemoryStore,
    TelemetryIngestionPipeline,
)
from src.memory.governance.access import AccessController
from src.memory.governance.audit import AuditLogger
from src.memory.governance.provenance import ProvenanceTracker
from src.memory.ingestion.correlators.events import correlate_events
from src.memory.ingestion.receivers.metrics import MetricsReceiver
from src.memory.ingestion.validators.telemetry import (
    TelemetryValidationError,
)
from src.memory.retention.compaction import compact_records
from src.memory.retention.deletion import delete_where
from src.memory.retention.lifecycle import RetentionPolicy, apply_retention
from src.memory.retrieval.incident_search import incident_search
from src.memory.retrieval.recent_context import recent_context
from src.memory.retrieval.similar_experiences import similar_experiences
from src.memory.retrieval.topology_context import topology_context
from src.memory.schemas import (
    Event,
    EventSeverity,
    Experience,
    Incident,
    IncidentSeverity,
    IncidentStatus,
    ResourceNode,
    ResourceRelation,
)


class MemoryTests(unittest.TestCase):
    def test_ingestion_pipeline_redacts_and_normalizes(self):
        pipeline = TelemetryIngestionPipeline()
        record = pipeline.process(
            {
                "metric": "cpu_usage",
                "value": 42.5,
                "source_id": "node-1",
                "source_type": "node",
                "labels": {"region": "eu"},
                "metadata": {"api_token": "secret-value"},
                "timestamp": "2026-11-19T10:00:00Z",
            }
        )
        self.assertEqual(record.metric, "cpu_usage")
        self.assertEqual(record.value, 42.5)
        self.assertEqual(record.metadata["api_token"], "[REDACTED]")
        self.assertEqual(record.timestamp.tzinfo, timezone.utc)

    def test_ingestion_validation_and_receiver(self):
        receiver = MetricsReceiver()
        with self.assertRaises(TypeError):
            receiver.submit("bad")
        receiver.submit({"metric": "memory"})
        batch = receiver.receive()
        self.assertEqual(len(batch), 1)
        self.assertEqual(receiver.receive(), [])
        with self.assertRaises(TelemetryValidationError):
            TelemetryIngestionPipeline().process({"metric": "cpu"})

    def test_in_memory_store_crud(self):
        store = InMemoryStore()
        node = ResourceNode("node-1", "server", "Node 1")
        self.assertIs(store.put(node), node)
        self.assertIs(store.get("node-1"), node)
        self.assertEqual(len(store.list()), 1)
        self.assertTrue(store.delete("node-1"))
        self.assertFalse(store.delete("node-1"))
        self.assertIsNone(store.get("node-1"))

    def test_event_correlation_and_recent_context(self):
        now = datetime.now(timezone.utc)
        event_a = Event(
            "warning",
            "node-a",
            "node",
            "A",
            timestamp=now - timedelta(minutes=5),
            correlation_id="incident-1",
        )
        event_b = Event(
            "error", "node-b", "node", "B", timestamp=now, correlation_id="incident-1"
        )
        groups = correlate_events([event_a, event_b])
        self.assertEqual(groups["incident-1"], [event_a, event_b])
        self.assertEqual(recent_context([event_a, event_b], 1)[0], event_b)

    def test_incident_search_and_experience_retrieval(self):
        incident = Incident(
            "Database failure",
            "Database pool exhausted",
            IncidentSeverity.HIGH,
            status=IncidentStatus.OPEN,
        )
        experience = Experience(
            incident_id=incident.incident_id,
            action_id="action-1",
            outcome="recovered",
            incident_signature="database_pool_exhausted",
            successful=True,
            reward=0.8,
        )
        self.assertEqual(
            incident_search([incident], query="pool", severity="high", status="open"),
            [incident],
        )
        self.assertEqual(
            similar_experiences(
                [experience],
                incident_signature="database",
                successful=True,
            ),
            [experience],
        )

    def test_topology_context_is_bounded_by_depth(self):
        nodes = [
            ResourceNode("a", "node", "A"),
            ResourceNode("b", "node", "B"),
            ResourceNode("c", "node", "C"),
        ]
        relations = [
            ResourceRelation("a", "b", "depends_on"),
            ResourceRelation("b", "c", "depends_on"),
        ]
        context = topology_context(nodes, relations, "a", max_depth=1)
        self.assertEqual(
            {node.resource_id for node in context["nodes"]},
            {"a", "b"},
        )
        self.assertEqual(len(context["relations"]), 1)

    def test_retention_compaction_and_deletion(self):
        now = datetime.now(timezone.utc)
        old = Event(
            "warning", "old", "node", "old", timestamp=now - timedelta(days=100)
        )
        new = Event("info", "new", "node", "new", timestamp=now)
        kept = apply_retention(
            [old, new],
            RetentionPolicy(max_age_days=30),
            now,
        )
        self.assertEqual(kept, [new])
        duplicate = Event("info", "new", "node", "duplicate", timestamp=now)
        compacted = compact_records([new, duplicate], ("event_type",))
        self.assertEqual(len(compacted), 1)
        kept, deleted = delete_where([new, old], lambda item: item == old)
        self.assertEqual(deleted, [old])
        self.assertEqual(kept, [new])

    def test_governance_audit_and_provenance(self):
        access = AccessController()
        access.grant("admin", "incident-1", {"read", "write"})
        self.assertTrue(access.is_allowed("admin", "incident-1", "read"))
        self.assertFalse(access.is_allowed("user", "incident-1", "read"))

        audit = AuditLogger()
        entry = audit.record("admin", "read", "incident-1")
        self.assertEqual(audit.list()[0], entry)

        provenance = ProvenanceTracker()
        record = provenance.record("incident-1", "detector", "created")
        self.assertEqual(provenance.history("incident-1"), [record])


if __name__ == "__main__":
    unittest.main()
