# OSHA

**AI-Driven Infrastructure Intelligence, Anomaly Detection, and Self-Healing Orchestration**

OSHA is an infrastructure intelligence and self-healing AI project designed to observe, understand, diagnose, and eventually assist in the remediation of problems across modern cloud and distributed computing environments.

The long-term goal of OSHA is to build an intelligent orchestration system capable of learning infrastructure behaviour from logs, metrics, events, topology, configuration, workload traces, security signals, and historical incidents.

OSHA is being developed as an independent platform rather than as a wrapper around an external AI agent. Its models, telemetry pipelines, correlation engine, policy controls, and orchestration components are intended to remain part of the OSHA architecture.

> **Project Status:** Active research and development.
>
> Many capabilities described in this README represent the target architecture and development direction. They should not be interpreted as production-ready features unless explicitly marked as implemented.

---

## 1. Vision

Modern infrastructure produces enormous amounts of operational information:

- system logs
- application logs
- CPU and memory metrics
- storage metrics
- network telemetry
- container and Kubernetes events
- cloud events
- GPU telemetry
- workload traces
- configuration changes
- service dependencies
- security findings
- incidents
- recovery actions

Traditional monitoring systems are excellent at collecting and visualizing this information, but infrastructure engineers are still responsible for correlating many independent signals and deciding what they mean.

OSHA aims to introduce an intelligence layer above those signals.

The long-term objective is to build a secure, auditable
operational-intelligence platform.

## Telemetry collector worker

The Compose telemetry worker runs the Linux collector every
`OSHA_COLLECTOR_INTERVAL_SECONDS` (30 seconds by default). It appends JSONL files
under `OSHA_DATA_DIR/live/<source>/<worker-id>.jsonl`. Kubernetes collection is
included when in-cluster credentials or a visible `KUBECONFIG` are available;
set `OSHA_COLLECTOR_SOURCES=linux,kubernetes` to require both sources. The
Kubernetes collector defaults to namespace `osha`, includes pod logs, and reads
up to 200 lines per container per cycle.

Run one collection cycle locally with:

```bash
OSHA_COLLECTOR_SOURCES=linux,kubernetes \
OSHA_KUBERNETES_NAMESPACE=osha \
python -m src.workers.main --once
```

In Docker Compose, view worker output with `docker compose logs -f worker`.
The container's Linux collector sees the container environment, not the host.
Collecting Linux telemetry from the host or every Kubernetes node requires
running the worker in those host/node contexts with the appropriate
read-only access. Kubernetes API collection requires permission to list pods
and events and read pod logs in the selected namespace. Collected pod and
journal logs may contain sensitive text; do not treat the JSONL output as
sanitized training data.

## Security boundary integration

The security package exposes central integration points rather than adding
cryptography to each collector. The shared adapters live in
`src/security/integration.py`:

- Training sources can require a signed sidecar for every consumed dataset.
  The sidecar name is `<data-file>.osha-signed.json`; verification occurs
  before records are read.
- `ModelArtifactManager` can sign promoted model manifests. The AI registry
  and standalone runtime can verify those manifests before loading model
  bytes. A signature establishes integrity and trusted provenance, not model
  quality or qualification.
- `RemediationAuthorizationGate` evaluates authenticated workload identity,
  permissions, approval, risk, evidence, and rollback policy immediately
  before execute-mode adapter dispatch. Execute mode stays disabled by
  default and requires both this gate and signed execution auditing.
- `SecureTelemetryTransport` seals normalized records. Correlation loading
  can require and open those envelopes before records enter the pipeline.
  Receivers bind the expected collector identity to its trusted signing key.
- These boundaries use `SecurityAuditRecorder` for signed events and an
  integrity chain.

These controls are enabled by injecting trusted verifier/signer services into
the corresponding managers. The project does not generate production signing,
encryption, or workload-identity credentials. Provision trusted keys,
producer/pipeline allow-lists, workload authentication, and approval/resource
resolvers from the deployment's approved identity and key-management systems.
Unsigned legacy inputs remain supported unless the relevant `require_signed_*`
or `require_secure` option is enabled.

The current `IntegrityChain` is process-local and intended for development and
testing. Multi-process or multi-replica deployments need a durable backend
with atomic append semantics and external chain-head anchoring before signed
audit can be treated as durable production evidence.