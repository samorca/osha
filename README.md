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

## PostgreSQL telemetry persistence

The API and runtime initialize the configured database during startup; the
telemetry worker also initializes it and persists each newly appended collector
batch to `osha_telemetry_records`. JSONL files remain available as a local
collection archive. PostgreSQL is configured with `OSHA_DATABASE_*` variables;
the password is resolved through the environment-backed secret provider and is
not stored in the repository or emitted in application status.

## Integrated telemetry frameworks

The API, runtime, and telemetry worker compose the shared telemetry pipeline:
normalization and redaction, correlation, intelligence, memory, decisions,
orchestration, observability, and guarded self-healing. The API exposes
`GET /api/v1/frameworks/status` and `POST /api/v1/frameworks/analyze`; the
worker sends newly collected records through the same pipeline.

Memory snapshots are stored under
`$OSHA_DATA_DIR/memory/<process-role>.json`. Compose mounts this directory as
persistent data. Helm stores it persistently when `persistence.enabled` is
true and uses pod-local ephemeral storage otherwise. The integration framework
does not connect provider adapters automatically, and remediation remains
dry-run-only; registering integration code does not by itself make an external
provider operational.

## Model environment configuration

OSHA's six model families (`event`, `anomaly`, `incident_root_cause`,
`decision_policy_risk`, `remediation_recommendation`, and
`capacity_forecasting`) load promoted local artifacts from
`OSHA_AI_MODELS_DIRECTORY`. Model-serving settings for device selection,
loading, inference limits, family-specific inference, and routing are listed in
[.env.example](./.env.example). These local model artifacts do not require
external OpenAI, Anthropic, or Hugging Face API keys. If private model
downloads are introduced later, their provider credentials should be added
through a secret manager rather than committed to `.env.example`.

## Capacity forecasting

Capacity forecasts are exposed at `POST /api/v1/forecasting/capacity`. Supply
timestamped percentage observations (0–100) for one resource and metric;
responses include forecast points, confidence, a threshold status, and an
estimated time to the critical threshold. The promoted
`osha-capacity-autoregressive` model forecasts CPU, memory, and disk. Network
utilization uses the explicit `linear-capacity-v1` fallback because the
available network traces are traffic rates, not utilization percentages. If
the promoted artifact is not installed, the endpoint also reports the baseline
model it used. It reports `insufficient_data` when the configured minimum
history is unavailable and does not execute remediation.

Example request:

```json
{
  "resource_id": "node-01",
  "metric": "memory",
  "history": [
    {"timestamp": "2026-11-25T18:00:00Z", "value": 62.0},
    {"timestamp": "2026-11-25T19:00:00Z", "value": 65.0},
    {"timestamp": "2026-11-25T20:00:00Z", "value": 68.0},
    {"timestamp": "2026-11-25T21:00:00Z", "value": 71.0}
  ],
  "horizon_hours": 12,
  "step_minutes": 60
}
```

Train, evaluate, and promote all six model families through one entry point.
It runs the task-specific event, anomaly, and operational trainers, evaluates
each on its held-out test split, and promotes all six under the same version
only after each trainer has produced valid test metrics:

```bash
.venv/bin/python -m src.training.train_all \
  --data-root data \
  --version 1.0.1
```

`--version` can be omitted to generate a unique version. The suite expects
prepared event, anomaly, and 5-minute capacity data beneath `data/`, plus
operational training data in the same root. Empty operational inputs fail
instead of silently training on synthetic records. All checkpoints, evaluation
reports, candidate artifacts, and promoted model files use the configured
training paths. The capacity model is promoted only when its held-out MAE beats
the persistence baseline for every trained metric; network utilization remains
on the explicit linear fallback because available network records are traffic
rates, not percentage utilization.

The Docker image contains a read-only seed of the currently promoted models.
Compose and Helm copy that seed into their model volume on startup without
overwriting versions already present. To train all six models in Compose:

```bash
docker compose --profile training run --rm trainer --version 1.0.2
docker compose restart api runtime worker
```

The Compose trainer reads the workspace `./data` directory and writes promoted
models to the shared `osha-models` volume.

For Helm, training is an opt-in Job. Put prepared training files under the
`data/` subpath of the release's shared storage, use a storage class that
supports `ReadWriteMany`, and set `persistence.enabled=true`,
`persistence.accessModes=[ReadWriteMany]`, `training.enabled=true`, and a new
`training.version`. The Job writes checkpoints, reports, and promoted models
to the shared claim. After it completes, restart the serving workloads so
their model registries discover the new version:

```bash
kubectl rollout restart deployment/<release>-osha-api \
  deployment/<release>-osha-runtime \
  deployment/<release>-osha-worker \
  --namespace <namespace>
```

Keep `training.enabled=false` for normal Helm upgrades after the Job finishes.

`OSHA_FORECASTING_*` settings in `.env.example` control enablement, minimum
history, lookback window, default horizon, and forecast step.

For local Compose use, keep `.env` untracked, copy [.env.example](./.env.example)
to `.env`, and set a strong `OSHA_DATABASE_PASSWORD`. Compose injects `.env` into
the API, runtime, and worker containers; its explicit service settings continue
to take precedence. PostgreSQL uses a named data volume and is health-checked
before OSHA services start.

For Helm, create the chart's expected Secret in the release namespace before
installing/upgrading:

```bash
kubectl create secret generic osha-postgresql-credentials \
  --from-literal=password="$OSHA_DATABASE_PASSWORD" \
  --namespace osha
```

The chart uses that Secret for both PostgreSQL initialization and application
connections. Do not commit the password or place it in `values.yaml`. The
process-local memory backend remains available for unconfigured development,
but is not durable across restarts or processes.

Inspect stored Compose telemetry with:

```bash
docker compose exec postgres psql -U osha -d osha \
  -c "SELECT source, count(*) FROM osha_telemetry_records GROUP BY source ORDER BY source;"
```

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