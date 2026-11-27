# OTGH

OTGH is a Go-based telemetry and infrastructure intelligence delivery gateway designed to move structured operational data reliably between infrastructure systems and external destinations.

It provides a common export pipeline with:

- structured records and batches
- routing
- bounded in-memory queues
- durable disk spooling
- retry policy with exponential backoff and jitter
- delivery acknowledgements
- worker pools
- delivery metrics
- HTTP export
- gRPC export
- S3-compatible object storage export
- signed webhook delivery
- an OTGH-native protocol layer
- graceful shutdown and restart recovery

OTGH is designed as delivery infrastructure. It separates the concerns of collecting or producing telemetry from reliably transporting that information to its configured destinations.

---

# Table of Contents

1. [Architecture](#architecture)
2. [Delivery Model](#delivery-model)
3. [Project Structure](#project-structure)
4. [Core Export Components](#core-export-components)
5. [Native Protocol](#native-protocol)
6. [HTTP Exporter](#http-exporter)
7. [gRPC Exporter](#grpc-exporter)
8. [Object Store Exporter](#object-store-exporter)
9. [Webhook Exporter](#webhook-exporter)
10. [Configuration](#configuration)
11. [Example Configuration](#example-configuration)
12. [Routing](#routing)
13. [Queue and Backpressure](#queue-and-backpressure)
14. [Durable Spool](#durable-spool)
15. [Retry Model](#retry-model)
16. [Acknowledgements](#acknowledgements)
17. [Worker Runtime](#worker-runtime)
18. [Metrics](#metrics)
19. [Startup](#startup)
20. [Graceful Shutdown](#graceful-shutdown)
21. [Crash Recovery](#crash-recovery)
22. [Security](#security)
23. [Webhook Security](#webhook-security)
24. [TLS and mTLS](#tls-and-mtls)
25. [Secrets](#secrets)
26. [Build](#build)
27. [Testing](#testing)
28. [Race Detection](#race-detection)
29. [Docker](#docker)
30. [Docker Compose](#docker-compose)
31. [Development Workflow](#development-workflow)
32. [Operations](#operations)
33. [Troubleshooting](#troubleshooting)
34. [Current Development Status](#current-development-status)
35. [Roadmap](#roadmap)

---

# Architecture

The high-level OTGH delivery path is:

```text
                     Producers
                        |
                        v
                    Records
                        |
                        v
                     Batcher
                        |
                        v
                     Router
                        |
                        v
                 Bounded Queue
                        |
                        v
                  Worker Pool
                        |
              +---------+---------+
              |                   |
              v                   v
        Durable Spool       Exporter Registry
                                  |
               +------------------+-------------------+
               |                  |                   |
               v                  v                   v
             HTTP               gRPC            Object Store
                                                       |
                                                       v
                                                  S3-compatible

                                  |
                                  v
                               Webhook
```

The exporter subsystem does not assume every destination uses the same transport.

Instead:

```text
export.Exporter
       |
       +----------------+
       |                |
       v                v
HTTP Exporter       gRPC Exporter
       |
       +----------------+
       |                |
       v                v
Object Store        Webhook
```

Each exporter translates transport-specific behavior into the common OTGH delivery model.

---

# Delivery Model

The durable delivery lifecycle is:

```text
Batch
  |
  v
Queue
  |
  v
Worker dequeues batch
  |
  v
Persist batch to spool
  |
  v
Start acknowledgement tracking
  |
  v
Attempt destination delivery
  |
  +----------------------------+
  |                            |
  v                            v
Success                       Failure
  |                            |
  v                            v
Apply acknowledgement       Classify failure
  |                            |
  v                            v
Complete?                  Retryable?
 /     \                    /       \
yes     no                yes        no
 |       |                 |          |
 v       v                 v          v
Delete  Preserve        Backoff     Failed
spool   spool             |
entry                    retry
```

A core reliability principle is:

```text
A batch is not removed from durable storage merely because it was sent.

It is removed only after OTGH determines that delivery has been successfully
acknowledged according to the destination semantics.
```

---

# Project Structure

The current architecture is organized around the following areas:

```text
.
├── cmd/
│   └── otgh/
│       ├── main.go
│       └── v1/
│           ├── runtime.go
│           ├── exporters.go
│           ├── workers.go
│           └── shutdown.go
│
├── internal/
│   ├── config/
│   │   └── config.go
│   │
│   └── export/
│       ├── acknowledgement.go
│       ├── metrics.go
│       ├── queue.go
│       ├── retry.go
│       ├── spool.go
│       ├── worker.go
│       │
│       ├── http/
│       │   ├── config.go
│       │   ├── client.go
│       │   ├── exporter.go
│       │   └── exporter_test.go
│       │
│       ├── grpc/
│       │   ├── config.go
│       │   └── exporter.go
│       │
│       ├── objectstore/
│       │   ├── config.go
│       │   └── exporter.go
│       │
│       ├── webhook/
│       │   ├── config.go
│       │   └── exporter.go
│       │
│       └── protocol/
│           ├── acknowledgement.go
│           ├── compression.go
│           ├── decoder.go
│           ├── encoder.go
│           ├── envelope.go
│           ├── errors.go
│           ├── frame.go
│           ├── handshake.go
│           ├── protocol.go
│           ├── receiver.go
│           ├── sender.go
│           └── version.go
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── go.mod
├── go.sum
└── README.md
```

The repository tree may evolve as integration work continues.

---

# Core Export Components

The core exporter package is transport-neutral.

The important components are:

```text
Record
Batch
Destination
Route
Router
Queue
Spool
RetryExecutor
AcknowledgementManager
ExporterRegistry
Worker
DeliveryMetrics
```

Transport implementations depend on those abstractions rather than redefining delivery behavior independently.

---

# Native Protocol

OTGH includes its own protocol package:

```text
internal/export/protocol/
```

The protocol layer is separate from HTTP, gRPC, object-store, and webhook transport adapters.

Its role is to provide a common OTGH message representation and session model.

## Protocol version

Protocol versions use:

```text
major.minor.patch
```

For example:

```text
1.0.0
```

The protocol package supports:

- version parsing
- validation
- comparison
- compatibility checking
- version negotiation

Version compatibility and capability compatibility are deliberately separate concepts.

---

## Envelope

The protocol envelope contains message-level information such as:

```text
Version
MessageID
MessageType
Priority
TenantID
CorrelationID
Sequence
CreatedAt
Source
Resource
Encoding
Compression
Payload
Integrity
Metadata
```

Infrastructure-specific message families include concepts such as:

```text
logs
metrics
traces
events

resource state
resource relationships
topology changes

security events
security findings
security incidents

anomalies
forecasts
situations
blast radius

decisions
remediation results
```

---

## Framing

OTGH protocol frames use a fixed binary header.

Conceptually:

```text
+------------------------------------+
| OTGH Frame Header                  |
+------------------------------------+
| Magic                              |
| Flags                              |
| Header Version                     |
| Reserved                           |
| Body Length                        |
| CRC32C                             |
+------------------------------------+
| Encoded Envelope                   |
| ...                                |
+------------------------------------+
```

The framing layer and envelope layer have different integrity responsibilities.

```text
Frame CRC32C
    |
    +--> framing/transport corruption detection

Envelope SHA-256
    |
    +--> payload integrity verification
```

Neither mechanism is intended to replace authenticated transport or sender authentication.

---

## Encoding

The initial envelope representation supports JSON.

The protocol model reserves a binary encoding identifier, but OTGH does not silently label JSON as binary.

If binary encoding is introduced later, it should have an explicitly specified representation.

---

## Compression

The compression subsystem currently supports:

```text
none
gzip
```

Compression is implemented through a registry abstraction so additional algorithms can be introduced without coupling every sender and receiver directly to a specific compressor.

Decoded payload sizes are bounded to avoid unbounded decompression.

---

## Handshake

OTGH peers negotiate their connection before ordinary protocol traffic.

A handshake advertises:

```text
protocol versions
encodings
compression algorithms
capabilities
maximum frame size
maximum payload size
maximum batch size
```

Negotiated capabilities may include:

```text
telemetry
resource state
resource relationships
topology
security
anomaly detection
forecasting
situations
blast radius
decisions
remediation results
acknowledgements
partial acknowledgements
sequencing
batching
```

The handshake determines what both peers may safely exchange.

---

## Protocol sessions

Each peer connection has a separate session.

```text
NEW
 |
 v
HANDSHAKING
 |
 v
READY
 |
 v
CLOSING
 |
 v
CLOSED
```

A session tracks:

```text
local node
remote node
session ID

negotiated version
negotiated encoding
negotiated compression
negotiated capabilities

frame limits
payload limits
batch limits

outgoing sequence
incoming sequence
```

---

## Protocol acknowledgements

Wire-level protocol acknowledgements support:

```text
ACCEPTED
PARTIAL
REJECTED
DEFERRED
```

These are separate from the process-local delivery acknowledgement manager.

The protocol acknowledgement describes what a remote OTGH peer reports.

The local acknowledgement manager tracks OTGH's internal delivery lifecycle.

---

# HTTP Exporter

Location:

```text
internal/export/http/
```

The HTTP exporter provides:

- reusable HTTP connections
- TLS
- optional mTLS
- bearer authentication
- API key authentication
- basic authentication
- request timeouts
- response-header timeouts
- TLS handshake timeout
- request-size limits
- response-size limits
- optional static headers
- context cancellation
- retry classification

The logical flow is:

```text
Batch
  |
  v
JSON
  |
  v
HTTP request
  |
  v
Destination
  |
  v
HTTP response
  |
  v
DeliveryResult
```

Successful `2xx` responses may either:

1. return no body, meaning the complete batch was accepted, or
2. return a structured OTGH-aware delivery response.

Transport-specific errors are translated into the shared retry model.

---

# gRPC Exporter

Location:

```text
internal/export/grpc/
```

The gRPC exporter currently provides:

- reusable gRPC ClientConn
- TLS
- optional mTLS
- bearer metadata
- API-key metadata
- message-size limits
- keepalive configuration
- connection recovery configuration
- standard gRPC health checking
- retry classification
- context deadlines

The current initial unary method is:

```text
/otgh.v1.ExportService/Export
```

The current adapter uses:

```text
google.protobuf.BytesValue
```

The request bytes contain JSON-serialized OTGH batch data.

The response bytes contain the OTGH export response representation.

This direct RPC adapter is an initial contract.

A dedicated generated protobuf service should replace it later without changing the common `export.Exporter` interface.

---

# Object Store Exporter

Location:

```text
internal/export/objectstore/
```

The object-store exporter currently uses an S3-compatible storage model.

One batch maps to one object.

Example object layout:

```text
<prefix>/
    YYYY/
        MM/
            DD/
                <destination>/
                    <batch-id>.json
```

Example:

```text
otgh/
2026/
11/
26/
archive-primary/
batch-1042.json
```

Configuration supports:

- S3-compatible endpoints
- regions
- buckets
- prefixes
- environment/default credentials
- static credentials
- anonymous mode where explicitly configured
- path-style addressing
- object metadata
- object tags
- upload timeouts
- maximum object sizes
- storage classes
- server-side encryption
- KMS-backed encryption

The exporter uses object writes as the destination acknowledgement boundary.

---

# Webhook Exporter

Location:

```text
internal/export/webhook/
```

Webhook delivery includes:

- HTTPS
- optional mTLS
- bearer authentication
- basic authentication
- custom non-sensitive headers
- HMAC-SHA256 signing
- delivery IDs
- timestamps
- outbound network policy
- payload limits
- response limits
- connection pooling
- retry classification

The signing contract is:

```text
delivery-id.timestamp.raw-body
```

Conceptual request headers:

```text
Webhook-Id: webhook-...
Webhook-Timestamp: 1795707000
Webhook-Key-Id: production-primary
Webhook-Signature: v1=<HMAC-SHA256>
```

The exact bytes serialized into the request body are the bytes used by the signing calculation.

---

# Configuration

OTGH loads:

```text
otgh.json
```

by default.

The path can be changed with:

```bash
export OTGH_CONFIG=/etc/otgh/nitial_delay": 1000000000,
    "max_delay": 30000000000,
    "multiplier": 2,
    "jitter": 0.2,
    "max_elapsed_time": 300000000000
  },

  "spool": {
    "directory": "/var/lib/otgh/spool",
    "directory_mode": 488,
    "file_mode": 384,
    "max_entries": 100000,
    "max_bytes": 10737418240,
    "max_entry_bytes": 67108864,
    "sync_directory": true
  },

  "protocol": {
    "node_id": "otgh-node-01"
  },

  "exporters": {
    "http": [],
    "grpc": [],
    "object_store": [],
    "webhook": []
  },

  "routes": []
}
```

## Important note about durations

Go `time.Duration` values encoded directly through standard JSON are represented numerically in nanoseconds by the configuration structures currently implemented.

For example:

```text
1 second  = 1000000000
30 seconds = 30000000000
5 minutes = 300000000000
```

A future configuration improvement should introduce human-readable duration strings such as:

```json
"shutdown_timeout": "30s"
```

rather than exposing nanosecond values directly.

---

# Environment Overrides

The current configuration loader recognizes:

```text
OTGH_CONFIG
OTGH_NODE_ID
OTGH_SPOOL_DIR
OTGH_WORKERS
```

Example:

```bash
export OTGH_CONFIG=/etc/otgh/otgh.json
export OTGH_NODE_ID=otgh-prod-01
export OTGH_SPOOL_DIR=/var/lib/otgh/spool
export OTGH_WORKERS=8
```

---

# Routing

Routing maps records to configured destination IDs.

Conceptually:

```text
Record
  |
  v
Router
  |
  +--------------------+
  |                    |
  v                    v
HTTP destination      Archive destination
```

Routes can select based on attributes represented by the routing model, including:

```text
record type
minimum priority
tenant
metadata
destination IDs
```

Exporter destination IDs must be globally unique.

A route referencing an unknown destination fails configuration validation.

---

# Queue and Backpressure

OTGH uses a bounded queue.

The queue provides:

- FIFO delivery order
- bounded capacity
- blocking enqueue/dequeue
- non-blocking operations
- queue-full signaling
- cancellation
- safe close
- draining after close
- statistics
- concurrent producer/consumer support

The queue is intentionally:

```text
in-memory
```

It provides backpressure, not durable storage.

Durability is the responsibility of the spool.

---

# Durable Spool

The spool protects batches that must survive process restart.

A batch is serialized into a versioned spool envelope containing:

```text
format version
stored timestamp
batch
checksum
```

Spool writes use:

```text
temporary file
      |
      v
write
      |
      v
sync file
      |
      v
close file
      |
      v
rename into active spool
      |
      v
optional directory sync
```

The spool provides:

- maximum entry count
- maximum total storage
- maximum entry size
- deterministic filename derivation
- checksum verification
- recovery
- acknowledgement-driven deletion
- quarantine
- context cancellation
- concurrent process-local access

The current spool assumes:

```text
one OTGH process owns one spool directory
```

It does not implement multi-process locking.

---

## Spool filenames

Batch IDs are not directly trusted as filesystem paths.

The spool derives a safe deterministic key before constructing the filename.

This avoids treating user- or producer-controlled identifiers as filesystem path components.

---

## Spool quarantine

Corrupt or otherwise unprocessable spool entries can be moved outside the active delivery set.

Conceptually:

```text
active spool
     |
     v
corrupt/unprocessable
     |
     v
quarantine/
```

Quarantine preserves evidence for diagnosis instead of silently deleting problematic data.

---

# Retry Model

Retry is centralized in:

```text
internal/export/retry.go
```

The retry engine controls:

```text
maximum attempts
initial delay
maximum delay
exponential multiplier
jitter
maximum elapsed time
context cancellation
```

Conceptually:

```text
attempt
   |
   v
failure
   |
   v
classifier
 /        \
retry     permanent
 |            |
 v            v
backoff      stop
 |
 v
next attempt
```

Transports classify failures.

The retry engine controls retry behavior.

This prevents the shared retry package from needing to understand:

```text
HTTP status codes
gRPC status codes
object-storage provider errors
webhook semantics
```

---

# Acknowledgements

Local delivery acknowledgement states include:

```text
pending
in_flight
partial
acknowledged
rejected
retrying
failed
```

For a fully acknowledged batch:

```text
accepted records == total records
rejected records == 0
```

Only after complete delivery may the worker remove the corresponding durable spool entry.

---

# Worker Runtime

The worker pipeline connects:

```text
Queue
  |
  v
Worker
  |
  v
Spool
  |
  v
AcknowledgementManager
  |
  v
RetryExecutor
  |
  v
ExporterRegistry
  |
  v
Destination
```

The runtime starts a bounded number of workers:

```text
application.worker_count
```

rather than creating an unbounded goroutine for every queued item.

Workers share:

```text
Queue
Spool
ExporterRegistry
AcknowledgementManager
DeliveryMetrics
```

Each worker has its own retry executor configured with the shared retry policy.

---

# Metrics

OTGH maintains internal delivery metrics without coupling the core package directly to a specific metrics backend.

Metrics include:

## Ingestion

```text
records received
records batched
batches created
```

## Queue

```text
batches queued
batches dequeued
queue-full events
queue depth
queue capacity
```

## Delivery

```text
export attempts
export successes
export failures
records accepted
records rejected
bytes delivered
```

## Retry

```text
retries scheduled
retries exhausted
retries suppressed
```

## Acknowledgements

```text
acknowledged batches
partial batches
failed batches
dead-lettered/rejected batches
```

## Spool

```text
spool writes
spool reads
spool deletes
spool recoveries
spool quarantines
spool errors
spool entries
spool bytes
```

## Workers

```text
workers started
workers stopped
active workers
active deliveries
```

## Latency

```text
count
total
minimum
maximum
last
mean
```

The internal snapshot can later be exposed through Prometheus, OpenTelemetry, an OTGH API, or another observability adapter.

---

# Startup

The command entry point follows:

```text
Load configuration
       |
       v
Validate configuration
       |
       v
Create Runtime
       |
       v
Configure exporters
       |
       v
Build route registry
       |
       v
Create workers
       |
       v
Recover durable spool
       |
       v
Start worker pool
```

Durable recovery occurs before normal queue processing.

If recovery fails, startup should not silently continue as if recovery succeeded.

---

# Graceful Shutdown

Shutdown follows an ordered process.

```text
SIGINT / SIGTERM
       |
       v
Cancel runtime context
       |
       v
Runtime enters STOPPING
       |
       v
Queue.Close()
       |
       v
Reject new work
       |
       v
Drain queued work
       |
       v
Stop workers
       |
       v
Close exporters
       |
       v
Close acknowledgement manager
       |
       v
Close spool
       |
       v
STOPPED
```

Shutdown has a configured timeout.

If graceful shutdown cannot complete within that timeout, the application can perform forced cleanup.

Unacknowledged batches are not intentionally deleted during shutdown.

---

# Crash Recovery

On restart:

```text
New process
    |
    v
Open spool
    |
    v
Discover pending entries
    |
    v
Validate stored data
    |
    v
Recover batches
    |
    v
Reconstruct acknowledgement tracking
    |
    v
Resume delivery
```

This is why the spool must be stored on persistent disk.

For containers:

```text
DO NOT place production spool data only inside the container writable layer.
```

Use a persistent volume or host-mounted persistent storage.

---

# Security

OTGH handles infrastructure data and should be treated as security-sensitive middleware.

Production deployment should consider:

- least privilege
- non-root execution
- TLS
- mTLS for sensitive destinations
- secret management
- restricted egress
- persistent spool protection
- filesystem permissions
- webhook signing
- private-network restrictions
- audit logging
- dependency updates

---

# Webhook Security

The webhook exporter supports HMAC-SHA256 signing.

The signature covers:

```text
delivery-id.timestamp.raw-body
```

This lets receivers authenticate the delivery ID and timestamp along with the exact request bytes.

The receiving side should independently implement:

- signature verification
- timestamp tolerance
- delivery-ID deduplication
- constant-time signature comparison
- payload validation
- idempotent processing

OTGH also performs outbound destination restrictions when configured not to allow private network targets.

The exporter validates resolved IP addresses at connection time rather than trusting only the configured hostname.

---

# TLS and mTLS

HTTP, gRPC, and webhook configurations support TLS.

Configuration can include:

```text
CA file
client certificate
client private key
server name
minimum TLS version
```

Providing a client certificate and private key enables mutual TLS where supported by the destination.

The object-store client uses its provider SDK transport behavior.

---

# Secrets

Do not commit production secrets into:

```text
README.md
otgh.json
Dockerfile
docker-compose.yml
Git history
```

Sensitive values include:

```text
bearer tokens
API keys
basic-auth passwords
object-store secret keys
session tokens
webhook HMAC secrets
client private keys
```

Use deployment-specific secret management.

Examples include:

```text
container secrets
Kubernetes Secrets
environment injection
external secret managers
mounted secret files
```

The exact mechanism should match the deployment environment.

---

# Build

Install/download modules:

```bash
go mod download
```

Tidy:

```bash
go mod tidy
```

Build:

```bash
go build ./cmd/otgh
```

Using the Makefile:

```bash
make build
```

Expected output:

```text
bin/otgh
```

---

# Run Locally

By default OTGH expects:

```text
otgh.json
```

Run:

```bash
make run
```

or:

```bash
go run ./cmd/otgh
```

Or specify a configuration file:

```bash
OTGH_CONFIG=./config/otgh.json go run ./cmd/otgh
```

---

# Testing

Run all tests:

```bash
go test ./...
```

Verbose:

```bash
go test -v ./...
```

Specific package:

```bash
go test -v ./internal/export/http
```

Specific test:

```bash
go test -v ./internal/export/http -run TestExporterConcurrent
```

---

# Race Detection

Concurrency is a core part of OTGH.

Run:

```bash
go test -race ./...
```

The queue and exporter tests should deliberately exercise concurrent operations so race-enabled tests cover actual shared execution.

For repeated concurrency testing:

```bash
go test -race -count=10 ./internal/export/http
```

or:

```bash
go test -race -count=10 ./internal/export -run TestQueue
```

---

# Coverage

Generate:

```bash
go test -coverprofile=coverage.out ./...
```

Display function coverage:

```bash
go tool cover -func=coverage.out
```

HTML:

```bash
go tool cover -html=coverage.out
```

Using the Makefile:

```bash
make test-cover
```

---

# Static Validation

Format:

```bash
gofmt -w .
```

or:

```bash
make fmt
```

Check formatting:

```bash
make fmt-check
```

Vet:

```bash
go vet ./...
```

or:

```bash
make vet
```

Complete project verification:

```bash
make verify
```

---

# Docker

Build:

```bash
docker build -t otgh:dev .
```

or:

```bash
make docker-build
```

The container build should use a multi-stage approach:

```text
Go build stage
     |
     v
compile static binary
     |
     v
minimal runtime stage
```

The runtime container should not require the Go compiler.

---

# Docker Compose

Create local runtime directories:

```bash
mkdir -p config
mkdir -p data/spool
```

Place configuration at:

```text
config/otgh.json
```

Validate Compose:

```bash
docker compose config
```

Start:

```bash
docker compose up -d --build
```

or:

```bash
make compose-up
```

Inspect:

```bash
docker compose ps
```

Logs:

```bash
docker compose logs -f otgh
```

Stop:

```bash
docker compose down
```

---

# Docker Persistence

The spool should be mounted from persistent storage.

Example:

```text
host:
./data/spool

container:
/var/lib/otgh/spool
```

A pending spool file can represent delivery work that is not yet acknowledged.

Deleting the spool can therefore mean deleting recoverable pending delivery state.

---

# Docker Security

The intended container posture includes:

```text
non-root user
read-only root filesystem
drop Linux capabilities
no-new-privileges
explicit writable spool mount
temporary /tmp filesystem
```

Operational security should be reviewed against the real deployment environment before production use.

---

# Health and Readiness

OTGH currently has exporter-level health capabilities for some transports, but the command application does not yet expose a dedicated operational HTTP server such as:

```text
/healthz
/readyz
/metrics
```

Therefore a Docker/Kubernetes HTTP health check should not be configured against a nonexistent endpoint.

A future operational server should expose separate:

```text
liveness
readiness
metrics
```

semantics.

---

# Development Workflow

A recommended development cycle is:

```bash
gofmt -w .
go mod tidy
go vet ./...
go test ./...
go test -race ./...
go build ./cmd/otgh
```

Then test container construction:

```bash
docker build -t otgh:dev .
```

Then validate Compose:

```bash
docker compose config
```

---

# Operations

Useful runtime concerns include:

## Queue pressure

Watch:

```text
queue depth
queue capacity
queue-full events
```

A persistently full queue can indicate that destinations cannot keep up with producers.

## Retry pressure

Watch:

```text
retries scheduled
retries exhausted
retry-suppressed failures
```

## Spool pressure

Watch:

```text
spool entry count
spool bytes
spool errors
quarantine activity
```

Unexpected spool growth can indicate persistent destination failures.

## Delivery performance

Watch:

```text
active deliveries
delivery failures
accepted records
rejected records
delivery latency
```

---

# Troubleshooting

## Configuration fails at startup

Run:

```bash
OTGH_CONFIG=./config/otgh.json go run ./cmd/otgh
```

Check:

```text
unknown fields
duplicate exporter IDs
unknown route destination IDs
invalid endpoint URLs
invalid TLS configuration
missing signing secrets
invalid spool limits
```

The configuration loader is intentionally strict.

---

## Spool grows continuously

Check:

1. destination availability
2. exporter retry classifications
3. authentication credentials
4. TLS trust
5. destination-side throttling
6. permanent rejection responses
7. spool error logs

Do not immediately delete spool files.

---

## HTTP delivery fails

Check:

```text
endpoint URL
TLS CA trust
client certificates
authentication
request size
response size
429/5xx responses
```

---

## gRPC delivery fails

Check:

```text
target address
TLS server name
CA trust
authentication metadata
message-size limits
gRPC health service
RPC contract compatibility
```

The current gRPC RPC contract remains an initial adapter and should eventually move to generated protobuf bindings.

---

## Object-store delivery fails

Check:

```text
bucket
region
endpoint
credentials
path-style addressing
permissions
encryption policy
KMS key
object size
```

For S3-compatible storage, also check whether the implementation matches the SDK behavior being used.

---

## Webhook delivery fails

Check:

```text
HTTPS certificate
callback URL
private-network policy
HMAC secret
key ID
destination response status
payload size
response size
authentication
```

---

# Current Development Status

OTGH is under active development.

The repository currently contains substantial implementation work across:

```text
queue
retry
spool
acknowledgements
workers
metrics

native protocol

HTTP exporter
gRPC exporter
object-store exporter
webhook exporter

configuration
runtime wiring
startup/shutdown
container scaffolding
```

However, implementation presence does not mean every component has already been integration-tested together.

Before production use, the complete repository should successfully pass:

```bash
go mod tidy
go vet ./...
go test ./...
go test -race ./...
go build ./cmd/otgh
docker build .
docker compose config
```

Any compile failures should be treated as integration work to resolve rather than bypassed.

---

# Known Integration Work

Areas that should receive focused verification include:

1. interfaces shared between:
   ```text
   worker.go
   registry
   router
   exporter implementations
   runtime wiring
   ```

2. complete queue/spool tests

3. HTTP exporter test coverage

4. gRPC server-side interoperability tests

5. object-store tests with an S3-compatible test backend or mock API

6. webhook signature tests

7. spool crash/restart recovery tests

8. protocol encoder/decoder round-trip tests

9. handshake negotiation tests

10. protocol sequencing tests

11. shutdown race tests

12. end-to-end runtime tests

13. configuration parsing tests

14. Docker/container startup tests

---

# Roadmap

Logical next engineering steps include:

## 1. Integration compilation

Run the whole repository and correct interface mismatches.

```bash
go test ./...
```

## 2. Complete unit tests

Especially:

```text
spool
retry
acknowledgements
protocol
gRPC
object store
webhook
runtime
```

## 3. Protocol tests

Add round-trip tests:

```text
Envelope
  -> Encode
  -> Frame
  -> Decode
  -> Envelope
```

Test:

```text
bad checksum
oversized frame
oversized decompression
unsupported version
unknown capability
bad sequence
replayed sequence
handshake rejection
```

## 4. Generated protobuf service

Replace the temporary direct `BytesValue` gRPC RPC contract with a dedicated versioned protobuf API.

Potential layout:

```text
proto/
└── otgh/
    └── v1/
        └── export.proto
```

## 5. Operational server

Add application endpoints for:

```text
/healthz
/readyz
/metrics
```

Only after these exist should Docker/Kubernetes HTTP probes point to them.

## 6. Observability adapter

Expose `DeliveryMetricsSnapshot` through an operational metrics endpoint or dedicated observability integration.

## 7. Secret providers

Separate secret references from normal static configuration.

## 8. Kubernetes resources

Add:

```text
Deployment
ConfigMap
Secret references
PersistentVolumeClaim
Service
ServiceAccount
NetworkPolicy
PodDisruptionBudget
resource requests/limits
health probes
```

after the application has real operational endpoints.

## 9. CI

Add automated:

```text
format check
vet
unit tests
race tests
build
container build
vulnerability/dependency checks
```

## 10. End-to-end testing

Exercise:

```text
producer
  -> OTGH route
  -> queue
  -> spool
  -> worker
  -> transport
  -> acknowledgement
  -> spool deletion
```

and failure recovery:

```text
destination unavailable
  -> retries
  -> process restart
  -> spool recovery
  -> destination restored
  -> delivery acknowledged
```

---

# Design Principles

OTGH development follows several important principles.

## Durability before delivery

Pending delivery work should be recoverable when appropriate.

## Transport independence

HTTP, gRPC, object storage, webhooks, and native protocol mechanisms should share common delivery semantics without leaking transport-specific behavior into the core engine.

## Bounded concurrency

Queues and worker counts should provide explicit pressure boundaries.

## Explicit retry policy

Retry classification and retry scheduling are separate concerns.

## Acknowledgement-driven cleanup

Successful network writes are not automatically equivalent to complete delivery acknowledgement.

## Context propagation

Blocking operations should respect cancellation and deadlines.

## Defensive limits

Network payloads, responses, frames, decompression, queues, and spool entries should be bounded.

## Secure defaults

Network trust, authentication, webhook signing, file permissions, and secret handling should be explicit.

## Recoverability

Process termination should not silently erase unacknowledged durable work.

---

# Contributing

Before submitting changes:

```bash
gofmt -w .
go mod tidy
go vet ./...
go test ./...
go test -race ./...
```

Keep new exporter implementations behind the common export interfaces where possible.

Do not duplicate retry, queue, spool, or acknowledgement logic inside individual transport packages unless the behavior is genuinely transport-specific.

---

# License

A project license has not been defined in this README.

Before public distribution, add the selected license in:

```text
LICENSE
```

and update this section accordingly.
