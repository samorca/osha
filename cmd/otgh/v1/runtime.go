package v1

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/samorca/otgh/internal/config"
	"github.com/samorca/otgh/internal/export"
	"github.com/samorca/otgh/internal/export/protocol"
)

// RuntimeState represents the lifecycle of the OTGH application runtime.
type RuntimeState uint8

const (
	RuntimeStateNew RuntimeState = iota
	RuntimeStateStarting
	RuntimeStateRunning
	RuntimeStateStopping
	RuntimeStateStopped
	RuntimeStateFailed
)

// String returns a stable diagnostic representation.
func (s RuntimeState) String() string {
	switch s {
	case RuntimeStateNew:
		return "new"

	case RuntimeStateStarting:
		return "starting"

	case RuntimeStateRunning:
		return "running"

	case RuntimeStateStopping:
		return "stopping"

	case RuntimeStateStopped:
		return "stopped"

	case RuntimeStateFailed:
		return "failed"

	default:
		return "unknown"
	}
}

// Valid reports whether the runtime state is recognized.
func (s RuntimeState) Valid() bool {
	return s <= RuntimeStateFailed
}

// Runtime owns the shared OTGH application components.
//
// Exporter construction lives in exporters.go.
// Worker construction and execution live in workers.go.
// Ordered shutdown lives in shutdown.go.
//
// Runtime itself is responsible for shared component ownership and lifecycle
// state.
type Runtime struct {
	config config.Config

	protocol *protocol.Protocol

	router *export.Router

	exporters *export.ExporterRegistry

	queue *export.Queue

	spool *export.Spool

	batchIDs export.BatchIDGenerator

	acknowledgements *export.AcknowledgementManager

	metrics *export.DeliveryMetrics

	retryConfig export.RetryConfig

	workersMu sync.RWMutex
	workers   []*export.Worker

	stateMu sync.RWMutex
	state   RuntimeState

	startedAt time.Time
	stoppedAt time.Time

	failure error

	runCancelMu sync.Mutex
	runCancel   context.CancelFunc

	closed atomic.Bool
}

// NewRuntime validates configuration and constructs OTGH's shared runtime
// components.
//
// It deliberately does not:
//
//   - construct destination exporters;
//   - start worker goroutines;
//   - register operating-system signal handlers.
//
// Those responsibilities belong to the surrounding application-wiring files.
func NewRuntime(
	cfg config.Config,
) (*Runtime, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create OTGH runtime: %w",
			err,
		)
	}

	queue, err := export.NewQueue(
		cfg.Queue.ExportConfig(),
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create OTGH queue: %w",
			err,
		)
	}

	spool, err := export.NewSpool(
		cfg.Spool.ExportConfig(),
	)
	if err != nil {
		queue.Close()

		return nil, fmt.Errorf(
			"create OTGH spool: %w",
			err,
		)
	}

	protocolRuntime, err :=
		protocol.NewProtocol(
			cfg.Protocol,
		)
	if err != nil {
		queue.Close()
		spool.Close()

		return nil, fmt.Errorf(
			"create OTGH protocol: %w",
			err,
		)
	}

	router := export.NewRouter()

	exporterRegistry :=
		export.NewExporterRegistry()

	metrics := export.NewDeliveryMetrics()

	acknowledgements :=
		export.NewAcknowledgementManager(
			export.AcknowledgementObserverFunc(func(event export.AcknowledgementEvent) {
				recordAcknowledgementMetric(
					metrics,
					event,
				)
			}),
		)

	runtime := &Runtime{
		config: cfg,

		protocol: protocolRuntime,

		router: router,

		exporters: exporterRegistry,

		queue: queue,

		spool: spool,

		batchIDs: export.NewSequentialBatchIDGenerator(
			cfg.Application.NodeID,
		),

		acknowledgements:
			acknowledgements,

		metrics: metrics,

		retryConfig:
			cfg.Retry.ExportConfig(),

		workers: make(
			[]*export.Worker,
			0,
			cfg.Application.WorkerCount,
		),

		state: RuntimeStateNew,
	}

	return runtime, nil
}

// Config returns a defensive application configuration copy.
//
// Individual nested exporter configuration types already protect mutable maps
// when normalized, but callers should still treat the returned configuration
// as read-only runtime information.
func (r *Runtime) Config() config.Config {
	if r == nil {
		return config.Config{}
	}

	return r.config
}

// Protocol returns the shared OTGH protocol runtime.
func (r *Runtime) Protocol() *protocol.Protocol {
	if r == nil {
		return nil
	}

	return r.protocol
}

// Router returns the shared destination router.
func (r *Runtime) Router() *export.Router {
	if r == nil {
		return nil
	}

	return r.router
}

// Exporters returns the live exporter registry.
func (r *Runtime) Exporters() *export.ExporterRegistry {
	if r == nil {
		return nil
	}

	return r.exporters
}

// Queue returns the in-memory delivery queue.
func (r *Runtime) Queue() *export.Queue {
	if r == nil {
		return nil
	}

	return r.queue
}

// Spool returns the durable delivery spool.
func (r *Runtime) Spool() *export.Spool {
	if r == nil {
		return nil
	}

	return r.spool
}

// Acknowledgements returns the process-local delivery acknowledgement manager.
func (r *Runtime) Acknowledgements() *export.AcknowledgementManager {
	if r == nil {
		return nil
	}

	return r.acknowledgements
}

// Metrics returns the runtime metrics collector.
func (r *Runtime) Metrics() *export.DeliveryMetrics {
	if r == nil {
		return nil
	}

	return r.metrics
}

// RetryConfig returns the immutable shared delivery retry configuration.
//
// Each worker/exporter pairing will later construct a RetryExecutor using this
// policy together with that exporter's transport-specific RetryClassifier.
func (r *Runtime) RetryConfig() export.RetryConfig {
	if r == nil {
		return export.RetryConfig{}
	}

	return r.retryConfig
}

// State returns the current runtime state.
func (r *Runtime) State() RuntimeState {
	if r == nil {
		return RuntimeStateStopped
	}

	r.stateMu.RLock()
	defer r.stateMu.RUnlock()

	return r.state
}

// Running reports whether the runtime has reached its running state.
func (r *Runtime) Running() bool {
	return r != nil &&
		r.State() == RuntimeStateRunning
}

// Closed reports whether final shutdown has completed.
func (r *Runtime) Closed() bool {
	if r == nil {
		return true
	}

	return r.closed.Load()
}

// Failure returns the failure that moved the runtime into failed state.
func (r *Runtime) Failure() error {
	if r == nil {
		return nil
	}

	r.stateMu.RLock()
	defer r.stateMu.RUnlock()

	return r.failure
}

// StartedAt returns the time the runtime entered Running state.
func (r *Runtime) StartedAt() time.Time {
	if r == nil {
		return time.Time{}
	}

	r.stateMu.RLock()
	defer r.stateMu.RUnlock()

	return r.startedAt
}

// StoppedAt returns the time final shutdown completed.
func (r *Runtime) StoppedAt() time.Time {
	if r == nil {
		return time.Time{}
	}

	r.stateMu.RLock()
	defer r.stateMu.RUnlock()

	return r.stoppedAt
}

// beginStart performs the guarded New -> Starting transition.
//
// workers.go will use this before constructing and starting workers.
func (r *Runtime) beginStart() error {
	if r == nil {
		return errors.New(
			"OTGH runtime is nil",
		)
	}

	r.stateMu.Lock()
	defer r.stateMu.Unlock()

	if r.closed.Load() {
		return errors.New(
			"OTGH runtime is closed",
		)
	}

	switch r.state {
	case RuntimeStateNew:
		r.state = RuntimeStateStarting
		return nil

	case RuntimeStateStarting:
		return errors.New(
			"OTGH runtime is already starting",
		)

	case RuntimeStateRunning:
		return errors.New(
			"OTGH runtime is already running",
		)

	case RuntimeStateStopping,
		RuntimeStateStopped:
		return errors.New(
			"OTGH runtime has been stopped",
		)

	case RuntimeStateFailed:
		if r.failure != nil {
			return fmt.Errorf(
				"OTGH runtime previously failed: %w",
				r.failure,
			)
		}

		return errors.New(
			"OTGH runtime previously failed",
		)

	default:
		return fmt.Errorf(
			"OTGH runtime has invalid state %d",
			r.state,
		)
	}
}

// markRunning completes startup.
func (r *Runtime) markRunning() error {
	if r == nil {
		return errors.New(
			"OTGH runtime is nil",
		)
	}

	r.stateMu.Lock()
	defer r.stateMu.Unlock()

	if r.state != RuntimeStateStarting {
		return fmt.Errorf(
			"cannot mark OTGH runtime running from state %s",
			r.state,
		)
	}

	r.state = RuntimeStateRunning
	r.startedAt = time.Now().UTC()

	return nil
}

// markStopping begins shutdown.
//
// Calling markStopping more than once is safe.
func (r *Runtime) markStopping() {
	if r == nil {
		return
	}

	r.stateMu.Lock()
	defer r.stateMu.Unlock()

	switch r.state {
	case RuntimeStateStopped:
		return

	case RuntimeStateFailed:
		// Preserve failed state until resources have been closed. shutdown.go
		// can then record StoppedAt without erasing the failure.

	default:
		r.state = RuntimeStateStopping
	}
}

// markStopped records completed shutdown.
func (r *Runtime) markStopped() {
	if r == nil {
		return
	}

	r.stateMu.Lock()
	defer r.stateMu.Unlock()

	r.stoppedAt = time.Now().UTC()

	if r.state != RuntimeStateFailed {
		r.state = RuntimeStateStopped
	}

	r.closed.Store(true)
}

// markFailed records a fatal runtime failure.
func (r *Runtime) markFailed(
	err error,
) {
	if r == nil || err == nil {
		return
	}

	r.stateMu.Lock()

	if r.failure == nil {
		r.failure = err
	}

	r.state = RuntimeStateFailed

	r.stateMu.Unlock()

	r.cancelRun()
}

// setRunCancel stores the cancellation function controlling runtime workers.
func (r *Runtime) setRunCancel(
	cancel context.CancelFunc,
) {
	if r == nil {
		return
	}

	r.runCancelMu.Lock()
	defer r.runCancelMu.Unlock()

	r.runCancel = cancel
}

// cancelRun requests worker shutdown.
func (r *Runtime) cancelRun() {
	if r == nil {
		return
	}

	r.runCancelMu.Lock()

	cancel := r.runCancel
	r.runCancel = nil

	r.runCancelMu.Unlock()

	if cancel != nil {
		cancel()
	}
}

// addWorker registers one runtime-owned worker.
func (r *Runtime) addWorker(
	worker *export.Worker,
) error {
	if r == nil {
		return errors.New(
			"OTGH runtime is nil",
		)
	}

	if worker == nil {
		return errors.New(
			"cannot register nil OTGH worker",
		)
	}

	r.workersMu.Lock()
	defer r.workersMu.Unlock()

	for _, existing := range r.workers {
		if existing != nil &&
			existing.ID() == worker.ID() {
			return fmt.Errorf(
				"worker %q is already registered",
				worker.ID(),
			)
		}
	}

	r.workers = append(
		r.workers,
		worker,
	)

	return nil
}

// Workers returns a defensive slice containing currently registered workers.
func (r *Runtime) Workers() []*export.Worker {
	if r == nil {
		return nil
	}

	r.workersMu.RLock()
	defer r.workersMu.RUnlock()

	return append(
		[]*export.Worker(nil),
		r.workers...,
	)
}

// Snapshot returns a diagnostic summary of the runtime.
func (r *Runtime) Snapshot() RuntimeSnapshot {
	if r == nil {
		return RuntimeSnapshot{
			State: RuntimeStateStopped,
		}
	}

	r.stateMu.RLock()

	snapshot := RuntimeSnapshot{
		NodeID: r.config.Application.NodeID,

		State: r.state,

		StartedAt: r.startedAt,
		StoppedAt: r.stoppedAt,

		Closed: r.closed.Load(),
	}

	if r.failure != nil {
		snapshot.Failure =
			r.failure.Error()
	}

	r.stateMu.RUnlock()

	r.workersMu.RLock()
	snapshot.Workers =
		len(r.workers)
	r.workersMu.RUnlock()

	snapshot.Exporters =
		r.exporters.Count()

	snapshot.Metrics =
		r.metrics.Snapshot(
			r.queue,
			r.spool,
		)

	return snapshot
}

// RuntimeSnapshot is an immutable diagnostic view of OTGH.
type RuntimeSnapshot struct {
	NodeID string

	State RuntimeState

	Workers   int
	Exporters int

	StartedAt time.Time
	StoppedAt time.Time

	Closed bool

	Failure string

	Metrics export.DeliveryMetricsSnapshot
}

func recordAcknowledgementMetric(
	metrics *export.DeliveryMetrics,
	event export.AcknowledgementEvent,
) {
	if metrics == nil {
		return
	}

	// Only transitions into the target state are counted. Re-notifying the
	// same state must not increase cumulative metrics.
	if event.Previous == event.Current {
		return
	}

	switch event.Current {
	case export.AcknowledgementAcknowledged:
		metrics.Acknowledged()

	case export.AcknowledgementPartial:
		metrics.PartialAcknowledgement()

	case export.AcknowledgementFailed:
		metrics.BatchFailed()

	case export.AcknowledgementRejected:
		metrics.BatchDeadLettered()
	}
}
