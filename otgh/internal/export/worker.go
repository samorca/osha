
package export

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrWorkerClosed indicates that the worker has been permanently closed.
	ErrWorkerClosed = errors.New("worker closed")

	// ErrWorkerRunning indicates that Run was called while the worker was
	// already active.
	ErrWorkerRunning = errors.New("worker already running")

	// ErrIncompleteDelivery indicates that an exporter returned without an
	// error but did not completely acknowledge the batch.
	ErrIncompleteDelivery = errors.New("incomplete batch delivery")
)

// WorkerConfig controls one delivery worker.
type WorkerConfig struct {
	// ID identifies the worker in diagnostics.
	ID string

	// HealthCheckBeforeExport requests an exporter health check before each
	// batch begins its delivery attempts.
	//
	// This is optional because some destinations do not provide a useful
	// health endpoint and a health check adds another network operation.
	HealthCheckBeforeExport bool
}

// Validate verifies the worker configuration.
func (c WorkerConfig) Validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return errors.New("worker id is required")
	}

	return nil
}

// Worker processes queued batches and delivers them through the exporter
// registry.
//
// Delivery lifecycle:
//
//   1. Dequeue a batch.
//   2. Persist it in the spool.
//   3. Start acknowledgement tracking.
//   4. Execute delivery with retry policy.
//   5. Apply exporter results.
//   6. Delete the spool entry only after complete acknowledgement.
//   7. Remove process-local acknowledgement state.
//
// A failed or cancelled delivery remains in the spool for later recovery.
type Worker struct {
	config WorkerConfig

	queue          *Queue
	spool          *Spool
	exporters      *ExporterRegistry
	retry          *RetryExecutor
	acknowledgements *AcknowledgementManager
	metrics        *DeliveryMetrics

	runMu sync.Mutex

	running atomic.Bool
	closed  atomic.Bool
}

// NewWorker creates a delivery worker.
func NewWorker(
	config WorkerConfig,
	queue *Queue,
	spool *Spool,
	exporters *ExporterRegistry,
	retry *RetryExecutor,
	acknowledgements *AcknowledgementManager,
	metrics *DeliveryMetrics,
) (*Worker, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("create worker: %w", err)
	}

	if queue == nil {
		return nil, errors.New("create worker: queue is required")
	}

	if spool == nil {
		return nil, errors.New("create worker: spool is required")
	}

	if exporters == nil {
		return nil, errors.New(
			"create worker: exporter registry is required",
		)
	}

	if retry == nil {
		return nil, errors.New(
			"create worker: retry executor is required",
		)
	}

	if acknowledgements == nil {
		return nil, errors.New(
			"create worker: acknowledgement manager is required",
		)
	}

	if metrics == nil {
		metrics = NewDeliveryMetrics()
	}

	return &Worker{
		config:           config,
		queue:            queue,
		spool:            spool,
		exporters:        exporters,
		retry:            retry,
		acknowledgements: acknowledgements,
		metrics:          metrics,
	}, nil
}

// ID returns the configured worker identity.
func (w *Worker) ID() string {
	if w == nil {
		return ""
	}

	return w.config.ID
}

// Run starts the worker's queue-consumption loop.
//
// Run blocks until:
//
//   - the context is cancelled;
//   - the queue is closed and drained;
//   - a queue error prevents continued operation.
//
// Individual delivery failure does not stop the worker. Failed batches remain
// durably stored in the spool so other batches can continue to be processed.
func (w *Worker) Run(ctx context.Context) error {
	if w == nil {
		return errors.New("worker is nil")
	}

	if ctx == nil {
		return errors.New("worker context is nil")
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	w.runMu.Lock()

	if w.closed.Load() {
		w.runMu.Unlock()
		return ErrWorkerClosed
	}

	if !w.running.CompareAndSwap(false, true) {
		w.runMu.Unlock()
		return ErrWorkerRunning
	}

	w.runMu.Unlock()

	w.metrics.WorkerStarted()

	defer func() {
		w.running.Store(false)
		w.metrics.WorkerStopped()
	}()

	for {
		batch, err := w.queue.Dequeue(ctx)
		if err != nil {
			switch {
			case errors.Is(err, ErrQueueClosed):
				return nil

			case errors.Is(err, context.Canceled),
				errors.Is(err, context.DeadlineExceeded):
				return err

			default:
				return fmt.Errorf(
					"worker %q dequeue batch: %w",
					w.config.ID,
					err,
				)
			}
		}

		w.metrics.BatchDequeued()

		// Delivery errors are intentionally not returned from Run. The batch
		// remains protected by the spool and the worker continues processing
		// unrelated work.
		_ = w.ProcessBatch(ctx, batch)

		if err := ctx.Err(); err != nil {
			return err
		}
	}
}

// ProcessBatch performs the complete durable delivery lifecycle for one batch.
//
// ProcessBatch can also be used by recovery code to process a batch loaded
// directly from the spool.
func (w *Worker) ProcessBatch(
	ctx context.Context,
	batch Batch,
) error {
	if w == nil {
		return errors.New("worker is nil")
	}

	if ctx == nil {
		return errors.New("worker context is nil")
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if w.closed.Load() {
		return ErrWorkerClosed
	}

	if err := batch.Validate(); err != nil {
		return fmt.Errorf(
			"worker %q received invalid batch: %w",
			w.config.ID,
			err,
		)
	}

	if err := w.ensureSpooled(ctx, batch); err != nil {
		w.metrics.SpoolError()

		return fmt.Errorf(
			"worker %q persist batch %q: %w",
			w.config.ID,
			batch.ID,
			err,
		)
	}

	if err := w.ensureTracked(ctx, batch); err != nil {
		return fmt.Errorf(
			"worker %q track batch %q: %w",
			w.config.ID,
			batch.ID,
			err,
		)
	}

	if w.config.HealthCheckBeforeExport {
		if err := w.exporters.Health(
			ctx,
			batch.Destination.ID,
		); err != nil {
			_, _ = w.acknowledgements.MarkRetrying(
				context.WithoutCancel(ctx),
				batch.ID,
				"exporter_unavailable",
				err.Error(),
			)

			return fmt.Errorf(
				"worker %q destination %q unavailable: %w",
				w.config.ID,
				batch.Destination.ID,
				err,
			)
		}
	}

	result, err := w.retry.Execute(
		ctx,
		func(
			attemptCtx context.Context,
			attempt int,
		) (DeliveryResult, error) {
			return w.attemptDelivery(
				attemptCtx,
				batch,
				attempt,
			)
		},
	)

	if err != nil {
		w.handleDeliveryFailure(ctx, batch, result, err)
		return fmt.Errorf(
			"worker %q deliver batch %q: %w",
			w.config.ID,
			batch.ID,
			err,
		)
	}

	ack, ackErr := w.acknowledgements.Get(ctx, batch.ID)
	if ackErr != nil {
		return fmt.Errorf(
			"worker %q retrieve acknowledgement for batch %q: %w",
			w.config.ID,
			batch.ID,
			ackErr,
		)
	}

	if !ack.Complete() {
		err := fmt.Errorf(
			"%w: batch %q accepted %d of %d records",
			ErrIncompleteDelivery,
			batch.ID,
			ack.AcceptedRecords,
			ack.TotalRecords,
		)

		w.metrics.PartialAcknowledgement()

		_, _ = w.acknowledgements.MarkRetrying(
			context.WithoutCancel(ctx),
			batch.ID,
			"incomplete_delivery",
			err.Error(),
		)

		return err
	}

	if err := w.completeAcknowledgedBatch(ctx, batch); err != nil {
		return fmt.Errorf(
			"worker %q complete batch %q: %w",
			w.config.ID,
			batch.ID,
			err,
		)
	}

	return nil
}

// Recover processes all batches currently stored in the durable spool.
//
// Entries are processed in the stable order returned by Spool.Recover.
// Successfully acknowledged entries are deleted by ProcessBatch. Failed
// entries remain in the spool.
func (w *Worker) Recover(ctx context.Context) error {
	if w == nil {
		return errors.New("worker is nil")
	}

	if ctx == nil {
		return errors.New("worker recovery context is nil")
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if w.closed.Load() {
		return ErrWorkerClosed
	}

	batches, err := w.spool.Recover(ctx)
	if err != nil {
		w.metrics.SpoolError()

		return fmt.Errorf(
			"worker %q recover spool: %w",
			w.config.ID,
			err,
		)
	}

	if len(batches) > 0 {
		w.metrics.SpoolRecovery(uint64(len(batches)))
	}

	var recoveryErrors []error

	for _, batch := range batches {
		if err := ctx.Err(); err != nil {
			return errors.Join(
				append(recoveryErrors, err)...,
			)
		}

		w.metrics.SpoolRead()

		if err := w.ProcessBatch(ctx, batch); err != nil {
			recoveryErrors = append(
				recoveryErrors,
				fmt.Errorf(
					"recover batch %q: %w",
					batch.ID,
					err,
				),
			)
		}
	}

	return errors.Join(recoveryErrors...)
}

// Running reports whether Run is currently active.
func (w *Worker) Running() bool {
	if w == nil {
		return false
	}

	return w.running.Load()
}

// Close prevents future Run and ProcessBatch calls.
//
// Close does not close shared queue, spool, exporter registry, or
// acknowledgement manager. Their lifecycle may be shared by multiple workers
// and must be coordinated by the application entry point.
func (w *Worker) Close() {
	if w == nil {
		return
	}

	w.closed.Store(true)
}

// Closed reports whether the worker has been closed.
func (w *Worker) Closed() bool {
	if w == nil {
		return true
	}

	return w.closed.Load()
}

func (w *Worker) attemptDelivery(
	ctx context.Context,
	batch Batch,
	attempt int,
) (DeliveryResult, error) {
	if err := ctx.Err(); err != nil {
		return DeliveryResult{}, err
	}

	if _, err := w.acknowledgements.BeginAttempt(
		ctx,
		batch.ID,
		attempt,
	); err != nil {
		return DeliveryResult{}, fmt.Errorf(
			"begin delivery attempt %d: %w",
			attempt,
			err,
		)
	}

	w.metrics.ExportAttempt()
	w.metrics.DeliveryStarted()

	startedAt := time.Now().UTC()

	result, exportErr := w.exporters.ExportTo(ctx, batch)

	finishedAt := time.Now().UTC()
	duration := finishedAt.Sub(startedAt)

	w.metrics.DeliveryFinished()

	result = normaliseDeliveryResult(
		batch,
		result,
		attempt,
		startedAt,
		finishedAt,
		exportErr,
	)

	if _, err := w.acknowledgements.ApplyResult(
		context.WithoutCancel(ctx),
		result,
	); err != nil {
		w.metrics.ExportFailed(
			uint64Safe(result.RejectedRecords),
			duration,
		)

		if exportErr != nil {
			return result, errors.Join(
				exportErr,
				fmt.Errorf(
					"apply delivery result: %w",
					err,
				),
			)
		}

		return result, fmt.Errorf(
			"apply delivery result: %w",
			err,
		)
	}

	if exportErr != nil {
		w.metrics.ExportFailed(
			uint64Safe(result.RejectedRecords),
			duration,
		)

		return result, exportErr
	}

	if !result.Successful() {
		w.metrics.ExportFailed(
			uint64Safe(result.RejectedRecords),
			duration,
		)

		return result, fmt.Errorf(
			"%w: batch %q status=%s accepted=%d rejected=%d",
			ErrIncompleteDelivery,
			batch.ID,
			result.Status,
			result.AcceptedRecords,
			result.RejectedRecords,
		)
	}

	w.metrics.ExportSucceeded(
		uint64Safe(result.AcceptedRecords),
		uint64(batch.PayloadSize()),
		duration,
	)

	return result, nil
}

func (w *Worker) ensureSpooled(
	ctx context.Context,
	batch Batch,
) error {
	err := w.spool.Put(ctx, batch)

	if err == nil {
		w.metrics.SpoolWrite()
		return nil
	}

	if errors.Is(err, ErrSpoolEntryExists) {
		existing, getErr := w.spool.Get(ctx, batch.ID)
		if getErr != nil {
			return errors.Join(err, getErr)
		}

		if existing.Destination.ID != batch.Destination.ID {
			return fmt.Errorf(
				"%w: batch %q spool destination %q differs from requested destination %q",
				ErrAcknowledgementConflict,
				batch.ID,
				existing.Destination.ID,
				batch.Destination.ID,
			)
		}

		w.metrics.SpoolRead()
		return nil
	}

	return err
}

func (w *Worker) ensureTracked(
	ctx context.Context,
	batch Batch,
) error {
	_, err := w.acknowledgements.Track(ctx, batch)

	if err == nil {
		return nil
	}

	if !errors.Is(err, ErrAcknowledgementExists) {
		return err
	}

	existing, getErr := w.acknowledgements.Get(ctx, batch.ID)
	if getErr != nil {
		return errors.Join(err, getErr)
	}

	if existing.DestinationID != batch.Destination.ID ||
		existing.TotalRecords != batch.RecordCount() {
		return fmt.Errorf(
			"%w: existing acknowledgement does not match batch %q",
			ErrAcknowledgementConflict,
			batch.ID,
		)
	}

	if existing.State.Terminal() {
		return fmt.Errorf(
			"%w: existing acknowledgement for batch %q is terminal: %s",
			ErrAcknowledgementConflict,
			batch.ID,
			existing.State,
		)
	}

	return nil
}

func (w *Worker) completeAcknowledgedBatch(
	ctx context.Context,
	batch Batch,
) error {
	if err := w.spool.Delete(ctx, batch.ID); err != nil {
		w.metrics.SpoolError()
		return fmt.Errorf(
			"delete acknowledged spool entry: %w",
			err,
		)
	}

	w.metrics.SpoolDelete()
	w.metrics.Acknowledged()

	if _, err := w.acknowledgements.Remove(
		ctx,
		batch.ID,
	); err != nil {
		return fmt.Errorf(
			"remove acknowledgement state: %w",
			err,
		)
	}

	return nil
}

func (w *Worker) handleDeliveryFailure(
	ctx context.Context,
	batch Batch,
	result DeliveryResult,
	deliveryErr error,
) {
	safeCtx := context.WithoutCancel(ctx)

	switch {
	case errors.Is(deliveryErr, ErrRetryExhausted):
		w.metrics.RetryExhausted()
		w.metrics.BatchFailed()

		_, _ = w.acknowledgements.MarkFailed(
			safeCtx,
			batch.ID,
			"retry_exhausted",
			deliveryErr.Error(),
		)

	case errors.Is(deliveryErr, ErrRetryNotAllowed):
		w.metrics.RetrySuppressed()
		w.metrics.BatchFailed()

		_, _ = w.acknowledgements.MarkFailed(
			safeCtx,
			batch.ID,
			"non_retryable",
			deliveryErr.Error(),
		)

	case errors.Is(deliveryErr, context.Canceled),
		errors.Is(deliveryErr, context.DeadlineExceeded):
		_, _ = w.acknowledgements.MarkRetrying(
			safeCtx,
			batch.ID,
			"delivery_interrupted",
			deliveryErr.Error(),
		)

	default:
		if result.Status == DeliveryDeadLettered {
			w.metrics.BatchDeadLettered()

			_, _ = w.acknowledgements.MarkFailed(
				safeCtx,
				batch.ID,
				"dead_lettered",
				deliveryErr.Error(),
			)

			return
		}

		_, _ = w.acknowledgements.MarkRetrying(
			safeCtx,
			batch.ID,
			"delivery_failed",
			deliveryErr.Error(),
		)
	}
}

// normaliseDeliveryResult guarantees enough identity and timing information
// for acknowledgement tracking, even when an exporter fails before it can
// construct a complete result.
func normaliseDeliveryResult(
	batch Batch,
	result DeliveryResult,
	attempt int,
	startedAt time.Time,
	finishedAt time.Time,
	exportErr error,
) DeliveryResult {
	if result.BatchID == "" {
		result.BatchID = batch.ID
	}

	if result.DestinationID == "" {
		result.DestinationID = batch.Destination.ID
	}

	if result.Attempt <= 0 {
		result.Attempt = attempt
	}

	if result.StartedAt.IsZero() {
		result.StartedAt = startedAt
	}

	if result.FinishedAt.IsZero() {
		result.FinishedAt = finishedAt
	}

	if exportErr != nil {
		if result.Status == DeliveryPending ||
			result.Status == DeliveryInFlight ||
			result.Status == DeliveryAcknowledged {
			result.Status = DeliveryRetrying
		}

		if result.ErrorMessage == "" {
			result.ErrorMessage = exportErr.Error()
		}

		result.Retryable = true
	}

	return result
}

func uint64Safe(value int) uint64 {
	if value <= 0 {
		return 0
	}

	return uint64(value)
}
