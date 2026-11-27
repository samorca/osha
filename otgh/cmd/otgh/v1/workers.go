package v1

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/samorca/otgh/internal/export"
)

// Start prepares workers, performs durable spool recovery, and starts the
// runtime worker pool.
//
// Start blocks until:
//
//   - the caller's context is cancelled;
//   - the queue is closed and drained;
//   - a worker encounters a fatal queue/run-loop failure.
//
// Individual batch-delivery failures do not normally stop a worker. Worker
// keeps those batches in the durable spool and continues processing unrelated
// work.
func (r *Runtime) Start(
	ctx context.Context,
) error {
	if r == nil {
		return errors.New(
			"OTGH runtime is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"OTGH runtime context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := r.beginStart(); err != nil {
		return err
	}

	runCtx, cancel := context.WithCancel(ctx)

	r.setRunCancel(cancel)

	defer func() {
		cancel()
		r.setRunCancel(nil)
	}()

	if r.exporters.Count() == 0 &&
		r.config.Exporters.Count() > 0 {
		err := errors.New(
			"OTGH exporters have not been configured",
		)

		r.markFailed(err)

		return err
	}

	if err := r.createWorkers(); err != nil {
		r.markFailed(err)

		return err
	}

	workers := r.Workers()

	if len(workers) == 0 {
		err := errors.New(
			"OTGH runtime has no workers",
		)

		r.markFailed(err)

		return err
	}

	// Recover durable batches before workers begin consuming newly queued
	// work. Only one worker performs recovery so the same spool entry cannot
	// be processed concurrently by several workers during startup.
	//
	// If recovery fails, the spool entries remain durable. Startup stops so
	// an operator does not unknowingly run with an incomplete recovery state.
	if err := workers[0].Recover(runCtx); err != nil {
		if ctxErr := runCtx.Err(); ctxErr != nil {
			return ctxErr
		}

		recoveryErr := fmt.Errorf(
			"recover OTGH durable spool: %w",
			err,
		)

		r.markFailed(recoveryErr)

		return recoveryErr
	}

	if err := r.markRunning(); err != nil {
		r.markFailed(err)

		return err
	}

	return r.runWorkers(runCtx, workers)
}

// createWorkers constructs the configured number of workers.
//
// Every worker consumes from the same bounded queue and shares:
//
//   - exporter registry;
//   - durable spool;
//   - acknowledgement manager;
//   - metrics collector.
//
// Each worker owns a separate RetryExecutor. RetryExecutor configuration is
// immutable, while the classifier delegates transport-specific decisions to
// registered exporters.
func (r *Runtime) createWorkers() error {
	if r == nil {
		return errors.New(
			"OTGH runtime is nil",
		)
	}

	r.workersMu.RLock()
	existingWorkers := len(r.workers)
	r.workersMu.RUnlock()

	if existingWorkers > 0 {
		return errors.New(
			"OTGH workers are already configured",
		)
	}

	classifier := &exporterRetryClassifier{
		registry: r.exporters,
	}

	for index := 0;
		index < r.config.Application.WorkerCount;
		index++ {
		workerID := fmt.Sprintf(
			"otgh-worker-%d",
			index+1,
		)

		retryExecutor, err :=
			export.NewRetryExecutor(
				r.retryConfig,
				classifier,
				export.RetryObserverFunc(
					func(
						event export.RetryEvent,
					) {
						r.metrics.
							RetryScheduled()
					},
				),
			)
		if err != nil {
			r.closeRegisteredWorkers()

			return fmt.Errorf(
				"create retry executor for worker %q: %w",
				workerID,
				err,
			)
		}

		worker, err := export.NewWorker(
			export.WorkerConfig{
				ID: workerID,

				// Export endpoint health checks are not performed before
				// every batch. That would add another network operation for
				// each delivery and some destinations do not implement a
				// compatible health endpoint.
				HealthCheckBeforeExport: false,
			},
			r.queue,
			r.spool,
			r.exporters,
			retryExecutor,
			r.acknowledgements,
			r.metrics,
		)
		if err != nil {
			r.closeRegisteredWorkers()

			return fmt.Errorf(
				"create worker %q: %w",
				workerID,
				err,
			)
		}

		if err := r.addWorker(worker); err != nil {
			worker.Close()
			r.closeRegisteredWorkers()

			return fmt.Errorf(
				"register worker %q: %w",
				workerID,
				err,
			)
		}
	}

	return nil
}

// runWorkers executes the bounded worker pool and waits for every worker.
//
// The first unexpected worker failure cancels the shared run context. Expected
// cancellation errors are ignored during coordinated shutdown.
func (r *Runtime) runWorkers(
	ctx context.Context,
	workers []*export.Worker,
) error {
	if len(workers) == 0 {
		return errors.New(
			"cannot run an empty OTGH worker pool",
		)
	}

	workerCtx, cancel :=
		context.WithCancel(ctx)
	defer cancel()

	results := make(
		chan workerResult,
		len(workers),
	)

	var waitGroup sync.WaitGroup

	waitGroup.Add(len(workers))

	for _, worker := range workers {
		worker := worker

		go func() {
			defer waitGroup.Done()

			err := worker.Run(workerCtx)

			results <- workerResult{
				workerID: worker.ID(),
				err:      err,
			}
		}()
	}

	// Close the results channel only after every worker has finished sending.
	go func() {
		waitGroup.Wait()
		close(results)
	}()

	var (
		runErrors []error
		cancelled bool
	)

	for result := range results {
		if result.err == nil {
			continue
		}

		if expectedWorkerStop(
			result.err,
			workerCtx,
		) {
			continue
		}

		runErrors = append(
			runErrors,
			fmt.Errorf(
				"worker %q failed: %w",
				result.workerID,
				result.err,
			),
		)

		if !cancelled {
			cancelled = true
			cancel()
		}
	}

	if len(runErrors) > 0 {
		err := errors.Join(
			runErrors...,
		)

		r.markFailed(err)

		return err
	}

	// Cancellation from the caller is the normal signal-driven shutdown path.
	// Return its error so main.go can distinguish cancellation from a
	// naturally closed/drained queue when deciding what to log.
	if err := ctx.Err(); err != nil {
		return err
	}

	return nil
}

// StopWorkers requests worker cancellation and closes each worker.
//
// The queue, spool, exporter registry, and acknowledgement manager are shared
// resources. shutdown.go will close those components in the correct order.
func (r *Runtime) StopWorkers() {
	if r == nil {
		return
	}

	r.cancelRun()

	for _, worker := range r.Workers() {
		if worker != nil {
			worker.Close()
		}
	}
}

func (r *Runtime) closeRegisteredWorkers() {
	if r == nil {
		return
	}

	r.workersMu.Lock()
	defer r.workersMu.Unlock()

	for _, worker := range r.workers {
		if worker != nil {
			worker.Close()
		}
	}

	r.workers = nil
}

type workerResult struct {
	workerID string
	err      error
}

func expectedWorkerStop(
	err error,
	ctx context.Context,
) bool {
	if err == nil {
		return true
	}

	if errors.Is(
		err,
		context.Canceled,
	) || errors.Is(
		err,
		context.DeadlineExceeded,
	) || errors.Is(
		err,
		export.ErrQueueClosed,
	) || errors.Is(
		err,
		export.ErrWorkerClosed,
	) {
		return true
	}

	if ctx != nil &&
		ctx.Err() != nil {
		return true
	}

	return false
}

// exporterRetryClassifier delegates retry decisions to registered exporters.
//
// This preserves the separation we established earlier:
//
//   - retry.go controls attempts, delays, jitter, and elapsed-time budget;
//   - HTTP understands HTTP failures;
//   - gRPC understands gRPC status codes;
//   - object-store understands provider errors;
//   - webhook understands webhook/HTTP outcomes.
//
// The classifier contains no mutable state and is safe for concurrent use.
type exporterRetryClassifier struct {
	registry *export.ExporterRegistry
}

// ClassifyRetry implements export.RetryClassifier.
func (c *exporterRetryClassifier) ClassifyRetry(
	err error,
) export.RetryDecision {
	if err == nil {
		return export.RetryDecision{
			Retry:  false,
			Reason: "operation succeeded",
		}
	}

	if errors.Is(
		err,
		context.Canceled,
	) || errors.Is(
		err,
		context.DeadlineExceeded,
	) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "delivery context ended",
		}
	}

	if c == nil || c.registry == nil {
		return export.RetryDecision{
			Retry:  false,
			Reason: "exporter registry unavailable",
		}
	}

	exporterIDs := c.registry.IDs()

	var reasons []string

	for _, exporterID := range exporterIDs {
		instance, getErr :=
			c.registry.Get(exporterID)
		if getErr != nil ||
			instance == nil {
			continue
		}

		classifier, ok := instance.(export.RetryClassifier)
		if !ok {
			continue
		}

		decision :=
			classifier.ClassifyRetry(err)

		if decision.Reason != "" {
			reasons = append(
				reasons,
				fmt.Sprintf(
					"%s: %s",
					exporterID,
					decision.Reason,
				),
			)
		}

		if decision.Retry {
			return export.RetryDecision{
				Retry: true,

				Reason: fmt.Sprintf(
					"exporter %q classified failure as retryable: %s",
					exporterID,
					decision.Reason,
				),
			}
		}
	}

	reason := "no exporter classified failure as retryable"

	if len(reasons) > 0 {
		reason = errors.Join(
			stringErrors(reasons)...,
		).Error()
	}

	return export.RetryDecision{
		Retry:  false,
		Reason: reason,
	}
}

func stringErrors(
	values []string,
) []error {
	result := make(
		[]error,
		0,
		len(values),
	)

	for _, value := range values {
		if value == "" {
			continue
		}

		result = append(
			result,
			errors.New(value),
		)
	}

	return result
}

// Compile-time interface assertion.
var _ export.RetryClassifier =
	(*exporterRetryClassifier)(nil)
