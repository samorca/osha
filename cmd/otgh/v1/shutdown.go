package v1

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// shutdownState protects idempotent runtime shutdown.
//
// We keep this package-level map out of Runtime itself because runtime.go was
// intentionally limited to component ownership/lifecycle state. If we later
// decide to make Shutdown state part of Runtime directly, this can move there
// without changing the public Shutdown API.
var runtimeShutdowns sync.Map

type runtimeShutdownState struct {
	once sync.Once

	done chan struct{}

	mu  sync.Mutex
	err error
}

func newRuntimeShutdownState() *runtimeShutdownState {
	return &runtimeShutdownState{
		done: make(chan struct{}),
	}
}

// Shutdown gracefully stops OTGH and releases shared resources.
//
// Shutdown is idempotent. Concurrent callers wait for the same shutdown
// operation and receive the same final error.
//
// Shutdown order:
//
//   1. Mark runtime as stopping.
//   2. Close queue to reject new batches.
//   3. Give workers an opportunity to drain queued batches.
//   4. Cancel/close workers.
//   5. Close exporters.
//   6. Close acknowledgement tracking.
//   7. Close durable spool.
//   8. Mark runtime stopped.
//
// Failed deliveries remain durable in the spool. Shutdown does not delete
// unacknowledged spool entries.
func (r *Runtime) Shutdown(
	ctx context.Context,
) error {
	if r == nil {
		return nil
	}

	if ctx == nil {
		return errors.New(
			"OTGH shutdown context is nil",
		)
	}

	state := shutdownStateFor(r)

	state.once.Do(func() {
		go func() {
			defer close(state.done)

			err := r.shutdown(ctx)

			state.mu.Lock()
			state.err = err
			state.mu.Unlock()
		}()
	})

	select {
	case <-state.done:
		state.mu.Lock()
		err := state.err
		state.mu.Unlock()

		return err

	case <-ctx.Done():
		return ctx.Err()
	}
}

// ShutdownWithTimeout performs shutdown using the configured application
// timeout.
//
// It deliberately derives shutdown from context.Background rather than the
// runtime's already-cancelled run context. A SIGTERM/caller cancellation that
// stops normal processing should not instantly cancel cleanup as well.
func (r *Runtime) ShutdownWithTimeout() error {
	if r == nil {
		return nil
	}

	timeout :=
		r.config.Application.ShutdownTimeout

	if timeout <= 0 {
		return errors.New(
			"OTGH shutdown timeout must be greater than zero",
		)
	}

	ctx, cancel := context.WithTimeout(
		context.Background(),
		timeout,
	)
	defer cancel()

	return r.Shutdown(ctx)
}

func (r *Runtime) shutdown(
	ctx context.Context,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	r.markStopping()

	var shutdownErrors []error

	// Stop accepting new batches immediately. Existing queue entries remain
	// available for workers to drain because Queue.Close preserves queued
	// items.
	if r.queue != nil {
		r.queue.Close()
	}

	// Give running workers time to drain the in-memory queue.
	//
	// Worker.Run exits naturally once Queue.Dequeue observes a closed and
	// drained queue.
	if r.queue != nil &&
		!r.queue.Empty() {
		if err := r.queue.WaitUntilEmpty(
			ctx,
			10*time.Millisecond,
		); err != nil {
			shutdownErrors = append(
				shutdownErrors,
				fmt.Errorf(
					"wait for OTGH queue drain: %w",
					err,
				),
			)
		}
	}

	// Whether drain completed or timed out, stop workers now. Any batch that
	// had reached ProcessBatch was placed in the durable spool before export.
	r.StopWorkers()

	// We do not currently have a WaitWorkers method because Start owns the
	// worker WaitGroup. Closing workers here prevents any new ProcessBatch
	// calls, while cancellation interrupts exporter operations that honor
	// their context.
	for _, worker := range r.Workers() {
		if worker == nil {
			continue
		}

		if worker.Running() {
			if err := waitWorkerStopped(
				ctx,
				worker,
			); err != nil {
				shutdownErrors = append(
					shutdownErrors,
					fmt.Errorf(
						"stop worker %q: %w",
						worker.ID(),
						err,
					),
				)
			}
		}
	}

	// Workers are no longer expected to issue exporter operations. Closing
	// network/storage clients before workers stop could interrupt work before
	// its spool state is settled.
	if r.exporters != nil {
		if err := r.exporters.CloseAll(); err != nil {
			shutdownErrors = append(
				shutdownErrors,
				fmt.Errorf(
					"close OTGH exporters: %w",
					err,
				),
			)
		}
	}

	if r.acknowledgements != nil {
		r.acknowledgements.Close()
	}

	if r.spool != nil {
		r.spool.Close()
	}

	r.markStopped()

	// Remove shutdown coordinator after completion. Runtime itself is closed,
	// so a future Shutdown call remains harmless via the fast path below.
	runtimeShutdowns.Delete(r)

	return errors.Join(
		shutdownErrors...,
	)
}

// ForceShutdown immediately requests cancellation and closes runtime
// resources without waiting for the queue to drain.
//
// This is intended as the fallback when graceful Shutdown exceeds its
// deadline. Durable spool entries are intentionally preserved.
func (r *Runtime) ForceShutdown() error {
	if r == nil {
		return nil
	}

	if r.Closed() {
		return nil
	}

	r.markStopping()

	r.cancelRun()

	if r.queue != nil {
		r.queue.Close()
	}

	r.StopWorkers()

	var shutdownErrors []error

	if r.exporters != nil {
		if err := r.exporters.CloseAll(); err != nil {
			shutdownErrors = append(
				shutdownErrors,
				fmt.Errorf(
					"force close OTGH exporters: %w",
					err,
				),
			)
		}
	}

	if r.acknowledgements != nil {
		r.acknowledgements.Close()
	}

	if r.spool != nil {
		r.spool.Close()
	}

	r.markStopped()

	runtimeShutdowns.Delete(r)

	return errors.Join(
		shutdownErrors...,
	)
}

func shutdownStateFor(
	r *Runtime,
) *runtimeShutdownState {
	if existing, found :=
		runtimeShutdowns.Load(r); found {
		return existing.(*runtimeShutdownState)
	}

	created :=
		newRuntimeShutdownState()

	actual, _ :=
		runtimeShutdowns.LoadOrStore(
			r,
			created,
		)

	return actual.(*runtimeShutdownState)
}

func waitWorkerStopped(
	ctx context.Context,
	worker interface {
		Running() bool
	},
) error {
	if ctx == nil {
		return errors.New(
			"worker stop context is nil",
		)
	}

	if worker == nil {
		return nil
	}

	ticker := time.NewTicker(
		10 * time.Millisecond,
	)
	defer ticker.Stop()

	for {
		if !worker.Running() {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-ticker.C:
		}
	}
}
