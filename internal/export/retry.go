package export

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"time"
)

var (
	// ErrRetryExhausted indicates that the configured retry budget has been
	// exhausted without a successful operation.
	ErrRetryExhausted = errors.New("retry exhausted")

	// ErrRetryNotAllowed indicates that an error was classified as
	// non-retryable.
	ErrRetryNotAllowed = errors.New("retry not allowed")
)

// RetryConfig controls retry behaviour for failed delivery operations.
//
// OTGH deliberately owns retry policy centrally rather than allowing every
// exporter to implement a different retry strategy.
type RetryConfig struct {
	// MaxAttempts is the maximum total number of operation attempts,
	// including the initial attempt.
	//
	// For example, MaxAttempts=4 means:
	//
	//   attempt 1
	//   retry attempt 2
	//   retry attempt 3
	//   retry attempt 4
	MaxAttempts int

	// InitialDelay is the base delay before the first retry.
	InitialDelay time.Duration

	// MaxDelay caps the calculated delay between attempts.
	MaxDelay time.Duration

	// Multiplier controls exponential growth.
	//
	// Example:
	//
	//   InitialDelay = 1s
	//   Multiplier   = 2
	//
	// produces base delays:
	//
	//   1s, 2s, 4s, 8s...
	Multiplier float64

	// Jitter is the proportional random variation applied to each delay.
	//
	// Valid values are in the range [0, 1].
	//
	// Jitter=0.20 means a calculated delay may vary by up to 20 percent
	// below or above its base value.
	Jitter float64

	// MaxElapsedTime limits the total retry window.
	//
	// Zero disables the elapsed-time limit, leaving MaxAttempts as the
	// controlling retry budget.
	MaxElapsedTime time.Duration
}

// Validate verifies the retry configuration.
func (c RetryConfig) Validate() error {
	if c.MaxAttempts <= 0 {
		return errors.New("retry max attempts must be greater than zero")
	}

	if c.InitialDelay <= 0 {
		return errors.New("retry initial delay must be greater than zero")
	}

	if c.MaxDelay <= 0 {
		return errors.New("retry max delay must be greater than zero")
	}

	if c.MaxDelay < c.InitialDelay {
		return errors.New(
			"retry max delay must be greater than or equal to initial delay",
		)
	}

	if math.IsNaN(c.Multiplier) ||
		math.IsInf(c.Multiplier, 0) ||
		c.Multiplier < 1 {
		return errors.New(
			"retry multiplier must be a finite value greater than or equal to one",
		)
	}

	if math.IsNaN(c.Jitter) ||
		math.IsInf(c.Jitter, 0) ||
		c.Jitter < 0 ||
		c.Jitter > 1 {
		return errors.New(
			"retry jitter must be a finite value between zero and one",
		)
	}

	if c.MaxElapsedTime < 0 {
		return errors.New(
			"retry max elapsed time cannot be negative",
		)
	}

	return nil
}

// RetryDecision describes whether an operation should be retried.
//
// An optional Reason helps diagnostics without forcing the retry subsystem to
// interpret transport-specific errors.
type RetryDecision struct {
	Retry  bool
	Reason string
}

// RetryClassifier determines whether a failed operation can be retried.
//
// Concrete transports understand their own failures best. For example, an
// HTTP exporter can distinguish a retryable server failure from a permanent
// request validation failure without teaching retry.go about HTTP status
// codes.
type RetryClassifier interface {
	ClassifyRetry(error) RetryDecision
}

// RetryClassifierFunc adapts a function to RetryClassifier.
type RetryClassifierFunc func(error) RetryDecision

// ClassifyRetry implements RetryClassifier.
func (f RetryClassifierFunc) ClassifyRetry(err error) RetryDecision {
	if f == nil {
		return RetryDecision{
			Retry:  false,
			Reason: "retry classifier is nil",
		}
	}

	return f(err)
}

// RetryOperation is a unit of work that may be retried.
//
// attempt starts at 1 and increases for each execution.
type RetryOperation func(
	ctx context.Context,
	attempt int,
) (DeliveryResult, error)

// RetryEvent describes one retry lifecycle event.
//
// This can later feed metrics, structured logs, or tracing without hard
// coupling the retry engine to those systems.
type RetryEvent struct {
	Attempt     int
	NextAttempt int

	Delay time.Duration

	Error error

	Decision RetryDecision

	Elapsed time.Duration
}

// RetryObserver receives retry lifecycle events.
//
// Implementations must return quickly and must not panic. RetryExecutor does
// not recover observer panics because silently swallowing application defects
// would make production failures harder to diagnose.
type RetryObserver interface {
	OnRetry(RetryEvent)
}

// RetryObserverFunc adapts a function to RetryObserver.
type RetryObserverFunc func(RetryEvent)

// OnRetry implements RetryObserver.
func (f RetryObserverFunc) OnRetry(event RetryEvent) {
	if f != nil {
		f(event)
	}
}

// retryClock allows tests to control time without adding arbitrary sleeps.
//
// Production code uses systemRetryClock.
type retryClock interface {
	Now() time.Time
	Sleep(context.Context, time.Duration) error
}

type systemRetryClock struct{}

func (systemRetryClock) Now() time.Time {
	return time.Now()
}

func (systemRetryClock) Sleep(
	ctx context.Context,
	delay time.Duration,
) error {
	if ctx == nil {
		return errors.New("sleep context is nil")
	}

	if delay <= 0 {
		if err := ctx.Err(); err != nil {
			return err
		}

		return nil
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()

	case <-timer.C:
		return nil
	}
}

// jitterSource supplies a value in [0, 1).
//
// Abstracting it lets tests use deterministic jitter.
type jitterSource interface {
	Float64() float64
}

type systemJitterSource struct{}

func (systemJitterSource) Float64() float64 {
	return rand.Float64()
}

// RetryExecutor applies one consistent retry policy to delivery operations.
//
// RetryExecutor contains immutable configuration and is safe for concurrent
// calls to Execute when its classifier and observer implementations are also
// safe for concurrent use.
type RetryExecutor struct {
	config RetryConfig

	classifier RetryClassifier
	observer   RetryObserver

	clock  retryClock
	jitter jitterSource
}

// NewRetryExecutor creates a production retry executor.
func NewRetryExecutor(
	config RetryConfig,
	classifier RetryClassifier,
	observer RetryObserver,
) (*RetryExecutor, error) {
	return newRetryExecutor(
		config,
		classifier,
		observer,
		systemRetryClock{},
		systemJitterSource{},
	)
}

func newRetryExecutor(
	config RetryConfig,
	classifier RetryClassifier,
	observer RetryObserver,
	clock retryClock,
	jitter jitterSource,
) (*RetryExecutor, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("create retry executor: %w", err)
	}

	if classifier == nil {
		return nil, errors.New(
			"create retry executor: retry classifier is required",
		)
	}

	if clock == nil {
		return nil, errors.New(
			"create retry executor: retry clock is required",
		)
	}

	if jitter == nil {
		return nil, errors.New(
			"create retry executor: jitter source is required",
		)
	}

	return &RetryExecutor{
		config:     config,
		classifier: classifier,
		observer:   observer,
		clock:      clock,
		jitter:     jitter,
	}, nil
}

// Execute runs an operation using the configured retry policy.
//
// Execute returns immediately on:
//
//   - success;
//   - context cancellation;
//   - a non-retryable error;
//   - maximum attempts;
//   - maximum elapsed retry time.
//
// RetryExecutor never mutates DeliveryResult. Exporters remain responsible
// for constructing the attempt's delivery result.
func (r *RetryExecutor) Execute(
	ctx context.Context,
	operation RetryOperation,
) (DeliveryResult, error) {
	if r == nil {
		return DeliveryResult{}, errors.New(
			"retry executor is nil",
		)
	}

	if ctx == nil {
		return DeliveryResult{}, errors.New(
			"retry context is nil",
		)
	}

	if operation == nil {
		return DeliveryResult{}, errors.New(
			"retry operation is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return DeliveryResult{}, err
	}

	startedAt := r.clock.Now()

	var (
		lastResult DeliveryResult
		lastErr    error
	)

	for attempt := 1; attempt <= r.config.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return lastResult, err
		}

		if r.elapsedLimitReached(startedAt) {
			if lastErr == nil {
				lastErr = errors.New(
					"maximum retry elapsed time reached before attempt",
				)
			}

			return lastResult, fmt.Errorf(
				"%w after %d attempts: %w",
				ErrRetryExhausted,
				attempt-1,
				lastErr,
			)
		}

		result, err := operation(ctx, attempt)
		lastResult = result

		if err == nil {
			return result, nil
		}

		lastErr = err

		decision := r.classifier.ClassifyRetry(err)

		if !decision.Retry {
			return result, fmt.Errorf(
				"%w: attempt %d: %w",
				ErrRetryNotAllowed,
				attempt,
				err,
			)
		}

		if attempt >= r.config.MaxAttempts {
			return result, fmt.Errorf(
				"%w after %d attempts: %w",
				ErrRetryExhausted,
				attempt,
				err,
			)
		}

		delay := r.delayForRetry(attempt)

		if r.config.MaxElapsedTime > 0 {
			elapsed := r.clock.Now().Sub(startedAt)

			if elapsed < 0 {
				elapsed = 0
			}

			remaining := r.config.MaxElapsedTime - elapsed

			if remaining <= 0 {
				return result, fmt.Errorf(
					"%w after %d attempts: %w",
					ErrRetryExhausted,
					attempt,
					err,
				)
			}

			if delay > remaining {
				delay = remaining
			}
		}

		if r.observer != nil {
			elapsed := r.clock.Now().Sub(startedAt)

			if elapsed < 0 {
				elapsed = 0
			}

			r.observer.OnRetry(RetryEvent{
				Attempt:     attempt,
				NextAttempt: attempt + 1,
				Delay:       delay,
				Error:       err,
				Decision:    decision,
				Elapsed:     elapsed,
			})
		}

		if err := r.clock.Sleep(ctx, delay); err != nil {
			return result, err
		}
	}

	// The loop is expected to return before reaching this point.
	// Keep a defensive fallback so changing retry limits later cannot cause
	// a silent successful return.
	if lastErr == nil {
		lastErr = errors.New("retry completed without result")
	}

	return lastResult, fmt.Errorf(
		"%w: %w",
		ErrRetryExhausted,
		lastErr,
	)
}

// Delay returns the retry delay after the specified failed attempt.
//
// attempt must be greater than zero. Attempt 1 means the delay that occurs
// before attempt 2.
func (r *RetryExecutor) Delay(attempt int) (time.Duration, error) {
	if r == nil {
		return 0, errors.New("retry executor is nil")
	}

	if attempt <= 0 {
		return 0, errors.New(
			"retry attempt must be greater than zero",
		)
	}

	return r.delayForRetry(attempt), nil
}

// Config returns a copy of the executor's retry configuration.
func (r *RetryExecutor) Config() RetryConfig {
	if r == nil {
		return RetryConfig{}
	}

	return r.config
}

func (r *RetryExecutor) elapsedLimitReached(
	startedAt time.Time,
) bool {
	if r.config.MaxElapsedTime <= 0 {
		return false
	}

	elapsed := r.clock.Now().Sub(startedAt)

	if elapsed < 0 {
		return false
	}

	return elapsed >= r.config.MaxElapsedTime
}

// delayForRetry calculates exponential backoff with proportional jitter.
//
// failedAttempt=1 calculates the delay between attempts 1 and 2.
func (r *RetryExecutor) delayForRetry(
	failedAttempt int,
) time.Duration {
	exponent := float64(failedAttempt - 1)

	baseFloat := float64(r.config.InitialDelay) *
		math.Pow(r.config.Multiplier, exponent)

	maxDelayFloat := float64(r.config.MaxDelay)

	if math.IsInf(baseFloat, 1) ||
		baseFloat > maxDelayFloat {
		baseFloat = maxDelayFloat
	}

	if baseFloat < 0 {
		baseFloat = maxDelayFloat
	}

	base := time.Duration(baseFloat)

	if base > r.config.MaxDelay {
		base = r.config.MaxDelay
	}

	if r.config.Jitter == 0 {
		return base
	}

	// Convert [0,1) into [-1,1), then scale by the configured jitter.
	randomUnit := (r.jitter.Float64() * 2) - 1

	factor := 1 + (randomUnit * r.config.Jitter)

	jitteredFloat := float64(base) * factor

	if jitteredFloat < 0 {
		jitteredFloat = 0
	}

	// MaxDelay is treated as a hard upper bound even after positive jitter.
	if jitteredFloat > maxDelayFloat {
		jitteredFloat = maxDelayFloat
	}

	return time.Duration(jitteredFloat)
}
