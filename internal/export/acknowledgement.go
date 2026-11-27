
package export

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

var (
	// ErrAcknowledgementNotFound indicates that no acknowledgement state is
	// registered for the requested batch.
	ErrAcknowledgementNotFound = errors.New(
		"acknowledgement not found",
	)

	// ErrAcknowledgementExists indicates that tracking has already been
	// started for the batch.
	ErrAcknowledgementExists = errors.New(
		"acknowledgement already exists",
	)

	// ErrAcknowledgementConflict indicates that acknowledgement data does not
	// match the tracked batch.
	ErrAcknowledgementConflict = errors.New(
		"acknowledgement conflict",
	)

	// ErrAcknowledgementClosed indicates that the manager has been closed.
	ErrAcknowledgementClosed = errors.New(
		"acknowledgement manager closed",
	)
)

// AcknowledgementState represents OTGH's current understanding of a batch's
// delivery outcome.
type AcknowledgementState uint8

const (
	AcknowledgementPending AcknowledgementState = iota
	AcknowledgementInFlight
	AcknowledgementPartial
	AcknowledgementAcknowledged
	AcknowledgementRejected
	AcknowledgementRetrying
	AcknowledgementFailed
)

// String returns a stable textual representation.
func (s AcknowledgementState) String() string {
	switch s {
	case AcknowledgementPending:
		return "pending"

	case AcknowledgementInFlight:
		return "in_flight"

	case AcknowledgementPartial:
		return "partial"

	case AcknowledgementAcknowledged:
		return "acknowledged"

	case AcknowledgementRejected:
		return "rejected"

	case AcknowledgementRetrying:
		return "retrying"

	case AcknowledgementFailed:
		return "failed"

	default:
		return "unknown"
	}
}

// Valid reports whether the state is recognised.
func (s AcknowledgementState) Valid() bool {
	return s <= AcknowledgementFailed
}

// Terminal reports whether no further delivery attempt is normally expected.
//
// Acknowledged means successful completion.
// Rejected and Failed require deliberate higher-level handling rather than
// remaining active delivery states.
func (s AcknowledgementState) Terminal() bool {
	switch s {
	case AcknowledgementAcknowledged,
		AcknowledgementRejected,
		AcknowledgementFailed:
		return true

	default:
		return false
	}
}

// Acknowledgement represents delivery state for one batch.
//
// It is intentionally transport-neutral. HTTP, gRPC, object storage,
// webhooks, and OTGH-native transports all translate destination responses
// into this common representation.
type Acknowledgement struct {
	BatchID       BatchID
	DestinationID DestinationID

	State AcknowledgementState

	TotalRecords    int
	AcceptedRecords int
	RejectedRecords int

	Attempt int

	Retryable bool

	ErrorCode    string
	ErrorMessage string

	CreatedAt time.Time
	UpdatedAt time.Time

	AcknowledgedAt time.Time
}

// Validate verifies acknowledgement consistency.
func (a Acknowledgement) Validate() error {
	if strings.TrimSpace(string(a.BatchID)) == "" {
		return errors.New("acknowledgement batch id is required")
	}

	if strings.TrimSpace(string(a.DestinationID)) == "" {
		return errors.New(
			"acknowledgement destination id is required",
		)
	}

	if !a.State.Valid() {
		return fmt.Errorf(
			"invalid acknowledgement state %d",
			a.State,
		)
	}

	if a.TotalRecords <= 0 {
		return errors.New(
			"acknowledgement total records must be greater than zero",
		)
	}

	if a.AcceptedRecords < 0 {
		return errors.New(
			"acknowledgement accepted records cannot be negative",
		)
	}

	if a.RejectedRecords < 0 {
		return errors.New(
			"acknowledgement rejected records cannot be negative",
		)
	}

	if a.AcceptedRecords+a.RejectedRecords > a.TotalRecords {
		return fmt.Errorf(
			"acknowledgement accounts for %d records but batch contains %d",
			a.AcceptedRecords+a.RejectedRecords,
			a.TotalRecords,
		)
	}

	if a.Attempt < 0 {
		return errors.New(
			"acknowledgement attempt cannot be negative",
		)
	}

	if a.CreatedAt.IsZero() {
		return errors.New(
			"acknowledgement created_at is required",
		)
	}

	if a.UpdatedAt.IsZero() {
		return errors.New(
			"acknowledgement updated_at is required",
		)
	}

	if a.UpdatedAt.Before(a.CreatedAt) {
		return errors.New(
			"acknowledgement updated_at precedes created_at",
		)
	}

	if !a.AcknowledgedAt.IsZero() &&
		a.AcknowledgedAt.Before(a.CreatedAt) {
		return errors.New(
			"acknowledged_at precedes created_at",
		)
	}

	if a.State == AcknowledgementAcknowledged {
		if a.AcceptedRecords != a.TotalRecords {
			return errors.New(
				"fully acknowledged batch must accept every record",
			)
		}

		if a.RejectedRecords != 0 {
			return errors.New(
				"fully acknowledged batch cannot contain rejected records",
			)
		}

		if a.AcknowledgedAt.IsZero() {
			return errors.New(
				"fully acknowledged batch requires acknowledged_at",
			)
		}

		if a.Retryable {
			return errors.New(
				"fully acknowledged batch cannot be retryable",
			)
		}
	}

	return nil
}

// Complete reports whether every record has been accepted.
func (a Acknowledgement) Complete() bool {
	return a.State == AcknowledgementAcknowledged &&
		a.AcceptedRecords == a.TotalRecords &&
		a.RejectedRecords == 0
}

// OutstandingRecords returns the number of records not yet accepted or
// rejected by the destination.
func (a Acknowledgement) OutstandingRecords() int {
	outstanding := a.TotalRecords -
		a.AcceptedRecords -
		a.RejectedRecords

	if outstanding < 0 {
		return 0
	}

	return outstanding
}

// AcknowledgementEvent describes a state transition.
//
// This can later be consumed by OTGH metrics or structured logging without
// coupling the manager directly to either subsystem.
type AcknowledgementEvent struct {
	Previous AcknowledgementState
	Current  AcknowledgementState

	Acknowledgement Acknowledgement
}

// AcknowledgementObserver receives acknowledgement state changes.
type AcknowledgementObserver interface {
	OnAcknowledgement(AcknowledgementEvent)
}

// AcknowledgementObserverFunc adapts a function to the observer interface.
type AcknowledgementObserverFunc func(AcknowledgementEvent)

// OnAcknowledgement implements AcknowledgementObserver.
func (f AcknowledgementObserverFunc) OnAcknowledgement(
	event AcknowledgementEvent,
) {
	if f != nil {
		f(event)
	}
}

// AcknowledgementManager tracks delivery acknowledgement state.
//
// State is process-local. Durable batch recovery remains the responsibility of
// Spool. After restart, worker.go can reconstruct acknowledgement tracking
// from recovered batches before attempting delivery again.
//
// The manager is safe for concurrent use.
type AcknowledgementManager struct {
	mu sync.RWMutex

	entries map[BatchID]Acknowledgement

	observer AcknowledgementObserver

	closed bool
}

// NewAcknowledgementManager creates an empty manager.
func NewAcknowledgementManager(
	observer AcknowledgementObserver,
) *AcknowledgementManager {
	return &AcknowledgementManager{
		entries:  make(map[BatchID]Acknowledgement),
		observer: observer,
	}
}

// Track starts acknowledgement tracking for a batch.
func (m *AcknowledgementManager) Track(
	ctx context.Context,
	batch Batch,
) (Acknowledgement, error) {
	if m == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement manager is nil",
		)
	}

	if ctx == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return Acknowledgement{}, err
	}

	if err := batch.Validate(); err != nil {
		return Acknowledgement{}, fmt.Errorf(
			"track acknowledgement: %w",
			err,
		)
	}

	now := time.Now().UTC()

	ack := Acknowledgement{
		BatchID:       batch.ID,
		DestinationID: batch.Destination.ID,
		State:         AcknowledgementPending,
		TotalRecords:  batch.RecordCount(),
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	if err := ack.Validate(); err != nil {
		return Acknowledgement{}, fmt.Errorf(
			"create acknowledgement: %w",
			err,
		)
	}

	m.mu.Lock()

	if m.closed {
		m.mu.Unlock()
		return Acknowledgement{}, ErrAcknowledgementClosed
	}

	if _, exists := m.entries[batch.ID]; exists {
		m.mu.Unlock()

		return Acknowledgement{}, fmt.Errorf(
			"%w: batch %q",
			ErrAcknowledgementExists,
			batch.ID,
		)
	}

	m.entries[batch.ID] = ack

	observer := m.observer

	m.mu.Unlock()

	m.notify(observer, AcknowledgementEvent{
		Previous:        AcknowledgementPending,
		Current:         AcknowledgementPending,
		Acknowledgement: ack,
	})

	return ack, nil
}

// BeginAttempt marks a batch as currently being delivered.
func (m *AcknowledgementManager) BeginAttempt(
	ctx context.Context,
	batchID BatchID,
	attempt int,
) (Acknowledgement, error) {
	if attempt <= 0 {
		return Acknowledgement{}, errors.New(
			"delivery attempt must be greater than zero",
		)
	}

	return m.update(
		ctx,
		batchID,
		func(ack *Acknowledgement) error {
			if ack.State == AcknowledgementAcknowledged {
				return errors.New(
					"cannot begin attempt for acknowledged batch",
				)
			}

			if ack.State == AcknowledgementRejected {
				return errors.New(
					"cannot begin attempt for rejected batch",
				)
			}

			if ack.State == AcknowledgementFailed {
				return errors.New(
					"cannot begin attempt for failed batch",
				)
			}

			if attempt < ack.Attempt {
				return fmt.Errorf(
					"attempt %d is older than current attempt %d",
					attempt,
					ack.Attempt,
				)
			}

			ack.Attempt = attempt
			ack.State = AcknowledgementInFlight
			ack.Retryable = false
			ack.ErrorCode = ""
			ack.ErrorMessage = ""

			return nil
		},
	)
}

// ApplyResult updates acknowledgement state from one exporter delivery result.
func (m *AcknowledgementManager) ApplyResult(
	ctx context.Context,
	result DeliveryResult,
) (Acknowledgement, error) {
	if strings.TrimSpace(string(result.BatchID)) == "" {
		return Acknowledgement{}, errors.New(
			"delivery result batch id is required",
		)
	}

	if strings.TrimSpace(string(result.DestinationID)) == "" {
		return Acknowledgement{}, errors.New(
			"delivery result destination id is required",
		)
	}

	if !result.Status.Valid() {
		return Acknowledgement{}, fmt.Errorf(
			"delivery result has invalid status %d",
			result.Status,
		)
	}

	return m.update(
		ctx,
		result.BatchID,
		func(ack *Acknowledgement) error {
			if ack.DestinationID != result.DestinationID {
				return fmt.Errorf(
					"%w: result destination %q does not match tracked destination %q",
					ErrAcknowledgementConflict,
					result.DestinationID,
					ack.DestinationID,
				)
			}

			if result.Attempt < 0 {
				return errors.New(
					"delivery attempt cannot be negative",
				)
			}

			if result.Attempt < ack.Attempt {
				return fmt.Errorf(
					"%w: result attempt %d is older than tracked attempt %d",
					ErrAcknowledgementConflict,
					result.Attempt,
					ack.Attempt,
				)
			}

			if result.AcceptedRecords < 0 ||
				result.RejectedRecords < 0 {
				return errors.New(
					"delivery record counts cannot be negative",
				)
			}

			accounted := result.AcceptedRecords +
				result.RejectedRecords

			if accounted > ack.TotalRecords {
				return fmt.Errorf(
					"%w: result accounts for %d records but batch contains %d",
					ErrAcknowledgementConflict,
					accounted,
					ack.TotalRecords,
				)
			}

			ack.Attempt = result.Attempt
			ack.AcceptedRecords = result.AcceptedRecords
			ack.RejectedRecords = result.RejectedRecords
			ack.Retryable = result.Retryable
			ack.ErrorCode = result.ErrorCode
			ack.ErrorMessage = result.ErrorMessage
			ack.AcknowledgedAt = time.Time{}

			switch result.Status {
			case DeliveryPending:
				ack.State = AcknowledgementPending

			case DeliveryInFlight:
				ack.State = AcknowledgementInFlight

			case DeliveryRetrying:
				ack.State = AcknowledgementRetrying

			case DeliveryFailed:
				ack.State = AcknowledgementFailed

			case DeliveryDeadLettered:
				ack.State = AcknowledgementRejected
				ack.Retryable = false

			case DeliveryAcknowledged:
				if result.AcceptedRecords == ack.TotalRecords &&
					result.RejectedRecords == 0 {
					ack.State = AcknowledgementAcknowledged
					ack.Retryable = false
					ack.AcknowledgedAt = time.Now().UTC()
				} else {
					ack.State = AcknowledgementPartial
				}

			default:
				return fmt.Errorf(
					"unsupported delivery status %d",
					result.Status,
				)
			}

			return nil
		},
	)
}

// MarkRetrying marks a tracked batch as waiting for another attempt.
func (m *AcknowledgementManager) MarkRetrying(
	ctx context.Context,
	batchID BatchID,
	errCode string,
	errMessage string,
) (Acknowledgement, error) {
	return m.update(
		ctx,
		batchID,
		func(ack *Acknowledgement) error {
			if ack.State.Terminal() {
				return fmt.Errorf(
					"cannot retry terminal acknowledgement state %q",
					ack.State,
				)
			}

			ack.State = AcknowledgementRetrying
			ack.Retryable = true
			ack.ErrorCode = strings.TrimSpace(errCode)
			ack.ErrorMessage = strings.TrimSpace(errMessage)

			return nil
		},
	)
}

// MarkFailed marks a batch as permanently failed.
func (m *AcknowledgementManager) MarkFailed(
	ctx context.Context,
	batchID BatchID,
	errCode string,
	errMessage string,
) (Acknowledgement, error) {
	return m.update(
		ctx,
		batchID,
		func(ack *Acknowledgement) error {
			if ack.State == AcknowledgementAcknowledged {
				return errors.New(
					"cannot fail an acknowledged batch",
				)
			}

			ack.State = AcknowledgementFailed
			ack.Retryable = false
			ack.ErrorCode = strings.TrimSpace(errCode)
			ack.ErrorMessage = strings.TrimSpace(errMessage)

			return nil
		},
	)
}

// Get returns a copy of the acknowledgement associated with batchID.
func (m *AcknowledgementManager) Get(
	ctx context.Context,
	batchID BatchID,
) (Acknowledgement, error) {
	if m == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement manager is nil",
		)
	}

	if ctx == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return Acknowledgement{}, err
	}

	if strings.TrimSpace(string(batchID)) == "" {
		return Acknowledgement{}, errors.New(
			"acknowledgement batch id is required",
		)
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return Acknowledgement{}, ErrAcknowledgementClosed
	}

	ack, exists := m.entries[batchID]
	if !exists {
		return Acknowledgement{}, fmt.Errorf(
			"%w: batch %q",
			ErrAcknowledgementNotFound,
			batchID,
		)
	}

	return ack, nil
}

// Remove removes acknowledgement state after the caller no longer requires
// process-local tracking.
//
// In the normal success path worker.go will:
//
//   1. receive full destination acknowledgement;
//   2. delete the durable spool entry;
//   3. remove acknowledgement tracking.
func (m *AcknowledgementManager) Remove(
	ctx context.Context,
	batchID BatchID,
) (Acknowledgement, error) {
	if m == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement manager is nil",
		)
	}

	if ctx == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return Acknowledgement{}, err
	}

	if strings.TrimSpace(string(batchID)) == "" {
		return Acknowledgement{}, errors.New(
			"acknowledgement batch id is required",
		)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return Acknowledgement{}, ErrAcknowledgementClosed
	}

	ack, exists := m.entries[batchID]
	if !exists {
		return Acknowledgement{}, fmt.Errorf(
			"%w: batch %q",
			ErrAcknowledgementNotFound,
			batchID,
		)
	}

	delete(m.entries, batchID)

	return ack, nil
}

// List returns a snapshot of all tracked acknowledgements.
//
// No ordering guarantee is provided.
func (m *AcknowledgementManager) List(
	ctx context.Context,
) ([]Acknowledgement, error) {
	if m == nil {
		return nil, errors.New(
			"acknowledgement manager is nil",
		)
	}

	if ctx == nil {
		return nil, errors.New(
			"acknowledgement context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, ErrAcknowledgementClosed
	}

	result := make(
		[]Acknowledgement,
		0,
		len(m.entries),
	)

	for _, ack := range m.entries {
		result = append(result, ack)
	}

	return result, nil
}

// Count returns the number of tracked batches.
func (m *AcknowledgementManager) Count() int {
	if m == nil {
		return 0
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.entries)
}

// Close prevents future state-changing and retrieval operations.
//
// Existing entries remain in memory until the manager becomes unreachable.
// Durable pending batches remain protected by Spool.
func (m *AcknowledgementManager) Close() {
	if m == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.closed = true
}

// Closed reports whether the manager has been closed.
func (m *AcknowledgementManager) Closed() bool {
	if m == nil {
		return true
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.closed
}

// update performs one validated acknowledgement state transition.
//
// Observer notification occurs after releasing the mutex so observer code
// cannot deadlock acknowledgement tracking.
func (m *AcknowledgementManager) update(
	ctx context.Context,
	batchID BatchID,
	mutate func(*Acknowledgement) error,
) (Acknowledgement, error) {
	if m == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement manager is nil",
		)
	}

	if ctx == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return Acknowledgement{}, err
	}

	if strings.TrimSpace(string(batchID)) == "" {
		return Acknowledgement{}, errors.New(
			"acknowledgement batch id is required",
		)
	}

	if mutate == nil {
		return Acknowledgement{}, errors.New(
			"acknowledgement mutation is nil",
		)
	}

	m.mu.Lock()

	if m.closed {
		m.mu.Unlock()

		return Acknowledgement{},
			ErrAcknowledgementClosed
	}

	current, exists := m.entries[batchID]
	if !exists {
		m.mu.Unlock()

		return Acknowledgement{}, fmt.Errorf(
			"%w: batch %q",
			ErrAcknowledgementNotFound,
			batchID,
		)
	}

	previousState := current.State

	next := current

	if err := mutate(&next); err != nil {
		m.mu.Unlock()
		return Acknowledgement{}, err
	}

	if err := ctx.Err(); err != nil {
		m.mu.Unlock()
		return Acknowledgement{}, err
	}

	next.UpdatedAt = time.Now().UTC()

	if err := next.Validate(); err != nil {
		m.mu.Unlock()

		return Acknowledgement{}, fmt.Errorf(
			"invalid acknowledgement transition: %w",
			err,
		)
	}

	m.entries[batchID] = next

	observer := m.observer

	m.mu.Unlock()

	m.notify(observer, AcknowledgementEvent{
		Previous:        previousState,
		Current:         next.State,
		Acknowledgement: next,
	})

	return next, nil
}

func (m *AcknowledgementManager) notify(
	observer AcknowledgementObserver,
	event AcknowledgementEvent,
) {
	if observer == nil {
		return
	}

	observer.OnAcknowledgement(event)
}
