package export

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Common batching errors.
var (
	ErrBatcherClosed    = errors.New("batcher closed")
	ErrBatchFull        = errors.New("batch is full")
	ErrRecordTooLarge   = errors.New("record exceeds maximum batch payload size")
	ErrBatchUnavailable = errors.New("batch unavailable")
)

// BatchIDGenerator generates unique batch identifiers.
//
// Keeping ID generation behind an interface allows the production runtime,
// tests, and future distributed deployments to choose an appropriate
// identifier strategy independently from the batching logic.
type BatchIDGenerator interface {
	NewBatchID() BatchID
}

// BatchIDGeneratorFunc adapts a function to BatchIDGenerator.
type BatchIDGeneratorFunc func() BatchID

// NewBatchID implements BatchIDGenerator.
func (f BatchIDGeneratorFunc) NewBatchID() BatchID {
	return f()
}

// SequentialBatchIDGenerator provides process-local batch identifiers.
//
// It is primarily useful as a safe default. Distributed deployments may
// provide a UUID, ULID, or another globally unique identifier generator.
type SequentialBatchIDGenerator struct {
	prefix  string
	counter atomic.Uint64
}

// NewSequentialBatchIDGenerator creates a process-local ID generator.
func NewSequentialBatchIDGenerator(prefix string) *SequentialBatchIDGenerator {
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		prefix = "batch"
	}

	return &SequentialBatchIDGenerator{
		prefix: prefix,
	}
}

// NewBatchID returns the next process-local batch identifier.
func (g *SequentialBatchIDGenerator) NewBatchID() BatchID {
	if g == nil {
		return ""
	}

	value := g.counter.Add(1)

	return BatchID(fmt.Sprintf("%s-%d", g.prefix, value))
}

// BatcherConfig controls how records are grouped before delivery.
type BatcherConfig struct {
	// MaxRecords is the maximum number of records allowed in one batch.
	MaxRecords int

	// MaxPayloadBytes is the maximum combined payload size allowed in a
	// single batch. This counts Record.Payload bytes only.
	MaxPayloadBytes int64
}

// Validate verifies the batching configuration.
func (c BatcherConfig) Validate() error {
	if c.MaxRecords <= 0 {
		return errors.New("batcher max records must be greater than zero")
	}

	if c.MaxPayloadBytes <= 0 {
		return errors.New("batcher max payload bytes must be greater than zero")
	}

	return nil
}

// Batcher groups records for one destination.
//
// A Batcher instance owns exactly one destination. This prevents records
// destined for different systems from accidentally being mixed into the same
// delivery batch.
//
// Batcher is safe for concurrent use.
type Batcher struct {
	mu sync.Mutex

	config      BatcherConfig
	destination Destination
	idGenerator BatchIDGenerator

	pending      []Record
	payloadBytes int64

	closed bool
}

// NewBatcher creates a new batcher for one destination.
func NewBatcher(
	config BatcherConfig,
	destination Destination,
	idGenerator BatchIDGenerator,
) (*Batcher, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("create batcher: %w", err)
	}

	if err := destination.Validate(); err != nil {
		return nil, fmt.Errorf("create batcher destination: %w", err)
	}

	if idGenerator == nil {
		return nil, errors.New("create batcher: batch id generator is required")
	}

	return &Batcher{
		config:      config,
		destination: cloneDestination(destination),
		idGenerator: idGenerator,
		pending:     make([]Record, 0, config.MaxRecords),
	}, nil
}

// Add appends a record to the current batch.
//
// Add never silently flushes or discards records. If adding the record would
// exceed the configured batch limits, ErrBatchFull is returned and the
// existing pending records remain untouched.
//
// The caller can Flush the current batch and retry Add with the record.
func (b *Batcher) Add(record Record) error {
	if b == nil {
		return errors.New("batcher is nil")
	}

	if err := record.Validate(); err != nil {
		return fmt.Errorf("add record: %w", err)
	}

	recordSize := int64(record.Size())

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrBatcherClosed
	}

	if recordSize > b.config.MaxPayloadBytes {
		return fmt.Errorf(
			"%w: record %q contains %d payload bytes, maximum is %d",
			ErrRecordTooLarge,
			record.ID,
			recordSize,
			b.config.MaxPayloadBytes,
		)
	}

	if len(b.pending) >= b.config.MaxRecords {
		return ErrBatchFull
	}

	if recordSize > b.config.MaxPayloadBytes-b.payloadBytes {
		return ErrBatchFull
	}

	b.pending = append(b.pending, cloneRecord(record))
	b.payloadBytes += recordSize

	return nil
}

// CanAdd reports whether record can be added without exceeding configured
// limits.
//
// The result is only a snapshot. Another goroutine can modify the batch before
// a subsequent Add call, so callers must still handle Add errors.
func (b *Batcher) CanAdd(record Record) bool {
	if b == nil {
		return false
	}

	if err := record.Validate(); err != nil {
		return false
	}

	recordSize := int64(record.Size())

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return false
	}

	if recordSize > b.config.MaxPayloadBytes {
		return false
	}

	if len(b.pending) >= b.config.MaxRecords {
		return false
	}

	return recordSize <= b.config.MaxPayloadBytes-b.payloadBytes
}

// Full reports whether the current batch has reached either configured limit.
func (b *Batcher) Full() bool {
	if b == nil {
		return false
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return false
	}

	return len(b.pending) >= b.config.MaxRecords ||
		b.payloadBytes >= b.config.MaxPayloadBytes
}

// Empty reports whether no records are currently pending.
func (b *Batcher) Empty() bool {
	return b.Len() == 0
}

// Len returns the number of records currently pending.
func (b *Batcher) Len() int {
	if b == nil {
		return 0
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	return len(b.pending)
}

// PayloadBytes returns the current combined unframed payload size.
func (b *Batcher) PayloadBytes() int64 {
	if b == nil {
		return 0
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	return b.payloadBytes
}

// RemainingRecords returns the number of additional records that can fit based
// solely on the configured record-count limit.
func (b *Batcher) RemainingRecords() int {
	if b == nil {
		return 0
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	remaining := b.config.MaxRecords - len(b.pending)
	if remaining < 0 {
		return 0
	}

	return remaining
}

// RemainingPayloadBytes returns the remaining payload capacity.
func (b *Batcher) RemainingPayloadBytes() int64 {
	if b == nil {
		return 0
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	remaining := b.config.MaxPayloadBytes - b.payloadBytes
	if remaining < 0 {
		return 0
	}

	return remaining
}

// Flush removes all currently pending records and returns them as a validated
// delivery Batch.
//
// Flush returns ErrBatchUnavailable when no records are pending.
//
// Records are copied into the returned batch so later batcher operations do
// not mutate the flushed batch.
//
// If batch construction unexpectedly fails validation, the pending records
// remain in the batcher.
func (b *Batcher) Flush(now time.Time) (Batch, error) {
	if b == nil {
		return Batch{}, errors.New("batcher is nil")
	}

	if now.IsZero() {
		return Batch{}, errors.New("flush time is required")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return Batch{}, ErrBatcherClosed
	}

	if len(b.pending) == 0 {
		return Batch{}, ErrBatchUnavailable
	}

	batchID := b.idGenerator.NewBatchID()
	if strings.TrimSpace(string(batchID)) == "" {
		return Batch{}, errors.New("batch id generator returned an empty id")
	}

	records := make([]Record, len(b.pending))
	for i := range b.pending {
		records[i] = cloneRecord(b.pending[i])
	}

	batch := Batch{
		ID:          batchID,
		Destination: cloneDestination(b.destination),
		Records:     records,
		CreatedAt:   now,
	}

	if err := batch.Validate(); err != nil {
		return Batch{}, fmt.Errorf("create batch: %w", err)
	}

	b.pending = b.pending[:0]
	b.payloadBytes = 0

	return batch, nil
}

// Snapshot returns a defensive copy of the records currently waiting in the
// batcher. It is intended for diagnostics and tests.
//
// Snapshot does not remove records.
func (b *Batcher) Snapshot() []Record {
	if b == nil {
		return nil
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	records := make([]Record, len(b.pending))

	for i := range b.pending {
		records[i] = cloneRecord(b.pending[i])
	}

	return records
}

// Destination returns a defensive copy of this batcher's destination.
func (b *Batcher) Destination() Destination {
	if b == nil {
		return Destination{}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	return cloneDestination(b.destination)
}

// Config returns this batcher's immutable batching configuration.
func (b *Batcher) Config() BatcherConfig {
	if b == nil {
		return BatcherConfig{}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	return b.config
}

// Close prevents future Add and Flush operations.
//
// Existing pending records are intentionally preserved in memory so the caller
// can inspect the batcher state before discarding the instance. Production
// shutdown should normally Flush pending records before calling Close.
func (b *Batcher) Close() {
	if b == nil {
		return
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed = true
}

// Closed reports whether the batcher has been closed.
func (b *Batcher) Closed() bool {
	if b == nil {
		return true
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	return b.closed
}

// Reset discards all pending records.
//
// This operation should be used carefully because it intentionally removes
// records that have not been exported.
func (b *Batcher) Reset() error {
	if b == nil {
		return errors.New("batcher is nil")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrBatcherClosed
	}

	b.pending = b.pending[:0]
	b.payloadBytes = 0

	return nil
}

// cloneRecord creates a defensive copy of a Record including mutable fields.
//
// Payload, metadata, resource labels, and resource identity are copied so
// callers cannot change records after they have entered a batch.
func cloneRecord(record Record) Record {
	cloned := record

	if record.Payload != nil {
		cloned.Payload = append([]byte(nil), record.Payload...)
	}

	if record.Metadata != nil {
		cloned.Metadata = make(map[string]string, len(record.Metadata))

		for key, value := range record.Metadata {
			cloned.Metadata[key] = value
		}
	}

	if record.Resource != nil {
		resource := *record.Resource

		if record.Resource.Labels != nil {
			resource.Labels = make(
				map[string]string,
				len(record.Resource.Labels),
			)

			for key, value := range record.Resource.Labels {
				resource.Labels[key] = value
			}
		}

		cloned.Resource = &resource
	}

	return cloned
}
