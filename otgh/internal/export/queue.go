
package export

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

var (
	// ErrQueueClosed indicates that the queue no longer accepts new work.
	ErrQueueClosed = errors.New("queue closed")

	// ErrQueueFull indicates that a non-blocking enqueue could not proceed
	// because the queue has reached its configured capacity.
	ErrQueueFull = errors.New("queue full")

	// ErrQueueEmpty indicates that a non-blocking dequeue found no work.
	ErrQueueEmpty = errors.New("queue empty")
)

// QueueConfig controls the in-memory delivery queue.
type QueueConfig struct {
	// Capacity is the maximum number of batches that may wait in memory.
	//
	// Capacity must be greater than zero.
	Capacity int
}

// Validate verifies the queue configuration.
func (c QueueConfig) Validate() error {
	if c.Capacity <= 0 {
		return errors.New("queue capacity must be greater than zero")
	}

	return nil
}

// QueueStats is a point-in-time view of queue state.
type QueueStats struct {
	Capacity int
	Depth    int

	Enqueued uint64
	Dequeued uint64

	Closed bool
}

// Queue is a bounded, concurrency-safe FIFO queue for delivery batches.
//
// Queue intentionally stores Batch rather than Record. Records are grouped by
// Batcher before entering the delivery queue.
//
// The queue supports both:
//
//   - blocking context-aware operations for workers and producers;
//   - non-blocking operations for callers that prefer explicit backpressure.
//
// Queue does not persist data to disk. spool.go will provide durable overflow
// and restart recovery independently from this in-memory component.
type Queue struct {
	mu sync.Mutex

	capacity int
	items    []Batch

	closed bool

	enqueued uint64
	dequeued uint64

	// notEmpty is replaced and closed whenever the queue changes from empty
	// to containing data, waking blocked dequeue operations.
	notEmpty chan struct{}

	// notFull is replaced and closed whenever queue capacity becomes
	// available, waking blocked enqueue operations.
	notFull chan struct{}
}

// NewQueue creates an empty bounded queue.
func NewQueue(config QueueConfig) (*Queue, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("create queue: %w", err)
	}

	return &Queue{
		capacity: config.Capacity,
		items:    make([]Batch, 0, config.Capacity),
		notEmpty: make(chan struct{}),
		notFull:  make(chan struct{}),
	}, nil
}

// Enqueue waits until the batch can be added, the context is cancelled, or
// the queue is closed.
//
// A defensive copy is stored so callers cannot mutate a batch after it has
// entered the queue.
func (q *Queue) Enqueue(ctx context.Context, batch Batch) error {
	if q == nil {
		return errors.New("queue is nil")
	}

	if ctx == nil {
		return errors.New("enqueue context is nil")
	}

	if err := batch.Validate(); err != nil {
		return fmt.Errorf("enqueue batch: %w", err)
	}

	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		q.mu.Lock()

		if q.closed {
			q.mu.Unlock()
			return ErrQueueClosed
		}

		if len(q.items) < q.capacity {
			wasEmpty := len(q.items) == 0

			q.items = append(q.items, cloneBatch(batch))
			q.enqueued++

			if wasEmpty {
				close(q.notEmpty)
				q.notEmpty = make(chan struct{})
			}

			q.mu.Unlock()

			return nil
		}

		wait := q.notFull

		q.mu.Unlock()

		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-wait:
			// Capacity may now be available. Loop and re-check under lock.
		}
	}
}

// TryEnqueue attempts to add a batch without blocking.
func (q *Queue) TryEnqueue(batch Batch) error {
	if q == nil {
		return errors.New("queue is nil")
	}

	if err := batch.Validate(); err != nil {
		return fmt.Errorf("enqueue batch: %w", err)
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	if q.closed {
		return ErrQueueClosed
	}

	if len(q.items) >= q.capacity {
		return ErrQueueFull
	}

	wasEmpty := len(q.items) == 0

	q.items = append(q.items, cloneBatch(batch))
	q.enqueued++

	if wasEmpty {
		close(q.notEmpty)
		q.notEmpty = make(chan struct{})
	}

	return nil
}

// Dequeue waits until a batch becomes available, the context is cancelled,
// or the queue has been closed and drained.
//
// Closing a queue does not discard work already waiting inside it. Consumers
// may continue draining queued batches after Close.
func (q *Queue) Dequeue(ctx context.Context) (Batch, error) {
	if q == nil {
		return Batch{}, errors.New("queue is nil")
	}

	if ctx == nil {
		return Batch{}, errors.New("dequeue context is nil")
	}

	for {
		if err := ctx.Err(); err != nil {
			return Batch{}, err
		}

		q.mu.Lock()

		if len(q.items) > 0 {
			wasFull := len(q.items) == q.capacity

			batch := q.removeFirstLocked()

			if wasFull {
				close(q.notFull)
				q.notFull = make(chan struct{})
			}

			q.mu.Unlock()

			return batch, nil
		}

		if q.closed {
			q.mu.Unlock()
			return Batch{}, ErrQueueClosed
		}

		wait := q.notEmpty

		q.mu.Unlock()

		select {
		case <-ctx.Done():
			return Batch{}, ctx.Err()

		case <-wait:
			// Work may now exist, or the queue may have been closed.
			// Loop and re-check under lock.
		}
	}
}

// TryDequeue removes the oldest batch without blocking.
func (q *Queue) TryDequeue() (Batch, error) {
	if q == nil {
		return Batch{}, errors.New("queue is nil")
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.items) == 0 {
		if q.closed {
			return Batch{}, ErrQueueClosed
		}

		return Batch{}, ErrQueueEmpty
	}

	wasFull := len(q.items) == q.capacity

	batch := q.removeFirstLocked()

	if wasFull {
		close(q.notFull)
		q.notFull = make(chan struct{})
	}

	return batch, nil
}

// Peek returns a defensive copy of the oldest queued batch without removing
// it.
func (q *Queue) Peek() (Batch, error) {
	if q == nil {
		return Batch{}, errors.New("queue is nil")
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.items) == 0 {
		if q.closed {
			return Batch{}, ErrQueueClosed
		}

		return Batch{}, ErrQueueEmpty
	}

	return cloneBatch(q.items[0]), nil
}

// Len returns the number of batches currently waiting.
func (q *Queue) Len() int {
	if q == nil {
		return 0
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	return len(q.items)
}

// Capacity returns the configured maximum number of queued batches.
func (q *Queue) Capacity() int {
	if q == nil {
		return 0
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	return q.capacity
}

// Available returns the number of currently available queue slots.
func (q *Queue) Available() int {
	if q == nil {
		return 0
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	available := q.capacity - len(q.items)
	if available < 0 {
		return 0
	}

	return available
}

// Full reports whether the queue has reached capacity.
func (q *Queue) Full() bool {
	if q == nil {
		return false
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	return len(q.items) >= q.capacity
}

// Empty reports whether no batches remain in the queue.
func (q *Queue) Empty() bool {
	return q.Len() == 0
}

// Closed reports whether Close has been called.
func (q *Queue) Closed() bool {
	if q == nil {
		return true
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	return q.closed
}

// Stats returns a snapshot of queue statistics.
func (q *Queue) Stats() QueueStats {
	if q == nil {
		return QueueStats{}
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	return QueueStats{
		Capacity: q.capacity,
		Depth:    len(q.items),
		Enqueued: q.enqueued,
		Dequeued: q.dequeued,
		Closed:   q.closed,
	}
}

// Snapshot returns defensive copies of all currently queued batches in FIFO
// order. It is primarily intended for diagnostics, tests, and spool handoff.
func (q *Queue) Snapshot() []Batch {
	if q == nil {
		return nil
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	batches := make([]Batch, len(q.items))

	for i := range q.items {
		batches[i] = cloneBatch(q.items[i])
	}

	return batches
}

// Close prevents new batches from being enqueued.
//
// Existing batches remain available to consumers until the queue is drained.
// All blocked Enqueue and Dequeue callers are awakened so they can re-check
// queue state.
func (q *Queue) Close() {
	if q == nil {
		return
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	if q.closed {
		return
	}

	q.closed = true

	close(q.notEmpty)
	q.notEmpty = make(chan struct{})

	close(q.notFull)
	q.notFull = make(chan struct{})
}

// WaitUntilEmpty blocks until the queue is empty or ctx is cancelled.
//
// This is useful during graceful shutdown after producers have stopped.
//
// This method uses a small polling interval deliberately rather than adding
// another internal condition channel. Shutdown waiting is not on the hot path.
func (q *Queue) WaitUntilEmpty(
	ctx context.Context,
	pollInterval time.Duration,
) error {
	if q == nil {
		return errors.New("queue is nil")
	}

	if ctx == nil {
		return errors.New("wait context is nil")
	}

	if pollInterval <= 0 {
		return errors.New("poll interval must be greater than zero")
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		if q.Empty() {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-ticker.C:
		}
	}
}

// removeFirstLocked removes and copies the oldest item.
//
// q.mu must be held by the caller.
func (q *Queue) removeFirstLocked() Batch {
	batch := cloneBatch(q.items[0])

	// Clear the removed entry before slicing so references held by the batch
	// can be reclaimed by the garbage collector.
	q.items[0] = Batch{}
	q.items = q.items[1:]

	q.dequeued++

	// Periodically compact the backing slice when the queue has been drained
	// substantially. This prevents a long-lived queue from permanently
	// retaining a very large backing array after temporary traffic bursts.
	if len(q.items) == 0 {
		q.items = make([]Batch, 0, q.capacity)
	}

	return batch
}

// cloneBatch returns a deep defensive copy of a Batch.
//
// Destination metadata, records, payloads, record metadata, and resource
// labels are copied so producers cannot modify queued delivery work.
func cloneBatch(batch Batch) Batch {
	cloned := batch
	cloned.Destination = cloneDestination(batch.Destination)

	if batch.Records != nil {
		cloned.Records = make([]Record, len(batch.Records))

		for i := range batch.Records {
			cloned.Records[i] = cloneRecord(batch.Records[i])
		}
	}

	return cloned
}
