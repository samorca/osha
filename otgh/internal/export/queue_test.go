
package export

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testBatch(id string) Batch {
	now := time.Now().UTC()

	return Batch{
		ID: BatchID(id),
		Destination: Destination{
			ID:   DestinationID("test-destination"),
			Type: "test",
		},
		Records: []Record{
			{
				ID:          RecordID(id + "-record"),
				Type:        RecordTypeEvent,
				Priority:    PriorityNormal,
				ContentType: ContentTypeJSON,
				Compression: CompressionNone,
				CreatedAt:   now,
				Payload:     []byte(`{"test":true}`),
			},
		},
		CreatedAt: now,
	}
}

func newTestQueue(t *testing.T, capacity int) *Queue {
	t.Helper()

	queue, err := NewQueue(QueueConfig{
		Capacity: capacity,
	})
	if err != nil {
		t.Fatalf("NewQueue() error = %v", err)
	}

	return queue
}

func TestQueueConfigValidate(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		wantErr  bool
	}{
		{
			name:     "valid",
			capacity: 1,
			wantErr:  false,
		},
		{
			name:     "zero",
			capacity: 0,
			wantErr:  true,
		},
		{
			name:     "negative",
			capacity: -1,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := (QueueConfig{
				Capacity: tt.capacity,
			}).Validate()

			if tt.wantErr && err == nil {
				t.Fatal("Validate() expected error, got nil")
			}

			if !tt.wantErr && err != nil {
				t.Fatalf("Validate() unexpected error = %v", err)
			}
		})
	}
}

func TestNewQueueRejectsInvalidCapacity(t *testing.T) {
	_, err := NewQueue(QueueConfig{
		Capacity: 0,
	})

	if err == nil {
		t.Fatal("NewQueue() expected error, got nil")
	}
}

func TestQueueFIFO(t *testing.T) {
	queue := newTestQueue(t, 3)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	expected := []string{
		"batch-1",
		"batch-2",
		"batch-3",
	}

	for _, id := range expected {
		if err := queue.Enqueue(ctx, testBatch(id)); err != nil {
			t.Fatalf("Enqueue(%q) error = %v", id, err)
		}
	}

	for _, wantID := range expected {
		batch, err := queue.Dequeue(ctx)
		if err != nil {
			t.Fatalf("Dequeue() error = %v", err)
		}

		if got := string(batch.ID); got != wantID {
			t.Fatalf(
				"Dequeue() ID = %q, want %q",
				got,
				wantID,
			)
		}
	}

	if !queue.Empty() {
		t.Fatalf("queue should be empty, depth = %d", queue.Len())
	}
}

func TestTryEnqueueFull(t *testing.T) {
	queue := newTestQueue(t, 1)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() unexpected error = %v", err)
	}

	err := queue.TryEnqueue(testBatch("batch-2"))

	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf(
			"TryEnqueue() error = %v, want ErrQueueFull",
			err,
		)
	}
}

func TestTryDequeueEmpty(t *testing.T) {
	queue := newTestQueue(t, 1)

	_, err := queue.TryDequeue()

	if !errors.Is(err, ErrQueueEmpty) {
		t.Fatalf(
			"TryDequeue() error = %v, want ErrQueueEmpty",
			err,
		)
	}
}

func TestEnqueueBlocksUntilCapacityAvailable(t *testing.T) {
	queue := newTestQueue(t, 1)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := queue.Enqueue(ctx, testBatch("batch-1")); err != nil {
		t.Fatalf("initial Enqueue() error = %v", err)
	}

	result := make(chan error, 1)

	go func() {
		result <- queue.Enqueue(ctx, testBatch("batch-2"))
	}()

	select {
	case err := <-result:
		t.Fatalf(
			"second Enqueue() returned before capacity was available: %v",
			err,
		)

	case <-time.After(50 * time.Millisecond):
		// Expected: producer is blocked.
	}

	first, err := queue.Dequeue(ctx)
	if err != nil {
		t.Fatalf("Dequeue() error = %v", err)
	}

	if first.ID != BatchID("batch-1") {
		t.Fatalf(
			"Dequeue() ID = %q, want batch-1",
			first.ID,
		)
	}

	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("blocked Enqueue() error = %v", err)
		}

	case <-ctx.Done():
		t.Fatalf(
			"blocked Enqueue() did not wake after capacity became available: %v",
			ctx.Err(),
		)
	}

	second, err := queue.Dequeue(ctx)
	if err != nil {
		t.Fatalf("second Dequeue() error = %v", err)
	}

	if second.ID != BatchID("batch-2") {
		t.Fatalf(
			"second Dequeue() ID = %q, want batch-2",
			second.ID,
		)
	}
}

func TestDequeueBlocksUntilItemAvailable(t *testing.T) {
	queue := newTestQueue(t, 1)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	type result struct {
		batch Batch
		err   error
	}

	results := make(chan result, 1)

	go func() {
		batch, err := queue.Dequeue(ctx)

		results <- result{
			batch: batch,
			err:   err,
		}
	}()

	select {
	case received := <-results:
		t.Fatalf(
			"Dequeue() returned before an item was available: batch=%q err=%v",
			received.batch.ID,
			received.err,
		)

	case <-time.After(50 * time.Millisecond):
		// Expected: consumer is blocked.
	}

	if err := queue.Enqueue(ctx, testBatch("batch-1")); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	select {
	case received := <-results:
		if received.err != nil {
			t.Fatalf("Dequeue() error = %v", received.err)
		}

		if received.batch.ID != BatchID("batch-1") {
			t.Fatalf(
				"Dequeue() ID = %q, want batch-1",
				received.batch.ID,
			)
		}

	case <-ctx.Done():
		t.Fatalf(
			"Dequeue() did not wake after enqueue: %v",
			ctx.Err(),
		)
	}
}

func TestEnqueueHonoursContextCancellation(t *testing.T) {
	queue := newTestQueue(t, 1)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := queue.Enqueue(ctx, testBatch("batch-2"))

	if !errors.Is(err, context.Canceled) {
		t.Fatalf(
			"Enqueue() error = %v, want context.Canceled",
			err,
		)
	}
}

func TestDequeueHonoursContextCancellation(t *testing.T) {
	queue := newTestQueue(t, 1)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := queue.Dequeue(ctx)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf(
			"Dequeue() error = %v, want context.Canceled",
			err,
		)
	}
}

func TestClosePreventsNewEnqueue(t *testing.T) {
	queue := newTestQueue(t, 2)

	queue.Close()

	err := queue.TryEnqueue(testBatch("batch-1"))

	if !errors.Is(err, ErrQueueClosed) {
		t.Fatalf(
			"TryEnqueue() error = %v, want ErrQueueClosed",
			err,
		)
	}

	if !queue.Closed() {
		t.Fatal("queue should report closed")
	}
}

func TestCloseAllowsDrain(t *testing.T) {
	queue := newTestQueue(t, 2)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	queue.Close()

	batch, err := queue.TryDequeue()
	if err != nil {
		t.Fatalf(
			"TryDequeue() should drain closed queue: %v",
			err,
		)
	}

	if batch.ID != BatchID("batch-1") {
		t.Fatalf(
			"TryDequeue() ID = %q, want batch-1",
			batch.ID,
		)
	}

	_, err = queue.TryDequeue()

	if !errors.Is(err, ErrQueueClosed) {
		t.Fatalf(
			"drained closed queue error = %v, want ErrQueueClosed",
			err,
		)
	}
}

func TestCloseWakesBlockedDequeue(t *testing.T) {
	queue := newTestQueue(t, 1)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	result := make(chan error, 1)

	go func() {
		_, err := queue.Dequeue(ctx)
		result <- err
	}()

	time.Sleep(25 * time.Millisecond)
	queue.Close()

	select {
	case err := <-result:
		if !errors.Is(err, ErrQueueClosed) {
			t.Fatalf(
				"Dequeue() error = %v, want ErrQueueClosed",
				err,
			)
		}

	case <-ctx.Done():
		t.Fatalf(
			"blocked Dequeue() was not awakened by Close(): %v",
			ctx.Err(),
		)
	}
}

func TestCloseWakesBlockedEnqueue(t *testing.T) {
	queue := newTestQueue(t, 1)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	result := make(chan error, 1)

	go func() {
		result <- queue.Enqueue(ctx, testBatch("batch-2"))
	}()

	time.Sleep(25 * time.Millisecond)
	queue.Close()

	select {
	case err := <-result:
		if !errors.Is(err, ErrQueueClosed) {
			t.Fatalf(
				"Enqueue() error = %v, want ErrQueueClosed",
				err,
			)
		}

	case <-ctx.Done():
		t.Fatalf(
			"blocked Enqueue() was not awakened by Close(): %v",
			ctx.Err(),
		)
	}
}

func TestPeekDoesNotRemoveBatch(t *testing.T) {
	queue := newTestQueue(t, 2)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	batch, err := queue.Peek()
	if err != nil {
		t.Fatalf("Peek() error = %v", err)
	}

	if batch.ID != BatchID("batch-1") {
		t.Fatalf(
			"Peek() ID = %q, want batch-1",
			batch.ID,
		)
	}

	if queue.Len() != 1 {
		t.Fatalf(
			"Peek() changed queue depth: got %d, want 1",
			queue.Len(),
		)
	}
}

func TestQueueDefensiveCopyOnEnqueue(t *testing.T) {
	queue := newTestQueue(t, 1)

	batch := testBatch("batch-1")
	batch.Records[0].Payload = []byte(`{"value":"original"}`)
	batch.Records[0].Metadata = map[string]string{
		"source": "original",
	}

	if err := queue.TryEnqueue(batch); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	batch.Records[0].Payload[0] = 'X'
	batch.Records[0].Metadata["source"] = "modified"
	batch.Destination.Type = "modified"

	queued, err := queue.TryDequeue()
	if err != nil {
		t.Fatalf("TryDequeue() error = %v", err)
	}

	if string(queued.Records[0].Payload) != `{"value":"original"}` {
		t.Fatalf(
			"queued payload was mutated: %q",
			string(queued.Records[0].Payload),
		)
	}

	if queued.Records[0].Metadata["source"] != "original" {
		t.Fatalf(
			"queued metadata was mutated: %q",
			queued.Records[0].Metadata["source"],
		)
	}

	if queued.Destination.Type != "test" {
		t.Fatalf(
			"queued destination was mutated: %q",
			queued.Destination.Type,
		)
	}
}

func TestQueueStats(t *testing.T) {
	queue := newTestQueue(t, 3)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	if err := queue.TryEnqueue(testBatch("batch-2")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	if _, err := queue.TryDequeue(); err != nil {
		t.Fatalf("TryDequeue() error = %v", err)
	}

	stats := queue.Stats()

	if stats.Capacity != 3 {
		t.Errorf(
			"Stats().Capacity = %d, want 3",
			stats.Capacity,
		)
	}

	if stats.Depth != 1 {
		t.Errorf(
			"Stats().Depth = %d, want 1",
			stats.Depth,
		)
	}

	if stats.Enqueued != 2 {
		t.Errorf(
			"Stats().Enqueued = %d, want 2",
			stats.Enqueued,
		)
	}

	if stats.Dequeued != 1 {
		t.Errorf(
			"Stats().Dequeued = %d, want 1",
			stats.Dequeued,
		)
	}
}

func TestWaitUntilEmpty(t *testing.T) {
	queue := newTestQueue(t, 2)

	if err := queue.TryEnqueue(testBatch("batch-1")); err != nil {
		t.Fatalf("TryEnqueue() error = %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	done := make(chan error, 1)

	go func() {
		done <- queue.WaitUntilEmpty(
			ctx,
			time.Millisecond,
		)
	}()

	time.Sleep(10 * time.Millisecond)

	if _, err := queue.TryDequeue(); err != nil {
		t.Fatalf("TryDequeue() error = %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("WaitUntilEmpty() error = %v", err)
		}

	case <-ctx.Done():
		t.Fatalf(
			"WaitUntilEmpty() did not complete: %v",
			ctx.Err(),
		)
	}
}

// TestQueueConcurrentProducersConsumers intentionally exercises Queue under
// substantial concurrent access.
//
// Run this test with:
//
//	go test -race ./internal/export
//
// The race detector then observes simultaneous accesses performed by these
// goroutines.
func TestQueueConcurrentProducersConsumers(t *testing.T) {
	const (
		producers        = 8
		consumers        = 6
		batchesPerWorker = 250
	)

	queue := newTestQueue(t, 64)

	ctx, cancel := context.WithTimeout(
		context.Background(),
		10*time.Second,
	)
	defer cancel()

	totalExpected := int64(producers * batchesPerWorker)

	var produced atomic.Int64
	var consumed atomic.Int64

	producerErrors := make(chan error, producers)
	consumerErrors := make(chan error, consumers)

	var producersWG sync.WaitGroup
	var consumersWG sync.WaitGroup

	producersWG.Add(producers)

	for producer := 0; producer < producers; producer++ {
		producer := producer

		go func() {
			defer producersWG.Done()

			for item := 0; item < batchesPerWorker; item++ {
				id := fmt.Sprintf(
					"producer-%d-batch-%d",
					producer,
					item,
				)

				if err := queue.Enqueue(ctx, testBatch(id)); err != nil {
					producerErrors <- err
					return
				}

				produced.Add(1)

				// Exercise concurrent read-only queue operations while
				// producers and consumers are modifying queue state.
				_ = queue.Len()
				_ = queue.Available()
				_ = queue.Full()
				_ = queue.Stats()
			}
		}()
	}

	consumersWG.Add(consumers)

	for consumer := 0; consumer < consumers; consumer++ {
		go func() {
			defer consumersWG.Done()

			for {
				batch, err := queue.Dequeue(ctx)
				if err != nil {
					if errors.Is(err, ErrQueueClosed) {
						return
					}

					consumerErrors <- err
					return
				}

				if batch.ID == "" {
					consumerErrors <- errors.New(
						"dequeued batch has empty id",
					)
					return
				}

				consumed.Add(1)

				_ = queue.Stats()
				_ = queue.Len()
			}
		}()
	}

	producersWG.Wait()

	close(producerErrors)

	for err := range producerErrors {
		if err != nil {
			t.Fatalf("producer error = %v", err)
		}
	}

	if produced.Load() != totalExpected {
		t.Fatalf(
			"produced = %d, want %d",
			produced.Load(),
			totalExpected,
		)
	}

	queue.Close()

	consumersWG.Wait()

	close(consumerErrors)

	for err := range consumerErrors {
		if err != nil {
			t.Fatalf("consumer error = %v", err)
		}
	}

	if consumed.Load() != totalExpected {
		t.Fatalf(
			"consumed = %d, want %d",
			consumed.Load(),
			totalExpected,
		)
	}

	stats := queue.Stats()

	if stats.Depth != 0 {
		t.Fatalf(
			"final queue depth = %d, want 0",
			stats.Depth,
		)
	}

	if stats.Enqueued != uint64(totalExpected) {
		t.Fatalf(
			"enqueued = %d, want %d",
			stats.Enqueued,
			totalExpected,
		)
	}

	if stats.Dequeued != uint64(totalExpected) {
		t.Fatalf(
			"dequeued = %d, want %d",
			stats.Dequeued,
			totalExpected,
		)
	}
}

// TestQueueConcurrentClose exercises Close while producers and consumers are
// actively using the queue.
//
// This is not intended to assert how many records are delivered. Its purpose
// is to ensure concurrent shutdown is safe and all blocked goroutines can
// terminate.
func TestQueueConcurrentClose(t *testing.T) {
	queue := newTestQueue(t, 16)

	ctx, cancel := context.WithTimeout(
		context.Background(),
		5*time.Second,
	)
	defer cancel()

	var wg sync.WaitGroup

	const workers = 12

	wg.Add(workers)

	for worker := 0; worker < workers; worker++ {
		worker := worker

		go func() {
			defer wg.Done()

			for i := 0; ; i++ {
				id := fmt.Sprintf(
					"worker-%d-%d",
					worker,
					i,
				)

				err := queue.Enqueue(ctx, testBatch(id))
				if err != nil {
					if errors.Is(err, ErrQueueClosed) ||
						errors.Is(err, context.Canceled) ||
						errors.Is(err, context.DeadlineExceeded) {
						return
					}

					t.Errorf("Enqueue() error = %v", err)
					return
				}

				_, err = queue.Dequeue(ctx)
				if err != nil {
					if errors.Is(err, ErrQueueClosed) ||
						errors.Is(err, context.Canceled) ||
						errors.Is(err, context.DeadlineExceeded) {
						return
					}

					t.Errorf("Dequeue() error = %v", err)
					return
				}

				_ = queue.Stats()
				_ = queue.Snapshot()
			}
		}()
	}

	time.Sleep(50 * time.Millisecond)

	// Multiple concurrent Close calls exercise idempotent shutdown under
	// the race detector.
	var closeWG sync.WaitGroup

	for i := 0; i < 8; i++ {
		closeWG.Add(1)

		go func() {
			defer closeWG.Done()
			queue.Close()
		}()
	}

	closeWG.Wait()

	finished := make(chan struct{})

	go func() {
		wg.Wait()
		close(finished)
	}()

	select {
	case <-finished:

	case <-ctx.Done():
		t.Fatalf(
			"workers did not terminate after Close(): %v",
			ctx.Err(),
		)
	}

	if !queue.Closed() {
		t.Fatal("queue should be closed")
	}
}
