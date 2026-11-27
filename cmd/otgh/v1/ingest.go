package v1

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/samorca/otgh/internal/export"
)

// SubmitRecords routes, batches, and enqueues records for asynchronous delivery.
// The returned count is the number of input records accepted into every
// destination queue selected by the configured routes.
func (r *Runtime) SubmitRecords(
	ctx context.Context,
	records []export.Record,
) (int, error) {
	if r == nil {
		return 0, errors.New("OTGH runtime is nil")
	}
	if ctx == nil {
		return 0, errors.New("OTGH ingestion context is nil")
	}
	if len(records) == 0 {
		return 0, errors.New("OTGH ingestion requires at least one record")
	}
	if r.Closed() {
		return 0, errors.New("OTGH runtime is closed")
	}
	if r.router == nil || r.queue == nil || r.batchIDs == nil {
		return 0, errors.New("OTGH ingestion components are unavailable")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	r.metrics.RecordReceived(uint64(len(records)))

	type destinationRecords struct {
		destination export.Destination
		records     []export.Record
	}

	grouped := make(map[export.DestinationID]*destinationRecords)
	destinationOrder := make([]export.DestinationID, 0)
	for _, record := range records {
		route, err := r.router.Route(record)
		if err != nil {
			return 0, fmt.Errorf("route record %q: %w", record.ID, err)
		}
		for _, destination := range route.Destinations {
			group, exists := grouped[destination.ID]
			if !exists {
				group = &destinationRecords{destination: destination}
				grouped[destination.ID] = group
				destinationOrder = append(destinationOrder, destination.ID)
			}
			group.records = append(group.records, record)
		}
	}

	for _, destinationID := range destinationOrder {
		group := grouped[destinationID]
		batcher, err := export.NewBatcher(
			r.config.Batching.ExportConfig(),
			group.destination,
			r.batchIDs,
		)
		if err != nil {
			return 0, fmt.Errorf("create batcher for %q: %w", destinationID, err)
		}

		for _, record := range group.records {
			if err := batcher.Add(record); err != nil {
				if !errors.Is(err, export.ErrBatchFull) {
					return 0, fmt.Errorf("batch record %q: %w", record.ID, err)
				}
				if err := r.flushBatch(ctx, batcher); err != nil {
					return 0, err
				}
				if err := batcher.Add(record); err != nil {
					return 0, fmt.Errorf("batch record %q after flush: %w", record.ID, err)
				}
			}
		}

		if !batcher.Empty() {
			if err := r.flushBatch(ctx, batcher); err != nil {
				return 0, err
			}
		}
		batcher.Close()
	}

	return len(records), nil
}

func (r *Runtime) flushBatch(
	ctx context.Context,
	batcher *export.Batcher,
) error {
	batch, err := batcher.Flush(time.Now().UTC())
	if err != nil {
		return fmt.Errorf("flush OTGH batch: %w", err)
	}
	if err := r.queue.Enqueue(ctx, batch); err != nil {
		return fmt.Errorf("enqueue OTGH batch %q: %w", batch.ID, err)
	}
	r.metrics.RecordBatched(uint64(batch.RecordCount()))
	r.metrics.BatchCreated()
	r.metrics.BatchQueued()
	return nil
}