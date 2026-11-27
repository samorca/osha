package export

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// Common exporter errors.
//
// Concrete exporters may wrap these errors so callers can use errors.Is
// without depending on transport-specific implementations.
var (
	ErrExporterNotFound    = errors.New("exporter not found")
	ErrExporterExists      = errors.New("exporter already registered")
	ErrExporterUnavailable = errors.New("exporter unavailable")
	ErrExporterClosed      = errors.New("exporter closed")
	ErrExportRejected      = errors.New("export rejected")
)

// Exporter describes a destination implementation capable of delivering
// an OTGH batch.
//
// Implementations must:
//
//   - honour context cancellation and deadlines;
//   - be safe for concurrent use unless explicitly documented otherwise;
//   - return a DeliveryResult for completed delivery attempts;
//   - return an error when delivery could not be completed;
//   - avoid implementing retry loops internally when the shared OTGH retry
//     subsystem owns retry behaviour.
//
// The exporter is responsible for translating the generic Batch into the
// representation required by its destination.
type Exporter interface {
	// ID returns the identity under which the exporter is registered.
	ID() DestinationID

	// Type returns the exporter implementation type, such as "http",
	// "grpc", "objectstore", "webhook", or an OTGH-native transport.
	Type() string

	// Export attempts to deliver a batch.
	//
	// Implementations must honour ctx and should return promptly when it is
	// cancelled.
	Export(ctx context.Context, batch Batch) (DeliveryResult, error)

	// Health checks whether the exporter can currently be used.
	//
	// Health should be lightweight. It must not mutate remote state.
	Health(ctx context.Context) error

	// Close releases resources owned by the exporter.
	//
	// Close should be safe to call during graceful shutdown.
	Close() error
}

// ExporterRegistry stores exporter implementations by DestinationID.
//
// The registry is thread-safe and intentionally independent from Router.
// Router owns routing policy; ExporterRegistry owns live delivery
// implementations.
type ExporterRegistry struct {
	mu sync.RWMutex

	exporters map[DestinationID]Exporter
}

// NewExporterRegistry creates an empty exporter registry.
func NewExporterRegistry() *ExporterRegistry {
	return &ExporterRegistry{
		exporters: make(map[DestinationID]Exporter),
	}
}

// Register adds an exporter.
//
// Register rejects duplicate IDs instead of replacing a running exporter.
// Replacing live exporters requires lifecycle coordination and should be
// performed explicitly through Replace.
func (r *ExporterRegistry) Register(exporter Exporter) error {
	if r == nil {
		return errors.New("exporter registry is nil")
	}

	if exporter == nil {
		return errors.New("exporter is nil")
	}

	id := exporter.ID()

	if strings.TrimSpace(string(id)) == "" {
		return errors.New("exporter id is required")
	}

	if strings.TrimSpace(exporter.Type()) == "" {
		return errors.New("exporter type is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.exporters == nil {
		r.exporters = make(map[DestinationID]Exporter)
	}

	if _, exists := r.exporters[id]; exists {
		return fmt.Errorf("%w: %q", ErrExporterExists, id)
	}

	r.exporters[id] = exporter

	return nil
}

// Replace atomically replaces the exporter registered under exporter.ID().
//
// The old exporter is returned to the caller so its lifecycle can be handled
// deliberately. Replace does not call Close while holding the registry lock.
func (r *ExporterRegistry) Replace(exporter Exporter) (Exporter, error) {
	if r == nil {
		return nil, errors.New("exporter registry is nil")
	}

	if exporter == nil {
		return nil, errors.New("exporter is nil")
	}

	id := exporter.ID()

	if strings.TrimSpace(string(id)) == "" {
		return nil, errors.New("exporter id is required")
	}

	if strings.TrimSpace(exporter.Type()) == "" {
		return nil, errors.New("exporter type is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.exporters == nil {
		r.exporters = make(map[DestinationID]Exporter)
	}

	previous := r.exporters[id]
	r.exporters[id] = exporter

	return previous, nil
}

// Get returns an exporter by destination ID.
func (r *ExporterRegistry) Get(id DestinationID) (Exporter, error) {
	if r == nil {
		return nil, errors.New("exporter registry is nil")
	}

	if strings.TrimSpace(string(id)) == "" {
		return nil, errors.New("exporter id is required")
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	exporter, exists := r.exporters[id]
	if !exists {
		return nil, fmt.Errorf("%w: %q", ErrExporterNotFound, id)
	}

	return exporter, nil
}

// Remove removes and returns an exporter.
//
// Remove deliberately does not call Close. The caller controls whether the
// exporter should be drained, immediately closed, or reused elsewhere.
func (r *ExporterRegistry) Remove(id DestinationID) (Exporter, error) {
	if r == nil {
		return nil, errors.New("exporter registry is nil")
	}

	if strings.TrimSpace(string(id)) == "" {
		return nil, errors.New("exporter id is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	exporter, exists := r.exporters[id]
	if !exists {
		return nil, fmt.Errorf("%w: %q", ErrExporterNotFound, id)
	}

	delete(r.exporters, id)

	return exporter, nil
}

// IDs returns a snapshot of all registered exporter IDs.
//
// No ordering guarantee is provided.
func (r *ExporterRegistry) IDs() []DestinationID {
	if r == nil {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	ids := make([]DestinationID, 0, len(r.exporters))

	for id := range r.exporters {
		ids = append(ids, id)
	}

	return ids
}

// Count returns the number of registered exporters.
func (r *ExporterRegistry) Count() int {
	if r == nil {
		return 0
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.exporters)
}

// Health checks a registered exporter.
//
// The registry lock is released before calling into exporter code so a slow or
// blocked health check cannot prevent registry reads and writes.
func (r *ExporterRegistry) Health(
	ctx context.Context,
	id DestinationID,
) error {
	exporter, err := r.Get(id)
	if err != nil {
		return err
	}

	if ctx == nil {
		return errors.New("health context is nil")
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := exporter.Health(ctx); err != nil {
		return fmt.Errorf(
			"%w: exporter %q: %w",
			ErrExporterUnavailable,
			id,
			err,
		)
	}

	return nil
}

// CloseAll removes every exporter from the registry and then closes them.
//
// Exporters are detached under the registry lock and closed afterwards.
// This prevents exporter shutdown from blocking registry operations while
// holding an internal mutex.
//
// All close operations are attempted. When multiple exporters fail to close,
// the returned error contains all failures using errors.Join.
func (r *ExporterRegistry) CloseAll() error {
	if r == nil {
		return nil
	}

	r.mu.Lock()

	exporters := make([]Exporter, 0, len(r.exporters))

	for _, exporter := range r.exporters {
		exporters = append(exporters, exporter)
	}

	r.exporters = make(map[DestinationID]Exporter)

	r.mu.Unlock()

	var closeErrors []error

	for _, exporter := range exporters {
		if err := exporter.Close(); err != nil {
			closeErrors = append(
				closeErrors,
				fmt.Errorf(
					"close exporter %q: %w",
					exporter.ID(),
					err,
				),
			)
		}
	}

	return errors.Join(closeErrors...)
}

// ExportTo validates a batch and sends it through the exporter registered for
// the batch destination.
//
// ExportTo does not retry failed deliveries. Retry policy will be owned by
// retry.go and worker.go so all exporter implementations have consistent retry
// semantics.
func (r *ExporterRegistry) ExportTo(
	ctx context.Context,
	batch Batch,
) (DeliveryResult, error) {
	if r == nil {
		return DeliveryResult{}, errors.New("exporter registry is nil")
	}

	if ctx == nil {
		return DeliveryResult{}, errors.New("export context is nil")
	}

	if err := ctx.Err(); err != nil {
		return DeliveryResult{}, err
	}

	if err := batch.Validate(); err != nil {
		return DeliveryResult{}, fmt.Errorf("export batch: %w", err)
	}

	exporter, err := r.Get(batch.Destination.ID)
	if err != nil {
		return DeliveryResult{}, err
	}

	if exporter.ID() != batch.Destination.ID {
		return DeliveryResult{}, fmt.Errorf(
			"exporter id %q does not match destination id %q",
			exporter.ID(),
			batch.Destination.ID,
		)
	}

	result, err := exporter.Export(ctx, batch)
	if err != nil {
		return result, fmt.Errorf(
			"export batch %q using exporter %q: %w",
			batch.ID,
			exporter.ID(),
			err,
		)
	}

	if err := validateDeliveryResult(batch, result); err != nil {
		return result, fmt.Errorf(
			"exporter %q returned invalid delivery result: %w",
			exporter.ID(),
			err,
		)
	}

	return result, nil
}

func validateDeliveryResult(
	batch Batch,
	result DeliveryResult,
) error {
	if result.BatchID != batch.ID {
		return fmt.Errorf(
			"result batch id %q does not match batch %q",
			result.BatchID,
			batch.ID,
		)
	}

	if result.DestinationID != batch.Destination.ID {
		return fmt.Errorf(
			"result destination id %q does not match destination %q",
			result.DestinationID,
			batch.Destination.ID,
		)
	}

	if !result.Status.Valid() {
		return fmt.Errorf(
			"invalid delivery status %d",
			result.Status,
		)
	}

	if result.Attempt < 0 {
		return errors.New("delivery attempt cannot be negative")
	}

	if result.AcceptedRecords < 0 {
		return errors.New("accepted record count cannot be negative")
	}

	if result.RejectedRecords < 0 {
		return errors.New("rejected record count cannot be negative")
	}

	total := result.AcceptedRecords + result.RejectedRecords
	if total > batch.RecordCount() {
		return fmt.Errorf(
			"delivery result accounts for %d records but batch contains %d",
			total,
			batch.RecordCount(),
		)
	}

	if !result.FinishedAt.IsZero() &&
		!result.StartedAt.IsZero() &&
		result.FinishedAt.Before(result.StartedAt) {
		return errors.New("delivery finished_at precedes started_at")
	}

	return nil
}
