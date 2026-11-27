
package export

import (
	"sync"
	"sync/atomic"
	"time"
)

// DeliveryMetrics tracks OTGH's internal export pipeline.
//
// It deliberately does not depend on Prometheus, OpenTelemetry, or another
// external metrics framework. External integrations can translate Snapshot()
// into their own metric representation.
//
// Independent high-frequency counters use typed atomic values. Compound
// latency state uses a mutex because count, sum, minimum, maximum, and last
// duration must remain internally consistent.
type DeliveryMetrics struct {
	// Ingestion and batching.
	recordsReceived atomic.Uint64
	recordsBatched  atomic.Uint64
	batchesCreated  atomic.Uint64

	// Queue behaviour.
	batchesQueued   atomic.Uint64
	batchesDequeued atomic.Uint64
	queueFull       atomic.Uint64

	// Delivery attempts.
	exportAttempts atomic.Uint64
	exportSuccess  atomic.Uint64
	exportFailures atomic.Uint64

	// Record-level delivery outcomes.
	recordsAccepted atomic.Uint64
	recordsRejected atomic.Uint64

	// Retry behaviour.
	retriesScheduled  atomic.Uint64
	retriesExhausted  atomic.Uint64
	retriesSuppressed atomic.Uint64

	// Acknowledgement behaviour.
	acknowledgedBatches atomic.Uint64
	partialBatches      atomic.Uint64
	failedBatches       atomic.Uint64
	deadLetteredBatches atomic.Uint64

	// Spool behaviour.
	spoolWrites      atomic.Uint64
	spoolReads       atomic.Uint64
	spoolDeletes     atomic.Uint64
	spoolRecoveries  atomic.Uint64
	spoolQuarantines atomic.Uint64
	spoolErrors      atomic.Uint64

	// Worker behaviour.
	workersStarted   atomic.Uint64
	workersStopped   atomic.Uint64
	activeWorkers    atomic.Int64
	activeDeliveries atomic.Int64

	// Bytes successfully accepted by a destination.
	bytesDelivered atomic.Uint64

	latencyMu sync.RWMutex
	latency   latencyAccumulator
}

// latencyAccumulator stores compound delivery latency state.
type latencyAccumulator struct {
	count uint64
	total time.Duration
	min   time.Duration
	max   time.Duration
	last  time.Duration
}

// DeliveryMetricsSnapshot is an immutable point-in-time view of OTGH metrics.
type DeliveryMetricsSnapshot struct {
	Timestamp time.Time

	// Ingestion and batching.
	RecordsReceived uint64
	RecordsBatched  uint64
	BatchesCreated  uint64

	// Queue state and cumulative operations.
	BatchesQueued   uint64
	BatchesDequeued uint64
	QueueFull       uint64

	QueueDepth    int
	QueueCapacity int

	// Delivery.
	ExportAttempts uint64
	ExportSuccess  uint64
	ExportFailures uint64

	RecordsAccepted uint64
	RecordsRejected uint64
	BytesDelivered  uint64

	// Retry.
	RetriesScheduled  uint64
	RetriesExhausted  uint64
	RetriesSuppressed uint64

	// Acknowledgements.
	AcknowledgedBatches uint64
	PartialBatches      uint64
	FailedBatches       uint64
	DeadLetteredBatches uint64

	// Spool.
	SpoolWrites      uint64
	SpoolReads       uint64
	SpoolDeletes     uint64
	SpoolRecoveries  uint64
	SpoolQuarantines uint64
	SpoolErrors      uint64

	SpoolEntries int
	SpoolBytes   int64
	SpoolMaxBytes int64

	// Workers.
	WorkersStarted   uint64
	WorkersStopped   uint64
	ActiveWorkers    int64
	ActiveDeliveries int64

	// Delivery latency.
	DeliveryLatencyCount uint64
	DeliveryLatencyTotal time.Duration
	DeliveryLatencyMin   time.Duration
	DeliveryLatencyMax   time.Duration
	DeliveryLatencyLast  time.Duration
	DeliveryLatencyMean  time.Duration
}

// NewDeliveryMetrics creates an empty metrics collector.
func NewDeliveryMetrics() *DeliveryMetrics {
	return &DeliveryMetrics{}
}

// RecordReceived increments the number of records entering OTGH.
func (m *DeliveryMetrics) RecordReceived(count uint64) {
	if m == nil || count == 0 {
		return
	}

	m.recordsReceived.Add(count)
}

// RecordBatched records records that were placed into delivery batches.
func (m *DeliveryMetrics) RecordBatched(count uint64) {
	if m == nil || count == 0 {
		return
	}

	m.recordsBatched.Add(count)
}

// BatchCreated records a newly created delivery batch.
func (m *DeliveryMetrics) BatchCreated() {
	if m == nil {
		return
	}

	m.batchesCreated.Add(1)
}

// BatchQueued records a batch entering the in-memory queue.
func (m *DeliveryMetrics) BatchQueued() {
	if m == nil {
		return
	}

	m.batchesQueued.Add(1)
}

// BatchDequeued records a batch leaving the in-memory queue.
func (m *DeliveryMetrics) BatchDequeued() {
	if m == nil {
		return
	}

	m.batchesDequeued.Add(1)
}

// QueueFull records backpressure caused by queue capacity.
func (m *DeliveryMetrics) QueueFull() {
	if m == nil {
		return
	}

	m.queueFull.Add(1)
}

// ExportAttempt records one exporter invocation.
func (m *DeliveryMetrics) ExportAttempt() {
	if m == nil {
		return
	}

	m.exportAttempts.Add(1)
}

// ExportSucceeded records a successful delivery attempt.
//
// acceptedRecords is the number of destination-accepted records.
// deliveredBytes should represent bytes the caller considers successfully
// delivered, not merely attempted.
func (m *DeliveryMetrics) ExportSucceeded(
	acceptedRecords uint64,
	deliveredBytes uint64,
	duration time.Duration,
) {
	if m == nil {
		return
	}

	m.exportSuccess.Add(1)

	if acceptedRecords > 0 {
		m.recordsAccepted.Add(acceptedRecords)
	}

	if deliveredBytes > 0 {
		m.bytesDelivered.Add(deliveredBytes)
	}

	m.ObserveDeliveryLatency(duration)
}

// ExportFailed records a failed delivery attempt.
//
// rejectedRecords should only represent records explicitly rejected by the
// destination. A transport failure does not automatically mean that every
// record was rejected.
func (m *DeliveryMetrics) ExportFailed(
	rejectedRecords uint64,
	duration time.Duration,
) {
	if m == nil {
		return
	}

	m.exportFailures.Add(1)

	if rejectedRecords > 0 {
		m.recordsRejected.Add(rejectedRecords)
	}

	m.ObserveDeliveryLatency(duration)
}

// RecordsAccepted increments the accepted-record counter independently of an
// export attempt.
//
// This is useful for partial-delivery workflows.
func (m *DeliveryMetrics) RecordsAccepted(count uint64) {
	if m == nil || count == 0 {
		return
	}

	m.recordsAccepted.Add(count)
}

// RecordsRejected increments the rejected-record counter independently of an
// export attempt.
func (m *DeliveryMetrics) RecordsRejected(count uint64) {
	if m == nil || count == 0 {
		return
	}

	m.recordsRejected.Add(count)
}

// RetryScheduled records a retry that has been accepted by policy.
func (m *DeliveryMetrics) RetryScheduled() {
	if m == nil {
		return
	}

	m.retriesScheduled.Add(1)
}

// RetryExhausted records a batch whose retry budget was exhausted.
func (m *DeliveryMetrics) RetryExhausted() {
	if m == nil {
		return
	}

	m.retriesExhausted.Add(1)
}

// RetrySuppressed records a failure classified as non-retryable.
func (m *DeliveryMetrics) RetrySuppressed() {
	if m == nil {
		return
	}

	m.retriesSuppressed.Add(1)
}

// Acknowledged records complete destination acknowledgement.
func (m *DeliveryMetrics) Acknowledged() {
	if m == nil {
		return
	}

	m.acknowledgedBatches.Add(1)
}

// PartialAcknowledgement records a partially accepted batch.
func (m *DeliveryMetrics) PartialAcknowledgement() {
	if m == nil {
		return
	}

	m.partialBatches.Add(1)
}

// BatchFailed records a permanently failed batch.
func (m *DeliveryMetrics) BatchFailed() {
	if m == nil {
		return
	}

	m.failedBatches.Add(1)
}

// BatchDeadLettered records a batch moved out of normal delivery processing.
func (m *DeliveryMetrics) BatchDeadLettered() {
	if m == nil {
		return
	}

	m.deadLetteredBatches.Add(1)
}

// SpoolWrite records a durable spool write.
func (m *DeliveryMetrics) SpoolWrite() {
	if m == nil {
		return
	}

	m.spoolWrites.Add(1)
}

// SpoolRead records a durable spool read.
func (m *DeliveryMetrics) SpoolRead() {
	if m == nil {
		return
	}

	m.spoolReads.Add(1)
}

// SpoolDelete records removal of an acknowledged/finished spool entry.
func (m *DeliveryMetrics) SpoolDelete() {
	if m == nil {
		return
	}

	m.spoolDeletes.Add(1)
}

// SpoolRecovery records recovery of a durable pending batch.
func (m *DeliveryMetrics) SpoolRecovery(count uint64) {
	if m == nil || count == 0 {
		return
	}

	m.spoolRecoveries.Add(count)
}

// SpoolQuarantine records an entry moved to quarantine.
func (m *DeliveryMetrics) SpoolQuarantine() {
	if m == nil {
		return
	}

	m.spoolQuarantines.Add(1)
}

// SpoolError records a spool operation failure.
func (m *DeliveryMetrics) SpoolError() {
	if m == nil {
		return
	}

	m.spoolErrors.Add(1)
}

// WorkerStarted records a worker entering its run loop.
func (m *DeliveryMetrics) WorkerStarted() {
	if m == nil {
		return
	}

	m.workersStarted.Add(1)
	m.activeWorkers.Add(1)
}

// WorkerStopped records a worker leaving its run loop.
func (m *DeliveryMetrics) WorkerStopped() {
	if m == nil {
		return
	}

	m.workersStopped.Add(1)
	decrementAtomicNonNegative(&m.activeWorkers)
}

// DeliveryStarted records a worker beginning one delivery attempt.
func (m *DeliveryMetrics) DeliveryStarted() {
	if m == nil {
		return
	}

	m.activeDeliveries.Add(1)
}

// DeliveryFinished records the completion of one delivery attempt.
func (m *DeliveryMetrics) DeliveryFinished() {
	if m == nil {
		return
	}

	decrementAtomicNonNegative(&m.activeDeliveries)
}

// ObserveDeliveryLatency records one delivery duration.
//
// Negative durations are ignored because they cannot represent a valid
// completed delivery interval.
func (m *DeliveryMetrics) ObserveDeliveryLatency(
	duration time.Duration,
) {
	if m == nil || duration < 0 {
		return
	}

	m.latencyMu.Lock()
	defer m.latencyMu.Unlock()

	m.latency.count++
	m.latency.total += duration
	m.latency.last = duration

	if m.latency.count == 1 || duration < m.latency.min {
		m.latency.min = duration
	}

	if m.latency.count == 1 || duration > m.latency.max {
		m.latency.max = duration
	}
}

// Snapshot returns a consistent point-in-time view.
//
// Queue and spool are optional. Passing nil omits their live gauge values.
// Their own Stats methods provide synchronisation for those structures.
func (m *DeliveryMetrics) Snapshot(
	queue *Queue,
	spool *Spool,
) DeliveryMetricsSnapshot {
	if m == nil {
		return DeliveryMetricsSnapshot{
			Timestamp: time.Now().UTC(),
		}
	}

	snapshot := DeliveryMetricsSnapshot{
		Timestamp: time.Now().UTC(),

		RecordsReceived: m.recordsReceived.Load(),
		RecordsBatched:  m.recordsBatched.Load(),
		BatchesCreated:  m.batchesCreated.Load(),

		BatchesQueued:   m.batchesQueued.Load(),
		BatchesDequeued: m.batchesDequeued.Load(),
		QueueFull:       m.queueFull.Load(),

		ExportAttempts: m.exportAttempts.Load(),
		ExportSuccess:  m.exportSuccess.Load(),
		ExportFailures: m.exportFailures.Load(),

		RecordsAccepted: m.recordsAccepted.Load(),
		RecordsRejected: m.recordsRejected.Load(),
		BytesDelivered:  m.bytesDelivered.Load(),

		RetriesScheduled:  m.retriesScheduled.Load(),
		RetriesExhausted:  m.retriesExhausted.Load(),
		RetriesSuppressed: m.retriesSuppressed.Load(),

		AcknowledgedBatches: m.acknowledgedBatches.Load(),
		PartialBatches:      m.partialBatches.Load(),
		FailedBatches:       m.failedBatches.Load(),
		DeadLetteredBatches: m.deadLetteredBatches.Load(),

		SpoolWrites:      m.spoolWrites.Load(),
		SpoolReads:       m.spoolReads.Load(),
		SpoolDeletes:     m.spoolDeletes.Load(),
		SpoolRecoveries:  m.spoolRecoveries.Load(),
		SpoolQuarantines: m.spoolQuarantines.Load(),
		SpoolErrors:      m.spoolErrors.Load(),

		WorkersStarted:   m.workersStarted.Load(),
		WorkersStopped:   m.workersStopped.Load(),
		ActiveWorkers:    m.activeWorkers.Load(),
		ActiveDeliveries: m.activeDeliveries.Load(),
	}

	if queue != nil {
		queueStats := queue.Stats()

		snapshot.QueueDepth = queueStats.Depth
		snapshot.QueueCapacity = queueStats.Capacity
	}

	if spool != nil {
		spoolStats := spool.Stats()

		snapshot.SpoolEntries = spoolStats.Entries
		snapshot.SpoolBytes = spoolStats.Bytes
		snapshot.SpoolMaxBytes = spoolStats.MaxBytes
	}

	m.latencyMu.RLock()

	snapshot.DeliveryLatencyCount = m.latency.count
	snapshot.DeliveryLatencyTotal = m.latency.total
	snapshot.DeliveryLatencyMin = m.latency.min
	snapshot.DeliveryLatencyMax = m.latency.max
	snapshot.DeliveryLatencyLast = m.latency.last

	if m.latency.count > 0 {
		snapshot.DeliveryLatencyMean = time.Duration(
			int64(m.latency.total) / int64(m.latency.count),
		)
	}

	m.latencyMu.RUnlock()

	return snapshot
}

// Reset clears cumulative metrics.
//
// Reset is intended primarily for tests and controlled metric-window
// management. It should not normally be called by production workers.
func (m *DeliveryMetrics) Reset() {
	if m == nil {
		return
	}

	m.recordsReceived.Store(0)
	m.recordsBatched.Store(0)
	m.batchesCreated.Store(0)

	m.batchesQueued.Store(0)
	m.batchesDequeued.Store(0)
	m.queueFull.Store(0)

	m.exportAttempts.Store(0)
	m.exportSuccess.Store(0)
	m.exportFailures.Store(0)

	m.recordsAccepted.Store(0)
	m.recordsRejected.Store(0)
	m.bytesDelivered.Store(0)

	m.retriesScheduled.Store(0)
	m.retriesExhausted.Store(0)
	m.retriesSuppressed.Store(0)

	m.acknowledgedBatches.Store(0)
	m.partialBatches.Store(0)
	m.failedBatches.Store(0)
	m.deadLetteredBatches.Store(0)

	m.spoolWrites.Store(0)
	m.spoolReads.Store(0)
	m.spoolDeletes.Store(0)
	m.spoolRecoveries.Store(0)
	m.spoolQuarantines.Store(0)
	m.spoolErrors.Store(0)

	m.workersStarted.Store(0)
	m.workersStopped.Store(0)
	m.activeWorkers.Store(0)
	m.activeDeliveries.Store(0)

	m.latencyMu.Lock()
	m.latency = latencyAccumulator{}
	m.latencyMu.Unlock()
}

// decrementAtomicNonNegative decreases a gauge while preventing it from
// becoming negative if a caller accidentally reports completion twice.
func decrementAtomicNonNegative(value *atomic.Int64) {
	if value == nil {
		return
	}

	for {
		current := value.Load()

		if current <= 0 {
			return
		}

		if value.CompareAndSwap(current, current-1) {
			return
		}
	}
}
