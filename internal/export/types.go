package export

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// RecordID uniquely identifies a record inside OTGH.
//
// It is intentionally represented as a string so OTGH does not force
// producers to use a particular UUID/ULID library or identifier format.
type RecordID string

// BatchID uniquely identifies a delivery batch.
type BatchID string

// DestinationID uniquely identifies a configured export destination.
type DestinationID string

// CorrelationID connects records that belong to the same situation,
// incident, trace, topology event, or other logical operation.
type CorrelationID string

// TenantID identifies the tenant or organisation that owns a record.
//
// An empty TenantID is allowed for deployments that do not use
// multi-tenancy.
type TenantID string

// RecordType describes the semantic class of data being transported.
type RecordType string

const (
	RecordTypeUnknown RecordType = "unknown"

	// Traditional telemetry.
	RecordTypeLog    RecordType = "log"
	RecordTypeMetric RecordType = "metric"
	RecordTypeTrace  RecordType = "trace"
	RecordTypeEvent  RecordType = "event"

	// Infrastructure state.
	RecordTypeResourceState        RecordType = "resource_state"
	RecordTypeResourceRelationship RecordType = "resource_relationship"
	RecordTypeTopologyChange       RecordType = "topology_change"

	// Security.
	RecordTypeSecurityEvent   RecordType = "security_event"
	RecordTypeSecurityFinding RecordType = "security_finding"
	RecordTypeSecurityIncident RecordType = "security_incident"

	// Intelligence.
	RecordTypeAnomaly     RecordType = "anomaly"
	RecordTypeForecast    RecordType = "forecast"
	RecordTypeSituation   RecordType = "situation"
	RecordTypeBlastRadius RecordType = "blast_radius"

	// Control-plane outcomes.
	RecordTypeDecision          RecordType = "decision"
	RecordTypeRemediationResult RecordType = "remediation_result"
)

// Valid reports whether the record type is recognised by this version of OTGH.
func (t RecordType) Valid() bool {
	switch t {
	case RecordTypeLog,
		RecordTypeMetric,
		RecordTypeTrace,
		RecordTypeEvent,
		RecordTypeResourceState,
		RecordTypeResourceRelationship,
		RecordTypeTopologyChange,
		RecordTypeSecurityEvent,
		RecordTypeSecurityFinding,
		RecordTypeSecurityIncident,
		RecordTypeAnomaly,
		RecordTypeForecast,
		RecordTypeSituation,
		RecordTypeBlastRadius,
		RecordTypeDecision,
		RecordTypeRemediationResult:
		return true
	default:
		return false
	}
}

// Priority determines how urgently a record should be delivered.
//
// Priority does not itself grant permission to perform an infrastructure
// action. Authorisation remains the responsibility of OSHA's policy and
// control planes.
type Priority uint8

const (
	PriorityBulk Priority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
	PriorityControl
)

func (p Priority) String() string {
	switch p {
	case PriorityBulk:
		return "bulk"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	case PriorityControl:
		return "control"
	default:
		return "unknown"
	}
}

// Valid reports whether the priority is recognised.
func (p Priority) Valid() bool {
	return p <= PriorityControl
}

// DeliveryStatus represents the current delivery lifecycle state.
type DeliveryStatus uint8

const (
	DeliveryPending DeliveryStatus = iota
	DeliveryInFlight
	DeliveryAcknowledged
	DeliveryRetrying
	DeliveryFailed
	DeliveryDeadLettered
)

func (s DeliveryStatus) String() string {
	switch s {
	case DeliveryPending:
		return "pending"
	case DeliveryInFlight:
		return "in_flight"
	case DeliveryAcknowledged:
		return "acknowledged"
	case DeliveryRetrying:
		return "retrying"
	case DeliveryFailed:
		return "failed"
	case DeliveryDeadLettered:
		return "dead_lettered"
	default:
		return "unknown"
	}
}

// Valid reports whether the delivery status is recognised.
func (s DeliveryStatus) Valid() bool {
	return s <= DeliveryDeadLettered
}

// ContentType describes how Payload is encoded.
//
// These are deliberately generic transport-level content types.
// Additional encodings can be introduced without changing RecordType.
type ContentType string

const (
	ContentTypeJSON       ContentType = "application/json"
	ContentTypeProtobuf   ContentType = "application/x-protobuf"
	ContentTypeOctetStream ContentType = "application/octet-stream"
	ContentTypeText       ContentType = "text/plain"
)

// Compression describes payload compression.
type Compression string

const (
	CompressionNone Compression = "none"
	CompressionGZIP Compression = "gzip"
)

// Source identifies the system and component that produced a record.
type Source struct {
	System    string `json:"system,omitempty"`
	Component string `json:"component,omitempty"`
	Instance  string `json:"instance,omitempty"`
	Region    string `json:"region,omitempty"`
	Cluster   string `json:"cluster,omitempty"`
	Node      string `json:"node,omitempty"`
}

// ResourceRef provides lightweight infrastructure identity.
//
// The complete infrastructure resource model belongs to OSHA. OTGH only
// carries enough identity to route, correlate, and deliver the record.
type ResourceRef struct {
	ID          string            `json:"id,omitempty"`
	Type        string            `json:"type,omitempty"`
	Name        string            `json:"name,omitempty"`
	Provider    string            `json:"provider,omitempty"`
	Region      string            `json:"region,omitempty"`
	Cluster     string            `json:"cluster,omitempty"`
	Namespace   string            `json:"namespace,omitempty"`
	Environment string            `json:"environment,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// Destination identifies where a record or batch should be delivered.
type Destination struct {
	ID       DestinationID     `json:"id"`
	Type     string            `json:"type"`
	Endpoint string            `json:"endpoint,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate checks the minimum destination contract.
//
// Exporter-specific validation, such as URL parsing or TLS requirements,
// belongs in the relevant exporter package.
func (d Destination) Validate() error {
	if strings.TrimSpace(string(d.ID)) == "" {
		return errors.New("destination id is required")
	}

	if strings.TrimSpace(d.Type) == "" {
		return errors.New("destination type is required")
	}

	return nil
}

// Record is OTGH's canonical unit of transport.
//
// Payload is kept as bytes so the export subsystem does not need to know
// the internal Go/Python model that produced the information.
//
// Metadata should contain routing and delivery metadata only. Large
// telemetry payloads belong in Payload.
type Record struct {
	ID            RecordID          `json:"id"`
	TenantID      TenantID          `json:"tenant_id,omitempty"`
	CorrelationID CorrelationID     `json:"correlation_id,omitempty"`

	Type     RecordType `json:"type"`
	Priority Priority   `json:"priority"`

	Source   Source       `json:"source"`
	Resource *ResourceRef `json:"resource,omitempty"`

	ContentType ContentType `json:"content_type"`
	Compression Compression `json:"compression"`

	CreatedAt time.Time `json:"created_at"`

	Payload  []byte            `json:"payload"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate checks whether the record satisfies OTGH's transport contract.
func (r Record) Validate() error {
	if strings.TrimSpace(string(r.ID)) == "" {
		return errors.New("record id is required")
	}

	if !r.Type.Valid() {
		return fmt.Errorf("invalid record type %q", r.Type)
	}

	if !r.Priority.Valid() {
		return fmt.Errorf("invalid priority %d", r.Priority)
	}

	if r.CreatedAt.IsZero() {
		return errors.New("record created_at is required")
	}

	if len(r.Payload) == 0 {
		return errors.New("record payload must not be empty")
	}

	switch r.ContentType {
	case ContentTypeJSON,
		ContentTypeProtobuf,
		ContentTypeOctetStream,
		ContentTypeText:
		// Valid.
	default:
		return fmt.Errorf("unsupported content type %q", r.ContentType)
	}

	switch r.Compression {
	case CompressionNone, CompressionGZIP:
		// Valid.
	default:
		return fmt.Errorf("unsupported compression %q", r.Compression)
	}

	return nil
}

// Size returns the payload size in bytes.
//
// Transport framing, metadata, compression overhead, and protocol headers
// are deliberately not included.
func (r Record) Size() int {
	return len(r.Payload)
}

// Batch groups records for delivery to a destination.
//
// Records belonging to different destinations should normally be separated
// by the router/batcher rather than combined into one Batch.
type Batch struct {
	ID          BatchID     `json:"id"`
	Destination Destination `json:"destination"`

	Records []Record `json:"records"`

	CreatedAt time.Time `json:"created_at"`
}

// Validate verifies the batch and all records inside it.
func (b Batch) Validate() error {
	if strings.TrimSpace(string(b.ID)) == "" {
		return errors.New("batch id is required")
	}

	if err := b.Destination.Validate(); err != nil {
		return fmt.Errorf("invalid batch destination: %w", err)
	}

	if b.CreatedAt.IsZero() {
		return errors.New("batch created_at is required")
	}

	if len(b.Records) == 0 {
		return errors.New("batch must contain at least one record")
	}

	for i := range b.Records {
		if err := b.Records[i].Validate(); err != nil {
			return fmt.Errorf("record %d: %w", i, err)
		}
	}

	return nil
}

// RecordCount returns the number of records in the batch.
func (b Batch) RecordCount() int {
	return len(b.Records)
}

// PayloadSize returns the combined unframed payload size of all records.
func (b Batch) PayloadSize() int64 {
	var total int64

	for i := range b.Records {
		total += int64(len(b.Records[i].Payload))
	}

	return total
}

// DeliveryResult describes the outcome of one export attempt.
//
// It is deliberately protocol-neutral. HTTP, gRPC, object storage, and
// future OTGH-native transports translate their responses into this type.
type DeliveryResult struct {
	BatchID       BatchID        `json:"batch_id"`
	DestinationID DestinationID  `json:"destination_id"`
	Status        DeliveryStatus `json:"status"`

	Attempt int `json:"attempt"`

	AcceptedRecords int `json:"accepted_records"`
	RejectedRecords int `json:"rejected_records"`

	Retryable bool `json:"retryable"`

	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`

	ErrorCode    string `json:"error_code,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`

	Metadata map[string]string `json:"metadata,omitempty"`
}

// Successful reports whether the complete batch was acknowledged.
func (r DeliveryResult) Successful() bool {
	return r.Status == DeliveryAcknowledged &&
		r.RejectedRecords == 0 &&
		r.ErrorCode == "" &&
		r.ErrorMessage == ""
}

// Duration returns the duration of the delivery attempt.
//
// A zero duration is returned when the timestamps cannot represent a
// completed interval.
func (r DeliveryResult) Duration() time.Duration {
	if r.StartedAt.IsZero() ||
		r.FinishedAt.IsZero() ||
		r.FinishedAt.Before(r.StartedAt) {
		return 0
	}

	return r.FinishedAt.Sub(r.StartedAt)
}