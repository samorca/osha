package protocol

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"
)

// MessageType identifies the semantic payload carried by an Envelope.
//
// OTGH deliberately supports infrastructure-intelligence objects in addition
// to conventional telemetry.
type MessageType string

const (
	MessageTypeUnknown MessageType = "unknown"

	MessageTypeLog    MessageType = "log"
	MessageTypeMetric MessageType = "metric"
	MessageTypeTrace  MessageType = "trace"
	MessageTypeEvent  MessageType = "event"

	MessageTypeResourceState        MessageType = "resource_state"
	MessageTypeResourceRelationship MessageType = "resource_relationship"
	MessageTypeTopologyChange       MessageType = "topology_change"

	MessageTypeSecurityEvent    MessageType = "security_event"
	MessageTypeSecurityFinding  MessageType = "security_finding"
	MessageTypeSecurityIncident MessageType = "security_incident"

	MessageTypeAnomaly     MessageType = "anomaly"
	MessageTypeForecast    MessageType = "forecast"
	MessageTypeSituation   MessageType = "situation"
	MessageTypeBlastRadius MessageType = "blast_radius"

	MessageTypeDecision          MessageType = "decision"
	MessageTypeRemediationResult MessageType = "remediation_result"

	MessageTypeAcknowledgement MessageType = "acknowledgement"
	MessageTypeError           MessageType = "error"

	MessageTypeHandshake        MessageType = "handshake"
	MessageTypeHandshakeResult  MessageType = "handshake_result"
)

// Valid reports whether the message type is recognised.
func (t MessageType) Valid() bool {
	switch t {
	case MessageTypeLog,
		MessageTypeMetric,
		MessageTypeTrace,
		MessageTypeEvent,
		MessageTypeResourceState,
		MessageTypeResourceRelationship,
		MessageTypeTopologyChange,
		MessageTypeSecurityEvent,
		MessageTypeSecurityFinding,
		MessageTypeSecurityIncident,
		MessageTypeAnomaly,
		MessageTypeForecast,
		MessageTypeSituation,
		MessageTypeBlastRadius,
		MessageTypeDecision,
		MessageTypeRemediationResult,
		MessageTypeAcknowledgement,
		MessageTypeError,
		MessageTypeHandshake,
		MessageTypeHandshakeResult:
		return true

	default:
		return false
	}
}

// MessagePriority controls protocol-level delivery importance.
//
// Priority does not grant permission to carry out infrastructure changes.
// OSHA's policy/control layer remains responsible for authorization.
type MessagePriority uint8

const (
	PriorityBulk MessagePriority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
	PriorityControl
)

// String returns a stable textual priority representation.
func (p MessagePriority) String() string {
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
func (p MessagePriority) Valid() bool {
	return p <= PriorityControl
}

// Encoding identifies the payload representation inside an envelope.
type Encoding string

const (
	EncodingJSON   Encoding = "json"
	EncodingBinary Encoding = "binary"
)

// Valid reports whether the encoding is supported at the envelope-model
// level.
//
// Actual encoder support is implemented by encoder.go and decoder.go.
func (e Encoding) Valid() bool {
	switch e {
	case EncodingJSON,
		EncodingBinary:
		return true

	default:
		return false
	}
}

// Compression identifies compression applied to Payload.
type Compression string

const (
	CompressionNone Compression = "none"
	CompressionGZIP Compression = "gzip"
)

// Valid reports whether the compression identifier is recognised.
func (c Compression) Valid() bool {
	switch c {
	case CompressionNone,
		CompressionGZIP:
		return true

	default:
		return false
	}
}

// Resource describes the infrastructure context associated with a protocol
// message.
//
// This is not intended to replace OSHA's complete resource model. It contains
// the portable identity information needed by a receiving OTGH/OSHA system.
type Resource struct {
	ID   string `json:"id,omitempty"`
	Type string `json:"type,omitempty"`
	Name string `json:"name,omitempty"`

	Provider    string `json:"provider,omitempty"`
	Region      string `json:"region,omitempty"`
	Environment string `json:"environment,omitempty"`

	Cluster   string `json:"cluster,omitempty"`
	Namespace string `json:"namespace,omitempty"`
	Node      string `json:"node,omitempty"`

	ParentID string `json:"parent_id,omitempty"`

	Labels map[string]string `json:"labels,omitempty"`
}

// Empty reports whether the Resource contains no identity or contextual
// information.
func (r Resource) Empty() bool {
	return r.ID == "" &&
		r.Type == "" &&
		r.Name == "" &&
		r.Provider == "" &&
		r.Region == "" &&
		r.Environment == "" &&
		r.Cluster == "" &&
		r.Namespace == "" &&
		r.Node == "" &&
		r.ParentID == "" &&
		len(r.Labels) == 0
}

// Source identifies the software component originating the message.
type Source struct {
	System    string `json:"system,omitempty"`
	Component string `json:"component,omitempty"`
	Instance  string `json:"instance,omitempty"`

	Region  string `json:"region,omitempty"`
	Cluster string `json:"cluster,omitempty"`
	Node    string `json:"node,omitempty"`
}

// Validate verifies the minimum source identity.
func (s Source) Validate() error {
	if strings.TrimSpace(s.System) == "" {
		return errors.New(
			"protocol source system is required",
		)
	}

	return nil
}

// Integrity contains payload integrity metadata.
//
// SHA-256 here detects corruption or accidental payload modification. It is
// not a sender-authentication mechanism. Authentication/signatures, if
// introduced, must be handled separately.
type Integrity struct {
	Algorithm string `json:"algorithm"`
	Digest    string `json:"digest"`
}

const IntegrityAlgorithmSHA256 = "sha256"

// Validate verifies integrity metadata syntax.
func (i Integrity) Validate() error {
	if i.Algorithm != IntegrityAlgorithmSHA256 {
		return fmt.Errorf(
			"%w: unsupported integrity algorithm %q",
			ErrInvalidChecksum,
			i.Algorithm,
		)
	}

	if len(i.Digest) != sha256.Size*2 {
		return fmt.Errorf(
			"%w: invalid SHA-256 digest length",
			ErrInvalidChecksum,
		)
	}

	decoded, err := hex.DecodeString(i.Digest)
	if err != nil {
		return fmt.Errorf(
			"%w: malformed digest: %v",
			ErrInvalidChecksum,
			err,
		)
	}

	if len(decoded) != sha256.Size {
		return fmt.Errorf(
			"%w: invalid SHA-256 digest",
			ErrInvalidChecksum,
		)
	}

	return nil
}

// Envelope is the canonical OTGH protocol message.
//
// An Envelope carries exactly one logical payload together with enough
// metadata for:
//
//   - protocol-version validation;
//   - routing;
//   - sequencing;
//   - correlation;
//   - infrastructure context;
//   - encoding/compression interpretation;
//   - payload integrity verification.
//
// Transport framing is deliberately separate and belongs to frame.go.
type Envelope struct {
	Version Version `json:"version"`

	MessageID string `json:"message_id"`

	MessageType MessageType     `json:"message_type"`
	Priority    MessagePriority `json:"priority"`

	TenantID      string `json:"tenant_id,omitempty"`
	CorrelationID string `json:"correlation_id,omitempty"`

	// Sequence is monotonic within the session/stream that owns the message.
	//
	// Sequence zero is reserved for messages that do not participate in
	// ordered stream delivery, such as some handshake messages.
	Sequence uint64 `json:"sequence,omitempty"`

	CreatedAt time.Time `json:"created_at"`

	Source Source `json:"source"`

	Resource *Resource `json:"resource,omitempty"`

	Encoding    Encoding    `json:"encoding"`
	Compression Compression `json:"compression"`

	// Payload contains the encoded, potentially compressed message body.
	Payload []byte `json:"payload"`

	Integrity Integrity `json:"integrity"`

	// Metadata contains small routing/protocol attributes.
	//
	// Large application data belongs in Payload.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate verifies envelope structure and payload integrity.
func (e Envelope) Validate() error {
	if err := e.ValidateStructure(); err != nil {
		return err
	}

	if err := e.VerifyIntegrity(); err != nil {
		return err
	}

	return nil
}

// ValidateStructure validates the envelope without verifying the payload
// digest.
//
// This separation is useful for decoders that need to validate framing and
// metadata before performing additional payload work.
func (e Envelope) ValidateStructure() error {
	if err := e.Version.Validate(); err != nil {
		return fmt.Errorf(
			"%w: version: %v",
			ErrInvalidEnvelope,
			err,
		)
	}

	if !e.Version.SupportedByCurrent() {
		return fmt.Errorf(
			"%w: version %s",
			ErrIncompatibleVersion,
			e.Version,
		)
	}

	if strings.TrimSpace(e.MessageID) == "" {
		return fmt.Errorf(
			"%w: message_id is required",
			ErrInvalidEnvelope,
		)
	}

	if !e.MessageType.Valid() {
		return fmt.Errorf(
			"%w: invalid message type %q",
			ErrInvalidEnvelope,
			e.MessageType,
		)
	}

	if !e.Priority.Valid() {
		return fmt.Errorf(
			"%w: invalid priority %d",
			ErrInvalidEnvelope,
			e.Priority,
		)
	}

	if e.CreatedAt.IsZero() {
		return fmt.Errorf(
			"%w: created_at is required",
			ErrInvalidEnvelope,
		)
	}

	if err := e.Source.Validate(); err != nil {
		return fmt.Errorf(
			"%w: %v",
			ErrInvalidEnvelope,
			err,
		)
	}

	if !e.Encoding.Valid() {
		return fmt.Errorf(
			"%w: %w: %q",
			ErrInvalidEnvelope,
			ErrUnsupportedEncoding,
			e.Encoding,
		)
	}

	if !e.Compression.Valid() {
		return fmt.Errorf(
			"%w: %w: %q",
			ErrInvalidEnvelope,
			ErrUnsupportedCompression,
			e.Compression,
		)
	}

	if len(e.Payload) == 0 {
		return fmt.Errorf(
			"%w: payload is empty",
			ErrInvalidEnvelope,
		)
	}

	if err := e.Integrity.Validate(); err != nil {
		return fmt.Errorf(
			"%w: integrity: %v",
			ErrInvalidEnvelope,
			err,
		)
	}

	return nil
}

// VerifyIntegrity recalculates the payload digest and compares it with the
// envelope's integrity metadata.
func (e Envelope) VerifyIntegrity() error {
	if err := e.Integrity.Validate(); err != nil {
		return err
	}

	expected := sha256.Sum256(e.Payload)

	expectedDigest := hex.EncodeToString(expected[:])

	if !strings.EqualFold(
		e.Integrity.Digest,
		expectedDigest,
	) {
		return fmt.Errorf(
			"%w: payload digest mismatch",
			ErrInvalidChecksum,
		)
	}

	return nil
}

// PayloadSize returns the encoded/compressed payload size in bytes.
func (e Envelope) PayloadSize() int {
	return len(e.Payload)
}

// Clone returns a defensive copy of the envelope.
func (e Envelope) Clone() Envelope {
	cloned := e

	if e.Payload != nil {
		cloned.Payload = append(
			[]byte(nil),
			e.Payload...,
		)
	}

	if e.Metadata != nil {
		cloned.Metadata = cloneStringMap(
			e.Metadata,
		)
	}

	if e.Resource != nil {
		resource := *e.Resource

		if e.Resource.Labels != nil {
			resource.Labels = cloneStringMap(
				e.Resource.Labels,
			)
		}

		cloned.Resource = &resource
	}

	return cloned
}

// NewEnvelope constructs a canonical envelope and calculates payload
// integrity metadata.
//
// Compression is descriptive at this layer. The payload provided here must
// already be in the representation indicated by Encoding and Compression.
func NewEnvelope(
	version Version,
	messageID string,
	messageType MessageType,
	priority MessagePriority,
	source Source,
	encoding Encoding,
	compression Compression,
	payload []byte,
) (Envelope, error) {
	messageID = strings.TrimSpace(messageID)

	if messageID == "" {
		return Envelope{}, fmt.Errorf(
			"%w: message id is required",
			ErrInvalidEnvelope,
		)
	}

	if len(payload) == 0 {
		return Envelope{}, fmt.Errorf(
			"%w: payload is empty",
			ErrInvalidPayload,
		)
	}

	payloadCopy := append(
		[]byte(nil),
		payload...,
	)

	digest := sha256.Sum256(payloadCopy)

	envelope := Envelope{
		Version:     version,
		MessageID:   messageID,
		MessageType: messageType,
		Priority:    priority,
		CreatedAt:   time.Now().UTC(),
		Source:      source,
		Encoding:    encoding,
		Compression: compression,
		Payload:     payloadCopy,
		Integrity: Integrity{
			Algorithm: IntegrityAlgorithmSHA256,
			Digest:    hex.EncodeToString(digest[:]),
		},
	}

	if err := envelope.Validate(); err != nil {
		return Envelope{}, err
	}

	return envelope, nil
}

// SetPayload replaces the payload and recalculates integrity metadata.
//
// The payload is defensively copied.
func (e *Envelope) SetPayload(
	payload []byte,
) error {
	if e == nil {
		return errors.New("envelope is nil")
	}

	if len(payload) == 0 {
		return fmt.Errorf(
			"%w: payload is empty",
			ErrInvalidPayload,
		)
	}

	e.Payload = append(
		e.Payload[:0],
		payload...,
	)

	digest := sha256.Sum256(e.Payload)

	e.Integrity = Integrity{
		Algorithm: IntegrityAlgorithmSHA256,
		Digest:    hex.EncodeToString(digest[:]),
	}

	return nil
}

// SetResource assigns a defensive copy of infrastructure context.
func (e *Envelope) SetResource(
	resource Resource,
) error {
	if e == nil {
		return errors.New("envelope is nil")
	}

	if resource.Empty() {
		e.Resource = nil
		return nil
	}

	copy := resource

	if resource.Labels != nil {
		copy.Labels = cloneStringMap(
			resource.Labels,
		)
	}

	e.Resource = &copy

	return nil
}

// SetMetadata replaces metadata with a defensive copy.
func (e *Envelope) SetMetadata(
	metadata map[string]string,
) error {
	if e == nil {
		return errors.New("envelope is nil")
	}

	e.Metadata = cloneStringMap(metadata)

	return nil
}
