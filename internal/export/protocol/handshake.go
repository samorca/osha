package protocol

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Capability identifies an optional OTGH protocol capability.
//
// A peer must not send capability-dependent messages unless that capability
// was successfully negotiated during the handshake.
type Capability string

const (
	CapabilityTelemetry             Capability = "telemetry"
	CapabilityResourceState         Capability = "resource_state"
	CapabilityResourceRelationships Capability = "resource_relationships"
	CapabilityTopology              Capability = "topology"
	CapabilitySecurity              Capability = "security"
	CapabilityAnomalyDetection      Capability = "anomaly_detection"
	CapabilityForecasting           Capability = "forecasting"
	CapabilitySituations            Capability = "situations"
	CapabilityBlastRadius           Capability = "blast_radius"
	CapabilityDecisions             Capability = "decisions"
	CapabilityRemediationResults    Capability = "remediation_results"
	CapabilityAcknowledgements      Capability = "acknowledgements"
	CapabilityPartialAcknowledgement Capability = "partial_acknowledgement"
	CapabilitySequencing            Capability = "sequencing"
	CapabilityBatching              Capability = "batching"
)

// Valid reports whether the capability is recognised.
func (c Capability) Valid() bool {
	switch c {
	case CapabilityTelemetry,
		CapabilityResourceState,
		CapabilityResourceRelationships,
		CapabilityTopology,
		CapabilitySecurity,
		CapabilityAnomalyDetection,
		CapabilityForecasting,
		CapabilitySituations,
		CapabilityBlastRadius,
		CapabilityDecisions,
		CapabilityRemediationResults,
		CapabilityAcknowledgements,
		CapabilityPartialAcknowledgement,
		CapabilitySequencing,
		CapabilityBatching:
		return true

	default:
		return false
	}
}

// RequiredCapabilityForMessage returns the capability required to send a
// message type.
//
// Handshake, handshake result, acknowledgement, and protocol error messages
// are control-plane messages and do not require an optional capability.
func RequiredCapabilityForMessage(
	messageType MessageType,
) (Capability, bool) {
	switch messageType {
	case MessageTypeLog,
		MessageTypeMetric,
		MessageTypeTrace,
		MessageTypeEvent:
		return CapabilityTelemetry, true

	case MessageTypeResourceState:
		return CapabilityResourceState, true

	case MessageTypeResourceRelationship:
		return CapabilityResourceRelationships, true

	case MessageTypeTopologyChange:
		return CapabilityTopology, true

	case MessageTypeSecurityEvent,
		MessageTypeSecurityFinding,
		MessageTypeSecurityIncident:
		return CapabilitySecurity, true

	case MessageTypeAnomaly:
		return CapabilityAnomalyDetection, true

	case MessageTypeForecast:
		return CapabilityForecasting, true

	case MessageTypeSituation:
		return CapabilitySituations, true

	case MessageTypeBlastRadius:
		return CapabilityBlastRadius, true

	case MessageTypeDecision:
		return CapabilityDecisions, true

	case MessageTypeRemediationResult:
		return CapabilityRemediationResults, true

	case MessageTypeHandshake,
		MessageTypeHandshakeResult,
		MessageTypeAcknowledgement,
		MessageTypeError:
		return "", false

	default:
		return "", false
	}
}

// HandshakeRequest is sent by a peer before ordinary protocol messages.
//
// It advertises concrete versions, encodings, compression algorithms,
// capabilities, and receive limits supported by the initiating peer.
type HandshakeRequest struct {
	NodeID    string `json:"node_id"`
	SessionID string `json:"session_id"`

	AgentName    string `json:"agent_name,omitempty"`
	AgentVersion string `json:"agent_version,omitempty"`

	Versions     []Version     `json:"versions"`
	Encodings    []Encoding    `json:"encodings"`
	Compressions []Compression `json:"compressions"`
	Capabilities []Capability  `json:"capabilities"`

	// MaxFrameSize is the largest frame body the initiating peer can receive.
	MaxFrameSize uint64 `json:"max_frame_size"`

	// MaxPayloadSize is the largest logical payload the initiating peer can
	// process after decompression.
	MaxPayloadSize uint64 `json:"max_payload_size"`

	// MaxBatchRecords is the largest number of records the initiating peer is
	// willing to accept in one logical batch. Zero means batching is not
	// advertised.
	MaxBatchRecords uint32 `json:"max_batch_records,omitempty"`

	SentAt time.Time `json:"sent_at"`

	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate verifies a handshake request.
func (h HandshakeRequest) Validate() error {
	if strings.TrimSpace(h.NodeID) == "" {
		return errors.New("handshake node_id is required")
	}

	if strings.TrimSpace(h.SessionID) == "" {
		return errors.New("handshake session_id is required")
	}

	if h.SentAt.IsZero() {
		return errors.New("handshake sent_at is required")
	}

	if len(h.Versions) == 0 {
		return errors.New(
			"handshake must advertise at least one protocol version",
		)
	}

	if err := validateVersions(h.Versions); err != nil {
		return fmt.Errorf("handshake versions: %w", err)
	}

	if len(h.Encodings) == 0 {
		return errors.New(
			"handshake must advertise at least one encoding",
		)
	}

	if err := validateEncodings(h.Encodings); err != nil {
		return fmt.Errorf("handshake encodings: %w", err)
	}

	if len(h.Compressions) == 0 {
		return errors.New(
			"handshake must advertise at least one compression algorithm",
		)
	}

	if err := validateCompressions(h.Compressions); err != nil {
		return fmt.Errorf("handshake compressions: %w", err)
	}

	if err := validateCapabilities(h.Capabilities); err != nil {
		return fmt.Errorf("handshake capabilities: %w", err)
	}

	if h.MaxFrameSize == 0 {
		return errors.New(
			"handshake max_frame_size must be greater than zero",
		)
	}

	if h.MaxPayloadSize == 0 {
		return errors.New(
			"handshake max_payload_size must be greater than zero",
		)
	}

	if containsCapability(
		h.Capabilities,
		CapabilityBatching,
	) && h.MaxBatchRecords == 0 {
		return errors.New(
			"batching capability requires max_batch_records",
		)
	}

	return nil
}

// HandshakeStatus describes the outcome of protocol negotiation.
type HandshakeStatus string

const (
	HandshakeStatusAccepted HandshakeStatus = "accepted"
	HandshakeStatusRejected HandshakeStatus = "rejected"
)

// Valid reports whether the status is recognised.
func (s HandshakeStatus) Valid() bool {
	switch s {
	case HandshakeStatusAccepted,
		HandshakeStatusRejected:
		return true

	default:
		return false
	}
}

// HandshakeResult contains the negotiated session contract.
//
// When Status is accepted, the selected version, encoding, compression,
// capabilities, and limits define what both peers may use.
//
// When Status is rejected, Error contains the reason.
type HandshakeResult struct {
	Status HandshakeStatus `json:"status"`

	SessionID string `json:"session_id"`

	Version     Version     `json:"version,omitempty"`
	Encoding    Encoding    `json:"encoding,omitempty"`
	Compression Compression `json:"compression,omitempty"`

	Capabilities []Capability `json:"capabilities,omitempty"`

	MaxFrameSize   uint64 `json:"max_frame_size,omitempty"`
	MaxPayloadSize uint64 `json:"max_payload_size,omitempty"`

	MaxBatchRecords uint32 `json:"max_batch_records,omitempty"`

	AcceptedAt time.Time `json:"accepted_at,omitempty"`

	Error *ProtocolError `json:"error,omitempty"`

	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate verifies a handshake result.
func (r HandshakeResult) Validate() error {
	if !r.Status.Valid() {
		return fmt.Errorf(
			"invalid handshake status %q",
			r.Status,
		)
	}

	if strings.TrimSpace(r.SessionID) == "" {
		return errors.New(
			"handshake result session_id is required",
		)
	}

	switch r.Status {
	case HandshakeStatusAccepted:
		if err := r.Version.Validate(); err != nil {
			return fmt.Errorf(
				"handshake result version: %w",
				err,
			)
		}

		if !r.Encoding.Valid() {
			return fmt.Errorf(
				"handshake result has invalid encoding %q",
				r.Encoding,
			)
		}

		if !r.Compression.Valid() {
			return fmt.Errorf(
				"handshake result has invalid compression %q",
				r.Compression,
			)
		}

		if err := validateCapabilities(
			r.Capabilities,
		); err != nil {
			return fmt.Errorf(
				"handshake result capabilities: %w",
				err,
			)
		}

		if r.MaxFrameSize == 0 {
			return errors.New(
				"accepted handshake requires max_frame_size",
			)
		}

		if r.MaxPayloadSize == 0 {
			return errors.New(
				"accepted handshake requires max_payload_size",
			)
		}

		if containsCapability(
			r.Capabilities,
			CapabilityBatching,
		) && r.MaxBatchRecords == 0 {
			return errors.New(
				"negotiated batching requires max_batch_records",
			)
		}

		if r.AcceptedAt.IsZero() {
			return errors.New(
				"accepted handshake requires accepted_at",
			)
		}

		if r.Error != nil {
			return errors.New(
				"accepted handshake cannot contain an error",
			)
		}

	case HandshakeStatusRejected:
		if r.Error == nil {
			return errors.New(
				"rejected handshake requires an error",
			)
		}

		if err := r.Error.Validate(); err != nil {
			return fmt.Errorf(
				"rejected handshake error: %w",
				err,
			)
		}
	}

	return nil
}

// Accepted reports whether negotiation succeeded.
func (r HandshakeResult) Accepted() bool {
	return r.Status == HandshakeStatusAccepted
}

// HasCapability reports whether the session negotiated a capability.
func (r HandshakeResult) HasCapability(
	capability Capability,
) bool {
	return containsCapability(
		r.Capabilities,
		capability,
	)
}

// AllowsMessage reports whether the negotiated session permits a message
// type.
func (r HandshakeResult) AllowsMessage(
	messageType MessageType,
) bool {
	if !r.Accepted() || !messageType.Valid() {
		return false
	}

	required, requiredExists :=
		RequiredCapabilityForMessage(messageType)

	if !requiredExists {
		return true
	}

	return r.HasCapability(required)
}

// HandshakePolicy describes the local peer's protocol support and limits.
type HandshakePolicy struct {
	NodeID string

	Versions     []Version
	Encodings    []Encoding
	Compressions []Compression
	Capabilities []Capability

	MaxFrameSize   uint64
	MaxPayloadSize uint64

	MaxBatchRecords uint32

	// RequireCapabilities rejects peers that do not support all listed
	// capabilities.
	RequireCapabilities []Capability

	Metadata map[string]string
}

// Validate verifies local handshake policy.
func (p HandshakePolicy) Validate() error {
	if strings.TrimSpace(p.NodeID) == "" {
		return errors.New(
			"handshake policy node_id is required",
		)
	}

	if len(p.Versions) == 0 {
		return errors.New(
			"handshake policy requires protocol versions",
		)
	}

	if err := validateVersions(p.Versions); err != nil {
		return fmt.Errorf(
			"handshake policy versions: %w",
			err,
		)
	}

	if len(p.Encodings) == 0 {
		return errors.New(
			"handshake policy requires encodings",
		)
	}

	if err := validateEncodings(p.Encodings); err != nil {
		return fmt.Errorf(
			"handshake policy encodings: %w",
			err,
		)
	}

	if len(p.Compressions) == 0 {
		return errors.New(
			"handshake policy requires compression algorithms",
		)
	}

	if err := validateCompressions(
		p.Compressions,
	); err != nil {
		return fmt.Errorf(
			"handshake policy compressions: %w",
			err,
		)
	}

	if err := validateCapabilities(
		p.Capabilities,
	); err != nil {
		return fmt.Errorf(
			"handshake policy capabilities: %w",
			err,
		)
	}

	if err := validateCapabilities(
		p.RequireCapabilities,
	); err != nil {
		return fmt.Errorf(
			"handshake required capabilities: %w",
			err,
		)
	}

	for _, required := range p.RequireCapabilities {
		if !containsCapability(
			p.Capabilities,
			required,
		) {
			return fmt.Errorf(
				"required capability %q is not locally supported",
				required,
			)
		}
	}

	if p.MaxFrameSize == 0 {
		return errors.New(
			"handshake policy max_frame_size must be greater than zero",
		)
	}

	if p.MaxPayloadSize == 0 {
		return errors.New(
			"handshake policy max_payload_size must be greater than zero",
		)
	}

	if containsCapability(
		p.Capabilities,
		CapabilityBatching,
	) && p.MaxBatchRecords == 0 {
		return errors.New(
			"local batching capability requires max_batch_records",
		)
	}

	return nil
}

// DefaultHandshakePolicy creates a baseline policy for one OTGH node.
func DefaultHandshakePolicy(
	nodeID string,
) HandshakePolicy {
	return HandshakePolicy{
		NodeID: nodeID,

		Versions: SupportedVersions(),

		Encodings: []Encoding{
			EncodingJSON,
		},

		Compressions: []Compression{
			CompressionNone,
			CompressionGZIP,
		},

		Capabilities: []Capability{
			CapabilityTelemetry,
			CapabilityResourceState,
			CapabilityResourceRelationships,
			CapabilityTopology,
			CapabilitySecurity,
			CapabilityAnomalyDetection,
			CapabilityForecasting,
			CapabilitySituations,
			CapabilityBlastRadius,
			CapabilityDecisions,
			CapabilityRemediationResults,
			CapabilityAcknowledgements,
			CapabilityPartialAcknowledgement,
			CapabilitySequencing,
			CapabilityBatching,
		},

		MaxFrameSize:   DefaultMaxFrameSize,
		MaxPayloadSize: 32 * 1024 * 1024,

		MaxBatchRecords: 1_000,
	}
}

// HandshakeNegotiator selects a session contract between a remote request and
// local policy.
//
// It contains immutable policy and is safe for concurrent use.
type HandshakeNegotiator struct {
	policy HandshakePolicy
}

// NewHandshakeNegotiator creates a negotiator.
func NewHandshakeNegotiator(
	policy HandshakePolicy,
) (*HandshakeNegotiator, error) {
	if err := policy.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create handshake negotiator: %w",
			err,
		)
	}

	policy.Versions = cloneVersions(
		policy.Versions,
	)
	policy.Encodings = cloneEncodings(
		policy.Encodings,
	)
	policy.Compressions = cloneCompressions(
		policy.Compressions,
	)
	policy.Capabilities = cloneCapabilities(
		policy.Capabilities,
	)
	policy.RequireCapabilities = cloneCapabilities(
		policy.RequireCapabilities,
	)
	policy.Metadata = cloneStringMap(
		policy.Metadata,
	)

	return &HandshakeNegotiator{
		policy: policy,
	}, nil
}

// Policy returns a defensive copy of local handshake policy.
func (n *HandshakeNegotiator) Policy() HandshakePolicy {
	if n == nil {
		return HandshakePolicy{}
	}

	policy := n.policy

	policy.Versions = cloneVersions(
		n.policy.Versions,
	)
	policy.Encodings = cloneEncodings(
		n.policy.Encodings,
	)
	policy.Compressions = cloneCompressions(
		n.policy.Compressions,
	)
	policy.Capabilities = cloneCapabilities(
		n.policy.Capabilities,
	)
	policy.RequireCapabilities = cloneCapabilities(
		n.policy.RequireCapabilities,
	)
	policy.Metadata = cloneStringMap(
		n.policy.Metadata,
	)

	return policy
}

// Negotiate evaluates a remote handshake request.
//
// Negotiation selects:
//
//   - the highest exact protocol version shared by both peers;
//   - the local peer's highest-preference shared encoding;
//   - the local peer's highest-preference shared compression algorithm;
//   - the intersection of explicitly supported capabilities;
//   - the smaller receive limit advertised by either peer.
func (n *HandshakeNegotiator) Negotiate(
	request HandshakeRequest,
) HandshakeResult {
	if n == nil {
		return rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeInternal,
			"handshake negotiator is nil",
			false,
		)
	}

	if err := request.Validate(); err != nil {
		return rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeMalformedMessage,
			err.Error(),
			false,
		)
	}

	version, err := NegotiateVersion(
		n.policy.Versions,
		request.Versions,
	)
	if err != nil {
		return rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeUnsupportedVersion,
			err.Error(),
			false,
		)
	}

	encoding, found := selectEncoding(
		n.policy.Encodings,
		request.Encodings,
	)
	if !found {
		return rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeUnsupportedEncoding,
			"no shared protocol encoding",
			false,
		)
	}

	compression, found := selectCompression(
		n.policy.Compressions,
		request.Compressions,
	)
	if !found {
		return rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeUnsupportedCompression,
			"no shared compression algorithm",
			false,
		)
	}

	capabilities := intersectCapabilities(
		n.policy.Capabilities,
		request.Capabilities,
	)

	for _, required := range n.policy.RequireCapabilities {
		if !containsCapability(
			capabilities,
			required,
		) {
			return rejectedHandshakeResult(
				request.SessionID,
				ErrorCodeHandshakeRejected,
				fmt.Sprintf(
					"required capability %q is unavailable",
					required,
				),
				false,
			)
		}
	}

	maxFrameSize := minimumUint64(
		n.policy.MaxFrameSize,
		request.MaxFrameSize,
	)

	maxPayloadSize := minimumUint64(
		n.policy.MaxPayloadSize,
		request.MaxPayloadSize,
	)

	var maxBatchRecords uint32

	if containsCapability(
		capabilities,
		CapabilityBatching,
	) {
		maxBatchRecords = minimumUint32(
			n.policy.MaxBatchRecords,
			request.MaxBatchRecords,
		)
	}

	result := HandshakeResult{
		Status: HandshakeStatusAccepted,

		SessionID: request.SessionID,

		Version:     version,
		Encoding:    encoding,
		Compression: compression,

		Capabilities: capabilities,

		MaxFrameSize:   maxFrameSize,
		MaxPayloadSize: maxPayloadSize,

		MaxBatchRecords: maxBatchRecords,

		AcceptedAt: time.Now().UTC(),

		Metadata: cloneStringMap(
			n.policy.Metadata,
		),
	}

	if err := result.Validate(); err != nil {
		return rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeInternal,
			fmt.Sprintf(
				"negotiated result is invalid: %v",
				err,
			),
			false,
		)
	}

	return result
}

// NewHandshakeRequest constructs a request from local policy.
func NewHandshakeRequest(
	policy HandshakePolicy,
	sessionID string,
	agentName string,
	agentVersion string,
) (HandshakeRequest, error) {
	if err := policy.Validate(); err != nil {
		return HandshakeRequest{}, err
	}

	sessionID = strings.TrimSpace(sessionID)

	if sessionID == "" {
		return HandshakeRequest{}, errors.New(
			"handshake session_id is required",
		)
	}

	request := HandshakeRequest{
		NodeID:    policy.NodeID,
		SessionID: sessionID,

		AgentName: strings.TrimSpace(
			agentName,
		),
		AgentVersion: strings.TrimSpace(
			agentVersion,
		),

		Versions: cloneVersions(
			policy.Versions,
		),
		Encodings: cloneEncodings(
			policy.Encodings,
		),
		Compressions: cloneCompressions(
			policy.Compressions,
		),
		Capabilities: cloneCapabilities(
			policy.Capabilities,
		),

		MaxFrameSize:   policy.MaxFrameSize,
		MaxPayloadSize: policy.MaxPayloadSize,

		MaxBatchRecords: policy.MaxBatchRecords,

		SentAt: time.Now().UTC(),

		Metadata: cloneStringMap(
			policy.Metadata,
		),
	}

	if err := request.Validate(); err != nil {
		return HandshakeRequest{}, err
	}

	return request, nil
}

func rejectedHandshakeResult(
	sessionID string,
	code ErrorCode,
	message string,
	retryable bool,
) HandshakeResult {
	sessionID = strings.TrimSpace(
		sessionID,
	)

	if sessionID == "" {
		sessionID = "unknown"
	}

	return HandshakeResult{
		Status:    HandshakeStatusRejected,
		SessionID: sessionID,
		Error: NewProtocolError(
			code,
			message,
			retryable,
		),
	}
}

func selectEncoding(
	local []Encoding,
	remote []Encoding,
) (Encoding, bool) {
	for _, preferred := range local {
		for _, candidate := range remote {
			if preferred == candidate {
				return preferred, true
			}
		}
	}

	return "", false
}

func selectCompression(
	local []Compression,
	remote []Compression,
) (Compression, bool) {
	for _, preferred := range local {
		for _, candidate := range remote {
			if preferred == candidate {
				return preferred, true
			}
		}
	}

	return "", false
}

func intersectCapabilities(
	local []Capability,
	remote []Capability,
) []Capability {
	remoteSet := make(
		map[Capability]struct{},
		len(remote),
	)

	for _, capability := range remote {
		remoteSet[capability] = struct{}{}
	}

	result := make(
		[]Capability,
		0,
		len(local),
	)

	seen := make(
		map[Capability]struct{},
		len(local),
	)

	for _, capability := range local {
		if _, exists := remoteSet[capability]; !exists {
			continue
		}

		if _, duplicate := seen[capability]; duplicate {
			continue
		}

		seen[capability] = struct{}{}
		result = append(
			result,
			capability,
		)
	}

	return result
}

func containsCapability(
	capabilities []Capability,
	target Capability,
) bool {
	for _, capability := range capabilities {
		if capability == target {
			return true
		}
	}

	return false
}

func validateVersions(
	versions []Version,
) error {
	seen := make(
		map[Version]struct{},
		len(versions),
	)

	for i, version := range versions {
		if err := version.Validate(); err != nil {
			return fmt.Errorf(
				"version %d: %w",
				i,
				err,
			)
		}

		if _, exists := seen[version]; exists {
			return fmt.Errorf(
				"duplicate protocol version %s",
				version,
			)
		}

		seen[version] = struct{}{}
	}

	return nil
}

func validateEncodings(
	encodings []Encoding,
) error {
	seen := make(
		map[Encoding]struct{},
		len(encodings),
	)

	for i, encoding := range encodings {
		if !encoding.Valid() {
			return fmt.Errorf(
				"encoding %d is invalid: %q",
				i,
				encoding,
			)
		}

		if _, exists := seen[encoding]; exists {
			return fmt.Errorf(
				"duplicate encoding %q",
				encoding,
			)
		}

		seen[encoding] = struct{}{}
	}

	return nil
}

func validateCompressions(
	compressions []Compression,
) error {
	seen := make(
		map[Compression]struct{},
		len(compressions),
	)

	for i, compression := range compressions {
		if !compression.Valid() {
			return fmt.Errorf(
				"compression %d is invalid: %q",
				i,
				compression,
			)
		}

		if _, exists := seen[compression]; exists {
			return fmt.Errorf(
				"duplicate compression %q",
				compression,
			)
		}

		seen[compression] = struct{}{}
	}

	return nil
}

func validateCapabilities(
	capabilities []Capability,
) error {
	seen := make(
		map[Capability]struct{},
		len(capabilities),
	)

	for i, capability := range capabilities {
		if !capability.Valid() {
			return fmt.Errorf(
				"capability %d is invalid: %q",
				i,
				capability,
			)
		}

		if _, exists := seen[capability]; exists {
			return fmt.Errorf(
				"duplicate capability %q",
				capability,
			)
		}

		seen[capability] = struct{}{}
	}

	return nil
}

// SortCapabilities returns a sorted defensive copy.
//
// Negotiation itself preserves local preference order. Sorting is provided
// for deterministic diagnostics and testing.
func SortCapabilities(
	capabilities []Capability,
) []Capability {
	sorted := cloneCapabilities(
		capabilities,
	)

	sort.Slice(
		sorted,
		func(i, j int) bool {
			return sorted[i] < sorted[j]
		},
	)

	return sorted
}

func cloneVersions(
	input []Version,
) []Version {
	if input == nil {
		return nil
	}

	return append(
		[]Version(nil),
		input...,
	)
}

func cloneEncodings(
	input []Encoding,
) []Encoding {
	if input == nil {
		return nil
	}

	return append(
		[]Encoding(nil),
		input...,
	)
}

func cloneCompressions(
	input []Compression,
) []Compression {
	if input == nil {
		return nil
	}

	return append(
		[]Compression(nil),
		input...,
	)
}

func cloneCapabilities(
	input []Capability,
) []Capability {
	if input == nil {
		return nil
	}

	return append(
		[]Capability(nil),
		input...,
	)
}

func minimumUint64(
	left uint64,
	right uint64,
) uint64 {
	if left < right {
		return left
	}

	return right
}

func minimumUint32(
	left uint32,
	right uint32,
) uint32 {
	if left < right {
		return left
	}

	return right
}
