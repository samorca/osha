package protocol

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ProtocolState represents the lifecycle of one OTGH protocol session.
type ProtocolState uint8

const (
	ProtocolStateNew ProtocolState = iota
	ProtocolStateHandshaking
	ProtocolStateReady
	ProtocolStateClosing
	ProtocolStateClosed
)

// String returns a stable state representation.
func (s ProtocolState) String() string {
	switch s {
	case ProtocolStateNew:
		return "new"

	case ProtocolStateHandshaking:
		return "handshaking"

	case ProtocolStateReady:
		return "ready"

	case ProtocolStateClosing:
		return "closing"

	case ProtocolStateClosed:
		return "closed"

	default:
		return "unknown"
	}
}

// Valid reports whether the state is recognised.
func (s ProtocolState) Valid() bool {
	return s <= ProtocolStateClosed
}

// ProtocolConfig controls one OTGH protocol runtime.
type ProtocolConfig struct {
	NodeID string

	Handshake HandshakePolicy

	Encoder EncoderConfig
	Decoder DecoderConfig

	Compression CompressionConfig
}

// DefaultProtocolConfig creates a baseline configuration for an OTGH node.
func DefaultProtocolConfig(
	nodeID string,
) ProtocolConfig {
	return ProtocolConfig{
		NodeID: strings.TrimSpace(nodeID),

		Handshake: DefaultHandshakePolicy(
			nodeID,
		),

		Encoder: DefaultEncoderConfig(),
		Decoder: DefaultDecoderConfig(),

		Compression: DefaultCompressionConfig(),
	}
}

// Validate verifies the complete protocol configuration.
func (c ProtocolConfig) Validate() error {
	nodeID := strings.TrimSpace(c.NodeID)

	if nodeID == "" {
		return errors.New(
			"protocol node_id is required",
		)
	}

	if err := c.Handshake.Validate(); err != nil {
		return fmt.Errorf(
			"protocol handshake configuration: %w",
			err,
		)
	}

	if strings.TrimSpace(c.Handshake.NodeID) !=
		nodeID {
		return fmt.Errorf(
			"protocol node_id %q does not match handshake node_id %q",
			nodeID,
			c.Handshake.NodeID,
		)
	}

	if err := c.Encoder.Validate(); err != nil {
		return fmt.Errorf(
			"protocol encoder configuration: %w",
			err,
		)
	}

	if err := c.Decoder.Validate(); err != nil {
		return fmt.Errorf(
			"protocol decoder configuration: %w",
			err,
		)
	}

	if err := c.Compression.Validate(); err != nil {
		return fmt.Errorf(
			"protocol compression configuration: %w",
			err,
		)
	}

	if c.Encoder.MaxFrameSize >
		c.Handshake.MaxFrameSize {
		return errors.New(
			"encoder max frame size exceeds locally advertised handshake limit",
		)
	}

	if c.Decoder.MaxFrameSize >
		c.Handshake.MaxFrameSize {
		return errors.New(
			"decoder max frame size exceeds locally advertised handshake limit",
		)
	}

	if c.Encoder.MaxPayloadSize >
		c.Handshake.MaxPayloadSize {
		return errors.New(
			"encoder max payload size exceeds locally advertised handshake limit",
		)
	}

	if c.Decoder.MaxPayloadSize >
		c.Handshake.MaxPayloadSize {
		return errors.New(
			"decoder max payload size exceeds locally advertised handshake limit",
		)
	}

	return nil
}

// Session describes the negotiated state of one peer connection.
//
// Session contains protocol state only. sender.go and receiver.go will own
// transport I/O.
type Session struct {
	mu sync.RWMutex

	localNodeID  string
	remoteNodeID string
	sessionID    string

	state ProtocolState

	negotiated HandshakeResult

	createdAt time.Time
	readyAt   time.Time
	closedAt  time.Time

	closeReason string

	// sendSequence is the last sequence allocated for an outgoing ordered
	// message.
	sendSequence atomic.Uint64

	// receiveSequence is protected by mu because receiving needs compound
	// validation and update semantics.
	receiveSequence uint64
}

// NewSession constructs a new un-negotiated protocol session.
func NewSession(
	localNodeID string,
	sessionID string,
) (*Session, error) {
	localNodeID = strings.TrimSpace(
		localNodeID,
	)

	sessionID = strings.TrimSpace(
		sessionID,
	)

	if localNodeID == "" {
		return nil, errors.New(
			"session local node_id is required",
		)
	}

	if sessionID == "" {
		return nil, errors.New(
			"session id is required",
		)
	}

	return &Session{
		localNodeID: localNodeID,
		sessionID:   sessionID,
		state:       ProtocolStateNew,
		createdAt:   time.Now().UTC(),
	}, nil
}

// ID returns the session identifier.
func (s *Session) ID() string {
	if s == nil {
		return ""
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.sessionID
}

// LocalNodeID returns the local OTGH node identity.
func (s *Session) LocalNodeID() string {
	if s == nil {
		return ""
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.localNodeID
}

// RemoteNodeID returns the peer identity once known.
func (s *Session) RemoteNodeID() string {
	if s == nil {
		return ""
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.remoteNodeID
}

// SetRemoteNodeID records the remote peer identity.
func (s *Session) SetRemoteNodeID(
	nodeID string,
) error {
	if s == nil {
		return errors.New("session is nil")
	}

	nodeID = strings.TrimSpace(nodeID)

	if nodeID == "" {
		return errors.New(
			"remote node_id is required",
		)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == ProtocolStateClosed {
		return ErrSessionClosed
	}

	if s.remoteNodeID != "" &&
		s.remoteNodeID != nodeID {
		return fmt.Errorf(
			"%w: remote node changed from %q to %q",
			ErrProtocol,
			s.remoteNodeID,
			nodeID,
		)
	}

	s.remoteNodeID = nodeID

	return nil
}

// State returns the current session lifecycle state.
func (s *Session) State() ProtocolState {
	if s == nil {
		return ProtocolStateClosed
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.state
}

// BeginHandshake transitions a new session into handshaking state.
func (s *Session) BeginHandshake() error {
	if s == nil {
		return errors.New("session is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	switch s.state {
	case ProtocolStateNew:
		s.state = ProtocolStateHandshaking
		return nil

	case ProtocolStateHandshaking:
		return nil

	case ProtocolStateReady:
		return errors.New(
			"session is already negotiated",
		)

	case ProtocolStateClosing,
		ProtocolStateClosed:
		return ErrSessionClosed

	default:
		return fmt.Errorf(
			"%w: invalid session state %d",
			ErrProtocol,
			s.state,
		)
	}
}

// ApplyHandshake records a successful negotiated session contract.
func (s *Session) ApplyHandshake(
	result HandshakeResult,
) error {
	if s == nil {
		return errors.New("session is nil")
	}

	if err := result.Validate(); err != nil {
		return fmt.Errorf(
			"apply handshake: %w",
			err,
		)
	}

	if !result.Accepted() {
		return fmt.Errorf(
			"%w: handshake was rejected",
			ErrHandshakeRejected,
		)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != ProtocolStateHandshaking &&
		s.state != ProtocolStateNew {
		return fmt.Errorf(
			"%w: cannot apply handshake while session is %s",
			ErrProtocol,
			s.state,
		)
	}

	if result.SessionID != s.sessionID {
		return fmt.Errorf(
			"%w: handshake session %q does not match local session %q",
			ErrHandshakeRejected,
			result.SessionID,
			s.sessionID,
		)
	}

	s.negotiated = cloneHandshakeResult(
		result,
	)

	s.state = ProtocolStateReady
	s.readyAt = time.Now().UTC()

	return nil
}

// Negotiated returns a defensive copy of the negotiated handshake result.
func (s *Session) Negotiated() (
	HandshakeResult,
	bool,
) {
	if s == nil {
		return HandshakeResult{}, false
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.state != ProtocolStateReady &&
		s.state != ProtocolStateClosing {
		return HandshakeResult{}, false
	}

	return cloneHandshakeResult(
		s.negotiated,
	), true
}

// Ready reports whether ordinary protocol traffic may be exchanged.
func (s *Session) Ready() bool {
	return s != nil &&
		s.State() == ProtocolStateReady
}

// AllowsMessage verifies both session readiness and negotiated capability.
func (s *Session) AllowsMessage(
	messageType MessageType,
) bool {
	if s == nil {
		return false
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.state != ProtocolStateReady {
		return false
	}

	return s.negotiated.AllowsMessage(
		messageType,
	)
}

// NextSequence allocates the next outgoing ordered sequence number.
//
// Sequence zero is reserved for non-ordered control traffic.
func (s *Session) NextSequence() (
	uint64,
	error,
) {
	if s == nil {
		return 0, errors.New("session is nil")
	}

	if !s.Ready() {
		return 0, ErrHandshakeRequired
	}

	next := s.sendSequence.Add(1)

	// Atomic wrap-around is practically unreachable, but sequence zero is
	// reserved and must never be emitted.
	if next == 0 {
		return 0, fmt.Errorf(
			"%w: outgoing sequence exhausted",
			ErrSequenceViolation,
		)
	}

	return next, nil
}

// AcceptSequence validates and records an incoming ordered sequence number.
//
// OTGH v1 expects strictly increasing contiguous ordered messages:
//
//	1, 2, 3, 4...
//
// Duplicate, stale, skipped, and zero sequence values are rejected.
//
// Control messages that do not participate in sequencing should not call this
// method.
func (s *Session) AcceptSequence(
	sequence uint64,
) error {
	if s == nil {
		return errors.New("session is nil")
	}

	if sequence == 0 {
		return fmt.Errorf(
			"%w: sequence zero is reserved",
			ErrSequenceViolation,
		)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != ProtocolStateReady {
		return ErrHandshakeRequired
	}

	expected := s.receiveSequence + 1

	if expected == 0 {
		return fmt.Errorf(
			"%w: incoming sequence exhausted",
			ErrSequenceViolation,
		)
	}

	if sequence != expected {
		return fmt.Errorf(
			"%w: received sequence %d, expected %d",
			ErrSequenceViolation,
			sequence,
			expected,
		)
	}

	s.receiveSequence = sequence

	return nil
}

// LastSentSequence returns the most recently allocated outgoing sequence.
func (s *Session) LastSentSequence() uint64 {
	if s == nil {
		return 0
	}

	return s.sendSequence.Load()
}

// LastReceivedSequence returns the most recently accepted incoming sequence.
func (s *Session) LastReceivedSequence() uint64 {
	if s == nil {
		return 0
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.receiveSequence
}

// Close transitions the session to closed state.
//
// Calling Close multiple times is safe.
func (s *Session) Close(
	reason string,
) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == ProtocolStateClosed {
		return
	}

	s.state = ProtocolStateClosing
	s.closeReason = strings.TrimSpace(reason)

	s.state = ProtocolStateClosed
	s.closedAt = time.Now().UTC()
}

// CloseReason returns the local session shutdown reason.
func (s *Session) CloseReason() string {
	if s == nil {
		return ""
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.closeReason
}

// SessionSnapshot is an immutable diagnostic view of a session.
type SessionSnapshot struct {
	SessionID string

	LocalNodeID  string
	RemoteNodeID string

	State ProtocolState

	Version     Version
	Encoding    Encoding
	Compression Compression

	Capabilities []Capability

	MaxFrameSize   uint64
	MaxPayloadSize uint64

	MaxBatchRecords uint32

	LastSentSequence     uint64
	LastReceivedSequence uint64

	CreatedAt time.Time
	ReadyAt   time.Time
	ClosedAt  time.Time

	CloseReason string
}

// Snapshot returns current session state.
func (s *Session) Snapshot() SessionSnapshot {
	if s == nil {
		return SessionSnapshot{
			State: ProtocolStateClosed,
		}
	}

	s.mu.RLock()

	snapshot := SessionSnapshot{
		SessionID: s.sessionID,

		LocalNodeID:  s.localNodeID,
		RemoteNodeID: s.remoteNodeID,

		State: s.state,

		Version:     s.negotiated.Version,
		Encoding:    s.negotiated.Encoding,
		Compression: s.negotiated.Compression,

		Capabilities: cloneCapabilities(
			s.negotiated.Capabilities,
		),

		MaxFrameSize: s.negotiated.MaxFrameSize,
		MaxPayloadSize: s.negotiated.
			MaxPayloadSize,

		MaxBatchRecords: s.negotiated.
			MaxBatchRecords,

		LastReceivedSequence: s.receiveSequence,

		CreatedAt: s.createdAt,
		ReadyAt:   s.readyAt,
		ClosedAt:  s.closedAt,

		CloseReason: s.closeReason,
	}

	s.mu.RUnlock()

	snapshot.LastSentSequence =
		s.sendSequence.Load()

	return snapshot
}

// Protocol combines OTGH's protocol components.
//
// Protocol owns configuration and reusable stateless components. Individual
// peer state belongs to Session.
type Protocol struct {
	config ProtocolConfig

	encoder *Encoder
	decoder *Decoder

	compression *CompressionRegistry
	negotiator  *HandshakeNegotiator
}

// NewProtocol builds the complete OTGH protocol runtime.
func NewProtocol(
	config ProtocolConfig,
) (*Protocol, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create protocol: %w",
			err,
		)
	}

	compression, err :=
		NewCompressionRegistry(
			config.Compression,
		)
	if err != nil {
		return nil, fmt.Errorf(
			"create protocol compression: %w",
			err,
		)
	}

	encoder, err := NewEncoder(
		config.Encoder,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create protocol encoder: %w",
			err,
		)
	}

	decoder, err := NewDecoder(
		config.Decoder,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create protocol decoder: %w",
			err,
		)
	}

	negotiator, err :=
		NewHandshakeNegotiator(
			config.Handshake,
		)
	if err != nil {
		return nil, fmt.Errorf(
			"create protocol negotiator: %w",
			err,
		)
	}

	return &Protocol{
		config: config,

		encoder: encoder,
		decoder: decoder,

		compression: compression,
		negotiator:  negotiator,
	}, nil
}

// Config returns a defensive configuration copy.
func (p *Protocol) Config() ProtocolConfig {
	if p == nil {
		return ProtocolConfig{}
	}

	config := p.config

	config.Handshake.Versions =
		cloneVersions(
			p.config.Handshake.Versions,
		)

	config.Handshake.Encodings =
		cloneEncodings(
			p.config.Handshake.Encodings,
		)

	config.Handshake.Compressions =
		cloneCompressions(
			p.config.Handshake.Compressions,
		)

	config.Handshake.Capabilities =
		cloneCapabilities(
			p.config.Handshake.Capabilities,
		)

	config.Handshake.RequireCapabilities =
		cloneCapabilities(
			p.config.Handshake.
				RequireCapabilities,
		)

	config.Handshake.Metadata =
		cloneStringMap(
			p.config.Handshake.Metadata,
		)

	return config
}

// Encoder returns the protocol encoder.
func (p *Protocol) Encoder() *Encoder {
	if p == nil {
		return nil
	}

	return p.encoder
}

// Decoder returns the protocol decoder.
func (p *Protocol) Decoder() *Decoder {
	if p == nil {
		return nil
	}

	return p.decoder
}

// Compression returns the compression registry.
func (p *Protocol) Compression() *CompressionRegistry {
	if p == nil {
		return nil
	}

	return p.compression
}

// Negotiator returns the handshake negotiator.
func (p *Protocol) Negotiator() *HandshakeNegotiator {
	if p == nil {
		return nil
	}

	return p.negotiator
}

// NewSession creates an un-negotiated session owned by this protocol runtime.
func (p *Protocol) NewSession(
	sessionID string,
) (*Session, error) {
	if p == nil {
		return nil, errors.New(
			"protocol is nil",
		)
	}

	return NewSession(
		p.config.NodeID,
		sessionID,
	)
}

// BuildHandshake creates this node's handshake request for a session.
func (p *Protocol) BuildHandshake(
	session *Session,
	agentName string,
	agentVersion string,
) (HandshakeRequest, error) {
	if p == nil {
		return HandshakeRequest{}, errors.New(
			"protocol is nil",
		)
	}

	if session == nil {
		return HandshakeRequest{}, errors.New(
			"session is nil",
		)
	}

	if session.LocalNodeID() !=
		p.config.NodeID {
		return HandshakeRequest{}, errors.New(
			"session belongs to a different local node",
		)
	}

	if err := session.BeginHandshake(); err != nil {
		return HandshakeRequest{}, err
	}

	request, err := NewHandshakeRequest(
		p.config.Handshake,
		session.ID(),
		agentName,
		agentVersion,
	)
	if err != nil {
		return HandshakeRequest{}, err
	}

	return request, nil
}

// AcceptHandshake negotiates an incoming request and associates the remote
// node identity with the session.
//
// The caller remains responsible for encoding and transmitting the result.
func (p *Protocol) AcceptHandshake(
	session *Session,
	request HandshakeRequest,
) (HandshakeResult, error) {
	if p == nil {
		return HandshakeResult{}, errors.New(
			"protocol is nil",
		)
	}

	if session == nil {
		return HandshakeResult{}, errors.New(
			"session is nil",
		)
	}

	if request.SessionID != session.ID() {
		result := rejectedHandshakeResult(
			request.SessionID,
			ErrorCodeHandshakeRejected,
			"handshake session identifier does not match active session",
			false,
		)

		return result, ErrHandshakeRejected
	}

	if err := session.SetRemoteNodeID(
		request.NodeID,
	); err != nil {
		return HandshakeResult{}, err
	}

	if err := session.BeginHandshake(); err != nil {
		return HandshakeResult{}, err
	}

	result := p.negotiator.Negotiate(
		request,
	)

	if !result.Accepted() {
		return result, ErrHandshakeRejected
	}

	if err := session.ApplyHandshake(
		result,
	); err != nil {
		return result, err
	}

	return result, nil
}

// ApplyHandshakeResult applies the result received by an initiating peer.
func (p *Protocol) ApplyHandshakeResult(
	session *Session,
	result HandshakeResult,
) error {
	if p == nil {
		return errors.New("protocol is nil")
	}

	if session == nil {
		return errors.New("session is nil")
	}

	if err := result.Validate(); err != nil {
		return err
	}

	if !result.Accepted() {
		return fmt.Errorf(
			"%w: %s",
			ErrHandshakeRejected,
			handshakeErrorMessage(result),
		)
	}

	return session.ApplyHandshake(
		result,
	)
}

// ValidateOutgoingEnvelope verifies that an ordinary outgoing envelope obeys
// the negotiated session contract.
func (p *Protocol) ValidateOutgoingEnvelope(
	session *Session,
	envelope Envelope,
) error {
	if p == nil {
		return errors.New("protocol is nil")
	}

	if session == nil {
		return errors.New("session is nil")
	}

	if !session.Ready() {
		return ErrHandshakeRequired
	}

	if err := envelope.Validate(); err != nil {
		return err
	}

	negotiated, exists :=
		session.Negotiated()

	if !exists {
		return ErrHandshakeRequired
	}

	if envelope.Version !=
		negotiated.Version {
		return fmt.Errorf(
			"%w: envelope version %s does not match negotiated version %s",
			ErrIncompatibleVersion,
			envelope.Version,
			negotiated.Version,
		)
	}

	if envelope.Encoding !=
		negotiated.Encoding {
		return fmt.Errorf(
			"%w: envelope encoding %q does not match negotiated encoding %q",
			ErrUnsupportedEncoding,
			envelope.Encoding,
			negotiated.Encoding,
		)
	}

	if envelope.Compression !=
		negotiated.Compression {
		return fmt.Errorf(
			"%w: envelope compression %q does not match negotiated compression %q",
			ErrUnsupportedCompression,
			envelope.Compression,
			negotiated.Compression,
		)
	}

	if !negotiated.AllowsMessage(
		envelope.MessageType,
	) {
		return fmt.Errorf(
			"%w: message type %q requires an unnegotiated capability",
			ErrUnsupportedMessage,
			envelope.MessageType,
		)
	}

	if uint64(len(envelope.Payload)) >
		negotiated.MaxPayloadSize {
		return fmt.Errorf(
			"%w: payload is %d bytes, negotiated maximum is %d",
			ErrInvalidPayload,
			len(envelope.Payload),
			negotiated.MaxPayloadSize,
		)
	}

	return nil
}

func cloneHandshakeResult(
	input HandshakeResult,
) HandshakeResult {
	cloned := input

	cloned.Capabilities =
		cloneCapabilities(
			input.Capabilities,
		)

	cloned.Metadata =
		cloneStringMap(
			input.Metadata,
		)

	if input.Error != nil {
		cloned.Error =
			cloneProtocolError(
				input.Error,
			)
	}

	return cloned
}

func handshakeErrorMessage(
	result HandshakeResult,
) string {
	if result.Error == nil {
		return "remote peer rejected handshake"
	}

	if strings.TrimSpace(
		result.Error.Message,
	) == "" {
		return string(
			result.Error.Code,
		)
	}

	return result.Error.Message
}
