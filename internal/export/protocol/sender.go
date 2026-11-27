package protocol

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// MessageIDGenerator generates unique identifiers for protocol messages.
type MessageIDGenerator interface {
	NewMessageID() string
}

// MessageIDGeneratorFunc adapts a function to MessageIDGenerator.
type MessageIDGeneratorFunc func() string

// NewMessageID implements MessageIDGenerator.
func (f MessageIDGeneratorFunc) NewMessageID() string {
	if f == nil {
		return ""
	}

	return f()
}

// SenderConfig controls outgoing protocol-message construction.
type SenderConfig struct {
	// Source identifies this OTGH component in outgoing messages.
	Source Source
}

// Validate verifies sender configuration.
func (c SenderConfig) Validate() error {
	if err := c.Source.Validate(); err != nil {
		return fmt.Errorf(
			"sender source: %w",
			err,
		)
	}

	return nil
}

// OutgoingMessage is the logical message supplied to Sender.
//
// Sender adds negotiated protocol details such as version, encoding,
// compression, sequencing, timestamps, and integrity information.
type OutgoingMessage struct {
	Type MessageType

	Priority MessagePriority

	TenantID      string
	CorrelationID string

	Resource *Resource

	Payload []byte

	Metadata map[string]string

	// Ordered determines whether Sender allocates a session sequence number.
	//
	// Ordinary data-plane traffic should normally be ordered when the
	// Sequencing capability has been negotiated.
	Ordered bool
}

// Validate verifies message-level requirements before protocol/session fields
// are applied.
func (m OutgoingMessage) Validate() error {
	if !m.Type.Valid() {
		return fmt.Errorf(
			"%w: invalid message type %q",
			ErrUnsupportedMessage,
			m.Type,
		)
	}

	if !m.Priority.Valid() {
		return fmt.Errorf(
			"%w: invalid message priority %d",
			ErrInvalidEnvelope,
			m.Priority,
		)
	}

	if len(m.Payload) == 0 {
		return fmt.Errorf(
			"%w: outgoing payload is empty",
			ErrInvalidPayload,
		)
	}

	return nil
}

// Sender creates validated OTGH protocol frames for one negotiated Session.
//
// Sender does not own the underlying network connection. HTTP, gRPC, TCP, or
// another transport can write the encoded frame produced here.
//
// Send operations are serialized so sequence allocation and wire writes occur
// in deterministic order when a shared writer is used.
type Sender struct {
	config SenderConfig

	protocol *Protocol
	session  *Session

	idGenerator MessageIDGenerator

	mu sync.Mutex
}

// NewSender creates a sender for one session.
func NewSender(
	config SenderConfig,
	protocol *Protocol,
	session *Session,
	idGenerator MessageIDGenerator,
) (*Sender, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create protocol sender: %w",
			err,
		)
	}

	if protocol == nil {
		return nil, errors.New(
			"create protocol sender: protocol is required",
		)
	}

	if session == nil {
		return nil, errors.New(
			"create protocol sender: session is required",
		)
	}

	if idGenerator == nil {
		return nil, errors.New(
			"create protocol sender: message id generator is required",
		)
	}

	if session.LocalNodeID() !=
		protocol.config.NodeID {
		return nil, errors.New(
			"create protocol sender: session belongs to another local node",
		)
	}

	return &Sender{
		config:      config,
		protocol:    protocol,
		session:     session,
		idGenerator: idGenerator,
	}, nil
}

// Session returns the sender session.
func (s *Sender) Session() *Session {
	if s == nil {
		return nil
	}

	return s.session
}

// Build constructs and encodes one ordinary outgoing protocol message.
func (s *Sender) Build(
	ctx context.Context,
	message OutgoingMessage,
) ([]byte, Envelope, error) {
	if s == nil {
		return nil, Envelope{}, errors.New(
			"protocol sender is nil",
		)
	}

	if ctx == nil {
		return nil, Envelope{}, errors.New(
			"sender context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return nil, Envelope{}, err
	}

	if err := message.Validate(); err != nil {
		return nil, Envelope{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, Envelope{}, err
	}

	if !s.session.Ready() {
		return nil, Envelope{}, ErrHandshakeRequired
	}

	negotiated, exists :=
		s.session.Negotiated()
	if !exists {
		return nil, Envelope{}, ErrHandshakeRequired
	}

	if !negotiated.AllowsMessage(
		message.Type,
	) {
		return nil, Envelope{}, fmt.Errorf(
			"%w: message type %q is not allowed by negotiated capabilities",
			ErrUnsupportedMessage,
			message.Type,
		)
	}

	messageID := strings.TrimSpace(
		s.idGenerator.NewMessageID(),
	)
	if messageID == "" {
		return nil, Envelope{}, errors.New(
			"message id generator returned an empty id",
		)
	}

	var sequence uint64

	if message.Ordered {
		if !negotiated.HasCapability(
			CapabilitySequencing,
		) {
			return nil, Envelope{}, fmt.Errorf(
				"%w: ordered message requires sequencing capability",
				ErrUnsupportedMessage,
			)
		}

		next, err :=
			s.session.NextSequence()
		if err != nil {
			return nil, Envelope{}, err
		}

		sequence = next
	}

	envelope, err := NewEnvelope(
		negotiated.Version,
		messageID,
		message.Type,
		message.Priority,
		s.config.Source,
		negotiated.Encoding,
		negotiated.Compression,
		message.Payload,
	)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"create outgoing envelope: %w",
			err,
		)
	}

	envelope.TenantID = strings.TrimSpace(
		message.TenantID,
	)

	envelope.CorrelationID = strings.TrimSpace(
		message.CorrelationID,
	)

	envelope.Sequence = sequence

	if message.Resource != nil {
		if err := envelope.SetResource(
			*message.Resource,
		); err != nil {
			return nil, Envelope{}, err
		}
	}

	if err := envelope.SetMetadata(
		message.Metadata,
	); err != nil {
		return nil, Envelope{}, err
	}

	if err := s.protocol.ValidateOutgoingEnvelope(
		s.session,
		envelope,
	); err != nil {
		return nil, Envelope{}, err
	}

	encoded, err :=
		s.protocol.encoder.Encode(
			envelope,
		)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode outgoing message %q: %w",
			messageID,
			err,
		)
	}

	negotiatedFrameLimit :=
		negotiated.MaxFrameSize

	if uint64(len(encoded)) <
		uint64(FrameHeaderSize) {
		return nil, Envelope{}, fmt.Errorf(
			"%w: encoded frame is smaller than header",
			ErrInvalidFrame,
		)
	}

	frameBodySize :=
		uint64(len(encoded) - FrameHeaderSize)

	if frameBodySize >
		negotiatedFrameLimit {
		return nil, Envelope{}, fmt.Errorf(
			"%w: encoded frame body is %d bytes, negotiated maximum is %d",
			ErrFrameTooLarge,
			frameBodySize,
			negotiatedFrameLimit,
		)
	}

	return encoded, envelope.Clone(), nil
}

// Send builds a message and writes the complete frame to writer.
//
// Sender does not close or otherwise own writer.
func (s *Sender) Send(
	ctx context.Context,
	writer io.Writer,
	message OutgoingMessage,
) (Envelope, error) {
	if s == nil {
		return Envelope{}, errors.New(
			"protocol sender is nil",
		)
	}

	if writer == nil {
		return Envelope{}, errors.New(
			"protocol writer is nil",
		)
	}

	// Build performs the sender lock internally. We deliberately do not hold
	// that lock across the potentially slow writer operation here. Transports
	// that share a stream between goroutines should serialize writes at the
	// transport/connection layer.
	encoded, envelope, err :=
		s.Build(ctx, message)
	if err != nil {
		return Envelope{}, err
	}

	if err := ctx.Err(); err != nil {
		return Envelope{}, err
	}

	if err := writeAll(
		writer,
		encoded,
	); err != nil {
		return Envelope{}, fmt.Errorf(
			"send protocol message %q: %w",
			envelope.MessageID,
			err,
		)
	}

	return envelope, nil
}

// BuildHandshake constructs a handshake control envelope.
//
// Handshake messages intentionally bypass ordinary session readiness because
// they are what establishes readiness.
func (s *Sender) BuildHandshake(
	ctx context.Context,
	agentName string,
	agentVersion string,
) ([]byte, Envelope, HandshakeRequest, error) {
	if s == nil {
		return nil, Envelope{}, HandshakeRequest{},
			errors.New("protocol sender is nil")
	}

	if ctx == nil {
		return nil, Envelope{}, HandshakeRequest{},
			errors.New("sender context is nil")
	}

	if err := ctx.Err(); err != nil {
		return nil, Envelope{}, HandshakeRequest{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	request, err :=
		s.protocol.BuildHandshake(
			s.session,
			agentName,
			agentVersion,
		)
	if err != nil {
		return nil, Envelope{}, HandshakeRequest{}, err
	}

	payload, err := json.Marshal(request)
	if err != nil {
		return nil, Envelope{}, HandshakeRequest{},
			fmt.Errorf(
				"encode handshake request: %w",
				err,
			)
	}

	messageID := strings.TrimSpace(
		s.idGenerator.NewMessageID(),
	)
	if messageID == "" {
		return nil, Envelope{}, HandshakeRequest{},
			errors.New(
				"message id generator returned an empty id",
			)
	}

	envelope, err := NewEnvelope(
		CurrentVersion,
		messageID,
		MessageTypeHandshake,
		PriorityControl,
		s.config.Source,
		EncodingJSON,
		CompressionNone,
		payload,
	)
	if err != nil {
		return nil, Envelope{}, HandshakeRequest{}, err
	}

	envelope.CorrelationID =
		request.SessionID

	encoded, err := s.protocol.encoder.Encode(
		envelope,
	)
	if err != nil {
		return nil, Envelope{}, HandshakeRequest{},
			fmt.Errorf(
				"encode handshake envelope: %w",
				err,
			)
	}

	return encoded,
		envelope.Clone(),
		request,
		nil
}

// BuildHandshakeResult builds the responder's handshake-result control
// envelope.
//
// Handshake result traffic also bypasses ordinary Ready checks.
func (s *Sender) BuildHandshakeResult(
	ctx context.Context,
	result HandshakeResult,
) ([]byte, Envelope, error) {
	if s == nil {
		return nil, Envelope{},
			errors.New("protocol sender is nil")
	}

	if ctx == nil {
		return nil, Envelope{},
			errors.New("sender context is nil")
	}

	if err := ctx.Err(); err != nil {
		return nil, Envelope{}, err
	}

	if err := result.Validate(); err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"invalid handshake result: %w",
			err,
		)
	}

	if result.SessionID !=
		s.session.ID() {
		return nil, Envelope{}, fmt.Errorf(
			"%w: handshake result session %q does not match sender session %q",
			ErrHandshakeRejected,
			result.SessionID,
			s.session.ID(),
		)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	payload, err := json.Marshal(result)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode handshake result: %w",
			err,
		)
	}

	messageID := strings.TrimSpace(
		s.idGenerator.NewMessageID(),
	)
	if messageID == "" {
		return nil, Envelope{},
			errors.New(
				"message id generator returned an empty id",
			)
	}

	version := CurrentVersion

	if result.Accepted() {
		version = result.Version
	}

	envelope, err := NewEnvelope(
		version,
		messageID,
		MessageTypeHandshakeResult,
		PriorityControl,
		s.config.Source,
		EncodingJSON,
		CompressionNone,
		payload,
	)
	if err != nil {
		return nil, Envelope{}, err
	}

	envelope.CorrelationID =
		result.SessionID

	encoded, err := s.protocol.encoder.Encode(
		envelope,
	)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode handshake result envelope: %w",
			err,
		)
	}

	return encoded, envelope.Clone(), nil
}

// BuildAcknowledgement constructs a protocol acknowledgement envelope.
//
// Acknowledgement messages use the negotiated version after session setup but
// remain uncompressed JSON control messages in protocol v1.
func (s *Sender) BuildAcknowledgement(
	ctx context.Context,
	ack Acknowledgement,
) ([]byte, Envelope, error) {
	if s == nil {
		return nil, Envelope{},
			errors.New("protocol sender is nil")
	}

	if ctx == nil {
		return nil, Envelope{},
			errors.New("sender context is nil")
	}

	if err := ctx.Err(); err != nil {
		return nil, Envelope{}, err
	}

	if !s.session.Ready() {
		return nil, Envelope{}, ErrHandshakeRequired
	}

	if err := ack.Validate(); err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"invalid acknowledgement: %w",
			err,
		)
	}

	if ack.SessionID != s.session.ID() {
		return nil, Envelope{}, fmt.Errorf(
			"%w: acknowledgement session %q does not match sender session %q",
			ErrAcknowledgement,
			ack.SessionID,
			s.session.ID(),
		)
	}

	negotiated, exists :=
		s.session.Negotiated()
	if !exists {
		return nil, Envelope{}, ErrHandshakeRequired
	}

	if !negotiated.HasCapability(
		CapabilityAcknowledgements,
	) {
		return nil, Envelope{}, fmt.Errorf(
			"%w: acknowledgements were not negotiated",
			ErrUnsupportedMessage,
		)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	payload, err := json.Marshal(ack)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode acknowledgement: %w",
			err,
		)
	}

	messageID := strings.TrimSpace(
		s.idGenerator.NewMessageID(),
	)
	if messageID == "" {
		return nil, Envelope{},
			errors.New(
				"message id generator returned an empty id",
			)
	}

	envelope, err := NewEnvelope(
		negotiated.Version,
		messageID,
		MessageTypeAcknowledgement,
		PriorityControl,
		s.config.Source,
		EncodingJSON,
		CompressionNone,
		payload,
	)
	if err != nil {
		return nil, Envelope{}, err
	}

	envelope.CorrelationID =
		ack.CorrelationID

	encoded, err := s.protocol.encoder.Encode(
		envelope,
	)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode acknowledgement envelope: %w",
			err,
		)
	}

	return encoded, envelope.Clone(), nil
}

// BuildError constructs a protocol-error envelope.
func (s *Sender) BuildError(
	ctx context.Context,
	protocolErr *ProtocolError,
	correlationID string,
) ([]byte, Envelope, error) {
	if s == nil {
		return nil, Envelope{},
			errors.New("protocol sender is nil")
	}

	if ctx == nil {
		return nil, Envelope{},
			errors.New("sender context is nil")
	}

	if err := ctx.Err(); err != nil {
		return nil, Envelope{}, err
	}

	if protocolErr == nil {
		return nil, Envelope{},
			errors.New(
				"protocol error is required",
			)
	}

	if err := protocolErr.Validate(); err != nil {
		return nil, Envelope{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	wireError := cloneProtocolError(
		protocolErr,
	)

	payload, err := json.Marshal(
		wireError,
	)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode protocol error: %w",
			err,
		)
	}

	messageID := strings.TrimSpace(
		s.idGenerator.NewMessageID(),
	)
	if messageID == "" {
		return nil, Envelope{},
			errors.New(
				"message id generator returned an empty id",
			)
	}

	version := CurrentVersion

	if negotiated, exists :=
		s.session.Negotiated(); exists {
		version = negotiated.Version
	}

	envelope, err := NewEnvelope(
		version,
		messageID,
		MessageTypeError,
		PriorityControl,
		s.config.Source,
		EncodingJSON,
		CompressionNone,
		payload,
	)
	if err != nil {
		return nil, Envelope{}, err
	}

	envelope.CorrelationID =
		strings.TrimSpace(correlationID)

	encoded, err := s.protocol.encoder.Encode(
		envelope,
	)
	if err != nil {
		return nil, Envelope{}, fmt.Errorf(
			"encode protocol error envelope: %w",
			err,
		)
	}

	return encoded, envelope.Clone(), nil
}

// NewTimestampMessageIDGenerator returns a simple process-local message ID
// generator.
//
// It creates identifiers using the current UTC Unix nanosecond timestamp and
// an atomic sequence suffix. This avoids introducing an external identifier
// dependency into the protocol package.
func NewTimestampMessageIDGenerator(
	prefix string,
) MessageIDGenerator {
	prefix = strings.TrimSpace(prefix)

	if prefix == "" {
		prefix = "otgh"
	}

	return &timestampMessageIDGenerator{
		prefix: prefix,
	}
}

type timestampMessageIDGenerator struct {
	prefix string

	counter atomic.Uint64
}

func (g *timestampMessageIDGenerator) NewMessageID() string {
	if g == nil {
		return ""
	}

	sequence := g.counter.Add(1)

	return fmt.Sprintf(
		"%s-%d-%d",
		g.prefix,
		time.Now().UTC().UnixNano(),
		sequence,
	)
}
