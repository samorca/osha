package protocol

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// ReceivedKind identifies the decoded result returned by Receiver.
type ReceivedKind uint8

const (
	ReceivedKindUnknown ReceivedKind = iota
	ReceivedKindMessage
	ReceivedKindHandshake
	ReceivedKindHandshakeResult
	ReceivedKindAcknowledgement
	ReceivedKindProtocolError
)

// String returns a stable representation of the received kind.
func (k ReceivedKind) String() string {
	switch k {
	case ReceivedKindMessage:
		return "message"

	case ReceivedKindHandshake:
		return "handshake"

	case ReceivedKindHandshakeResult:
		return "handshake_result"

	case ReceivedKindAcknowledgement:
		return "acknowledgement"

	case ReceivedKindProtocolError:
		return "protocol_error"

	default:
		return "unknown"
	}
}

// Valid reports whether the received kind is recognized.
func (k ReceivedKind) Valid() bool {
	return k >= ReceivedKindMessage &&
		k <= ReceivedKindProtocolError
}

// ReceivedMessage contains one successfully decoded protocol message.
//
// Exactly one control-message field is normally populated based on Kind.
// Ordinary infrastructure messages use Envelope and leave the control fields
// nil.
type ReceivedMessage struct {
	Kind ReceivedKind

	Envelope Envelope

	Handshake       *HandshakeRequest
	HandshakeResult *HandshakeResult
	Acknowledgement *Acknowledgement
	ProtocolError   *ProtocolError
}

// Validate verifies the decoded result.
func (m ReceivedMessage) Validate() error {
	if !m.Kind.Valid() {
		return fmt.Errorf(
			"%w: invalid received kind %d",
			ErrMalformedMessage,
			m.Kind,
		)
	}

	if err := m.Envelope.ValidateStructure(); err != nil {
		return fmt.Errorf(
			"received envelope structure: %w",
			err,
		)
	}

	switch m.Kind {
	case ReceivedKindMessage:
		if isControlMessage(m.Envelope.MessageType) {
			return fmt.Errorf(
				"%w: control message %q classified as ordinary message",
				ErrMalformedMessage,
				m.Envelope.MessageType,
			)
		}

		if m.Handshake != nil ||
			m.HandshakeResult != nil ||
			m.Acknowledgement != nil ||
			m.ProtocolError != nil {
			return errors.New(
				"ordinary message contains control payload",
			)
		}

	case ReceivedKindHandshake:
		if m.Envelope.MessageType !=
			MessageTypeHandshake {
			return errors.New(
				"handshake result has incorrect envelope message type",
			)
		}

		if m.Handshake == nil {
			return errors.New(
				"received handshake payload is missing",
			)
		}

		if err := m.Handshake.Validate(); err != nil {
			return fmt.Errorf(
				"received handshake: %w",
				err,
			)
		}

	case ReceivedKindHandshakeResult:
		if m.Envelope.MessageType !=
			MessageTypeHandshakeResult {
			return errors.New(
				"handshake result has incorrect envelope message type",
			)
		}

		if m.HandshakeResult == nil {
			return errors.New(
				"received handshake result payload is missing",
			)
		}

		if err := m.HandshakeResult.Validate(); err != nil {
			return fmt.Errorf(
				"received handshake result: %w",
				err,
			)
		}

	case ReceivedKindAcknowledgement:
		if m.Envelope.MessageType !=
			MessageTypeAcknowledgement {
			return errors.New(
				"acknowledgement has incorrect envelope message type",
			)
		}

		if m.Acknowledgement == nil {
			return errors.New(
				"received acknowledgement payload is missing",
			)
		}

		if err := m.Acknowledgement.Validate(); err != nil {
			return fmt.Errorf(
				"received acknowledgement: %w",
				err,
			)
		}

	case ReceivedKindProtocolError:
		if m.Envelope.MessageType !=
			MessageTypeError {
			return errors.New(
				"protocol error has incorrect envelope message type",
			)
		}

		if m.ProtocolError == nil {
			return errors.New(
				"received protocol error payload is missing",
			)
		}

		if err := m.ProtocolError.Validate(); err != nil {
			return fmt.Errorf(
				"received protocol error: %w",
				err,
			)
		}
	}

	return nil
}

// Receiver decodes incoming OTGH frames and enforces the active Session
// contract.
//
// Receiver handles protocol validation only. It does not own the underlying
// network connection and does not automatically send acknowledgements or
// protocol-error responses.
type Receiver struct {
	protocol *Protocol
	session  *Session
}

// NewReceiver creates a receiver for one protocol session.
func NewReceiver(
	protocol *Protocol,
	session *Session,
) (*Receiver, error) {
	if protocol == nil {
		return nil, errors.New(
			"create protocol receiver: protocol is required",
		)
	}

	if session == nil {
		return nil, errors.New(
			"create protocol receiver: session is required",
		)
	}

	if session.LocalNodeID() !=
		protocol.config.NodeID {
		return nil, errors.New(
			"create protocol receiver: session belongs to another local node",
		)
	}

	return &Receiver{
		protocol: protocol,
		session:  session,
	}, nil
}

// Session returns the receiver session.
func (r *Receiver) Session() *Session {
	if r == nil {
		return nil
	}

	return r.session
}

// Receive decodes one complete serialized OTGH frame.
func (r *Receiver) Receive(
	ctx context.Context,
	data []byte,
) (ReceivedMessage, error) {
	if r == nil {
		return ReceivedMessage{}, errors.New(
			"protocol receiver is nil",
		)
	}

	if ctx == nil {
		return ReceivedMessage{}, errors.New(
			"receiver context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return ReceivedMessage{}, err
	}

	if len(data) == 0 {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: received frame is empty",
			ErrInvalidFrame,
		)
	}

	envelope, err := r.protocol.decoder.Decode(data)
	if err != nil {
		return ReceivedMessage{}, err
	}

	if err := ctx.Err(); err != nil {
		return ReceivedMessage{}, err
	}

	return r.processEnvelope(envelope)
}

// ReceiveFrom reads and decodes exactly one OTGH frame from reader.
//
// Receiver does not close reader.
func (r *Receiver) ReceiveFrom(
	ctx context.Context,
	reader io.Reader,
) (ReceivedMessage, error) {
	if r == nil {
		return ReceivedMessage{}, errors.New(
			"protocol receiver is nil",
		)
	}

	if ctx == nil {
		return ReceivedMessage{}, errors.New(
			"receiver context is nil",
		)
	}

	if reader == nil {
		return ReceivedMessage{}, errors.New(
			"protocol reader is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return ReceivedMessage{}, err
	}

	envelope, err :=
		r.protocol.decoder.DecodeFrom(reader)
	if err != nil {
		return ReceivedMessage{}, err
	}

	if err := ctx.Err(); err != nil {
		return ReceivedMessage{}, err
	}

	return r.processEnvelope(envelope)
}

func (r *Receiver) processEnvelope(
	envelope Envelope,
) (ReceivedMessage, error) {
	switch envelope.MessageType {
	case MessageTypeHandshake:
		return r.processHandshake(envelope)

	case MessageTypeHandshakeResult:
		return r.processHandshakeResult(envelope)

	case MessageTypeAcknowledgement:
		return r.processAcknowledgement(envelope)

	case MessageTypeError:
		return r.processProtocolError(envelope)

	default:
		return r.processOrdinaryMessage(envelope)
	}
}

func (r *Receiver) processOrdinaryMessage(
	envelope Envelope,
) (ReceivedMessage, error) {
	if !r.session.Ready() {
		return ReceivedMessage{}, ErrHandshakeRequired
	}

	negotiated, exists :=
		r.session.Negotiated()
	if !exists {
		return ReceivedMessage{}, ErrHandshakeRequired
	}

	if err := validateEnvelopeAgainstSession(
		envelope,
		negotiated,
	); err != nil {
		return ReceivedMessage{}, err
	}

	if !negotiated.AllowsMessage(
		envelope.MessageType,
	) {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: message type %q requires an unnegotiated capability",
			ErrUnsupportedMessage,
			envelope.MessageType,
		)
	}

	sequencingEnabled :=
		negotiated.HasCapability(
			CapabilitySequencing,
		)

	switch {
	case sequencingEnabled &&
		envelope.Sequence == 0:
		return ReceivedMessage{}, fmt.Errorf(
			"%w: negotiated sequencing requires a non-zero sequence",
			ErrSequenceViolation,
		)

	case sequencingEnabled:
		if err := r.session.AcceptSequence(
			envelope.Sequence,
		); err != nil {
			return ReceivedMessage{}, err
		}

	case !sequencingEnabled &&
		envelope.Sequence != 0:
		return ReceivedMessage{}, fmt.Errorf(
			"%w: message contains sequence %d but sequencing was not negotiated",
			ErrSequenceViolation,
			envelope.Sequence,
		)
	}

	received := ReceivedMessage{
		Kind:     ReceivedKindMessage,
		Envelope: envelope.Clone(),
	}

	if err := received.Validate(); err != nil {
		return ReceivedMessage{}, err
	}

	return received, nil
}

func (r *Receiver) processHandshake(
	envelope Envelope,
) (ReceivedMessage, error) {
	if r.session.State() == ProtocolStateReady {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake received after session became ready",
			ErrHandshakeRejected,
		)
	}

	if envelope.Encoding != EncodingJSON {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake encoding must be JSON",
			ErrUnsupportedEncoding,
		)
	}

	if envelope.Compression != CompressionNone {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake cannot be compressed",
			ErrUnsupportedCompression,
		)
	}

	if envelope.Sequence != 0 {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake sequence must be zero",
			ErrSequenceViolation,
		)
	}

	var request HandshakeRequest

	if err := decodeStrictJSONPayload(
		envelope.Payload,
		&request,
	); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if err := request.Validate(); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if request.SessionID != r.session.ID() {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake session %q does not match active session %q",
			ErrHandshakeRejected,
			request.SessionID,
			r.session.ID(),
		)
	}

	if envelope.CorrelationID != "" &&
		envelope.CorrelationID != request.SessionID {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake correlation id does not match session id",
			ErrHandshakeRejected,
		)
	}

	requestCopy := cloneHandshakeRequest(request)

	received := ReceivedMessage{
		Kind:       ReceivedKindHandshake,
		Envelope:   envelope.Clone(),
		Handshake:  &requestCopy,
	}

	if err := received.Validate(); err != nil {
		return ReceivedMessage{}, err
	}

	return received, nil
}

func (r *Receiver) processHandshakeResult(
	envelope Envelope,
) (ReceivedMessage, error) {
	state := r.session.State()

	if state != ProtocolStateHandshaking &&
		state != ProtocolStateNew {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result received while session is %s",
			ErrHandshakeRejected,
			state,
		)
	}

	if envelope.Encoding != EncodingJSON {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result encoding must be JSON",
			ErrUnsupportedEncoding,
		)
	}

	if envelope.Compression != CompressionNone {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result cannot be compressed",
			ErrUnsupportedCompression,
		)
	}

	if envelope.Sequence != 0 {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result sequence must be zero",
			ErrSequenceViolation,
		)
	}

	var result HandshakeResult

	if err := decodeStrictJSONPayload(
		envelope.Payload,
		&result,
	); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if err := result.Validate(); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if result.SessionID != r.session.ID() {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result session %q does not match active session %q",
			ErrHandshakeRejected,
			result.SessionID,
			r.session.ID(),
		)
	}

	if envelope.CorrelationID != "" &&
		envelope.CorrelationID != result.SessionID {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: handshake result correlation id does not match session id",
			ErrHandshakeRejected,
		)
	}

	resultCopy := cloneHandshakeResult(result)

	received := ReceivedMessage{
		Kind:            ReceivedKindHandshakeResult,
		Envelope:        envelope.Clone(),
		HandshakeResult: &resultCopy,
	}

	if err := received.Validate(); err != nil {
		return ReceivedMessage{}, err
	}

	return received, nil
}

func (r *Receiver) processAcknowledgement(
	envelope Envelope,
) (ReceivedMessage, error) {
	if !r.session.Ready() {
		return ReceivedMessage{}, ErrHandshakeRequired
	}

	negotiated, exists :=
		r.session.Negotiated()
	if !exists {
		return ReceivedMessage{}, ErrHandshakeRequired
	}

	if !negotiated.HasCapability(
		CapabilityAcknowledgements,
	) {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: acknowledgements were not negotiated",
			ErrUnsupportedMessage,
		)
	}

	if err := validateControlEnvelopeAgainstSession(
		envelope,
		negotiated,
	); err != nil {
		return ReceivedMessage{}, err
	}

	if envelope.Sequence != 0 {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: acknowledgement sequence must be zero",
			ErrSequenceViolation,
		)
	}

	var acknowledgement Acknowledgement

	if err := decodeStrictJSONPayload(
		envelope.Payload,
		&acknowledgement,
	); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: acknowledgement payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if err := acknowledgement.Validate(); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: acknowledgement payload: %v",
			ErrAcknowledgement,
			err,
		)
	}

	if acknowledgement.SessionID !=
		r.session.ID() {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: acknowledgement session %q does not match active session %q",
			ErrAcknowledgement,
			acknowledgement.SessionID,
			r.session.ID(),
		)
	}

	if acknowledgement.Status == AckStatusPartial &&
		!negotiated.HasCapability(
			CapabilityPartialAcknowledgement,
		) {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: partial acknowledgements were not negotiated",
			ErrUnsupportedMessage,
		)
	}

	ackCopy := acknowledgement.Clone()

	received := ReceivedMessage{
		Kind:            ReceivedKindAcknowledgement,
		Envelope:        envelope.Clone(),
		Acknowledgement: &ackCopy,
	}

	if err := received.Validate(); err != nil {
		return ReceivedMessage{}, err
	}

	return received, nil
}

func (r *Receiver) processProtocolError(
	envelope Envelope,
) (ReceivedMessage, error) {
	if envelope.Encoding != EncodingJSON {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: protocol error encoding must be JSON",
			ErrUnsupportedEncoding,
		)
	}

	if envelope.Compression != CompressionNone {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: protocol error cannot be compressed",
			ErrUnsupportedCompression,
		)
	}

	if envelope.Sequence != 0 {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: protocol error sequence must be zero",
			ErrSequenceViolation,
		)
	}

	// Protocol errors are permitted before a session becomes Ready so peers
	// can explain handshake failures.
	if r.session.Ready() {
		negotiated, exists :=
			r.session.Negotiated()
		if !exists {
			return ReceivedMessage{}, ErrHandshakeRequired
		}

		if envelope.Version != negotiated.Version {
			return ReceivedMessage{}, fmt.Errorf(
				"%w: protocol error version %s does not match negotiated version %s",
				ErrIncompatibleVersion,
				envelope.Version,
				negotiated.Version,
			)
		}
	}

	var wireError ProtocolError

	if err := decodeStrictJSONPayload(
		envelope.Payload,
		&wireError,
	); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: protocol error payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if err := wireError.Validate(); err != nil {
		return ReceivedMessage{}, fmt.Errorf(
			"%w: protocol error payload: %v",
			ErrMalformedMessage,
			err,
		)
	}

	localError := ErrorFromRemote(
		wireError.Code,
		wireError.Message,
		wireError.Retryable,
		wireError.CorrelationID,
		wireError.Details,
	)

	received := ReceivedMessage{
		Kind:          ReceivedKindProtocolError,
		Envelope:      envelope.Clone(),
		ProtocolError: localError,
	}

	if err := received.Validate(); err != nil {
		return ReceivedMessage{}, err
	}

	return received, nil
}

func validateEnvelopeAgainstSession(
	envelope Envelope,
	negotiated HandshakeResult,
) error {
	if envelope.Version != negotiated.Version {
		return fmt.Errorf(
			"%w: envelope version %s does not match negotiated version %s",
			ErrIncompatibleVersion,
			envelope.Version,
			negotiated.Version,
		)
	}

	if envelope.Encoding != negotiated.Encoding {
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

	if uint64(len(envelope.Payload)) >
		negotiated.MaxPayloadSize {
		return fmt.Errorf(
			"%w: logical payload is %d bytes, negotiated maximum is %d",
			ErrInvalidPayload,
			len(envelope.Payload),
			negotiated.MaxPayloadSize,
		)
	}

	return nil
}

func validateControlEnvelopeAgainstSession(
	envelope Envelope,
	negotiated HandshakeResult,
) error {
	if envelope.Version != negotiated.Version {
		return fmt.Errorf(
			"%w: control envelope version %s does not match negotiated version %s",
			ErrIncompatibleVersion,
			envelope.Version,
			negotiated.Version,
		)
	}

	if envelope.Encoding != EncodingJSON {
		return fmt.Errorf(
			"%w: protocol v1 control messages require JSON",
			ErrUnsupportedEncoding,
		)
	}

	if envelope.Compression != CompressionNone {
		return fmt.Errorf(
			"%w: protocol v1 control messages cannot be compressed",
			ErrUnsupportedCompression,
		)
	}

	return nil
}

func decodeStrictJSONPayload(
	payload []byte,
	destination any,
) error {
	if len(payload) == 0 {
		return fmt.Errorf(
			"%w: JSON payload is empty",
			ErrInvalidPayload,
		)
	}

	if destination == nil {
		return errors.New(
			"JSON destination is nil",
		)
	}

	decoder := json.NewDecoder(
		bytes.NewReader(payload),
	)

	decoder.DisallowUnknownFields()

	if err := decoder.Decode(
		destination,
	); err != nil {
		return err
	}

	var extra json.RawMessage

	err := decoder.Decode(&extra)

	if errors.Is(err, io.EOF) {
		return nil
	}

	if err == nil {
		return errors.New(
			"JSON payload contains multiple values",
		)
	}

	return fmt.Errorf(
		"JSON payload contains trailing data: %w",
		err,
	)
}

func cloneHandshakeRequest(
	input HandshakeRequest,
) HandshakeRequest {
	cloned := input

	cloned.Versions =
		cloneVersions(input.Versions)

	cloned.Encodings =
		cloneEncodings(input.Encodings)

	cloned.Compressions =
		cloneCompressions(
			input.Compressions,
		)

	cloned.Capabilities =
		cloneCapabilities(
			input.Capabilities,
		)

	cloned.Metadata =
		cloneStringMap(input.Metadata)

	return cloned
}
