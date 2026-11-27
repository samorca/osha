package protocol

import (
	"errors"
	"fmt"
	"strings"
)

// Sentinel errors allow callers to classify protocol failures with errors.Is
// without depending on error-message text.
var (
	ErrProtocol             = errors.New("protocol error")
	ErrMalformedMessage     = errors.New("malformed protocol message")
	ErrUnsupportedMessage   = errors.New("unsupported message")
	ErrInvalidFrame         = errors.New("invalid protocol frame")
	ErrFrameTooLarge        = errors.New("protocol frame too large")
	ErrInvalidEnvelope      = errors.New("invalid protocol envelope")
	ErrInvalidPayload       = errors.New("invalid protocol payload")
	ErrInvalidChecksum      = errors.New("invalid payload checksum")
	ErrUnsupportedEncoding  = errors.New("unsupported encoding")
	ErrUnsupportedCompression = errors.New("unsupported compression")
	ErrHandshakeRequired    = errors.New("protocol handshake required")
	ErrHandshakeRejected    = errors.New("protocol handshake rejected")
	ErrSessionClosed        = errors.New("protocol session closed")
	ErrSequenceViolation    = errors.New("protocol sequence violation")
	ErrAcknowledgement      = errors.New("protocol acknowledgement error")
	ErrAuthentication       = errors.New("protocol authentication error")
	ErrAuthorization        = errors.New("protocol authorization error")
	ErrRateLimited          = errors.New("protocol rate limited")
	ErrTemporaryUnavailable = errors.New("protocol temporarily unavailable")
)

// ErrorCode is the stable machine-readable error identifier sent between
// protocol peers.
//
// Human-readable messages may change over time. ErrorCode values form part of
// the protocol contract and therefore should not be renamed casually.
type ErrorCode string

const (
	ErrorCodeUnknown ErrorCode = "unknown"

	ErrorCodeMalformedMessage      ErrorCode = "malformed_message"
	ErrorCodeUnsupportedMessage    ErrorCode = "unsupported_message"
	ErrorCodeInvalidFrame          ErrorCode = "invalid_frame"
	ErrorCodeFrameTooLarge         ErrorCode = "frame_too_large"
	ErrorCodeInvalidEnvelope       ErrorCode = "invalid_envelope"
	ErrorCodeInvalidPayload        ErrorCode = "invalid_payload"
	ErrorCodeInvalidChecksum       ErrorCode = "invalid_checksum"
	ErrorCodeUnsupportedVersion    ErrorCode = "unsupported_version"
	ErrorCodeUnsupportedEncoding   ErrorCode = "unsupported_encoding"
	ErrorCodeUnsupportedCompression ErrorCode = "unsupported_compression"

	ErrorCodeHandshakeRequired ErrorCode = "handshake_required"
	ErrorCodeHandshakeRejected ErrorCode = "handshake_rejected"

	ErrorCodeSessionClosed     ErrorCode = "session_closed"
	ErrorCodeSequenceViolation ErrorCode = "sequence_violation"

	ErrorCodeAcknowledgement ErrorCode = "acknowledgement_error"

	ErrorCodeAuthenticationFailed ErrorCode = "authentication_failed"
	ErrorCodePermissionDenied      ErrorCode = "permission_denied"

	ErrorCodeRateLimited          ErrorCode = "rate_limited"
	ErrorCodeTemporaryUnavailable ErrorCode = "temporary_unavailable"

	ErrorCodeInternal ErrorCode = "internal_error"
)

// Valid reports whether the error code is recognised by this protocol
// implementation.
func (c ErrorCode) Valid() bool {
	switch c {
	case ErrorCodeMalformedMessage,
		ErrorCodeUnsupportedMessage,
		ErrorCodeInvalidFrame,
		ErrorCodeFrameTooLarge,
		ErrorCodeInvalidEnvelope,
		ErrorCodeInvalidPayload,
		ErrorCodeInvalidChecksum,
		ErrorCodeUnsupportedVersion,
		ErrorCodeUnsupportedEncoding,
		ErrorCodeUnsupportedCompression,
		ErrorCodeHandshakeRequired,
		ErrorCodeHandshakeRejected,
		ErrorCodeSessionClosed,
		ErrorCodeSequenceViolation,
		ErrorCodeAcknowledgement,
		ErrorCodeAuthenticationFailed,
		ErrorCodePermissionDenied,
		ErrorCodeRateLimited,
		ErrorCodeTemporaryUnavailable,
		ErrorCodeInternal:
		return true

	default:
		return false
	}
}

// Retryable reports whether the protocol error normally represents a
// temporary condition.
//
// This is a protocol-level hint only. The OTGH retry engine remains
// responsible for applying retry limits, delays, policy, and cancellation.
func (c ErrorCode) Retryable() bool {
	switch c {
	case ErrorCodeRateLimited,
		ErrorCodeTemporaryUnavailable:
		return true

	default:
		return false
	}
}

// ProtocolError is the structured error representation used internally by the
// OTGH-native protocol.
//
// Code is intended for machines.
// Message is intended for operators and diagnostics.
// Retryable explicitly communicates whether the peer considers retry useful.
//
// Cause is local process state and is deliberately excluded from serialized
// representations.
type ProtocolError struct {
	Code ErrorCode `json:"code"`

	Message string `json:"message,omitempty"`

	Retryable bool `json:"retryable"`

	// CorrelationID associates the error with a request, exchange, or
	// protocol operation when available.
	CorrelationID string `json:"correlation_id,omitempty"`

	// Details may contain small non-sensitive diagnostic attributes.
	//
	// Secrets, credentials, tokens, raw authentication headers, and large
	// payloads must never be placed here.
	Details map[string]string `json:"details,omitempty"`

	cause error
}

// Error implements the error interface.
func (e *ProtocolError) Error() string {
	if e == nil {
		return "<nil>"
	}

	switch {
	case e.Code != "" && e.Message != "":
		return fmt.Sprintf(
			"%s: %s",
			e.Code,
			e.Message,
		)

	case e.Code != "":
		return string(e.Code)

	case e.Message != "":
		return e.Message

	default:
		return ErrProtocol.Error()
	}
}

// Unwrap exposes the local underlying error for errors.Is/errors.As.
//
// Cause is not serialized to remote peers.
func (e *ProtocolError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.cause
}

// Validate verifies the structured protocol error.
func (e ProtocolError) Validate() error {
	if !e.Code.Valid() {
		return fmt.Errorf(
			"%w: invalid error code %q",
			ErrProtocol,
			e.Code,
		)
	}

	if strings.TrimSpace(e.Message) == "" {
		return fmt.Errorf(
			"%w: protocol error message is required",
			ErrProtocol,
		)
	}

	return nil
}

// WithCause returns a copy of the protocol error with a local cause.
//
// The cause remains process-local and is never serialized.
func (e ProtocolError) WithCause(
	cause error,
) *ProtocolError {
	e.cause = cause

	return &e
}

// NewProtocolError constructs a validated protocol error.
//
// If retryable is true for a code that is normally non-retryable, the explicit
// caller decision is preserved. This permits context-specific policy while
// ErrorCode.Retryable continues to provide the default classification.
func NewProtocolError(
	code ErrorCode,
	message string,
	retryable bool,
) *ProtocolError {
	return &ProtocolError{
		Code:      code,
		Message:   strings.TrimSpace(message),
		Retryable: retryable,
	}
}

// NewDefaultProtocolError constructs an error using the code's default
// retryability classification.
func NewDefaultProtocolError(
	code ErrorCode,
	message string,
) *ProtocolError {
	return NewProtocolError(
		code,
		message,
		code.Retryable(),
	)
}

// WithCorrelationID returns a copy carrying a correlation identifier.
func (e ProtocolError) WithCorrelationID(
	id string,
) *ProtocolError {
	e.CorrelationID = strings.TrimSpace(id)

	return &e
}

// WithDetails returns a copy containing defensive copies of diagnostic
// details.
func (e ProtocolError) WithDetails(
	details map[string]string,
) *ProtocolError {
	e.Details = cloneStringMap(details)

	return &e
}

// IsRetryableError reports whether err represents a retryable protocol
// failure.
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	var protocolErr *ProtocolError

	if errors.As(err, &protocolErr) {
		return protocolErr.Retryable
	}

	return errors.Is(
		err,
		ErrRateLimited,
	) || errors.Is(
		err,
		ErrTemporaryUnavailable,
	)
}

// CodeFromError translates known local protocol errors into stable wire error
// codes.
//
// Unknown errors intentionally map to internal_error rather than leaking
// implementation-specific information to a peer.
func CodeFromError(err error) ErrorCode {
	if err == nil {
		return ErrorCodeUnknown
	}

	var protocolErr *ProtocolError

	if errors.As(err, &protocolErr) &&
		protocolErr.Code.Valid() {
		return protocolErr.Code
	}

	switch {
	case errors.Is(err, ErrMalformedMessage):
		return ErrorCodeMalformedMessage

	case errors.Is(err, ErrUnsupportedMessage):
		return ErrorCodeUnsupportedMessage

	case errors.Is(err, ErrInvalidFrame):
		return ErrorCodeInvalidFrame

	case errors.Is(err, ErrFrameTooLarge):
		return ErrorCodeFrameTooLarge

	case errors.Is(err, ErrInvalidEnvelope):
		return ErrorCodeInvalidEnvelope

	case errors.Is(err, ErrInvalidPayload):
		return ErrorCodeInvalidPayload

	case errors.Is(err, ErrInvalidChecksum):
		return ErrorCodeInvalidChecksum

	case errors.Is(err, ErrInvalidVersion),
		errors.Is(err, ErrIncompatibleVersion),
		errors.Is(err, ErrNoCompatibleVersion):
		return ErrorCodeUnsupportedVersion

	case errors.Is(err, ErrUnsupportedEncoding):
		return ErrorCodeUnsupportedEncoding

	case errors.Is(err, ErrUnsupportedCompression):
		return ErrorCodeUnsupportedCompression

	case errors.Is(err, ErrHandshakeRequired):
		return ErrorCodeHandshakeRequired

	case errors.Is(err, ErrHandshakeRejected):
		return ErrorCodeHandshakeRejected

	case errors.Is(err, ErrSessionClosed):
		return ErrorCodeSessionClosed

	case errors.Is(err, ErrSequenceViolation):
		return ErrorCodeSequenceViolation

	case errors.Is(err, ErrAcknowledgement):
		return ErrorCodeAcknowledgement

	case errors.Is(err, ErrAuthentication):
		return ErrorCodeAuthenticationFailed

	case errors.Is(err, ErrAuthorization):
		return ErrorCodePermissionDenied

	case errors.Is(err, ErrRateLimited):
		return ErrorCodeRateLimited

	case errors.Is(err, ErrTemporaryUnavailable):
		return ErrorCodeTemporaryUnavailable

	default:
		return ErrorCodeInternal
	}
}

// SentinelFromCode converts a wire error code into its corresponding local
// sentinel error.
//
// Unknown codes return ErrProtocol.
func SentinelFromCode(
	code ErrorCode,
) error {
	switch code {
	case ErrorCodeMalformedMessage:
		return ErrMalformedMessage

	case ErrorCodeUnsupportedMessage:
		return ErrUnsupportedMessage

	case ErrorCodeInvalidFrame:
		return ErrInvalidFrame

	case ErrorCodeFrameTooLarge:
		return ErrFrameTooLarge

	case ErrorCodeInvalidEnvelope:
		return ErrInvalidEnvelope

	case ErrorCodeInvalidPayload:
		return ErrInvalidPayload

	case ErrorCodeInvalidChecksum:
		return ErrInvalidChecksum

	case ErrorCodeUnsupportedVersion:
		return ErrIncompatibleVersion

	case ErrorCodeUnsupportedEncoding:
		return ErrUnsupportedEncoding

	case ErrorCodeUnsupportedCompression:
		return ErrUnsupportedCompression

	case ErrorCodeHandshakeRequired:
		return ErrHandshakeRequired

	case ErrorCodeHandshakeRejected:
		return ErrHandshakeRejected

	case ErrorCodeSessionClosed:
		return ErrSessionClosed

	case ErrorCodeSequenceViolation:
		return ErrSequenceViolation

	case ErrorCodeAcknowledgement:
		return ErrAcknowledgement

	case ErrorCodeAuthenticationFailed:
		return ErrAuthentication

	case ErrorCodePermissionDenied:
		return ErrAuthorization

	case ErrorCodeRateLimited:
		return ErrRateLimited

	case ErrorCodeTemporaryUnavailable:
		return ErrTemporaryUnavailable

	default:
		return ErrProtocol
	}
}

// ErrorFromRemote constructs a local ProtocolError from information received
// from a remote peer.
//
// The local cause is mapped to the corresponding sentinel so errors.Is can be
// used by callers.
func ErrorFromRemote(
	code ErrorCode,
	message string,
	retryable bool,
	correlationID string,
	details map[string]string,
) *ProtocolError {
	if !code.Valid() {
		code = ErrorCodeInternal
	}

	protocolErr := ProtocolError{
		Code:          code,
		Message:       strings.TrimSpace(message),
		Retryable:     retryable,
		CorrelationID: strings.TrimSpace(correlationID),
		Details:       cloneStringMap(details),
		cause:         SentinelFromCode(code),
	}

	if protocolErr.Message == "" {
		protocolErr.Message = string(code)
	}

	return &protocolErr
}

func cloneStringMap(
	input map[string]string,
) map[string]string {
	if input == nil {
		return nil
	}

	output := make(
		map[string]string,
		len(input),
	)

	for key, value := range input {
		output[key] = value
	}

	return output
}
