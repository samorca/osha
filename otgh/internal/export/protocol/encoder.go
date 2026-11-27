package protocol

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// EncoderConfig controls OTGH protocol encoding.
type EncoderConfig struct {
	// MaxFrameSize is the maximum encoded frame body size.
	MaxFrameSize uint64

	// MaxPayloadSize limits the uncompressed application payload.
	MaxPayloadSize uint64

	// GZIPLevel controls gzip compression when an envelope requests it.
	//
	// Valid values are those accepted by compress/gzip.
	GZIPLevel int
}

// DefaultEncoderConfig returns production-oriented defaults.
func DefaultEncoderConfig() EncoderConfig {
	return EncoderConfig{
		MaxFrameSize:   DefaultMaxFrameSize,
		MaxPayloadSize: 32 * 1024 * 1024,
		GZIPLevel:      gzip.DefaultCompression,
	}
}

// Validate validates encoder configuration.
func (c EncoderConfig) Validate() error {
	if c.MaxFrameSize == 0 {
		return errors.New(
			"encoder max frame size must be greater than zero",
		)
	}

	if c.MaxPayloadSize == 0 {
		return errors.New(
			"encoder max payload size must be greater than zero",
		)
	}

	if _, err := gzip.NewWriterLevel(
		io.Discard,
		c.GZIPLevel,
	); err != nil {
		return fmt.Errorf(
			"invalid gzip compression level %d: %w",
			c.GZIPLevel,
			err,
		)
	}

	return nil
}

// Encoder converts OTGH Envelopes into validated Frames and serialized wire
// bytes.
//
// Encoder contains immutable configuration and is safe for concurrent use.
type Encoder struct {
	config EncoderConfig
}

// NewEncoder creates an encoder.
func NewEncoder(
	config EncoderConfig,
) (*Encoder, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create protocol encoder: %w",
			err,
		)
	}

	return &Encoder{
		config: config,
	}, nil
}

// Config returns a copy of the encoder configuration.
func (e *Encoder) Config() EncoderConfig {
	if e == nil {
		return EncoderConfig{}
	}

	return e.config
}

// Encode converts an envelope into one complete serialized OTGH frame.
//
// The returned bytes contain:
//
//	fixed frame header
//		+
//	encoded envelope body
//
// Encode does not modify the caller's envelope.
func (e *Encoder) Encode(
	envelope Envelope,
) ([]byte, error) {
	if e == nil {
		return nil, errors.New(
			"protocol encoder is nil",
		)
	}

	frame, err := e.EncodeFrame(envelope)
	if err != nil {
		return nil, err
	}

	encoded, err := frame.MarshalBinary(
		e.config.MaxFrameSize,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"serialize protocol frame: %w",
			err,
		)
	}

	return encoded, nil
}

// EncodeFrame converts an Envelope into a Frame without serializing the final
// frame header/body combination.
//
// This is useful for tests and transports that operate on Frame directly.
func (e *Encoder) EncodeFrame(
	envelope Envelope,
) (Frame, error) {
	if e == nil {
		return Frame{}, errors.New(
			"protocol encoder is nil",
		)
	}

	prepared, err := e.prepareEnvelope(
		envelope,
	)
	if err != nil {
		return Frame{}, err
	}

	body, err := marshalEnvelope(prepared)
	if err != nil {
		return Frame{}, fmt.Errorf(
			"encode envelope: %w",
			err,
		)
	}

	if uint64(len(body)) > e.config.MaxFrameSize {
		return Frame{}, fmt.Errorf(
			"%w: encoded envelope is %d bytes, maximum is %d",
			ErrFrameTooLarge,
			len(body),
			e.config.MaxFrameSize,
		)
	}

	flags := frameFlagsForEnvelope(prepared)

	frame, err := NewFrame(
		body,
		flags,
		e.config.MaxFrameSize,
	)
	if err != nil {
		return Frame{}, fmt.Errorf(
			"create protocol frame: %w",
			err,
		)
	}

	return frame, nil
}

// EncodeTo writes one complete serialized OTGH frame to writer.
//
// The method performs a full write check so a writer that returns a short
// write without an error is not silently treated as successful.
func (e *Encoder) EncodeTo(
	writer io.Writer,
	envelope Envelope,
) error {
	if e == nil {
		return errors.New(
			"protocol encoder is nil",
		)
	}

	if writer == nil {
		return errors.New(
			"protocol writer is nil",
		)
	}

	encoded, err := e.Encode(envelope)
	if err != nil {
		return err
	}

	if err := writeAll(
		writer,
		encoded,
	); err != nil {
		return fmt.Errorf(
			"write encoded protocol frame: %w",
			err,
		)
	}

	return nil
}

// prepareEnvelope performs payload transformation required by the envelope's
// declared compression.
//
// An Envelope entering this method is treated as containing its logical,
// uncompressed payload. The returned copy contains the wire payload.
//
// After transformation, integrity metadata is recalculated so the digest
// protects the exact payload bytes transmitted inside the envelope.
func (e *Encoder) prepareEnvelope(
	envelope Envelope,
) (Envelope, error) {
	if envelope.Version.Validate() != nil {
		return Envelope{}, fmt.Errorf(
			"%w: invalid protocol version",
			ErrInvalidEnvelope,
		)
	}

	if uint64(len(envelope.Payload)) >
		e.config.MaxPayloadSize {
		return Envelope{}, fmt.Errorf(
			"%w: payload is %d bytes, maximum is %d",
			ErrInvalidPayload,
			len(envelope.Payload),
			e.config.MaxPayloadSize,
		)
	}

	prepared := envelope.Clone()

	switch prepared.Compression {
	case CompressionNone:
		if err := prepared.SetPayload(
			prepared.Payload,
		); err != nil {
			return Envelope{}, err
		}

	case CompressionGZIP:
		compressed, err := compressGZIP(
			prepared.Payload,
			e.config.GZIPLevel,
		)
		if err != nil {
			return Envelope{}, fmt.Errorf(
				"compress protocol payload: %w",
				err,
			)
		}

		if err := prepared.SetPayload(
			compressed,
		); err != nil {
			return Envelope{}, err
		}

	default:
		return Envelope{}, fmt.Errorf(
			"%w: %q",
			ErrUnsupportedCompression,
			prepared.Compression,
		)
	}

	if err := prepared.Validate(); err != nil {
		return Envelope{}, fmt.Errorf(
			"prepared envelope validation failed: %w",
			err,
		)
	}

	return prepared, nil
}

func marshalEnvelope(
	envelope Envelope,
) ([]byte, error) {
	switch envelope.Encoding {
	case EncodingJSON:
		encoded, err := json.Marshal(envelope)
		if err != nil {
			return nil, fmt.Errorf(
				"marshal JSON envelope: %w",
				err,
			)
		}

		return encoded, nil

	case EncodingBinary:
		// Binary envelope encoding is intentionally reserved for a future
		// explicitly specified wire representation. We do not silently map
		// "binary" to JSON because that would make the protocol contract
		// ambiguous.
		return nil, fmt.Errorf(
			"%w: binary envelope encoding is not implemented",
			ErrUnsupportedEncoding,
		)

	default:
		return nil, fmt.Errorf(
			"%w: %q",
			ErrUnsupportedEncoding,
			envelope.Encoding,
		)
	}
}

func frameFlagsForEnvelope(
	envelope Envelope,
) FrameFlags {
	var flags FrameFlags

	switch envelope.MessageType {
	case MessageTypeHandshake,
		MessageTypeHandshakeResult,
		MessageTypeAcknowledgement,
		MessageTypeError:
		flags |= FrameFlags(
			FrameFlagControl,
		)
	}

	if envelope.Compression != CompressionNone {
		flags |= FrameFlags(
			FrameFlagCompressed,
		)
	}

	// Every current v1 envelope is encoded into exactly one frame.
	flags |= FrameFlags(
		FrameFlagFinal,
	)

	return flags
}

func compressGZIP(
	payload []byte,
	level int,
) ([]byte, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: cannot compress empty payload",
			ErrInvalidPayload,
		)
	}

	var buffer bytes.Buffer

	writer, err := gzip.NewWriterLevel(
		&buffer,
		level,
	)
	if err != nil {
		return nil, err
	}

	if _, err := writer.Write(payload); err != nil {
		_ = writer.Close()

		return nil, fmt.Errorf(
			"write gzip payload: %w",
			err,
		)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf(
			"finalize gzip payload: %w",
			err,
		)
	}

	return append(
		[]byte(nil),
		buffer.Bytes()...,
	), nil
}

// writeAll writes the complete byte slice or returns an error.
func writeAll(
	writer io.Writer,
	data []byte,
) error {
	for len(data) > 0 {
		written, err := writer.Write(data)

		if written < 0 ||
			written > len(data) {
			return fmt.Errorf(
				"invalid writer count %d for %d remaining bytes",
				written,
				len(data),
			)
		}

		if written > 0 {
			data = data[written:]
		}

		if err != nil {
			return err
		}

		if written == 0 {
			return io.ErrShortWrite
		}
	}

	return nil
}
