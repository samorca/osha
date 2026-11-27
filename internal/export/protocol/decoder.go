package protocol

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// DecoderConfig controls OTGH protocol decoding and resource limits.
type DecoderConfig struct {
	// MaxFrameSize is the maximum encoded frame body accepted from a peer.
	MaxFrameSize uint64

	// MaxPayloadSize is the maximum logical payload size after
	// decompression.
	//
	// This is important because a small compressed payload can expand into
	// significantly more data.
	MaxPayloadSize uint64
}

// DefaultDecoderConfig returns conservative decoder defaults.
func DefaultDecoderConfig() DecoderConfig {
	return DecoderConfig{
		MaxFrameSize:   DefaultMaxFrameSize,
		MaxPayloadSize: 32 * 1024 * 1024,
	}
}

// Validate verifies decoder configuration.
func (c DecoderConfig) Validate() error {
	if c.MaxFrameSize == 0 {
		return errors.New(
			"decoder max frame size must be greater than zero",
		)
	}

	if c.MaxPayloadSize == 0 {
		return errors.New(
			"decoder max payload size must be greater than zero",
		)
	}

	return nil
}

// Decoder converts OTGH wire frames back into logical Envelopes.
//
// Decoder contains immutable configuration and is safe for concurrent use.
type Decoder struct {
	config DecoderConfig
}

// NewDecoder creates a protocol decoder.
func NewDecoder(
	config DecoderConfig,
) (*Decoder, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create protocol decoder: %w",
			err,
		)
	}

	return &Decoder{
		config: config,
	}, nil
}

// Config returns a copy of the decoder configuration.
func (d *Decoder) Config() DecoderConfig {
	if d == nil {
		return DecoderConfig{}
	}

	return d.config
}

// Decode parses exactly one complete serialized OTGH frame.
//
// The returned Envelope contains the logical uncompressed payload.
//
// Wire-level integrity is validated before decompression:
//   - frame CRC32C;
//   - envelope SHA-256 digest.
//
// This means the digest is checked against the exact payload bytes received
// from the peer before any transformation is performed.
func (d *Decoder) Decode(
	data []byte,
) (Envelope, error) {
	if d == nil {
		return Envelope{}, errors.New(
			"protocol decoder is nil",
		)
	}

	frame, err := UnmarshalFrame(
		data,
		d.config.MaxFrameSize,
	)
	if err != nil {
		return Envelope{}, fmt.Errorf(
			"decode protocol frame: %w",
			err,
		)
	}

	return d.DecodeFrame(frame)
}

// DecodeFrame converts a validated Frame into a logical Envelope.
func (d *Decoder) DecodeFrame(
	frame Frame,
) (Envelope, error) {
	if d == nil {
		return Envelope{}, errors.New(
			"protocol decoder is nil",
		)
	}

	if err := frame.Validate(
		d.config.MaxFrameSize,
	); err != nil {
		return Envelope{}, fmt.Errorf(
			"validate protocol frame: %w",
			err,
		)
	}

	envelope, err := unmarshalEnvelope(
		frame.Body,
	)
	if err != nil {
		return Envelope{}, fmt.Errorf(
			"decode protocol envelope: %w",
			err,
		)
	}

	if err := envelope.Validate(); err != nil {
		return Envelope{}, fmt.Errorf(
			"validate protocol envelope: %w",
			err,
		)
	}

	if err := validateFrameEnvelopeFlags(
		frame.Header.Flags,
		envelope,
	); err != nil {
		return Envelope{}, err
	}

	logical, err := d.restoreLogicalPayload(
		envelope,
	)
	if err != nil {
		return Envelope{}, err
	}

	return logical, nil
}

// DecodeFrom reads exactly one OTGH frame from reader.
//
// The method first reads the fixed-size frame header, validates the advertised
// body length against configured limits, and only then allocates memory for
// the body.
//
// This avoids trusting an unbounded peer-controlled size before validation.
func (d *Decoder) DecodeFrom(
	reader io.Reader,
) (Envelope, error) {
	if d == nil {
		return Envelope{}, errors.New(
			"protocol decoder is nil",
		)
	}

	if reader == nil {
		return Envelope{}, errors.New(
			"protocol reader is nil",
		)
	}

	headerBytes := make(
		[]byte,
		FrameHeaderSize,
	)

	if _, err := io.ReadFull(
		reader,
		headerBytes,
	); err != nil {
		return Envelope{}, fmt.Errorf(
			"read protocol frame header: %w",
			err,
		)
	}

	header, err := UnmarshalHeader(
		headerBytes,
		d.config.MaxFrameSize,
	)
	if err != nil {
		return Envelope{}, fmt.Errorf(
			"decode protocol frame header: %w",
			err,
		)
	}

	// UnmarshalHeader has already enforced MaxFrameSize, so converting this
	// body size into an allocation is bounded by decoder configuration.
	body := make(
		[]byte,
		int(header.BodyLength),
	)

	if _, err := io.ReadFull(
		reader,
		body,
	); err != nil {
		return Envelope{}, fmt.Errorf(
			"read protocol frame body: %w",
			err,
		)
	}

	frame := Frame{
		Header: header,
		Body:   body,
	}

	return d.DecodeFrame(frame)
}

func unmarshalEnvelope(
	data []byte,
) (Envelope, error) {
	if len(data) == 0 {
		return Envelope{}, fmt.Errorf(
			"%w: encoded envelope is empty",
			ErrInvalidEnvelope,
		)
	}

	decoder := json.NewDecoder(
		bytes.NewReader(data),
	)

	// Strict field names help catch protocol-version/schema mismatches instead
	// of silently ignoring fields that this implementation does not know.
	decoder.DisallowUnknownFields()

	var envelope Envelope

	if err := decoder.Decode(
		&envelope,
	); err != nil {
		return Envelope{}, fmt.Errorf(
			"%w: decode JSON envelope: %v",
			ErrMalformedMessage,
			err,
		)
	}

	if err := ensureJSONDecoderEOF(
		decoder,
	); err != nil {
		return Envelope{}, err
	}

	return envelope, nil
}

// restoreLogicalPayload reverses wire compression and returns a copy whose
// Payload is suitable for application processing.
//
// The logical envelope's Compression value remains unchanged. It describes
// how the payload arrived on the wire.
//
// Integrity continues to describe the original wire payload. Callers should
// not call logical.Validate() after decompression because Validate verifies
// the wire digest against Payload. Structural validation has already occurred
// before this transformation.
func (d *Decoder) restoreLogicalPayload(
	envelope Envelope,
) (Envelope, error) {
	switch envelope.Compression {
	case CompressionNone:
		if uint64(len(envelope.Payload)) >
			d.config.MaxPayloadSize {
			return Envelope{}, fmt.Errorf(
				"%w: payload is %d bytes, maximum is %d",
				ErrInvalidPayload,
				len(envelope.Payload),
				d.config.MaxPayloadSize,
			)
		}

		return envelope.Clone(), nil

	case CompressionGZIP:
		payload, err := decompressGZIPLimited(
			envelope.Payload,
			d.config.MaxPayloadSize,
		)
		if err != nil {
			return Envelope{}, fmt.Errorf(
				"decompress protocol payload: %w",
				err,
			)
		}

		logical := envelope.Clone()
		logical.Payload = payload

		return logical, nil

	default:
		return Envelope{}, fmt.Errorf(
			"%w: %q",
			ErrUnsupportedCompression,
			envelope.Compression,
		)
	}
}

func validateFrameEnvelopeFlags(
	flags FrameFlags,
	envelope Envelope,
) error {
	expectedControl := isControlMessage(
		envelope.MessageType,
	)

	if flags.Has(FrameFlagControl) != expectedControl {
		return fmt.Errorf(
			"%w: control flag does not match message type %q",
			ErrInvalidFrame,
			envelope.MessageType,
		)
	}

	expectedCompressed :=
		envelope.Compression != CompressionNone

	if flags.Has(FrameFlagCompressed) !=
		expectedCompressed {
		return fmt.Errorf(
			"%w: compressed flag does not match envelope compression %q",
			ErrInvalidFrame,
			envelope.Compression,
		)
	}

	// Protocol v1 currently requires one complete envelope per frame.
	if !flags.Has(FrameFlagFinal) {
		return fmt.Errorf(
			"%w: protocol v1 frame must carry final flag",
			ErrInvalidFrame,
		)
	}

	return nil
}

func isControlMessage(
	messageType MessageType,
) bool {
	switch messageType {
	case MessageTypeHandshake,
		MessageTypeHandshakeResult,
		MessageTypeAcknowledgement,
		MessageTypeError:
		return true

	default:
		return false
	}
}

func decompressGZIPLimited(
	compressed []byte,
	maxPayloadSize uint64,
) ([]byte, error) {
	if len(compressed) == 0 {
		return nil, fmt.Errorf(
			"%w: compressed payload is empty",
			ErrInvalidPayload,
		)
	}

	if maxPayloadSize == 0 {
		return nil, errors.New(
			"maximum payload size must be greater than zero",
		)
	}

	reader, err := gzip.NewReader(
		bytes.NewReader(compressed),
	)
	if err != nil {
		return nil, fmt.Errorf(
			"%w: invalid gzip payload: %v",
			ErrInvalidPayload,
			err,
		)
	}
	defer reader.Close()

	// OTGH accepts exactly one gzip member for a protocol payload.
	//
	// Disabling multistream mode avoids interpreting concatenated gzip
	// members as one logical application payload.
	reader.Multistream(false)

	// Read one byte beyond the configured maximum so oversized
	// decompression can be detected deterministically.
	limited := io.LimitReader(
		reader,
		int64(maxPayloadSize)+1,
	)

	payload, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf(
			"%w: read gzip payload: %v",
			ErrInvalidPayload,
			err,
		)
	}

	if uint64(len(payload)) >
		maxPayloadSize {
		return nil, fmt.Errorf(
			"%w: decompressed payload exceeds maximum size %d",
			ErrInvalidPayload,
			maxPayloadSize,
		)
	}

	// Continue to EOF when the payload did not hit the limit. This ensures
	// gzip validates its own end-of-stream checksum and trailer.
	//
	// Go's gzip reader reports checksum errors while reading through the end
	// of the gzip stream.
	var tail [1]byte

	n, err := reader.Read(tail[:])

	switch {
	case err == nil && n > 0:
		return nil, fmt.Errorf(
			"%w: decompressed payload exceeds maximum size %d",
			ErrInvalidPayload,
			maxPayloadSize,
		)

	case errors.Is(err, io.EOF):
		// Expected.

	case err != nil:
		return nil, fmt.Errorf(
			"%w: finalize gzip payload: %v",
			ErrInvalidPayload,
			err,
		)
	}

	return append(
		[]byte(nil),
		payload...,
	), nil
}

func ensureJSONDecoderEOF(
	decoder *json.Decoder,
) error {
	var extra json.RawMessage

	err := decoder.Decode(&extra)

	if errors.Is(err, io.EOF) {
		return nil
	}

	if err == nil {
		return fmt.Errorf(
			"%w: multiple JSON values in envelope body",
			ErrMalformedMessage,
		)
	}

	return fmt.Errorf(
		"%w: trailing envelope data: %v",
		ErrMalformedMessage,
		err,
	)
}
