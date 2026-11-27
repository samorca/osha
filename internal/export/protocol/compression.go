package protocol

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
)

// Compressor defines the compression contract used by the OTGH protocol.
//
// Implementations must be safe for concurrent use unless explicitly
// documented otherwise.
type Compressor interface {
	// Algorithm returns the Compression identifier handled by this
	// implementation.
	Algorithm() Compression

	// Compress returns a new byte slice containing the compressed payload.
	Compress(payload []byte) ([]byte, error)

	// Decompress restores a payload while enforcing the supplied maximum
	// uncompressed size.
	Decompress(
		payload []byte,
		maxOutputBytes uint64,
	) ([]byte, error)
}

// CompressionConfig controls OTGH compression behaviour.
type CompressionConfig struct {
	// GZIPLevel is the compression level passed to Go's gzip writer.
	GZIPLevel int
}

// DefaultCompressionConfig returns the default protocol compression
// configuration.
func DefaultCompressionConfig() CompressionConfig {
	return CompressionConfig{
		GZIPLevel: gzip.DefaultCompression,
	}
}

// Validate verifies compression configuration.
func (c CompressionConfig) Validate() error {
	writer, err := gzip.NewWriterLevel(
		io.Discard,
		c.GZIPLevel,
	)
	if err != nil {
		return fmt.Errorf(
			"invalid gzip compression level %d: %w",
			c.GZIPLevel,
			err,
		)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf(
			"validate gzip writer: %w",
			err,
		)
	}

	return nil
}

// CompressionRegistry owns the compression algorithms supported by this OTGH
// protocol implementation.
//
// The registry is immutable after construction and therefore safe for
// concurrent use.
type CompressionRegistry struct {
	compressors map[Compression]Compressor
}

// NewCompressionRegistry creates the standard OTGH compression registry.
func NewCompressionRegistry(
	config CompressionConfig,
) (*CompressionRegistry, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create compression registry: %w",
			err,
		)
	}

	registry := &CompressionRegistry{
		compressors: make(
			map[Compression]Compressor,
			2,
		),
	}

	registry.compressors[CompressionNone] =
		noCompression{}

	registry.compressors[CompressionGZIP] =
		gzipCompressor{
			level: config.GZIPLevel,
		}

	return registry, nil
}

// Supported reports whether the registry contains an implementation for the
// requested compression algorithm.
func (r *CompressionRegistry) Supported(
	algorithm Compression,
) bool {
	if r == nil {
		return false
	}

	_, exists := r.compressors[algorithm]

	return exists
}

// Algorithms returns the compression algorithms supported by this registry.
func (r *CompressionRegistry) Algorithms() []Compression {
	if r == nil {
		return nil
	}

	// Return a deterministic order rather than map-iteration order.
	algorithms := make(
		[]Compression,
		0,
		2,
	)

	if r.Supported(CompressionNone) {
		algorithms = append(
			algorithms,
			CompressionNone,
		)
	}

	if r.Supported(CompressionGZIP) {
		algorithms = append(
			algorithms,
			CompressionGZIP,
		)
	}

	return algorithms
}

// Compress applies the requested algorithm.
//
// CompressionNone still returns a defensive copy so callers never receive the
// original mutable payload slice.
func (r *CompressionRegistry) Compress(
	algorithm Compression,
	payload []byte,
) ([]byte, error) {
	if r == nil {
		return nil, errors.New(
			"compression registry is nil",
		)
	}

	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: cannot compress empty payload",
			ErrInvalidPayload,
		)
	}

	compressor, exists := r.compressors[algorithm]
	if !exists {
		return nil, fmt.Errorf(
			"%w: %q",
			ErrUnsupportedCompression,
			algorithm,
		)
	}

	output, err := compressor.Compress(payload)
	if err != nil {
		return nil, fmt.Errorf(
			"compress using %q: %w",
			algorithm,
			err,
		)
	}

	if len(output) == 0 {
		return nil, fmt.Errorf(
			"%w: compressor %q produced empty output",
			ErrInvalidPayload,
			algorithm,
		)
	}

	return output, nil
}

// Decompress restores a payload and enforces a maximum output size.
//
// The output limit applies even to CompressionNone so a caller can enforce
// one consistent payload-size policy regardless of compression algorithm.
func (r *CompressionRegistry) Decompress(
	algorithm Compression,
	payload []byte,
	maxOutputBytes uint64,
) ([]byte, error) {
	if r == nil {
		return nil, errors.New(
			"compression registry is nil",
		)
	}

	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: cannot decompress empty payload",
			ErrInvalidPayload,
		)
	}

	if maxOutputBytes == 0 {
		return nil, errors.New(
			"maximum decompressed payload size must be greater than zero",
		)
	}

	compressor, exists := r.compressors[algorithm]
	if !exists {
		return nil, fmt.Errorf(
			"%w: %q",
			ErrUnsupportedCompression,
			algorithm,
		)
	}

	output, err := compressor.Decompress(
		payload,
		maxOutputBytes,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"decompress using %q: %w",
			algorithm,
			err,
		)
	}

	if uint64(len(output)) > maxOutputBytes {
		return nil, fmt.Errorf(
			"%w: decompressed payload is %d bytes, maximum is %d",
			ErrInvalidPayload,
			len(output),
			maxOutputBytes,
		)
	}

	return output, nil
}

// noCompression provides the pass-through implementation.
type noCompression struct{}

func (noCompression) Algorithm() Compression {
	return CompressionNone
}

func (noCompression) Compress(
	payload []byte,
) ([]byte, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: payload is empty",
			ErrInvalidPayload,
		)
	}

	return append(
		[]byte(nil),
		payload...,
	), nil
}

func (noCompression) Decompress(
	payload []byte,
	maxOutputBytes uint64,
) ([]byte, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: payload is empty",
			ErrInvalidPayload,
		)
	}

	if uint64(len(payload)) > maxOutputBytes {
		return nil, fmt.Errorf(
			"%w: payload is %d bytes, maximum is %d",
			ErrInvalidPayload,
			len(payload),
			maxOutputBytes,
		)
	}

	return append(
		[]byte(nil),
		payload...,
	), nil
}

// gzipCompressor implements gzip payload compression.
type gzipCompressor struct {
	level int
}

func (g gzipCompressor) Algorithm() Compression {
	return CompressionGZIP
}

func (g gzipCompressor) Compress(
	payload []byte,
) ([]byte, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: payload is empty",
			ErrInvalidPayload,
		)
	}

	var buffer bytes.Buffer

	writer, err := gzip.NewWriterLevel(
		&buffer,
		g.level,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create gzip writer: %w",
			err,
		)
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

	compressed := buffer.Bytes()

	if len(compressed) == 0 {
		return nil, fmt.Errorf(
			"%w: gzip produced empty output",
			ErrInvalidPayload,
		)
	}

	return append(
		[]byte(nil),
		compressed...,
	), nil
}

func (g gzipCompressor) Decompress(
	payload []byte,
	maxOutputBytes uint64,
) ([]byte, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf(
			"%w: compressed payload is empty",
			ErrInvalidPayload,
		)
	}

	if maxOutputBytes == 0 {
		return nil, errors.New(
			"maximum decompressed payload size must be greater than zero",
		)
	}

	reader, err := gzip.NewReader(
		bytes.NewReader(payload),
	)
	if err != nil {
		return nil, fmt.Errorf(
			"%w: invalid gzip header: %v",
			ErrInvalidPayload,
			err,
		)
	}
	defer reader.Close()

	// OTGH treats one payload as exactly one gzip stream.
	//
	// Concatenated gzip members are not accepted as one protocol payload.
	reader.Multistream(false)

	// LimitReader is given one extra byte. If that extra byte can be read,
	// the logical payload exceeded maxOutputBytes.
	limited := io.LimitReader(
		reader,
		int64(maxOutputBytes)+1,
	)

	output, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf(
			"%w: read gzip payload: %v",
			ErrInvalidPayload,
			err,
		)
	}

	if uint64(len(output)) > maxOutputBytes {
		return nil, fmt.Errorf(
			"%w: decompressed payload exceeds maximum size %d",
			ErrInvalidPayload,
			maxOutputBytes,
		)
	}

	// Read through the gzip trailer so gzip can perform its stream checksum
	// validation.
	var tail [1]byte

	n, err := reader.Read(tail[:])

	switch {
	case err == nil && n > 0:
		return nil, fmt.Errorf(
			"%w: decompressed payload exceeds maximum size %d",
			ErrInvalidPayload,
			maxOutputBytes,
		)

	case errors.Is(err, io.EOF):
		// Expected complete stream.

	case err != nil:
		return nil, fmt.Errorf(
			"%w: finalize gzip payload: %v",
			ErrInvalidPayload,
			err,
		)
	}

	return append(
		[]byte(nil),
		output...,
	), nil
}
