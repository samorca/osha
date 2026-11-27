package protocol

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
)

const (
	// FrameMagic identifies OTGH protocol frames on a byte stream.
	//
	// These bytes belong to the OTGH-native wire format and allow a decoder
	// to reject unrelated or corrupted input early.
	FrameMagic uint32 = 0x4F544748 // ASCII: OTGH

	// FrameHeaderSize is the fixed number of bytes preceding a frame body.
	//
	// Wire layout:
	//
	//   0..3    magic
	//   4       flags
	//   5       header format version
	//   6..7    reserved
	//   8..15   body length
	//   16..19  body CRC32C
	//
	// All integer fields use network byte order (big endian).
	FrameHeaderSize = 20

	// FrameHeaderVersion identifies the framing structure itself.
	//
	// This is separate from the OTGH protocol Version carried inside the
	// envelope. Framing can therefore evolve independently from message
	// semantics.
	FrameHeaderVersion uint8 = 1

	// DefaultMaxFrameSize limits the frame body accepted by default.
	//
	// Applications may configure a lower or higher limit according to their
	// deployment requirements.
	DefaultMaxFrameSize uint64 = 64 * 1024 * 1024
)

// FrameFlag describes transport-level characteristics of a frame.
type FrameFlag uint8

const (
	FrameFlagNone FrameFlag = 0

	// FrameFlagControl marks protocol-control traffic such as handshake,
	// acknowledgement, or protocol-error messages.
	FrameFlagControl FrameFlag = 1 << iota

	// FrameFlagCompressed indicates that the envelope/payload representation
	// carried by the frame includes compressed data.
	FrameFlagCompressed

	// FrameFlagFinal indicates the final frame belonging to a logical
	// operation when future protocol features introduce multi-frame
	// exchanges.
	FrameFlagFinal
)

const knownFrameFlags = FrameFlagControl |
	FrameFlagCompressed |
	FrameFlagFinal

// FrameFlags represents a combination of FrameFlag values.
type FrameFlags uint8

// Has reports whether a flag is set.
func (f FrameFlags) Has(flag FrameFlag) bool {
	return uint8(f)&uint8(flag) != 0
}

// Valid reports whether no unknown flag bits are set.
func (f FrameFlags) Valid() bool {
	return uint8(f)&^uint8(knownFrameFlags) == 0
}

// FrameHeader describes the fixed binary header that precedes every frame
// body.
type FrameHeader struct {
	Magic uint32

	Flags FrameFlags

	HeaderVersion uint8

	BodyLength uint64

	Checksum uint32
}

// Validate verifies the frame header against a caller-supplied maximum body
// size.
func (h FrameHeader) Validate(maxBodySize uint64) error {
	if h.Magic != FrameMagic {
		return fmt.Errorf(
			"%w: invalid magic 0x%08x",
			ErrInvalidFrame,
			h.Magic,
		)
	}

	if h.HeaderVersion != FrameHeaderVersion {
		return fmt.Errorf(
			"%w: unsupported header version %d",
			ErrInvalidFrame,
			h.HeaderVersion,
		)
	}

	if !h.Flags.Valid() {
		return fmt.Errorf(
			"%w: unknown frame flags 0x%02x",
			ErrInvalidFrame,
			uint8(h.Flags),
		)
	}

	if h.BodyLength == 0 {
		return fmt.Errorf(
			"%w: body length must be greater than zero",
			ErrInvalidFrame,
		)
	}

	if maxBodySize == 0 {
		return fmt.Errorf(
			"%w: maximum body size must be greater than zero",
			ErrInvalidFrame,
		)
	}

	if h.BodyLength > maxBodySize {
		return fmt.Errorf(
			"%w: frame body is %d bytes, maximum is %d",
			ErrFrameTooLarge,
			h.BodyLength,
			maxBodySize,
		)
	}

	return nil
}

// Frame is one complete OTGH wire unit.
//
// Body contains an encoded Envelope. frame.go deliberately does not decode
// that body. Encoding and decoding belong to encoder.go and decoder.go.
type Frame struct {
	Header FrameHeader

	Body []byte
}

// NewFrame constructs a frame and calculates its transport-level checksum.
func NewFrame(
	body []byte,
	flags FrameFlags,
	maxBodySize uint64,
) (Frame, error) {
	if len(body) == 0 {
		return Frame{}, fmt.Errorf(
			"%w: frame body is empty",
			ErrInvalidFrame,
		)
	}

	if maxBodySize == 0 {
		return Frame{}, fmt.Errorf(
			"%w: maximum body size must be greater than zero",
			ErrInvalidFrame,
		)
	}

	if uint64(len(body)) > maxBodySize {
		return Frame{}, fmt.Errorf(
			"%w: frame body is %d bytes, maximum is %d",
			ErrFrameTooLarge,
			len(body),
			maxBodySize,
		)
	}

	if !flags.Valid() {
		return Frame{}, fmt.Errorf(
			"%w: unknown frame flags 0x%02x",
			ErrInvalidFrame,
			uint8(flags),
		)
	}

	bodyCopy := append([]byte(nil), body...)

	frame := Frame{
		Header: FrameHeader{
			Magic:         FrameMagic,
			Flags:         flags,
			HeaderVersion: FrameHeaderVersion,
			BodyLength:    uint64(len(bodyCopy)),
			Checksum:      calculateFrameChecksum(bodyCopy),
		},
		Body: bodyCopy,
	}

	if err := frame.Validate(maxBodySize); err != nil {
		return Frame{}, err
	}

	return frame, nil
}

// Validate verifies header consistency, body length, configured size limits,
// and transport-level checksum.
func (f Frame) Validate(maxBodySize uint64) error {
	if err := f.Header.Validate(maxBodySize); err != nil {
		return err
	}

	if uint64(len(f.Body)) != f.Header.BodyLength {
		return fmt.Errorf(
			"%w: header declares %d body bytes but frame contains %d",
			ErrInvalidFrame,
			f.Header.BodyLength,
			len(f.Body),
		)
	}

	expected := calculateFrameChecksum(f.Body)

	if f.Header.Checksum != expected {
		return fmt.Errorf(
			"%w: frame checksum mismatch",
			ErrInvalidChecksum,
		)
	}

	return nil
}

// Size returns the complete serialized frame size.
func (f Frame) Size() uint64 {
	return uint64(FrameHeaderSize) + uint64(len(f.Body))
}

// BodySize returns the encoded body size.
func (f Frame) BodySize() uint64 {
	return uint64(len(f.Body))
}

// Clone returns a defensive copy.
func (f Frame) Clone() Frame {
	cloned := f

	if f.Body != nil {
		cloned.Body = append([]byte(nil), f.Body...)
	}

	return cloned
}

// MarshalHeader serializes a validated frame header into its fixed binary
// representation.
func MarshalHeader(
	header FrameHeader,
	maxBodySize uint64,
) ([]byte, error) {
	if err := header.Validate(maxBodySize); err != nil {
		return nil, err
	}

	data := make([]byte, FrameHeaderSize)

	binary.BigEndian.PutUint32(
		data[0:4],
		header.Magic,
	)

	data[4] = byte(header.Flags)
	data[5] = header.HeaderVersion

	// Bytes 6 and 7 are reserved and MUST be zero in framing version 1.
	data[6] = 0
	data[7] = 0

	binary.BigEndian.PutUint64(
		data[8:16],
		header.BodyLength,
	)

	binary.BigEndian.PutUint32(
		data[16:20],
		header.Checksum,
	)

	return data, nil
}

// UnmarshalHeader parses and validates one fixed-size frame header.
func UnmarshalHeader(
	data []byte,
	maxBodySize uint64,
) (FrameHeader, error) {
	if len(data) != FrameHeaderSize {
		return FrameHeader{}, fmt.Errorf(
			"%w: header size is %d, expected %d",
			ErrInvalidFrame,
			len(data),
			FrameHeaderSize,
		)
	}

	if data[6] != 0 || data[7] != 0 {
		return FrameHeader{}, fmt.Errorf(
			"%w: reserved header bytes must be zero",
			ErrInvalidFrame,
		)
	}

	header := FrameHeader{
		Magic: binary.BigEndian.Uint32(
			data[0:4],
		),
		Flags: FrameFlags(data[4]),
		HeaderVersion: data[5],
		BodyLength: binary.BigEndian.Uint64(
			data[8:16],
		),
		Checksum: binary.BigEndian.Uint32(
			data[16:20],
		),
	}

	if err := header.Validate(maxBodySize); err != nil {
		return FrameHeader{}, err
	}

	return header, nil
}

// MarshalBinary serializes a complete validated frame.
func (f Frame) MarshalBinary(
	maxBodySize uint64,
) ([]byte, error) {
	if err := f.Validate(maxBodySize); err != nil {
		return nil, err
	}

	header, err := MarshalHeader(
		f.Header,
		maxBodySize,
	)
	if err != nil {
		return nil, err
	}

	totalSize := FrameHeaderSize + len(f.Body)

	output := make([]byte, totalSize)

	copy(
		output[:FrameHeaderSize],
		header,
	)

	copy(
		output[FrameHeaderSize:],
		f.Body,
	)

	return output, nil
}

// UnmarshalFrame parses one complete frame from data.
//
// data must contain exactly one frame. Stream-oriented parsing will be handled
// by decoder.go, which can read exactly FrameHeaderSize bytes followed by the
// advertised body length.
func UnmarshalFrame(
	data []byte,
	maxBodySize uint64,
) (Frame, error) {
	if len(data) < FrameHeaderSize {
		return Frame{}, fmt.Errorf(
			"%w: frame contains %d bytes, smaller than header size %d",
			ErrInvalidFrame,
			len(data),
			FrameHeaderSize,
		)
	}

	header, err := UnmarshalHeader(
		data[:FrameHeaderSize],
		maxBodySize,
	)
	if err != nil {
		return Frame{}, err
	}

	expectedSize := uint64(FrameHeaderSize) +
		header.BodyLength

	if uint64(len(data)) != expectedSize {
		return Frame{}, fmt.Errorf(
			"%w: frame contains %d bytes, expected %d",
			ErrInvalidFrame,
			len(data),
			expectedSize,
		)
	}

	body := append(
		[]byte(nil),
		data[FrameHeaderSize:]...,
	)

	frame := Frame{
		Header: header,
		Body:   body,
	}

	if err := frame.Validate(maxBodySize); err != nil {
		return Frame{}, err
	}

	return frame, nil
}

// Equal reports whether two frames contain identical framing metadata and
// body bytes.
func (f Frame) Equal(other Frame) bool {
	return f.Header == other.Header &&
		bytes.Equal(f.Body, other.Body)
}

// SetBody replaces the body and recalculates its dependent header fields.
func (f *Frame) SetBody(
	body []byte,
	maxBodySize uint64,
) error {
	if f == nil {
		return errors.New("frame is nil")
	}

	if len(body) == 0 {
		return fmt.Errorf(
			"%w: frame body is empty",
			ErrInvalidFrame,
		)
	}

	if maxBodySize == 0 {
		return fmt.Errorf(
			"%w: maximum body size must be greater than zero",
			ErrInvalidFrame,
		)
	}

	if uint64(len(body)) > maxBodySize {
		return fmt.Errorf(
			"%w: frame body is %d bytes, maximum is %d",
			ErrFrameTooLarge,
			len(body),
			maxBodySize,
		)
	}

	f.Body = append(f.Body[:0], body...)

	f.Header.Magic = FrameMagic
	f.Header.HeaderVersion = FrameHeaderVersion
	f.Header.BodyLength = uint64(len(f.Body))
	f.Header.Checksum = calculateFrameChecksum(f.Body)

	return f.Validate(maxBodySize)
}

// SetFlags replaces the frame flags after validating them.
func (f *Frame) SetFlags(flags FrameFlags) error {
	if f == nil {
		return errors.New("frame is nil")
	}

	if !flags.Valid() {
		return fmt.Errorf(
			"%w: unknown frame flags 0x%02x",
			ErrInvalidFrame,
			uint8(flags),
		)
	}

	f.Header.Flags = flags

	return nil
}

// AddFlag adds one known flag.
func (f *Frame) AddFlag(flag FrameFlag) error {
	if f == nil {
		return errors.New("frame is nil")
	}

	if uint8(flag)&^uint8(knownFrameFlags) != 0 ||
		flag == FrameFlagNone {
		return fmt.Errorf(
			"%w: invalid frame flag 0x%02x",
			ErrInvalidFrame,
			uint8(flag),
		)
	}

	flags := FrameFlags(
		uint8(f.Header.Flags) |
			uint8(flag),
	)

	return f.SetFlags(flags)
}

// RemoveFlag removes a flag.
func (f *Frame) RemoveFlag(flag FrameFlag) error {
	if f == nil {
		return errors.New("frame is nil")
	}

	if uint8(flag)&^uint8(knownFrameFlags) != 0 {
		return fmt.Errorf(
			"%w: invalid frame flag 0x%02x",
			ErrInvalidFrame,
			uint8(flag),
		)
	}

	flags := FrameFlags(
		uint8(f.Header.Flags) &
			^uint8(flag),
	)

	return f.SetFlags(flags)
}

// calculateFrameChecksum computes a CRC32C checksum for framing-level
// corruption detection.
//
// Envelope payload integrity remains separately protected by the SHA-256
// digest defined in envelope.go. The two mechanisms serve different layers:
//
//   frame CRC32C:
//       fast transport/frame corruption detection.
//
//   envelope SHA-256:
//       payload integrity verification after decoding.
func calculateFrameChecksum(
	body []byte,
) uint32 {
	return crc32.Checksum(
		body,
		crc32.MakeTable(
			crc32.Castagnoli,
		),
	)
}
