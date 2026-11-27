
package export

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	spoolFormatVersion = 1
	spoolFileExtension = ".otgh-spool"
)

var (
	// ErrSpoolClosed indicates that the spool is no longer available.
	ErrSpoolClosed = errors.New("spool closed")

	// ErrSpoolEntryNotFound indicates that no entry exists for a batch ID.
	ErrSpoolEntryNotFound = errors.New("spool entry not found")

	// ErrSpoolEntryExists indicates that a batch already exists in the spool.
	ErrSpoolEntryExists = errors.New("spool entry already exists")

	// ErrSpoolCorrupt indicates that an entry cannot be trusted or decoded.
	ErrSpoolCorrupt = errors.New("spool entry corrupt")

	// ErrSpoolFull indicates that adding another entry would exceed a
	// configured spool limit.
	ErrSpoolFull = errors.New("spool full")
)

// SpoolConfig controls durable local batch storage.
type SpoolConfig struct {
	// Directory is the local directory used for spool entries.
	Directory string

	// DirectoryMode is applied when the spool directory is created.
	DirectoryMode fs.FileMode

	// FileMode is applied to newly created spool files.
	FileMode fs.FileMode

	// MaxEntries limits the number of pending spool files.
	//
	// Zero disables the entry-count limit.
	MaxEntries int

	// MaxBytes limits the combined on-disk size tracked by the spool.
	//
	// Zero disables the total-size limit.
	MaxBytes int64

	// MaxEntryBytes limits the encoded size of one spool entry.
	MaxEntryBytes int64

	// SyncDirectory requests a directory sync after publishing or deleting
	// an entry.
	//
	// Directory syncing improves durability of filesystem metadata updates,
	// although exact guarantees still depend on the operating system,
	// filesystem, storage device, and mount configuration.
	SyncDirectory bool
}

// DefaultSpoolConfig returns conservative local spool defaults.
func DefaultSpoolConfig(directory string) SpoolConfig {
	return SpoolConfig{
		Directory:     directory,
		DirectoryMode: 0o750,
		FileMode:      0o600,
		MaxEntries:    100_000,
		MaxBytes:      10 * 1024 * 1024 * 1024,
		MaxEntryBytes: 64 * 1024 * 1024,
		SyncDirectory: true,
	}
}

// Validate verifies the spool configuration.
func (c SpoolConfig) Validate() error {
	if strings.TrimSpace(c.Directory) == "" {
		return errors.New("spool directory is required")
	}

	if c.DirectoryMode == 0 {
		return errors.New("spool directory mode is required")
	}

	if c.FileMode == 0 {
		return errors.New("spool file mode is required")
	}

	if c.MaxEntries < 0 {
		return errors.New("spool max entries cannot be negative")
	}

	if c.MaxBytes < 0 {
		return errors.New("spool max bytes cannot be negative")
	}

	if c.MaxEntryBytes <= 0 {
		return errors.New(
			"spool max entry bytes must be greater than zero",
		)
	}

	if c.MaxBytes > 0 && c.MaxEntryBytes > c.MaxBytes {
		return errors.New(
			"spool max entry bytes cannot exceed max bytes",
		)
	}

	return nil
}

// SpoolEntryMetadata describes one durable batch without loading its payload.
type SpoolEntryMetadata struct {
	BatchID       BatchID
	DestinationID DestinationID

	Path string

	CreatedAt time.Time
	StoredAt  time.Time

	SizeBytes int64
}

// SpoolStats is a point-in-time view of durable spool state.
type SpoolStats struct {
	Directory string

	Entries int
	Bytes   int64

	MaxEntries int
	MaxBytes   int64

	Closed bool
}

// spoolEnvelope is the versioned representation persisted on disk.
//
// Batch is encoded with the standard JSON representation defined by the
// transport types. Payload []byte values are encoded safely by encoding/json.
type spoolEnvelope struct {
	Version int `json:"version"`

	StoredAt time.Time `json:"stored_at"`

	Batch Batch `json:"batch"`

	// Checksum is the SHA-256 digest of Batch's canonical JSON produced by
	// encoding/json. It detects truncation and accidental modification.
	Checksum string `json:"checksum"`
}

// Spool persists pending batches as independent files.
//
// Each successful Put publishes one complete entry. Entries can be loaded,
// listed, recovered, acknowledged through Delete, or quarantined when corrupt.
//
// Spool is safe for concurrent use inside one process. It does not provide
// multi-process locking. Only one OTGH process should own a spool directory.
type Spool struct {
	mu sync.RWMutex

	config SpoolConfig

	entries int
	bytes   int64

	closed bool
}

// NewSpool creates or opens a spool directory and scans existing entries.
func NewSpool(config SpoolConfig) (*Spool, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("create spool: %w", err)
	}

	directory, err := filepath.Abs(config.Directory)
	if err != nil {
		return nil, fmt.Errorf(
			"resolve spool directory %q: %w",
			config.Directory,
			err,
		)
	}

	config.Directory = filepath.Clean(directory)

	if err := os.MkdirAll(
		config.Directory,
		config.DirectoryMode,
	); err != nil {
		return nil, fmt.Errorf(
			"create spool directory %q: %w",
			config.Directory,
			err,
		)
	}

	info, err := os.Stat(config.Directory)
	if err != nil {
		return nil, fmt.Errorf(
			"stat spool directory %q: %w",
			config.Directory,
			err,
		)
	}

	if !info.IsDir() {
		return nil, fmt.Errorf(
			"spool path %q is not a directory",
			config.Directory,
		)
	}

	spool := &Spool{
		config: config,
	}

	if err := spool.scanLocked(); err != nil {
		return nil, fmt.Errorf("scan spool: %w", err)
	}

	if config.MaxEntries > 0 &&
		spool.entries > config.MaxEntries {
		return nil, fmt.Errorf(
			"%w: existing entries %d exceed configured maximum %d",
			ErrSpoolFull,
			spool.entries,
			config.MaxEntries,
		)
	}

	if config.MaxBytes > 0 &&
		spool.bytes > config.MaxBytes {
		return nil, fmt.Errorf(
			"%w: existing bytes %d exceed configured maximum %d",
			ErrSpoolFull,
			spool.bytes,
			config.MaxBytes,
		)
	}

	return spool, nil
}

// Put persists a batch.
//
// Put never overwrites an existing entry with the same BatchID. Call Delete
// after acknowledgement before reusing a BatchID.
func (s *Spool) Put(
	ctx context.Context,
	batch Batch,
) error {
	if s == nil {
		return errors.New("spool is nil")
	}

	if ctx == nil {
		return errors.New("spool put context is nil")
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := batch.Validate(); err != nil {
		return fmt.Errorf("spool batch: %w", err)
	}

	encoded, err := encodeSpoolEnvelope(batch, time.Now().UTC())
	if err != nil {
		return fmt.Errorf("encode spool batch %q: %w", batch.ID, err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSpoolClosed
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	entrySize := int64(len(encoded))

	if entrySize > s.config.MaxEntryBytes {
		return fmt.Errorf(
			"%w: batch %q encoded size %d exceeds entry limit %d",
			ErrSpoolFull,
			batch.ID,
			entrySize,
			s.config.MaxEntryBytes,
		)
	}

	if s.config.MaxEntries > 0 &&
		s.entries >= s.config.MaxEntries {
		return fmt.Errorf(
			"%w: entry limit %d reached",
			ErrSpoolFull,
			s.config.MaxEntries,
		)
	}

	if s.config.MaxBytes > 0 &&
		entrySize > s.config.MaxBytes-s.bytes {
		return fmt.Errorf(
			"%w: adding %d bytes would exceed byte limit %d",
			ErrSpoolFull,
			entrySize,
			s.config.MaxBytes,
		)
	}

	target := s.entryPath(batch.ID)

	if _, err := os.Stat(target); err == nil {
		return fmt.Errorf(
			"%w: batch %q",
			ErrSpoolEntryExists,
			batch.ID,
		)
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf(
			"stat spool entry for batch %q: %w",
			batch.ID,
			err,
		)
	}

	if err := writeFileAtomic(
		ctx,
		s.config.Directory,
		target,
		encoded,
		s.config.FileMode,
	); err != nil {
		return fmt.Errorf(
			"persist spool batch %q: %w",
			batch.ID,
			err,
		)
	}

	if s.config.SyncDirectory {
		if err := syncDirectory(s.config.Directory); err != nil {
			// The entry is already visible. Attempt to remove it so Put does
			// not report failure while silently leaving an untracked entry.
			removeErr := os.Remove(target)

			if removeErr == nil {
				_ = syncDirectory(s.config.Directory)
			}

			return errors.Join(
				fmt.Errorf(
					"sync spool directory after batch %q: %w",
					batch.ID,
					err,
				),
				removeErr,
			)
		}
	}

	s.entries++
	s.bytes += entrySize

	return nil
}

// Get loads and verifies one batch.
func (s *Spool) Get(
	ctx context.Context,
	id BatchID,
) (Batch, error) {
	if s == nil {
		return Batch{}, errors.New("spool is nil")
	}

	if ctx == nil {
		return Batch{}, errors.New("spool get context is nil")
	}

	if err := ctx.Err(); err != nil {
		return Batch{}, err
	}

	if strings.TrimSpace(string(id)) == "" {
		return Batch{}, errors.New("spool batch id is required")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return Batch{}, ErrSpoolClosed
	}

	path := s.entryPath(id)

	envelope, _, err := readSpoolEnvelope(
		ctx,
		path,
		s.config.MaxEntryBytes,
	)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Batch{}, fmt.Errorf(
				"%w: batch %q",
				ErrSpoolEntryNotFound,
				id,
			)
		}

		return Batch{}, fmt.Errorf(
			"read spool batch %q: %w",
			id,
			err,
		)
	}

	if envelope.Batch.ID != id {
		return Batch{}, fmt.Errorf(
			"%w: requested batch %q but entry contains %q",
			ErrSpoolCorrupt,
			id,
			envelope.Batch.ID,
		)
	}

	return cloneBatch(envelope.Batch), nil
}

// Delete removes an acknowledged or otherwise completed batch.
func (s *Spool) Delete(
	ctx context.Context,
	id BatchID,
) error {
	if s == nil {
		return errors.New("spool is nil")
	}

	if ctx == nil {
		return errors.New("spool delete context is nil")
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if strings.TrimSpace(string(id)) == "" {
		return errors.New("spool batch id is required")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSpoolClosed
	}

	path := s.entryPath(id)

	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf(
				"%w: batch %q",
				ErrSpoolEntryNotFound,
				id,
			)
		}

		return fmt.Errorf(
			"stat spool batch %q: %w",
			id,
			err,
		)
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := os.Remove(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf(
				"%w: batch %q",
				ErrSpoolEntryNotFound,
				id,
			)
		}

		return fmt.Errorf(
			"delete spool batch %q: %w",
			id,
			err,
		)
	}

	if s.config.SyncDirectory {
		if err := syncDirectory(s.config.Directory); err != nil {
			return fmt.Errorf(
				"sync spool directory after deleting batch %q: %w",
				id,
				err,
			)
		}
	}

	s.entries--
	s.bytes -= info.Size()

	if s.entries < 0 {
		s.entries = 0
	}

	if s.bytes < 0 {
		s.bytes = 0
	}

	return nil
}

// List returns metadata for all valid spool files.
//
// List reads entry headers and verifies checksums. A corrupt entry causes List
// to return an error rather than silently excluding pending work.
func (s *Spool) List(
	ctx context.Context,
) ([]SpoolEntryMetadata, error) {
	if s == nil {
		return nil, errors.New("spool is nil")
	}

	if ctx == nil {
		return nil, errors.New("spool list context is nil")
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, ErrSpoolClosed
	}

	paths, err := s.entryPathsLocked()
	if err != nil {
		return nil, err
	}

	entries := make([]SpoolEntryMetadata, 0, len(paths))

	for _, path := range paths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		envelope, size, err := readSpoolEnvelope(
			ctx,
			path,
			s.config.MaxEntryBytes,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"read spool entry %q: %w",
				path,
				err,
			)
		}

		entries = append(entries, SpoolEntryMetadata{
			BatchID:       envelope.Batch.ID,
			DestinationID: envelope.Batch.Destination.ID,
			Path:          path,
			CreatedAt:     envelope.Batch.CreatedAt,
			StoredAt:      envelope.StoredAt,
			SizeBytes:     size,
		})
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].StoredAt.Equal(entries[j].StoredAt) {
			return entries[i].BatchID < entries[j].BatchID
		}

		return entries[i].StoredAt.Before(entries[j].StoredAt)
	})

	return entries, nil
}

// Recover loads all pending batches in stable oldest-first order.
//
// Recover does not delete entries. The caller should delete each entry only
// after successful acknowledgement from its destination.
func (s *Spool) Recover(
	ctx context.Context,
) ([]Batch, error) {
	entries, err := s.List(ctx)
	if err != nil {
		return nil, err
	}

	batches := make([]Batch, 0, len(entries))

	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		batch, err := s.Get(ctx, entry.BatchID)
		if err != nil {
			return nil, fmt.Errorf(
				"recover batch %q: %w",
				entry.BatchID,
				err,
			)
		}

		batches = append(batches, batch)
	}

	return batches, nil
}

// Quarantine moves a corrupt or unprocessable entry outside the active spool.
//
// The quarantined file remains available for manual diagnosis and is no
// longer returned by List or Recover.
func (s *Spool) Quarantine(
	ctx context.Context,
	id BatchID,
	reason string,
) (string, error) {
	if s == nil {
		return "", errors.New("spool is nil")
	}

	if ctx == nil {
		return "", errors.New("spool quarantine context is nil")
	}

	if err := ctx.Err(); err != nil {
		return "", err
	}

	if strings.TrimSpace(string(id)) == "" {
		return "", errors.New("spool batch id is required")
	}

	reason = sanitiseFileComponent(reason)
	if reason == "" {
		reason = "unknown"
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return "", ErrSpoolClosed
	}

	source := s.entryPath(id)

	info, err := os.Stat(source)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf(
				"%w: batch %q",
				ErrSpoolEntryNotFound,
				id,
			)
		}

		return "", fmt.Errorf(
			"stat spool batch %q: %w",
			id,
			err,
		)
	}

	quarantineDirectory := filepath.Join(
		s.config.Directory,
		"quarantine",
	)

	if err := os.MkdirAll(
		quarantineDirectory,
		s.config.DirectoryMode,
	); err != nil {
		return "", fmt.Errorf(
			"create quarantine directory: %w",
			err,
		)
	}

	destination := filepath.Join(
		quarantineDirectory,
		fmt.Sprintf(
			"%s-%d-%s%s",
			spoolKey(id),
			time.Now().UTC().UnixNano(),
			reason,
			spoolFileExtension,
		),
	)

	if err := os.Rename(source, destination); err != nil {
		return "", fmt.Errorf(
			"quarantine spool batch %q: %w",
			id,
			err,
		)
	}

	if s.config.SyncDirectory {
		if err := syncDirectory(s.config.Directory); err != nil {
			return "", fmt.Errorf(
				"sync spool directory after quarantine: %w",
				err,
			)
		}

		if err := syncDirectory(quarantineDirectory); err != nil {
			return "", fmt.Errorf(
				"sync quarantine directory: %w",
				err,
			)
		}
	}

	s.entries--
	s.bytes -= info.Size()

	if s.entries < 0 {
		s.entries = 0
	}

	if s.bytes < 0 {
		s.bytes = 0
	}

	return destination, nil
}

// Stats returns current spool counters.
func (s *Spool) Stats() SpoolStats {
	if s == nil {
		return SpoolStats{}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return SpoolStats{
		Directory:  s.config.Directory,
		Entries:    s.entries,
		Bytes:      s.bytes,
		MaxEntries: s.config.MaxEntries,
		MaxBytes:   s.config.MaxBytes,
		Closed:     s.closed,
	}
}

// Close prevents future spool operations.
//
// Spool methods open and close files per operation, so Close has no persistent
// file descriptor to release.
func (s *Spool) Close() {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.closed = true
}

// Closed reports whether the spool has been closed.
func (s *Spool) Closed() bool {
	if s == nil {
		return true
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.closed
}

func (s *Spool) scanLocked() error {
	paths, err := s.entryPathsLocked()
	if err != nil {
		return err
	}

	var totalBytes int64

	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf(
				"stat spool entry %q: %w",
				path,
				err,
			)
		}

		if info.Size() > s.config.MaxEntryBytes {
			return fmt.Errorf(
				"%w: entry %q size %d exceeds limit %d",
				ErrSpoolCorrupt,
				path,
				info.Size(),
				s.config.MaxEntryBytes,
			)
		}

		totalBytes += info.Size()
	}

	s.entries = len(paths)
	s.bytes = totalBytes

	return nil
}

func (s *Spool) entryPathsLocked() ([]string, error) {
	pattern := filepath.Join(
		s.config.Directory,
		"*"+spoolFileExtension,
	)

	paths, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf(
			"list spool entries: %w",
			err,
		)
	}

	sort.Strings(paths)

	return paths, nil
}

func (s *Spool) entryPath(id BatchID) string {
	return filepath.Join(
		s.config.Directory,
		spoolKey(id)+spoolFileExtension,
	)
}

// spoolKey converts any BatchID into a fixed safe filename.
//
// Hashing prevents path traversal, separator injection, platform-specific
// filename issues, and accidental exposure of customer-provided identifiers.
func spoolKey(id BatchID) string {
	digest := sha256.Sum256([]byte(id))

	return hex.EncodeToString(digest[:])
}

func encodeSpoolEnvelope(
	batch Batch,
	storedAt time.Time,
) ([]byte, error) {
	batchJSON, err := json.Marshal(batch)
	if err != nil {
		return nil, fmt.Errorf("marshal batch: %w", err)
	}

	digest := sha256.Sum256(batchJSON)

	envelope := spoolEnvelope{
		Version:  spoolFormatVersion,
		StoredAt: storedAt,
		Batch:    batch,
		Checksum: hex.EncodeToString(digest[:]),
	}

	encoded, err := json.Marshal(envelope)
	if err != nil {
		return nil, fmt.Errorf("marshal spool envelope: %w", err)
	}

	return encoded, nil
}

func readSpoolEnvelope(
	ctx context.Context,
	path string,
	maxBytes int64,
) (spoolEnvelope, int64, error) {
	if err := ctx.Err(); err != nil {
		return spoolEnvelope{}, 0, err
	}

	file, err := os.Open(path)
	if err != nil {
		return spoolEnvelope{}, 0, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return spoolEnvelope{}, 0, fmt.Errorf(
			"stat entry: %w",
			err,
		)
	}

	if !info.Mode().IsRegular() {
		return spoolEnvelope{}, 0, fmt.Errorf(
			"%w: entry is not a regular file",
			ErrSpoolCorrupt,
		)
	}

	if info.Size() > maxBytes {
		return spoolEnvelope{}, info.Size(), fmt.Errorf(
			"%w: entry size %d exceeds limit %d",
			ErrSpoolCorrupt,
			info.Size(),
			maxBytes,
		)
	}

	limited := io.LimitReader(file, maxBytes+1)

	data, err := io.ReadAll(limited)
	if err != nil {
		return spoolEnvelope{}, info.Size(), fmt.Errorf(
			"read entry: %w",
			err,
		)
	}

	if int64(len(data)) > maxBytes {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: entry exceeds limit %d",
			ErrSpoolCorrupt,
			maxBytes,
		)
	}

	if err := ctx.Err(); err != nil {
		return spoolEnvelope{}, 0, err
	}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var envelope spoolEnvelope

	if err := decoder.Decode(&envelope); err != nil {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: decode entry: %v",
			ErrSpoolCorrupt,
			err,
		)
	}

	if err := ensureJSONEOF(decoder); err != nil {
		return spoolEnvelope{}, int64(len(data)), err
	}

	if envelope.Version != spoolFormatVersion {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: unsupported format version %d",
			ErrSpoolCorrupt,
			envelope.Version,
		)
	}

	if envelope.StoredAt.IsZero() {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: stored_at is required",
			ErrSpoolCorrupt,
		)
	}

	if err := envelope.Batch.Validate(); err != nil {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: invalid batch: %v",
			ErrSpoolCorrupt,
			err,
		)
	}

	batchJSON, err := json.Marshal(envelope.Batch)
	if err != nil {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: marshal batch for verification: %v",
			ErrSpoolCorrupt,
			err,
		)
	}

	digest := sha256.Sum256(batchJSON)
	expected := hex.EncodeToString(digest[:])

	if !strings.EqualFold(envelope.Checksum, expected) {
		return spoolEnvelope{}, int64(len(data)), fmt.Errorf(
			"%w: checksum mismatch",
			ErrSpoolCorrupt,
		)
	}

	return envelope, int64(len(data)), nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra json.RawMessage

	err := decoder.Decode(&extra)

	if errors.Is(err, io.EOF) {
		return nil
	}

	if err == nil {
		return fmt.Errorf(
			"%w: entry contains multiple JSON values",
			ErrSpoolCorrupt,
		)
	}

	return fmt.Errorf(
		"%w: trailing data: %v",
		ErrSpoolCorrupt,
		err,
	)
}

func writeFileAtomic(
	ctx context.Context,
	directory string,
	target string,
	data []byte,
	mode fs.FileMode,
) (returnErr error) {
	if err := ctx.Err(); err != nil {
		return err
	}

	temp, err := os.CreateTemp(
		directory,
		".otgh-spool-tmp-*",
	)
	if err != nil {
		return fmt.Errorf("create temporary spool file: %w", err)
	}

	tempName := temp.Name()

	defer func() {
		closeErr := temp.Close()
		removeErr := os.Remove(tempName)

		if returnErr == nil &&
			closeErr != nil &&
			!errors.Is(closeErr, os.ErrClosed) {
			returnErr = fmt.Errorf(
				"close temporary spool file: %w",
				closeErr,
			)
		}

		if returnErr == nil &&
			removeErr != nil &&
			!errors.Is(removeErr, os.ErrNotExist) {
			returnErr = fmt.Errorf(
				"remove temporary spool file: %w",
				removeErr,
			)
		}
	}()

	if err := temp.Chmod(mode); err != nil {
		return fmt.Errorf(
			"set temporary spool file mode: %w",
			err,
		)
	}

	if _, err := temp.Write(data); err != nil {
		return fmt.Errorf(
			"write temporary spool file: %w",
			err,
		)
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := temp.Sync(); err != nil {
		return fmt.Errorf(
			"sync temporary spool file: %w",
			err,
		)
	}

	if err := temp.Close(); err != nil {
		return fmt.Errorf(
			"close temporary spool file: %w",
			err,
		)
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := os.Rename(tempName, target); err != nil {
		return fmt.Errorf(
			"publish spool file: %w",
			err,
		)
	}

	return nil
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer directory.Close()

	return directory.Sync()
}

func sanitiseFileComponent(value string) string {
	value = strings.TrimSpace(value)

	if value == "" {
		return ""
	}

	var builder strings.Builder

	for _, character := range value {
		switch {
		case character >= 'a' && character <= 'z':
			builder.WriteRune(character)

		case character >= 'A' && character <= 'Z':
			builder.WriteRune(character)

		case character >= '0' && character <= '9':
			builder.WriteRune(character)

		case character == '-', character == '_':
			builder.WriteRune(character)

		default:
			builder.WriteByte('-')
		}

		if builder.Len() >= 64 {
			break
		}
	}

	return strings.Trim(builder.String(), "-_")
}
