package objectstore

import (
	"errors"
	"fmt"
	"net/url"
	"path"
	"strings"
	"time"
)

// Provider identifies the object-storage API family.
//
// S3Compatible represents services implementing the S3 object API. Provider
// identity is kept separate so additional object-storage implementations can
// be introduced without changing the export subsystem.
type Provider string

const (
	ProviderS3Compatible Provider = "s3_compatible"
)

// Valid reports whether the provider is supported.
func (p Provider) Valid() bool {
	switch p {
	case ProviderS3Compatible:
		return true

	default:
		return false
	}
}

// CredentialType identifies how OTGH obtains object-storage credentials.
type CredentialType string

const (
	// CredentialEnvironment delegates credential discovery to the object
	// storage SDK/environment.
	CredentialEnvironment CredentialType = "environment"

	// CredentialStatic uses explicitly configured access and secret keys.
	//
	// These values should come from a secret provider or environment
	// variables rather than source control.
	CredentialStatic CredentialType = "static"

	// CredentialAnonymous performs unsigned/anonymous access.
	//
	// This should only be used against storage intentionally configured for
	// anonymous writes.
	CredentialAnonymous CredentialType = "anonymous"
)

// Valid reports whether the credential type is recognized.
func (c CredentialType) Valid() bool {
	switch c {
	case CredentialEnvironment,
		CredentialStatic,
		CredentialAnonymous:
		return true

	default:
		return false
	}
}

// CredentialConfig controls authentication to the object store.
type CredentialConfig struct {
	Type CredentialType

	AccessKey string
	SecretKey string

	// SessionToken supports temporary credentials where required by the
	// provider.
	SessionToken string
}

// Validate verifies credential configuration.
func (c CredentialConfig) Validate() error {
	if !c.Type.Valid() {
		return fmt.Errorf(
			"unsupported object-store credential type %q",
			c.Type,
		)
	}

	switch c.Type {
	case CredentialEnvironment:
		if c.AccessKey != "" ||
			c.SecretKey != "" ||
			c.SessionToken != "" {
			return errors.New(
				"explicit object-store credentials cannot be supplied when credential type is environment",
			)
		}

	case CredentialStatic:
		if strings.TrimSpace(c.AccessKey) == "" {
			return errors.New(
				"static object-store credentials require an access key",
			)
		}

		if strings.TrimSpace(c.SecretKey) == "" {
			return errors.New(
				"static object-store credentials require a secret key",
			)
		}

	case CredentialAnonymous:
		if c.AccessKey != "" ||
			c.SecretKey != "" ||
			c.SessionToken != "" {
			return errors.New(
				"anonymous object-store access cannot contain credentials",
			)
		}
	}

	return nil
}

// EncryptionType identifies server-side encryption requested for objects.
//
// Encryption names are modeled as OTGH configuration values. exporter.go will
// translate them into the provider SDK's request representation.
type EncryptionType string

const (
	EncryptionNone EncryptionType = "none"

	// EncryptionS3Managed represents storage-provider-managed server-side
	// encryption compatible with the common SSE-S3 model.
	EncryptionS3Managed EncryptionType = "s3_managed"

	// EncryptionKMS represents provider key-management-backed server-side
	// encryption compatible with the SSE-KMS model.
	EncryptionKMS EncryptionType = "kms"
)

// Valid reports whether the encryption type is supported.
func (e EncryptionType) Valid() bool {
	switch e {
	case EncryptionNone,
		EncryptionS3Managed,
		EncryptionKMS:
		return true

	default:
		return false
	}
}

// EncryptionConfig controls server-side object encryption.
type EncryptionConfig struct {
	Type EncryptionType

	// KMSKeyID identifies the key when Type is EncryptionKMS.
	KMSKeyID string

	// Context contains non-secret encryption-context attributes where the
	// object-store provider supports them.
	Context map[string]string
}

// Validate verifies encryption configuration.
func (c EncryptionConfig) Validate() error {
	if !c.Type.Valid() {
		return fmt.Errorf(
			"unsupported object-store encryption type %q",
			c.Type,
		)
	}

	switch c.Type {
	case EncryptionNone,
		EncryptionS3Managed:
		if strings.TrimSpace(c.KMSKeyID) != "" {
			return fmt.Errorf(
				"KMS key id requires encryption type %q",
				EncryptionKMS,
			)
		}

		if len(c.Context) > 0 {
			return fmt.Errorf(
				"encryption context requires encryption type %q",
				EncryptionKMS,
			)
		}

	case EncryptionKMS:
		if strings.TrimSpace(c.KMSKeyID) == "" {
			return errors.New(
				"KMS encryption requires a key id",
			)
		}
	}

	for key, value := range c.Context {
		if strings.TrimSpace(key) == "" {
			return errors.New(
				"encryption context key cannot be empty",
			)
		}

		if strings.ContainsAny(
			key,
			"\r\n",
		) || strings.ContainsAny(
			value,
			"\r\n",
		) {
			return errors.New(
				"encryption context cannot contain line breaks",
			)
		}
	}

	return nil
}

// UploadConfig controls individual object upload limits and timing.
type UploadConfig struct {
	// Timeout is the maximum duration for one object upload unless the caller
	// provides a shorter context deadline.
	Timeout time.Duration

	// MaxObjectBytes prevents unexpectedly large batches from being archived
	// into a single object.
	MaxObjectBytes int64

	// ContentType identifies the serialized object representation.
	ContentType string

	// StorageClass optionally requests a provider-specific storage class.
	//
	// Empty delegates storage-class selection to the provider/bucket.
	StorageClass string
}

// DefaultUploadConfig returns baseline upload limits.
func DefaultUploadConfig() UploadConfig {
	return UploadConfig{
		Timeout:        60 * time.Second,
		MaxObjectBytes: 128 * 1024 * 1024,
		ContentType:    "application/json",
	}
}

// Validate verifies upload configuration.
func (c UploadConfig) Validate() error {
	if c.Timeout <= 0 {
		return errors.New(
			"object-store upload timeout must be greater than zero",
		)
	}

	if c.MaxObjectBytes <= 0 {
		return errors.New(
			"object-store max object bytes must be greater than zero",
		)
	}

	if strings.TrimSpace(c.ContentType) == "" {
		return errors.New(
			"object-store content type is required",
		)
	}

	if strings.ContainsAny(
		c.StorageClass,
		"\r\n",
	) {
		return errors.New(
			"object-store storage class contains a line break",
		)
	}

	return nil
}

// RetryTransportConfig controls provider-client connection recovery.
//
// Batch retry policy remains centralized in internal/export/retry.go. These
// values are intended only for low-level storage client transport behavior
// where the selected SDK exposes equivalent controls.
type RetryTransportConfig struct {
	MaxAttempts int
}

// DefaultRetryTransportConfig returns conservative transport behavior.
func DefaultRetryTransportConfig() RetryTransportConfig {
	return RetryTransportConfig{
		MaxAttempts: 1,
	}
}

// Validate verifies provider transport retry configuration.
func (c RetryTransportConfig) Validate() error {
	if c.MaxAttempts <= 0 {
		return errors.New(
			"object-store transport max attempts must be greater than zero",
		)
	}

	return nil
}

// Config describes one OTGH object-storage export destination.
type Config struct {
	// ID must match the DestinationID registered with OTGH.
	ID string

	Provider Provider

	// Endpoint optionally overrides the provider's normal object-store
	// endpoint.
	//
	// For S3-compatible private storage this can point to the storage API
	// endpoint.
	Endpoint string

	// Region identifies the storage region where required.
	Region string

	// Bucket is the destination object bucket/container.
	Bucket string

	// Prefix is prepended to generated object keys.
	//
	// Example:
	//
	//   osha/telemetry/
	//
	// exporter.go will append tenant/date/batch-specific key segments.
	Prefix string

	Credentials CredentialConfig

	Encryption EncryptionConfig

	Upload UploadConfig

	TransportRetry RetryTransportConfig

	// UsePathStyle controls S3-compatible addressing behavior where supported
	// by the selected SDK.
	UsePathStyle bool

	// Metadata contains static non-sensitive object metadata applied to each
	// uploaded batch.
	Metadata map[string]string

	// Tags contains static object tags.
	Tags map[string]string
}

// DefaultConfig returns baseline object-store configuration.
//
// ID, Endpoint, Region, and Bucket remain deployment-specific.
func DefaultConfig() Config {
	return Config{
		Provider: ProviderS3Compatible,

		Credentials: CredentialConfig{
			Type: CredentialEnvironment,
		},

		Encryption: EncryptionConfig{
			Type: EncryptionS3Managed,
		},

		Upload: DefaultUploadConfig(),

		TransportRetry: DefaultRetryTransportConfig(),
	}
}

// Validate performs complete object-store configuration validation.
func (c Config) Validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return errors.New(
			"object-store exporter id is required",
		)
	}

	if !c.Provider.Valid() {
		return fmt.Errorf(
			"unsupported object-store provider %q",
			c.Provider,
		)
	}

	if strings.TrimSpace(c.Bucket) == "" {
		return errors.New(
			"object-store bucket is required",
		)
	}

	if strings.ContainsAny(
		c.Bucket,
		"\r\n",
	) {
		return errors.New(
			"object-store bucket contains a line break",
		)
	}

	if c.Endpoint != "" {
		parsed, err := url.ParseRequestURI(
			strings.TrimSpace(c.Endpoint),
		)
		if err != nil {
			return fmt.Errorf(
				"invalid object-store endpoint %q: %w",
				c.Endpoint,
				err,
			)
		}

		switch parsed.Scheme {
		case "http", "https":
			// Supported endpoint schemes.

		default:
			return fmt.Errorf(
				"unsupported object-store endpoint scheme %q",
				parsed.Scheme,
			)
		}

		if parsed.Host == "" {
			return errors.New(
				"object-store endpoint must include a host",
			)
		}
	}

	if strings.ContainsAny(
		c.Region,
		"\r\n",
	) {
		return errors.New(
			"object-store region contains a line break",
		)
	}

	if err := validatePrefix(
		c.Prefix,
	); err != nil {
		return err
	}

	if err := c.Credentials.Validate(); err != nil {
		return fmt.Errorf(
			"object-store credentials: %w",
			err,
		)
	}

	if err := c.Encryption.Validate(); err != nil {
		return fmt.Errorf(
			"object-store encryption: %w",
			err,
		)
	}

	if err := c.Upload.Validate(); err != nil {
		return fmt.Errorf(
			"object-store upload: %w",
			err,
		)
	}

	if err := c.TransportRetry.Validate(); err != nil {
		return fmt.Errorf(
			"object-store transport retry: %w",
			err,
		)
	}

	if err := validateStringMap(
		"metadata",
		c.Metadata,
	); err != nil {
		return err
	}

	if err := validateStringMap(
		"tags",
		c.Tags,
	); err != nil {
		return err
	}

	return nil
}

// Normalized returns a defensive configuration copy with stable prefix and
// default values applied.
func (c Config) Normalized() Config {
	normalized := c

	normalized.ID = strings.TrimSpace(c.ID)
	normalized.Endpoint = strings.TrimSpace(
		c.Endpoint,
	)
	normalized.Region = strings.TrimSpace(
		c.Region,
	)
	normalized.Bucket = strings.TrimSpace(
		c.Bucket,
	)

	normalized.Prefix = normalizePrefix(
		c.Prefix,
	)

	if normalized.Provider == "" {
		normalized.Provider =
			ProviderS3Compatible
	}

	if normalized.Credentials.Type == "" {
		normalized.Credentials.Type =
			CredentialEnvironment
	}

	if normalized.Encryption.Type == "" {
		normalized.Encryption.Type =
			EncryptionS3Managed
	}

	if normalized.Upload.ContentType == "" {
		normalized.Upload.ContentType =
			"application/json"
	}

	normalized.Metadata = cloneStringMap(
		c.Metadata,
	)

	normalized.Tags = cloneStringMap(
		c.Tags,
	)

	if c.Encryption.Context != nil {
		normalized.Encryption.Context =
			cloneStringMap(
				c.Encryption.Context,
			)
	}

	return normalized
}

// ObjectKeyPrefix returns the normalized configured object prefix.
func (c Config) ObjectKeyPrefix() string {
	return normalizePrefix(c.Prefix)
}

func validatePrefix(
	prefix string,
) error {
	prefix = strings.TrimSpace(prefix)

	if prefix == "" {
		return nil
	}

	if strings.ContainsAny(
		prefix,
		"\r\n",
	) {
		return errors.New(
			"object-store prefix contains a line break",
		)
	}

	if strings.Contains(
		prefix,
		"\\",
	) {
		return errors.New(
			"object-store prefix must use forward slashes",
		)
	}

	segments := strings.Split(
		prefix,
		"/",
	)

	for _, segment := range segments {
		if segment == ".." {
			return errors.New(
				"object-store prefix cannot contain parent-directory segments",
			)
		}
	}

	return nil
}

func normalizePrefix(
	prefix string,
) string {
	prefix = strings.TrimSpace(prefix)

	if prefix == "" {
		return ""
	}

	prefix = strings.ReplaceAll(
		prefix,
		"\\",
		"/",
	)

	prefix = strings.Trim(
		prefix,
		"/",
	)

	if prefix == "" {
		return ""
	}

	// path.Clean removes repeated separators and "." segments. Validation is
	// responsible for rejecting ".." before this normalization is used for a
	// production configuration.
	prefix = path.Clean(prefix)

	if prefix == "." {
		return ""
	}

	return prefix + "/"
}

func validateStringMap(
	name string,
	values map[string]string,
) error {
	for key, value := range values {
		if strings.TrimSpace(key) == "" {
			return fmt.Errorf(
				"object-store %s key cannot be empty",
				name,
			)
		}

		if strings.ContainsAny(
			key,
			"\r\n",
		) || strings.ContainsAny(
			value,
			"\r\n",
		) {
			return fmt.Errorf(
				"object-store %s cannot contain line breaks",
				name,
			)
		}
	}

	return nil
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
