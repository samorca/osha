package grpc

import (
	"crypto/tls"
	"errors"
	"fmt"
	"strings"
	"time"
)

// AuthenticationType identifies authentication metadata attached to outgoing
// gRPC calls.
//
// Transport security and application authentication remain separate:
// TLS/mTLS protects and authenticates the transport, while these credentials
// represent application-level identity.
type AuthenticationType string

const (
	AuthenticationNone   AuthenticationType = "none"
	AuthenticationBearer AuthenticationType = "bearer"
	AuthenticationAPIKey AuthenticationType = "api_key"
)

// Valid reports whether the authentication type is recognized.
func (a AuthenticationType) Valid() bool {
	switch a {
	case AuthenticationNone,
		AuthenticationBearer,
		AuthenticationAPIKey:
		return true

	default:
		return false
	}
}

// TLSConfig controls gRPC transport security.
type TLSConfig struct {
	// Enabled requires TLS transport security.
	//
	// Production deployments should normally keep this enabled.
	Enabled bool

	// CAFile optionally contains PEM-encoded CA certificates used to verify
	// the remote gRPC server.
	//
	// When empty, the system certificate pool is used.
	CAFile string

	// CertFile and KeyFile enable mutual TLS client authentication.
	//
	// Both must be configured together.
	CertFile string
	KeyFile  string

	// ServerName optionally overrides the DNS name used during remote
	// certificate verification.
	ServerName string

	// MinVersion is a crypto/tls protocol version.
	//
	// Zero is normalized to TLS 1.2.
	MinVersion uint16
}

// Validate verifies TLS configuration.
func (c TLSConfig) Validate() error {
	caFile := strings.TrimSpace(c.CAFile)
	certFile := strings.TrimSpace(c.CertFile)
	keyFile := strings.TrimSpace(c.KeyFile)
	serverName := strings.TrimSpace(c.ServerName)

	if !c.Enabled {
		if caFile != "" ||
			certFile != "" ||
			keyFile != "" ||
			serverName != "" {
			return errors.New(
				"gRPC TLS files or server name require TLS to be enabled",
			)
		}

		return nil
	}

	if (certFile == "") != (keyFile == "") {
		return errors.New(
			"gRPC client TLS certificate and private key must be configured together",
		)
	}

	minVersion := c.MinVersion

	if minVersion == 0 {
		minVersion = tls.VersionTLS12
	}

	switch minVersion {
	case tls.VersionTLS12,
		tls.VersionTLS13:
		return nil

	default:
		return fmt.Errorf(
			"unsupported gRPC minimum TLS version 0x%x",
			minVersion,
		)
	}
}

// AuthenticationConfig controls application authentication metadata.
type AuthenticationConfig struct {
	Type AuthenticationType

	// BearerToken is attached as authorization metadata when Type is bearer.
	BearerToken string

	// APIKey is attached when Type is api_key.
	APIKey string

	// APIKeyMetadataKey defaults to x-api-key when omitted.
	APIKeyMetadataKey string
}

// Validate verifies authentication configuration.
func (c AuthenticationConfig) Validate() error {
	if !c.Type.Valid() {
		return fmt.Errorf(
			"unsupported gRPC authentication type %q",
			c.Type,
		)
	}

	switch c.Type {
	case AuthenticationNone:
		if c.BearerToken != "" ||
			c.APIKey != "" ||
			c.APIKeyMetadataKey != "" {
			return errors.New(
				"gRPC credentials supplied while authentication type is none",
			)
		}

	case AuthenticationBearer:
		if strings.TrimSpace(c.BearerToken) == "" {
			return errors.New(
				"gRPC bearer authentication requires a token",
			)
		}

	case AuthenticationAPIKey:
		if strings.TrimSpace(c.APIKey) == "" {
			return errors.New(
				"gRPC API key authentication requires a key",
			)
		}

		key := strings.TrimSpace(
			c.APIKeyMetadataKey,
		)

		if key != "" &&
			!validMetadataKey(key) {
			return fmt.Errorf(
				"invalid gRPC API key metadata key %q",
				key,
			)
		}
	}

	return nil
}

// TimeoutConfig controls connection and RPC timing.
type TimeoutConfig struct {
	// Connect is the maximum time allowed for initial connection setup when
	// the caller requests a blocking connection establishment.
	Connect time.Duration

	// Request is the default deadline used for one export RPC when the caller
	// has not supplied a shorter context deadline.
	Request time.Duration

	// Health is the default deadline for exporter health checks.
	Health time.Duration
}

// DefaultTimeoutConfig returns baseline timing limits.
func DefaultTimeoutConfig() TimeoutConfig {
	return TimeoutConfig{
		Connect: 10 * time.Second,
		Request: 30 * time.Second,
		Health:  5 * time.Second,
	}
}

// Validate verifies timeout configuration.
func (c TimeoutConfig) Validate() error {
	if c.Connect <= 0 {
		return errors.New(
			"gRPC connect timeout must be greater than zero",
		)
	}

	if c.Request <= 0 {
		return errors.New(
			"gRPC request timeout must be greater than zero",
		)
	}

	if c.Health <= 0 {
		return errors.New(
			"gRPC health timeout must be greater than zero",
		)
	}

	return nil
}

// KeepaliveConfig controls idle connection liveness checks.
//
// These values describe our OTGH policy. exporter.go will translate them into
// the gRPC-Go keepalive client configuration.
type KeepaliveConfig struct {
	// Time is the period after which a keepalive ping may be sent on an idle
	// connection.
	Time time.Duration

	// Timeout is how long to wait for a keepalive ping acknowledgement before
	// considering the connection unavailable.
	Timeout time.Duration

	// PermitWithoutStream controls whether keepalive may run when there are no
	// active RPC streams.
	PermitWithoutStream bool
}

// DefaultKeepaliveConfig returns conservative keepalive defaults.
func DefaultKeepaliveConfig() KeepaliveConfig {
	return KeepaliveConfig{
		Time:                60 * time.Second,
		Timeout:             20 * time.Second,
		PermitWithoutStream: false,
	}
}

// Validate verifies keepalive configuration.
func (c KeepaliveConfig) Validate() error {
	if c.Time <= 0 {
		return errors.New(
			"gRPC keepalive time must be greater than zero",
		)
	}

	if c.Timeout <= 0 {
		return errors.New(
			"gRPC keepalive timeout must be greater than zero",
		)
	}

	return nil
}

// ReconnectConfig controls connection-recovery backoff.
//
// This is connection-level recovery, not OTGH's batch-delivery retry policy.
// Batch retries remain centralized in internal/export/retry.go.
type ReconnectConfig struct {
	// BaseDelay is the first reconnection delay.
	BaseDelay time.Duration

	// Multiplier grows subsequent delays.
	Multiplier float64

	// Jitter adds proportional timing variation.
	Jitter float64

	// MaxDelay caps reconnect delay.
	MaxDelay time.Duration
}

// DefaultReconnectConfig returns baseline reconnection parameters.
func DefaultReconnectConfig() ReconnectConfig {
	return ReconnectConfig{
		BaseDelay:  time.Second,
		Multiplier: 1.6,
		Jitter:     0.2,
		MaxDelay:   30 * time.Second,
	}
}

// Validate verifies reconnection parameters.
func (c ReconnectConfig) Validate() error {
	if c.BaseDelay <= 0 {
		return errors.New(
			"gRPC reconnect base delay must be greater than zero",
		)
	}

	if c.Multiplier < 1 {
		return errors.New(
			"gRPC reconnect multiplier must be greater than or equal to one",
		)
	}

	if c.Jitter < 0 ||
		c.Jitter > 1 {
		return errors.New(
			"gRPC reconnect jitter must be between zero and one",
		)
	}

	if c.MaxDelay <= 0 {
		return errors.New(
			"gRPC reconnect max delay must be greater than zero",
		)
	}

	if c.MaxDelay < c.BaseDelay {
		return errors.New(
			"gRPC reconnect max delay must be greater than or equal to base delay",
		)
	}

	return nil
}

// Config describes one OTGH gRPC export destination.
type Config struct {
	// ID must correspond to the DestinationID registered with OTGH.
	ID string

	// Endpoint is the gRPC target.
	//
	// Examples can include DNS names and resolver-compatible gRPC targets.
	Endpoint string

	// Authority optionally overrides the HTTP/2 :authority value and TLS
	// server name behavior where supported by the client.
	Authority string

	Authentication AuthenticationConfig

	TLS TLSConfig

	Timeouts TimeoutConfig

	Keepalive KeepaliveConfig

	Reconnect ReconnectConfig

	// MaxSendMessageBytes prevents an export RPC from sending unexpectedly
	// large messages.
	MaxSendMessageBytes int

	// MaxReceiveMessageBytes limits how much response data gRPC may decode.
	MaxReceiveMessageBytes int

	// WaitForReady determines whether an RPC may wait for the channel to
	// become ready instead of immediately failing for transient connectivity.
	WaitForReady bool

	// Metadata contains static non-secret metadata attached to calls.
	//
	// Credentials should use Authentication rather than this map.
	Metadata map[string]string
}

// DefaultConfig returns baseline gRPC exporter configuration.
//
// ID and Endpoint remain empty because the caller must explicitly provide the
// destination identity.
func DefaultConfig() Config {
	return Config{
		Authentication: AuthenticationConfig{
			Type: AuthenticationNone,
		},

		TLS: TLSConfig{
			Enabled:    true,
			MinVersion: tls.VersionTLS12,
		},

		Timeouts: DefaultTimeoutConfig(),

		Keepalive: DefaultKeepaliveConfig(),

		Reconnect: DefaultReconnectConfig(),

		MaxSendMessageBytes:    64 * 1024 * 1024,
		MaxReceiveMessageBytes: 16 * 1024 * 1024,

		WaitForReady: true,
	}
}

// Validate performs complete gRPC configuration validation.
func (c Config) Validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return errors.New(
			"gRPC exporter id is required",
		)
	}

	if strings.TrimSpace(c.Endpoint) == "" {
		return errors.New(
			"gRPC exporter endpoint is required",
		)
	}

	if strings.ContainsAny(
		c.Endpoint,
		"\r\n",
	) {
		return errors.New(
			"gRPC exporter endpoint contains a line break",
		)
	}

	if strings.ContainsAny(
		c.Authority,
		"\r\n",
	) {
		return errors.New(
			"gRPC authority contains a line break",
		)
	}

	if err := c.Authentication.Validate(); err != nil {
		return fmt.Errorf(
			"gRPC authentication configuration: %w",
			err,
		)
	}

	if err := c.TLS.Validate(); err != nil {
		return fmt.Errorf(
			"gRPC TLS configuration: %w",
			err,
		)
	}

	if err := c.Timeouts.Validate(); err != nil {
		return fmt.Errorf(
			"gRPC timeout configuration: %w",
			err,
		)
	}

	if err := c.Keepalive.Validate(); err != nil {
		return fmt.Errorf(
			"gRPC keepalive configuration: %w",
			err,
		)
	}

	if err := c.Reconnect.Validate(); err != nil {
		return fmt.Errorf(
			"gRPC reconnect configuration: %w",
			err,
		)
	}

	if c.MaxSendMessageBytes <= 0 {
		return errors.New(
			"gRPC max send message bytes must be greater than zero",
		)
	}

	if c.MaxReceiveMessageBytes <= 0 {
		return errors.New(
			"gRPC max receive message bytes must be greater than zero",
		)
	}

	for key, value := range c.Metadata {
		if !validMetadataKey(key) {
			return fmt.Errorf(
				"invalid gRPC metadata key %q",
				key,
			)
		}

		if strings.ContainsAny(
			value,
			"\r\n",
		) {
			return fmt.Errorf(
				"gRPC metadata %q contains a line break",
				key,
			)
		}

		if sensitiveMetadataKey(key) {
			return fmt.Errorf(
				"sensitive gRPC metadata %q must be configured through Authentication",
				key,
			)
		}
	}

	return nil
}

// Normalized returns a defensive configuration copy with defaults that can be
// safely derived at runtime.
func (c Config) Normalized() Config {
	normalized := c

	normalized.ID = strings.TrimSpace(
		c.ID,
	)

	normalized.Endpoint = strings.TrimSpace(
		c.Endpoint,
	)

	normalized.Authority = strings.TrimSpace(
		c.Authority,
	)

	if normalized.Authentication.Type == "" {
		normalized.Authentication.Type =
			AuthenticationNone
	}

	if normalized.Authentication.Type ==
		AuthenticationAPIKey &&
		strings.TrimSpace(
			normalized.Authentication.
				APIKeyMetadataKey,
		) == "" {
		normalized.Authentication.
			APIKeyMetadataKey = "x-api-key"
	}

	if normalized.TLS.Enabled &&
		normalized.TLS.MinVersion == 0 {
		normalized.TLS.MinVersion =
			tls.VersionTLS12
	}

	normalized.Metadata = cloneMetadata(
		c.Metadata,
	)

	return normalized
}

func validMetadataKey(
	key string,
) bool {
	key = strings.TrimSpace(key)

	if key == "" {
		return false
	}

	// gRPC metadata keys are represented using lowercase ASCII keys in our
	// OTGH configuration. Constraining them here prevents ambiguous
	// configuration and simplifies signing/logging rules later.
	if key != strings.ToLower(key) {
		return false
	}

	for _, character := range key {
		switch {
		case character >= 'a' &&
			character <= 'z':
			continue

		case character >= '0' &&
			character <= '9':
			continue

		case character == '-',
			character == '_',
			character == '.':
			continue

		default:
			return false
		}
	}

	return true
}

func sensitiveMetadataKey(
	key string,
) bool {
	switch strings.ToLower(
		strings.TrimSpace(key),
	) {
	case "authorization",
		"proxy-authorization",
		"x-api-key":
		return true

	default:
		return false
	}
}

func cloneMetadata(
	input map[string]string,
) map[string]string {
	if input == nil {
		return nil
	}

	cloned := make(
		map[string]string,
		len(input),
	)

	for key, value := range input {
		cloned[key] = value
	}

	return cloned
}
