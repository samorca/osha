package webhook

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"
)

// AuthenticationType identifies optional HTTP-level authentication.
//
// Webhook signing and HTTP authentication are separate controls. A destination
// can require both.
type AuthenticationType string

const (
	AuthenticationNone   AuthenticationType = "none"
	AuthenticationBearer AuthenticationType = "bearer"
	AuthenticationBasic  AuthenticationType = "basic"
)

// Valid reports whether the authentication type is recognized.
func (a AuthenticationType) Valid() bool {
	switch a {
	case AuthenticationNone,
		AuthenticationBearer,
		AuthenticationBasic:
		return true

	default:
		return false
	}
}

// AuthenticationConfig controls optional HTTP authentication.
type AuthenticationConfig struct {
	Type AuthenticationType

	BearerToken string

	Username string
	Password string
}

// Validate verifies HTTP authentication configuration.
func (c AuthenticationConfig) Validate() error {
	if !c.Type.Valid() {
		return fmt.Errorf(
			"unsupported webhook authentication type %q",
			c.Type,
		)
	}

	switch c.Type {
	case AuthenticationNone:
		if c.BearerToken != "" ||
			c.Username != "" ||
			c.Password != "" {
			return errors.New(
				"webhook credentials supplied while authentication type is none",
			)
		}

	case AuthenticationBearer:
		if strings.TrimSpace(c.BearerToken) == "" {
			return errors.New(
				"webhook bearer authentication requires a token",
			)
		}

	case AuthenticationBasic:
		if strings.TrimSpace(c.Username) == "" {
			return errors.New(
				"webhook basic authentication requires a username",
			)
		}

		if c.Password == "" {
			return errors.New(
				"webhook basic authentication requires a password",
			)
		}
	}

	return nil
}

// SignatureAlgorithm identifies the webhook payload authentication algorithm.
type SignatureAlgorithm string

const (
	SignatureHMACSHA256 SignatureAlgorithm = "hmac-sha256"
)

// Valid reports whether the signing algorithm is supported.
func (a SignatureAlgorithm) Valid() bool {
	return a == SignatureHMACSHA256
}

// SigningConfig controls webhook payload signing.
//
// OTGH signs:
//
//	delivery-id + "." + timestamp + "." + raw-body
//
// where timestamp is the UTC Unix timestamp represented in decimal.
//
// Binding the delivery ID, timestamp, and exact raw payload into the HMAC lets
// the receiver authenticate all three values together.
type SigningConfig struct {
	Enabled bool

	Algorithm SignatureAlgorithm

	// Secret is the shared HMAC secret.
	//
	// It should be injected from a secret store/environment and must never be
	// logged.
	Secret string

	// KeyID identifies the signing key so receivers can support rotation.
	KeyID string

	DeliveryIDHeader string
	TimestampHeader  string
	SignatureHeader  string
	KeyIDHeader      string
}

// DefaultSigningConfig returns OTGH's webhook signing defaults.
//
// Secret and KeyID remain empty because the deployment must supply them.
func DefaultSigningConfig() SigningConfig {
	return SigningConfig{
		Enabled:   true,
		Algorithm: SignatureHMACSHA256,

		DeliveryIDHeader: "Webhook-Id",
		TimestampHeader:  "Webhook-Timestamp",
		SignatureHeader:  "Webhook-Signature",
		KeyIDHeader:      "Webhook-Key-Id",
	}
}

// Validate verifies signing configuration.
func (c SigningConfig) Validate() error {
	if !c.Enabled {
		if c.Secret != "" ||
			c.KeyID != "" {
			return errors.New(
				"webhook signing secret/key id supplied while signing is disabled",
			)
		}

		return nil
	}

	if !c.Algorithm.Valid() {
		return fmt.Errorf(
			"unsupported webhook signature algorithm %q",
			c.Algorithm,
		)
	}

	if len(c.Secret) < 32 {
		return errors.New(
			"webhook signing secret must contain at least 32 bytes",
		)
	}

	if strings.TrimSpace(c.KeyID) == "" {
		return errors.New(
			"webhook signing key id is required",
		)
	}

	headers := map[string]string{
		"delivery id": c.DeliveryIDHeader,
		"timestamp":   c.TimestampHeader,
		"signature":   c.SignatureHeader,
		"key id":      c.KeyIDHeader,
	}

	seen := make(
		map[string]string,
		len(headers),
	)

	for purpose, header := range headers {
		header = strings.TrimSpace(header)

		if !validHeaderName(header) {
			return fmt.Errorf(
				"invalid webhook %s header %q",
				purpose,
				header,
			)
		}

		canonical := strings.ToLower(header)

		if existing, exists :=
			seen[canonical]; exists {
			return fmt.Errorf(
				"webhook %s header duplicates %s header",
				purpose,
				existing,
			)
		}

		seen[canonical] = purpose
	}

	return nil
}

// TLSConfig controls webhook HTTPS transport security.
type TLSConfig struct {
	Enabled bool

	CAFile string

	CertFile string
	KeyFile  string

	ServerName string

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
				"webhook TLS files/server name require TLS to be enabled",
			)
		}

		return nil
	}

	if (certFile == "") != (keyFile == "") {
		return errors.New(
			"webhook client certificate and private key must be configured together",
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
			"unsupported webhook minimum TLS version 0x%x",
			minVersion,
		)
	}
}

// TimeoutConfig controls webhook HTTP timing.
type TimeoutConfig struct {
	Request time.Duration

	Connect time.Duration

	TLSHandshake time.Duration

	ResponseHeader time.Duration

	IdleConnection time.Duration
}

// DefaultTimeoutConfig returns baseline webhook timeout values.
func DefaultTimeoutConfig() TimeoutConfig {
	return TimeoutConfig{
		Request:        15 * time.Second,
		Connect:        5 * time.Second,
		TLSHandshake:   5 * time.Second,
		ResponseHeader: 10 * time.Second,
		IdleConnection: 60 * time.Second,
	}
}

// Validate verifies timeout configuration.
func (c TimeoutConfig) Validate() error {
	values := []struct {
		name  string
		value time.Duration
	}{
		{"request", c.Request},
		{"connect", c.Connect},
		{"TLS handshake", c.TLSHandshake},
		{"response header", c.ResponseHeader},
		{"idle connection", c.IdleConnection},
	}

	for _, item := range values {
		if item.value <= 0 {
			return fmt.Errorf(
				"webhook %s timeout must be greater than zero",
				item.name,
			)
		}
	}

	return nil
}

// NetworkPolicy controls which webhook endpoints may be targeted.
//
// Webhooks can turn a publisher into an outbound-request engine. The endpoint
// should therefore not silently gain access to loopback/private/link-local
// infrastructure unless the deployment explicitly opts into it.
type NetworkPolicy struct {
	// AllowHTTP permits plain HTTP callback destinations.
	//
	// Production deployments should leave this false.
	AllowHTTP bool

	// AllowPrivateNetworks permits literal private/link-local IP targets.
	//
	// DNS resolution must also be checked at connection time by the exporter
	// because a hostname can resolve to a private address.
	AllowPrivateNetworks bool
}

// Config describes one OTGH webhook destination.
type Config struct {
	ID string

	URL string

	UserAgent string

	ContentType string

	Headers map[string]string

	Authentication AuthenticationConfig

	Signing SigningConfig

	TLS TLSConfig

	Timeouts TimeoutConfig

	Network NetworkPolicy

	MaxPayloadBytes int64

	MaxResponseBytes int64

	// FollowRedirects is false by default because a redirect can change the
	// hosting trust boundary after URL validation.
	FollowRedirects bool

	// MaxIdleConnectionsPerHost controls reusable HTTP connections.
	MaxIdleConnectionsPerHost int
}

// DefaultConfig returns secure baseline webhook settings.
//
// ID, URL, signing Secret, and signing KeyID remain deployment-specific.
func DefaultConfig() Config {
	return Config{
		UserAgent:   "otgh-webhook/1",
		ContentType: "application/json",

		Authentication: AuthenticationConfig{
			Type: AuthenticationNone,
		},

		Signing: DefaultSigningConfig(),

		TLS: TLSConfig{
			Enabled:    true,
			MinVersion: tls.VersionTLS12,
		},

		Timeouts: DefaultTimeoutConfig(),

		Network: NetworkPolicy{
			AllowHTTP:            false,
			AllowPrivateNetworks: false,
		},

		MaxPayloadBytes:  4 * 1024 * 1024,
		MaxResponseBytes: 256 * 1024,

		FollowRedirects: false,

		MaxIdleConnectionsPerHost: 10,
	}
}

// Validate performs complete webhook configuration validation.
func (c Config) Validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return errors.New(
			"webhook exporter id is required",
		)
	}

	rawURL := strings.TrimSpace(c.URL)

	if rawURL == "" {
		return errors.New(
			"webhook URL is required",
		)
	}

	parsed, err := url.ParseRequestURI(rawURL)
	if err != nil {
		return fmt.Errorf(
			"invalid webhook URL %q: %w",
			rawURL,
			err,
		)
	}

	if parsed.Host == "" {
		return errors.New(
			"webhook URL must contain a host",
		)
	}

	if parsed.User != nil {
		return errors.New(
			"webhook URL must not contain embedded credentials",
		)
	}

	switch parsed.Scheme {
	case "https":
		if !c.TLS.Enabled {
			return errors.New(
				"https webhook requires TLS to be enabled",
			)
		}

	case "http":
		if !c.Network.AllowHTTP {
			return errors.New(
				"plain HTTP webhook URLs are disabled by network policy",
			)
		}

		if c.TLS.Enabled {
			return errors.New(
				"HTTP webhook cannot enable TLS",
			)
		}

	default:
		return fmt.Errorf(
			"unsupported webhook URL scheme %q",
			parsed.Scheme,
		)
	}

	host := parsed.Hostname()

	if host == "" {
		return errors.New(
			"webhook URL hostname is empty",
		)
	}

	if ip := net.ParseIP(host); ip != nil &&
		!c.Network.AllowPrivateNetworks &&
		nonPublicIP(ip) {
		return fmt.Errorf(
			"webhook IP %q is blocked by network policy",
			host,
		)
	}

	if strings.TrimSpace(c.ContentType) == "" {
		return errors.New(
			"webhook content type is required",
		)
	}

	for name, value := range c.Headers {
		if !validHeaderName(name) {
			return fmt.Errorf(
				"invalid webhook header %q",
				name,
			)
		}

		if strings.ContainsAny(
			value,
			"\r\n",
		) {
			return fmt.Errorf(
				"webhook header %q contains a line break",
				name,
			)
		}

		if sensitiveHeader(name) {
			return fmt.Errorf(
				"sensitive webhook header %q must use Authentication or Signing configuration",
				name,
			)
		}
	}

	if err := c.Authentication.Validate(); err != nil {
		return fmt.Errorf(
			"webhook authentication: %w",
			err,
		)
	}

	if err := c.Signing.Validate(); err != nil {
		return fmt.Errorf(
			"webhook signing: %w",
			err,
		)
	}

	if err := c.TLS.Validate(); err != nil {
		return fmt.Errorf(
			"webhook TLS: %w",
			err,
		)
	}

	if err := c.Timeouts.Validate(); err != nil {
		return fmt.Errorf(
			"webhook timeouts: %w",
			err,
		)
	}

	if c.MaxPayloadBytes <= 0 {
		return errors.New(
			"webhook max payload bytes must be greater than zero",
		)
	}

	if c.MaxResponseBytes <= 0 {
		return errors.New(
			"webhook max response bytes must be greater than zero",
		)
	}

	if c.MaxIdleConnectionsPerHost <= 0 {
		return errors.New(
			"webhook max idle connections per host must be greater than zero",
		)
	}

	if c.FollowRedirects &&
		!c.Network.AllowPrivateNetworks {
		// Redirects require re-validating every destination and every
		// connection target. exporter.go could support this later, but the
		// safest initial contract is to reject this ambiguous combination.
		return errors.New(
			"webhook redirects cannot be enabled while private-network targets are blocked",
		)
	}

	return nil
}

// Normalized applies deterministic defaults and defensive copies.
func (c Config) Normalized() Config {
	normalized := c

	normalized.ID = strings.TrimSpace(
		c.ID,
	)

	normalized.URL = strings.TrimSpace(
		c.URL,
	)

	normalized.UserAgent = strings.TrimSpace(
		c.UserAgent,
	)

	if normalized.UserAgent == "" {
		normalized.UserAgent =
			"otgh-webhook/1"
	}

	normalized.ContentType = strings.TrimSpace(
		c.ContentType,
	)

	if normalized.ContentType == "" {
		normalized.ContentType =
			"application/json"
	}

	if normalized.Authentication.Type == "" {
		normalized.Authentication.Type =
			AuthenticationNone
	}

	if normalized.Signing.Enabled &&
		normalized.Signing.Algorithm == "" {
		normalized.Signing.Algorithm =
			SignatureHMACSHA256
	}

	if normalized.Signing.DeliveryIDHeader == "" {
		normalized.Signing.DeliveryIDHeader =
			"Webhook-Id"
	}

	if normalized.Signing.TimestampHeader == "" {
		normalized.Signing.TimestampHeader =
			"Webhook-Timestamp"
	}

	if normalized.Signing.SignatureHeader == "" {
		normalized.Signing.SignatureHeader =
			"Webhook-Signature"
	}

	if normalized.Signing.KeyIDHeader == "" {
		normalized.Signing.KeyIDHeader =
			"Webhook-Key-Id"
	}

	if normalized.TLS.Enabled &&
		normalized.TLS.MinVersion == 0 {
		normalized.TLS.MinVersion =
			tls.VersionTLS12
	}

	normalized.Headers = cloneHeaders(
		c.Headers,
	)

	return normalized
}

func validHeaderName(
	name string,
) bool {
	name = strings.TrimSpace(name)

	if name == "" {
		return false
	}

	for _, character := range name {
		switch {
		case character >= 'a' &&
			character <= 'z',
			character >= 'A' &&
				character <= 'Z',
			character >= '0' &&
				character <= '9':
			continue

		case character == '!',
			character == '#',
			character == '$',
			character == '%',
			character == '&',
			character == '\'',
			character == '*',
			character == '+',
			character == '-',
			character == '.',
			character == '^',
			character == '_',
			character == '`',
			character == '|',
			character == '~':
			continue

		default:
			return false
		}
	}

	return true
}

func sensitiveHeader(
	name string,
) bool {
	switch strings.ToLower(
		strings.TrimSpace(name),
	) {
	case "authorization",
		"proxy-authorization",
		"cookie",
		"set-cookie",
		"x-api-key",
		"webhook-id",
		"webhook-timestamp",
		"webhook-signature",
		"webhook-key-id":
		return true

	default:
		return false
	}
}

func nonPublicIP(
	ip net.IP,
) bool {
	if ip == nil {
		return true
	}

	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() ||
		ip.IsMulticast()
}

func cloneHeaders(
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
