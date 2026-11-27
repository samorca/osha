package http

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// AuthenticationType identifies how the HTTP exporter authenticates to its
// destination.
type AuthenticationType string

const (
	AuthenticationNone   AuthenticationType = "none"
	AuthenticationBearer AuthenticationType = "bearer"
	AuthenticationAPIKey AuthenticationType = "api_key"
	AuthenticationBasic  AuthenticationType = "basic"
)

// Valid reports whether the authentication type is recognized.
func (a AuthenticationType) Valid() bool {
	switch a {
	case AuthenticationNone,
		AuthenticationBearer,
		AuthenticationAPIKey,
		AuthenticationBasic:
		return true

	default:
		return false
	}
}

// TLSConfig controls TLS verification and optional mutual TLS.
//
// Certificates are configured by file path rather than embedding certificate
// contents directly in application configuration structures.
type TLSConfig struct {
	// Enabled requires an HTTPS endpoint.
	Enabled bool

	// CAFile optionally contains a PEM encoded CA certificate bundle used to
	// verify the remote server.
	CAFile string

	// CertFile and KeyFile enable client certificate authentication.
	//
	// Both must be supplied together.
	CertFile string
	KeyFile  string

	// ServerName optionally overrides the hostname used for TLS certificate
	// verification.
	ServerName string

	// MinVersion is a crypto/tls protocol-version constant.
	//
	// Zero means TLS 1.2.
	MinVersion uint16
}

// Validate verifies TLS configuration.
func (c TLSConfig) Validate() error {
	if !c.Enabled {
		if strings.TrimSpace(c.CAFile) != "" ||
			strings.TrimSpace(c.CertFile) != "" ||
			strings.TrimSpace(c.KeyFile) != "" ||
			strings.TrimSpace(c.ServerName) != "" {
			return errors.New(
				"TLS files or server name require TLS to be enabled",
			)
		}

		return nil
	}

	certFile := strings.TrimSpace(c.CertFile)
	keyFile := strings.TrimSpace(c.KeyFile)

	if (certFile == "") != (keyFile == "") {
		return errors.New(
			"TLS client certificate and key must be configured together",
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
			"unsupported minimum TLS version 0x%x",
			minVersion,
		)
	}
}

// AuthenticationConfig controls credentials attached to HTTP requests.
//
// Secrets should be supplied through environment variables, container
// secrets, Kubernetes Secrets, or another secret provider. They should not be
// committed to source control.
type AuthenticationConfig struct {
	Type AuthenticationType

	// BearerToken is used with AuthenticationBearer.
	BearerToken string

	// APIKey is used with AuthenticationAPIKey.
	APIKey string

	// APIKeyHeader defaults to X-API-Key when omitted.
	APIKeyHeader string

	// Username and Password are used with AuthenticationBasic.
	Username string
	Password string
}

// Validate verifies authentication configuration.
func (c AuthenticationConfig) Validate() error {
	if !c.Type.Valid() {
		return fmt.Errorf(
			"unsupported HTTP authentication type %q",
			c.Type,
		)
	}

	switch c.Type {
	case AuthenticationNone:
		if c.hasCredentialMaterial() {
			return errors.New(
				"HTTP credentials supplied while authentication type is none",
			)
		}

	case AuthenticationBearer:
		if strings.TrimSpace(c.BearerToken) == "" {
			return errors.New(
				"bearer authentication requires a token",
			)
		}

	case AuthenticationAPIKey:
		if strings.TrimSpace(c.APIKey) == "" {
			return errors.New(
				"API key authentication requires a key",
			)
		}

		header := strings.TrimSpace(c.APIKeyHeader)
		if header != "" && !validHeaderName(header) {
			return fmt.Errorf(
				"invalid API key header name %q",
				header,
			)
		}

	case AuthenticationBasic:
		if strings.TrimSpace(c.Username) == "" {
			return errors.New(
				"basic authentication requires a username",
			)
		}

		if c.Password == "" {
			return errors.New(
				"basic authentication requires a password",
			)
		}
	}

	return nil
}

func (c AuthenticationConfig) hasCredentialMaterial() bool {
	return c.BearerToken != "" ||
		c.APIKey != "" ||
		c.APIKeyHeader != "" ||
		c.Username != "" ||
		c.Password != ""
}

// HTTPTimeoutConfig controls HTTP client timing limits.
type HTTPTimeoutConfig struct {
	// Request is the overall request timeout.
	Request time.Duration

	// Connect limits establishing a network connection.
	Connect time.Duration

	// TLSHandshake limits TLS negotiation.
	TLSHandshake time.Duration

	// ResponseHeader limits waiting for response headers after the request has
	// been written.
	ResponseHeader time.Duration

	// ExpectContinue limits waiting for a 100-continue response.
	ExpectContinue time.Duration

	// IdleConnection controls how long reusable idle connections remain in the
	// pool.
	IdleConnection time.Duration
}

// DefaultHTTPTimeoutConfig returns conservative defaults for small/mid-sized
// API delivery requests.
func DefaultHTTPTimeoutConfig() HTTPTimeoutConfig {
	return HTTPTimeoutConfig{
		Request:        30 * time.Second,
		Connect:        5 * time.Second,
		TLSHandshake:   5 * time.Second,
		ResponseHeader: 15 * time.Second,
		ExpectContinue: time.Second,
		IdleConnection: 90 * time.Second,
	}
}

// Validate verifies HTTP timeout configuration.
func (c HTTPTimeoutConfig) Validate() error {
	values := []struct {
		name  string
		value time.Duration
	}{
		{"request", c.Request},
		{"connect", c.Connect},
		{"TLS handshake", c.TLSHandshake},
		{"response header", c.ResponseHeader},
		{"expect continue", c.ExpectContinue},
		{"idle connection", c.IdleConnection},
	}

	for _, item := range values {
		if item.value <= 0 {
			return fmt.Errorf(
				"HTTP %s timeout must be greater than zero",
				item.name,
			)
		}
	}

	return nil
}

// ConnectionConfig controls HTTP connection pooling.
type ConnectionConfig struct {
	// MaxIdleConnections limits idle connections across all hosts.
	MaxIdleConnections int

	// MaxIdleConnectionsPerHost limits reusable idle connections per
	// destination host.
	MaxIdleConnectionsPerHost int

	// MaxConnectionsPerHost limits total connections per host.
	//
	// Zero means net/http may choose an unlimited total connection count.
	MaxConnectionsPerHost int
}

// DefaultConnectionConfig returns baseline pool settings.
func DefaultConnectionConfig() ConnectionConfig {
	return ConnectionConfig{
		MaxIdleConnections:        100,
		MaxIdleConnectionsPerHost: 20,
		MaxConnectionsPerHost:     100,
	}
}

// Validate verifies connection-pool configuration.
func (c ConnectionConfig) Validate() error {
	if c.MaxIdleConnections <= 0 {
		return errors.New(
			"max idle connections must be greater than zero",
		)
	}

	if c.MaxIdleConnectionsPerHost <= 0 {
		return errors.New(
			"max idle connections per host must be greater than zero",
		)
	}

	if c.MaxConnectionsPerHost < 0 {
		return errors.New(
			"max connections per host cannot be negative",
		)
	}

	if c.MaxIdleConnectionsPerHost >
		c.MaxIdleConnections {
		return errors.New(
			"max idle connections per host cannot exceed total max idle connections",
		)
	}

	return nil
}

// Config describes one OTGH HTTP export destination.
type Config struct {
	// ID must match the DestinationID registered with the export system.
	ID string

	// Endpoint is the complete HTTP(S) destination URL.
	Endpoint string

	// Method is normally POST.
	//
	// Empty defaults to POST.
	Method string

	// ContentType defaults to application/json.
	ContentType string

	// UserAgent identifies OTGH to the destination.
	UserAgent string

	// Headers contains static non-sensitive request headers.
	//
	// Authentication headers should be configured via Authentication instead
	// of placing secrets in this map.
	Headers map[string]string

	Authentication AuthenticationConfig

	TLS TLSConfig

	Timeouts HTTPTimeoutConfig

	Connections ConnectionConfig

	// MaxRequestBytes prevents unexpectedly large export requests.
	MaxRequestBytes int64

	// MaxResponseBytes limits the response body read into memory.
	MaxResponseBytes int64

	// FollowRedirects determines whether redirects are allowed.
	//
	// Export destinations should normally be stable explicit endpoints, so
	// the default configuration disables redirects.
	FollowRedirects bool
}

// DefaultConfig returns baseline HTTP-export configuration.
//
// Endpoint and ID remain empty because the caller must provide them.
func DefaultConfig() Config {
	return Config{
		Method:      "POST",
		ContentType: "application/json",
		UserAgent:   "otgh/1",

		Authentication: AuthenticationConfig{
			Type: AuthenticationNone,
		},

		TLS: TLSConfig{
			Enabled:    true,
			MinVersion: tls.VersionTLS12,
		},

		Timeouts: DefaultHTTPTimeoutConfig(),

		Connections: DefaultConnectionConfig(),

		MaxRequestBytes:  64 * 1024 * 1024,
		MaxResponseBytes: 1024 * 1024,

		FollowRedirects: false,
	}
}

// Validate performs complete configuration validation.
func (c Config) Validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return errors.New(
			"HTTP exporter id is required",
		)
	}

	endpoint := strings.TrimSpace(c.Endpoint)

	if endpoint == "" {
		return errors.New(
			"HTTP exporter endpoint is required",
		)
	}

	parsed, err := url.ParseRequestURI(endpoint)
	if err != nil {
		return fmt.Errorf(
			"invalid HTTP exporter endpoint %q: %w",
			endpoint,
			err,
		)
	}

	if parsed.Host == "" {
		return errors.New(
			"HTTP exporter endpoint must contain a host",
		)
	}

	switch parsed.Scheme {
	case "http":
		if c.TLS.Enabled {
			return errors.New(
				"TLS-enabled HTTP exporter requires an https endpoint",
			)
		}

	case "https":
		if !c.TLS.Enabled {
			return errors.New(
				"https endpoint requires TLS to be enabled",
			)
		}

	default:
		return fmt.Errorf(
			"unsupported HTTP endpoint scheme %q",
			parsed.Scheme,
		)
	}

	method := strings.TrimSpace(c.Method)

	if method == "" {
		method = "POST"
	}

	if !validHTTPMethod(method) {
		return fmt.Errorf(
			"invalid HTTP method %q",
			method,
		)
	}

	if strings.TrimSpace(c.ContentType) == "" {
		return errors.New(
			"HTTP content type is required",
		)
	}

	for name, value := range c.Headers {
		if !validHeaderName(name) {
			return fmt.Errorf(
				"invalid HTTP header name %q",
				name,
			)
		}

		if containsLineBreak(value) {
			return fmt.Errorf(
				"HTTP header %q contains a line break",
				name,
			)
		}

		if sensitiveHeader(name) {
			return fmt.Errorf(
				"sensitive HTTP header %q must be configured through authentication or secret-aware configuration",
				name,
			)
		}
	}

	if err := c.Authentication.Validate(); err != nil {
		return fmt.Errorf(
			"HTTP authentication configuration: %w",
			err,
		)
	}

	if err := c.TLS.Validate(); err != nil {
		return fmt.Errorf(
			"HTTP TLS configuration: %w",
			err,
		)
	}

	if err := c.Timeouts.Validate(); err != nil {
		return fmt.Errorf(
			"HTTP timeout configuration: %w",
			err,
		)
	}

	if err := c.Connections.Validate(); err != nil {
		return fmt.Errorf(
			"HTTP connection configuration: %w",
			err,
		)
	}

	if c.MaxRequestBytes <= 0 {
		return errors.New(
			"HTTP max request bytes must be greater than zero",
		)
	}

	if c.MaxResponseBytes <= 0 {
		return errors.New(
			"HTTP max response bytes must be greater than zero",
		)
	}

	return nil
}

// Normalized returns a configuration copy with protocol defaults applied.
func (c Config) Normalized() Config {
	normalized := c

	normalized.ID = strings.TrimSpace(
		normalized.ID,
	)

	normalized.Endpoint = strings.TrimSpace(
		normalized.Endpoint,
	)

	normalized.Method = strings.ToUpper(
		strings.TrimSpace(normalized.Method),
	)

	if normalized.Method == "" {
		normalized.Method = "POST"
	}

	normalized.ContentType = strings.TrimSpace(
		normalized.ContentType,
	)

	if normalized.ContentType == "" {
		normalized.ContentType = "application/json"
	}

	normalized.UserAgent = strings.TrimSpace(
		normalized.UserAgent,
	)

	if normalized.UserAgent == "" {
		normalized.UserAgent = "otgh/1"
	}

	if normalized.Authentication.Type == "" {
		normalized.Authentication.Type =
			AuthenticationNone
	}

	if normalized.Authentication.Type ==
		AuthenticationAPIKey &&
		strings.TrimSpace(
			normalized.Authentication.APIKeyHeader,
		) == "" {
		normalized.Authentication.APIKeyHeader =
			"X-API-Key"
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

func validHTTPMethod(method string) bool {
	if method == "" {
		return false
	}

	for _, character := range method {
		switch {
		case character >= 'A' && character <= 'Z',
			character >= 'a' && character <= 'z':
			continue

		default:
			return false
		}
	}

	return true
}

func validHeaderName(name string) bool {
	name = strings.TrimSpace(name)

	if name == "" {
		return false
	}

	for _, character := range name {
		switch {
		case character >= 'a' && character <= 'z',
			character >= 'A' && character <= 'Z',
			character >= '0' && character <= '9':
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

func containsLineBreak(value string) bool {
	return strings.ContainsAny(
		value,
		"\r\n",
	)
}

func sensitiveHeader(name string) bool {
	switch strings.ToLower(
		strings.TrimSpace(name),
	) {
	case "authorization",
		"proxy-authorization",
		"cookie",
		"set-cookie",
		"x-api-key":
		return true

	default:
		return false
	}
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

	for name, value := range input {
		cloned[name] = value
	}

	return cloned
}
