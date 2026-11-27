package http

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	nethttp "net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrClientClosed      = errors.New("HTTP client closed")
	ErrResponseTooLarge  = errors.New("HTTP response too large")
	ErrRequestTooLarge   = errors.New("HTTP request too large")
	ErrRedirectDisallowed = errors.New("HTTP redirect disallowed")
)

// Response contains the bounded HTTP response information needed by the
// exporter.
//
// The underlying response body is read and closed by Client.Do.
type Response struct {
	StatusCode int
	Status     string

	Header nethttp.Header

	Body []byte

	ReceivedAt time.Time
}

// Success reports whether the response is a 2xx status code.
func (r Response) Success() bool {
	return r.StatusCode >= 200 &&
		r.StatusCode <= 299
}

// Client is OTGH's reusable HTTP delivery client.
//
// It owns one net/http Client and Transport. Both are intended to be reused
// across requests so connections can be pooled.
//
// Client is safe for concurrent use.
type Client struct {
	config Config

	httpClient *nethttp.Client
	transport  *nethttp.Transport

	closeOnce sync.Once
	closed    atomic.Bool
}

// NewClient builds a reusable HTTP client from Config.
func NewClient(
	config Config,
) (*Client, error) {
	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create HTTP client: %w",
			err,
		)
	}

	tlsConfig, err := buildTLSConfig(
		config,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create HTTP TLS configuration: %w",
			err,
		)
	}

	dialer := &net.Dialer{
		Timeout:   config.Timeouts.Connect,
		KeepAlive: 30 * time.Second,
	}

	transport := &nethttp.Transport{
		Proxy: nethttp.ProxyFromEnvironment,

		DialContext: dialer.DialContext,

		ForceAttemptHTTP2: true,

		MaxIdleConns: config.Connections.
			MaxIdleConnections,

		MaxIdleConnsPerHost: config.Connections.
			MaxIdleConnectionsPerHost,

		MaxConnsPerHost: config.Connections.
			MaxConnectionsPerHost,

		IdleConnTimeout: config.Timeouts.
			IdleConnection,

		TLSHandshakeTimeout: config.Timeouts.
			TLSHandshake,

		ResponseHeaderTimeout: config.Timeouts.
			ResponseHeader,

		ExpectContinueTimeout: config.Timeouts.
			ExpectContinue,

		TLSClientConfig: tlsConfig,
	}

	httpClient := &nethttp.Client{
		Transport: transport,
		Timeout:   config.Timeouts.Request,
	}

	if !config.FollowRedirects {
		httpClient.CheckRedirect =
			func(
				request *nethttp.Request,
				via []*nethttp.Request,
			) error {
				return nethttp.ErrUseLastResponse
			}
	}

	return &Client{
		config:     config,
		httpClient: httpClient,
		transport:  transport,
	}, nil
}

// Config returns a defensive copy of the normalized configuration.
//
// Credential values are retained because this structure is intended for
// internal programmatic use. Callers must never log the returned value.
func (c *Client) Config() Config {
	if c == nil {
		return Config{}
	}

	config := c.config
	config.Headers = cloneHeaders(
		c.config.Headers,
	)

	return config
}

// Do sends one HTTP request body to the configured destination.
//
// payload must already be serialized by the exporter.
//
// Do:
//   - enforces MaxRequestBytes;
//   - applies configured headers and authentication;
//   - respects context cancellation/deadlines;
//   - performs the request through the reusable connection pool;
//   - reads at most MaxResponseBytes + 1;
//   - closes the network response body before returning.
func (c *Client) Do(
	ctx context.Context,
	payload []byte,
) (Response, error) {
	if c == nil {
		return Response{}, errors.New(
			"HTTP client is nil",
		)
	}

	if ctx == nil {
		return Response{}, errors.New(
			"HTTP request context is nil",
		)
	}

	if c.closed.Load() {
		return Response{}, ErrClientClosed
	}

	if err := ctx.Err(); err != nil {
		return Response{}, err
	}

	if len(payload) == 0 {
		return Response{}, errors.New(
			"HTTP request payload is empty",
		)
	}

	if int64(len(payload)) >
		c.config.MaxRequestBytes {
		return Response{}, fmt.Errorf(
			"%w: request contains %d bytes, maximum is %d",
			ErrRequestTooLarge,
			len(payload),
			c.config.MaxRequestBytes,
		)
	}

	request, err :=
		nethttp.NewRequestWithContext(
			ctx,
			c.config.Method,
			c.config.Endpoint,
			bytes.NewReader(payload),
		)
	if err != nil {
		return Response{}, fmt.Errorf(
			"create HTTP request: %w",
			err,
		)
	}

	request.ContentLength =
		int64(len(payload))

	if err := c.applyHeaders(request); err != nil {
		return Response{}, err
	}

	response, err := c.httpClient.Do(
		request,
	)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Response{}, ctxErr
		}

		return Response{}, fmt.Errorf(
			"execute HTTP request: %w",
			err,
		)
	}

	defer response.Body.Close()

	body, err := readResponseBody(
		response.Body,
		c.config.MaxResponseBytes,
	)
	if err != nil {
		return Response{}, fmt.Errorf(
			"read HTTP response: %w",
			err,
		)
	}

	return Response{
		StatusCode: response.StatusCode,
		Status:     response.Status,

		Header: response.Header.Clone(),

		Body: append(
			[]byte(nil),
			body...,
		),

		ReceivedAt: time.Now().UTC(),
	}, nil
}

// Health performs a lightweight HTTP request to the configured endpoint.
//
// It intentionally sends no export payload. The caller should only use this
// when the destination endpoint/method semantics make an empty request a
// meaningful health check.
//
// Most OTGH deployments should eventually configure a separate health
// endpoint rather than assume the export endpoint supports this operation.
func (c *Client) Health(
	ctx context.Context,
) error {
	if c == nil {
		return errors.New(
			"HTTP client is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"HTTP health context is nil",
		)
	}

	if c.closed.Load() {
		return ErrClientClosed
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	request, err :=
		nethttp.NewRequestWithContext(
			ctx,
			nethttp.MethodHead,
			c.config.Endpoint,
			nil,
		)
	if err != nil {
		return fmt.Errorf(
			"create HTTP health request: %w",
			err,
		)
	}

	if err := c.applyHeaders(request); err != nil {
		return err
	}

	response, err := c.httpClient.Do(request)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}

		return fmt.Errorf(
			"execute HTTP health request: %w",
			err,
		)
	}
	defer response.Body.Close()

	_, _ = io.Copy(
		io.Discard,
		io.LimitReader(response.Body, 4096),
	)

	if response.StatusCode >= 200 &&
		response.StatusCode <= 399 {
		return nil
	}

	return fmt.Errorf(
		"HTTP health endpoint returned status %d",
		response.StatusCode,
	)
}

// Close releases idle network connections.
//
// It is safe to call Close multiple times.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}

	c.closeOnce.Do(func() {
		c.closed.Store(true)

		if c.transport != nil {
			c.transport.CloseIdleConnections()
		}
	})

	return nil
}

// Closed reports whether Close has been called.
func (c *Client) Closed() bool {
	if c == nil {
		return true
	}

	return c.closed.Load()
}

func (c *Client) applyHeaders(
	request *nethttp.Request,
) error {
	if request == nil {
		return errors.New(
			"HTTP request is nil",
		)
	}

	request.Header.Set(
		"Content-Type",
		c.config.ContentType,
	)

	if c.config.UserAgent != "" {
		request.Header.Set(
			"User-Agent",
			c.config.UserAgent,
		)
	}

	for name, value := range c.config.Headers {
		request.Header.Set(
			name,
			value,
		)
	}

	switch c.config.Authentication.Type {
	case AuthenticationNone:
		return nil

	case AuthenticationBearer:
		request.Header.Set(
			"Authorization",
			"Bearer "+
				c.config.Authentication.BearerToken,
		)

	case AuthenticationAPIKey:
		request.Header.Set(
			c.config.Authentication.APIKeyHeader,
			c.config.Authentication.APIKey,
		)

	case AuthenticationBasic:
		request.SetBasicAuth(
			c.config.Authentication.Username,
			c.config.Authentication.Password,
		)

	default:
		return fmt.Errorf(
			"unsupported HTTP authentication type %q",
			c.config.Authentication.Type,
		)
	}

	return nil
}

func buildTLSConfig(
	config Config,
) (*tls.Config, error) {
	if !config.TLS.Enabled {
		return nil, nil
	}

	minVersion := config.TLS.MinVersion

	if minVersion == 0 {
		minVersion = tls.VersionTLS12
	}

	tlsConfig := &tls.Config{
		MinVersion: minVersion,
		ServerName: strings.TrimSpace(
			config.TLS.ServerName,
		),
	}

	if strings.TrimSpace(config.TLS.CAFile) != "" {
		pool, err := loadCertPool(
			config.TLS.CAFile,
		)
		if err != nil {
			return nil, err
		}

		tlsConfig.RootCAs = pool
	}

	certFile := strings.TrimSpace(
		config.TLS.CertFile,
	)

	keyFile := strings.TrimSpace(
		config.TLS.KeyFile,
	)

	if certFile != "" && keyFile != "" {
		certificate, err :=
			tls.LoadX509KeyPair(
				certFile,
				keyFile,
			)
		if err != nil {
			return nil, fmt.Errorf(
				"load HTTP client TLS certificate: %w",
				err,
			)
		}

		tlsConfig.Certificates =
			[]tls.Certificate{
				certificate,
			}
	}

	return tlsConfig, nil
}

func loadCertPool(
	caFile string,
) (*x509.CertPool, error) {
	caFile = strings.TrimSpace(caFile)

	if caFile == "" {
		return nil, errors.New(
			"CA file path is empty",
		)
	}

	pemData, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf(
			"read CA file %q: %w",
			caFile,
			err,
		)
	}

	// Begin with system roots so configuring an additional CA file extends
	// trust rather than unexpectedly discarding all system trust anchors.
	pool, err := x509.SystemCertPool()
	if err != nil || pool == nil {
		pool = x509.NewCertPool()
	}

	if ok := pool.AppendCertsFromPEM(
		pemData,
	); !ok {
		return nil, fmt.Errorf(
			"CA file %q contains no usable certificates",
			caFile,
		)
	}

	return pool, nil
}

func readResponseBody(
	reader io.Reader,
	maxBytes int64,
) ([]byte, error) {
	if reader == nil {
		return nil, errors.New(
			"HTTP response body is nil",
		)
	}

	if maxBytes <= 0 {
		return nil, errors.New(
			"maximum HTTP response size must be greater than zero",
		)
	}

	limited := io.LimitReader(
		reader,
		maxBytes+1,
	)

	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}

	if int64(len(body)) > maxBytes {
		return nil, fmt.Errorf(
			"%w: response exceeds %d bytes",
			ErrResponseTooLarge,
			maxBytes,
		)
	}

	return body, nil
}
