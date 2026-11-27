package webhook

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	nethttp "net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/samorca/otgh/internal/export"
)

var (
	ErrWebhookExporterClosed = errors.New(
		"webhook exporter closed",
	)

	ErrWebhookRejected = errors.New(
		"webhook delivery rejected",
	)

	ErrWebhookPayloadTooLarge = errors.New(
		"webhook payload too large",
	)

	ErrWebhookResponseTooLarge = errors.New(
		"webhook response too large",
	)

	ErrWebhookBlockedAddress = errors.New(
		"webhook destination address blocked",
	)
)

// DeliveryIDGenerator creates unique webhook delivery identifiers.
type DeliveryIDGenerator interface {
	NewDeliveryID() string
}

// DeliveryIDGeneratorFunc adapts a function to DeliveryIDGenerator.
type DeliveryIDGeneratorFunc func() string

// NewDeliveryID implements DeliveryIDGenerator.
func (f DeliveryIDGeneratorFunc) NewDeliveryID() string {
	if f == nil {
		return ""
	}

	return f()
}

// WebhookError describes a webhook delivery failure.
type WebhookError struct {
	StatusCode int
	Status     string

	Retryable bool

	RetryAfter time.Duration

	ResponseBody string

	Cause error
}

// Error implements error.
func (e *WebhookError) Error() string {
	if e == nil {
		return "<nil>"
	}

	switch {
	case e.StatusCode > 0 &&
		e.ResponseBody != "":
		return fmt.Sprintf(
			"webhook delivery failed with HTTP status %d (%s): %s",
			e.StatusCode,
			e.Status,
			e.ResponseBody,
		)

	case e.StatusCode > 0:
		return fmt.Sprintf(
			"webhook delivery failed with HTTP status %d (%s)",
			e.StatusCode,
			e.Status,
		)

	case e.Cause != nil:
		return fmt.Sprintf(
			"webhook delivery failed: %v",
			e.Cause,
		)

	default:
		return "webhook delivery failed"
	}
}

// Unwrap exposes the local cause.
func (e *WebhookError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.Cause
}

// DeliveryResponse is an optional response body returned by an OTGH-aware
// webhook destination.
//
// An empty successful response means the complete batch was accepted.
type DeliveryResponse struct {
	AcceptedRecords int `json:"accepted_records"`
	RejectedRecords int `json:"rejected_records"`

	Retryable bool `json:"retryable"`

	Message string `json:"message,omitempty"`

	ErrorCode string `json:"error_code,omitempty"`
}

// Exporter delivers event-oriented OTGH batches to one webhook endpoint.
//
// Exporter is safe for concurrent use. It signs each serialized payload
// independently and reuses one HTTP transport for connection pooling.
type Exporter struct {
	config Config

	client    *nethttp.Client
	transport *nethttp.Transport

	idGenerator DeliveryIDGenerator

	closeOnce sync.Once
	closed    atomic.Bool
}

// NewExporter creates a webhook exporter.
func NewExporter(
	config Config,
	idGenerator DeliveryIDGenerator,
) (*Exporter, error) {
	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create webhook exporter: %w",
			err,
		)
	}

	if idGenerator == nil {
		idGenerator =
			NewTimestampDeliveryIDGenerator(
				"webhook",
			)
	}

	tlsConfig, err := buildTLSConfig(
		config,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create webhook TLS configuration: %w",
			err,
		)
	}

	dialer := &net.Dialer{
		Timeout:   config.Timeouts.Connect,
		KeepAlive: 30 * time.Second,
	}

	transport := &nethttp.Transport{
		Proxy: nethttp.ProxyFromEnvironment,

		DialContext: secureDialContext(
			dialer,
			config.Network.
				AllowPrivateNetworks,
		),

		ForceAttemptHTTP2: true,

		MaxIdleConns: config.
			MaxIdleConnectionsPerHost,

		MaxIdleConnsPerHost: config.
			MaxIdleConnectionsPerHost,

		MaxConnsPerHost: config.
			MaxIdleConnectionsPerHost * 4,

		IdleConnTimeout: config.Timeouts.
			IdleConnection,

		TLSHandshakeTimeout: config.Timeouts.
			TLSHandshake,

		ResponseHeaderTimeout: config.Timeouts.
			ResponseHeader,

		ExpectContinueTimeout: time.Second,

		TLSClientConfig: tlsConfig,
	}

	client := &nethttp.Client{
		Transport: transport,
		Timeout:   config.Timeouts.Request,
	}

	if !config.FollowRedirects {
		client.CheckRedirect =
			func(
				request *nethttp.Request,
				via []*nethttp.Request,
			) error {
				return nethttp.ErrUseLastResponse
			}
	}

	return &Exporter{
		config:      config,
		client:      client,
		transport:   transport,
		idGenerator: idGenerator,
	}, nil
}

// ID implements export.Exporter.
func (e *Exporter) ID() export.DestinationID {
	if e == nil {
		return ""
	}

	return export.DestinationID(
		e.config.ID,
	)
}

// Type implements export.Exporter.
func (e *Exporter) Type() string {
	return "webhook"
}

// Export serializes, signs, and delivers one webhook batch.
func (e *Exporter) Export(
	ctx context.Context,
	batch export.Batch,
) (export.DeliveryResult, error) {
	startedAt := time.Now().UTC()

	result := export.DeliveryResult{
		BatchID:       batch.ID,
		DestinationID: batch.Destination.ID,
		Status:        export.DeliveryInFlight,
		Attempt:       1,
		StartedAt:     startedAt,
	}

	if e == nil {
		return webhookFailedResult(
			result,
			"webhook_exporter_nil",
			"webhook exporter is nil",
			false,
			errors.New("webhook exporter is nil"),
		)
	}

	if ctx == nil {
		return webhookFailedResult(
			result,
			"context_nil",
			"webhook export context is nil",
			false,
			errors.New(
				"webhook export context is nil",
			),
		)
	}

	if e.closed.Load() {
		return webhookFailedResult(
			result,
			"webhook_exporter_closed",
			ErrWebhookExporterClosed.Error(),
			false,
			ErrWebhookExporterClosed,
		)
	}

	if err := ctx.Err(); err != nil {
		return webhookFailedResult(
			result,
			"context_cancelled",
			err.Error(),
			false,
			err,
		)
	}

	if err := batch.Validate(); err != nil {
		return webhookFailedResult(
			result,
			"invalid_batch",
			err.Error(),
			false,
			fmt.Errorf(
				"webhook batch validation: %w",
				err,
			),
		)
	}

	if batch.Destination.ID != e.ID() {
		err := fmt.Errorf(
			"batch destination %q does not match webhook exporter %q",
			batch.Destination.ID,
			e.ID(),
		)

		return webhookFailedResult(
			result,
			"destination_mismatch",
			err.Error(),
			false,
			err,
		)
	}

	payload, err := json.Marshal(batch)
	if err != nil {
		return webhookFailedResult(
			result,
			"batch_encoding_failed",
			err.Error(),
			false,
			fmt.Errorf(
				"encode webhook batch %q: %w",
				batch.ID,
				err,
			),
		)
	}

	if int64(len(payload)) >
		e.config.MaxPayloadBytes {
		err := fmt.Errorf(
			"%w: encoded batch is %d bytes, maximum is %d",
			ErrWebhookPayloadTooLarge,
			len(payload),
			e.config.MaxPayloadBytes,
		)

		return webhookFailedResult(
			result,
			"webhook_payload_too_large",
			err.Error(),
			false,
			err,
		)
	}

	deliveryID := strings.TrimSpace(
		e.idGenerator.NewDeliveryID(),
	)

	if deliveryID == "" {
		err := errors.New(
			"webhook delivery id generator returned an empty id",
		)

		return webhookFailedResult(
			result,
			"delivery_id_failed",
			err.Error(),
			false,
			err,
		)
	}

	timestamp := strconv.FormatInt(
		time.Now().UTC().Unix(),
		10,
	)

	request, err :=
		nethttp.NewRequestWithContext(
			ctx,
			nethttp.MethodPost,
			e.config.URL,
			bytes.NewReader(payload),
		)
	if err != nil {
		return webhookFailedResult(
			result,
			"request_creation_failed",
			err.Error(),
			false,
			err,
		)
	}

	request.ContentLength = int64(
		len(payload),
	)

	if err := e.applyHeaders(
		request,
		deliveryID,
		timestamp,
		payload,
	); err != nil {
		return webhookFailedResult(
			result,
			"request_headers_failed",
			err.Error(),
			false,
			err,
		)
	}

	response, err := e.client.Do(request)

	result.FinishedAt = time.Now().UTC()

	if err != nil {
		retryable := retryableTransportError(
			err,
		)

		if retryable {
			result.Status =
				export.DeliveryRetrying
		} else {
			result.Status =
				export.DeliveryFailed
		}

		result.Retryable = retryable

		result.ErrorCode =
			classifyTransportErrorCode(
				err,
			)

		result.ErrorMessage = err.Error()

		return result, &WebhookError{
			Retryable: retryable,
			Cause:     err,
		}
	}

	defer response.Body.Close()

	body, err := readResponseBody(
		response.Body,
		e.config.MaxResponseBytes,
	)
	if err != nil {
		result.Status =
			export.DeliveryFailed

		result.ErrorCode =
			"webhook_response_too_large"

		result.ErrorMessage =
			err.Error()

		return result, err
	}

	result.Metadata = map[string]string{
		"webhook.delivery_id":
			deliveryID,

		"webhook.status_code":
			strconv.Itoa(
				response.StatusCode,
			),
	}

	if !successfulStatus(
		response.StatusCode,
	) {
		webhookErr :=
			classifyHTTPError(
				response,
				body,
			)

		if webhookErr.Retryable {
			result.Status =
				export.DeliveryRetrying
		} else {
			result.Status =
				export.DeliveryFailed
		}

		result.RejectedRecords =
			batch.RecordCount()

		result.Retryable =
			webhookErr.Retryable

		result.ErrorCode =
			"http_status_" +
				strconv.Itoa(
					response.StatusCode,
				)

		result.ErrorMessage =
			webhookErr.Error()

		if webhookErr.RetryAfter > 0 {
			result.Metadata["webhook.retry_after"] =
				webhookErr.
					RetryAfter.
					String()
		}

		return result, errors.Join(
			ErrWebhookRejected,
			webhookErr,
		)
	}

	deliveryResponse, err :=
		parseDeliveryResponse(
			body,
			batch.RecordCount(),
		)
	if err != nil {
		result.Status =
			export.DeliveryFailed

		result.ErrorCode =
			"invalid_webhook_response"

		result.ErrorMessage =
			err.Error()

		return result, err
	}

	result.AcceptedRecords =
		deliveryResponse.
			AcceptedRecords

	result.RejectedRecords =
		deliveryResponse.
			RejectedRecords

	result.Retryable =
		deliveryResponse.Retryable

	result.ErrorCode =
		strings.TrimSpace(
			deliveryResponse.
				ErrorCode,
		)

	result.ErrorMessage =
		strings.TrimSpace(
			deliveryResponse.Message,
		)

	switch {
	case result.AcceptedRecords ==
		batch.RecordCount() &&
		result.RejectedRecords == 0:
		result.Status =
			export.DeliveryAcknowledged

		result.Retryable = false
		result.ErrorCode = ""
		result.ErrorMessage = ""

		return result, nil

	case result.Retryable:
		result.Status =
			export.DeliveryRetrying

		return result, &WebhookError{
			StatusCode: response.StatusCode,
			Status:     response.Status,
			Retryable:  true,
			ResponseBody: result.
				ErrorMessage,
			Cause: ErrWebhookRejected,
		}

	default:
		result.Status =
			export.DeliveryAcknowledged

		return result, nil
	}
}

// Health performs a HEAD request to the configured callback URL.
//
// This is optional because not every webhook receiver supports HEAD.
func (e *Exporter) Health(
	ctx context.Context,
) error {
	if e == nil {
		return errors.New(
			"webhook exporter is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"webhook health context is nil",
		)
	}

	if e.closed.Load() {
		return ErrWebhookExporterClosed
	}

	request, err :=
		nethttp.NewRequestWithContext(
			ctx,
			nethttp.MethodHead,
			e.config.URL,
			nil,
		)
	if err != nil {
		return err
	}

	if err := e.applyAuthentication(
		request,
	); err != nil {
		return err
	}

	response, err := e.client.Do(request)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}

		return err
	}
	defer response.Body.Close()

	_, _ = io.Copy(
		io.Discard,
		io.LimitReader(
			response.Body,
			4096,
		),
	)

	if response.StatusCode >= 200 &&
		response.StatusCode <= 399 {
		return nil
	}

	return fmt.Errorf(
		"webhook health endpoint returned status %d",
		response.StatusCode,
	)
}

// Close implements export.Exporter.
func (e *Exporter) Close() error {
	if e == nil {
		return nil
	}

	e.closeOnce.Do(func() {
		e.closed.Store(true)

		if e.transport != nil {
			e.transport.CloseIdleConnections()
		}
	})

	return nil
}

// Closed reports whether Close has been called.
func (e *Exporter) Closed() bool {
	if e == nil {
		return true
	}

	return e.closed.Load()
}

// ClassifyRetry implements export.RetryClassifier.
func (e *Exporter) ClassifyRetry(
	err error,
) export.RetryDecision {
	if err == nil {
		return export.RetryDecision{
			Retry:  false,
			Reason: "operation succeeded",
		}
	}

	if errors.Is(err, context.Canceled) ||
		errors.Is(
			err,
			context.DeadlineExceeded,
		) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "request context ended",
		}
	}

	if errors.Is(
		err,
		ErrWebhookExporterClosed,
	) || errors.Is(
		err,
		ErrWebhookPayloadTooLarge,
	) || errors.Is(
		err,
		ErrWebhookResponseTooLarge,
	) || errors.Is(
		err,
		ErrWebhookBlockedAddress,
	) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "permanent local webhook failure",
		}
	}

	var webhookErr *WebhookError

	if errors.As(err, &webhookErr) {
		return export.RetryDecision{
			Retry: webhookErr.Retryable,

			Reason: fmt.Sprintf(
				"webhook HTTP status %d retryable=%t",
				webhookErr.StatusCode,
				webhookErr.Retryable,
			),
		}
	}

	return export.RetryDecision{
		Retry: retryableTransportError(
			err,
		),

		Reason: "webhook transport classification",
	}
}

func (e *Exporter) applyHeaders(
	request *nethttp.Request,
	deliveryID string,
	timestamp string,
	payload []byte,
) error {
	if request == nil {
		return errors.New(
			"webhook request is nil",
		)
	}

	request.Header.Set(
		"Content-Type",
		e.config.ContentType,
	)

	request.Header.Set(
		"User-Agent",
		e.config.UserAgent,
	)

	for key, value := range e.config.Headers {
		request.Header.Set(key, value)
	}

	if err := e.applyAuthentication(
		request,
	); err != nil {
		return err
	}

	if !e.config.Signing.Enabled {
		return nil
	}

	signature, err := signPayload(
		e.config.Signing,
		deliveryID,
		timestamp,
		payload,
	)
	if err != nil {
		return err
	}

	request.Header.Set(
		e.config.Signing.DeliveryIDHeader,
		deliveryID,
	)

	request.Header.Set(
		e.config.Signing.TimestampHeader,
		timestamp,
	)

	request.Header.Set(
		e.config.Signing.SignatureHeader,
		signature,
	)

	request.Header.Set(
		e.config.Signing.KeyIDHeader,
		e.config.Signing.KeyID,
	)

	return nil
}

func (e *Exporter) applyAuthentication(
	request *nethttp.Request,
) error {
	switch e.config.Authentication.Type {
	case AuthenticationNone:
		return nil

	case AuthenticationBearer:
		request.Header.Set(
			"Authorization",
			"Bearer "+
				e.config.Authentication.
					BearerToken,
		)

	case AuthenticationBasic:
		request.SetBasicAuth(
			e.config.Authentication.Username,
			e.config.Authentication.Password,
		)

	default:
		return fmt.Errorf(
			"unsupported webhook authentication type %q",
			e.config.Authentication.Type,
		)
	}

	return nil
}

// signPayload returns versioned hexadecimal HMAC-SHA256.
//
// Signed bytes:
//
//	delivery-id.timestamp.raw-body
func signPayload(
	config SigningConfig,
	deliveryID string,
	timestamp string,
	payload []byte,
) (string, error) {
	if !config.Enabled {
		return "", errors.New(
			"webhook signing is disabled",
		)
	}

	if config.Algorithm !=
		SignatureHMACSHA256 {
		return "", fmt.Errorf(
			"unsupported webhook signature algorithm %q",
			config.Algorithm,
		)
	}

	deliveryID = strings.TrimSpace(
		deliveryID,
	)

	timestamp = strings.TrimSpace(
		timestamp,
	)

	if deliveryID == "" {
		return "", errors.New(
			"webhook delivery id is required",
		)
	}

	if timestamp == "" {
		return "", errors.New(
			"webhook timestamp is required",
		)
	}

	if len(payload) == 0 {
		return "", errors.New(
			"webhook payload is empty",
		)
	}

	mac := hmac.New(
		sha256.New,
		[]byte(config.Secret),
	)

	_, _ = mac.Write(
		[]byte(deliveryID),
	)

	_, _ = mac.Write(
		[]byte("."),
	)

	_, _ = mac.Write(
		[]byte(timestamp),
	)

	_, _ = mac.Write(
		[]byte("."),
	)

	_, _ = mac.Write(payload)

	return "v1=" +
		hex.EncodeToString(
			mac.Sum(nil),
		), nil
}

func secureDialContext(
	dialer *net.Dialer,
	allowPrivate bool,
) func(
	context.Context,
	string,
	string,
) (net.Conn, error) {
	return func(
		ctx context.Context,
		network string,
		address string,
	) (net.Conn, error) {
		if dialer == nil {
			return nil, errors.New(
				"webhook network dialer is nil",
			)
		}

		if allowPrivate {
			return dialer.DialContext(
				ctx,
				network,
				address,
			)
		}

		host, port, err :=
			net.SplitHostPort(address)
		if err != nil {
			return nil, fmt.Errorf(
				"parse webhook destination address %q: %w",
				address,
				err,
			)
		}

		resolver := net.DefaultResolver

		addresses, err :=
			resolver.LookupIPAddr(
				ctx,
				host,
			)
		if err != nil {
			return nil, fmt.Errorf(
				"resolve webhook destination %q: %w",
				host,
				err,
			)
		}

		if len(addresses) == 0 {
			return nil, fmt.Errorf(
				"webhook destination %q resolved to no addresses",
				host,
			)
		}

		var lastErr error

		for _, candidate := range addresses {
			if nonPublicIP(candidate.IP) {
				lastErr = fmt.Errorf(
					"%w: %s resolved to %s",
					ErrWebhookBlockedAddress,
					host,
					candidate.IP,
				)

				continue
			}

			target := net.JoinHostPort(
				candidate.IP.String(),
				port,
			)

			connection, err :=
				dialer.DialContext(
					ctx,
					network,
					target,
				)
			if err == nil {
				return connection, nil
			}

			lastErr = err
		}

		if lastErr == nil {
			lastErr =
				ErrWebhookBlockedAddress
		}

		return nil, lastErr
	}
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

	if strings.TrimSpace(
		config.TLS.CAFile,
	) != "" {
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
				"load webhook client certificate: %w",
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
	pemData, err := os.ReadFile(
		strings.TrimSpace(caFile),
	)
	if err != nil {
		return nil, fmt.Errorf(
			"read webhook CA file: %w",
			err,
		)
	}

	pool, err := x509.SystemCertPool()
	if err != nil || pool == nil {
		pool = x509.NewCertPool()
	}

	if ok := pool.AppendCertsFromPEM(
		pemData,
	); !ok {
		return nil, errors.New(
			"webhook CA file contains no usable certificates",
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
			"webhook response body is nil",
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
			"%w: maximum is %d bytes",
			ErrWebhookResponseTooLarge,
			maxBytes,
		)
	}

	return body, nil
}

func parseDeliveryResponse(
	body []byte,
	totalRecords int,
) (DeliveryResponse, error) {
	if len(
		strings.TrimSpace(
			string(body),
		),
	) == 0 {
		return DeliveryResponse{
			AcceptedRecords: totalRecords,
		}, nil
	}

	var response DeliveryResponse

	decoder := json.NewDecoder(
		bytes.NewReader(body),
	)

	decoder.DisallowUnknownFields()

	if err := decoder.Decode(
		&response,
	); err != nil {
		return DeliveryResponse{}, fmt.Errorf(
			"decode webhook response: %w",
			err,
		)
	}

	if response.AcceptedRecords < 0 ||
		response.RejectedRecords < 0 {
		return DeliveryResponse{}, errors.New(
			"webhook response record counts cannot be negative",
		)
	}

	accounted :=
		response.AcceptedRecords +
			response.RejectedRecords

	if accounted > totalRecords {
		return DeliveryResponse{}, fmt.Errorf(
			"webhook response accounts for %d records but batch contains %d",
			accounted,
			totalRecords,
		)
	}

	if accounted == 0 {
		response.AcceptedRecords =
			totalRecords
	}

	return response, nil
}

func classifyHTTPError(
	response *nethttp.Response,
	body []byte,
) *WebhookError {
	retryAfter := parseRetryAfter(
		response.Header.Get(
			"Retry-After",
		),
		time.Now().UTC(),
	)

	return &WebhookError{
		StatusCode: response.StatusCode,
		Status:     response.Status,

		Retryable: retryableStatusCode(
			response.StatusCode,
		),

		RetryAfter: retryAfter,

		ResponseBody: strings.TrimSpace(
			string(body),
		),

		Cause: ErrWebhookRejected,
	}
}

func retryableStatusCode(
	statusCode int,
) bool {
	switch statusCode {
	case nethttp.StatusRequestTimeout,
		nethttp.StatusTooEarly,
		nethttp.StatusTooManyRequests:
		return true
	}

	return statusCode >= 500 &&
		statusCode <= 599 &&
		statusCode !=
			nethttp.StatusNotImplemented
}

func successfulStatus(
	statusCode int,
) bool {
	return statusCode >= 200 &&
		statusCode <= 299
}

func retryableTransportError(
	err error,
) bool {
	if err == nil {
		return false
	}

	if errors.Is(err, context.Canceled) ||
		errors.Is(
			err,
			context.DeadlineExceeded,
		) ||
		errors.Is(
			err,
			ErrWebhookBlockedAddress,
		) {
		return false
	}

	var networkError net.Error

	return errors.As(
		err,
		&networkError,
	)
}

func classifyTransportErrorCode(
	err error,
) string {
	switch {
	case errors.Is(
		err,
		context.Canceled,
	):
		return "context_cancelled"

	case errors.Is(
		err,
		context.DeadlineExceeded,
	):
		return "deadline_exceeded"

	case errors.Is(
		err,
		ErrWebhookBlockedAddress,
	):
		return "blocked_destination"

	default:
		return "webhook_transport_error"
	}
}

func parseRetryAfter(
	value string,
	now time.Time,
) time.Duration {
	value = strings.TrimSpace(value)

	if value == "" {
		return 0
	}

	if seconds, err :=
		strconv.ParseUint(
			value,
			10,
			32,
		); err == nil {
		return time.Duration(seconds) *
			time.Second
	}

	target, err := nethttp.ParseTime(value)
	if err != nil {
		return 0
	}

	delay := target.Sub(now)

	if delay < 0 {
		return 0
	}

	return delay
}

// NewTimestampDeliveryIDGenerator returns a process-local delivery ID
// generator.
func NewTimestampDeliveryIDGenerator(
	prefix string,
) DeliveryIDGenerator {
	prefix = strings.TrimSpace(prefix)

	if prefix == "" {
		prefix = "webhook"
	}

	return &timestampDeliveryIDGenerator{
		prefix: prefix,
	}
}

type timestampDeliveryIDGenerator struct {
	prefix string

	counter atomic.Uint64
}

func (g *timestampDeliveryIDGenerator) NewDeliveryID() string {
	if g == nil {
		return ""
	}

	return fmt.Sprintf(
		"%s-%d-%d",
		g.prefix,
		time.Now().UTC().UnixNano(),
		g.counter.Add(1),
	)
}

func webhookFailedResult(
	result export.DeliveryResult,
	code string,
	message string,
	retryable bool,
	err error,
) (
	export.DeliveryResult,
	error,
) {
	result.FinishedAt = time.Now().UTC()
	result.Retryable = retryable
	result.ErrorCode = code
	result.ErrorMessage = message

	if retryable {
		result.Status =
			export.DeliveryRetrying
	} else {
		result.Status =
			export.DeliveryFailed
	}

	return result, err
}

// Compile-time interface assertions.
var (
	_ export.Exporter =
		(*Exporter)(nil)

	_ export.RetryClassifier =
		(*Exporter)(nil)
)
