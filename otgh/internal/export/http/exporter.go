package http

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	nethttp "net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/samorca/otgh/internal/export"
)

var (
	// ErrHTTPExporterClosed indicates that the exporter has been closed.
	ErrHTTPExporterClosed = errors.New(
		"HTTP exporter closed",
	)

	// ErrHTTPDeliveryRejected indicates that the remote endpoint rejected the
	// delivery with a non-success status.
	ErrHTTPDeliveryRejected = errors.New(
		"HTTP delivery rejected",
	)
)

// HTTPError describes an HTTP export failure.
//
// Retryable is a transport-specific classification consumed by OTGH's shared
// RetryExecutor. The exporter itself does not perform retries.
type HTTPError struct {
	StatusCode int
	Status     string

	Retryable bool

	RetryAfter time.Duration

	ResponseBody string

	Cause error
}

// Error implements error.
func (e *HTTPError) Error() string {
	if e == nil {
		return "<nil>"
	}

	switch {
	case e.StatusCode > 0 &&
		e.ResponseBody != "":
		return fmt.Sprintf(
			"HTTP export failed with status %d (%s): %s",
			e.StatusCode,
			e.Status,
			e.ResponseBody,
		)

	case e.StatusCode > 0:
		return fmt.Sprintf(
			"HTTP export failed with status %d (%s)",
			e.StatusCode,
			e.Status,
		)

	case e.Cause != nil:
		return fmt.Sprintf(
			"HTTP export failed: %v",
			e.Cause,
		)

	default:
		return "HTTP export failed"
	}
}

// Unwrap exposes the underlying local error.
func (e *HTTPError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.Cause
}

// ExportResponse is an optional structured response that an OTGH-aware HTTP
// destination may return.
//
// Empty response bodies are also valid for successful 2xx responses. In that
// case, the exporter treats the complete batch as accepted.
//
// A destination can return this structure when it needs to describe partial
// acceptance or give a machine-readable message.
type ExportResponse struct {
	AcceptedRecords int `json:"accepted_records"`
	RejectedRecords int `json:"rejected_records"`

	Retryable bool `json:"retryable"`

	Message string `json:"message,omitempty"`

	ErrorCode string `json:"error_code,omitempty"`
}

// Exporter delivers OTGH batches to one HTTP endpoint.
//
// Exporter is safe for concurrent use. It delegates requests to a reusable
// Client and does not implement retries itself.
type Exporter struct {
	config Config
	client *Client

	closeOnce sync.Once
	closed    atomic.Bool
}

// NewExporter creates an HTTP exporter with an internally managed client.
func NewExporter(
	config Config,
) (*Exporter, error) {
	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create HTTP exporter: %w",
			err,
		)
	}

	client, err := NewClient(config)
	if err != nil {
		return nil, err
	}

	return &Exporter{
		config: config,
		client: client,
	}, nil
}

// NewExporterWithClient creates an exporter using an existing Client.
//
// This constructor is useful for tests and controlled dependency injection.
// The supplied client must have the same normalized destination identity and
// endpoint as config.
func NewExporterWithClient(
	config Config,
	client *Client,
) (*Exporter, error) {
	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create HTTP exporter: %w",
			err,
		)
	}

	if client == nil {
		return nil, errors.New(
			"create HTTP exporter: client is required",
		)
	}

	clientConfig := client.Config()

	if clientConfig.ID != config.ID {
		return nil, fmt.Errorf(
			"HTTP client id %q does not match exporter id %q",
			clientConfig.ID,
			config.ID,
		)
	}

	if clientConfig.Endpoint != config.Endpoint {
		return nil, errors.New(
			"HTTP client endpoint does not match exporter endpoint",
		)
	}

	return &Exporter{
		config: config,
		client: client,
	}, nil
}

// ID implements export.Exporter.
func (e *Exporter) ID() export.DestinationID {
	if e == nil {
		return ""
	}

	return export.DestinationID(e.config.ID)
}

// Type implements export.Exporter.
func (e *Exporter) Type() string {
	return "http"
}

// Export serializes and delivers a batch.
//
// A successful response must use an HTTP 2xx status code.
//
// If the response body is empty, the complete batch is considered accepted.
// If a response body is present, it may contain ExportResponse JSON for
// record-level counts.
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
		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "http_exporter_nil"
		result.ErrorMessage = "HTTP exporter is nil"

		return result, errors.New(
			result.ErrorMessage,
		)
	}

	if ctx == nil {
		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "context_nil"
		result.ErrorMessage = "HTTP export context is nil"

		return result, errors.New(
			result.ErrorMessage,
		)
	}

	if e.closed.Load() {
		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "http_exporter_closed"
		result.ErrorMessage = ErrHTTPExporterClosed.Error()

		return result, ErrHTTPExporterClosed
	}

	if err := ctx.Err(); err != nil {
		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "context_cancelled"
		result.ErrorMessage = err.Error()

		return result, err
	}

	if err := batch.Validate(); err != nil {
		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "invalid_batch"
		result.ErrorMessage = err.Error()

		return result, fmt.Errorf(
			"HTTP export batch validation: %w",
			err,
		)
	}

	if batch.Destination.ID != e.ID() {
		err := fmt.Errorf(
			"batch destination %q does not match HTTP exporter %q",
			batch.Destination.ID,
			e.ID(),
		)

		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "destination_mismatch"
		result.ErrorMessage = err.Error()

		return result, err
	}

	payload, err := json.Marshal(batch)
	if err != nil {
		result.Status = export.DeliveryFailed
		result.FinishedAt = time.Now().UTC()
		result.ErrorCode = "batch_encoding_failed"
		result.ErrorMessage = err.Error()

		return result, fmt.Errorf(
			"encode HTTP export batch %q: %w",
			batch.ID,
			err,
		)
	}

	response, err := e.client.Do(
		ctx,
		payload,
	)

	result.FinishedAt = time.Now().UTC()

	if err != nil {
		retryable := retryableTransportError(
			err,
		)

		if retryable {
			result.Status = export.DeliveryRetrying
		} else {
			result.Status = export.DeliveryFailed
		}

		result.Retryable = retryable
		result.ErrorCode = classifyTransportErrorCode(
			err,
		)
		result.ErrorMessage = err.Error()

		return result, &HTTPError{
			Retryable: retryable,
			Cause:     err,
		}
	}

	if !response.Success() {
		httpErr := classifyHTTPResponseError(
			response,
		)

		if httpErr.Retryable {
			result.Status = export.DeliveryRetrying
		} else {
			result.Status = export.DeliveryFailed
		}

		result.RejectedRecords = batch.RecordCount()
		result.Retryable = httpErr.Retryable
		result.ErrorCode = httpStatusErrorCode(
			response.StatusCode,
		)
		result.ErrorMessage = httpErr.Error()

		result.Metadata = responseMetadata(
			response,
			httpErr.RetryAfter,
		)

		return result, errors.Join(
			ErrHTTPDeliveryRejected,
			httpErr,
		)
	}

	accepted, rejected, retryable, responseMessage, responseCode, parseErr :=
		parseSuccessfulResponse(
			response.Body,
			batch.RecordCount(),
		)

	if parseErr != nil {
		result.Status = export.DeliveryFailed
		result.Retryable = false
		result.ErrorCode = "invalid_success_response"
		result.ErrorMessage = parseErr.Error()

		return result, parseErr
	}

	result.AcceptedRecords = accepted
	result.RejectedRecords = rejected
	result.Retryable = retryable
	result.ErrorCode = responseCode
	result.ErrorMessage = responseMessage
	result.Metadata = responseMetadata(
		response,
		0,
	)

	switch {
	case accepted == batch.RecordCount() &&
		rejected == 0:
		result.Status = export.DeliveryAcknowledged
		result.Retryable = false
		result.ErrorCode = ""
		result.ErrorMessage = ""

		return result, nil

	case retryable:
		result.Status = export.DeliveryRetrying

		return result, &HTTPError{
			StatusCode: response.StatusCode,
			Status:     response.Status,
			Retryable:  true,
			ResponseBody: responseMessage,
			Cause:       export.ErrIncompleteDelivery,
		}

	default:
		result.Status = export.DeliveryAcknowledged

		// Returning nil here permits the acknowledgement manager to record
		// a partial receiver result. Worker-level completeness policy then
		// decides whether the durable spool entry can be removed.
		return result, nil
	}
}

// Health implements export.Exporter.
func (e *Exporter) Health(
	ctx context.Context,
) error {
	if e == nil {
		return errors.New(
			"HTTP exporter is nil",
		)
	}

	if e.closed.Load() {
		return ErrHTTPExporterClosed
	}

	return e.client.Health(ctx)
}

// Close implements export.Exporter.
func (e *Exporter) Close() error {
	if e == nil {
		return nil
	}

	var closeErr error

	e.closeOnce.Do(func() {
		e.closed.Store(true)

		closeErr = e.client.Close()
	})

	return closeErr
}

// Closed reports whether the exporter has been closed.
func (e *Exporter) Closed() bool {
	if e == nil {
		return true
	}

	return e.closed.Load()
}

// ClassifyRetry implements export.RetryClassifier.
//
// This allows the shared RetryExecutor to use HTTP-specific failure knowledge
// without teaching retry.go about HTTP status codes or network errors.
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
		errors.Is(err, context.DeadlineExceeded) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "request context ended",
		}
	}

	if errors.Is(err, ErrHTTPExporterClosed) ||
		errors.Is(err, ErrRequestTooLarge) ||
		errors.Is(err, ErrResponseTooLarge) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "permanent local HTTP exporter failure",
		}
	}

	var httpErr *HTTPError

	if errors.As(err, &httpErr) {
		return export.RetryDecision{
			Retry:  httpErr.Retryable,
			Reason: httpRetryReason(
				httpErr,
			),
		}
	}

	return export.RetryDecision{
		Retry: retryableTransportError(
			err,
		),
		Reason: "HTTP transport classification",
	}
}

func parseSuccessfulResponse(
	body []byte,
	totalRecords int,
) (
	accepted int,
	rejected int,
	retryable bool,
	message string,
	errorCode string,
	err error,
) {
	if len(body) == 0 {
		return totalRecords, 0, false, "", "", nil
	}

	if len(strings.TrimSpace(string(body))) == 0 {
		return totalRecords, 0, false, "", "", nil
	}

	var response ExportResponse

	decoder := json.NewDecoder(
		strings.NewReader(
			string(body),
		),
	)

	decoder.DisallowUnknownFields()

	if decodeErr := decoder.Decode(
		&response,
	); decodeErr != nil {
		return 0, 0, false, "", "",
			fmt.Errorf(
				"decode HTTP success response: %w",
				decodeErr,
			)
	}

	if response.AcceptedRecords < 0 ||
		response.RejectedRecords < 0 {
		return 0, 0, false, "", "",
			errors.New(
				"HTTP response record counts cannot be negative",
			)
	}

	accounted := response.AcceptedRecords +
		response.RejectedRecords

	if accounted > totalRecords {
		return 0, 0, false, "", "",
			fmt.Errorf(
				"HTTP response accounts for %d records but batch contains %d",
				accounted,
				totalRecords,
			)
	}

	if accounted == 0 {
		// A structured success response with no explicit counts is treated
		// like an empty successful response.
		response.AcceptedRecords =
			totalRecords
	}

	return response.AcceptedRecords,
		response.RejectedRecords,
		response.Retryable,
		strings.TrimSpace(response.Message),
		strings.TrimSpace(response.ErrorCode),
		nil
}

func classifyHTTPResponseError(
	response Response,
) *HTTPError {
	retryAfter := parseRetryAfter(
		response.Header.Get("Retry-After"),
		response.ReceivedAt,
	)

	retryable := retryableStatusCode(
		response.StatusCode,
	)

	body := strings.TrimSpace(
		string(response.Body),
	)

	return &HTTPError{
		StatusCode: response.StatusCode,
		Status:     response.Status,

		Retryable: retryable,

		RetryAfter: retryAfter,

		ResponseBody: body,

		Cause: ErrHTTPDeliveryRejected,
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
		statusCode != nethttp.StatusNotImplemented
}

func retryableTransportError(
	err error,
) bool {
	if err == nil {
		return false
	}

	if errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, ErrClientClosed) ||
		errors.Is(err, ErrRequestTooLarge) ||
		errors.Is(err, ErrResponseTooLarge) {
		return false
	}

	var netErr net.Error

	if errors.As(err, &netErr) {
		return true
	}

	var urlErr *url.Error

	if errors.As(err, &urlErr) {
		return retryableTransportError(
			urlErr.Err,
		)
	}

	return false
}

func classifyTransportErrorCode(
	err error,
) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "context_cancelled"

	case errors.Is(err, context.DeadlineExceeded):
		return "deadline_exceeded"

	case errors.Is(err, ErrClientClosed):
		return "http_client_closed"

	case errors.Is(err, ErrRequestTooLarge):
		return "request_too_large"

	case errors.Is(err, ErrResponseTooLarge):
		return "response_too_large"

	default:
		return "http_transport_error"
	}
}

func httpStatusErrorCode(
	statusCode int,
) string {
	return "http_status_" +
		strconv.Itoa(statusCode)
}

func httpRetryReason(
	httpErr *HTTPError,
) string {
	if httpErr == nil {
		return "HTTP error unavailable"
	}

	if httpErr.StatusCode > 0 {
		return fmt.Sprintf(
			"HTTP status %d retryable=%t",
			httpErr.StatusCode,
			httpErr.Retryable,
		)
	}

	return fmt.Sprintf(
		"HTTP transport retryable=%t",
		httpErr.Retryable,
	)
}

func responseMetadata(
	response Response,
	retryAfter time.Duration,
) map[string]string {
	metadata := map[string]string{
		"http.status_code": strconv.Itoa(
			response.StatusCode,
		),
	}

	if requestID := firstNonEmptyHeader(
		response.Header,
		"X-Request-ID",
		"X-Correlation-ID",
		"Traceparent",
	); requestID != "" {
		metadata["http.request_id"] =
			requestID
	}

	if retryAfter > 0 {
		metadata["http.retry_after"] =
			retryAfter.String()
	}

	return metadata
}

func firstNonEmptyHeader(
	header nethttp.Header,
	names ...string,
) string {
	for _, name := range names {
		value := strings.TrimSpace(
			header.Get(name),
		)

		if value != "" {
			return value
		}
	}

	return ""
}

func parseRetryAfter(
	value string,
	now time.Time,
) time.Duration {
	value = strings.TrimSpace(value)

	if value == "" {
		return 0
	}

	if seconds, err := strconv.ParseUint(
		value,
		10,
		32,
	); err == nil {
		return time.Duration(seconds) *
			time.Second
	}

	parsed, err := nethttp.ParseTime(value)
	if err != nil {
		return 0
	}

	if now.IsZero() {
		now = time.Now().UTC()
	}

	delay := parsed.Sub(now)

	if delay < 0 {
		return 0
	}

	return delay
}

// Compile-time interface assertions.
var (
	_ export.Exporter       = (*Exporter)(nil)
	_ export.RetryClassifier = (*Exporter)(nil)
)
