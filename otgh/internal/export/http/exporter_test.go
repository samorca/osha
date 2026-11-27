package http

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	nethttp "net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samorca/otgh/internal/export"
)

func testHTTPConfig(
	endpoint string,
) Config {
	config := DefaultConfig()

	config.ID = "test-http"
	config.Endpoint = endpoint

	// httptest.NewServer uses plain HTTP.
	config.TLS.Enabled = false
	config.TLS.MinVersion = 0

	config.Timeouts.Request = 2 * time.Second
	config.Timeouts.Connect = time.Second
	config.Timeouts.TLSHandshake = time.Second
	config.Timeouts.ResponseHeader = time.Second
	config.Timeouts.ExpectContinue = time.Second
	config.Timeouts.IdleConnection = 5 * time.Second

	config.MaxRequestBytes = 1024 * 1024
	config.MaxResponseBytes = 1024 * 1024

	return config
}

func testHTTPBatch(
	id string,
	destinationID export.DestinationID,
) export.Batch {
	now := time.Now().UTC()

	return export.Batch{
		ID: export.BatchID(id),

		Destination: export.Destination{
			ID:   destinationID,
			Type: "http",
		},

		Records: []export.Record{
			{
				ID: export.RecordID(
					id + "-record-1",
				),

				Type:     export.RecordTypeEvent,
				Priority: export.PriorityNormal,

				Source: export.Source{
					System:    "test",
					Component: "http-exporter-test",
				},

				ContentType: export.ContentTypeJSON,
				Compression: export.CompressionNone,

				CreatedAt: now,

				Payload: []byte(
					`{"value":1}`,
				),
			},
		},

		CreatedAt: now,
	}
}

func newTestExporter(
	t *testing.T,
	server *httptest.Server,
) *Exporter {
	t.Helper()

	exporter, err := NewExporter(
		testHTTPConfig(server.URL),
	)
	if err != nil {
		t.Fatalf(
			"NewExporter() error = %v",
			err,
		)
	}

	t.Cleanup(func() {
		_ = exporter.Close()
	})

	return exporter
}

func TestExporterSuccessEmptyResponse(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				if request.Method != nethttp.MethodPost {
					t.Errorf(
						"method = %q, want POST",
						request.Method,
					)
				}

				if got := request.Header.Get(
					"Content-Type",
				); got != "application/json" {
					t.Errorf(
						"Content-Type = %q",
						got,
					)
				}

				body, err := io.ReadAll(
					request.Body,
				)
				if err != nil {
					t.Errorf(
						"ReadAll() error = %v",
						err,
					)

					writer.WriteHeader(
						nethttp.StatusInternalServerError,
					)
					return
				}

				var batch export.Batch

				if err := json.Unmarshal(
					body,
					&batch,
				); err != nil {
					t.Errorf(
						"decode batch error = %v",
						err,
					)

					writer.WriteHeader(
						nethttp.StatusBadRequest,
					)
					return
				}

				if batch.ID != "batch-1" {
					t.Errorf(
						"batch ID = %q, want batch-1",
						batch.ID,
					)
				}

				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-1",
			exporter.ID(),
		),
	)

	if err != nil {
		t.Fatalf(
			"Export() error = %v",
			err,
		)
	}

	if result.Status !=
		export.DeliveryAcknowledged {
		t.Fatalf(
			"status = %s, want acknowledged",
			result.Status,
		)
	}

	if result.AcceptedRecords != 1 {
		t.Fatalf(
			"accepted = %d, want 1",
			result.AcceptedRecords,
		)
	}

	if result.RejectedRecords != 0 {
		t.Fatalf(
			"rejected = %d, want 0",
			result.RejectedRecords,
		)
	}

	if result.Retryable {
		t.Fatal(
			"successful delivery must not be retryable",
		)
	}
}

func TestExporterStructuredSuccess(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.Header().Set(
					"Content-Type",
					"application/json",
				)

				writer.WriteHeader(
					nethttp.StatusOK,
				)

				_, _ = writer.Write(
					[]byte(
						`{"accepted_records":1,"rejected_records":0}`,
					),
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-structured",
			exporter.ID(),
		),
	)

	if err != nil {
		t.Fatalf(
			"Export() error = %v",
			err,
		)
	}

	if !result.Successful() {
		t.Fatalf(
			"result = %+v, want successful",
			result,
		)
	}
}

func TestExporterBearerAuthentication(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				if got := request.Header.Get(
					"Authorization",
				); got != "Bearer secret-token" {
					t.Errorf(
						"Authorization = %q",
						got,
					)
				}

				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	config := testHTTPConfig(
		server.URL,
	)

	config.Authentication = AuthenticationConfig{
		Type:        AuthenticationBearer,
		BearerToken: "secret-token",
	}

	exporter, err := NewExporter(config)
	if err != nil {
		t.Fatalf(
			"NewExporter() error = %v",
			err,
		)
	}
	defer exporter.Close()

	_, err = exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-auth",
			exporter.ID(),
		),
	)

	if err != nil {
		t.Fatalf(
			"Export() error = %v",
			err,
		)
	}
}

func TestExporterAPIKeyAuthentication(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				if got := request.Header.Get(
					"X-OTGH-Key",
				); got != "test-key" {
					t.Errorf(
						"X-OTGH-Key = %q",
						got,
					)
				}

				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	config := testHTTPConfig(
		server.URL,
	)

	config.Authentication = AuthenticationConfig{
		Type:         AuthenticationAPIKey,
		APIKey:       "test-key",
		APIKeyHeader: "X-OTGH-Key",
	}

	exporter, err := NewExporter(config)
	if err != nil {
		t.Fatalf(
			"NewExporter() error = %v",
			err,
		)
	}
	defer exporter.Close()

	_, err = exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-api-key",
			exporter.ID(),
		),
	)

	if err != nil {
		t.Fatalf(
			"Export() error = %v",
			err,
		)
	}
}

func TestExporterTooManyRequestsRetryable(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.Header().Set(
					"Retry-After",
					"3",
				)

				writer.WriteHeader(
					nethttp.StatusTooManyRequests,
				)

				_, _ = writer.Write(
					[]byte("slow down"),
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-429",
			exporter.ID(),
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected error",
		)
	}

	if !result.Retryable {
		t.Fatal(
			"429 result should be retryable",
		)
	}

	if result.Status !=
		export.DeliveryRetrying {
		t.Fatalf(
			"status = %s, want retrying",
			result.Status,
		)
	}

	var httpErr *HTTPError

	if !errors.As(err, &httpErr) {
		t.Fatalf(
			"error = %T, want *HTTPError",
			err,
		)
	}

	if !httpErr.Retryable {
		t.Fatal(
			"HTTPError should be retryable",
		)
	}

	if httpErr.RetryAfter !=
		3*time.Second {
		t.Fatalf(
			"RetryAfter = %s, want 3s",
			httpErr.RetryAfter,
		)
	}

	decision := exporter.ClassifyRetry(
		err,
	)

	if !decision.Retry {
		t.Fatalf(
			"ClassifyRetry() = %+v, want retry",
			decision,
		)
	}
}

func TestExporterServerErrorRetryable(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.WriteHeader(
					nethttp.StatusServiceUnavailable,
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-503",
			exporter.ID(),
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected error",
		)
	}

	if !result.Retryable {
		t.Fatal(
			"503 should be retryable",
		)
	}

	if !exporter.ClassifyRetry(err).Retry {
		t.Fatal(
			"503 error should classify as retryable",
		)
	}
}

func TestExporterBadRequestNotRetryable(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.WriteHeader(
					nethttp.StatusBadRequest,
				)

				_, _ = writer.Write(
					[]byte("invalid request"),
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-400",
			exporter.ID(),
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected error",
		)
	}

	if result.Retryable {
		t.Fatal(
			"400 must not be retryable",
		)
	}

	if exporter.ClassifyRetry(err).Retry {
		t.Fatal(
			"400 should classify as non-retryable",
		)
	}
}

func TestExporterPartialResponse(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.Header().Set(
					"Content-Type",
					"application/json",
				)

				_, _ = writer.Write(
					[]byte(
						`{"accepted_records":1,"rejected_records":1,"retryable":false,"message":"partial"}`,
					),
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	batch := testHTTPBatch(
		"batch-partial",
		exporter.ID(),
	)

	second := batch.Records[0]
	second.ID = "batch-partial-record-2"
	second.Payload = []byte(
		`{"value":2}`,
	)

	batch.Records = append(
		batch.Records,
		second,
	)

	result, err := exporter.Export(
		context.Background(),
		batch,
	)

	if err != nil {
		t.Fatalf(
			"Export() error = %v",
			err,
		)
	}

	if result.AcceptedRecords != 1 {
		t.Fatalf(
			"accepted = %d, want 1",
			result.AcceptedRecords,
		)
	}

	if result.RejectedRecords != 1 {
		t.Fatalf(
			"rejected = %d, want 1",
			result.RejectedRecords,
		)
	}

	if result.Successful() {
		t.Fatal(
			"partial response must not be completely successful",
		)
	}
}

func TestExporterMalformedSuccessResponse(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.WriteHeader(
					nethttp.StatusOK,
				)

				_, _ = writer.Write(
					[]byte(`{"accepted_records":`),
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-malformed",
			exporter.ID(),
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected malformed-response error",
		)
	}

	if result.ErrorCode !=
		"invalid_success_response" {
		t.Fatalf(
			"ErrorCode = %q",
			result.ErrorCode,
		)
	}
}

func TestExporterContextCancellation(
	t *testing.T,
) {
	started := make(
		chan struct{},
	)
	releaseHandler := make(chan struct{})

	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				close(started)

				<-releaseHandler
				writer.WriteHeader(nethttp.StatusNoContent)
			},
		),
	)
	defer func() {
		close(releaseHandler)
		server.Close()
	}()

	exporter := newTestExporter(
		t,
		server,
	)

	ctx, cancel :=
		context.WithCancel(
			context.Background(),
		)

	resultCh := make(
		chan error,
		1,
	)

	go func() {
		_, err := exporter.Export(
			ctx,
			testHTTPBatch(
				"batch-cancel",
				exporter.ID(),
			),
		)

		resultCh <- err
	}()

	select {
	case <-started:

	case <-time.After(2 * time.Second):
		t.Fatal(
			"server did not receive request",
		)
	}

	cancel()

	select {
	case err := <-resultCh:
		if !errors.Is(
			err,
			context.Canceled,
		) {
			t.Fatalf(
				"error = %v, want context.Canceled",
				err,
			)
		}

	case <-time.After(2 * time.Second):
		t.Fatal(
			"Export() did not return after cancellation",
		)
	}
}

func TestExporterRequestTooLarge(
	t *testing.T,
) {
	var calls atomic.Int64

	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				calls.Add(1)

				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	config := testHTTPConfig(
		server.URL,
	)

	config.MaxRequestBytes = 8

	exporter, err := NewExporter(config)
	if err != nil {
		t.Fatalf(
			"NewExporter() error = %v",
			err,
		)
	}
	defer exporter.Close()

	_, err = exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-large",
			exporter.ID(),
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected request-size error",
		)
	}

	if calls.Load() != 0 {
		t.Fatalf(
			"server calls = %d, want 0",
			calls.Load(),
		)
	}
}

func TestExporterResponseTooLarge(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.WriteHeader(
					nethttp.StatusOK,
				)

				_, _ = writer.Write(
					[]byte(
						"0123456789abcdef",
					),
				)
			},
		),
	)
	defer server.Close()

	config := testHTTPConfig(
		server.URL,
	)

	config.MaxResponseBytes = 4

	exporter, err := NewExporter(config)
	if err != nil {
		t.Fatalf(
			"NewExporter() error = %v",
			err,
		)
	}
	defer exporter.Close()

	result, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-large-response",
			exporter.ID(),
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected response-size error",
		)
	}

	if result.Retryable {
		t.Fatal(
			"local response-size violation must not be retryable",
		)
	}
}

func TestExporterDestinationMismatch(
	t *testing.T,
) {
	var calls atomic.Int64

	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				calls.Add(1)

				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	_, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-wrong-destination",
			"another-destination",
		),
	)

	if err == nil {
		t.Fatal(
			"Export() expected destination mismatch",
		)
	}

	if calls.Load() != 0 {
		t.Fatalf(
			"server received %d requests, want 0",
			calls.Load(),
		)
	}
}

func TestExporterAfterClose(
	t *testing.T,
) {
	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	if err := exporter.Close(); err != nil {
		t.Fatalf(
			"Close() error = %v",
			err,
		)
	}

	_, err := exporter.Export(
		context.Background(),
		testHTTPBatch(
			"batch-closed",
			exporter.ID(),
		),
	)

	if !errors.Is(
		err,
		ErrHTTPExporterClosed,
	) {
		t.Fatalf(
			"error = %v, want ErrHTTPExporterClosed",
			err,
		)
	}

	if !exporter.Closed() {
		t.Fatal(
			"exporter should report closed",
		)
	}
}

func TestParseRetryAfterHTTPDate(
	t *testing.T,
) {
	now := time.Date(
		2026,
		time.November,
		26,
		15,
		0,
		0,
		0,
		time.UTC,
	)

	target := now.Add(
		10 * time.Second,
	)

	delay := parseRetryAfter(
		target.Format(
			nethttp.TimeFormat,
		),
		now,
	)

	if delay != 10*time.Second {
		t.Fatalf(
			"delay = %s, want 10s",
			delay,
		)
	}
}

// TestExporterConcurrent exercises the reusable exporter/client/transport
// from many goroutines.
//
// Run with:
//
//	go test -race ./internal/export/http
//
// The Go race detector then observes these simultaneous calls.
func TestExporterConcurrent(
	t *testing.T,
) {
	const (
		workers   = 16
		perWorker = 50
	)

	var requests atomic.Int64

	server := httptest.NewServer(
		nethttp.HandlerFunc(
			func(
				writer nethttp.ResponseWriter,
				request *nethttp.Request,
			) {
				requests.Add(1)

				writer.WriteHeader(
					nethttp.StatusNoContent,
				)
			},
		),
	)
	defer server.Close()

	exporter := newTestExporter(
		t,
		server,
	)

	ctx, cancel := context.WithTimeout(
		context.Background(),
		10*time.Second,
	)
	defer cancel()

	var wg sync.WaitGroup

	errorsCh := make(
		chan error,
		workers,
	)

	wg.Add(workers)

	for worker := 0; worker < workers; worker++ {
		worker := worker

		go func() {
			defer wg.Done()

			for item := 0; item < perWorker; item++ {
				id := fmt.Sprintf(
					"worker-%d-batch-%d",
					worker,
					item,
				)

				result, err := exporter.Export(
					ctx,
					testHTTPBatch(
						id,
						exporter.ID(),
					),
				)

				if err != nil {
					errorsCh <- fmt.Errorf(
						"%s: %w",
						id,
						err,
					)
					return
				}

				if !result.Successful() {
					errorsCh <- fmt.Errorf(
						"%s: unsuccessful result %+v",
						id,
						result,
					)
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errorsCh)

	for err := range errorsCh {
		if err != nil {
			t.Fatal(err)
		}
	}

	expected :=
		int64(workers * perWorker)

	if requests.Load() != expected {
		t.Fatalf(
			"requests = %d, want %d",
			requests.Load(),
			expected,
		)
	}
}
