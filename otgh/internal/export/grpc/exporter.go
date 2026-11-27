package grpc

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/samorca/otgh/internal/export"
	grpcgo "google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	grpc_health_v1 "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

const (
	// ExportMethod is the initial OTGH unary gRPC export contract.
	//
	// Request:
	//   google.protobuf.BytesValue containing JSON-encoded export.Batch.
	//
	// Response:
	//   google.protobuf.BytesValue containing JSON-encoded ExportResponse.
	//
	// This direct invocation contract can later be replaced by generated OTGH
	// protobuf bindings without changing the export.Exporter interface.
	ExportMethod = "/otgh.v1.ExportService/Export"

	// HealthServiceName is submitted to the standard gRPC health service.
	//
	// An empty service name asks for the overall server health status.
	HealthServiceName = ""
)

var (
	// ErrGRPCExporterClosed indicates that the exporter has been closed.
	ErrGRPCExporterClosed = errors.New(
		"gRPC exporter closed",
	)

	// ErrGRPCDeliveryRejected indicates that the remote OTGH gRPC service
	// rejected the batch.
	ErrGRPCDeliveryRejected = errors.New(
		"gRPC delivery rejected",
	)

	// ErrGRPCInvalidResponse indicates that the RPC completed but returned a
	// response that violates the OTGH export contract.
	ErrGRPCInvalidResponse = errors.New(
		"invalid gRPC export response",
	)
)

// ExportResponse is the response contract returned inside the gRPC
// BytesValue payload.
type ExportResponse struct {
	AcceptedRecords int `json:"accepted_records"`
	RejectedRecords int `json:"rejected_records"`

	Retryable bool `json:"retryable"`

	Message string `json:"message,omitempty"`

	ErrorCode string `json:"error_code,omitempty"`
}

// GRPCError describes a gRPC export failure.
type GRPCError struct {
	Code codes.Code

	Message string

	Retryable bool

	Cause error
}

// Error implements error.
func (e *GRPCError) Error() string {
	if e == nil {
		return "<nil>"
	}

	switch {
	case e.Message != "":
		return fmt.Sprintf(
			"gRPC export failed with code %s: %s",
			e.Code,
			e.Message,
		)

	case e.Cause != nil:
		return fmt.Sprintf(
			"gRPC export failed with code %s: %v",
			e.Code,
			e.Cause,
		)

	default:
		return fmt.Sprintf(
			"gRPC export failed with code %s",
			e.Code,
		)
	}
}

// Unwrap exposes the underlying local error.
func (e *GRPCError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.Cause
}

// Exporter delivers OTGH batches through one reusable gRPC ClientConn.
//
// Exporter is safe for concurrent use. gRPC connection recovery is handled by
// ClientConn. Batch retry timing and retry budgets remain the responsibility
// of export.RetryExecutor.
type Exporter struct {
	config Config

	conn *grpcgo.ClientConn

	healthClient grpc_health_v1.HealthClient

	closeOnce sync.Once
	closed    atomic.Bool
}

// NewExporter creates a gRPC exporter and its reusable ClientConn.
//
// grpc.NewClient creates the logical client channel. Network activity begins
// when an RPC or explicit connection attempt requires it.
func NewExporter(
	config Config,
) (*Exporter, error) {
	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create gRPC exporter: %w",
			err,
		)
	}

	options, err := buildDialOptions(config)
	if err != nil {
		return nil, fmt.Errorf(
			"create gRPC dial options: %w",
			err,
		)
	}

	conn, err := grpcgo.NewClient(
		config.Endpoint,
		options...,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"create gRPC client connection: %w",
			err,
		)
	}

	return &Exporter{
		config: config,

		conn: conn,

		healthClient: grpc_health_v1.
			NewHealthClient(conn),
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
	return "grpc"
}

// Export serializes and delivers one batch using the OTGH unary export
// contract.
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
		return failedResult(
			result,
			"grpc_exporter_nil",
			"gRPC exporter is nil",
			false,
			errors.New("gRPC exporter is nil"),
		)
	}

	if ctx == nil {
		return failedResult(
			result,
			"context_nil",
			"gRPC export context is nil",
			false,
			errors.New("gRPC export context is nil"),
		)
	}

	if e.closed.Load() {
		return failedResult(
			result,
			"grpc_exporter_closed",
			ErrGRPCExporterClosed.Error(),
			false,
			ErrGRPCExporterClosed,
		)
	}

	if err := ctx.Err(); err != nil {
		return failedResult(
			result,
			"context_cancelled",
			err.Error(),
			false,
			err,
		)
	}

	if err := batch.Validate(); err != nil {
		return failedResult(
			result,
			"invalid_batch",
			err.Error(),
			false,
			fmt.Errorf(
				"gRPC export batch validation: %w",
				err,
			),
		)
	}

	if batch.Destination.ID != e.ID() {
		err := fmt.Errorf(
			"batch destination %q does not match gRPC exporter %q",
			batch.Destination.ID,
			e.ID(),
		)

		return failedResult(
			result,
			"destination_mismatch",
			err.Error(),
			false,
			err,
		)
	}

	payload, err := json.Marshal(batch)
	if err != nil {
		return failedResult(
			result,
			"batch_encoding_failed",
			err.Error(),
			false,
			fmt.Errorf(
				"encode gRPC batch %q: %w",
				batch.ID,
				err,
			),
		)
	}

	if len(payload) >
		e.config.MaxSendMessageBytes {
		err := fmt.Errorf(
			"encoded batch contains %d bytes, maximum is %d",
			len(payload),
			e.config.MaxSendMessageBytes,
		)

		return failedResult(
			result,
			"grpc_request_too_large",
			err.Error(),
			false,
			err,
		)
	}

	rpcCtx, cancel := withDefaultTimeout(
		ctx,
		e.config.Timeouts.Request,
	)
	defer cancel()

	rpcCtx, err = e.outgoingContext(rpcCtx)
	if err != nil {
		return failedResult(
			result,
			"grpc_metadata_failed",
			err.Error(),
			false,
			err,
		)
	}

	request := wrapperspb.Bytes(payload)
	response := &wrapperspb.BytesValue{}

	err = e.conn.Invoke(
		rpcCtx,
		ExportMethod,
		request,
		response,
		grpcgo.MaxCallSendMsgSize(
			e.config.MaxSendMessageBytes,
		),
		grpcgo.MaxCallRecvMsgSize(
			e.config.MaxReceiveMessageBytes,
		),
		grpcgo.WaitForReady(
			e.config.WaitForReady,
		),
	)

	result.FinishedAt = time.Now().UTC()

	if err != nil {
		grpcErr := classifyGRPCError(err)

		if grpcErr.Retryable {
			result.Status = export.DeliveryRetrying
		} else {
			result.Status = export.DeliveryFailed
		}

		result.Retryable = grpcErr.Retryable
		result.ErrorCode = grpcErrorCode(
			grpcErr.Code,
		)
		result.ErrorMessage = grpcErr.Error()

		return result, grpcErr
	}

	exportResponse, err := parseExportResponse(
		response.Value,
		batch.RecordCount(),
	)
	if err != nil {
		result.Status = export.DeliveryFailed
		result.ErrorCode = "grpc_invalid_response"
		result.ErrorMessage = err.Error()

		return result, err
	}

	result.AcceptedRecords =
		exportResponse.AcceptedRecords

	result.RejectedRecords =
		exportResponse.RejectedRecords

	result.Retryable =
		exportResponse.Retryable

	result.ErrorCode = strings.TrimSpace(
		exportResponse.ErrorCode,
	)

	result.ErrorMessage = strings.TrimSpace(
		exportResponse.Message,
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

		return result, &GRPCError{
			Code:      codes.Unavailable,
			Message:   result.ErrorMessage,
			Retryable: true,
			Cause:     ErrGRPCDeliveryRejected,
		}

	default:
		// A non-retryable partial response is returned without an RPC error.
		// The acknowledgement manager and Worker determine whether durable
		// delivery state may be removed.
		result.Status =
			export.DeliveryAcknowledged

		return result, nil
	}
}

// Health implements export.Exporter using the standard gRPC health service.
func (e *Exporter) Health(
	ctx context.Context,
) error {
	if e == nil {
		return errors.New(
			"gRPC exporter is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"gRPC health context is nil",
		)
	}

	if e.closed.Load() {
		return ErrGRPCExporterClosed
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	healthCtx, cancel := withDefaultTimeout(
		ctx,
		e.config.Timeouts.Health,
	)
	defer cancel()

	healthCtx, err := e.outgoingContext(
		healthCtx,
	)
	if err != nil {
		return err
	}

	response, err := e.healthClient.Check(
		healthCtx,
		&grpc_health_v1.HealthCheckRequest{
			Service: HealthServiceName,
		},
		grpcgo.MaxCallRecvMsgSize(
			e.config.MaxReceiveMessageBytes,
		),
		grpcgo.WaitForReady(
			e.config.WaitForReady,
		),
	)
	if err != nil {
		return fmt.Errorf(
			"gRPC health check failed: %w",
			err,
		)
	}

	if response.Status !=
		grpc_health_v1.
			HealthCheckResponse_SERVING {
		return fmt.Errorf(
			"gRPC destination health status is %s",
			response.Status,
		)
	}

	return nil
}

// Connect begins connection establishment and waits until the channel becomes
// Ready or the configured connect timeout expires.
//
// Export and Health can also establish connectivity lazily, so calling
// Connect is optional.
func (e *Exporter) Connect(
	ctx context.Context,
) error {
	if e == nil {
		return errors.New(
			"gRPC exporter is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"gRPC connect context is nil",
		)
	}

	if e.closed.Load() {
		return ErrGRPCExporterClosed
	}

	connectCtx, cancel := withDefaultTimeout(
		ctx,
		e.config.Timeouts.Connect,
	)
	defer cancel()

	e.conn.Connect()

	for {
		state := e.conn.GetState()

		if state == connectivity.Ready {
			return nil
		}

		if state == connectivity.Shutdown {
			return ErrGRPCExporterClosed
		}

		if !e.conn.WaitForStateChange(
			connectCtx,
			state,
		) {
			if err := connectCtx.Err(); err != nil {
				return err
			}

			return errors.New(
				"gRPC connection state did not change",
			)
		}
	}
}

// Close implements export.Exporter.
func (e *Exporter) Close() error {
	if e == nil {
		return nil
	}

	var closeErr error

	e.closeOnce.Do(func() {
		e.closed.Store(true)

		if e.conn != nil {
			closeErr = e.conn.Close()
		}
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

	if errors.Is(err, ErrGRPCExporterClosed) ||
		errors.Is(err, ErrGRPCInvalidResponse) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "permanent local gRPC exporter failure",
		}
	}

	var grpcErr *GRPCError

	if errors.As(err, &grpcErr) {
		return export.RetryDecision{
			Retry: grpcErr.Retryable,
			Reason: fmt.Sprintf(
				"gRPC code %s retryable=%t",
				grpcErr.Code,
				grpcErr.Retryable,
			),
		}
	}

	code := status.Code(err)

	return export.RetryDecision{
		Retry: retryableGRPCCode(code),
		Reason: fmt.Sprintf(
			"gRPC code %s",
			code,
		),
	}
}

func buildDialOptions(
	config Config,
) ([]grpcgo.DialOption, error) {
	transportCredentials, err :=
		buildTransportCredentials(config)
	if err != nil {
		return nil, err
	}

	options := []grpcgo.DialOption{
		grpcgo.WithTransportCredentials(
			transportCredentials,
		),

		grpcgo.WithKeepaliveParams(
			keepalive.ClientParameters{
				Time: config.Keepalive.Time,

				Timeout: config.Keepalive.Timeout,

				PermitWithoutStream: config.
					Keepalive.
					PermitWithoutStream,
			},
		),

		grpcgo.WithConnectParams(
			grpcgo.ConnectParams{
				Backoff: backoff.Config{
					BaseDelay: config.
						Reconnect.
						BaseDelay,

					Multiplier: config.
						Reconnect.
						Multiplier,

					Jitter: config.
						Reconnect.
						Jitter,

					MaxDelay: config.
						Reconnect.
						MaxDelay,
				},

				MinConnectTimeout: config.
					Timeouts.
					Connect,
			},
		),

		grpcgo.WithDefaultCallOptions(
			grpcgo.MaxCallSendMsgSize(
				config.MaxSendMessageBytes,
			),

			grpcgo.MaxCallRecvMsgSize(
				config.MaxReceiveMessageBytes,
			),

			grpcgo.WaitForReady(
				config.WaitForReady,
			),
		),
	}

	if config.Authority != "" {
		options = append(
			options,
			grpcgo.WithAuthority(
				config.Authority,
			),
		)
	}

	return options, nil
}

func buildTransportCredentials(
	config Config,
) (
	credentials.TransportCredentials,
	error,
) {
	if !config.TLS.Enabled {
		return insecure.NewCredentials(), nil
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
				"load gRPC client certificate: %w",
				err,
			)
		}

		tlsConfig.Certificates =
			[]tls.Certificate{
				certificate,
			}
	}

	return credentials.NewTLS(
		tlsConfig,
	), nil
}

func loadCertPool(
	caFile string,
) (*x509.CertPool, error) {
	pemData, err := os.ReadFile(
		strings.TrimSpace(caFile),
	)
	if err != nil {
		return nil, fmt.Errorf(
			"read gRPC CA file: %w",
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
			"gRPC CA file contains no usable certificates",
		)
	}

	return pool, nil
}

func (e *Exporter) outgoingContext(
	ctx context.Context,
) (context.Context, error) {
	if ctx == nil {
		return nil, errors.New(
			"gRPC outgoing context is nil",
		)
	}

	values := make(
		[]string,
		0,
		2*(len(e.config.Metadata)+1),
	)

	for key, value := range e.config.Metadata {
		values = append(
			values,
			key,
			value,
		)
	}

	switch e.config.Authentication.Type {
	case AuthenticationNone:
		// Nothing to add.

	case AuthenticationBearer:
		values = append(
			values,
			"authorization",
			"Bearer "+
				e.config.Authentication.
					BearerToken,
		)

	case AuthenticationAPIKey:
		values = append(
			values,
			e.config.Authentication.
				APIKeyMetadataKey,
			e.config.Authentication.APIKey,
		)

	default:
		return nil, fmt.Errorf(
			"unsupported gRPC authentication type %q",
			e.config.Authentication.Type,
		)
	}

	if len(values) == 0 {
		return ctx, nil
	}

	return metadata.NewOutgoingContext(
		ctx,
		metadata.Pairs(values...),
	), nil
}

func withDefaultTimeout(
	ctx context.Context,
	timeout time.Duration,
) (
	context.Context,
	context.CancelFunc,
) {
	if _, exists := ctx.Deadline(); exists {
		return context.WithCancel(ctx)
	}

	return context.WithTimeout(
		ctx,
		timeout,
	)
}

func parseExportResponse(
	payload []byte,
	totalRecords int,
) (ExportResponse, error) {
	if len(payload) == 0 {
		return ExportResponse{
			AcceptedRecords: totalRecords,
		}, nil
	}

	var response ExportResponse

	decoder := json.NewDecoder(
		strings.NewReader(
			string(payload),
		),
	)

	decoder.DisallowUnknownFields()

	if err := decoder.Decode(
		&response,
	); err != nil {
		return ExportResponse{}, fmt.Errorf(
			"%w: decode response: %v",
			ErrGRPCInvalidResponse,
			err,
		)
	}

	if response.AcceptedRecords < 0 ||
		response.RejectedRecords < 0 {
		return ExportResponse{}, fmt.Errorf(
			"%w: response record counts cannot be negative",
			ErrGRPCInvalidResponse,
		)
	}

	accounted := response.AcceptedRecords +
		response.RejectedRecords

	if accounted > totalRecords {
		return ExportResponse{}, fmt.Errorf(
			"%w: response accounts for %d records but batch contains %d",
			ErrGRPCInvalidResponse,
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

func classifyGRPCError(
	err error,
) *GRPCError {
	code := status.Code(err)

	message := status.Convert(err).
		Message()

	return &GRPCError{
		Code:      code,
		Message:   message,
		Retryable: retryableGRPCCode(code),
		Cause:     err,
	}
}

func retryableGRPCCode(
	code codes.Code,
) bool {
	switch code {
	case codes.Unavailable,
		codes.ResourceExhausted,
		codes.Aborted:
		return true

	default:
		return false
	}
}

func grpcErrorCode(
	code codes.Code,
) string {
	name := strings.ToLower(
		code.String(),
	)

	name = strings.ReplaceAll(
		name,
		" ",
		"_",
	)

	return "grpc_" + name
}

func failedResult(
	result export.DeliveryResult,
	code string,
	message string,
	retryable bool,
	err error,
) (export.DeliveryResult, error) {
	result.FinishedAt = time.Now().UTC()
	result.Retryable = retryable
	result.ErrorCode = code
	result.ErrorMessage = message

	if retryable {
		result.Status = export.DeliveryRetrying
	} else {
		result.Status = export.DeliveryFailed
	}

	return result, err
}

// Compile-time interface assertions.
var (
	_ export.Exporter =
		(*Exporter)(nil)

	_ export.RetryClassifier =
		(*Exporter)(nil)

	_ net.Error = (*temporaryNetworkError)(nil)
)

// temporaryNetworkError is reserved for future connection-level wrappers.
//
// Its presence keeps network-failure classification extensible without
// exposing gRPC internals from the exporter API.
type temporaryNetworkError struct {
	err error
}

func (e *temporaryNetworkError) Error() string {
	if e == nil || e.err == nil {
		return "temporary network error"
	}

	return e.err.Error()
}

func (e *temporaryNetworkError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.err
}

func (e *temporaryNetworkError) Timeout() bool {
	return false
}

func (e *temporaryNetworkError) Temporary() bool {
	return true
}
