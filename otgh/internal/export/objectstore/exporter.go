package objectstore

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
	"github.com/aws/smithy-go/logging"

	"github.com/samorca/otgh/internal/export"
)

var (
	ErrObjectStoreExporterClosed = errors.New(
		"object-store exporter closed",
	)

	ErrObjectStoreUploadRejected = errors.New(
		"object-store upload rejected",
	)

	ErrObjectStoreObjectTooLarge = errors.New(
		"object-store object too large",
	)
)

// ObjectStoreAPI contains only the storage operations OTGH requires.
//
// Keeping the exporter dependent on this small interface instead of *s3.Client
// directly makes the delivery logic testable and leaves room for another
// S3-compatible adapter later.
type ObjectStoreAPI interface {
	PutObject(
		ctx context.Context,
		params *s3.PutObjectInput,
		optFns ...func(*s3.Options),
	) (*s3.PutObjectOutput, error)

	HeadBucket(
		ctx context.Context,
		params *s3.HeadBucketInput,
		optFns ...func(*s3.Options),
	) (*s3.HeadBucketOutput, error)
}

// StoreError describes a provider operation failure.
type StoreError struct {
	Operation string

	Code string

	Message string

	Retryable bool

	Cause error
}

func (e *StoreError) Error() string {
	if e == nil {
		return "<nil>"
	}

	switch {
	case e.Operation != "" &&
		e.Code != "" &&
		e.Message != "":
		return fmt.Sprintf(
			"object-store %s failed with code %s: %s",
			e.Operation,
			e.Code,
			e.Message,
		)

	case e.Operation != "" &&
		e.Message != "":
		return fmt.Sprintf(
			"object-store %s failed: %s",
			e.Operation,
			e.Message,
		)

	case e.Cause != nil:
		return fmt.Sprintf(
			"object-store operation failed: %v",
			e.Cause,
		)

	default:
		return "object-store operation failed"
	}
}

func (e *StoreError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.Cause
}

// Exporter writes OTGH delivery batches as immutable object-store objects.
//
// One batch maps to one object:
//
//	prefix/YYYY/MM/DD/<destination>/<batch>.json
//
// Batch IDs are sanitized before becoming part of an object key.
type Exporter struct {
	config Config

	client ObjectStoreAPI

	closeOnce sync.Once
	closed    atomic.Bool
}

// NewExporter constructs the provider client and exporter.
func NewExporter(
	ctx context.Context,
	config Config,
) (*Exporter, error) {
	if ctx == nil {
		return nil, errors.New(
			"create object-store exporter: context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create object-store exporter: %w",
			err,
		)
	}

	client, err := newS3Client(
		ctx,
		config,
	)
	if err != nil {
		return nil, err
	}

	return &Exporter{
		config: config,
		client: client,
	}, nil
}

// NewExporterWithClient creates an exporter around a supplied storage client.
//
// Primarily useful for tests and controlled adapters.
func NewExporterWithClient(
	config Config,
	client ObjectStoreAPI,
) (*Exporter, error) {
	config = config.Normalized()

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf(
			"create object-store exporter: %w",
			err,
		)
	}

	if client == nil {
		return nil, errors.New(
			"create object-store exporter: client is required",
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

	return export.DestinationID(
		e.config.ID,
	)
}

// Type implements export.Exporter.
func (e *Exporter) Type() string {
	return "objectstore"
}

// Export serializes one OTGH batch and stores it as one object.
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
		return storeFailedResult(
			result,
			"objectstore_exporter_nil",
			"object-store exporter is nil",
			false,
			errors.New(
				"object-store exporter is nil",
			),
		)
	}

	if ctx == nil {
		return storeFailedResult(
			result,
			"context_nil",
			"object-store export context is nil",
			false,
			errors.New(
				"object-store export context is nil",
			),
		)
	}

	if e.closed.Load() {
		return storeFailedResult(
			result,
			"objectstore_exporter_closed",
			ErrObjectStoreExporterClosed.Error(),
			false,
			ErrObjectStoreExporterClosed,
		)
	}

	if err := ctx.Err(); err != nil {
		return storeFailedResult(
			result,
			"context_cancelled",
			err.Error(),
			false,
			err,
		)
	}

	if err := batch.Validate(); err != nil {
		return storeFailedResult(
			result,
			"invalid_batch",
			err.Error(),
			false,
			fmt.Errorf(
				"object-store batch validation: %w",
				err,
			),
		)
	}

	if batch.Destination.ID != e.ID() {
		err := fmt.Errorf(
			"batch destination %q does not match object-store exporter %q",
			batch.Destination.ID,
			e.ID(),
		)

		return storeFailedResult(
			result,
			"destination_mismatch",
			err.Error(),
			false,
			err,
		)
	}

	payload, err := json.Marshal(batch)
	if err != nil {
		return storeFailedResult(
			result,
			"batch_encoding_failed",
			err.Error(),
			false,
			fmt.Errorf(
				"encode object-store batch %q: %w",
				batch.ID,
				err,
			),
		)
	}

	if int64(len(payload)) >
		e.config.Upload.MaxObjectBytes {
		err := fmt.Errorf(
			"%w: encoded batch is %d bytes, maximum is %d",
			ErrObjectStoreObjectTooLarge,
			len(payload),
			e.config.Upload.MaxObjectBytes,
		)

		return storeFailedResult(
			result,
			"object_too_large",
			err.Error(),
			false,
			err,
		)
	}

	objectKey, err := e.objectKey(batch)
	if err != nil {
		return storeFailedResult(
			result,
			"object_key_failed",
			err.Error(),
			false,
			err,
		)
	}

	uploadCtx, cancel := withUploadTimeout(
		ctx,
		e.config.Upload.Timeout,
	)
	defer cancel()

	input, err := e.putObjectInput(
		batch,
		objectKey,
		payload,
	)
	if err != nil {
		return storeFailedResult(
			result,
			"object_request_failed",
			err.Error(),
			false,
			err,
		)
	}

	output, err := e.client.PutObject(
		uploadCtx,
		input,
	)

	result.FinishedAt = time.Now().UTC()

	if err != nil {
		storeErr := classifyStoreError(
			"put_object",
			err,
		)

		if storeErr.Retryable {
			result.Status =
				export.DeliveryRetrying
		} else {
			result.Status =
				export.DeliveryFailed
		}

		result.Retryable =
			storeErr.Retryable

		result.ErrorCode =
			storeErrorCode(
				storeErr,
			)

		result.ErrorMessage =
			storeErr.Error()

		return result, storeErr
	}

	result.Status =
		export.DeliveryAcknowledged

	result.AcceptedRecords =
		batch.RecordCount()

	result.Retryable = false

	result.Metadata = map[string]string{
		"objectstore.bucket":
			e.config.Bucket,

		"objectstore.key":
			objectKey,
	}

	if output != nil {
		if output.ETag != nil &&
			strings.TrimSpace(
				*output.ETag,
			) != "" {
			result.Metadata["objectstore.etag"] =
				strings.Trim(
					*output.ETag,
					`"`,
				)
		}

		if output.VersionId != nil &&
			strings.TrimSpace(
				*output.VersionId,
			) != "" {
			result.Metadata["objectstore.version_id"] =
				*output.VersionId
		}
	}

	return result, nil
}

// Health implements export.Exporter.
//
// HeadBucket provides a lightweight check that the configured storage
// destination is reachable and addressable by this client.
func (e *Exporter) Health(
	ctx context.Context,
) error {
	if e == nil {
		return errors.New(
			"object-store exporter is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"object-store health context is nil",
		)
	}

	if e.closed.Load() {
		return ErrObjectStoreExporterClosed
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	healthCtx, cancel := withUploadTimeout(
		ctx,
		minimumDuration(
			e.config.Upload.Timeout,
			10*time.Second,
		),
	)
	defer cancel()

	_, err := e.client.HeadBucket(
		healthCtx,
		&s3.HeadBucketInput{
			Bucket: aws.String(
				e.config.Bucket,
			),
		},
	)
	if err != nil {
		return classifyStoreError(
			"head_bucket",
			err,
		)
	}

	return nil
}

// Close implements export.Exporter.
//
// AWS SDK S3 clients do not require explicit shutdown for normal use, so
// Close prevents new OTGH exporter operations.
func (e *Exporter) Close() error {
	if e == nil {
		return nil
	}

	e.closeOnce.Do(func() {
		e.closed.Store(true)
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
		errors.Is(err, context.DeadlineExceeded) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "request context ended",
		}
	}

	if errors.Is(
		err,
		ErrObjectStoreExporterClosed,
	) || errors.Is(
		err,
		ErrObjectStoreObjectTooLarge,
	) {
		return export.RetryDecision{
			Retry:  false,
			Reason: "permanent local object-store failure",
		}
	}

	var storeErr *StoreError

	if errors.As(err, &storeErr) {
		return export.RetryDecision{
			Retry: storeErr.Retryable,

			Reason: fmt.Sprintf(
				"object-store code %q retryable=%t",
				storeErr.Code,
				storeErr.Retryable,
			),
		}
	}

	return export.RetryDecision{
		Retry:  false,
		Reason: "unclassified object-store error",
	}
}

func newS3Client(
	ctx context.Context,
	config Config,
) (*s3.Client, error) {
	loadOptions := []func(
		*awsconfig.LoadOptions,
	) error{
		awsconfig.WithRegion(
			config.Region,
		),

		awsconfig.WithRetryMaxAttempts(
			config.TransportRetry.
				MaxAttempts,
		),
	}

	switch config.Credentials.Type {
	case CredentialEnvironment:
		// Use the AWS SDK's configured/default credential loading chain.

	case CredentialStatic:
		provider :=
			credentials.NewStaticCredentialsProvider(
				config.Credentials.AccessKey,
				config.Credentials.SecretKey,
				config.Credentials.SessionToken,
			)

		loadOptions = append(
			loadOptions,
			awsconfig.WithCredentialsProvider(
				provider,
			),
		)

	case CredentialAnonymous:
		loadOptions = append(
			loadOptions,
			awsconfig.WithCredentialsProvider(
				aws.AnonymousCredentials{},
			),
		)

	default:
		return nil, fmt.Errorf(
			"unsupported object-store credential type %q",
			config.Credentials.Type,
		)
	}

	awsConfig, err :=
		awsconfig.LoadDefaultConfig(
			ctx,
			loadOptions...,
		)
	if err != nil {
		return nil, fmt.Errorf(
			"load S3-compatible client configuration: %w",
			err,
		)
	}

	options := func(
		options *s3.Options,
	) {
		options.UsePathStyle =
			config.UsePathStyle

		if config.Endpoint != "" {
			options.BaseEndpoint =
				aws.String(
					config.Endpoint,
				)
		}
	}

	return s3.NewFromConfig(
		awsConfig,
		options,
	), nil
}

func (e *Exporter) putObjectInput(
	batch export.Batch,
	objectKey string,
	payload []byte,
) (*s3.PutObjectInput, error) {
	input := &s3.PutObjectInput{
		Bucket: aws.String(
			e.config.Bucket,
		),

		Key: aws.String(
			objectKey,
		),

		Body: bytes.NewReader(
			payload,
		),

		ContentLength: aws.Int64(
			int64(len(payload)),
		),

		ContentType: aws.String(
			e.config.Upload.ContentType,
		),

		Metadata: cloneStringMap(
			e.config.Metadata,
		),
	}

	if strings.TrimSpace(
		e.config.Upload.StorageClass,
	) != "" {
		input.StorageClass =
			types.StorageClass(
				e.config.Upload.
					StorageClass,
			)
	}

	switch e.config.Encryption.Type {
	case EncryptionNone:
		// No explicit server-side encryption request.

	case EncryptionS3Managed:
		input.ServerSideEncryption =
			types.ServerSideEncryptionAes256

	case EncryptionKMS:
		input.ServerSideEncryption =
			types.ServerSideEncryptionAwsKms

		input.SSEKMSKeyId =
			aws.String(
				e.config.Encryption.
					KMSKeyID,
			)

		if len(
			e.config.Encryption.Context,
		) > 0 {
			encodedContext, err :=
				json.Marshal(
					e.config.Encryption.
						Context,
				)
			if err != nil {
				return nil, fmt.Errorf(
					"encode object-store encryption context: %w",
					err,
				)
			}

			input.SSEKMSEncryptionContext =
				aws.String(
					base64.StdEncoding.
						EncodeToString(
							encodedContext,
						),
				)
		}

	default:
		return nil, fmt.Errorf(
			"unsupported object-store encryption type %q",
			e.config.Encryption.Type,
		)
	}

	if len(e.config.Tags) > 0 {
		input.Tagging = aws.String(
			encodeTags(
				e.config.Tags,
			),
		)
	}

	// Include OTGH identity metadata without overwriting explicit user
	// metadata.
	if input.Metadata == nil {
		input.Metadata = make(
			map[string]string,
			3,
		)
	}

	if _, exists :=
		input.Metadata["otgh-batch-id"]; !exists {
		input.Metadata["otgh-batch-id"] =
			string(batch.ID)
	}

	if _, exists :=
		input.Metadata["otgh-destination-id"]; !exists {
		input.Metadata["otgh-destination-id"] =
			string(
				batch.Destination.ID,
			)
	}

	if _, exists :=
		input.Metadata["otgh-record-count"]; !exists {
		input.Metadata["otgh-record-count"] =
			fmt.Sprintf(
				"%d",
				batch.RecordCount(),
			)
	}

	return input, nil
}

func (e *Exporter) objectKey(
	batch export.Batch,
) (string, error) {
	if strings.TrimSpace(
		string(batch.ID),
	) == "" {
		return "", errors.New(
			"object key requires batch id",
		)
	}

	createdAt := batch.CreatedAt.UTC()

	if createdAt.IsZero() {
		return "", errors.New(
			"object key requires batch creation time",
		)
	}

	destination :=
		sanitizeKeySegment(
			string(
				batch.Destination.ID,
			),
		)

	batchID :=
		sanitizeKeySegment(
			string(batch.ID),
		)

	if destination == "" ||
		batchID == "" {
		return "", errors.New(
			"object key contains empty sanitized identifier",
		)
	}

	key := path.Join(
		strings.TrimSuffix(
			e.config.ObjectKeyPrefix(),
			"/",
		),

		fmt.Sprintf(
			"%04d",
			createdAt.Year(),
		),

		fmt.Sprintf(
			"%02d",
			int(createdAt.Month()),
		),

		fmt.Sprintf(
			"%02d",
			createdAt.Day(),
		),

		destination,

		batchID+".json",
	)

	key = strings.TrimPrefix(
		key,
		"/",
	)

	if key == "" ||
		key == "." {
		return "", errors.New(
			"generated object key is empty",
		)
	}

	return key, nil
}

func sanitizeKeySegment(
	value string,
) string {
	value = strings.TrimSpace(value)

	if value == "" {
		return ""
	}

	var builder strings.Builder

	for _, character := range value {
		switch {
		case character >= 'a' &&
			character <= 'z':
			builder.WriteRune(character)

		case character >= 'A' &&
			character <= 'Z':
			builder.WriteRune(character)

		case character >= '0' &&
			character <= '9':
			builder.WriteRune(character)

		case character == '-',
			character == '_',
			character == '.':
			builder.WriteRune(character)

		default:
			builder.WriteByte('-')
		}
	}

	return strings.Trim(
		builder.String(),
		"-.",
	)
}

func encodeTags(
	tags map[string]string,
) string {
	if len(tags) == 0 {
		return ""
	}

	parts := make(
		[]string,
		0,
		len(tags),
	)

	for key, value := range tags {
		parts = append(
			parts,
			urlQueryEscape(key)+
				"="+
				urlQueryEscape(value),
		)
	}

	// Deterministic ordering improves tests and signatures/debugging.
	sortStrings(parts)

	return strings.Join(
		parts,
		"&",
	)
}

func urlQueryEscape(
	value string,
) string {
	// This implementation performs the small percent-encoding subset needed
	// by object tags without coupling object key construction to URL paths.
	var builder strings.Builder

	const hexadecimal = "0123456789ABCDEF"

	for i := 0; i < len(value); i++ {
		character := value[i]

		switch {
		case character >= 'a' &&
			character <= 'z',
			character >= 'A' &&
				character <= 'Z',
			character >= '0' &&
				character <= '9',
			character == '-',
			character == '_',
			character == '.',
			character == '~':
			builder.WriteByte(character)

		default:
			builder.WriteByte('%')
			builder.WriteByte(
				hexadecimal[
					character>>4,
				],
			)
			builder.WriteByte(
				hexadecimal[
					character&0x0f,
				],
			)
		}
	}

	return builder.String()
}

func sortStrings(
	values []string,
) {
	// Insertion sort is sufficient here because static object-tag sets are
	// expected to remain small and avoids another package-level dependency.
	for i := 1; i < len(values); i++ {
		current := values[i]
		position := i - 1

		for position >= 0 &&
			values[position] > current {
			values[position+1] =
				values[position]

			position--
		}

		values[position+1] = current
	}
}

func classifyStoreError(
	operation string,
	err error,
) *StoreError {
	if err == nil {
		return nil
	}

	if errors.Is(err, context.Canceled) {
		return &StoreError{
			Operation: operation,
			Code:      "context_cancelled",
			Message:   err.Error(),
			Retryable: false,
			Cause:     err,
		}
	}

	if errors.Is(
		err,
		context.DeadlineExceeded,
	) {
		return &StoreError{
			Operation: operation,
			Code:      "deadline_exceeded",
			Message:   err.Error(),
			Retryable: false,
			Cause:     err,
		}
	}

	var apiErr smithy.APIError

	if errors.As(err, &apiErr) {
		code := strings.TrimSpace(
			apiErr.ErrorCode(),
		)

		message := strings.TrimSpace(
			apiErr.ErrorMessage(),
		)

		return &StoreError{
			Operation: operation,
			Code:      code,
			Message:   message,

			Retryable: retryableStoreCode(
				code,
			),

			Cause: err,
		}
	}

	var netErr interface {
		Timeout() bool
		Temporary() bool
	}

	if errors.As(err, &netErr) {
		return &StoreError{
			Operation: operation,
			Code:      "network_error",
			Message:   err.Error(),
			Retryable: true,
			Cause:     err,
		}
	}

	return &StoreError{
		Operation: operation,
		Code:      "storage_error",
		Message:   err.Error(),
		Retryable: false,
		Cause:     err,
	}
}

func retryableStoreCode(
	code string,
) bool {
	switch strings.ToLower(
		strings.TrimSpace(code),
	) {
	case "requesttimeout",
		"requesttimeoutexception",
		"slowdown",
		"throttling",
		"throttlingexception",
		"serviceunavailable",
		"internalerror",
		"internalfailure":
		return true

	default:
		return false
	}
}

func storeErrorCode(
	err *StoreError,
) string {
	if err == nil {
		return "objectstore_error"
	}

	code := strings.ToLower(
		strings.TrimSpace(err.Code),
	)

	if code == "" {
		return "objectstore_error"
	}

	var builder strings.Builder

	for _, character := range code {
		switch {
		case character >= 'a' &&
			character <= 'z',
			character >= '0' &&
				character <= '9':
			builder.WriteRune(character)

		default:
			builder.WriteByte('_')
		}
	}

	return "objectstore_" +
		strings.Trim(
			builder.String(),
			"_",
		)
}

func withUploadTimeout(
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

func minimumDuration(
	left time.Duration,
	right time.Duration,
) time.Duration {
	if left < right {
		return left
	}

	return right
}

func storeFailedResult(
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

// Disable verbose provider SDK logging by default.
//
// OTGH should expose its own structured delivery logs rather than allowing
// provider libraries to emit request details that might contain environment
// or endpoint information.
type discardSDKLogger struct{}

func (discardSDKLogger) Logf(
	classification logging.Classification,
	format string,
	v ...interface{},
) {
}

// Compile-time interface assertions.
var (
	_ export.Exporter =
		(*Exporter)(nil)

	_ export.RetryClassifier =
		(*Exporter)(nil)
)
