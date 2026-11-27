package v1

import (
	"bytes"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/samorca/otgh/internal/export"
)

const (
	defaultIngestAddress  = ":8082"
	maxIngestBodyBytes    = 128 * 1024
	maxIngestRecordCount  = 256
	defaultIngestPriority = "normal"
)

type recordIngestRequest struct {
	Records []recordIngestItem `json:"records"`
}

type recordIngestItem struct {
	ID            string            `json:"id,omitempty"`
	TenantID      string            `json:"tenant_id,omitempty"`
	CorrelationID string            `json:"correlation_id,omitempty"`
	Type          export.RecordType `json:"type"`
	Priority      string            `json:"priority,omitempty"`
	Source        export.Source     `json:"source,omitempty"`
	Resource      *export.ResourceRef `json:"resource,omitempty"`
	ContentType   export.ContentType `json:"content_type,omitempty"`
	Compression   export.Compression `json:"compression,omitempty"`
	CreatedAt     time.Time          `json:"created_at,omitempty"`
	Payload       json.RawMessage    `json:"payload"`
	Metadata      map[string]string  `json:"metadata,omitempty"`
}

type ingestResponse struct {
	AcceptedRecords int    `json:"accepted_records"`
	RejectedRecords int    `json:"rejected_records"`
	Retryable       bool   `json:"retryable"`
	Message         string `json:"message"`
}

// NewIngestHTTPServer creates the OTGH producer API and health endpoints.
func NewIngestHTTPServer(
	runtime *Runtime,
	address string,
	bearerToken string,
) (*http.Server, error) {
	if runtime == nil {
		return nil, errors.New("OTGH runtime is required")
	}
	address = strings.TrimSpace(address)
	if address == "" {
		address = defaultIngestAddress
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health/live", func(writer http.ResponseWriter, _ *http.Request) {
		writeJSON(writer, http.StatusOK, map[string]string{"status": "live"})
	})
	mux.HandleFunc("GET /health/ready", func(writer http.ResponseWriter, _ *http.Request) {
		if !runtime.Running() {
			writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"status": "not_ready"})
			return
		}
		writeJSON(writer, http.StatusOK, map[string]string{"status": "ready"})
	})
	mux.Handle("POST /v1/records", recordIngestHandler(runtime, bearerToken))

	return &http.Server{
		Addr:              address,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    16 * 1024,
	}, nil
}

func recordIngestHandler(runtime *Runtime, bearerToken string) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if bearerToken != "" && !authorized(request, bearerToken) {
			writeJSON(writer, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
		if !runtime.Running() {
			writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"error": "OTGH is not ready"})
			return
		}

		request.Body = http.MaxBytesReader(writer, request.Body, maxIngestBodyBytes)
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		var payload recordIngestRequest
		if err := decoder.Decode(&payload); err != nil {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				writeJSON(writer, http.StatusRequestEntityTooLarge, map[string]string{"error": "request body is too large"})
				return
			}
			writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
			return
		}
		var extra any
		if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
			writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "request must contain one JSON document"})
			return
		}
		if len(payload.Records) == 0 || len(payload.Records) > maxIngestRecordCount {
			writeJSON(writer, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("records must contain between 1 and %d items", maxIngestRecordCount)})
			return
		}

		records := make([]export.Record, 0, len(payload.Records))
		var totalPayloadBytes int64
		for index, item := range payload.Records {
			if strings.TrimSpace(item.ID) == "" {
				recordID, idErr := newRecordID()
				if idErr != nil {
					writeJSON(writer, http.StatusInternalServerError, map[string]string{"error": "could not generate record ID"})
					return
				}
				item.ID = recordID
			}
			record, err := normalizeIngestRecord(item)
			if err != nil {
				writeJSON(writer, http.StatusUnprocessableEntity, map[string]string{"error": fmt.Sprintf("record %d: %v", index, err)})
				return
			}
			totalPayloadBytes += int64(len(record.Payload))
			records = append(records, record)
		}
		if totalPayloadBytes > runtime.Config().Batching.MaxPayloadBytes {
			writeJSON(writer, http.StatusRequestEntityTooLarge, map[string]string{"error": "record payloads exceed OTGH batch capacity"})
			return
		}

		accepted, err := runtime.SubmitRecords(request.Context(), records)
		if err != nil {
			writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(writer, http.StatusAccepted, ingestResponse{
			AcceptedRecords: accepted,
			RejectedRecords: 0,
			Retryable:       false,
			Message:         "records queued for delivery",
		})
	})
}

func normalizeIngestRecord(item recordIngestItem) (export.Record, error) {
	payload := bytes.TrimSpace(item.Payload)
	if len(payload) == 0 || !json.Valid(payload) {
		return export.Record{}, errors.New("payload must be a non-empty JSON value")
	}
	priority, err := parsePriority(item.Priority)
	if err != nil {
		return export.Record{}, err
	}
	if item.ContentType == "" {
		item.ContentType = export.ContentTypeJSON
	}
	if item.Compression == "" {
		item.Compression = export.CompressionNone
	}
	if item.CreatedAt.IsZero() {
		item.CreatedAt = time.Now().UTC()
	}
	if strings.TrimSpace(item.Source.System) == "" {
		item.Source.System = "otgh"
	}
	if strings.TrimSpace(item.Source.Component) == "" {
		item.Source.Component = "ingest"
	}

	record := export.Record{
		ID:            export.RecordID(item.ID),
		TenantID:      export.TenantID(item.TenantID),
		CorrelationID: export.CorrelationID(item.CorrelationID),
		Type:          item.Type,
		Priority:      priority,
		Source:        item.Source,
		Resource:      item.Resource,
		ContentType:   item.ContentType,
		Compression:   item.Compression,
		CreatedAt:     item.CreatedAt,
		Payload:       append([]byte(nil), payload...),
		Metadata:      item.Metadata,
	}
	if err := record.Validate(); err != nil {
		return export.Record{}, err
	}
	return record, nil
}

func newRecordID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return "otgh-" + hex.EncodeToString(value[:]), nil
}

func parsePriority(value string) (export.Priority, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", defaultIngestPriority:
		return export.PriorityNormal, nil
	case "bulk":
		return export.PriorityBulk, nil
	case "high":
		return export.PriorityHigh, nil
	case "critical":
		return export.PriorityCritical, nil
	case "control":
		return export.PriorityControl, nil
	default:
		return 0, fmt.Errorf("unsupported priority %q", value)
	}
}

func authorized(request *http.Request, token string) bool {
	header := strings.TrimSpace(request.Header.Get("Authorization"))
	if !strings.HasPrefix(header, "Bearer ") {
		return false
	}
	provided := strings.TrimSpace(strings.TrimPrefix(header, "Bearer "))
	return len(provided) == len(token) && subtle.ConstantTimeCompare([]byte(provided), []byte(token)) == 1
}

func writeJSON(writer http.ResponseWriter, statusCode int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(statusCode)
	_ = json.NewEncoder(writer).Encode(value)
}