package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/samorca/otgh/internal/config"
	"github.com/samorca/otgh/internal/export"
)

func TestSubmitRecordsRoutesBatchesAndQueues(t *testing.T) {
	cfg := config.Default()
	cfg.Spool.Directory = t.TempDir()
	cfg.Batching.MaxRecords = 2

	runtime, err := NewRuntime(cfg)
	if err != nil {
		t.Fatalf("NewRuntime() error = %v", err)
	}
	defer runtime.ForceShutdown()

	destination := export.Destination{
		ID:       "osha",
		Type:     "http",
		Endpoint: "http://osha-api/api/v1/otgh/batches",
	}
	if err := runtime.Router().RegisterDestination(destination); err != nil {
		t.Fatalf("RegisterDestination() error = %v", err)
	}
	if err := runtime.Router().ReplaceRoutes([]export.Route{{
		Name:           "all-to-osha",
		DestinationIDs: []export.DestinationID{destination.ID},
	}}); err != nil {
		t.Fatalf("ReplaceRoutes() error = %v", err)
	}

	createdAt := time.Now().UTC()
	records := make([]export.Record, 3)
	for index := range records {
		records[index] = export.Record{
			ID:          export.RecordID(fmt.Sprintf("record-%d", index+1)),
			Type:        export.RecordTypeEvent,
			Priority:    export.PriorityNormal,
			Source:      export.Source{System: "test"},
			ContentType: export.ContentTypeJSON,
			Compression: export.CompressionNone,
			CreatedAt:   createdAt,
			Payload:     []byte(fmt.Sprintf("{\"index\":%d}", index)),
		}
	}

	accepted, err := runtime.SubmitRecords(context.Background(), records)
	if err != nil {
		t.Fatalf("SubmitRecords() error = %v", err)
	}
	if accepted != len(records) {
		t.Fatalf("SubmitRecords() accepted = %d, want %d", accepted, len(records))
	}

	batches := runtime.Queue().Snapshot()
	if len(batches) != 2 {
		t.Fatalf("queued batches = %d, want 2", len(batches))
	}
	if batches[0].RecordCount() != 2 || batches[1].RecordCount() != 1 {
		t.Fatalf(
			"batch record counts = %d, %d; want 2, 1",
			batches[0].RecordCount(),
			batches[1].RecordCount(),
		)
	}
	if batches[0].Destination.ID != destination.ID || batches[1].Destination.ID != destination.ID {
		t.Fatal("records were queued for an unexpected destination")
	}
}

func TestIngestHTTPServerAuthenticatesAndQueuesRecords(t *testing.T) {
	cfg := config.Default()
	cfg.Spool.Directory = t.TempDir()
	runtime, err := NewRuntime(cfg)
	if err != nil {
		t.Fatalf("NewRuntime() error = %v", err)
	}
	defer runtime.ForceShutdown()

	destination := export.Destination{ID: "osha", Type: "http"}
	if err := runtime.Router().RegisterDestination(destination); err != nil {
		t.Fatalf("RegisterDestination() error = %v", err)
	}
	if err := runtime.Router().ReplaceRoutes([]export.Route{{
		Name:           "all-to-osha",
		DestinationIDs: []export.DestinationID{destination.ID},
	}}); err != nil {
		t.Fatalf("ReplaceRoutes() error = %v", err)
	}
	if err := runtime.beginStart(); err != nil {
		t.Fatalf("beginStart() error = %v", err)
	}
	if err := runtime.markRunning(); err != nil {
		t.Fatalf("markRunning() error = %v", err)
	}

	server, err := NewIngestHTTPServer(runtime, "127.0.0.1:0", "test-token")
	if err != nil {
		t.Fatalf("NewIngestHTTPServer() error = %v", err)
	}

	unauthorized := httptest.NewRecorder()
	unauthorizedRequest := httptest.NewRequest(
		http.MethodPost,
		"/v1/records",
		strings.NewReader(`{"records":[]}`),
	)
	server.Handler.ServeHTTP(unauthorized, unauthorizedRequest)
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d, want %d", unauthorized.Code, http.StatusUnauthorized)
	}

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/records",
		strings.NewReader(`{"records":[{"type":"event","payload":{"message":"disk full"}}]}`),
	)
	request.Header.Set("Authorization", "Bearer test-token")
	response := httptest.NewRecorder()
	server.Handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("ingest status = %d, body=%s", response.Code, response.Body.String())
	}

	var result ingestResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.AcceptedRecords != 1 || result.RejectedRecords != 0 {
		t.Fatalf("ingest result = %+v, want one accepted record", result)
	}
	if queued := runtime.Queue().Snapshot(); len(queued) != 1 || queued[0].RecordCount() != 1 {
		t.Fatalf("queued batches = %+v, want one one-record batch", queued)
	}
}