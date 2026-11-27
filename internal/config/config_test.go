package config

import (
	"encoding/json"
	"testing"

	exporthttp "github.com/samorca/otgh/internal/export/http"
)

func TestParseAddsOSHAExporterAndRouteFromEnvironment(t *testing.T) {
	t.Setenv("OTGH_OSHA_ENDPOINT", "http://osha-api:8088/api/v1/otgh/batches")
	t.Setenv("OTGH_OSHA_TOKEN", "test-token")

	data, err := json.Marshal(Default())
	if err != nil {
		t.Fatalf("marshal default config: %v", err)
	}

	config, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if len(config.Exporters.HTTP) != 1 {
		t.Fatalf("HTTP exporters = %d, want 1", len(config.Exporters.HTTP))
	}
	if got := config.Exporters.HTTP[0]; got.ID != "osha" || got.Endpoint != "http://osha-api:8088/api/v1/otgh/batches" {
		t.Fatalf("OSHA exporter = %+v", got)
	}
	if config.Exporters.HTTP[0].TLS.Enabled {
		t.Fatal("HTTP OSHA endpoint should disable TLS")
	}
	if config.Exporters.HTTP[0].Authentication.Type != exporthttp.AuthenticationBearer ||
		config.Exporters.HTTP[0].Authentication.BearerToken != "test-token" {
		t.Fatal("OSHA bearer token was not applied")
	}
	if len(config.Routes) != 1 || config.Routes[0].Name != "osha-analysis" ||
		len(config.Routes[0].DestinationIDs) != 1 || config.Routes[0].DestinationIDs[0] != "osha" {
		t.Fatalf("OSHA routes = %+v", config.Routes)
	}
}