package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/samorca/otgh/internal/export"
	exportgrpc "github.com/samorca/otgh/internal/export/grpc"
	exporthttp "github.com/samorca/otgh/internal/export/http"
	"github.com/samorca/otgh/internal/export/objectstore"
	"github.com/samorca/otgh/internal/export/protocol"
	"github.com/samorca/otgh/internal/export/webhook"
)

const (
	// CurrentConfigVersion identifies the application configuration schema.
	CurrentConfigVersion = 1

	// DefaultConfigFile is the conventional OTGH configuration file.
	DefaultConfigFile = "otgh.json"
)

var (
	ErrInvalidConfig = errors.New(
		"invalid OTGH configuration",
	)

	ErrUnsupportedConfigVersion = errors.New(
		"unsupported OTGH configuration version",
	)
)

// ApplicationConfig controls process-level OTGH behavior.
type ApplicationConfig struct {
	// NodeID uniquely identifies this OTGH instance.
	NodeID string `json:"node_id"`

	// ShutdownTimeout limits graceful process shutdown.
	ShutdownTimeout time.Duration `json:"shutdown_timeout"`

	// WorkerCount is the number of concurrent export workers.
	WorkerCount int `json:"worker_count"`
}

// Validate verifies application-level settings.
func (c ApplicationConfig) Validate() error {
	if strings.TrimSpace(c.NodeID) == "" {
		return errors.New(
			"application node_id is required",
		)
	}

	if c.ShutdownTimeout <= 0 {
		return errors.New(
			"application shutdown_timeout must be greater than zero",
		)
	}

	if c.WorkerCount <= 0 {
		return errors.New(
			"application worker_count must be greater than zero",
		)
	}

	return nil
}

// BatchingConfig represents application-level batching settings.
//
// It is converted into export.BatcherConfig when runtime components are
// created.
type BatchingConfig struct {
	MaxRecords int `json:"max_records"`

	MaxPayloadBytes int64 `json:"max_payload_bytes"`
}

// Validate verifies batching configuration.
func (c BatchingConfig) Validate() error {
	return c.ExportConfig().Validate()
}

// ExportConfig converts application configuration into the export package
// representation.
func (c BatchingConfig) ExportConfig() export.BatcherConfig {
	return export.BatcherConfig{
		MaxRecords:      c.MaxRecords,
		MaxPayloadBytes: c.MaxPayloadBytes,
	}
}

// QueueConfig represents application queue settings.
type QueueConfig struct {
	Capacity int `json:"capacity"`
}

// Validate verifies queue configuration.
func (c QueueConfig) Validate() error {
	return c.ExportConfig().Validate()
}

// ExportConfig converts queue settings into the export package type.
func (c QueueConfig) ExportConfig() export.QueueConfig {
	return export.QueueConfig{
		Capacity: c.Capacity,
	}
}

// RetryConfig stores retry values in application configuration.
type RetryConfig struct {
	MaxAttempts int `json:"max_attempts"`

	InitialDelay time.Duration `json:"initial_delay"`

	MaxDelay time.Duration `json:"max_delay"`

	Multiplier float64 `json:"multiplier"`

	Jitter float64 `json:"jitter"`

	MaxElapsedTime time.Duration `json:"max_elapsed_time"`
}

// Validate verifies retry configuration.
func (c RetryConfig) Validate() error {
	return c.ExportConfig().Validate()
}

// ExportConfig converts application values into export.RetryConfig.
func (c RetryConfig) ExportConfig() export.RetryConfig {
	return export.RetryConfig{
		MaxAttempts: c.MaxAttempts,

		InitialDelay: c.InitialDelay,
		MaxDelay:     c.MaxDelay,

		Multiplier: c.Multiplier,
		Jitter:     c.Jitter,

		MaxElapsedTime: c.MaxElapsedTime,
	}
}

// SpoolConfig describes durable local spool settings.
type SpoolConfig struct {
	Directory string `json:"directory"`

	DirectoryMode uint32 `json:"directory_mode"`

	FileMode uint32 `json:"file_mode"`

	MaxEntries int `json:"max_entries"`

	MaxBytes int64 `json:"max_bytes"`

	MaxEntryBytes int64 `json:"max_entry_bytes"`

	SyncDirectory bool `json:"sync_directory"`
}

// ExportConfig converts application values into export.SpoolConfig.
func (c SpoolConfig) ExportConfig() export.SpoolConfig {
	return export.SpoolConfig{
		Directory: c.Directory,

		DirectoryMode: os.FileMode(
			c.DirectoryMode,
		),

		FileMode: os.FileMode(
			c.FileMode,
		),

		MaxEntries: c.MaxEntries,

		MaxBytes: c.MaxBytes,

		MaxEntryBytes: c.MaxEntryBytes,

		SyncDirectory: c.SyncDirectory,
	}
}

// Validate verifies spool settings.
func (c SpoolConfig) Validate() error {
	return c.ExportConfig().Validate()
}

// ExportersConfig contains every configured export destination.
type ExportersConfig struct {
	HTTP []exporthttp.Config `json:"http,omitempty"`

	GRPC []exportgrpc.Config `json:"grpc,omitempty"`

	ObjectStore []objectstore.Config `json:"object_store,omitempty"`

	Webhook []webhook.Config `json:"webhook,omitempty"`
}

// Count returns the total number of configured exporters.
func (c ExportersConfig) Count() int {
	return len(c.HTTP) +
		len(c.GRPC) +
		len(c.ObjectStore) +
		len(c.Webhook)
}

// Validate verifies every exporter and ensures exporter IDs are globally
// unique.
func (c ExportersConfig) Validate() error {
	ids := make(
		map[string]string,
		c.Count(),
	)

	registerID := func(
		id string,
		exporterType string,
	) error {
		id = strings.TrimSpace(id)

		if id == "" {
			return fmt.Errorf(
				"%s exporter has empty id",
				exporterType,
			)
		}

		if existing, exists := ids[id]; exists {
			return fmt.Errorf(
				"exporter id %q is used by both %s and %s exporters",
				id,
				existing,
				exporterType,
			)
		}

		ids[id] = exporterType

		return nil
	}

	for index, item := range c.HTTP {
		normalized := item.Normalized()

		if err := normalized.Validate(); err != nil {
			return fmt.Errorf(
				"HTTP exporter %d: %w",
				index,
				err,
			)
		}

		if err := registerID(
			normalized.ID,
			"http",
		); err != nil {
			return err
		}
	}

	for index, item := range c.GRPC {
		normalized := item.Normalized()

		if err := normalized.Validate(); err != nil {
			return fmt.Errorf(
				"gRPC exporter %d: %w",
				index,
				err,
			)
		}

		if err := registerID(
			normalized.ID,
			"grpc",
		); err != nil {
			return err
		}
	}

	for index, item := range c.ObjectStore {
		normalized := item.Normalized()

		if err := normalized.Validate(); err != nil {
			return fmt.Errorf(
				"object-store exporter %d: %w",
				index,
				err,
			)
		}

		if err := registerID(
			normalized.ID,
			"objectstore",
		); err != nil {
			return err
		}
	}

	for index, item := range c.Webhook {
		normalized := item.Normalized()

		if err := normalized.Validate(); err != nil {
			return fmt.Errorf(
				"webhook exporter %d: %w",
				index,
				err,
			)
		}

		if err := registerID(
			normalized.ID,
			"webhook",
		); err != nil {
			return err
		}
	}

	return nil
}

// RouteConfig is the serializable representation of export.Route.
type RouteConfig struct {
	Name string `json:"name"`

	RecordTypes []export.RecordType `json:"record_types,omitempty"`

	MinPriority export.Priority `json:"min_priority"`

	TenantID export.TenantID `json:"tenant_id,omitempty"`

	Metadata map[string]string `json:"metadata,omitempty"`

	DestinationIDs []export.DestinationID `json:"destination_ids"`
}

// ExportRoute converts the serializable form into export.Route.
func (c RouteConfig) ExportRoute() export.Route {
	route := export.Route{
		Name: c.Name,

		RecordTypes: append(
			[]export.RecordType(nil),
			c.RecordTypes...,
		),

		MinPriority: c.MinPriority,

		TenantID: c.TenantID,

		DestinationIDs: append(
			[]export.DestinationID(nil),
			c.DestinationIDs...,
		),
	}

	if c.Metadata != nil {
		route.Metadata = make(
			map[string]string,
			len(c.Metadata),
		)

		for key, value := range c.Metadata {
			route.Metadata[key] = value
		}
	}

	return route
}

// Config is the complete OTGH application configuration.
type Config struct {
	Version int `json:"version"`

	Application ApplicationConfig `json:"application"`

	Batching BatchingConfig `json:"batching"`

	Queue QueueConfig `json:"queue"`

	Retry RetryConfig `json:"retry"`

	Spool SpoolConfig `json:"spool"`

	Protocol protocol.ProtocolConfig `json:"protocol"`

	Exporters ExportersConfig `json:"exporters"`

	Routes []RouteConfig `json:"routes,omitempty"`
}

// Default returns production-oriented baseline values.
//
// Destination URLs, credentials, signing secrets, buckets, and routes are not
// invented here. Those must be explicitly supplied by deployment
// configuration.
func Default() Config {
	nodeID := "otgh-node"

	defaultSpool :=
		export.DefaultSpoolConfig(
			"./data/spool",
		)

	return Config{
		Version: CurrentConfigVersion,

		Application: ApplicationConfig{
			NodeID:          nodeID,
			ShutdownTimeout: 30 * time.Second,
			WorkerCount:     4,
		},

		Batching: BatchingConfig{
			MaxRecords:      500,
			MaxPayloadBytes: 8 * 1024 * 1024,
		},

		Queue: QueueConfig{
			Capacity: 2048,
		},

		Retry: RetryConfig{
			MaxAttempts: 5,

			InitialDelay: time.Second,

			MaxDelay: 30 * time.Second,

			Multiplier: 2,

			Jitter: 0.2,

			MaxElapsedTime: 5 * time.Minute,
		},

		Spool: SpoolConfig{
			Directory: defaultSpool.Directory,

			DirectoryMode: uint32(
				defaultSpool.DirectoryMode,
			),

			FileMode: uint32(
				defaultSpool.FileMode,
			),

			MaxEntries: defaultSpool.MaxEntries,

			MaxBytes: defaultSpool.MaxBytes,

			MaxEntryBytes: defaultSpool.
				MaxEntryBytes,

			SyncDirectory: defaultSpool.
				SyncDirectory,
		},

		Protocol: protocol.DefaultProtocolConfig(
			nodeID,
		),

		Exporters: ExportersConfig{},

		Routes: nil,
	}
}

// Validate verifies the complete configuration.
func (c Config) Validate() error {
	if c.Version != CurrentConfigVersion {
		return fmt.Errorf(
			"%w: got %d, supported %d",
			ErrUnsupportedConfigVersion,
			c.Version,
			CurrentConfigVersion,
		)
	}

	if err := c.Application.Validate(); err != nil {
		return fmt.Errorf(
			"%w: application: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if err := c.Batching.Validate(); err != nil {
		return fmt.Errorf(
			"%w: batching: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if err := c.Queue.Validate(); err != nil {
		return fmt.Errorf(
			"%w: queue: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if err := c.Retry.Validate(); err != nil {
		return fmt.Errorf(
			"%w: retry: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if err := c.Spool.Validate(); err != nil {
		return fmt.Errorf(
			"%w: spool: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if c.Protocol.NodeID !=
		c.Application.NodeID {
		return fmt.Errorf(
			"%w: protocol node_id %q does not match application node_id %q",
			ErrInvalidConfig,
			c.Protocol.NodeID,
			c.Application.NodeID,
		)
	}

	if err := c.Protocol.Validate(); err != nil {
		return fmt.Errorf(
			"%w: protocol: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if err := c.Exporters.Validate(); err != nil {
		return fmt.Errorf(
			"%w: exporters: %v",
			ErrInvalidConfig,
			err,
		)
	}

	if err := c.validateRoutes(); err != nil {
		return fmt.Errorf(
			"%w: routes: %v",
			ErrInvalidConfig,
			err,
		)
	}

	return nil
}

func (c Config) validateRoutes() error {
	exporterIDs := make(
		map[export.DestinationID]struct{},
		c.Exporters.Count(),
	)

	for _, item := range c.Exporters.HTTP {
		exporterIDs[
			export.DestinationID(
				item.ID,
			),
		] = struct{}{}
	}

	for _, item := range c.Exporters.GRPC {
		exporterIDs[
			export.DestinationID(
				item.ID,
			),
		] = struct{}{}
	}

	for _, item := range c.Exporters.ObjectStore {
		exporterIDs[
			export.DestinationID(
				item.ID,
			),
		] = struct{}{}
	}

	for _, item := range c.Exporters.Webhook {
		exporterIDs[
			export.DestinationID(
				item.ID,
			),
		] = struct{}{}
	}

	names := make(
		map[string]struct{},
		len(c.Routes),
	)

	for index, routeConfig := range c.Routes {
		route := routeConfig.ExportRoute()

		if err := route.Validate(); err != nil {
			return fmt.Errorf(
				"route %d: %w",
				index,
				err,
			)
		}

		if _, exists := names[route.Name]; exists {
			return fmt.Errorf(
				"duplicate route name %q",
				route.Name,
			)
		}

		names[route.Name] = struct{}{}

		for _, destinationID :=
			range route.DestinationIDs {
			if _, exists :=
				exporterIDs[destinationID]; !exists {
				return fmt.Errorf(
					"route %q references unknown exporter %q",
					route.Name,
					destinationID,
				)
			}
		}
	}

	return nil
}

// RoutesForExport returns defensive export.Route values.
func (c Config) RoutesForExport() []export.Route {
	if len(c.Routes) == 0 {
		return nil
	}

	routes := make(
		[]export.Route,
		len(c.Routes),
	)

	for index, route :=
		range c.Routes {
		routes[index] =
			route.ExportRoute()
	}

	return routes
}

// LoadFile loads configuration from a JSON file.
//
// The loader rejects unknown fields and trailing JSON values before complete
// validation.
func LoadFile(
	filename string,
) (Config, error) {
	filename = strings.TrimSpace(
		filename,
	)

	if filename == "" {
		return Config{}, errors.New(
			"configuration filename is required",
		)
	}

	data, err := os.ReadFile(filename)
	if err != nil {
		return Config{}, fmt.Errorf(
			"read configuration file %q: %w",
			filename,
			err,
		)
	}

	config, err := Parse(data)
	if err != nil {
		return Config{}, fmt.Errorf(
			"parse configuration file %q: %w",
			filename,
			err,
		)
	}

	return config, nil
}

// Parse strictly parses one complete JSON configuration document.
func Parse(
	data []byte,
) (Config, error) {
	if len(bytes.TrimSpace(data)) == 0 {
		return Config{}, errors.New(
			"configuration is empty",
		)
	}

	decoder := json.NewDecoder(
		bytes.NewReader(data),
	)

	decoder.DisallowUnknownFields()

	config := Default()

	if err := decoder.Decode(
		&config,
	); err != nil {
		return Config{}, fmt.Errorf(
			"decode configuration JSON: %w",
			err,
		)
	}

	var extra json.RawMessage

	err := decoder.Decode(&extra)

	if !errors.Is(err, io.EOF) {
		if err == nil {
			return Config{}, errors.New(
				"configuration contains multiple JSON documents",
			)
		}

		return Config{}, fmt.Errorf(
			"configuration contains trailing data: %w",
			err,
		)
	}

	applyEnvironment(
		&config,
	)

	normalize(&config)

	if err := config.Validate(); err != nil {
		return Config{}, err
	}

	return config, nil
}

// Load loads the default configuration file path.
//
// OTGH_CONFIG may override the filename.
func Load() (Config, error) {
	filename := strings.TrimSpace(
		os.Getenv("OTGH_CONFIG"),
	)

	if filename == "" {
		filename = DefaultConfigFile
	}

	return LoadFile(filename)
}

// normalize applies stable runtime normalization after parsing/environment
// application.
func normalize(
	config *Config,
) {
	if config == nil {
		return
	}

	config.Application.NodeID =
		strings.TrimSpace(
			config.Application.NodeID,
		)

	config.Spool.Directory =
		filepath.Clean(
			config.Spool.Directory,
		)

	config.Protocol.NodeID =
		config.Application.NodeID

	config.Protocol.Handshake.NodeID =
		config.Application.NodeID

	for index := range config.Exporters.HTTP {
		config.Exporters.HTTP[index] =
			config.Exporters.HTTP[
				index,
			].Normalized()
	}

	for index := range config.Exporters.GRPC {
		config.Exporters.GRPC[index] =
			config.Exporters.GRPC[
				index,
			].Normalized()
	}

	for index :=
		range config.Exporters.ObjectStore {
		config.Exporters.ObjectStore[index] =
			config.Exporters.ObjectStore[
				index,
			].Normalized()
	}

	for index :=
		range config.Exporters.Webhook {
		config.Exporters.Webhook[index] =
			config.Exporters.Webhook[
				index,
			].Normalized()
	}
}

// applyEnvironment applies deployment-level overrides.
//
// Only application/runtime values belong here. Exporter credentials and
// signing secrets should instead be wired from a dedicated secret-loading
// layer rather than creating an ever-growing environment parser in config.go.
func applyEnvironment(
	config *Config,
) {
	if config == nil {
		return
	}

	if value := strings.TrimSpace(
		os.Getenv("OTGH_NODE_ID"),
	); value != "" {
		config.Application.NodeID =
			value
	}

	if value := strings.TrimSpace(
		os.Getenv("OTGH_SPOOL_DIR"),
	); value != "" {
		config.Spool.Directory =
			value
	}

	if value := strings.TrimSpace(
		os.Getenv("OTGH_WORKERS"),
	); value != "" {
		if parsed, err :=
			strconv.Atoi(value); err == nil &&
			parsed > 0 {
			config.Application.WorkerCount =
				parsed
		}
	}

	if endpoint := strings.TrimSpace(
		os.Getenv("OTGH_OSHA_ENDPOINT"),
	); endpoint != "" {
		configureOSHAExporter(config, endpoint)
	}
}

func configureOSHAExporter(config *Config, endpoint string) {
	var exporter exporthttp.Config
	found := false
	for index := range config.Exporters.HTTP {
		if config.Exporters.HTTP[index].ID == "osha" {
			exporter = config.Exporters.HTTP[index]
			found = true
			break
		}
	}
	if !found {
		exporter = exporthttp.DefaultConfig()
		exporter.ID = "osha"
	}
	exporter.Endpoint = endpoint
	if strings.HasPrefix(strings.ToLower(endpoint), "http://") {
		exporter.TLS.Enabled = false
	}
	if token := strings.TrimSpace(os.Getenv("OTGH_OSHA_TOKEN")); token != "" {
		exporter.Authentication = exporthttp.AuthenticationConfig{
			Type:       exporthttp.AuthenticationBearer,
			BearerToken: token,
		}
	}
	if found {
		for index := range config.Exporters.HTTP {
			if config.Exporters.HTTP[index].ID == "osha" {
				config.Exporters.HTTP[index] = exporter
				break
			}
		}
	} else {
		config.Exporters.HTTP = append(config.Exporters.HTTP, exporter)
	}

	routeFound := false
	for index := range config.Routes {
		if config.Routes[index].Name == "osha-analysis" {
			config.Routes[index].DestinationIDs = []export.DestinationID{"osha"}
			routeFound = true
			break
		}
	}
	if !routeFound {
		config.Routes = append(config.Routes, RouteConfig{
			Name:           "osha-analysis",
			DestinationIDs: []export.DestinationID{"osha"},
		})
	}
}
