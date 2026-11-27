package v1

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/samorca/otgh/internal/export"
	exportgrpc "github.com/samorca/otgh/internal/export/grpc"
	exporthttp "github.com/samorca/otgh/internal/export/http"
	"github.com/samorca/otgh/internal/export/objectstore"
	"github.com/samorca/otgh/internal/export/webhook"
)

// ConfigureExporters constructs and registers all exporters declared in the
// application configuration.
//
// Configuration order is preserved within each exporter family:
//
//   1. HTTP
//   2. gRPC
//   3. object store
//   4. webhook
//
// Routes are installed only after every exporter and destination has been
// registered successfully.
//
// If any step fails, all exporters created during this call are closed and
// removed from the registry.
func (r *Runtime) ConfigureExporters(
	ctx context.Context,
) (returnErr error) {
	if r == nil {
		return errors.New(
			"OTGH runtime is nil",
		)
	}

	if ctx == nil {
		return errors.New(
			"configure exporters context is nil",
		)
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if r.Closed() {
		return errors.New(
			"OTGH runtime is closed",
		)
	}

	if r.exporters == nil {
		return errors.New(
			"OTGH exporter registry is nil",
		)
	}

	if r.router == nil {
		return errors.New(
			"OTGH router is nil",
		)
	}

	if r.exporters.Count() != 0 {
		return errors.New(
			"OTGH exporters are already configured",
		)
	}

	// Any failure after the first successful registration must release all
	// exporters created by this call. CloseAll also removes them from the
	// registry.
	defer func() {
		if returnErr == nil {
			return
		}

		if closeErr := r.exporters.CloseAll(); closeErr != nil {
			returnErr = errors.Join(
				returnErr,
				fmt.Errorf(
					"rollback configured exporters: %w",
					closeErr,
				),
			)
		}
	}()

	if err := r.configureHTTPExporters(ctx); err != nil {
		return err
	}

	if err := r.configureGRPCExporters(ctx); err != nil {
		return err
	}

	if err := r.configureObjectStoreExporters(ctx); err != nil {
		return err
	}

	if err := r.configureWebhookExporters(ctx); err != nil {
		return err
	}

	routes := r.config.RoutesForExport()

	if err := r.router.ReplaceRoutes(routes); err != nil {
		return fmt.Errorf(
			"configure OTGH routes: %w",
			err,
		)
	}

	return nil
}

func (r *Runtime) configureHTTPExporters(
	ctx context.Context,
) error {
	for index, exporterConfig :=
		range r.config.Exporters.HTTP {
		if err := ctx.Err(); err != nil {
			return err
		}

		exporterConfig =
			exporterConfig.Normalized()

		instance, err :=
			exporthttp.NewExporter(
				exporterConfig,
			)
		if err != nil {
			return fmt.Errorf(
				"create HTTP exporter %d (%q): %w",
				index,
				exporterConfig.ID,
				err,
			)
		}

		destination := export.Destination{
			ID: export.DestinationID(
				exporterConfig.ID,
			),

			Type: instance.Type(),

			Endpoint: exporterConfig.Endpoint,

			Metadata: map[string]string{
				"transport": "http",
			},
		}

		if err := r.registerExporter(
			instance,
			destination,
		); err != nil {
			closeErr := instance.Close()

			return errors.Join(
				fmt.Errorf(
					"register HTTP exporter %q: %w",
					exporterConfig.ID,
					err,
				),
				closeErr,
			)
		}
	}

	return nil
}

func (r *Runtime) configureGRPCExporters(
	ctx context.Context,
) error {
	for index, exporterConfig :=
		range r.config.Exporters.GRPC {
		if err := ctx.Err(); err != nil {
			return err
		}

		exporterConfig =
			exporterConfig.Normalized()

		instance, err :=
			exportgrpc.NewExporter(
				exporterConfig,
			)
		if err != nil {
			return fmt.Errorf(
				"create gRPC exporter %d (%q): %w",
				index,
				exporterConfig.ID,
				err,
			)
		}

		destination := export.Destination{
			ID: export.DestinationID(
				exporterConfig.ID,
			),

			Type: instance.Type(),

			Endpoint: exporterConfig.Endpoint,

			Metadata: map[string]string{
				"transport": "grpc",
			},
		}

		if err := r.registerExporter(
			instance,
			destination,
		); err != nil {
			closeErr := instance.Close()

			return errors.Join(
				fmt.Errorf(
					"register gRPC exporter %q: %w",
					exporterConfig.ID,
					err,
				),
				closeErr,
			)
		}
	}

	return nil
}

func (r *Runtime) configureObjectStoreExporters(
	ctx context.Context,
) error {
	for index, exporterConfig :=
		range r.config.Exporters.ObjectStore {
		if err := ctx.Err(); err != nil {
			return err
		}

		exporterConfig =
			exporterConfig.Normalized()

		instance, err :=
			objectstore.NewExporter(
				ctx,
				exporterConfig,
			)
		if err != nil {
			return fmt.Errorf(
				"create object-store exporter %d (%q): %w",
				index,
				exporterConfig.ID,
				err,
			)
		}

		destination := export.Destination{
			ID: export.DestinationID(
				exporterConfig.ID,
			),

			Type: instance.Type(),

			Endpoint: exporterConfig.Endpoint,

			Metadata: map[string]string{
				"transport": "objectstore",
				"provider": string(
					exporterConfig.Provider,
				),
				"bucket": exporterConfig.Bucket,
				"region": exporterConfig.Region,
			},
		}

		if err := r.registerExporter(
			instance,
			destination,
		); err != nil {
			closeErr := instance.Close()

			return errors.Join(
				fmt.Errorf(
					"register object-store exporter %q: %w",
					exporterConfig.ID,
					err,
				),
				closeErr,
			)
		}
	}

	return nil
}

func (r *Runtime) configureWebhookExporters(
	ctx context.Context,
) error {
	for index, exporterConfig :=
		range r.config.Exporters.Webhook {
		if err := ctx.Err(); err != nil {
			return err
		}

		exporterConfig =
			exporterConfig.Normalized()

		idPrefix := strings.TrimSpace(
			exporterConfig.ID,
		)

		if idPrefix == "" {
			idPrefix = "webhook"
		}

		instance, err :=
			webhook.NewExporter(
				exporterConfig,
				webhook.
					NewTimestampDeliveryIDGenerator(
						idPrefix,
					),
			)
		if err != nil {
			return fmt.Errorf(
				"create webhook exporter %d (%q): %w",
				index,
				exporterConfig.ID,
				err,
			)
		}

		destination := export.Destination{
			ID: export.DestinationID(
				exporterConfig.ID,
			),

			Type: instance.Type(),

			Endpoint: exporterConfig.URL,

			Metadata: map[string]string{
				"transport": "webhook",
			},
		}

		if err := r.registerExporter(
			instance,
			destination,
		); err != nil {
			closeErr := instance.Close()

			return errors.Join(
				fmt.Errorf(
					"register webhook exporter %q: %w",
					exporterConfig.ID,
					err,
				),
				closeErr,
			)
		}
	}

	return nil
}

// registerExporter registers the live implementation and its router
// destination as one logical operation.
//
// If destination registration fails, the exporter is removed from the
// registry before returning.
func (r *Runtime) registerExporter(
	instance export.Exporter,
	destination export.Destination,
) error {
	if instance == nil {
		return errors.New(
			"cannot register nil exporter",
		)
	}

	if err := destination.Validate(); err != nil {
		return fmt.Errorf(
			"invalid exporter destination: %w",
			err,
		)
	}

	if instance.ID() != destination.ID {
		return fmt.Errorf(
			"exporter ID %q does not match destination ID %q",
			instance.ID(),
			destination.ID,
		)
	}

	if strings.TrimSpace(
		instance.Type(),
	) != strings.TrimSpace(
		destination.Type,
	) {
		return fmt.Errorf(
			"exporter type %q does not match destination type %q",
			instance.Type(),
			destination.Type,
		)
	}

	if err := r.exporters.Register(
		instance,
	); err != nil {
		return err
	}

	if err := r.router.RegisterDestination(
		destination,
	); err != nil {
		removed, removeErr :=
			r.exporters.Remove(
				instance.ID(),
			)

		if removeErr == nil &&
			removed != nil {
			removeErr = removed.Close()
		}

		return errors.Join(
			fmt.Errorf(
				"register router destination: %w",
				err,
			),
			removeErr,
		)
	}

	return nil
}
