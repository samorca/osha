package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/samorca/otgh/internal/config"
	v1 "github.com/samorca/otgh/cmd/otgh/v1"
)

func main() {
	if err := run(); err != nil {
		log.Printf(
			"otgh: fatal: %v",
			err,
		)

		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf(
			"load OTGH configuration: %w",
			err,
		)
	}

	runtime, err := v1.NewRuntime(cfg)
	if err != nil {
		return fmt.Errorf(
			"create OTGH runtime: %w",
			err,
		)
	}

	// If startup fails after shared resources have been constructed, ensure
	// they are still released.
	started := false

	defer func() {
		if started {
			return
		}

		if runtime != nil &&
			!runtime.Closed() {
			if shutdownErr :=
				runtime.ForceShutdown(); shutdownErr != nil {
				log.Printf(
					"otgh: startup cleanup: %v",
					shutdownErr,
				)
			}
		}
	}()

	startupCtx := context.Background()

	if err := runtime.ConfigureExporters(
		startupCtx,
	); err != nil {
		return fmt.Errorf(
			"configure OTGH exporters: %w",
			err,
		)
	}
	listenAddress := strings.TrimSpace(os.Getenv("OTGH_LISTEN_ADDR"))
	if listenAddress == "" {
		listenAddress = ":8082"
	}
	ingestServer, err := v1.NewIngestHTTPServer(
		runtime,
		listenAddress,
		os.Getenv("OTGH_INGEST_TOKEN"),
	)
	if err != nil {
		return fmt.Errorf("create OTGH ingestion server: %w", err)
	}

	// SIGINT handles terminal interruption such as Ctrl-C.
	// SIGTERM is commonly used by service managers and container
	// orchestrators to request graceful termination.
	signalCtx, stopSignals :=
		signal.NotifyContext(
			context.Background(),
			os.Interrupt,
			syscall.SIGTERM,
		)
	defer stopSignals()
	runCtx, cancelRun := context.WithCancel(signalCtx)
	defer cancelRun()

	serverErrors := make(chan error, 1)
	go func() {
		err := ingestServer.ListenAndServe()
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErrors <- err
			cancelRun()
		}
	}()

	log.Printf(
		"otgh: starting node=%s workers=%d exporters=%d",
		cfg.Application.NodeID,
		cfg.Application.WorkerCount,
		cfg.Exporters.Count(),
	)

	started = true

	runErr := runtime.Start(runCtx)
	cancelRun()

	shutdownCtx, cancelShutdown := context.WithTimeout(
		context.Background(),
		cfg.Application.ShutdownTimeout,
	)
	ingestShutdownErr := ingestServer.Shutdown(shutdownCtx)
	cancelShutdown()
	if ingestShutdownErr != nil {
		_ = ingestServer.Close()
	}

	select {
	case serverErr := <-serverErrors:
		runErr = errors.Join(runErr, fmt.Errorf("OTGH ingestion server: %w", serverErr))
	default:
	}

	if runErr != nil &&
		!expectedRunTermination(
			runErr,
			runCtx,
		) {
		log.Printf(
			"otgh: runtime stopped with error: %v",
			runErr,
		)
	}

	log.Printf(
		"otgh: shutting down node=%s",
		cfg.Application.NodeID,
	)

	// Stop signal notification before cleanup. A second OS signal can then
	// use the platform's normal behavior instead of repeatedly entering our
	// graceful-shutdown path.
	stopSignals()

	shutdownErr :=
		runtime.ShutdownWithTimeout()
	if ingestShutdownErr != nil {
		shutdownErr = errors.Join(
			shutdownErr,
			fmt.Errorf("stop OTGH ingestion server: %w", ingestShutdownErr),
		)
	}

	if shutdownErr != nil {
		log.Printf(
			"otgh: graceful shutdown failed: %v",
			shutdownErr,
		)

		if forceErr :=
			runtime.ForceShutdown(); forceErr != nil {
			shutdownErr = errors.Join(
				shutdownErr,
				fmt.Errorf(
					"force shutdown: %w",
					forceErr,
				),
			)
		}
	}

	snapshot := runtime.Snapshot()

	log.Printf(
		"otgh: stopped node=%s state=%s workers=%d exporters=%d queue_depth=%d spool_entries=%d",
		snapshot.NodeID,
		snapshot.State,
		snapshot.Workers,
		snapshot.Exporters,
		snapshot.Metrics.QueueDepth,
		snapshot.Metrics.SpoolEntries,
	)

	switch {
	case runErr != nil &&
		!expectedRunTermination(
			runErr,
			runCtx,
		):
		if shutdownErr != nil {
			return errors.Join(
				runErr,
				shutdownErr,
			)
		}

		return runErr

	case shutdownErr != nil:
		return shutdownErr

	default:
		return nil
	}
}

func expectedRunTermination(
	err error,
	ctx context.Context,
) bool {
	if err == nil {
		return true
	}

	if errors.Is(
		err,
		context.Canceled,
	) {
		return true
	}

	if ctx != nil &&
		ctx.Err() != nil &&
		errors.Is(
			err,
			ctx.Err(),
		) {
		return true
	}

	return false
}
