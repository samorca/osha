from __future__ import annotations

"""
Executable entrypoint for the OSHA application runtime.

Run with:

    python -m src.runtime.main

The module owns the foreground process lifecycle:

    configuration
        ->
    service registry
        ->
    runtime bootstrap
        ->
    signal waiting
        ->
    graceful shutdown

Infrastructure remediation is not executed here.
"""

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time

from dataclasses import dataclass

from types import FrameType

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

from src.runtime.config import (
    RuntimeConfig,
    from_environment,
    safe_config_summary,
)

from src.runtime.runtime import (
    OshaRuntime,
    RuntimeLifecycleError,
)

from src.runtime.schema import (
    RuntimeState,
    ShutdownReason,
)

from src.runtime.services import (
    RuntimeService,
    RuntimeServiceRegistry,
    registry_from_config,
)
from src.database.runtime import create_database_pipeline


LOGGER = logging.getLogger(
    "osha.runtime"
)


# -------------------------------------------------------------------
# Process result
# -------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeProcessResult:
    """
    Final result returned by one runtime process.

    This is separate from RuntimeStatus because it
    describes the operating-system process outcome.
    """

    success: bool

    exit_code: int

    reason: str

    runtime_summary: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    shutdown_summary: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    error_type: Optional[str] = None

    error: Optional[str] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "runtime_summary": (
                dict(
                    self.runtime_summary
                )
                if self.runtime_summary
                is not None
                else None
            ),
            "shutdown_summary": (
                dict(
                    self.shutdown_summary
                )
                if self.shutdown_summary
                is not None
                else None
            ),
            "error_type": (
                self.error_type
            ),
            "error": self.error,
        }


# -------------------------------------------------------------------
# Signal state
# -------------------------------------------------------------------


class RuntimeSignalController:
    """
    Convert process signals into a thread event.

    Signal handlers do not execute runtime shutdown
    directly. Shutdown remains in the main control
    flow, where exceptions and lifecycle transitions
    can be handled safely.
    """

    def __init__(
        self,
    ) -> None:
        self._stop_event = (
            threading.Event()
        )

        self._signal_name: Optional[
            str
        ] = None

        self._previous_handlers: Dict[
            int,
            Any,
        ] = {}

    @property
    def stop_event(
        self,
    ) -> threading.Event:
        return self._stop_event

    @property
    def signal_name(
        self,
    ) -> Optional[str]:
        return self._signal_name

    def request_stop(
        self,
        *,
        reason: str = "requested",
    ) -> None:
        if self._signal_name is None:
            self._signal_name = reason

        self._stop_event.set()

    def install(
        self,
    ) -> None:
        for signal_number in (
            signal.SIGINT,
            signal.SIGTERM,
        ):
            self._previous_handlers[
                signal_number
            ] = signal.getsignal(
                signal_number
            )

            signal.signal(
                signal_number,
                self._handle_signal,
            )

    def restore(
        self,
    ) -> None:
        for (
            signal_number,
            handler,
        ) in self._previous_handlers.items():
            signal.signal(
                signal_number,
                handler,
            )

        self._previous_handlers.clear()

    def wait(
        self,
        timeout: Optional[
            float
        ] = None,
    ) -> bool:
        return self._stop_event.wait(
            timeout=timeout
        )

    def _handle_signal(
        self,
        signal_number: int,
        frame: Optional[
            FrameType
        ],
    ) -> None:
        del frame

        try:
            name = signal.Signals(
                signal_number
            ).name

        except ValueError:
            name = str(
                signal_number
            )

        if self._signal_name is None:
            self._signal_name = name

            LOGGER.info(
                "Shutdown signal received: %s",
                name,
            )

        self._stop_event.set()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def parse_args(
    argv: Optional[
        Sequence[
            str
        ]
    ] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the OSHA orchestration "
            "application runtime."
        )
    )

    parser.add_argument(
        "--host-id",
        default=None,
        help=(
            "Override the runtime host ID. "
            "Defaults to the local hostname."
        ),
    )

    parser.add_argument(
        "--started-by",
        default=None,
        help=(
            "Operator or service identity that "
            "started this runtime."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Bootstrap the runtime, emit a snapshot, "
            "then shut down immediately."
        ),
    )

    parser.add_argument(
        "--status-interval",
        type=float,
        default=None,
        help=(
            "Emit a safe runtime snapshot every N "
            "seconds. By default snapshots are not "
            "printed periodically."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit structured lifecycle events as "
            "JSON."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "critical",
            "error",
            "warning",
            "info",
            "debug",
        ),
        default=os.environ.get(
            "OSHA_RUNTIME_LOG_LEVEL",
            "info",
        ).strip().lower(),
        help=(
            "Runtime process log level. "
            "Default: OSHA_RUNTIME_LOG_LEVEL or info."
        ),
    )

    parser.add_argument(
        "--no-signal-handlers",
        action="store_true",
        help=(
            "Do not install SIGINT/SIGTERM handlers. "
            "Intended for embedding and controlled "
            "tests."
        ),
    )

    return parser.parse_args(
        argv
    )


def validate_args(
    args: argparse.Namespace,
) -> None:
    if (
        args.status_interval
        is not None
        and args.status_interval
        <= 0
    ):
        raise ValueError(
            "--status-interval must be greater "
            "than zero."
        )

    if not str(
        args.log_level
    ).strip():
        raise ValueError(
            "--log-level cannot be empty."
        )


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------


def configure_logging(
    level: str,
) -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            level.upper(),
            logging.INFO,
        ),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )


# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------


def serialize(
    value: Any,
) -> Any:
    """
    Convert OSHA runtime values into safe JSON-ready
    representations without exposing arbitrary object
    internals.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): serialize(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            serialize(
                item
            )
            for item in value
        ]

    converter = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(
        converter
    ):
        return serialize(
            converter()
        )

    return str(
        value
    )


def emit(
    event: str,
    payload: Mapping[
        str,
        Any,
    ],
    *,
    json_output: bool,
) -> None:
    message = {
        "event": event,
        **{
            str(key): serialize(
                value
            )
            for key, value
            in payload.items()
        },
    }

    if json_output:
        print(
            json.dumps(
                message,
                sort_keys=True,
                ensure_ascii=False,
            ),
            flush=True,
        )

        return

    print(
        f"\n[{event}]",
        flush=True,
    )

    print(
        json.dumps(
            message,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )


# -------------------------------------------------------------------
# Runtime composition
# -------------------------------------------------------------------


def build_registry(
    config: RuntimeConfig,
) -> RuntimeServiceRegistry:
    """
    Build the configured runtime-service registry.

    The persistence service is composed here so that
    each runtime process initializes and checks its
    configured database connection.
    """

    registry = registry_from_config(config)
    database = create_database_pipeline()
    database_service = registry.get("database")

    def start_database() -> Dict[str, Any]:
        try:
            result = database.start()
            database.telemetry_repository().ensure_schema()
            return result
        except Exception:
            if database.ready:
                database.stop()
            raise

    if database_service is None:
        database_service = RuntimeService(
            name="database",
            config=config.service("database"),
        )
        registry.register(database_service)

    database_service.start_hook = start_database
    database_service.stop_hook = database.stop
    database_service.readiness_probe = lambda: database.ready
    database_service.attributes["backend"] = database.config.backend
    registry.validate_dependencies()
    return registry


def build_runtime(
    config: RuntimeConfig,
) -> OshaRuntime:
    registry = build_registry(
        config
    )

    return OshaRuntime(
        config=config,
        registry=registry,
    )


# -------------------------------------------------------------------
# Safe runtime output
# -------------------------------------------------------------------


def runtime_snapshot(
    runtime: OshaRuntime,
) -> Dict[str, Any]:
    snapshot = runtime.snapshot()

    return snapshot.to_dict()


def runtime_summary(
    runtime: OshaRuntime,
) -> Dict[str, Any]:
    return runtime.summary()


# -------------------------------------------------------------------
# Shutdown reason
# -------------------------------------------------------------------


def shutdown_reason_for_signal(
    signal_name: Optional[
        str
    ],
) -> str:
    if signal_name in {
        "SIGINT",
        "SIGTERM",
    }:
        return (
            ShutdownReason.REQUESTED.value
        )

    return ShutdownReason.REQUESTED.value


# -------------------------------------------------------------------
# Main lifecycle
# -------------------------------------------------------------------


def run(
    args: argparse.Namespace,
) -> RuntimeProcessResult:
    config = from_environment()

    registry = build_registry(
        config
    )

    runtime = OshaRuntime(
        config=config,
        registry=registry,
    )

    controller = (
        RuntimeSignalController()
    )

    handlers_installed = False

    shutdown_summary: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    final_summary: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    if not args.no_signal_handlers:
        controller.install()
        handlers_installed = True

    host_id = (
        args.host_id
        or socket.gethostname()
    )

    started_by = (
        args.started_by
        or os.environ.get(
            "OSHA_STARTED_BY"
        )
        or "runtime-main"
    )

    try:
        emit(
            "runtime_starting",
            {
                "host_id": host_id,
                "started_by": started_by,
                "config": (
                    safe_config_summary(
                        config
                    )
                ),
                "registered_services": (
                    registry.names()
                ),
            },
            json_output=args.json,
        )

        bootstrap = runtime.start(
            host_id=host_id,
            started_by=started_by,
        )

        emit(
            "runtime_bootstrap",
            {
                "success": (
                    bootstrap.success
                ),
                "degraded": (
                    bootstrap.degraded
                ),
                "failed_service_names": (
                    list(
                        bootstrap.
                        failed_service_names
                    )
                ),
                "runtime": (
                    bootstrap.runtime.
                    to_dict()
                ),
                "service_results": [
                    serialize(
                        result
                    )
                    for result in (
                        bootstrap.
                        service_results
                    )
                ],
            },
            json_output=args.json,
        )

        if not bootstrap.success:
            final_summary = (
                runtime_summary(
                    runtime
                )
            )

            return RuntimeProcessResult(
                success=False,
                exit_code=1,
                reason=(
                    "Runtime bootstrap failed."
                ),
                runtime_summary=(
                    final_summary
                ),
            )

        snapshot = runtime_snapshot(
            runtime
        )

        emit(
            "runtime_ready",
            {
                "ready": runtime.ready,
                "degraded": (
                    bootstrap.degraded
                ),
                "snapshot": snapshot,
            },
            json_output=args.json,
        )

        if args.once:
            controller.request_stop(
                reason="once"
            )

        while not controller.stop_event.is_set():
            wait_interval = (
                args.status_interval
                if args.status_interval
                is not None
                else config.
                health_interval_seconds
            )

            controller.wait(
                timeout=wait_interval
            )

            if controller.stop_event.is_set():
                break

            if (
                args.status_interval
                is not None
            ):
                emit(
                    "runtime_status",
                    {
                        "snapshot": (
                            runtime_snapshot(
                                runtime
                            )
                        ),
                    },
                    json_output=(
                        args.json
                    ),
                )

        reason = (
            "One-shot runtime completed."
            if controller.signal_name
            == "once"
            else (
                "Runtime shutdown requested by "
                f"{controller.signal_name or 'process'}."
            )
        )

        emit(
            "runtime_stopping",
            {
                "reason": reason,
                "signal": (
                    controller.signal_name
                ),
            },
            json_output=args.json,
        )

        shutdown = runtime.shutdown(
            reason=reason,
            shutdown_reason=(
                shutdown_reason_for_signal(
                    controller.signal_name
                )
            ),
        )

        shutdown_summary = serialize(
            shutdown
        )

        final_summary = (
            runtime_summary(
                runtime
            )
        )

        emit(
            "runtime_stopped",
            {
                "success": (
                    getattr(
                        shutdown,
                        "success",
                        True,
                    )
                ),
                "shutdown": (
                    shutdown_summary
                ),
                "summary": (
                    final_summary
                ),
            },
            json_output=args.json,
        )

        shutdown_success = bool(
            getattr(
                shutdown,
                "success",
                True,
            )
        )

        return RuntimeProcessResult(
            success=shutdown_success,
            exit_code=(
                0
                if shutdown_success
                else 1
            ),
            reason=(
                reason
                if shutdown_success
                else (
                    "Runtime shutdown "
                    "completed with failures."
                )
            ),
            runtime_summary=(
                final_summary
            ),
            shutdown_summary=(
                shutdown_summary
            ),
        )

    except KeyboardInterrupt:
        LOGGER.info(
            "Runtime interrupted by keyboard."
        )

        controller.request_stop(
            reason="SIGINT"
        )

        if (
            runtime.status is not None
            and runtime.status.state
            != RuntimeState.STOPPED.value
        ):
            try:
                shutdown = runtime.shutdown(
                    reason=(
                        "Runtime interrupted "
                        "by keyboard."
                    ),
                    shutdown_reason=(
                        ShutdownReason.
                        REQUESTED.value
                    ),
                )

                shutdown_summary = (
                    serialize(
                        shutdown
                    )
                )

            except Exception:
                LOGGER.exception(
                    "Runtime shutdown after "
                    "keyboard interrupt failed."
                )

        final_summary = (
            runtime_summary(
                runtime
            )
            if runtime.status
            is not None
            else None
        )

        return RuntimeProcessResult(
            success=False,
            exit_code=130,
            reason=(
                "Runtime interrupted."
            ),
            runtime_summary=(
                final_summary
            ),
            shutdown_summary=(
                shutdown_summary
            ),
        )

    except Exception as exc:
        LOGGER.exception(
            "OSHA runtime process failed."
        )

        if (
            runtime.status is not None
            and runtime.status.state
            != RuntimeState.STOPPED.value
        ):
            try:
                shutdown = runtime.shutdown(
                    reason=(
                        "Runtime process failed: "
                        f"{type(exc).__name__}."
                    ),
                    shutdown_reason=(
                        ShutdownReason.FAILURE.value
                    ),
                )

                shutdown_summary = (
                    serialize(
                        shutdown
                    )
                )

            except Exception:
                LOGGER.exception(
                    "Failure-path runtime "
                    "shutdown also failed."
                )

        try:
            final_summary = (
                runtime_summary(
                    runtime
                )
                if runtime.status
                is not None
                else None
            )

        except Exception:
            final_summary = None

        return RuntimeProcessResult(
            success=False,
            exit_code=1,
            reason=(
                "Runtime process failed."
            ),
            runtime_summary=(
                final_summary
            ),
            shutdown_summary=(
                shutdown_summary
            ),
            error_type=(
                type(exc).__name__
            ),
            error=str(
                exc
            ),
        )

    finally:
        if handlers_installed:
            controller.restore()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------


def main(
    argv: Optional[
        Sequence[
            str
        ]
    ] = None,
) -> None:
    args = parse_args(
        argv
    )

    validate_args(
        args
    )

    configure_logging(
        args.log_level
    )

    result = run(
        args
    )

    if not result.success:
        emit(
            "runtime_process_result",
            result.to_dict(),
            json_output=args.json,
        )

    raise SystemExit(
        result.exit_code
    )


if __name__ == "__main__":
    main()
