from __future__ import annotations

"""
Read-only Linux host client for OSHA.

The first implementation supports local host
inspection through psutil and Python standard
library metadata.

No arbitrary command execution is provided.
"""

import os
import platform
import socket

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationConnectionError,
    IntegrationOperationError,
    wrap_provider_error,
)

from src.integrations.linux.config import (
    LinuxConfig,
    LinuxMode,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Optional psutil import
# ---------------------------------------------------------------

try:
    import psutil

except ImportError:
    psutil = None


# ---------------------------------------------------------------
# Client error
# ---------------------------------------------------------------


class LinuxClientError(IntegrationOperationError):
    default_code = "linux_client_error"


# ---------------------------------------------------------------
# Named tuple conversion
# ---------------------------------------------------------------


def _as_dict(
    value: Any,
) -> Dict[str, Any]:
    if value is None:
        return {}

    method = getattr(
        value,
        "_asdict",
        None,
    )

    if callable(method):
        return dict(method())

    if isinstance(
        value,
        dict,
    ):
        return dict(value)

    return {}


# ---------------------------------------------------------------
# Linux client
# ---------------------------------------------------------------


class LinuxClient:
    """
    Local read-only Linux client.

    REMOTE mode is represented in configuration
    but deliberately not implemented here until
    a controlled remote transport adapter is
    introduced.
    """

    def __init__(
        self,
        *,
        config: LinuxConfig,
        credential: Optional[SecretValue] = None,
    ) -> None:
        self.config = config

        self.config.validate()

        self.credential = credential

        self._connected = False

    # ---------------------------------------------------------
    # Connected
    # ---------------------------------------------------------

    @property
    def connected(
        self,
    ) -> bool:
        return self._connected

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    def connect(
        self,
    ) -> "LinuxClient":
        if self.connected:
            return self

        if self.config.connection.mode != LinuxMode.LOCAL.value:
            raise (
                IntegrationConnectionError(
                    "Remote Linux "
                    "integration is not "
                    "implemented in this "
                    "read-only client."
                )
            )

        if psutil is None:
            raise (IntegrationConnectionError("psutil is not installed."))

        self._connected = True

        return self

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(
        self,
    ) -> None:
        self._connected = False

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping(
        self,
    ) -> bool:
        """
        Verify that local system information can
        be read.
        """

        if not self.connected:
            return False

        try:
            psutil.cpu_count()

            return True

        except Exception:
            return False

    # ---------------------------------------------------------
    # Host identity
    # ---------------------------------------------------------

    def host_info(
        self,
    ) -> Dict[str, Any]:
        self._require_connected()

        try:
            value = {
                "hostname": (socket.gethostname()),
                "fqdn": (socket.getfqdn()),
                "system": (platform.system()),
                "release": (platform.release()),
                "version": (platform.version()),
                "machine": (platform.machine()),
                "architecture": (platform.architecture()[0]),
                "python_version": (platform.python_version()),
            }

            if self.config.include_boot_info:
                boot_timestamp = psutil.boot_time()

                value["boot_time"] = datetime.fromtimestamp(
                    boot_timestamp,
                    tz=timezone.utc,
                ).isoformat()

                value["uptime_seconds"] = max(
                    0.0,
                    datetime.now(timezone.utc).timestamp() - boot_timestamp,
                )

            return value

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Linux host information collection failed.",
            ) from exc

    # ---------------------------------------------------------
    # CPU
    # ---------------------------------------------------------

    def cpu_info(
        self,
    ) -> Dict[str, Any]:
        self._require_enabled(
            self.config.include_cpu,
            "cpu",
        )

        try:
            frequency = psutil.cpu_freq()

            return {
                "logical_count": (psutil.cpu_count(logical=True)),
                "physical_count": (psutil.cpu_count(logical=False)),
                "percent": (psutil.cpu_percent(interval=None)),
                "times": (_as_dict(psutil.cpu_times())),
                "stats": (_as_dict(psutil.cpu_stats())),
                "frequency": (_as_dict(frequency) if frequency is not None else None),
            }

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Linux CPU information collection failed.",
            ) from exc

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def memory_info(
        self,
    ) -> Dict[str, Any]:
        self._require_enabled(
            self.config.include_memory,
            "memory",
        )

        try:
            return {
                "virtual": (_as_dict(psutil.virtual_memory())),
                "swap": (_as_dict(psutil.swap_memory())),
            }

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Linux memory information collection failed.",
            ) from exc

    # ---------------------------------------------------------
    # Disks
    # ---------------------------------------------------------

    def disk_info(
        self,
    ) -> Dict[str, Any]:
        self._require_enabled(
            self.config.include_disks,
            "disks",
        )

        try:
            partitions: List[
                Dict[
                    str,
                    Any,
                ]
            ] = []

            for partition in psutil.disk_partitions(all=False):
                item = _as_dict(partition)

                try:
                    item["usage"] = _as_dict(psutil.disk_usage(partition.mountpoint))

                except (
                    PermissionError,
                    OSError,
                ):
                    item["usage"] = None

                partitions.append(item)

            io = psutil.disk_io_counters()

            return {
                "partitions": (partitions),
                "io": (_as_dict(io) if io is not None else None),
            }

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Linux disk information collection failed.",
            ) from exc

    # ---------------------------------------------------------
    # Network
    # ---------------------------------------------------------

    def network_info(
        self,
    ) -> Dict[str, Any]:
        self._require_enabled(
            self.config.include_network,
            "network",
        )

        try:
            addresses = {}

            for (
                interface,
                interface_addresses,
            ) in psutil.net_if_addrs().items():
                addresses[interface] = [
                    _as_dict(address) for address in interface_addresses
                ]

            stats = {
                name: _as_dict(value) for name, value in psutil.net_if_stats().items()
            }

            io = {
                name: _as_dict(value)
                for name, value in psutil.net_io_counters(pernic=True).items()
            }

            return {
                "interfaces": (addresses),
                "statistics": (stats),
                "io": io,
            }

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Linux network information collection failed.",
            ) from exc

    # ---------------------------------------------------------
    # Processes
    # ---------------------------------------------------------

    def processes(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        """
        Return bounded process metadata.

        Process command lines and environments
        are deliberately excluded.
        """

        self._require_enabled(
            self.config.include_processes,
            "processes",
        )

        if limit is not None and limit < 1:
            raise ValueError("process limit must be >= 1")

        output: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        try:
            iterator = psutil.process_iter(
                attrs=[
                    "pid",
                    "ppid",
                    "name",
                    "username",
                    "status",
                    "create_time",
                    "cpu_percent",
                    "memory_percent",
                ]
            )

            for process in iterator:
                try:
                    info = dict(process.info)

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue

                output.append(info)

                if limit is not None and len(output) >= limit:
                    break

            return output

        except Exception as exc:
            raise self._operation_error(
                exc,
                "Linux process information collection failed.",
            ) from exc

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    def environment(
        self,
    ) -> Dict[str, str]:
        """
        Host/process environment is disabled by
        default because it may contain secrets.

        Even when deliberately enabled, normal
        OSHA redaction/serialization controls
        must still apply upstream.
        """

        self._require_enabled(
            self.config.include_environment,
            "environment",
        )

        return dict(os.environ)

    # ---------------------------------------------------------
    # Complete snapshot
    # ---------------------------------------------------------

    def snapshot(
        self,
        *,
        process_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Collect only categories enabled by
        LinuxConfig.
        """

        self._require_connected()

        value: Dict[
            str,
            Any,
        ] = {
            "host": (self.host_info()),
        }

        if self.config.include_cpu:
            value["cpu"] = self.cpu_info()

        if self.config.include_memory:
            value["memory"] = self.memory_info()

        if self.config.include_disks:
            value["disks"] = self.disk_info()

        if self.config.include_network:
            value["network"] = self.network_info()

        if self.config.include_processes:
            value["processes"] = self.processes(limit=process_limit)

        # Deliberately do not add environment
        # automatically even when enabled.
        # It must be requested explicitly through
        # environment().

        return value

    # ---------------------------------------------------------
    # Guards
    # ---------------------------------------------------------

    def _require_connected(
        self,
    ) -> None:
        if not self.connected:
            raise (IntegrationConnectionError("Linux client is not connected."))

    def _require_enabled(
        self,
        enabled: bool,
        capability: str,
    ) -> None:
        self._require_connected()

        if not enabled:
            raise (
                IntegrationCapabilityError(
                    f"Linux information category is disabled: {capability}"
                )
            )

    # ---------------------------------------------------------
    # Error translation
    # ---------------------------------------------------------

    def _operation_error(
        self,
        error: BaseException,
        message: str,
    ) -> LinuxClientError:
        wrapped = wrap_provider_error(
            error,
            provider="linux",
            target_id=(self.config.target_id),
            message=message,
            retryable=False,
            code=("linux_read_failed"),
        )

        return LinuxClientError(
            message,
            code=wrapped.code,
            retryable=(wrapped.retryable),
            details=(wrapped.details),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connected": (self.connected),
            "mode": (self.config.connection.mode),
            "target_id": (self.config.target_id),
            "categories": {
                "cpu": (self.config.include_cpu),
                "memory": (self.config.include_memory),
                "disks": (self.config.include_disks),
                "network": (self.config.include_network),
                "processes": (self.config.include_processes),
                "environment": (self.config.include_environment),
            },
            "remote_execution_enabled": (False),
            "arbitrary_command_execution_enabled": (False),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "LinuxClientError",
    "LinuxClient",
]
