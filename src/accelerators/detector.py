from __future__ import annotations

"""
Accelerator discovery for OSHA.

AcceleratorDetector composes the concrete:

    CPU backend
    CUDA backend
    ROCm backend

into one framework-neutral AcceleratorInventory.

The detector owns:

    backend registration
    backend discovery ordering
    inventory aggregation
    duplicate-device validation
    backend discovery diagnostics
    health aggregation

The detector does NOT:

    choose a workload device
    perform CPU fallback
    move tensors
    train models
    execute inference

Device selection belongs to manager.py.
"""

from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.accelerators.base import (
    AcceleratorBackend,
    BackendDiscoveryResult,
)

from src.accelerators.config import (
    AcceleratorConfig,
)

from src.accelerators.cpu.backend import (
    CPUBackend,
)

from src.accelerators.cuda.backend import (
    CUDABackend,
)

from src.accelerators.errors import (
    AcceleratorDiscoveryError,
    AcceleratorDuplicateDeviceError,
    AcceleratorInventoryError,
    safe_accelerator_error_details,
)

from src.accelerators.rocm.backend import (
    ROCmBackend,
)

from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorHealth,
    AcceleratorHealthReport,
    AcceleratorHealthStatus,
    AcceleratorInventory,
    AcceleratorKind,
)


# ---------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------


class AcceleratorDetectionResult:
    """
    Full detector result.

    inventory contains the normalized devices.

    backend_results preserves bounded information
    about each backend discovery attempt.
    """

    def __init__(
        self,
        *,
        inventory: AcceleratorInventory,
        backend_results: Mapping[
            str,
            BackendDiscoveryResult,
        ],
        failures: Iterable[
            Mapping[
                str,
                Any,
            ]
        ] = (),
    ) -> None:
        inventory.validate()

        self.inventory = inventory

        self.backend_results = {
            str(name): result for name, result in backend_results.items()
        }

        self.failures = tuple(dict(failure) for failure in failures)

        self.validate()

    def validate(
        self,
    ) -> None:
        self.inventory.validate()

        for name, result in self.backend_results.items():
            result.validate()

            if name != result.backend:
                raise AcceleratorInventoryError(
                    "Backend discovery result key does not match its backend identity.",
                    details={
                        "key": name,
                        "backend": (result.backend),
                    },
                )

    @property
    def successful(
        self,
    ) -> bool:
        return not self.failures

    @property
    def failure_count(
        self,
    ) -> int:
        return len(self.failures)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "successful": (self.successful),
            "failure_count": (self.failure_count),
            "inventory": (self.inventory.to_dict()),
            "backends": {
                name: result.to_dict() for name, result in self.backend_results.items()
            },
            "failures": [dict(failure) for failure in self.failures],
        }


# ---------------------------------------------------------------
# Detector
# ---------------------------------------------------------------


class AcceleratorDetector:
    """
    Compose accelerator backends into one
    inventory.

    Default configured backends:

        CPU
        CUDA
        ROCm

    Backends are queried according to
    config.discovery.backend_priority.
    """

    def __init__(
        self,
        *,
        config: Optional[AcceleratorConfig] = None,
        backends: Optional[Iterable[AcceleratorBackend]] = None,
    ) -> None:
        self.config = config or AcceleratorConfig()

        self.config.validate()

        self._lock = RLock()

        self._backends: Dict[
            str,
            AcceleratorBackend,
        ] = {}

        self._last_result: Optional[AcceleratorDetectionResult] = None

        if backends is None:
            self._register_defaults()

        else:
            for backend in backends:
                self.register_backend(backend)

    # ---------------------------------------------------------
    # Default backends
    # ---------------------------------------------------------

    def _register_defaults(
        self,
    ) -> None:
        discovery = self.config.discovery

        if discovery.discover_cpu:
            self.register_backend(CPUBackend(config=self.config))

        if discovery.discover_cuda:
            self.register_backend(CUDABackend(config=self.config))

        if discovery.discover_rocm:
            self.register_backend(ROCmBackend(config=self.config))

    # ---------------------------------------------------------
    # Register backend
    # ---------------------------------------------------------

    def register_backend(
        self,
        backend: AcceleratorBackend,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(
            backend,
            AcceleratorBackend,
        ):
            raise TypeError("backend must implement AcceleratorBackend")

        kind = backend.kind

        with self._lock:
            if kind in self._backends and not replace:
                raise AcceleratorDiscoveryError(
                    "Accelerator backend is already registered.",
                    details={
                        "backend": kind,
                    },
                )

            self._backends[kind] = backend

    # ---------------------------------------------------------
    # Remove backend
    # ---------------------------------------------------------

    def remove_backend(
        self,
        kind: str,
    ) -> AcceleratorBackend:
        normalized = str(kind).strip().lower()

        with self._lock:
            backend = self._backends.pop(
                normalized,
                None,
            )

        if backend is None:
            raise AcceleratorDiscoveryError(
                "Accelerator backend is not registered.",
                details={
                    "backend": (normalized),
                },
            )

        return backend

    # ---------------------------------------------------------
    # Backend lookup
    # ---------------------------------------------------------

    def backend(
        self,
        kind: str,
    ) -> Optional[AcceleratorBackend]:
        normalized = str(kind).strip().lower()

        with self._lock:
            return self._backends.get(normalized)

    # ---------------------------------------------------------
    # Required backend
    # ---------------------------------------------------------

    def require_backend(
        self,
        kind: str,
    ) -> AcceleratorBackend:
        backend = self.backend(kind)

        if backend is None:
            raise AcceleratorDiscoveryError(
                "Requested accelerator backend is not registered.",
                details={
                    "backend": kind,
                },
            )

        return backend

    # ---------------------------------------------------------
    # Ordered backends
    # ---------------------------------------------------------

    def backends(
        self,
    ) -> Tuple[
        AcceleratorBackend,
        ...,
    ]:
        with self._lock:
            values = dict(self._backends)

        output: List[AcceleratorBackend] = []

        seen = set()

        for kind in self.config.discovery.backend_priority:
            backend = values.get(kind)

            if backend is None:
                continue

            output.append(backend)

            seen.add(kind)

        # Custom/replaced backends not represented in
        # backend_priority are appended deterministically.
        for kind in sorted(values):
            if kind in seen:
                continue

            output.append(values[kind])

        return tuple(output)

    # ---------------------------------------------------------
    # Device identity check
    # ---------------------------------------------------------

    @staticmethod
    def _validate_unique_devices(
        devices: Iterable[AcceleratorDevice],
    ) -> Tuple[
        AcceleratorDevice,
        ...,
    ]:
        output: List[AcceleratorDevice] = []

        seen_ids: Dict[
            str,
            AcceleratorDevice,
        ] = {}

        # Runtime-device duplication is normally invalid
        # between CUDA and ROCm because only one of those
        # logical GPU backends should claim a particular
        # PyTorch runtime device.
        seen_runtime_devices: Dict[
            str,
            AcceleratorDevice,
        ] = {}

        for device in devices:
            device.validate()

            existing = seen_ids.get(device.device_id)

            if existing is not None:
                raise (
                    AcceleratorDuplicateDeviceError(
                        "Accelerator inventory contains duplicate device identity.",
                        details={
                            "device_id": (device.device_id),
                            "first_kind": (existing.kind),
                            "second_kind": (device.kind),
                        },
                    )
                )

            seen_ids[device.device_id] = device

            runtime_key = device.runtime_device

            existing_runtime = seen_runtime_devices.get(runtime_key)

            if (
                existing_runtime is not None
                and existing_runtime.kind != device.kind
                and {
                    existing_runtime.kind,
                    device.kind,
                }
                <= {
                    AcceleratorKind.CUDA.value,
                    AcceleratorKind.ROCM.value,
                }
            ):
                raise (
                    AcceleratorDuplicateDeviceError(
                        "CUDA and ROCm backends claimed the same runtime device.",
                        details={
                            "runtime_device": (runtime_key),
                            "first_device_id": (existing_runtime.device_id),
                            "first_kind": (existing_runtime.kind),
                            "second_device_id": (device.device_id),
                            "second_kind": (device.kind),
                        },
                    )
                )

            seen_runtime_devices[runtime_key] = device

            output.append(device)

        return tuple(output)

    # ---------------------------------------------------------
    # Detect
    # ---------------------------------------------------------

    def detect(
        self,
        *,
        fail_fast: Optional[bool] = None,
    ) -> AcceleratorDetectionResult:
        """
        Query all registered backends and build
        one AcceleratorInventory.

        fail_fast defaults to the configured
        discovery.fail_on_backend_error policy.
        """

        if not self.config.enabled:
            inventory = AcceleratorInventory(
                devices=(),
                attributes={
                    "accelerators_enabled": (False),
                },
            )

            result = AcceleratorDetectionResult(
                inventory=inventory,
                backend_results={},
                failures=(),
            )

            with self._lock:
                self._last_result = result

            return result

        should_fail_fast = (
            self.config.discovery.fail_on_backend_error
            if fail_fast is None
            else bool(fail_fast)
        )

        devices: List[AcceleratorDevice] = []

        backend_results: Dict[
            str,
            BackendDiscoveryResult,
        ] = {}

        failures: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        for backend in self.backends():
            try:
                result = backend.discover()

                backend_results[backend.kind] = result

                devices.extend(result.devices)

                if result.error_type is not None:
                    failures.append(
                        {
                            "backend": (backend.kind),
                            "error_type": (result.error_type),
                            "code": ("backend_discovery_degraded"),
                        }
                    )

            except Exception as exc:
                if should_fail_fast:
                    raise

                failures.append(
                    {
                        "backend": (backend.kind),
                        "error_type": (type(exc).__name__),
                        "code": getattr(
                            exc,
                            "code",
                            ("accelerator_discovery_error"),
                        ),
                    }
                )

                backend_results[backend.kind] = BackendDiscoveryResult(
                    backend=backend.kind,
                    devices=(),
                    available=False,
                    error_type=(type(exc).__name__),
                    attributes={
                        "detector_caught_error": (True),
                    },
                )

        try:
            normalized_devices = self._validate_unique_devices(devices)

            inventory = AcceleratorInventory(
                devices=(normalized_devices),
                attributes={
                    "backend_priority": list(self.config.discovery.backend_priority),
                    "registered_backends": [
                        backend.kind for backend in self.backends()
                    ],
                    "backend_failure_count": (len(failures)),
                },
            )

            inventory.validate()

        except Exception as exc:
            if isinstance(
                exc,
                AcceleratorInventoryError,
            ):
                raise

            raise AcceleratorInventoryError(
                "Accelerator inventory could not be constructed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("construct_inventory"),
                    )
                ),
            ) from exc

        result = AcceleratorDetectionResult(
            inventory=inventory,
            backend_results=(backend_results),
            failures=failures,
        )

        with self._lock:
            self._last_result = result

        return result

    # ---------------------------------------------------------
    # Inventory convenience
    # ---------------------------------------------------------

    def inventory(
        self,
        *,
        refresh: bool = True,
    ) -> AcceleratorInventory:
        """
        Return the accelerator inventory.

        refresh=True performs current discovery.

        refresh=False returns the most recent
        detector snapshot when one exists.
        """

        if not refresh:
            with self._lock:
                cached = self._last_result

            if cached is not None:
                return cached.inventory

        return self.detect().inventory

    # ---------------------------------------------------------
    # Last result
    # ---------------------------------------------------------

    @property
    def last_result(
        self,
    ) -> Optional[AcceleratorDetectionResult]:
        with self._lock:
            return self._last_result

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health(
        self,
        *,
        deep: Optional[bool] = None,
        refresh_inventory: bool = True,
    ) -> AcceleratorHealthReport:
        """
        Aggregate backend health.

        Backend UNAVAILABLE is normal when that
        backend is optional. Overall readiness is
        interpreted by manager/config policy.
        """

        if not self.config.enabled:
            return AcceleratorHealthReport(
                components=(
                    AcceleratorHealth(
                        component=("accelerators"),
                        status=(AcceleratorHealthStatus.UNAVAILABLE.value),
                        message=("Accelerator subsystem is disabled."),
                        retryable=False,
                    ),
                ),
                inventory=(
                    AcceleratorInventory(
                        devices=(),
                        attributes={
                            "accelerators_enabled": (False),
                        },
                    )
                ),
                attributes={
                    "enabled": False,
                },
            )

        deep_check = self.config.health.deep_check if deep is None else bool(deep)

        components: List[AcceleratorHealth] = []

        for backend in self.backends():
            try:
                components.append(backend.health(deep=deep_check))

            except Exception as exc:
                components.append(
                    AcceleratorHealth(
                        component=("accelerator_backend:" + backend.kind),
                        status=(AcceleratorHealthStatus.UNHEALTHY.value),
                        message=("Accelerator backend health check failed."),
                        kind=backend.kind,
                        retryable=True,
                        error_type=(type(exc).__name__),
                        attributes={
                            "deep_check": (deep_check),
                        },
                    )
                )

        inventory = self.inventory(refresh=(refresh_inventory))

        # CPU can be an explicit baseline health
        # requirement.
        if self.config.health.require_cpu and not inventory.by_kind(
            AcceleratorKind.CPU.value,
            available_only=True,
        ):
            components.append(
                AcceleratorHealth(
                    component=("accelerator_policy:cpu"),
                    status=(AcceleratorHealthStatus.UNHEALTHY.value),
                    message=(
                        "CPU availability is "
                        "required but no "
                        "usable CPU backend "
                        "was detected."
                    ),
                    kind=(AcceleratorKind.CPU.value),
                    retryable=False,
                )
            )

        preferred = self.config.selection.preferred_kind

        if (
            self.config.health.require_preferred_backend
            and preferred != AcceleratorKind.AUTO.value
            and not inventory.by_kind(
                preferred,
                available_only=True,
            )
        ):
            components.append(
                AcceleratorHealth(
                    component=("accelerator_policy:" + preferred),
                    status=(AcceleratorHealthStatus.UNHEALTHY.value),
                    message=(
                        "The configured "
                        "preferred "
                        "accelerator backend "
                        "is required but "
                        "unavailable."
                    ),
                    kind=preferred,
                    retryable=True,
                )
            )

        return AcceleratorHealthReport(
            components=tuple(components),
            inventory=inventory,
            attributes={
                "enabled": True,
                "deep_check": (deep_check),
                "registered_backends": [backend.kind for backend in self.backends()],
            },
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        *,
        refresh: bool = True,
    ) -> Dict[str, Any]:
        inventory = self.inventory(refresh=refresh)

        return {
            "enabled": (self.config.enabled),
            "registered_backends": [backend.kind for backend in self.backends()],
            "backend_priority": list(self.config.discovery.backend_priority),
            "inventory": (inventory.to_dict()),
            "selection_performed": (False),
        }


# ---------------------------------------------------------------
# Default detector
# ---------------------------------------------------------------


def create_accelerator_detector(
    *,
    config: Optional[AcceleratorConfig] = None,
) -> AcceleratorDetector:
    return AcceleratorDetector(config=config)


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AcceleratorDetectionResult",
    "AcceleratorDetector",
    "create_accelerator_detector",
]
