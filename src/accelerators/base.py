from __future__ import annotations

"""
Base contracts for OSHA accelerator backends.

Concrete backends:

    src/accelerators/cpu/backend.py
    src/accelerators/cuda/backend.py
    src/accelerators/rocm/backend.py

all implement AcceleratorBackend.

Responsibilities of a backend:

    report backend availability
    discover devices
    refresh device information
    query memory information
    construct runtime devices
    synchronize execution when supported
    clear runtime caches when supported
    expose backend health

Responsibilities NOT owned here:

    cross-backend device selection
    fallback policy
    workload routing
    training
    AI model execution

Those belong to AcceleratorManager and callers.
"""

from abc import (
    ABC,
    abstractmethod,
)

from threading import (
    RLock,
)

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from src.accelerators.config import (
    AcceleratorConfig,
)

from src.accelerators.errors import (
    AcceleratorBackendOperationError,
    AcceleratorBackendUnavailableError,
    AcceleratorCacheClearError,
    AcceleratorDeviceNotFoundError,
    AcceleratorDeviceUnavailableError,
    AcceleratorHealthCheckError,
    AcceleratorMemoryQueryError,
    AcceleratorRuntimeDeviceError,
    AcceleratorSynchronizationError,
    safe_accelerator_error_details,
)

from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorHealth,
    AcceleratorHealthStatus,
    AcceleratorKind,
    AcceleratorMemory,
    AcceleratorState,
)


# ---------------------------------------------------------------
# Backend discovery result
# ---------------------------------------------------------------

class BackendDiscoveryResult:
    """
    Result of discovering devices from one
    concrete accelerator backend.

    A backend may be unavailable without making
    global accelerator discovery fail.
    """

    def __init__(
        self,
        *,
        backend: str,
        devices: Tuple[
            AcceleratorDevice,
            ...,
        ] = (),
        available: bool = False,
        error_type: Optional[
            str
        ] = None,
        attributes: Optional[
            Dict[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        self.backend = str(
            backend
        ).strip().lower()

        self.devices = tuple(
            devices
        )

        self.available = bool(
            available
        )

        self.error_type = (
            error_type
        )

        self.attributes = dict(
            attributes
            or {}
        )

        self.validate()

    def validate(
        self,
    ) -> None:
        if (
            self.backend
            not in {
                AcceleratorKind.
                CPU.value,

                AcceleratorKind.
                CUDA.value,

                AcceleratorKind.
                ROCM.value,
            }
        ):
            raise ValueError(
                "Backend discovery result "
                "contains an unsupported "
                "backend."
            )

        seen = set()

        for device in self.devices:
            device.validate()

            if (
                device.kind
                != self.backend
            ):
                raise ValueError(
                    "Discovered device kind "
                    "does not match backend."
                )

            if device.device_id in seen:
                raise ValueError(
                    "Backend discovery "
                    "returned duplicate "
                    "device IDs."
                )

            seen.add(
                device.device_id
            )

    @property
    def device_count(
        self,
    ) -> int:
        return len(
            self.devices
        )

    @property
    def available_devices(
        self,
    ) -> Tuple[
        AcceleratorDevice,
        ...,
    ]:
        return tuple(
            device
            for device
            in self.devices
            if device.available
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "backend": (
                self.backend
            ),

            "available": (
                self.available
            ),

            "device_count": (
                self.device_count
            ),

            "available_device_count": len(
                self.available_devices
            ),

            "devices": [
                device.to_dict()
                for device
                in self.devices
            ],

            "error_type": (
                self.error_type
            ),

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Accelerator backend
# ---------------------------------------------------------------

class AcceleratorBackend(
    ABC
):
    """
    Abstract contract for one accelerator backend.

    Concrete implementations own all
    backend-specific hardware/runtime behavior.

    The rest of OSHA should interact with devices
    through AcceleratorDevice and this interface
    rather than directly calling CUDA/ROCm APIs.
    """

    kind: str

    def __init__(
        self,
        *,
        config: Optional[
            AcceleratorConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or AcceleratorConfig()
        )

        self.config.validate()

        self._lock = RLock()

        if (
            self.kind
            not in {
                AcceleratorKind.
                CPU.value,

                AcceleratorKind.
                CUDA.value,

                AcceleratorKind.
                ROCM.value,
            }
        ):
            raise ValueError(
                "AcceleratorBackend.kind "
                "must be cpu, cuda, or "
                "rocm."
            )

    # ---------------------------------------------------------
    # Backend identity
    # ---------------------------------------------------------

    @property
    def backend_name(
        self,
    ) -> str:
        return self.kind

    # ---------------------------------------------------------
    # Availability
    # ---------------------------------------------------------

    @abstractmethod
    def available(
        self,
    ) -> bool:
        """
        Return whether this backend runtime is
        currently usable.

        Examples:

            CPU:
                normally True

            CUDA:
                runtime present and at least one
                usable device visible

            ROCm:
                compatible runtime and usable
                device visible
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Raw discovery
    # ---------------------------------------------------------

    @abstractmethod
    def _discover(
        self,
    ) -> Tuple[
        AcceleratorDevice,
        ...,
    ]:
        """
        Backend-specific hardware discovery.

        Concrete implementations must translate
        runtime/framework details into
        AcceleratorDevice objects.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Public discovery
    # ---------------------------------------------------------

    def discover(
        self,
    ) -> BackendDiscoveryResult:
        """
        Discover devices from this backend.

        Backend unavailability returns a valid
        discovery result when configuration allows
        graceful discovery.

        Unexpected discovery errors are raised
        only when fail_on_backend_error is enabled.
        """

        with self._lock:
            try:
                backend_available = (
                    self.available()
                )

            except Exception as exc:
                if (
                    self.config.
                    discovery.
                    fail_on_backend_error
                ):
                    raise (
                        AcceleratorBackendUnavailableError(
                            "Accelerator backend "
                            "availability check "
                            "failed.",
                            details=(
                                safe_accelerator_error_details(
                                    exc,
                                    operation=(
                                        "availability"
                                    ),
                                    backend=(
                                        self.kind
                                    ),
                                )
                            ),
                        )
                    ) from exc

                return BackendDiscoveryResult(
                    backend=self.kind,
                    devices=(),
                    available=False,
                    error_type=(
                        type(exc).__name__
                    ),
                    attributes={
                        "availability_check_failed": (
                            True
                        ),
                    },
                )

            if not backend_available:
                return BackendDiscoveryResult(
                    backend=self.kind,
                    devices=(),
                    available=False,
                    attributes={
                        "backend_unavailable": (
                            True
                        ),
                    },
                )

            try:
                devices = tuple(
                    self._discover()
                )

                for device in devices:
                    device.validate()

                    if (
                        device.kind
                        != self.kind
                    ):
                        raise ValueError(
                            "Backend returned a "
                            "device from another "
                            "accelerator kind."
                        )

                return BackendDiscoveryResult(
                    backend=self.kind,
                    devices=devices,
                    available=True,
                )

            except Exception as exc:
                if (
                    self.config.
                    discovery.
                    fail_on_backend_error
                ):
                    if isinstance(
                        exc,
                        AcceleratorBackendOperationError,
                    ):
                        raise

                    raise (
                        AcceleratorBackendOperationError(
                            "Accelerator backend "
                            "device discovery "
                            "failed.",
                            details=(
                                safe_accelerator_error_details(
                                    exc,
                                    operation=(
                                        "discover"
                                    ),
                                    backend=(
                                        self.kind
                                    ),
                                )
                            ),
                        )
                    ) from exc

                return BackendDiscoveryResult(
                    backend=self.kind,
                    devices=(),
                    available=True,
                    error_type=(
                        type(exc).__name__
                    ),
                    attributes={
                        "discovery_failed": (
                            True
                        ),
                    },
                )

    # ---------------------------------------------------------
    # Device lookup
    # ---------------------------------------------------------

    def device(
        self,
        device_id: str,
    ) -> AcceleratorDevice:
        """
        Resolve one device from a fresh backend
        discovery snapshot.
        """

        expected = str(
            device_id
        ).strip()

        result = self.discover()

        for device in result.devices:
            if (
                device.device_id
                == expected
            ):
                return device

        raise AcceleratorDeviceNotFoundError(
            "Accelerator device was not "
            "found in this backend.",
            details={
                "backend": self.kind,
                "device_id": expected,
            },
        )

    # ---------------------------------------------------------
    # Ensure usable device
    # ---------------------------------------------------------

    def require_available_device(
        self,
        device: AcceleratorDevice,
    ) -> None:
        device.validate()

        if (
            device.kind
            != self.kind
        ):
            raise AcceleratorDeviceUnavailableError(
                "Accelerator device does "
                "not belong to this "
                "backend.",
                details={
                    "backend": self.kind,
                    "device_id": (
                        device.device_id
                    ),
                    "device_kind": (
                        device.kind
                    ),
                },
            )

        if not device.available:
            raise AcceleratorDeviceUnavailableError(
                "Accelerator device is "
                "not available.",
                details={
                    "backend": self.kind,
                    "device_id": (
                        device.device_id
                    ),
                    "state": (
                        device.state
                    ),
                },
            )

    # ---------------------------------------------------------
    # Memory query implementation
    # ---------------------------------------------------------

    @abstractmethod
    def _memory(
        self,
        device: AcceleratorDevice,
    ) -> AcceleratorMemory:
        """
        Backend-specific memory query.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Public memory query
    # ---------------------------------------------------------

    def memory(
        self,
        device: AcceleratorDevice,
    ) -> AcceleratorMemory:
        self.require_available_device(
            device
        )

        if (
            self.config.
            runtime.
            synchronize_before_memory_query
        ):
            self.synchronize(
                device
            )

        try:
            memory = self._memory(
                device
            )

            if not isinstance(
                memory,
                AcceleratorMemory,
            ):
                raise TypeError(
                    "Backend memory query "
                    "must return "
                    "AcceleratorMemory."
                )

            memory.validate()

            return memory

        except AcceleratorMemoryQueryError:
            raise

        except Exception as exc:
            raise AcceleratorMemoryQueryError(
                "Accelerator memory query "
                "failed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation="memory",
                        backend=self.kind,
                        device_id=(
                            device.device_id
                        ),
                        runtime_device=(
                            device.
                            runtime_device
                        ),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Refresh one device
    # ---------------------------------------------------------

    def refresh(
        self,
        device: AcceleratorDevice,
    ) -> AcceleratorDevice:
        """
        Refresh volatile device properties.

        Base implementation currently refreshes
        memory while preserving stable hardware
        identity and capabilities.
        """

        self.require_available_device(
            device
        )

        memory = self.memory(
            device
        )

        return AcceleratorDevice(
            device_id=(
                device.device_id
            ),

            kind=device.kind,

            runtime_device=(
                device.runtime_device
            ),

            index=device.index,

            name=device.name,

            state=device.state,

            memory=memory,

            capabilities=(
                device.capabilities
            ),

            supported_precisions=(
                device.
                supported_precisions
            ),

            compute_capability=(
                device.
                compute_capability
            ),

            vendor=device.vendor,

            architecture=(
                device.architecture
            ),

            driver_version=(
                device.driver_version
            ),

            runtime_version=(
                device.runtime_version
            ),

            logical_processor_count=(
                device.
                logical_processor_count
            ),

            multiprocessor_count=(
                device.
                multiprocessor_count
            ),

            attributes=dict(
                device.attributes
            ),
        )

    # ---------------------------------------------------------
    # Runtime-device implementation
    # ---------------------------------------------------------

    @abstractmethod
    def _runtime_device(
        self,
        device: AcceleratorDevice,
    ) -> Any:
        """
        Build the native framework runtime device.

        This is intentionally typed as Any.

        For PyTorch backends it may return
        torch.device. A future backend could return
        another runtime's device representation.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Public runtime device
    # ---------------------------------------------------------

    def runtime_device(
        self,
        device: AcceleratorDevice,
    ) -> Any:
        self.require_available_device(
            device
        )

        try:
            return self._runtime_device(
                device
            )

        except AcceleratorRuntimeDeviceError:
            raise

        except Exception as exc:
            raise AcceleratorRuntimeDeviceError(
                "Accelerator runtime "
                "device could not be "
                "constructed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=(
                            "runtime_device"
                        ),
                        backend=self.kind,
                        device_id=(
                            device.device_id
                        ),
                        runtime_device=(
                            device.
                            runtime_device
                        ),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------

    def _synchronize(
        self,
        device: AcceleratorDevice,
    ) -> None:
        """
        Optional backend-specific synchronization.

        CPU backends may keep this as a no-op.
        """

    def synchronize(
        self,
        device: AcceleratorDevice,
    ) -> None:
        self.require_available_device(
            device
        )

        try:
            self._synchronize(
                device
            )

        except AcceleratorSynchronizationError:
            raise

        except Exception as exc:
            raise AcceleratorSynchronizationError(
                "Accelerator "
                "synchronization failed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=(
                            "synchronize"
                        ),
                        backend=self.kind,
                        device_id=(
                            device.device_id
                        ),
                        runtime_device=(
                            device.
                            runtime_device
                        ),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Cache management
    # ---------------------------------------------------------

    def _clear_cache(
        self,
        device: AcceleratorDevice,
    ) -> None:
        """
        Optional backend-specific cache clearing.

        CPU normally leaves this as a no-op.
        """

    def clear_cache(
        self,
        device: AcceleratorDevice,
    ) -> None:
        self.require_available_device(
            device
        )

        try:
            self._clear_cache(
                device
            )

        except AcceleratorCacheClearError:
            raise

        except Exception as exc:
            raise AcceleratorCacheClearError(
                "Accelerator cache could "
                "not be cleared.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=(
                            "clear_cache"
                        ),
                        backend=self.kind,
                        device_id=(
                            device.device_id
                        ),
                        runtime_device=(
                            device.
                            runtime_device
                        ),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Release
    # ---------------------------------------------------------

    def release(
        self,
        device: AcceleratorDevice,
    ) -> None:
        """
        Release backend-level transient resources.

        This does not destroy hardware context or
        delete model objects owned by callers.
        """

        self.require_available_device(
            device
        )

        if (
            self.config.
            runtime.
            clear_cache_on_release
        ):
            self.clear_cache(
                device
            )

    # ---------------------------------------------------------
    # Allocation failure handling
    # ---------------------------------------------------------

    def allocation_failure(
        self,
        device: AcceleratorDevice,
    ) -> None:
        """
        Optional recovery hook after caller-level
        allocation failure.

        The manager/training/AI layer can invoke
        this before deciding whether to retry or
        fall back.
        """

        self.require_available_device(
            device
        )

        if (
            self.config.
            runtime.
            clear_cache_on_allocation_failure
        ):
            self.clear_cache(
                device
            )

    # ---------------------------------------------------------
    # Health implementation
    # ---------------------------------------------------------

    def _health(
        self,
        *,
        deep: bool,
    ) -> AcceleratorHealth:
        """
        Optional backend-specific health
        implementation.

        Concrete implementations may override
        this for backend-specific diagnostics.
        """

        available = self.available()

        return AcceleratorHealth(
            component=(
                "accelerator_backend:"
                + self.kind
            ),

            status=(
                AcceleratorHealthStatus.
                HEALTHY.value
                if available
                else AcceleratorHealthStatus.
                UNAVAILABLE.value
            ),

            message=(
                "Accelerator backend is "
                "available."
                if available
                else (
                    "Accelerator backend "
                    "is unavailable."
                )
            ),

            kind=self.kind,

            attributes={
                "deep_check": (
                    deep
                ),
            },
        )

    # ---------------------------------------------------------
    # Public health
    # ---------------------------------------------------------

    def health(
        self,
        *,
        deep: Optional[
            bool
        ] = None,
    ) -> AcceleratorHealth:
        """
        Check backend health.

        Backends translate runtime failures into
        structured AcceleratorHealth rather than
        exposing arbitrary framework exceptions.
        """

        deep_check = (
            self.config.
            health.deep_check
            if deep is None
            else bool(
                deep
            )
        )

        try:
            health = self._health(
                deep=deep_check
            )

            if not isinstance(
                health,
                AcceleratorHealth,
            ):
                raise TypeError(
                    "Backend health check "
                    "must return "
                    "AcceleratorHealth."
                )

            health.validate()

            return health

        except Exception as exc:
            if isinstance(
                exc,
                AcceleratorHealthCheckError,
            ):
                raise

            if (
                self.config.
                discovery.
                fail_on_backend_error
            ):
                raise AcceleratorHealthCheckError(
                    "Accelerator backend "
                    "health check failed.",
                    details=(
                        safe_accelerator_error_details(
                            exc,
                            operation="health",
                            backend=self.kind,
                        )
                    ),
                ) from exc

            return AcceleratorHealth(
                component=(
                    "accelerator_backend:"
                    + self.kind
                ),

                status=(
                    AcceleratorHealthStatus.
                    UNHEALTHY.value
                ),

                message=(
                    "Accelerator backend "
                    "health check failed."
                ),

                kind=self.kind,

                retryable=True,

                error_type=(
                    type(exc).__name__
                ),

                attributes={
                    "deep_check": (
                        deep_check
                    ),
                },
            )

    # ---------------------------------------------------------
    # Backend summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        """
        Lightweight backend summary.

        This method may perform discovery but does
        not allocate models or run workloads.
        """

        discovery = self.discover()

        return {
            "kind": self.kind,

            "available": (
                discovery.available
            ),

            "device_count": (
                discovery.device_count
            ),

            "available_device_count": len(
                discovery.
                available_devices
            ),

            "error_type": (
                discovery.error_type
            ),

            "devices": [
                {
                    "device_id": (
                        device.device_id
                    ),

                    "runtime_device": (
                        device.
                        runtime_device
                    ),

                    "state": (
                        device.state
                    ),

                    "name": (
                        device.name
                    ),
                }
                for device
                in discovery.devices
            ],
        }


# ---------------------------------------------------------------
# CPU backend base marker
# ---------------------------------------------------------------

class CPUAcceleratorBackend(
    AcceleratorBackend
):
    """
    Base marker for CPU implementations.
    """

    kind = (
        AcceleratorKind.CPU.value
    )


# ---------------------------------------------------------------
# CUDA backend base marker
# ---------------------------------------------------------------

class CUDAAcceleratorBackend(
    AcceleratorBackend
):
    """
    Base marker for NVIDIA CUDA implementations.
    """

    kind = (
        AcceleratorKind.CUDA.value
    )


# ---------------------------------------------------------------
# ROCm backend base marker
# ---------------------------------------------------------------

class ROCmAcceleratorBackend(
    AcceleratorBackend
):
    """
    Base marker for AMD ROCm implementations.
    """

    kind = (
        AcceleratorKind.ROCM.value
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "BackendDiscoveryResult",

    "AcceleratorBackend",

    "CPUAcceleratorBackend",
    "CUDAAcceleratorBackend",
    "ROCmAcceleratorBackend",
]
