from __future__ import annotations

"""
Configuration for the OSHA accelerator subsystem.

This module defines accelerator policy for:

    backend discovery
    backend priority
    device selection
    CPU fallback
    memory requirements
    precision requirements
    mixed precision
    training/inference requirements
    cache behavior
    health checks

Hardware discovery belongs to detector.py and the
backend implementations.

Device selection belongs to manager.py.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

from src.accelerators.schema import (
    AcceleratorCapability,
    AcceleratorKind,
    AcceleratorPrecision,
    AcceleratorRequirements,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = (
    "OSHA_ACCELERATOR_"
)


# ---------------------------------------------------------------
# Supported backend priority values
# ---------------------------------------------------------------

CONCRETE_BACKENDS = (
    AcceleratorKind.CUDA.value,
    AcceleratorKind.ROCM.value,
    AcceleratorKind.CPU.value,
)


DEFAULT_BACKEND_PRIORITY = (
    AcceleratorKind.CUDA.value,
    AcceleratorKind.ROCM.value,
    AcceleratorKind.CPU.value,
)


# ---------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------

def _bool(
    value: Any,
    *,
    name: str,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if value is None:
        raise ValueError(
            f"{name} cannot be None"
        )

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"{name} must be a boolean"
    )


def _integer(
    value: Any,
    *,
    name: str,
    minimum: Optional[
        int
    ] = None,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be an integer"
        )

    try:
        result = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be an integer"
        ) from exc

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    return result


def _optional_integer(
    value: Any,
    *,
    name: str,
    minimum: Optional[
        int
    ] = None,
) -> Optional[int]:
    if value is None:
        return None

    if (
        isinstance(
            value,
            str,
        )
        and not value.strip()
    ):
        return None

    return _integer(
        value,
        name=name,
        minimum=minimum,
    )


def _number(
    value: Any,
    *,
    name: str,
    minimum: Optional[
        float
    ] = None,
    maximum: Optional[
        float
    ] = None,
) -> float:
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    if (
        maximum is not None
        and result > maximum
    ):
        raise ValueError(
            f"{name} must be <= {maximum}"
        )

    return result


def _unique_tuple(
    values: Tuple[
        str,
        ...,
    ],
    *,
    name: str,
) -> Tuple[str, ...]:
    output = []
    seen = set()

    for value in values:
        text = str(
            value
        ).strip().lower()

        if not text:
            raise ValueError(
                f"{name} cannot contain "
                "empty values"
            )

        if text in seen:
            raise ValueError(
                f"{name} cannot contain "
                f"duplicate value: {text}"
            )

        seen.add(
            text
        )

        output.append(
            text
        )

    return tuple(
        output
    )


# ---------------------------------------------------------------
# Discovery configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorDiscoveryConfig:
    """
    Controls which backends participate in
    hardware discovery.

    CPU should normally remain enabled because it
    provides the portable execution fallback.
    """

    discover_cpu: bool = True

    discover_cuda: bool = True

    discover_rocm: bool = True

    include_unavailable_devices: bool = False

    refresh_memory_on_detection: bool = True

    fail_on_backend_error: bool = False

    backend_priority: Tuple[
        str,
        ...,
    ] = (
        DEFAULT_BACKEND_PRIORITY
    )

    def validate(
        self,
    ) -> None:
        priority = _unique_tuple(
            self.backend_priority,
            name="backend_priority",
        )

        unsupported = [
            backend
            for backend
            in priority
            if backend
            not in CONCRETE_BACKENDS
        ]

        if unsupported:
            raise ValueError(
                "Unsupported accelerator "
                "backend priority values: "
                + ", ".join(
                    unsupported
                )
            )

        enabled = set()

        if self.discover_cpu:
            enabled.add(
                AcceleratorKind.
                CPU.value
            )

        if self.discover_cuda:
            enabled.add(
                AcceleratorKind.
                CUDA.value
            )

        if self.discover_rocm:
            enabled.add(
                AcceleratorKind.
                ROCM.value
            )

        if not enabled:
            raise ValueError(
                "At least one accelerator "
                "backend must be enabled"
            )

        missing = (
            enabled
            - set(
                priority
            )
        )

        if missing:
            raise ValueError(
                "backend_priority is "
                "missing enabled "
                "backends: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )


# ---------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorSelectionConfig:
    """
    Default manager-side device-selection policy.

    Callers can still supply stricter
    AcceleratorRequirements for individual
    workloads.
    """

    preferred_kind: str = (
        AcceleratorKind.AUTO.value
    )

    device_index: Optional[
        int
    ] = None

    device_id: Optional[
        str
    ] = None

    allow_cpu_fallback: bool = True

    allow_degraded_devices: bool = True

    require_accelerator: bool = False

    prefer_highest_free_memory: bool = True

    minimum_total_memory_bytes: Optional[
        int
    ] = None

    minimum_free_memory_bytes: Optional[
        int
    ] = None

    def validate(
        self,
    ) -> None:
        preferred = str(
            self.preferred_kind
        ).strip().lower()

        if (
            preferred
            not in {
                item.value
                for item
                in AcceleratorKind
            }
        ):
            raise ValueError(
                "Unsupported preferred "
                "accelerator kind: "
                f"{preferred}"
            )

        _optional_integer(
            self.device_index,
            name="device_index",
            minimum=0,
        )

        if self.device_id is not None:
            if not str(
                self.device_id
            ).strip():
                raise ValueError(
                    "device_id cannot "
                    "be empty"
                )

        _optional_integer(
            self.
            minimum_total_memory_bytes,
            name=(
                "minimum_total_memory_bytes"
            ),
            minimum=0,
        )

        _optional_integer(
            self.
            minimum_free_memory_bytes,
            name=(
                "minimum_free_memory_bytes"
            ),
            minimum=0,
        )

        if (
            self.require_accelerator
            and preferred
            == AcceleratorKind.
            CPU.value
        ):
            raise ValueError(
                "require_accelerator "
                "cannot be used with "
                "preferred_kind=cpu"
            )

        if (
            self.require_accelerator
            and self.
            allow_cpu_fallback
        ):
            raise ValueError(
                "require_accelerator and "
                "allow_cpu_fallback cannot "
                "both be enabled"
            )


# ---------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorPrecisionConfig:
    """
    Precision preferences used to build workload
    requirements.

    Mixed precision remains a workload policy,
    not something automatically enabled merely
    because a GPU supports it.
    """

    preferred_precision: str = (
        AcceleratorPrecision.
        FLOAT32.value
    )

    allow_mixed_precision: bool = False

    mixed_precision_candidates: Tuple[
        str,
        ...,
    ] = (
        AcceleratorPrecision.
        FLOAT16.value,
        AcceleratorPrecision.
        BFLOAT16.value,
    )

    require_requested_precision: bool = True

    def validate(
        self,
    ) -> None:
        valid_precisions = {
            item.value
            for item
            in AcceleratorPrecision
        }

        preferred = str(
            self.preferred_precision
        ).strip().lower()

        if preferred not in (
            valid_precisions
        ):
            raise ValueError(
                "Unsupported preferred "
                "precision: "
                f"{preferred}"
            )

        candidates = _unique_tuple(
            self.
            mixed_precision_candidates,
            name=(
                "mixed_precision_candidates"
            ),
        )

        unsupported = [
            precision
            for precision
            in candidates
            if precision
            not in valid_precisions
        ]

        if unsupported:
            raise ValueError(
                "Unsupported mixed "
                "precision candidates: "
                + ", ".join(
                    unsupported
                )
            )

        if self.allow_mixed_precision:
            allowed = {
                AcceleratorPrecision.
                FLOAT16.value,

                AcceleratorPrecision.
                BFLOAT16.value,
            }

            if not (
                set(
                    candidates
                )
                & allowed
            ):
                raise ValueError(
                    "Mixed precision must "
                    "include float16 or "
                    "bfloat16"
                )


# ---------------------------------------------------------------
# Runtime/cache configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorRuntimeConfig:
    """
    Controls safe accelerator runtime behavior.

    The backend remains responsible for deciding
    whether an operation such as cache clearing is
    meaningful for that backend.
    """

    synchronize_before_memory_query: bool = (
        False
    )

    clear_cache_on_release: bool = True

    clear_cache_on_allocation_failure: bool = (
        True
    )

    empty_cache_before_large_workload: bool = (
        False
    )

    large_workload_bytes: int = (
        4
        * 1024**3
    )

    def validate(
        self,
    ) -> None:
        _integer(
            self.large_workload_bytes,
            name="large_workload_bytes",
            minimum=1,
        )


# ---------------------------------------------------------------
# Health configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorHealthConfig:
    """
    Controls accelerator health inspection.

    Deep checks may perform backend-specific
    runtime initialization and should therefore
    remain opt-in.
    """

    enabled: bool = True

    deep_check: bool = False

    check_memory: bool = True

    require_cpu: bool = True

    require_preferred_backend: bool = False

    degraded_is_usable: bool = True

    minimum_free_memory_bytes: Optional[
        int
    ] = None

    def validate(
        self,
    ) -> None:
        _optional_integer(
            self.
            minimum_free_memory_bytes,
            name=(
                "health "
                "minimum_free_memory_bytes"
            ),
            minimum=0,
        )


# ---------------------------------------------------------------
# Top-level accelerator configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorConfig:
    """
    Top-level accelerator subsystem configuration.

    This configuration is shared by detector.py,
    manager.py and backend implementations.

    training/ and ai/ are expected to translate
    their local workload policy into
    AcceleratorRequirements rather than
    reimplementing hardware discovery.
    """

    enabled: bool = True

    discovery: AcceleratorDiscoveryConfig = field(
        default_factory=(
            AcceleratorDiscoveryConfig
        )
    )

    selection: AcceleratorSelectionConfig = field(
        default_factory=(
            AcceleratorSelectionConfig
        )
    )

    precision: AcceleratorPrecisionConfig = field(
        default_factory=(
            AcceleratorPrecisionConfig
        )
    )

    runtime: AcceleratorRuntimeConfig = field(
        default_factory=(
            AcceleratorRuntimeConfig
        )
    )

    health: AcceleratorHealthConfig = field(
        default_factory=(
            AcceleratorHealthConfig
        )
    )

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.discovery.validate()

        self.selection.validate()

        self.precision.validate()

        self.runtime.validate()

        self.health.validate()

        priority = (
            self.discovery.
            backend_priority
        )

        preferred = (
            self.selection.
            preferred_kind
        )

        if (
            preferred
            != AcceleratorKind.
            AUTO.value
            and preferred
            not in priority
        ):
            raise ValueError(
                "Preferred accelerator "
                "kind is not present in "
                "backend_priority"
            )

        if (
            preferred
            == AcceleratorKind.
            CPU.value
            and not self.discovery.
            discover_cpu
        ):
            raise ValueError(
                "CPU is preferred but CPU "
                "discovery is disabled"
            )

        if (
            preferred
            == AcceleratorKind.
            CUDA.value
            and not self.discovery.
            discover_cuda
        ):
            raise ValueError(
                "CUDA is preferred but "
                "CUDA discovery is "
                "disabled"
            )

        if (
            preferred
            == AcceleratorKind.
            ROCM.value
            and not self.discovery.
            discover_rocm
        ):
            raise ValueError(
                "ROCm is preferred but "
                "ROCm discovery is "
                "disabled"
            )

    # ---------------------------------------------------------
    # Build generic requirements
    # ---------------------------------------------------------

    def requirements(
        self,
        *,
        training: bool = False,
        inference: bool = False,
        minimum_total_memory_bytes: Optional[
            int
        ] = None,
        minimum_free_memory_bytes: Optional[
            int
        ] = None,
        preferred_kind: Optional[
            str
        ] = None,
        device_index: Optional[
            int
        ] = None,
        device_id: Optional[
            str
        ] = None,
        require_accelerator: Optional[
            bool
        ] = None,
        allow_cpu_fallback: Optional[
            bool
        ] = None,
        mixed_precision: Optional[
            bool
        ] = None,
        required_capabilities: Tuple[
            str,
            ...,
        ] = (),
        required_precisions: Tuple[
            str,
            ...,
        ] = (),
    ) -> AcceleratorRequirements:
        """
        Build workload requirements by combining
        configured defaults with caller-specific
        constraints.
        """

        self.validate()

        capabilities = list(
            required_capabilities
        )

        if training:
            capabilities.append(
                AcceleratorCapability.
                TRAINING.value
            )

        if inference:
            capabilities.append(
                AcceleratorCapability.
                INFERENCE.value
            )

        use_mixed_precision = (
            self.precision.
            allow_mixed_precision
            if mixed_precision
            is None
            else bool(
                mixed_precision
            )
        )

        precisions = list(
            required_precisions
        )

        if not precisions:
            if use_mixed_precision:
                precisions.append(
                    self.precision.
                    mixed_precision_candidates[
                        0
                    ]
                )

            elif (
                self.precision.
                require_requested_precision
            ):
                precisions.append(
                    self.precision.
                    preferred_precision
                )

        if use_mixed_precision:
            capabilities.append(
                AcceleratorCapability.
                MIXED_PRECISION.value
            )

        requirements = (
            AcceleratorRequirements(
                preferred_kind=(
                    preferred_kind
                    or self.selection.
                    preferred_kind
                ),

                device_index=(
                    device_index
                    if device_index
                    is not None
                    else self.selection.
                    device_index
                ),

                device_id=(
                    device_id
                    if device_id
                    is not None
                    else self.selection.
                    device_id
                ),

                allow_cpu_fallback=(
                    self.selection.
                    allow_cpu_fallback
                    if allow_cpu_fallback
                    is None
                    else bool(
                        allow_cpu_fallback
                    )
                ),

                require_accelerator=(
                    self.selection.
                    require_accelerator
                    if require_accelerator
                    is None
                    else bool(
                        require_accelerator
                    )
                ),

                minimum_total_memory_bytes=(
                    minimum_total_memory_bytes
                    if minimum_total_memory_bytes
                    is not None
                    else self.selection.
                    minimum_total_memory_bytes
                ),

                minimum_free_memory_bytes=(
                    minimum_free_memory_bytes
                    if minimum_free_memory_bytes
                    is not None
                    else self.selection.
                    minimum_free_memory_bytes
                ),

                required_capabilities=tuple(
                    dict.fromkeys(
                        capabilities
                    )
                ),

                required_precisions=tuple(
                    dict.fromkeys(
                        precisions
                    )
                ),

                mixed_precision=(
                    use_mixed_precision
                ),

                training=training,

                inference=inference,

                allow_degraded=(
                    self.selection.
                    allow_degraded_devices
                ),

                attributes={
                    "source": (
                        "accelerator_config"
                    ),
                },
            )
        )

        requirements.validate()

        return requirements

    # ---------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": (
                self.enabled
            ),

            "discovery": {
                "discover_cpu": (
                    self.discovery.
                    discover_cpu
                ),

                "discover_cuda": (
                    self.discovery.
                    discover_cuda
                ),

                "discover_rocm": (
                    self.discovery.
                    discover_rocm
                ),

                "include_unavailable_devices": (
                    self.discovery.
                    include_unavailable_devices
                ),

                "refresh_memory_on_detection": (
                    self.discovery.
                    refresh_memory_on_detection
                ),

                "fail_on_backend_error": (
                    self.discovery.
                    fail_on_backend_error
                ),

                "backend_priority": list(
                    self.discovery.
                    backend_priority
                ),
            },

            "selection": {
                "preferred_kind": (
                    self.selection.
                    preferred_kind
                ),

                "device_index": (
                    self.selection.
                    device_index
                ),

                "device_id": (
                    self.selection.
                    device_id
                ),

                "allow_cpu_fallback": (
                    self.selection.
                    allow_cpu_fallback
                ),

                "allow_degraded_devices": (
                    self.selection.
                    allow_degraded_devices
                ),

                "require_accelerator": (
                    self.selection.
                    require_accelerator
                ),

                "prefer_highest_free_memory": (
                    self.selection.
                    prefer_highest_free_memory
                ),

                "minimum_total_memory_bytes": (
                    self.selection.
                    minimum_total_memory_bytes
                ),

                "minimum_free_memory_bytes": (
                    self.selection.
                    minimum_free_memory_bytes
                ),
            },

            "precision": {
                "preferred_precision": (
                    self.precision.
                    preferred_precision
                ),

                "allow_mixed_precision": (
                    self.precision.
                    allow_mixed_precision
                ),

                "mixed_precision_candidates": list(
                    self.precision.
                    mixed_precision_candidates
                ),

                "require_requested_precision": (
                    self.precision.
                    require_requested_precision
                ),
            },

            "runtime": {
                "synchronize_before_memory_query": (
                    self.runtime.
                    synchronize_before_memory_query
                ),

                "clear_cache_on_release": (
                    self.runtime.
                    clear_cache_on_release
                ),

                "clear_cache_on_allocation_failure": (
                    self.runtime.
                    clear_cache_on_allocation_failure
                ),

                "empty_cache_before_large_workload": (
                    self.runtime.
                    empty_cache_before_large_workload
                ),

                "large_workload_bytes": (
                    self.runtime.
                    large_workload_bytes
                ),
            },

            "health": {
                "enabled": (
                    self.health.enabled
                ),

                "deep_check": (
                    self.health.deep_check
                ),

                "check_memory": (
                    self.health.
                    check_memory
                ),

                "require_cpu": (
                    self.health.
                    require_cpu
                ),

                "require_preferred_backend": (
                    self.health.
                    require_preferred_backend
                ),

                "degraded_is_usable": (
                    self.health.
                    degraded_is_usable
                ),

                "minimum_free_memory_bytes": (
                    self.health.
                    minimum_free_memory_bytes
                ),
            },

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Environment parsing
# ---------------------------------------------------------------

def _value(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
) -> Any:
    return values.get(
        prefix
        + name
    )


def _environment_bool(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: bool,
) -> bool:
    value = _value(
        values,
        prefix,
        name,
    )

    if value is None:
        return default

    return _bool(
        value,
        name=(
            prefix
            + name
        ),
    )


def _environment_int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[
        int
    ],
    *,
    minimum: int = 0,
) -> Optional[int]:
    value = _value(
        values,
        prefix,
        name,
    )

    if value is None:
        return default

    return _optional_integer(
        value,
        name=(
            prefix
            + name
        ),
        minimum=minimum,
    )


def _environment_text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[
        str
    ],
) -> Optional[str]:
    value = _value(
        values,
        prefix,
        name,
    )

    if value is None:
        return default

    text = str(
        value
    ).strip()

    return (
        text
        if text
        else default
    )


def _environment_csv(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Tuple[
        str,
        ...,
    ],
) -> Tuple[str, ...]:
    value = _value(
        values,
        prefix,
        name,
    )

    if value is None:
        return default

    parsed = tuple(
        item.strip().lower()
        for item
        in str(
            value
        ).split(
            ","
        )
        if item.strip()
    )

    if not parsed:
        return default

    return parsed


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------

def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = (
        DEFAULT_ENV_PREFIX
    ),
) -> AcceleratorConfig:
    """
    Build AcceleratorConfig from a flat mapping,
    normally environment variables.
    """

    defaults = AcceleratorConfig()

    config = AcceleratorConfig(
        enabled=(
            _environment_bool(
                values,
                prefix,
                "ENABLED",
                defaults.enabled,
            )
        ),

        discovery=(
            AcceleratorDiscoveryConfig(
                discover_cpu=(
                    _environment_bool(
                        values,
                        prefix,
                        "DISCOVER_CPU",
                        defaults.
                        discovery.
                        discover_cpu,
                    )
                ),

                discover_cuda=(
                    _environment_bool(
                        values,
                        prefix,
                        "DISCOVER_CUDA",
                        defaults.
                        discovery.
                        discover_cuda,
                    )
                ),

                discover_rocm=(
                    _environment_bool(
                        values,
                        prefix,
                        "DISCOVER_ROCM",
                        defaults.
                        discovery.
                        discover_rocm,
                    )
                ),

                include_unavailable_devices=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "INCLUDE_"
                            "UNAVAILABLE_DEVICES"
                        ),
                        defaults.
                        discovery.
                        include_unavailable_devices,
                    )
                ),

                refresh_memory_on_detection=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "REFRESH_MEMORY_"
                            "ON_DETECTION"
                        ),
                        defaults.
                        discovery.
                        refresh_memory_on_detection,
                    )
                ),

                fail_on_backend_error=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "FAIL_ON_BACKEND_"
                            "ERROR"
                        ),
                        defaults.
                        discovery.
                        fail_on_backend_error,
                    )
                ),

                backend_priority=(
                    _environment_csv(
                        values,
                        prefix,
                        "BACKEND_PRIORITY",
                        defaults.
                        discovery.
                        backend_priority,
                    )
                ),
            )
        ),

        selection=(
            AcceleratorSelectionConfig(
                preferred_kind=(
                    (
                        _environment_text(
                            values,
                            prefix,
                            "PREFERRED_KIND",
                            defaults.
                            selection.
                            preferred_kind,
                        )
                        or defaults.
                        selection.
                        preferred_kind
                    ).lower()
                ),

                device_index=(
                    _environment_int(
                        values,
                        prefix,
                        "DEVICE_INDEX",
                        defaults.
                        selection.
                        device_index,
                        minimum=0,
                    )
                ),

                device_id=(
                    _environment_text(
                        values,
                        prefix,
                        "DEVICE_ID",
                        defaults.
                        selection.
                        device_id,
                    )
                ),

                allow_cpu_fallback=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "ALLOW_CPU_"
                            "FALLBACK"
                        ),
                        defaults.
                        selection.
                        allow_cpu_fallback,
                    )
                ),

                allow_degraded_devices=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "ALLOW_DEGRADED_"
                            "DEVICES"
                        ),
                        defaults.
                        selection.
                        allow_degraded_devices,
                    )
                ),

                require_accelerator=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "REQUIRE_"
                            "ACCELERATOR"
                        ),
                        defaults.
                        selection.
                        require_accelerator,
                    )
                ),

                prefer_highest_free_memory=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "PREFER_HIGHEST_"
                            "FREE_MEMORY"
                        ),
                        defaults.
                        selection.
                        prefer_highest_free_memory,
                    )
                ),

                minimum_total_memory_bytes=(
                    _environment_int(
                        values,
                        prefix,
                        (
                            "MINIMUM_TOTAL_"
                            "MEMORY_BYTES"
                        ),
                        defaults.
                        selection.
                        minimum_total_memory_bytes,
                        minimum=0,
                    )
                ),

                minimum_free_memory_bytes=(
                    _environment_int(
                        values,
                        prefix,
                        (
                            "MINIMUM_FREE_"
                            "MEMORY_BYTES"
                        ),
                        defaults.
                        selection.
                        minimum_free_memory_bytes,
                        minimum=0,
                    )
                ),
            )
        ),

        precision=(
            AcceleratorPrecisionConfig(
                preferred_precision=(
                    (
                        _environment_text(
                            values,
                            prefix,
                            (
                                "PREFERRED_"
                                "PRECISION"
                            ),
                            defaults.
                            precision.
                            preferred_precision,
                        )
                        or defaults.
                        precision.
                        preferred_precision
                    ).lower()
                ),

                allow_mixed_precision=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "ALLOW_MIXED_"
                            "PRECISION"
                        ),
                        defaults.
                        precision.
                        allow_mixed_precision,
                    )
                ),

                mixed_precision_candidates=(
                    _environment_csv(
                        values,
                        prefix,
                        (
                            "MIXED_PRECISION_"
                            "CANDIDATES"
                        ),
                        defaults.
                        precision.
                        mixed_precision_candidates,
                    )
                ),

                require_requested_precision=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "REQUIRE_REQUESTED_"
                            "PRECISION"
                        ),
                        defaults.
                        precision.
                        require_requested_precision,
                    )
                ),
            )
        ),

        runtime=(
            AcceleratorRuntimeConfig(
                synchronize_before_memory_query=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "SYNCHRONIZE_BEFORE_"
                            "MEMORY_QUERY"
                        ),
                        defaults.
                        runtime.
                        synchronize_before_memory_query,
                    )
                ),

                clear_cache_on_release=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "CLEAR_CACHE_ON_"
                            "RELEASE"
                        ),
                        defaults.
                        runtime.
                        clear_cache_on_release,
                    )
                ),

                clear_cache_on_allocation_failure=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "CLEAR_CACHE_ON_"
                            "ALLOCATION_FAILURE"
                        ),
                        defaults.
                        runtime.
                        clear_cache_on_allocation_failure,
                    )
                ),

                empty_cache_before_large_workload=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "EMPTY_CACHE_BEFORE_"
                            "LARGE_WORKLOAD"
                        ),
                        defaults.
                        runtime.
                        empty_cache_before_large_workload,
                    )
                ),

                large_workload_bytes=(
                    int(
                        _environment_int(
                            values,
                            prefix,
                            (
                                "LARGE_WORKLOAD_"
                                "BYTES"
                            ),
                            defaults.
                            runtime.
                            large_workload_bytes,
                            minimum=1,
                        )
                    )
                ),
            )
        ),

        health=(
            AcceleratorHealthConfig(
                enabled=(
                    _environment_bool(
                        values,
                        prefix,
                        "HEALTH_ENABLED",
                        defaults.
                        health.enabled,
                    )
                ),

                deep_check=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "HEALTH_DEEP_"
                            "CHECK"
                        ),
                        defaults.
                        health.deep_check,
                    )
                ),

                check_memory=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "HEALTH_CHECK_"
                            "MEMORY"
                        ),
                        defaults.
                        health.check_memory,
                    )
                ),

                require_cpu=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "HEALTH_REQUIRE_"
                            "CPU"
                        ),
                        defaults.
                        health.require_cpu,
                    )
                ),

                require_preferred_backend=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "HEALTH_REQUIRE_"
                            "PREFERRED_BACKEND"
                        ),
                        defaults.
                        health.
                        require_preferred_backend,
                    )
                ),

                degraded_is_usable=(
                    _environment_bool(
                        values,
                        prefix,
                        (
                            "HEALTH_DEGRADED_"
                            "IS_USABLE"
                        ),
                        defaults.
                        health.
                        degraded_is_usable,
                    )
                ),

                minimum_free_memory_bytes=(
                    _environment_int(
                        values,
                        prefix,
                        (
                            "HEALTH_MINIMUM_"
                            "FREE_MEMORY_BYTES"
                        ),
                        defaults.
                        health.
                        minimum_free_memory_bytes,
                        minimum=0,
                    )
                ),
            )
        ),

        attributes={
            "config_source": (
                "mapping"
            ),

            "environment_prefix": (
                prefix
            ),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------

def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = (
        DEFAULT_ENV_PREFIX
    ),
) -> AcceleratorConfig:
    """
    Build configuration from process environment.
    """

    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return AcceleratorConfig(
        enabled=config.enabled,

        discovery=(
            config.discovery
        ),

        selection=(
            config.selection
        ),

        precision=(
            config.precision
        ),

        runtime=config.runtime,

        health=config.health,

        attributes={
            **dict(
                config.attributes
            ),

            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: AcceleratorConfig,
) -> Dict[str, Any]:
    """
    Return operator-friendly accelerator policy
    without framework/runtime internals.
    """

    config.validate()

    return {
        "enabled": (
            config.enabled
        ),

        "backend_priority": list(
            config.discovery.
            backend_priority
        ),

        "discover_cpu": (
            config.discovery.
            discover_cpu
        ),

        "discover_cuda": (
            config.discovery.
            discover_cuda
        ),

        "discover_rocm": (
            config.discovery.
            discover_rocm
        ),

        "preferred_kind": (
            config.selection.
            preferred_kind
        ),

        "device_index": (
            config.selection.
            device_index
        ),

        "device_id": (
            config.selection.
            device_id
        ),

        "allow_cpu_fallback": (
            config.selection.
            allow_cpu_fallback
        ),

        "require_accelerator": (
            config.selection.
            require_accelerator
        ),

        "prefer_highest_free_memory": (
            config.selection.
            prefer_highest_free_memory
        ),

        "minimum_total_memory_bytes": (
            config.selection.
            minimum_total_memory_bytes
        ),

        "minimum_free_memory_bytes": (
            config.selection.
            minimum_free_memory_bytes
        ),

        "preferred_precision": (
            config.precision.
            preferred_precision
        ),

        "mixed_precision": (
            config.precision.
            allow_mixed_precision
        ),

        "health_enabled": (
            config.health.enabled
        ),

        "deep_health_check": (
            config.health.
            deep_check
        ),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",

    "CONCRETE_BACKENDS",
    "DEFAULT_BACKEND_PRIORITY",

    "AcceleratorDiscoveryConfig",
    "AcceleratorSelectionConfig",
    "AcceleratorPrecisionConfig",
    "AcceleratorRuntimeConfig",
    "AcceleratorHealthConfig",

    "AcceleratorConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
