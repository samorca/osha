from __future__ import annotations

"""
Configuration for OSHA AI model serving.

This configuration controls:

    promoted model discovery
    device selection
    model loading/cache policy
    inference limits
    anomaly threshold defaults
    runtime replacement policy

Model training configuration remains under
src/training/config.py.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

from src.core.validation import (
    boolean,
    integer,
    number,
    unique_strings,
)

from src.ai.schema import (
    AIDeviceFamily,
    AIModelFamily,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = "OSHA_AI_"


# ---------------------------------------------------------------
# Supported devices
# ---------------------------------------------------------------

SUPPORTED_DEVICES = (
    AIDeviceFamily.AUTO.value,
    AIDeviceFamily.CPU.value,
    AIDeviceFamily.CUDA.value,
    AIDeviceFamily.ROCM.value,
)


# ---------------------------------------------------------------
# Model path configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AIPathConfig:
    """
    Filesystem locations used by AI serving.

    models_directory should normally point at
    promoted models, not training checkpoints.
    """

    models_directory: str = "models"

    cache_directory: str = (
        "artifacts/ai-cache"
    )

    log_directory: str = (
        "logs/ai"
    )

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "models_directory",
                self.models_directory,
            ),
            (
                "cache_directory",
                self.cache_directory,
            ),
            (
                "log_directory",
                self.log_directory,
            ),
        ):
            if not str(
                value
            ).strip():
                raise ValueError(
                    f"{name} cannot be empty"
                )

    def resolved(
        self,
    ) -> Dict[str, str]:
        self.validate()

        return {
            "models": str(
                Path(
                    self.models_directory
                ).expanduser().resolve()
            ),

            "cache": str(
                Path(
                    self.cache_directory
                ).expanduser().resolve()
            ),

            "logs": str(
                Path(
                    self.log_directory
                ).expanduser().resolve()
            ),
        }


# ---------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AIDeviceConfig:
    """
    Model-serving device policy.

    Actual hardware inspection remains the
    responsibility of src/accelerators/.
    """

    preferred: str = (
        AIDeviceFamily.AUTO.value
    )

    device_index: Optional[
        int
    ] = None

    allow_cpu_fallback: bool = True

    mixed_precision: bool = False

    def validate(
        self,
    ) -> None:
        preferred = str(
            self.preferred
        ).strip().lower()

        if (
            preferred
            not in SUPPORTED_DEVICES
        ):
            raise ValueError(
                "Unsupported AI device: "
                f"{self.preferred}"
            )

        if self.device_index is not None:
            integer(
                self.device_index,
                name="device_index",
                minimum=0,
            )


# ---------------------------------------------------------------
# Model loading/cache configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AIModelLoadingConfig:
    """
    Controls runtime model loading.

    Lazy loading keeps models unloaded until they
    are first required for inference.
    """

    lazy_loading: bool = True

    verify_checksum: bool = True

    maximum_loaded_models: int = 4

    unload_unused_models: bool = True

    allow_runtime_replacement: bool = False

    require_promoted_state: bool = True

    def validate(
        self,
    ) -> None:
        integer(
            self.maximum_loaded_models,
            name="maximum_loaded_models",
            minimum=1,
        )


# ---------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AIInferenceConfig:
    """
    Generic serving limits.

    These are service/runtime limits, not model
    training parameters.
    """

    maximum_batch_size: int = 64

    maximum_payload_bytes: int = (
        4
        * 1024
        * 1024
    )

    timeout_seconds: float = 30.0

    enable_batch_inference: bool = True

    reject_non_finite_inputs: bool = True

    include_representations: bool = False

    def validate(
        self,
    ) -> None:
        integer(
            self.maximum_batch_size,
            name="maximum_batch_size",
            minimum=1,
        )

        integer(
            self.maximum_payload_bytes,
            name="maximum_payload_bytes",
            minimum=1,
        )

        number(
            self.timeout_seconds,
            name="timeout_seconds",
            minimum=0.001,
        )


# ---------------------------------------------------------------
# Event inference configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class EventInferenceConfig:
    """
    Event-model serving defaults.
    """

    max_length: int = 128

    top_k: int = 1

    return_source_prediction: bool = True

    return_component_prediction: bool = True

    return_event_prediction: bool = True

    def validate(
        self,
    ) -> None:
        integer(
            self.max_length,
            name="event max_length",
            minimum=1,
        )

        integer(
            self.top_k,
            name="event top_k",
            minimum=1,
        )


# ---------------------------------------------------------------
# Anomaly inference configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AnomalyInferenceConfig:
    """
    Anomaly-model serving defaults.

    default_threshold is a fallback serving
    threshold. A promoted model artifact may later
    carry a validated/calibrated threshold that
    should take precedence.
    """

    default_threshold: float = 0.50

    minimum_threshold: float = 0.0

    maximum_threshold: float = 1.0

    allow_request_threshold_override: bool = (
        False
    )

    def validate(
        self,
    ) -> None:
        threshold = number(
            self.default_threshold,
            name="default_threshold",
            minimum=0.0,
            maximum=1.0,
        )

        minimum = number(
            self.minimum_threshold,
            name="minimum_threshold",
            minimum=0.0,
            maximum=1.0,
        )

        maximum = number(
            self.maximum_threshold,
            name="maximum_threshold",
            minimum=0.0,
            maximum=1.0,
        )

        if minimum > maximum:
            raise ValueError(
                "minimum_threshold "
                "cannot be greater than "
                "maximum_threshold"
            )

        if not (
            minimum
            <= threshold
            <= maximum
        ):
            raise ValueError(
                "default_threshold must "
                "fall within configured "
                "threshold limits"
            )


# ---------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AIRoutingConfig:
    """
    Controls how inference requests select model
    versions.
    """

    use_latest_version_when_unspecified: bool = (
        True
    )

    require_explicit_family: bool = True

    allow_fallback_model: bool = False

    fallback_event_model: Optional[
        str
    ] = None

    fallback_anomaly_model: Optional[
        str
    ] = None

    def validate(
        self,
    ) -> None:
        if (
            self.fallback_event_model
            and not self.
            allow_fallback_model
        ):
            raise ValueError(
                "fallback_event_model "
                "requires "
                "allow_fallback_model"
            )

        if (
            self.fallback_anomaly_model
            and not self.
            allow_fallback_model
        ):
            raise ValueError(
                "fallback_anomaly_model "
                "requires "
                "allow_fallback_model"
            )


# ---------------------------------------------------------------
# Main AI configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AIConfig:
    """
    Top-level OSHA AI-serving configuration.
    """

    enabled: bool = True

    allowed_families: Tuple[
        str,
        ...,
    ] = (
        AIModelFamily.EVENT.value,
        AIModelFamily.ANOMALY.value,
    )

    paths: AIPathConfig = field(
        default_factory=AIPathConfig
    )

    device: AIDeviceConfig = field(
        default_factory=AIDeviceConfig
    )

    loading: AIModelLoadingConfig = field(
        default_factory=(
            AIModelLoadingConfig
        )
    )

    inference: AIInferenceConfig = field(
        default_factory=(
            AIInferenceConfig
        )
    )

    event: EventInferenceConfig = field(
        default_factory=(
            EventInferenceConfig
        )
    )

    anomaly: AnomalyInferenceConfig = field(
        default_factory=(
            AnomalyInferenceConfig
        )
    )

    routing: AIRoutingConfig = field(
        default_factory=(
            AIRoutingConfig
        )
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.paths.validate()

        self.device.validate()

        self.loading.validate()

        self.inference.validate()

        self.event.validate()

        self.anomaly.validate()

        self.routing.validate()

        families = unique_strings(
            self.allowed_families,
            name="allowed_families",
            allow_empty=False,
        )

        valid_families = {
            item.value
            for item
            in AIModelFamily
        }

        unsupported = [
            family
            for family
            in families
            if family
            not in valid_families
        ]

        if unsupported:
            raise ValueError(
                "Unsupported AI model "
                "families: "
                + ", ".join(
                    unsupported
                )
            )

    # ---------------------------------------------------------
    # Family enabled
    # ---------------------------------------------------------

    def family_allowed(
        self,
        family: str,
    ) -> bool:
        self.validate()

        return bool(
            self.enabled
            and str(
                family
            ).strip().lower()
            in set(
                self.allowed_families
            )
        )

    # ---------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": self.enabled,

            "allowed_families": list(
                self.allowed_families
            ),

            "paths": (
                self.paths.resolved()
            ),

            "device": {
                "preferred": (
                    self.device.preferred
                ),

                "device_index": (
                    self.device.
                    device_index
                ),

                "allow_cpu_fallback": (
                    self.device.
                    allow_cpu_fallback
                ),

                "mixed_precision": (
                    self.device.
                    mixed_precision
                ),
            },

            "loading": {
                "lazy_loading": (
                    self.loading.
                    lazy_loading
                ),

                "verify_checksum": (
                    self.loading.
                    verify_checksum
                ),

                "maximum_loaded_models": (
                    self.loading.
                    maximum_loaded_models
                ),

                "unload_unused_models": (
                    self.loading.
                    unload_unused_models
                ),

                "allow_runtime_replacement": (
                    self.loading.
                    allow_runtime_replacement
                ),

                "require_promoted_state": (
                    self.loading.
                    require_promoted_state
                ),
            },

            "inference": {
                "maximum_batch_size": (
                    self.inference.
                    maximum_batch_size
                ),

                "maximum_payload_bytes": (
                    self.inference.
                    maximum_payload_bytes
                ),

                "timeout_seconds": (
                    self.inference.
                    timeout_seconds
                ),

                "enable_batch_inference": (
                    self.inference.
                    enable_batch_inference
                ),

                "reject_non_finite_inputs": (
                    self.inference.
                    reject_non_finite_inputs
                ),

                "include_representations": (
                    self.inference.
                    include_representations
                ),
            },

            "event": {
                "max_length": (
                    self.event.max_length
                ),

                "top_k": (
                    self.event.top_k
                ),

                "return_source_prediction": (
                    self.event.
                    return_source_prediction
                ),

                "return_component_prediction": (
                    self.event.
                    return_component_prediction
                ),

                "return_event_prediction": (
                    self.event.
                    return_event_prediction
                ),
            },

            "anomaly": {
                "default_threshold": (
                    self.anomaly.
                    default_threshold
                ),

                "minimum_threshold": (
                    self.anomaly.
                    minimum_threshold
                ),

                "maximum_threshold": (
                    self.anomaly.
                    maximum_threshold
                ),

                "allow_request_threshold_override": (
                    self.anomaly.
                    allow_request_threshold_override
                ),
            },

            "routing": {
                "use_latest_version_when_unspecified": (
                    self.routing.
                    use_latest_version_when_unspecified
                ),

                "require_explicit_family": (
                    self.routing.
                    require_explicit_family
                ),

                "allow_fallback_model": (
                    self.routing.
                    allow_fallback_model
                ),

                "fallback_event_model": (
                    self.routing.
                    fallback_event_model
                ),

                "fallback_anomaly_model": (
                    self.routing.
                    fallback_anomaly_model
                ),
            },

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------

def _get(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Any = None,
) -> Any:
    return values.get(
        prefix + name,
        default,
    )


def _text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[
        str
    ] = None,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
        default,
    )

    if value is None:
        return None

    text = str(
        value
    ).strip()

    return text or None


def _bool(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: bool,
) -> bool:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(
            value
        ).strip()
        == ""
    ):
        return default

    return boolean(
        value,
        name=prefix + name,
    )


def _int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(
            value
        ).strip()
        == ""
    ):
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=minimum,
    )


def _optional_int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[
        int
    ],
) -> Optional[int]:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(
            value
        ).strip()
        == ""
    ):
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=0,
    )


def _float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
    maximum: Optional[
        float
    ] = None,
) -> float:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(
            value
        ).strip()
        == ""
    ):
        return default

    kwargs: Dict[
        str,
        Any,
    ] = {
        "name": prefix + name,
        "minimum": minimum,
    }

    if maximum is not None:
        kwargs[
            "maximum"
        ] = maximum

    return number(
        value,
        **kwargs,
    )


def _csv(
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
) -> Tuple[
    str,
    ...,
]:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(
            value
        ).strip()
        == ""
    ):
        return default

    return tuple(
        item.lower()
        for item
        in unique_strings(
            str(
                value
            ).split(
                ","
            ),
            name=prefix + name,
            allow_empty=False,
        )
    )


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------

def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> AIConfig:
    defaults = AIConfig()

    config = AIConfig(
        enabled=_bool(
            values,
            prefix,
            "ENABLED",
            True,
        ),

        allowed_families=_csv(
            values,
            prefix,
            "ALLOWED_FAMILIES",
            defaults.allowed_families,
        ),

        paths=AIPathConfig(
            models_directory=(
                _text(
                    values,
                    prefix,
                    "MODELS_DIRECTORY",
                    defaults.paths.
                    models_directory,
                )
                or defaults.paths.
                models_directory
            ),

            cache_directory=(
                _text(
                    values,
                    prefix,
                    "CACHE_DIRECTORY",
                    defaults.paths.
                    cache_directory,
                )
                or defaults.paths.
                cache_directory
            ),

            log_directory=(
                _text(
                    values,
                    prefix,
                    "LOG_DIRECTORY",
                    defaults.paths.
                    log_directory,
                )
                or defaults.paths.
                log_directory
            ),
        ),

        device=AIDeviceConfig(
            preferred=(
                _text(
                    values,
                    prefix,
                    "DEVICE",
                    defaults.device.
                    preferred,
                )
                or defaults.device.
                preferred
            ).lower(),

            device_index=(
                _optional_int(
                    values,
                    prefix,
                    "DEVICE_INDEX",
                    defaults.device.
                    device_index,
                )
            ),

            allow_cpu_fallback=(
                _bool(
                    values,
                    prefix,
                    "ALLOW_CPU_FALLBACK",
                    defaults.device.
                    allow_cpu_fallback,
                )
            ),

            mixed_precision=_bool(
                values,
                prefix,
                "MIXED_PRECISION",
                defaults.device.
                mixed_precision,
            ),
        ),

        loading=AIModelLoadingConfig(
            lazy_loading=_bool(
                values,
                prefix,
                "LAZY_LOADING",
                defaults.loading.
                lazy_loading,
            ),

            verify_checksum=_bool(
                values,
                prefix,
                "VERIFY_CHECKSUM",
                defaults.loading.
                verify_checksum,
            ),

            maximum_loaded_models=(
                _int(
                    values,
                    prefix,
                    "MAXIMUM_LOADED_MODELS",
                    defaults.loading.
                    maximum_loaded_models,
                    minimum=1,
                )
            ),

            unload_unused_models=(
                _bool(
                    values,
                    prefix,
                    "UNLOAD_UNUSED_MODELS",
                    defaults.loading.
                    unload_unused_models,
                )
            ),

            allow_runtime_replacement=(
                _bool(
                    values,
                    prefix,
                    "ALLOW_RUNTIME_REPLACEMENT",
                    defaults.loading.
                    allow_runtime_replacement,
                )
            ),

            require_promoted_state=(
                _bool(
                    values,
                    prefix,
                    "REQUIRE_PROMOTED_STATE",
                    defaults.loading.
                    require_promoted_state,
                )
            ),
        ),

        inference=AIInferenceConfig(
            maximum_batch_size=_int(
                values,
                prefix,
                "MAXIMUM_BATCH_SIZE",
                defaults.inference.
                maximum_batch_size,
                minimum=1,
            ),

            maximum_payload_bytes=_int(
                values,
                prefix,
                "MAXIMUM_PAYLOAD_BYTES",
                defaults.inference.
                maximum_payload_bytes,
                minimum=1,
            ),

            timeout_seconds=_float(
                values,
                prefix,
                "TIMEOUT_SECONDS",
                defaults.inference.
                timeout_seconds,
                minimum=0.001,
            ),

            enable_batch_inference=(
                _bool(
                    values,
                    prefix,
                    "ENABLE_BATCH_INFERENCE",
                    defaults.inference.
                    enable_batch_inference,
                )
            ),

            reject_non_finite_inputs=(
                _bool(
                    values,
                    prefix,
                    "REJECT_NON_FINITE_INPUTS",
                    defaults.inference.
                    reject_non_finite_inputs,
                )
            ),

            include_representations=(
                _bool(
                    values,
                    prefix,
                    "INCLUDE_REPRESENTATIONS",
                    defaults.inference.
                    include_representations,
                )
            ),
        ),

        event=EventInferenceConfig(
            max_length=_int(
                values,
                prefix,
                "EVENT_MAX_LENGTH",
                defaults.event.
                max_length,
                minimum=1,
            ),

            top_k=_int(
                values,
                prefix,
                "EVENT_TOP_K",
                defaults.event.
                top_k,
                minimum=1,
            ),

            return_source_prediction=(
                _bool(
                    values,
                    prefix,
                    "EVENT_RETURN_SOURCE",
                    defaults.event.
                    return_source_prediction,
                )
            ),

            return_component_prediction=(
                _bool(
                    values,
                    prefix,
                    "EVENT_RETURN_COMPONENT",
                    defaults.event.
                    return_component_prediction,
                )
            ),

            return_event_prediction=(
                _bool(
                    values,
                    prefix,
                    "EVENT_RETURN_EVENT",
                    defaults.event.
                    return_event_prediction,
                )
            ),
        ),

        anomaly=AnomalyInferenceConfig(
            default_threshold=_float(
                values,
                prefix,
                "ANOMALY_DEFAULT_THRESHOLD",
                defaults.anomaly.
                default_threshold,
                minimum=0.0,
                maximum=1.0,
            ),

            minimum_threshold=_float(
                values,
                prefix,
                "ANOMALY_MINIMUM_THRESHOLD",
                defaults.anomaly.
                minimum_threshold,
                minimum=0.0,
                maximum=1.0,
            ),

            maximum_threshold=_float(
                values,
                prefix,
                "ANOMALY_MAXIMUM_THRESHOLD",
                defaults.anomaly.
                maximum_threshold,
                minimum=0.0,
                maximum=1.0,
            ),

            allow_request_threshold_override=(
                _bool(
                    values,
                    prefix,
                    "ANOMALY_ALLOW_THRESHOLD_OVERRIDE",
                    defaults.anomaly.
                    allow_request_threshold_override,
                )
            ),
        ),

        routing=AIRoutingConfig(
            use_latest_version_when_unspecified=(
                _bool(
                    values,
                    prefix,
                    "USE_LATEST_VERSION",
                    defaults.routing.
                    use_latest_version_when_unspecified,
                )
            ),

            require_explicit_family=(
                _bool(
                    values,
                    prefix,
                    "REQUIRE_EXPLICIT_FAMILY",
                    defaults.routing.
                    require_explicit_family,
                )
            ),

            allow_fallback_model=(
                _bool(
                    values,
                    prefix,
                    "ALLOW_FALLBACK_MODEL",
                    defaults.routing.
                    allow_fallback_model,
                )
            ),

            fallback_event_model=(
                _text(
                    values,
                    prefix,
                    "FALLBACK_EVENT_MODEL",
                    defaults.routing.
                    fallback_event_model,
                )
            ),

            fallback_anomaly_model=(
                _text(
                    values,
                    prefix,
                    "FALLBACK_ANOMALY_MODEL",
                    defaults.routing.
                    fallback_anomaly_model,
                )
            ),
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
    prefix: str = DEFAULT_ENV_PREFIX,
) -> AIConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return AIConfig(
        enabled=config.enabled,

        allowed_families=(
            config.allowed_families
        ),

        paths=config.paths,

        device=config.device,

        loading=config.loading,

        inference=config.inference,

        event=config.event,

        anomaly=config.anomaly,

        routing=config.routing,

        attributes={
            **config.attributes,

            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: AIConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "enabled": (
            config.enabled
        ),

        "allowed_families": list(
            config.allowed_families
        ),

        "paths": (
            config.paths.resolved()
        ),

        "device": (
            config.device.preferred
        ),

        "device_index": (
            config.device.device_index
        ),

        "cpu_fallback": (
            config.device.
            allow_cpu_fallback
        ),

        "mixed_precision": (
            config.device.
            mixed_precision
        ),

        "lazy_loading": (
            config.loading.
            lazy_loading
        ),

        "verify_checksum": (
            config.loading.
            verify_checksum
        ),

        "maximum_loaded_models": (
            config.loading.
            maximum_loaded_models
        ),

        "runtime_model_replacement": (
            config.loading.
            allow_runtime_replacement
        ),

        "maximum_batch_size": (
            config.inference.
            maximum_batch_size
        ),

        "timeout_seconds": (
            config.inference.
            timeout_seconds
        ),

        "default_anomaly_threshold": (
            config.anomaly.
            default_threshold
        ),

        "request_threshold_override": (
            config.anomaly.
            allow_request_threshold_override
        ),

        "automatic_training": False,

        "automatic_remediation": False,
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "SUPPORTED_DEVICES",

    "AIPathConfig",
    "AIDeviceConfig",

    "AIModelLoadingConfig",
    "AIInferenceConfig",

    "EventInferenceConfig",
    "AnomalyInferenceConfig",

    "AIRoutingConfig",

    "AIConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
