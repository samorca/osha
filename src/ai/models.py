from __future__ import annotations

"""
Concrete OSHA AI runtime model adapters.

Hardware discovery and device selection are
delegated to:

    src.accelerators.AcceleratorManager

Implemented model families:

    event
        text
        -> Vocabulary
        -> InfrastructureEventModel
        -> event/source/component predictions

    anomaly
        numeric features
        -> InfrastructureAnomalyModel
        -> anomaly probability and classification

These adapters load and execute promoted models.

They do not:

    train models
    evaluate candidates
    promote artifacts
    interpret incidents
    authorize remediation
"""

import math
import time

from pathlib import Path

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.accelerators import (
    AcceleratorConfig,
    AcceleratorManager,
    AcceleratorPrecision,
    AcceleratorSelection,
    AcceleratorSelectionConfig,
)

from src.ai.base import (
    AnomalyModelAdapter,
    EventModelAdapter,
    IncidentRootCauseModelAdapter,
    DecisionPolicyRiskModelAdapter,
    RemediationRecommendationModelAdapter,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.errors import (
    AIAnomalyFeatureError,
    AIAnomalyThresholdError,
    AIDeviceSelectionError,
    AIDeviceUnavailableError,
    AIEventTextMissingError,
    AIInferenceExecutionError,
    AIInferenceRequestError,
    AIInferenceNonFiniteInputError,
    AIInferenceShapeError,
    AILabelMapError,
    AIModelArtifactNotFoundError,
    AIModelConfigurationError,
    AIModelLoadError,
    AIVocabularyError,
    safe_ai_error_details,
)

from src.ai.schema import (
    AIModelDescriptor,
    AIModelFamily,
    AnomalyInference,
    EventInference,
    InferenceRequest,
    InferenceResult,
    InferenceState,
    Prediction,
)

from src.core.time import (
    utc_now,
)

from src.training.anomaly_model import (
    InfrastructureAnomalyModel,
)

from src.training.operational_models import (
    DecisionRiskModel,
    FEATURE_DIMENSION,
    IncidentRootCauseModel,
    RemediationRanker,
    feature_vector,
)

from src.training.event_dataset import (
    LabelMaps,
    Vocabulary,
)

from src.training.event_model import (
    InfrastructureEventModel,
)


# ------------------------------------------------------------------
# Optional PyTorch runtime
# ------------------------------------------------------------------

try:
    import torch

except ImportError:
    torch = None


# ------------------------------------------------------------------
# Runtime containers
# ------------------------------------------------------------------


class EventRuntime:
    """
    Assets required for event-model inference.
    """

    def __init__(
        self,
        *,
        model: InfrastructureEventModel,
        vocabulary: Vocabulary,
        labels: LabelMaps,
        max_length: int,
        device: Any,
        accelerator_selection: (AcceleratorSelection),
        accelerator_manager: (AcceleratorManager),
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.labels = labels

        self.max_length = int(max_length)

        self.device = device

        self.accelerator_selection = accelerator_selection

        self.accelerator_manager = accelerator_manager


class AnomalyRuntime:
    """
    Assets required for anomaly-model inference.
    """

    def __init__(
        self,
        *,
        model: InfrastructureAnomalyModel,
        input_dimension: int,
        threshold: float,
        device: Any,
        accelerator_selection: (AcceleratorSelection),
        accelerator_manager: (AcceleratorManager),
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model

        self.input_dimension = int(input_dimension)

        self.threshold = float(threshold)

        self.device = device

        self.accelerator_selection = accelerator_selection

        self.accelerator_manager = accelerator_manager

        self.feature_names = tuple(feature_names or ())


class OperationalRuntime:
    """Runtime state shared by the operational model families."""

    def __init__(
        self,
        *,
        model: Any,
        input_dimension: int,
        device: Any,
        accelerator_selection: AcceleratorSelection,
        accelerator_manager: AcceleratorManager,
        labels: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.model = model
        self.input_dimension = int(input_dimension)
        self.device = device
        self.accelerator_selection = accelerator_selection
        self.accelerator_manager = accelerator_manager
        self.labels = {
            key: tuple(value)
            for key, value in (labels or {}).items()
        }


# ------------------------------------------------------------------
# Runtime guard
# ------------------------------------------------------------------


def require_torch() -> None:
    if torch is None:
        raise AIModelLoadError("PyTorch is required for OSHA AI model serving.")


# ------------------------------------------------------------------
# Accelerator configuration translation
# ------------------------------------------------------------------


def accelerator_config_from_ai(
    config: AIConfig,
    *,
    consumer: str,
) -> AcceleratorConfig:
    """
    Translate AI-serving accelerator preferences
    into the shared accelerator subsystem.

    AIConfig remains responsible for inference
    policy.

    src.accelerators remains responsible for
    hardware discovery, device selection, and
    CPU fallback.
    """

    config.validate()

    preferred = str(config.device.preferred).strip().lower()

    allow_cpu_fallback = bool(config.device.allow_cpu_fallback)

    require_accelerator = bool(
        not allow_cpu_fallback
        and preferred
        in {
            "cuda",
            "rocm",
        }
    )

    result = AcceleratorConfig(
        selection=(
            AcceleratorSelectionConfig(
                preferred_kind=preferred,
                device_index=(config.device.device_index),
                allow_cpu_fallback=(allow_cpu_fallback),
                allow_degraded_devices=True,
                require_accelerator=(require_accelerator),
                prefer_highest_free_memory=True,
            )
        ),
        attributes={
            "consumer": consumer,
        },
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Accelerator manager construction
# ------------------------------------------------------------------


def create_inference_accelerator_manager(
    config: AIConfig,
    *,
    consumer: str,
) -> AcceleratorManager:
    return AcceleratorManager(
        config=accelerator_config_from_ai(
            config,
            consumer=consumer,
        )
    )


# ------------------------------------------------------------------
# Inference accelerator selection
# ------------------------------------------------------------------


def select_inference_accelerator(
    manager: AcceleratorManager,
    config: AIConfig,
) -> AcceleratorSelection:
    """
    Select an inference device through the shared
    accelerator policy.

    Mixed precision is only requested when enabled
    by AIConfig.
    """

    required_precisions = ()

    if config.device.mixed_precision:
        required_precisions = (AcceleratorPrecision.FLOAT16.value,)

    try:
        return manager.select_for_inference(
            preferred_kind=(config.device.preferred),
            device_index=(config.device.device_index),
            require_accelerator=(
                not config.device.allow_cpu_fallback
                and config.device.preferred
                in {
                    "cuda",
                    "rocm",
                }
            ),
            allow_cpu_fallback=(config.device.allow_cpu_fallback),
            mixed_precision=(config.device.mixed_precision),
            required_precisions=(required_precisions),
            refresh=True,
        )

    except Exception as exc:
        error_name = type(exc).__name__

        if error_name in {
            "AcceleratorFallbackDisabledError",
            "AcceleratorFallbackUnavailableError",
            "AcceleratorDeviceUnavailableError",
            "AcceleratorDeviceNotFoundError",
            "AcceleratorNoMatchingDeviceError",
        }:
            raise AIDeviceUnavailableError(
                "No accelerator device satisfies the AI-serving requirements.",
                details={
                    "preferred": (config.device.preferred),
                    "device_index": (config.device.device_index),
                    "cpu_fallback": (config.device.allow_cpu_fallback),
                    "accelerator_error_type": (error_name),
                },
            ) from exc

        raise AIDeviceSelectionError(
            "AI-serving accelerator selection failed.",
            details={
                "preferred": (config.device.preferred),
                "accelerator_error_type": (error_name),
            },
        ) from exc


# ------------------------------------------------------------------
# Promoted checkpoint loading
# ------------------------------------------------------------------


def load_torch_payload(
    path: str | Path,
    *,
    map_location: Any,
) -> Dict[str, Any]:
    """
    Load one promoted model checkpoint payload.
    """

    require_torch()

    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise AIModelArtifactNotFoundError(
            "Promoted model artifact was not found.",
            details={
                "path": str(source),
            },
        )

    if not source.is_file():
        raise AIModelLoadError(
            "Promoted model artifact is not a regular file.",
            details={
                "path": str(source),
            },
        )

    try:
        try:
            value = torch.load(
                source,
                map_location=(map_location),
                weights_only=False,
            )

        except TypeError:
            value = torch.load(
                source,
                map_location=(map_location),
            )

    except Exception as exc:
        raise AIModelLoadError(
            "Promoted model artifact could not be loaded.",
            details=(
                safe_ai_error_details(
                    exc,
                    operation=("load_model_artifact"),
                    device=str(map_location),
                )
            ),
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise AIModelLoadError("Promoted model artifact root must be an object.")

    return dict(value)


# ------------------------------------------------------------------
# Payload helpers
# ------------------------------------------------------------------


def require_mapping(
    payload: Mapping[
        str,
        Any,
    ],
    key: str,
) -> Dict[str, Any]:
    value = payload.get(key)

    if not isinstance(
        value,
        Mapping,
    ):
        raise AIModelConfigurationError(
            "Model payload field is missing or invalid.",
            details={
                "field": key,
            },
        )

    return dict(value)


def require_model_state(
    payload: Mapping[
        str,
        Any,
    ],
) -> Mapping[str, Any]:
    value = payload.get("model_state_dict")

    if not isinstance(
        value,
        Mapping,
    ):
        raise AIModelConfigurationError(
            "Model payload contains no valid model_state_dict."
        )

    return value


# ------------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------------


def top_prediction(
    logits: Any,
    labels: Mapping[
        int,
        str,
    ],
) -> Optional[Prediction]:
    """
    Decode one classification-logits row.
    """

    require_torch()

    if logits.ndim != 2:
        raise AIInferenceShapeError(
            "Classification logits must have shape [batch, classes]."
        )

    if logits.shape[0] != 1:
        raise AIInferenceShapeError(
            "Single inference expected one classification-logits row."
        )

    if logits.shape[1] < 1:
        return None

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    probability, index = torch.max(
        probabilities,
        dim=1,
    )

    class_id = int(index.item())

    label = labels.get(class_id)

    if label is None:
        return None

    return Prediction(
        label=str(label),
        score=float(
            logits[
                0,
                class_id,
            ].item()
        ),
        probability=float(probability.item()),
        attributes={
            "class_id": class_id,
        },
    )


# ------------------------------------------------------------------
# Common runtime release helper
# ------------------------------------------------------------------


def release_accelerator_runtime(
    *,
    manager: AcceleratorManager,
    selection: AcceleratorSelection,
    model: Any,
) -> None:
    """
    Move a model to CPU before releasing its
    accelerator selection.

    Cache and synchronization behavior remain
    owned by the selected backend.
    """

    try:
        if model is not None and hasattr(
            model,
            "to",
        ):
            model.to("cpu")

    finally:
        manager.release(selection)


# ------------------------------------------------------------------
# Event model adapter
# ------------------------------------------------------------------


class InfrastructureEventModelAdapter(EventModelAdapter):
    """
    Runtime adapter for
    InfrastructureEventModel.
    """

    def _load_model(
        self,
    ) -> EventRuntime:
        require_torch()

        accelerator_manager = create_inference_accelerator_manager(
            self.config,
            consumer=("ai.models.event"),
        )

        accelerator_selection = None

        try:
            accelerator_selection = select_inference_accelerator(
                accelerator_manager,
                self.config,
            )

            device = accelerator_manager.runtime_device(accelerator_selection)

            payload = load_torch_payload(
                self.descriptor.path,
                map_location=device,
            )

            model_config = require_mapping(
                payload,
                "model_config",
            )

            vocabulary_value = payload.get("vocabulary")

            if not isinstance(
                vocabulary_value,
                Mapping,
            ):
                raise AIVocabularyError("Event model artifact has no valid vocabulary.")

            labels_value = payload.get("labels")

            if not isinstance(
                labels_value,
                Mapping,
            ):
                raise AILabelMapError("Event model artifact has no valid label maps.")

            try:
                vocabulary = Vocabulary.from_dict(vocabulary_value)

                labels = LabelMaps.from_dict(labels_value)

            except Exception as exc:
                raise AIModelConfigurationError(
                    "Event runtime assets could not be restored.",
                    details=(
                        safe_ai_error_details(
                            exc,
                            operation=("restore_event_assets"),
                            model_id=(self.model_id),
                        )
                    ),
                ) from exc

            required = (
                "vocab_size",
                "num_sources",
                "num_components",
                "num_events",
            )

            missing = [name for name in required if name not in model_config]

            if missing:
                raise AIModelConfigurationError(
                    "Event model configuration is incomplete.",
                    details={
                        "missing": missing,
                    },
                )

            model = InfrastructureEventModel(
                vocab_size=int(model_config["vocab_size"]),
                num_sources=int(model_config["num_sources"]),
                num_components=int(model_config["num_components"]),
                num_events=int(model_config["num_events"]),
                embedding_dim=int(
                    model_config.get(
                        "embedding_dim",
                        192,
                    )
                ),
                hidden_dim=int(
                    model_config.get(
                        "hidden_dim",
                        256,
                    )
                ),
                dropout=float(
                    model_config.get(
                        "dropout",
                        0.20,
                    )
                ),
            )

            model.load_state_dict(
                require_model_state(payload),
                strict=True,
            )

            model.to(device)

            model.eval()

            self.descriptor.device = accelerator_selection.device_id

            self.descriptor.model_config = {
                **model_config,
                "accelerator_selection": (accelerator_selection.to_dict()),
            }

            self.descriptor.loaded_at = utc_now()

            self.descriptor.attributes.update(
                {
                    "runtime_device": str(device),
                    "accelerator_kind": (accelerator_selection.kind),
                    "accelerator_fallback": (accelerator_selection.fallback_used),
                }
            )

            return EventRuntime(
                model=model,
                vocabulary=vocabulary,
                labels=labels,
                max_length=int(
                    model_config.get(
                        "max_length",
                        self.config.event.max_length,
                    )
                ),
                device=device,
                accelerator_selection=(accelerator_selection),
                accelerator_manager=(accelerator_manager),
            )

        except Exception:
            if accelerator_selection is not None:
                try:
                    accelerator_manager.release(accelerator_selection)

                except Exception:
                    pass

            raise

    def _after_load(
        self,
    ) -> None:
        if not isinstance(
            self.model,
            EventRuntime,
        ):
            raise AIModelLoadError("Event adapter runtime state is invalid.")

    def _release_model(
        self,
        runtime: Any,
    ) -> None:
        if not isinstance(
            runtime,
            EventRuntime,
        ):
            return

        release_accelerator_runtime(
            manager=(runtime.accelerator_manager),
            selection=(runtime.accelerator_selection),
            model=runtime.model,
        )

    @staticmethod
    def _event_text(
        request: InferenceRequest,
    ) -> str:
        payload = request.payload

        for key in (
            "content",
            "message",
            "text",
            "log",
        ):
            value = payload.get(key)

            if value is None:
                continue

            text = str(value).strip()

            if text:
                return text

        nested = payload.get("payload")

        if isinstance(
            nested,
            Mapping,
        ):
            for key in (
                "content",
                "message",
                "text",
                "log",
            ):
                value = nested.get(key)

                if value is None:
                    continue

                text = str(value).strip()

                if text:
                    return text

        raise AIEventTextMissingError(
            "Event inference request contains no log text.",
            details={
                "request_id": (request.request_id),
            },
        )

    def _infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        runtime = self.model

        if not isinstance(
            runtime,
            EventRuntime,
        ):
            raise AIInferenceExecutionError("Event runtime is not available.")

        text = self._event_text(request)

        requested_max_length = request.options.get("max_length")

        if requested_max_length is None:
            max_length = runtime.max_length

        else:
            try:
                max_length = int(requested_max_length)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise AIInferenceShapeError(
                    "Event max_length must be an integer."
                ) from exc

        if max_length < 1:
            raise AIInferenceShapeError("Event max_length must be at least one.")

        max_length = min(
            max_length,
            runtime.max_length,
        )

        started = time.perf_counter()

        try:
            (
                input_ids,
                attention_mask,
            ) = runtime.vocabulary.encode(
                text,
                max_length,
            )

            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=runtime.device,
            )

            mask_tensor = torch.tensor(
                [attention_mask],
                dtype=torch.float32,
                device=runtime.device,
            )

            use_mixed_precision = bool(
                self.config.device.mixed_precision
                and runtime.accelerator_selection.device.accelerator
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=(
                        "cuda"
                        if runtime.accelerator_selection.device.accelerator
                        else "cpu"
                    ),
                    dtype=(
                        torch.float16
                        if runtime.accelerator_selection.device.accelerator
                        else torch.bfloat16
                    ),
                    enabled=(use_mixed_precision),
                ):
                    output = runtime.model(
                        input_ids=(input_tensor),
                        attention_mask=(mask_tensor),
                    )

            source_prediction = (
                top_prediction(
                    output["source_logits"],
                    runtime.labels.id_to_source,
                )
                if self.config.event.return_source_prediction
                else None
            )

            component_prediction = (
                top_prediction(
                    output["component_logits"],
                    runtime.labels.id_to_component,
                )
                if self.config.event.return_component_prediction
                else None
            )

            event_prediction = (
                top_prediction(
                    output["event_logits"],
                    runtime.labels.id_to_event,
                )
                if self.config.event.return_event_prediction
                else None
            )

            representation = output["representation"]

            latency_ms = (time.perf_counter() - started) * 1000.0

            event_result = EventInference(
                source=source_prediction,
                component=(component_prediction),
                event=event_prediction,
                representation_dimension=(int(representation.shape[-1])),
                attributes={
                    "token_count": int(sum(attention_mask)),
                    "max_length": (max_length),
                },
            ).finalize()

            predictions = [
                prediction
                for prediction in (
                    source_prediction,
                    component_prediction,
                    event_prediction,
                )
                if prediction is not None
            ]

            attributes: Dict[
                str,
                Any,
            ] = {
                "device": str(runtime.device),
                "accelerator": (runtime.accelerator_selection.to_dict()),
                "mixed_precision": (use_mixed_precision),
            }

            if self.config.inference.include_representations:
                attributes["representation"] = (
                    representation[0].detach().float().cpu().tolist()
                )

            return InferenceResult(
                request_id=str(request.request_id),
                model=(self.descriptor.identity),
                state=(InferenceState.COMPLETED.value),
                event=event_result,
                predictions=(predictions),
                provenance=(request.provenance),
                latency_ms=(latency_ms),
                attributes=attributes,
            ).finalize()

        except (
            AIEventTextMissingError,
            AIInferenceShapeError,
            AIInferenceExecutionError,
        ):
            raise

        except Exception as exc:
            raise AIInferenceExecutionError(
                "Event inference failed.",
                details=(
                    safe_ai_error_details(
                        exc,
                        operation=("event_inference"),
                        model_id=(self.model_id),
                        family=self.family,
                        request_id=(request.request_id),
                        device=str(runtime.device),
                    )
                ),
            ) from exc


# ------------------------------------------------------------------
# Anomaly model adapter
# ------------------------------------------------------------------


class InfrastructureAnomalyModelAdapter(AnomalyModelAdapter):
    """
    Runtime adapter for
    InfrastructureAnomalyModel.
    """

    def _load_model(
        self,
    ) -> AnomalyRuntime:
        require_torch()

        accelerator_manager = create_inference_accelerator_manager(
            self.config,
            consumer=("ai.models.anomaly"),
        )

        accelerator_selection = None

        try:
            accelerator_selection = select_inference_accelerator(
                accelerator_manager,
                self.config,
            )

            device = accelerator_manager.runtime_device(accelerator_selection)

            payload = load_torch_payload(
                self.descriptor.path,
                map_location=device,
            )

            model_config = require_mapping(
                payload,
                "model_config",
            )

            try:
                model = InfrastructureAnomalyModel.from_config(model_config)

                model.load_state_dict(
                    require_model_state(payload),
                    strict=True,
                )

                model.to(device)

                model.eval()

            except Exception as exc:
                raise AIModelLoadError(
                    "Anomaly model could not be reconstructed.",
                    details=(
                        safe_ai_error_details(
                            exc,
                            operation=("construct_anomaly_model"),
                            model_id=(self.model_id),
                            device=str(device),
                        )
                    ),
                ) from exc

            threshold = self._artifact_threshold(
                payload,
                model_config,
            )

            feature_names = self._feature_names(payload)

            self.descriptor.device = accelerator_selection.device_id

            self.descriptor.model_config = {
                **model_config,
                "accelerator_selection": (accelerator_selection.to_dict()),
            }

            self.descriptor.loaded_at = utc_now()

            self.descriptor.attributes.update(
                {
                    "runtime_device": str(device),
                    "accelerator_kind": (accelerator_selection.kind),
                    "accelerator_fallback": (accelerator_selection.fallback_used),
                }
            )

            return AnomalyRuntime(
                model=model,
                input_dimension=(model.input_dim),
                threshold=threshold,
                device=device,
                accelerator_selection=(accelerator_selection),
                accelerator_manager=(accelerator_manager),
                feature_names=(feature_names),
            )

        except Exception:
            if accelerator_selection is not None:
                try:
                    accelerator_manager.release(accelerator_selection)

                except Exception:
                    pass

            raise

    def _after_load(
        self,
    ) -> None:
        if not isinstance(
            self.model,
            AnomalyRuntime,
        ):
            raise AIModelLoadError("Anomaly adapter runtime state is invalid.")

    def _release_model(
        self,
        runtime: Any,
    ) -> None:
        if not isinstance(
            runtime,
            AnomalyRuntime,
        ):
            return

        release_accelerator_runtime(
            manager=(runtime.accelerator_manager),
            selection=(runtime.accelerator_selection),
            model=runtime.model,
        )

    def _artifact_threshold(
        self,
        payload: Mapping[
            str,
            Any,
        ],
        model_config: Mapping[
            str,
            Any,
        ],
    ) -> float:
        candidates: List[Any] = []

        extra = payload.get("extra")

        if isinstance(
            extra,
            Mapping,
        ):
            candidates.extend(
                (
                    extra.get("calibrated_threshold"),
                    extra.get("threshold"),
                    extra.get("temporary_threshold"),
                )
            )

        candidates.extend(
            (
                model_config.get("calibrated_threshold"),
                model_config.get("threshold"),
                model_config.get("temporary_threshold"),
                self.descriptor.attributes.get("calibrated_threshold"),
                self.descriptor.attributes.get("threshold"),
                self.config.anomaly.default_threshold,
            )
        )

        for candidate in candidates:
            if candidate is None:
                continue

            try:
                threshold = float(candidate)

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                math.isfinite(threshold)
                and self.config.anomaly.minimum_threshold
                <= threshold
                <= self.config.anomaly.maximum_threshold
            ):
                return threshold

        raise AIAnomalyThresholdError("No valid anomaly threshold is available.")

    @staticmethod
    def _feature_names(
        payload: Mapping[
            str,
            Any,
        ],
    ) -> Tuple[str, ...]:
        schema = payload.get("feature_schema")

        if not isinstance(
            schema,
            Mapping,
        ):
            return ()

        names = schema.get("feature_names")

        if not isinstance(
            names,
            Sequence,
        ) or isinstance(
            names,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            return ()

        return tuple(str(name) for name in names)

    def _threshold(
        self,
        request: InferenceRequest,
        runtime: AnomalyRuntime,
    ) -> float:
        candidate = request.options.get("threshold")

        if candidate is None:
            return runtime.threshold

        if not (self.config.anomaly.allow_request_threshold_override):
            raise AIAnomalyThresholdError(
                "Request-level anomaly threshold override is disabled.",
                details={
                    "request_id": (request.request_id),
                },
            )

        try:
            threshold = float(candidate)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise AIAnomalyThresholdError(
                "Request anomaly threshold is invalid."
            ) from exc

        if (
            not math.isfinite(threshold)
            or threshold < self.config.anomaly.minimum_threshold
            or threshold > self.config.anomaly.maximum_threshold
        ):
            raise AIAnomalyThresholdError(
                "Request anomaly threshold is outside configured limits."
            )

        return threshold

    def _features(
        self,
        request: InferenceRequest,
        runtime: AnomalyRuntime,
    ) -> List[float]:
        payload = request.payload

        raw = None

        for key in (
            "features",
            "feature_vector",
            "values",
            "vector",
            "x",
        ):
            if key in payload:
                raw = payload[key]
                break

        if raw is None and runtime.feature_names:
            raw = [payload.get(name) for name in runtime.feature_names]

        if not isinstance(
            raw,
            Sequence,
        ) or isinstance(
            raw,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            raise AIAnomalyFeatureError(
                "Anomaly inference request contains no numeric feature vector.",
                details={
                    "request_id": (request.request_id),
                },
            )

        values: List[float] = []

        for index, item in enumerate(raw):
            if isinstance(
                item,
                bool,
            ) or not isinstance(
                item,
                (
                    int,
                    float,
                ),
            ):
                raise AIAnomalyFeatureError(
                    "Anomaly feature must be numeric.",
                    details={
                        "index": index,
                    },
                )

            value = float(item)

            if self.config.inference.reject_non_finite_inputs and not math.isfinite(
                value
            ):
                raise (
                    AIInferenceNonFiniteInputError(
                        "Anomaly feature must be finite.",
                        details={
                            "index": index,
                        },
                    )
                )

            values.append(value)

        if len(values) != runtime.input_dimension:
            raise AIInferenceShapeError(
                "Anomaly feature dimension does not match the model.",
                details={
                    "expected": (runtime.input_dimension),
                    "actual": len(values),
                },
            )

        return values

    def _infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        runtime = self.model

        if not isinstance(
            runtime,
            AnomalyRuntime,
        ):
            raise AIInferenceExecutionError("Anomaly runtime is not available.")

        values = self._features(
            request,
            runtime,
        )

        threshold = self._threshold(
            request,
            runtime,
        )

        started = time.perf_counter()

        try:
            feature_tensor = torch.tensor(
                [values],
                dtype=torch.float32,
                device=runtime.device,
            )

            use_mixed_precision = bool(
                self.config.device.mixed_precision
                and runtime.accelerator_selection.device.accelerator
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=(
                        "cuda"
                        if runtime.accelerator_selection.device.accelerator
                        else "cpu"
                    ),
                    dtype=(
                        torch.float16
                        if runtime.accelerator_selection.device.accelerator
                        else torch.bfloat16
                    ),
                    enabled=(use_mixed_precision),
                ):
                    output = runtime.model(feature_tensor)

            logits = output["anomaly_logits"]

            probabilities = output["anomaly_probability"]

            representation = output["representation"]

            logit = float(logits[0].item())

            probability = float(probabilities[0].item())

            if not (math.isfinite(logit) and math.isfinite(probability)):
                raise AIInferenceExecutionError(
                    "Anomaly model produced a non-finite output."
                )

            anomalous = bool(probability >= threshold)

            latency_ms = (time.perf_counter() - started) * 1000.0

            anomaly_result = AnomalyInference(
                probability=probability,
                threshold=threshold,
                anomalous=(anomalous),
                score=logit,
                representation_dimension=(int(representation.shape[-1])),
                attributes={
                    "input_dimension": (runtime.input_dimension),
                },
            ).finalize()

            prediction = Prediction(
                label=("anomaly" if anomalous else "normal"),
                score=logit,
                probability=(probability if anomalous else 1.0 - probability),
                attributes={
                    "anomaly_probability": (probability),
                    "threshold": (threshold),
                },
            )

            attributes: Dict[
                str,
                Any,
            ] = {
                "device": str(runtime.device),
                "accelerator": (runtime.accelerator_selection.to_dict()),
                "mixed_precision": (use_mixed_precision),
            }

            if self.config.inference.include_representations:
                attributes["representation"] = (
                    representation[0].detach().float().cpu().tolist()
                )

            return InferenceResult(
                request_id=str(request.request_id),
                model=(self.descriptor.identity),
                state=(InferenceState.COMPLETED.value),
                anomaly=(anomaly_result),
                predictions=[prediction],
                provenance=(request.provenance),
                latency_ms=(latency_ms),
                attributes=attributes,
            ).finalize()

        except (
            AIAnomalyFeatureError,
            AIAnomalyThresholdError,
            AIInferenceNonFiniteInputError,
            AIInferenceShapeError,
            AIInferenceExecutionError,
        ):
            raise

        except Exception as exc:
            raise AIInferenceExecutionError(
                "Anomaly inference failed.",
                details=(
                    safe_ai_error_details(
                        exc,
                        operation=("anomaly_inference"),
                        model_id=(self.model_id),
                        family=self.family,
                        request_id=(request.request_id),
                        device=str(runtime.device),
                    )
                ),
            ) from exc


class _OperationalModelAdapterMixin:
    """Load and execute one fixed-feature operational classifier."""

    model_kind: str

    def _construct_model(
        self,
        model_config: Mapping[str, Any],
    ) -> Any:
        raise NotImplementedError

    def _restore_labels(
        self,
        payload: Mapping[str, Any],
        model_config: Mapping[str, Any],
    ) -> Dict[str, Sequence[str]]:
        return {}

    def _load_model(self) -> OperationalRuntime:
        require_torch()

        accelerator_manager = create_inference_accelerator_manager(
            self.config,
            consumer=f"ai.models.{self.model_kind}",
        )
        accelerator_selection = None

        try:
            accelerator_selection = select_inference_accelerator(
                accelerator_manager,
                self.config,
            )
            device = accelerator_manager.runtime_device(
                accelerator_selection
            )
            payload = load_torch_payload(
                self.descriptor.path,
                map_location=device,
            )
            model_config = require_mapping(
                payload,
                "model_config",
            )
            model = self._construct_model(model_config)
            model.load_state_dict(
                require_model_state(payload),
                strict=True,
            )
            model.to(device)
            model.eval()

            self.descriptor.device = (
                accelerator_selection.device_id
            )
            self.descriptor.model_config = {
                **model_config,
                "accelerator_selection": (
                    accelerator_selection.to_dict()
                ),
            }
            self.descriptor.loaded_at = utc_now()
            self.descriptor.attributes.update(
                {
                    "runtime_device": str(device),
                    "accelerator_kind": (
                        accelerator_selection.kind
                    ),
                    "accelerator_fallback": (
                        accelerator_selection.fallback_used
                    ),
                }
            )

            return OperationalRuntime(
                model=model,
                input_dimension=FEATURE_DIMENSION,
                device=device,
                accelerator_selection=accelerator_selection,
                accelerator_manager=accelerator_manager,
                labels=self._restore_labels(
                    payload,
                    model_config,
                ),
            )

        except Exception:
            if accelerator_selection is not None:
                try:
                    accelerator_manager.release(
                        accelerator_selection
                    )
                except Exception:
                    pass
            raise

    def _after_load(self) -> None:
        if not isinstance(self.model, OperationalRuntime):
            raise AIModelLoadError(
                "Operational adapter runtime state is invalid."
            )

    def _release_model(self, runtime: Any) -> None:
        if isinstance(runtime, OperationalRuntime):
            release_accelerator_runtime(
                manager=runtime.accelerator_manager,
                selection=runtime.accelerator_selection,
                model=runtime.model,
            )

    @staticmethod
    def _input_features(
        request: InferenceRequest,
        expected_dimension: int,
    ) -> List[float]:
        payload = request.payload
        if "features" in payload:
            raw_features = payload["features"]
            if not isinstance(raw_features, Sequence) or isinstance(
                raw_features,
                (str, bytes, bytearray),
            ):
                raise AIAnomalyFeatureError(
                    "Operational model features must be a numeric sequence."
                )
            values = list(raw_features)
        else:
            values = feature_vector(
                payload.get("input", payload)
            )

        if len(values) != expected_dimension:
            raise AIInferenceShapeError(
                "Operational model input has an unexpected feature dimension.",
                details={
                    "expected": expected_dimension,
                    "actual": len(values),
                },
            )

        output = []
        for index, value in enumerate(values):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise AIAnomalyFeatureError(
                    "Operational model feature is not numeric.",
                    details={"index": index},
                ) from exc
            if not math.isfinite(number):
                raise AIInferenceNonFiniteInputError(
                    "Operational model features must be finite.",
                    details={"index": index},
                )
            output.append(number)
        return output

    @staticmethod
    def _ranked_predictions(
        logits: Any,
        labels: Sequence[str],
        *,
        head: str,
        top_k: int,
    ) -> List[Prediction]:
        probabilities = torch.softmax(
            logits,
            dim=-1,
        )
        count = min(top_k, len(labels))
        scores, indices = torch.topk(
            probabilities,
            k=count,
            dim=-1,
        )
        predictions = []
        for rank, (score, index) in enumerate(
            zip(
                scores[0].detach().cpu().tolist(),
                indices[0].detach().cpu().tolist(),
                strict=True,
            ),
            start=1,
        ):
            predictions.append(
                Prediction(
                    label=labels[index],
                    score=float(score),
                    probability=float(score),
                    attributes={
                        "head": head,
                        "rank": rank,
                        "calibrated": False,
                    },
                )
            )
        return predictions

    def _infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        runtime = self.model
        if not isinstance(runtime, OperationalRuntime):
            raise AIModelLoadError(
                "Operational model is not loaded."
            )

        started = time.perf_counter()
        features = self._input_features(
            request,
            runtime.input_dimension,
        )
        tensor = torch.tensor(
            [features],
            dtype=torch.float32,
            device=runtime.device,
        )
        try:
            with torch.no_grad():
                output = runtime.model(tensor)

            top_k = request.options.get("top_k", 3)
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise AIInferenceRequestError(
                    "top_k must be an integer."
                )
            if not 1 <= top_k <= 8:
                raise AIInferenceRequestError(
                    "top_k must be between 1 and 8."
                )

            if self.family == AIModelFamily.INCIDENT_ROOT_CAUSE.value:
                predictions = self._ranked_predictions(
                    output["root_cause_logits"],
                    runtime.labels["root_cause"],
                    head="root_cause",
                    top_k=top_k,
                )
                predictions.extend(
                    self._ranked_predictions(
                        output["severity_logits"],
                        runtime.labels["severity"],
                        head="severity",
                        top_k=1,
                    )
                )
            elif self.family == AIModelFamily.DECISION_POLICY_RISK.value:
                predictions = self._ranked_predictions(
                    output["risk_logits"],
                    runtime.labels["risk"],
                    head="risk",
                    top_k=top_k,
                )
            else:
                score = float(
                    output["recommendation_score"][0]
                    .detach()
                    .cpu()
                    .item()
                )
                predictions = [
                    Prediction(
                        label="recommendation_score",
                        score=score,
                        attributes={
                            "ranking_only": True,
                            "authorizes_action": False,
                            "calibrated": False,
                        },
                    )
                ]

            latency_ms = (
                time.perf_counter() - started
            ) * 1000.0
            return InferenceResult(
                request_id=str(request.request_id),
                model=self.descriptor.identity,
                state=InferenceState.COMPLETED.value,
                predictions=predictions,
                provenance=request.provenance,
                latency_ms=latency_ms,
                attributes={
                    "experimental": True,
                    "input_dimension": runtime.input_dimension,
                    "scores_calibrated": False,
                    "remediation_authorized": False,
                },
            ).finalize()
        except (
            AIInferenceRequestError,
            AIAnomalyFeatureError,
            AIInferenceShapeError,
            AIInferenceNonFiniteInputError,
        ):
            raise
        except Exception as exc:
            raise AIInferenceExecutionError(
                "Operational model inference failed.",
                details=safe_ai_error_details(
                    exc,
                    operation="operational_inference",
                    model_id=self.model_id,
                    family=self.family,
                    request_id=request.request_id,
                    device=str(runtime.device),
                ),
            ) from exc


class InfrastructureIncidentRootCauseModelAdapter(
    _OperationalModelAdapterMixin,
    IncidentRootCauseModelAdapter,
):
    model_kind = "incident_root_cause"

    def _construct_model(
        self,
        model_config: Mapping[str, Any],
    ) -> IncidentRootCauseModel:
        input_dim = int(model_config.get("input_dim", -1))
        if input_dim != FEATURE_DIMENSION:
            raise AIModelConfigurationError(
                "Incident model feature dimension is unsupported.",
                details={"input_dim": input_dim},
            )
        return IncidentRootCauseModel(
            input_dim=input_dim,
            severity_count=int(
                model_config.get("severity_count", -1)
            ),
            root_cause_count=int(
                model_config.get("root_cause_count", -1)
            ),
        )

    def _restore_labels(
        self,
        payload: Mapping[str, Any],
        model_config: Mapping[str, Any],
    ) -> Dict[str, Sequence[str]]:
        labels = {
            "severity": payload.get("severity_labels"),
            "root_cause": payload.get("root_cause_labels"),
        }
        expected = {
            "severity": int(model_config["severity_count"]),
            "root_cause": int(model_config["root_cause_count"]),
        }
        for name, values in labels.items():
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes, bytearray))
                or len(values) != expected[name]
                or not all(isinstance(value, str) and value for value in values)
            ):
                raise AIModelConfigurationError(
                    "Incident model label map is invalid.",
                    details={"label_map": name},
                )
        return labels


class InfrastructureDecisionRiskModelAdapter(
    _OperationalModelAdapterMixin,
    DecisionPolicyRiskModelAdapter,
):
    model_kind = "decision_policy_risk"

    def _construct_model(
        self,
        model_config: Mapping[str, Any],
    ) -> DecisionRiskModel:
        input_dim = int(model_config.get("input_dim", -1))
        if input_dim != FEATURE_DIMENSION:
            raise AIModelConfigurationError(
                "Decision-risk model feature dimension is unsupported.",
                details={"input_dim": input_dim},
            )
        return DecisionRiskModel(
            input_dim=input_dim,
            risk_count=int(model_config.get("risk_count", -1)),
        )

    def _restore_labels(
        self,
        payload: Mapping[str, Any],
        model_config: Mapping[str, Any],
    ) -> Dict[str, Sequence[str]]:
        labels = payload.get("risk_labels")
        if (
            not isinstance(labels, Sequence)
            or isinstance(labels, (str, bytes, bytearray))
            or len(labels) != int(model_config["risk_count"])
            or not all(isinstance(value, str) and value for value in labels)
        ):
            raise AIModelConfigurationError(
                "Decision-risk model label map is invalid."
            )
        return {"risk": labels}


class InfrastructureRemediationRankerAdapter(
    _OperationalModelAdapterMixin,
    RemediationRecommendationModelAdapter,
):
    model_kind = "remediation_recommendation"

    def _construct_model(
        self,
        model_config: Mapping[str, Any],
    ) -> RemediationRanker:
        input_dim = int(model_config.get("input_dim", -1))
        if input_dim != FEATURE_DIMENSION:
            raise AIModelConfigurationError(
                "Remediation-ranker feature dimension is unsupported.",
                details={"input_dim": input_dim},
            )
        return RemediationRanker(
            input_dim=input_dim,
        )


# ------------------------------------------------------------------
# Adapter factory
# ------------------------------------------------------------------


def create_model_adapter(
    descriptor: AIModelDescriptor,
    *,
    config: Optional[AIConfig] = None,
):
    """
    Create the family-specific model adapter.

    Model loading remains explicit or controlled
    by AIModelRegistry.
    """

    descriptor.finalize()

    family = descriptor.identity.family

    if family == AIModelFamily.EVENT.value:
        return InfrastructureEventModelAdapter(
            descriptor,
            config=config,
        )

    if family == AIModelFamily.ANOMALY.value:
        return InfrastructureAnomalyModelAdapter(
            descriptor,
            config=config,
        )

    if family == AIModelFamily.INCIDENT_ROOT_CAUSE.value:
        return InfrastructureIncidentRootCauseModelAdapter(
            descriptor,
            config=config,
        )

    if family == AIModelFamily.DECISION_POLICY_RISK.value:
        return InfrastructureDecisionRiskModelAdapter(
            descriptor,
            config=config,
        )

    if family == AIModelFamily.REMEDIATION_RECOMMENDATION.value:
        return InfrastructureRemediationRankerAdapter(
            descriptor,
            config=config,
        )

    raise AIModelConfigurationError(
        "No concrete AI adapter is registered for the requested model family.",
        details={
            "family": family,
            "model_id": (descriptor.identity.model_id),
        },
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "EventRuntime",
    "AnomalyRuntime",
    "OperationalRuntime",
    "require_torch",
    "accelerator_config_from_ai",
    "create_inference_accelerator_manager",
    "select_inference_accelerator",
    "load_torch_payload",
    "require_mapping",
    "require_model_state",
    "top_prediction",
    "release_accelerator_runtime",
    "InfrastructureEventModelAdapter",
    "InfrastructureAnomalyModelAdapter",
    "InfrastructureIncidentRootCauseModelAdapter",
    "InfrastructureDecisionRiskModelAdapter",
    "InfrastructureRemediationRankerAdapter",
    "create_model_adapter",
]
