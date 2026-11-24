from __future__ import annotations

"""
Shared contracts for the OSHA AI serving layer.

The AI subsystem consumes promoted models and
produces structured inference results.

Responsibilities represented here:

    model identity
    model lifecycle state
    inference requests
    predictions
    event-model results
    anomaly-model results
    inference provenance
    inference timing/status

Training contracts remain in src/training/.
Infrastructure interpretation remains in
src/intelligence/.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

import math

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.core.identifiers import (
    random_id,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    required_string,
    unique_strings,
)


# ---------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------

AI_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Model family
# ---------------------------------------------------------------


class AIModelFamily(
    str,
    Enum,
):
    EVENT = "event"

    ANOMALY = "anomaly"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Model state
# ---------------------------------------------------------------


class AIModelState(
    str,
    Enum,
):
    REGISTERED = "registered"

    LOADING = "loading"

    READY = "ready"

    DEGRADED = "degraded"

    UNAVAILABLE = "unavailable"

    FAILED = "failed"

    UNLOADED = "unloaded"


# ---------------------------------------------------------------
# Inference state
# ---------------------------------------------------------------


class InferenceState(
    str,
    Enum,
):
    CREATED = "created"

    ROUTED = "routed"

    RUNNING = "running"

    COMPLETED = "completed"

    REJECTED = "rejected"

    FAILED = "failed"


# ---------------------------------------------------------------
# Device family
# ---------------------------------------------------------------


class AIDeviceFamily(
    str,
    Enum,
):
    CPU = "cpu"

    CUDA = "cuda"

    ROCM = "rocm"

    AUTO = "auto"


# ---------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------


@dataclass(frozen=True)
class AIModelIdentity:
    """
    Stable identity for one promoted AI model
    version.
    """

    model_name: str

    family: str

    version: str

    artifact_id: Optional[str] = None

    checkpoint_id: Optional[str] = None

    def validate(
        self,
    ) -> None:
        required_string(
            self.model_name,
            name="model_name",
            maximum_length=512,
        )

        required_string(
            self.version,
            name="model version",
            maximum_length=128,
        )

        if self.family not in {item.value for item in AIModelFamily}:
            raise ValueError(f"Unknown AI model family: {self.family}")

    @property
    def model_id(
        self,
    ) -> str:
        self.validate()

        return self.family + ":" + self.model_name + ":" + self.version

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "model_id": self.model_id,
            "model_name": (self.model_name),
            "family": self.family,
            "version": self.version,
            "artifact_id": (self.artifact_id),
            "checkpoint_id": (self.checkpoint_id),
        }


# ---------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------


@dataclass
class AIModelDescriptor:
    """
    Runtime-neutral description of one model
    available to the AI serving layer.

    path points to a promoted model artifact.
    """

    identity: AIModelIdentity

    path: str

    state: str = AIModelState.REGISTERED.value

    device: str = AIDeviceFamily.AUTO.value

    checksum: Optional[str] = None

    capabilities: List[str] = field(default_factory=list)

    model_config: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    registered_at: str = field(default_factory=utc_now)

    loaded_at: Optional[str] = None

    error_type: Optional[str] = None

    def finalize(
        self,
    ) -> "AIModelDescriptor":
        self.identity.validate()

        required_string(
            self.path,
            name="AI model path",
            maximum_length=4096,
        )

        if self.state not in {item.value for item in AIModelState}:
            raise ValueError(f"Unknown AI model state: {self.state}")

        if self.device not in {item.value for item in AIDeviceFamily} and not (
            str(self.device).startswith("cuda:")
        ):
            raise ValueError(f"Unknown AI model device: {self.device}")

        self.capabilities = unique_strings(self.capabilities)

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "identity": (self.identity.to_dict()),
            "path": self.path,
            "state": self.state,
            "device": self.device,
            "checksum": (self.checksum),
            "capabilities": list(self.capabilities),
            "model_config": (to_json_safe(self.model_config)),
            "attributes": (to_json_safe(self.attributes)),
            "registered_at": (self.registered_at),
            "loaded_at": (self.loaded_at),
            "error_type": (self.error_type),
        }


# ---------------------------------------------------------------
# Inference provenance
# ---------------------------------------------------------------


@dataclass
class InferenceProvenance:
    """
    References connecting inference to the
    infrastructure evidence from which it was
    derived.
    """

    provider: Optional[str] = None

    environment_id: Optional[str] = None

    resource_type: Optional[str] = None

    resource_id: Optional[str] = None

    incident_id: Optional[str] = None

    correlation_id: Optional[str] = None

    source_ids: List[str] = field(default_factory=list)

    reference_ids: List[str] = field(default_factory=list)

    observed_at: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "InferenceProvenance":
        self.source_ids = unique_strings(self.source_ids)

        self.reference_ids = unique_strings(self.reference_ids)

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "provider": self.provider,
            "environment_id": (self.environment_id),
            "resource_type": (self.resource_type),
            "resource_id": (self.resource_id),
            "incident_id": (self.incident_id),
            "correlation_id": (self.correlation_id),
            "source_ids": list(self.source_ids),
            "reference_ids": list(self.reference_ids),
            "observed_at": (self.observed_at),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Inference request
# ---------------------------------------------------------------


@dataclass
class InferenceRequest:
    """
    Provider-neutral request to the AI serving
    layer.

    payload remains structured input rather than
    framework-specific tensors.
    """

    family: str

    payload: Dict[
        str,
        Any,
    ]

    model_name: Optional[str] = None

    model_version: Optional[str] = None

    provenance: InferenceProvenance = field(default_factory=(InferenceProvenance))

    options: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    request_id: Optional[str] = None

    created_at: str = field(default_factory=utc_now)

    def finalize(
        self,
    ) -> "InferenceRequest":
        if self.family not in {item.value for item in AIModelFamily}:
            raise ValueError(f"Unknown inference family: {self.family}")

        if not isinstance(
            self.payload,
            dict,
        ):
            raise ValueError("Inference payload must be a dictionary")

        if not self.request_id:
            self.request_id = random_id("ai:request")

        self.provenance.finalize()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (AI_SCHEMA_VERSION),
            "request_id": (self.request_id),
            "family": self.family,
            "model_name": (self.model_name),
            "model_version": (self.model_version),
            "payload": (to_json_safe(self.payload)),
            "provenance": (self.provenance.to_dict()),
            "options": (to_json_safe(self.options)),
            "created_at": (self.created_at),
        }


# ---------------------------------------------------------------
# Generic prediction
# ---------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """
    One model prediction with an optional score
    and probability.
    """

    label: str

    score: Optional[float] = None

    probability: Optional[float] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        required_string(
            self.label,
            name="prediction label",
            maximum_length=1024,
        )

        if self.score is not None:
            if not math.isfinite(float(self.score)):
                raise ValueError("Prediction score must be finite")

        if self.probability is not None:
            probability = float(self.probability)

            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("Prediction probability must be between 0 and 1")

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "label": self.label,
            "score": self.score,
            "probability": (self.probability),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Event inference result
# ---------------------------------------------------------------


@dataclass
class EventInference:
    """
    Structured result from the event model's
    classification heads.
    """

    source: Optional[Prediction] = None

    component: Optional[Prediction] = None

    event: Optional[Prediction] = None

    representation_dimension: Optional[int] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "EventInference":
        for prediction in (
            self.source,
            self.component,
            self.event,
        ):
            if prediction is not None:
                prediction.validate()

        if (
            self.representation_dimension is not None
            and self.representation_dimension < 1
        ):
            raise ValueError("representation_dimension must be >= 1")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "source": (self.source.to_dict() if self.source is not None else None),
            "component": (
                self.component.to_dict() if self.component is not None else None
            ),
            "event": (self.event.to_dict() if self.event is not None else None),
            "representation_dimension": (self.representation_dimension),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Anomaly inference result
# ---------------------------------------------------------------


@dataclass
class AnomalyInference:
    """
    Raw AI-layer anomaly result.

    Whether an infrastructure incident is
    actionable remains an intelligence/decision
    concern.
    """

    probability: float

    threshold: float

    anomalous: bool

    score: Optional[float] = None

    representation_dimension: Optional[int] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "AnomalyInference":
        probability = float(self.probability)

        threshold = float(self.threshold)

        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("Anomaly probability must be between 0 and 1")

        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("Anomaly threshold must be between 0 and 1")

        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("Anomaly score must be finite")

        if (
            self.representation_dimension is not None
            and self.representation_dimension < 1
        ):
            raise ValueError("representation_dimension must be >= 1")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "probability": float(self.probability),
            "threshold": float(self.threshold),
            "anomalous": bool(self.anomalous),
            "score": self.score,
            "representation_dimension": (self.representation_dimension),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------


@dataclass
class InferenceResult:
    request_id: str

    model: AIModelIdentity

    state: str

    event: Optional[EventInference] = None

    anomaly: Optional[AnomalyInference] = None

    predictions: List[Prediction] = field(default_factory=list)

    provenance: InferenceProvenance = field(default_factory=(InferenceProvenance))

    latency_ms: Optional[float] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    error_type: Optional[str] = None

    result_id: Optional[str] = None

    completed_at: str = field(default_factory=utc_now)

    def finalize(
        self,
    ) -> "InferenceResult":
        required_string(
            self.request_id,
            name="request_id",
            maximum_length=512,
        )

        self.model.validate()

        if self.state not in {item.value for item in InferenceState}:
            raise ValueError(f"Unknown inference state: {self.state}")

        if self.event is not None and self.anomaly is not None:
            raise ValueError(
                "Inference result cannot contain both event and anomaly result payloads"
            )

        if self.model.family == AIModelFamily.EVENT.value and self.anomaly is not None:
            raise ValueError("Event model returned an anomaly result")

        if self.model.family == AIModelFamily.ANOMALY.value and self.event is not None:
            raise ValueError("Anomaly model returned an event result")

        if self.event is not None:
            self.event.finalize()

        if self.anomaly is not None:
            self.anomaly.finalize()

        for prediction in self.predictions:
            prediction.validate()

        self.provenance.finalize()

        if self.latency_ms is not None:
            latency = float(self.latency_ms)

            if not math.isfinite(latency) or latency < 0.0:
                raise ValueError("latency_ms must be finite and >= 0")

        if not self.result_id:
            self.result_id = random_id("ai:result")

        return self

    @property
    def successful(
        self,
    ) -> bool:
        return self.state == InferenceState.COMPLETED.value

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (AI_SCHEMA_VERSION),
            "result_id": (self.result_id),
            "request_id": (self.request_id),
            "model": (self.model.to_dict()),
            "state": self.state,
            "successful": (self.successful),
            "event": (self.event.to_dict() if self.event is not None else None),
            "anomaly": (self.anomaly.to_dict() if self.anomaly is not None else None),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "provenance": (self.provenance.to_dict()),
            "latency_ms": (self.latency_ms),
            "attributes": (to_json_safe(self.attributes)),
            "error_type": (self.error_type),
            "completed_at": (self.completed_at),
        }


# ---------------------------------------------------------------
# Batch inference request
# ---------------------------------------------------------------


@dataclass
class BatchInferenceRequest:
    requests: List[InferenceRequest]

    batch_id: Optional[str] = None

    created_at: str = field(default_factory=utc_now)

    def finalize(
        self,
    ) -> "BatchInferenceRequest":
        if not self.requests:
            raise ValueError("Inference batch cannot be empty")

        for request in self.requests:
            request.finalize()

        if not self.batch_id:
            self.batch_id = random_id("ai:batch")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (AI_SCHEMA_VERSION),
            "batch_id": (self.batch_id),
            "request_count": (len(self.requests)),
            "requests": [request.to_dict() for request in self.requests],
            "created_at": (self.created_at),
        }


# ---------------------------------------------------------------
# Batch inference result
# ---------------------------------------------------------------


@dataclass
class BatchInferenceResult:
    batch_id: str

    results: List[InferenceResult]

    completed_at: str = field(default_factory=utc_now)

    def finalize(
        self,
    ) -> "BatchInferenceResult":
        required_string(
            self.batch_id,
            name="batch_id",
            maximum_length=512,
        )

        for result in self.results:
            result.finalize()

        return self

    @property
    def success_count(
        self,
    ) -> int:
        return sum(1 for result in self.results if result.successful)

    @property
    def failure_count(
        self,
    ) -> int:
        return len(self.results) - self.success_count

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (AI_SCHEMA_VERSION),
            "batch_id": (self.batch_id),
            "result_count": (len(self.results)),
            "success_count": (self.success_count),
            "failure_count": (self.failure_count),
            "results": [result.to_dict() for result in self.results],
            "completed_at": (self.completed_at),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AI_SCHEMA_VERSION",
    "AIModelFamily",
    "AIModelState",
    "InferenceState",
    "AIDeviceFamily",
    "AIModelIdentity",
    "AIModelDescriptor",
    "InferenceProvenance",
    "InferenceRequest",
    "Prediction",
    "EventInference",
    "AnomalyInference",
    "InferenceResult",
    "BatchInferenceRequest",
    "BatchInferenceResult",
]
