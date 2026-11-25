from __future__ import annotations

"""
Public API contracts for OSHA.

These models define the external HTTP interface
used by:

    health routes
    accelerator routes
    model-management routes
    AI inference routes
    incident routes
    decision routes
    remediation routes

Important boundary:

    API request/response models
        are public, versioned contracts.

    Internal src.ai, src.accelerators,
    src.intelligence, src.decisions, and
    src.remediation models
        remain private domain contracts.

Route services are responsible for translating
between public API models and internal objects.

This module does not:

    create a FastAPI application
    register routes
    perform inference
    select accelerator devices
    load models
    mutate infrastructure
    execute remediation
"""

import math

from datetime import datetime, timezone

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from uuid import (
    UUID,
    uuid4,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ------------------------------------------------------------------
# API schema version
# ------------------------------------------------------------------

API_SCHEMA_VERSION = "1"


# ------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------

MAX_REQUEST_ID_LENGTH = 128

MAX_IDEMPOTENCY_KEY_LENGTH = 256

MAX_NAME_LENGTH = 256

MAX_MESSAGE_LENGTH = 4096

MAX_CONTENT_LENGTH = 131_072

MAX_BATCH_SIZE = 256

MAX_FEATURE_COUNT = 100_000

MAX_ATTRIBUTE_COUNT = 256

MAX_INCIDENT_SIGNAL_COUNT = 10_000

MAX_REMEDIATION_STEP_COUNT = 256


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_required_text(
    value: Any,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    if len(text) > maximum_length:
        raise ValueError(f"{field_name} cannot exceed {maximum_length} characters")

    return text


def clean_optional_text(
    value: Any,
    *,
    field_name: str,
    maximum_length: int,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if len(text) > maximum_length:
        raise ValueError(f"{field_name} cannot exceed {maximum_length} characters")

    return text


def finite_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(f"{field_name} must be numeric")

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")

    return number


def validate_attributes(
    value: Mapping[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    result = dict(value)

    if len(result) > MAX_ATTRIBUTE_COUNT:
        raise ValueError(
            f"attributes cannot contain more than {MAX_ATTRIBUTE_COUNT} entries"
        )

    for key in result:
        if not isinstance(
            key,
            str,
        ):
            raise ValueError("attribute keys must be strings")

        if not key.strip():
            raise ValueError("attribute keys cannot be empty")

        if len(key) > MAX_NAME_LENGTH:
            raise ValueError("attribute key exceeds the maximum allowed length")

    return result


# ------------------------------------------------------------------
# Common enums
# ------------------------------------------------------------------


class APIStatus(
    str,
    Enum,
):
    OK = "ok"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    ERROR = "error"


class HealthStatus(
    str,
    Enum,
):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class OperationState(
    str,
    Enum,
):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AcceleratorKindAPI(
    str,
    Enum,
):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


class PrecisionAPI(
    str,
    Enum,
):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"


class WorkloadKindAPI(
    str,
    Enum,
):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    EMBEDDING = "embedding"
    DATA_PROCESSING = "data_processing"
    CUSTOM = "custom"


class OptimizationLevelAPI(
    str,
    Enum,
):
    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class ModelFamilyAPI(
    str,
    Enum,
):
    EVENT = "event"
    ANOMALY = "anomaly"


class OperationalModelFamilyAPI(
    str,
    Enum,
):
    INCIDENT_ROOT_CAUSE = "incident_root_cause"
    DECISION_POLICY_RISK = "decision_policy_risk"
    REMEDIATION_RECOMMENDATION = "remediation_recommendation"


class ModelStateAPI(
    str,
    Enum,
):
    DISCOVERED = "discovered"
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNLOADED = "unloaded"


class InferenceStateAPI(
    str,
    Enum,
):
    COMPLETED = "completed"
    FAILED = "failed"


class IncidentStatusAPI(
    str,
    Enum,
):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SeverityAPI(
    str,
    Enum,
):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStatusAPI(
    str,
    Enum,
):
    PROPOSED = "proposed"
    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    EXPIRED = "expired"


class RemediationPlanStatusAPI(
    str,
    Enum,
):
    DRAFT = "draft"
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class RemediationRiskAPI(
    str,
    Enum,
):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ------------------------------------------------------------------
# Shared base models
# ------------------------------------------------------------------


class APIInputModel(BaseModel):
    """
    Strict public request model.

    Unknown fields are rejected so misspelled or
    unsupported caller input cannot be silently
    ignored.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class APIOutputModel(BaseModel):
    """
    Public response contract.

    from_attributes allows services to construct
    responses from domain objects while preserving
    response filtering.
    """

    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )


class APIResponseMetadata(APIOutputModel):
    schema_version: str = Field(
        default=API_SCHEMA_VERSION,
    )

    request_id: Optional[str] = Field(
        default=None,
        max_length=MAX_REQUEST_ID_LENGTH,
    )

    generated_at: datetime = Field(
        default_factory=utc_now,
    )

    duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
    )


# ------------------------------------------------------------------
# Error contracts
# ------------------------------------------------------------------


class APIErrorField(APIOutputModel):
    location: List[str | int] = Field(
        default_factory=list,
    )

    code: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )


class APIError(APIOutputModel):
    code: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )

    retryable: bool = False

    details: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    fields: List[APIErrorField] = Field(
        default_factory=list,
    )


class APIErrorResponse(APIOutputModel):
    status: str = Field(
        default=APIStatus.ERROR.value,
    )

    error: APIError

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Generic operation contracts
# ------------------------------------------------------------------


class OperationResponse(APIOutputModel):
    operation_id: UUID = Field(
        default_factory=uuid4,
    )

    state: str = Field(
        default=OperationState.COMPLETED.value,
    )

    message: Optional[str] = Field(
        default=None,
        max_length=MAX_MESSAGE_LENGTH,
    )

    result: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


class PaginationMetadata(APIOutputModel):
    limit: int = Field(
        ge=1,
        le=1000,
    )

    offset: int = Field(
        ge=0,
    )

    returned: int = Field(
        ge=0,
    )

    total: Optional[int] = Field(
        default=None,
        ge=0,
    )

    has_more: Optional[bool] = None


# ------------------------------------------------------------------
# Health contracts
# ------------------------------------------------------------------


class HealthComponentResponse(APIOutputModel):
    component: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    status: str

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )

    critical: bool = False

    error_type: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class HealthResponse(APIOutputModel):
    service: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    status: str

    ready: bool

    live: bool

    version: Optional[str] = Field(
        default=None,
        max_length=128,
    )

    components: List[HealthComponentResponse] = Field(
        default_factory=list,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


class LivenessResponse(APIOutputModel):
    status: str = Field(
        default=HealthStatus.HEALTHY.value,
    )

    live: bool = True

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


class ReadinessResponse(APIOutputModel):
    status: str

    ready: bool

    reasons: List[str] = Field(
        default_factory=list,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Inference provenance
# ------------------------------------------------------------------


class InferenceProvenanceRequest(APIInputModel):
    source: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    source_id: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    tenant_id: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    environment: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    cluster: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    host: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    correlation_id: Optional[str] = Field(
        default=None,
        max_length=MAX_REQUEST_ID_LENGTH,
    )

    observed_at: Optional[datetime] = None

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    @field_validator(
        "attributes",
    )
    @classmethod
    def validate_metadata(
        cls,
        value: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        return validate_attributes(value)


class InferenceProvenanceResponse(APIOutputModel):
    source: Optional[str] = None

    source_id: Optional[str] = None

    tenant_id: Optional[str] = None

    environment: Optional[str] = None

    cluster: Optional[str] = None

    host: Optional[str] = None

    correlation_id: Optional[str] = None

    observed_at: Optional[datetime] = None

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ------------------------------------------------------------------
# Event inference
# ------------------------------------------------------------------


class EventInferenceRequest(APIInputModel):
    content: str = Field(
        min_length=1,
        max_length=MAX_CONTENT_LENGTH,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    max_length: Optional[int] = Field(
        default=None,
        ge=1,
        le=65_536,
    )

    provenance: Optional[InferenceProvenanceRequest] = None

    payload: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    options: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    @field_validator(
        "content",
    )
    @classmethod
    def normalize_content(
        cls,
        value: str,
    ) -> str:
        return clean_required_text(
            value,
            field_name="content",
            maximum_length=(MAX_CONTENT_LENGTH),
        )

    @field_validator(
        "payload",
        "options",
    )
    @classmethod
    def validate_mappings(
        cls,
        value: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        return validate_attributes(value)


# ------------------------------------------------------------------
# Anomaly inference
# ------------------------------------------------------------------


class AnomalyInferenceRequest(APIInputModel):
    features: List[float] = Field(
        min_length=1,
        max_length=MAX_FEATURE_COUNT,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    provenance: Optional[InferenceProvenanceRequest] = None

    payload: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    options: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    @field_validator(
        "features",
    )
    @classmethod
    def validate_features(
        cls,
        values: List[float],
    ) -> List[float]:
        output = []

        for index, value in enumerate(values):
            output.append(
                finite_number(
                    value,
                    field_name=(f"features[{index}]"),
                )
            )

        return output

    @field_validator(
        "payload",
        "options",
    )
    @classmethod
    def validate_mappings(
        cls,
        value: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        return validate_attributes(value)


# ------------------------------------------------------------------
# Operational model inference
# ------------------------------------------------------------------


class OperationalInferenceRequest(APIInputModel):
    family: OperationalModelFamilyAPI

    input: Dict[
        str,
        Any,
    ] = Field(
        min_length=1,
        max_length=MAX_ATTRIBUTE_COUNT,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=8,
    )

    provenance: Optional[InferenceProvenanceRequest] = None

    @field_validator(
        "input",
    )
    @classmethod
    def validate_input(
        cls,
        value: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        return validate_attributes(value)


# ------------------------------------------------------------------
# Capacity forecasting
# ------------------------------------------------------------------


class CapacityMetricAPI(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


class CapacityForecastStatusAPI(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class CapacityTelemetryPointRequest(APIInputModel):
    timestamp: datetime
    value: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value


class CapacityForecastRequest(APIInputModel):
    resource_id: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    metric: CapacityMetricAPI
    history: List[CapacityTelemetryPointRequest] = Field(
        default_factory=list,
        max_length=10_000,
    )
    horizon_hours: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=24.0 * 30,
        allow_inf_nan=False,
    )
    step_minutes: Optional[int] = Field(default=None, ge=1, le=24 * 60)


class CapacityForecastPointResponse(APIOutputModel):
    timestamp: datetime
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class CapacityForecastResponse(APIOutputModel):
    resource_id: str
    metric: CapacityMetricAPI
    generated_at: datetime
    horizon_hours: float
    current_value: float
    threshold: float
    status: CapacityForecastStatusAPI
    points: List[CapacityForecastPointResponse] = Field(default_factory=list)
    predicted_threshold_time: Optional[datetime] = None
    time_to_threshold_hours: Optional[float] = None
    confidence: float
    model_name: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    metadata: APIResponseMetadata = Field(default_factory=APIResponseMetadata)


# ------------------------------------------------------------------
# Batch inference
# ------------------------------------------------------------------


class BatchInferenceItemRequest(APIInputModel):
    item_id: str = Field(
        min_length=1,
        max_length=MAX_REQUEST_ID_LENGTH,
    )

    family: str

    event: Optional[EventInferenceRequest] = None

    anomaly: Optional[AnomalyInferenceRequest] = None

    @model_validator(
        mode="after",
    )
    def validate_family_payload(
        self,
    ) -> "BatchInferenceItemRequest":
        if self.family == ModelFamilyAPI.EVENT.value:
            if self.event is None:
                raise ValueError("event payload is required for family=event")

            if self.anomaly is not None:
                raise ValueError("anomaly payload is not allowed for family=event")

        elif self.family == ModelFamilyAPI.ANOMALY.value:
            if self.anomaly is None:
                raise ValueError("anomaly payload is required for family=anomaly")

            if self.event is not None:
                raise ValueError("event payload is not allowed for family=anomaly")

        else:
            raise ValueError("family must be event or anomaly")

        return self


class BatchInferenceRequest(APIInputModel):
    items: List[BatchInferenceItemRequest] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )

    fail_fast: bool = False

    require_same_family: bool = False

    @field_validator(
        "items",
    )
    @classmethod
    def validate_unique_item_ids(
        cls,
        values: List[BatchInferenceItemRequest],
    ) -> List[BatchInferenceItemRequest]:
        seen = set()

        for item in values:
            if item.item_id in seen:
                raise ValueError("batch item IDs must be unique")

            seen.add(item.item_id)

        return values


# ------------------------------------------------------------------
# Inference responses
# ------------------------------------------------------------------


class ModelIdentityResponse(APIOutputModel):
    model_id: str

    name: str

    family: str

    version: str


class PredictionResponse(APIOutputModel):
    label: str

    score: Optional[float] = None

    probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class EventInferenceDetailResponse(APIOutputModel):
    source: Optional[PredictionResponse] = None

    component: Optional[PredictionResponse] = None

    event: Optional[PredictionResponse] = None

    representation_dimension: Optional[int] = Field(
        default=None,
        ge=0,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class AnomalyInferenceDetailResponse(APIOutputModel):
    probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    anomalous: bool

    score: Optional[float] = None

    representation_dimension: Optional[int] = Field(
        default=None,
        ge=0,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class InferenceResponse(APIOutputModel):
    request_id: str

    state: str = Field(
        default=InferenceStateAPI.COMPLETED.value,
    )

    model: ModelIdentityResponse

    predictions: List[PredictionResponse] = Field(
        default_factory=list,
    )

    event: Optional[EventInferenceDetailResponse] = None

    anomaly: Optional[AnomalyInferenceDetailResponse] = None

    provenance: Optional[InferenceProvenanceResponse] = None

    latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


class BatchInferenceItemResponse(APIOutputModel):
    item_id: str

    success: bool

    result: Optional[InferenceResponse] = None

    error: Optional[APIError] = None

    @model_validator(
        mode="after",
    )
    def validate_result_or_error(
        self,
    ) -> "BatchInferenceItemResponse":
        if self.success:
            if self.result is None:
                raise ValueError("successful item requires result")

            if self.error is not None:
                raise ValueError("successful item cannot contain an error")

        else:
            if self.error is None:
                raise ValueError("failed item requires error")

        return self


class BatchInferenceResponse(APIOutputModel):
    items: List[BatchInferenceItemResponse]

    total: int = Field(
        ge=0,
    )

    succeeded: int = Field(
        ge=0,
    )

    failed: int = Field(
        ge=0,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )

    @model_validator(
        mode="after",
    )
    def validate_counts(
        self,
    ) -> "BatchInferenceResponse":
        if self.succeeded + self.failed != self.total:
            raise ValueError("succeeded + failed must equal total")

        if len(self.items) != self.total:
            raise ValueError("items length must equal total")

        return self


# ------------------------------------------------------------------
# Accelerator requests
# ------------------------------------------------------------------


class AcceleratorSelectionRequest(APIInputModel):
    preferred_kind: str = Field(
        default=AcceleratorKindAPI.AUTO.value,
    )

    device_id: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    device_index: Optional[int] = Field(
        default=None,
        ge=0,
    )

    training: bool = False

    inference: bool = False

    require_accelerator: bool = False

    allow_cpu_fallback: bool = True

    allow_degraded: bool = True

    minimum_total_memory_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    minimum_free_memory_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    required_capabilities: List[str] = Field(
        default_factory=list,
        max_length=128,
    )

    required_precisions: List[str] = Field(
        default_factory=list,
        max_length=16,
    )

    mixed_precision: bool = False

    @model_validator(
        mode="after",
    )
    def validate_policy(
        self,
    ) -> "AcceleratorSelectionRequest":
        if self.require_accelerator and self.allow_cpu_fallback:
            raise ValueError(
                "require_accelerator and allow_cpu_fallback cannot both be enabled"
            )

        if (
            self.require_accelerator
            and self.preferred_kind == AcceleratorKindAPI.CPU.value
        ):
            raise ValueError("CPU cannot satisfy require_accelerator")

        return self


# ------------------------------------------------------------------
# Accelerator responses
# ------------------------------------------------------------------


class AcceleratorMemoryResponse(APIOutputModel):
    total_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    free_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    used_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    allocated_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    reserved_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    utilization_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class AcceleratorDeviceResponse(APIOutputModel):
    device_id: str

    kind: str

    runtime_device: str

    index: Optional[int] = None

    name: Optional[str] = None

    state: str

    available: bool

    degraded: bool = False

    accelerator: bool = False

    memory: AcceleratorMemoryResponse = Field(
        default_factory=AcceleratorMemoryResponse,
    )

    capabilities: List[str] = Field(
        default_factory=list,
    )

    supported_precisions: List[str] = Field(
        default_factory=list,
    )

    architecture: Optional[str] = None

    compute_capability: Optional[str] = None

    vendor: Optional[str] = None

    runtime_version: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class AcceleratorInventoryResponse(APIOutputModel):
    devices: List[AcceleratorDeviceResponse] = Field(
        default_factory=list,
    )

    device_count: int = Field(
        ge=0,
    )

    available_device_count: int = Field(
        ge=0,
    )

    accelerator_device_count: int = Field(
        ge=0,
    )

    detected_at: Optional[datetime] = None

    failures: List[
        Dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


class AcceleratorSelectionResponse(APIOutputModel):
    device: AcceleratorDeviceResponse

    reason: str

    fallback_used: bool

    candidates_considered: List[str] = Field(
        default_factory=list,
    )

    rejected_candidates: Dict[
        str,
        List[str],
    ] = Field(
        default_factory=dict,
    )

    selected_at: Optional[datetime] = None

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Optimization requests
# ------------------------------------------------------------------


class AcceleratorWorkloadRequest(APIInputModel):
    kind: str

    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
    )

    model_parameter_count: Optional[int] = Field(
        default=None,
        ge=0,
    )

    estimated_model_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    estimated_input_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    estimated_peak_memory_bytes: Optional[int] = Field(
        default=None,
        ge=0,
    )

    sequence_length: Optional[int] = Field(
        default=None,
        ge=1,
    )

    precision: str = Field(
        default=PrecisionAPI.FLOAT32.value,
    )

    mixed_precision_enabled: bool = False

    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
    )

    data_loader_workers: Optional[int] = Field(
        default=None,
        ge=0,
    )

    pin_memory_enabled: Optional[bool] = None

    non_blocking_transfers: Optional[bool] = None

    compiled: bool = False

    latency_sensitive: bool = False

    throughput_sensitive: bool = False

    deterministic: bool = False

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class OptimizationConstraintsRequest(APIInputModel):
    level: str = Field(
        default=OptimizationLevelAPI.BALANCED.value,
    )

    allow_precision_change: bool = True

    allow_mixed_precision: bool = True

    allow_batch_size_change: bool = True

    allow_compilation: bool = True

    allow_thread_change: bool = True

    allow_data_loader_change: bool = True

    allow_memory_tuning: bool = True

    minimum_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
    )

    maximum_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
    )

    maximum_memory_fraction: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )

    required_precision: Optional[str] = None

    @model_validator(
        mode="after",
    )
    def validate_batch_range(
        self,
    ) -> "OptimizationConstraintsRequest":
        if (
            self.minimum_batch_size is not None
            and self.maximum_batch_size is not None
            and self.minimum_batch_size > self.maximum_batch_size
        ):
            raise ValueError("minimum_batch_size cannot exceed maximum_batch_size")

        return self


class AcceleratorOptimizationRequest(APIInputModel):
    selection: AcceleratorSelectionRequest

    workload: AcceleratorWorkloadRequest

    constraints: OptimizationConstraintsRequest = Field(
        default_factory=OptimizationConstraintsRequest,
    )

    refresh_metrics: bool = True


# ------------------------------------------------------------------
# Optimization responses
# ------------------------------------------------------------------


class OptimizationRecommendationResponse(APIOutputModel):
    kind: str

    title: str

    reason: str

    current_value: Any = None

    recommended_value: Any = None

    priority: str

    confidence: str

    disposition: str

    expected_effect: Optional[str] = None

    risks: List[str] = Field(
        default_factory=list,
    )

    prerequisites: List[str] = Field(
        default_factory=list,
    )

    evidence: List[str] = Field(
        default_factory=list,
    )

    automatically_applicable: bool = False

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class AcceleratorOptimizationResponse(APIOutputModel):
    device_id: str

    backend: str

    optimizer_name: Optional[str] = None

    optimizer_version: Optional[str] = None

    recommendation_count: int = Field(
        ge=0,
    )

    high_priority_count: int = Field(
        ge=0,
    )

    recommendations: List[OptimizationRecommendationResponse] = Field(
        default_factory=list,
    )

    profile: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    automatically_applied: bool = False

    generated_at: Optional[datetime] = None

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Accelerator metrics response
# ------------------------------------------------------------------


class AcceleratorMetricsResponse(APIOutputModel):
    device_id: str

    kind: str

    runtime_device: str

    metrics: Dict[
        str,
        Any,
    ]

    sampled_at: Optional[datetime] = None

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Model contracts
# ------------------------------------------------------------------


class ModelResponse(APIOutputModel):
    model_id: str

    name: str

    family: str

    version: str

    state: str

    ready: bool

    loaded: bool

    device: Optional[str] = None

    artifact_path_exposed: bool = False

    model_configuration: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
        alias="model_config",
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    registered_at: Optional[datetime] = None

    loaded_at: Optional[datetime] = None


class ModelListResponse(APIOutputModel):
    models: List[ModelResponse] = Field(
        default_factory=list,
    )

    pagination: PaginationMetadata

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


class ModelOperationRequest(APIInputModel):
    force: bool = False

    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=MAX_IDEMPOTENCY_KEY_LENGTH,
    )


class ModelRefreshRequest(APIInputModel):
    fail_fast: bool = False

    replace: bool = False

    eager_load: Optional[bool] = None


class ModelRefreshResponse(APIOutputModel):
    discovered: int = Field(
        ge=0,
    )

    registered: int = Field(
        ge=0,
    )

    failed: int = Field(
        ge=0,
    )

    failures: List[
        Dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Incident contracts
# ------------------------------------------------------------------


class IncidentSignalRequest(APIInputModel):
    signal_id: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    source: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    signal_type: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    observed_at: datetime

    severity: str = Field(
        default=SeverityAPI.INFO.value,
    )

    payload: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class IncidentCreateRequest(APIInputModel):
    title: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    summary: Optional[str] = Field(
        default=None,
        max_length=MAX_MESSAGE_LENGTH,
    )

    severity: str = Field(
        default=SeverityAPI.MEDIUM.value,
    )

    service: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    environment: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    signals: List[IncidentSignalRequest] = Field(
        default_factory=list,
        max_length=MAX_INCIDENT_SIGNAL_COUNT,
    )

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=MAX_IDEMPOTENCY_KEY_LENGTH,
    )


class IncidentResponse(APIOutputModel):
    incident_id: str

    title: str

    summary: Optional[str] = None

    severity: str

    status: str

    service: Optional[str] = None

    environment: Optional[str] = None

    signal_count: int = Field(
        ge=0,
    )

    created_at: datetime

    updated_at: datetime

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class IncidentListResponse(APIOutputModel):
    incidents: List[IncidentResponse] = Field(
        default_factory=list,
    )

    pagination: PaginationMetadata

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


# ------------------------------------------------------------------
# Decision contracts
# ------------------------------------------------------------------


class DecisionEvaluateRequest(APIInputModel):
    incident_id: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    candidate_action: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    target: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    context: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    dry_run: bool = True


class DecisionResponse(APIOutputModel):
    decision_id: str

    incident_id: str

    status: str

    candidate_action: str

    reason: str

    risk: str

    approval_required: bool

    policy_results: List[
        Dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

    created_at: datetime

    expires_at: Optional[datetime] = None

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ------------------------------------------------------------------
# Remediation contracts
# ------------------------------------------------------------------


class RemediationStepRequest(APIInputModel):
    step_id: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    action: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    provider: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    target: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    parameters: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=86_400,
    )

    reversible: bool = False

    rollback_action: Optional[str] = Field(
        default=None,
        max_length=MAX_NAME_LENGTH,
    )

    @model_validator(
        mode="after",
    )
    def validate_rollback(
        self,
    ) -> "RemediationStepRequest":
        if self.reversible and not self.rollback_action:
            raise ValueError("reversible remediation step requires rollback_action")

        return self


class RemediationPlanCreateRequest(APIInputModel):
    incident_id: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    decision_id: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    title: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=MAX_MESSAGE_LENGTH,
    )

    risk: str = Field(
        default=RemediationRiskAPI.MEDIUM.value,
    )

    steps: List[RemediationStepRequest] = Field(
        min_length=1,
        max_length=MAX_REMEDIATION_STEP_COUNT,
    )

    dry_run: bool = True

    idempotency_key: str = Field(
        min_length=1,
        max_length=MAX_IDEMPOTENCY_KEY_LENGTH,
    )

    @field_validator(
        "steps",
    )
    @classmethod
    def validate_unique_steps(
        cls,
        values: List[RemediationStepRequest],
    ) -> List[RemediationStepRequest]:
        seen = set()

        for step in values:
            if step.step_id in seen:
                raise ValueError("remediation step IDs must be unique")

            seen.add(step.step_id)

        return values


class RemediationApprovalRequest(APIInputModel):
    approved: bool

    reason: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )

    approval_token: Optional[str] = Field(
        default=None,
        max_length=2048,
    )

    expected_plan_version: int = Field(
        ge=1,
    )


class RemediationExecutionRequest(APIInputModel):
    confirm: bool

    dry_run: bool = True

    idempotency_key: str = Field(
        min_length=1,
        max_length=MAX_IDEMPOTENCY_KEY_LENGTH,
    )

    expected_plan_version: int = Field(
        ge=1,
    )

    @model_validator(
        mode="after",
    )
    def validate_confirmation(
        self,
    ) -> "RemediationExecutionRequest":
        if not self.confirm:
            raise ValueError("remediation execution requires confirm=true")

        return self


class RemediationStepResponse(APIOutputModel):
    step_id: str

    action: str

    provider: str

    state: str

    reversible: bool

    started_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None

    error: Optional[APIError] = None

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class RemediationPlanResponse(APIOutputModel):
    plan_id: str

    version: int = Field(
        ge=1,
    )

    incident_id: str

    decision_id: str

    title: str

    description: Optional[str] = None

    risk: str

    status: str

    dry_run: bool

    approval_required: bool

    approved_by: Optional[str] = None

    approved_at: Optional[datetime] = None

    steps: List[RemediationStepResponse] = Field(
        default_factory=list,
    )

    created_at: datetime

    updated_at: datetime

    attributes: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "API_SCHEMA_VERSION",
    "MAX_REQUEST_ID_LENGTH",
    "MAX_IDEMPOTENCY_KEY_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_MESSAGE_LENGTH",
    "MAX_CONTENT_LENGTH",
    "MAX_BATCH_SIZE",
    "MAX_FEATURE_COUNT",
    "MAX_ATTRIBUTE_COUNT",
    "APIStatus",
    "HealthStatus",
    "OperationState",
    "AcceleratorKindAPI",
    "PrecisionAPI",
    "WorkloadKindAPI",
    "OptimizationLevelAPI",
    "ModelFamilyAPI",
    "ModelStateAPI",
    "InferenceStateAPI",
    "IncidentStatusAPI",
    "SeverityAPI",
    "DecisionStatusAPI",
    "RemediationPlanStatusAPI",
    "RemediationRiskAPI",
    "APIInputModel",
    "APIOutputModel",
    "APIResponseMetadata",
    "APIErrorField",
    "APIError",
    "APIErrorResponse",
    "OperationResponse",
    "PaginationMetadata",
    "HealthComponentResponse",
    "HealthResponse",
    "LivenessResponse",
    "ReadinessResponse",
    "InferenceProvenanceRequest",
    "InferenceProvenanceResponse",
    "EventInferenceRequest",
    "AnomalyInferenceRequest",
    "BatchInferenceItemRequest",
    "BatchInferenceRequest",
    "ModelIdentityResponse",
    "PredictionResponse",
    "EventInferenceDetailResponse",
    "AnomalyInferenceDetailResponse",
    "InferenceResponse",
    "BatchInferenceItemResponse",
    "BatchInferenceResponse",
    "AcceleratorSelectionRequest",
    "AcceleratorMemoryResponse",
    "AcceleratorDeviceResponse",
    "AcceleratorInventoryResponse",
    "AcceleratorSelectionResponse",
    "AcceleratorWorkloadRequest",
    "OptimizationConstraintsRequest",
    "AcceleratorOptimizationRequest",
    "OptimizationRecommendationResponse",
    "AcceleratorOptimizationResponse",
    "AcceleratorMetricsResponse",
    "ModelResponse",
    "ModelListResponse",
    "ModelOperationRequest",
    "ModelRefreshRequest",
    "ModelRefreshResponse",
    "IncidentSignalRequest",
    "IncidentCreateRequest",
    "IncidentResponse",
    "IncidentListResponse",
    "DecisionEvaluateRequest",
    "DecisionResponse",
    "RemediationStepRequest",
    "RemediationPlanCreateRequest",
    "RemediationApprovalRequest",
    "RemediationExecutionRequest",
    "RemediationStepResponse",
    "RemediationPlanResponse",
]
