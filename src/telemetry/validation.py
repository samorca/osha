from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.telemetry.schema import (
    Provider,
    Relationship,
    RelationshipType,
    ResourceIdentity,
    ResourceType,
    Severity,
    Signal,
    SignalType,
    TelemetryRecord,
)


class ValidationLevel(
    str,
    Enum,
):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """
    One validation problem discovered in a
    normalized telemetry object.
    """

    level: str

    code: str

    message: str

    field: Optional[str] = None

    value: Any = None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "value": self.value,
        }


@dataclass
class ValidationResult:
    """
    Collection of validation results.

    Errors make the object invalid.

    Warnings identify questionable or incomplete
    data but do not make the object unusable.
    """

    issues: List[
        ValidationIssue
    ] = field(
        default_factory=list
    )

    def add_error(
        self,
        code: str,
        message: str,
        *,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                level=(
                    ValidationLevel.ERROR.value
                ),
                code=code,
                message=message,
                field=field_name,
                value=value,
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        *,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                level=(
                    ValidationLevel.WARNING.value
                ),
                code=code,
                message=message,
                field=field_name,
                value=value,
            )
        )

    @property
    def errors(
        self,
    ) -> List[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.level
            == ValidationLevel.ERROR.value
        ]

    @property
    def warnings(
        self,
    ) -> List[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.level
            == ValidationLevel.WARNING.value
        ]

    @property
    def valid(
        self,
    ) -> bool:
        return not self.errors

    def extend(
        self,
        other: "ValidationResult",
    ) -> None:
        self.issues.extend(
            other.issues
        )

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "error_count": len(
                self.errors
            ),
            "warning_count": len(
                self.warnings
            ),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


def enum_values(
    enum_class,
) -> set[str]:
    return {
        item.value
        for item in enum_class
    }


PROVIDERS = enum_values(
    Provider
)

RESOURCE_TYPES = enum_values(
    ResourceType
)

SIGNAL_TYPES = enum_values(
    SignalType
)

SEVERITIES = enum_values(
    Severity
)

RELATIONSHIP_TYPES = (
    enum_values(
        RelationshipType
    )
)


def non_empty_string(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            str,
        )
        and bool(
            value.strip()
        )
    )


def valid_number(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return False

    if not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return False

    if isinstance(
        value,
        float,
    ):
        return math.isfinite(
            value
        )

    return True


def timestamp_valid(
    value: Any,
) -> bool:
    """
    Verify that a timestamp is non-empty and can
    be interpreted as ISO-8601.

    ISO dates ending in Z are normalized for
    Python's datetime parser.
    """

    if not non_empty_string(
        value
    ):
        return False

    text = str(
        value
    ).strip()

    if text.endswith(
        "Z"
    ):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        datetime.fromisoformat(
            text
        )

        return True

    except ValueError:
        return False


def mapping_valid(
    value: Any,
) -> bool:
    return isinstance(
        value,
        Mapping,
    )


def validate_attributes(
    attributes: Any,
    *,
    path: str,
) -> ValidationResult:
    result = ValidationResult()

    if not isinstance(
        attributes,
        Mapping,
    ):
        result.add_error(
            "invalid_attributes",
            (
                "Attributes must be "
                "a mapping."
            ),
            field_name=path,
            value=type(
                attributes
            ).__name__,
        )

        return result

    for key in attributes.keys():
        if not isinstance(
            key,
            str,
        ):
            result.add_error(
                "invalid_attribute_key",
                (
                    "Attribute keys must "
                    "be strings."
                ),
                field_name=path,
                value=str(key),
            )

    return result


def validate_resource(
    resource: ResourceIdentity,
) -> ValidationResult:
    """
    Validate a normalized ResourceIdentity.
    """

    result = ValidationResult()

    if not isinstance(
        resource,
        ResourceIdentity,
    ):
        result.add_error(
            "invalid_resource_object",
            (
                "Resource must be a "
                "ResourceIdentity instance."
            ),
        )

        return result

    if not non_empty_string(
        resource.provider
    ):
        result.add_error(
            "missing_provider",
            "Resource provider is required.",
            field_name="resource.provider",
        )

    elif (
        resource.provider
        not in PROVIDERS
    ):
        # Custom providers can still be supported
        # through Provider.OTHER.
        result.add_warning(
            "unknown_provider",
            (
                "Provider is not currently in "
                "the common Provider enum."
            ),
            field_name="resource.provider",
            value=resource.provider,
        )

    if not non_empty_string(
        resource.resource_type
    ):
        result.add_error(
            "missing_resource_type",
            (
                "Resource type is required."
            ),
            field_name=(
                "resource.resource_type"
            ),
        )

    elif (
        resource.resource_type
        not in RESOURCE_TYPES
    ):
        result.add_warning(
            "unknown_resource_type",
            (
                "Resource type is not in the "
                "common ResourceType enum."
            ),
            field_name=(
                "resource.resource_type"
            ),
            value=(
                resource.resource_type
            ),
        )

    if not non_empty_string(
        resource.provider_id
    ):
        result.add_error(
            "missing_provider_id",
            (
                "Provider resource ID "
                "is required."
            ),
            field_name=(
                "resource.provider_id"
            ),
        )

    if not non_empty_string(
        resource.environment_id
    ):
        result.add_error(
            "missing_environment",
            (
                "Resource environment_id "
                "is required."
            ),
            field_name=(
                "resource.environment_id"
            ),
        )

    if (
        resource.canonical_id
        is not None
        and not non_empty_string(
            resource.canonical_id
        )
    ):
        result.add_error(
            "invalid_canonical_id",
            (
                "canonical_id must be "
                "non-empty when supplied."
            ),
            field_name=(
                "resource.canonical_id"
            ),
        )

    if (
        resource.resource_type
        == ResourceType.UNKNOWN.value
    ):
        result.add_warning(
            "unknown_resource",
            (
                "Resource could not be mapped "
                "to a common resource type."
            ),
            field_name=(
                "resource.resource_type"
            ),
        )

    result.extend(
        validate_attributes(
            resource.attributes,
            path=(
                "resource.attributes"
            ),
        )
    )

    try:
        effective_id = (
            resource.effective_id()
        )

        if not non_empty_string(
            effective_id
        ):
            result.add_error(
                "invalid_effective_id",
                (
                    "Resource effective ID "
                    "is empty."
                ),
            )

    except Exception as exc:
        result.add_error(
            "effective_id_failure",
            (
                "Unable to generate "
                "resource effective ID."
            ),
            value=str(exc),
        )

    return result


def validate_signal(
    signal: Signal,
) -> ValidationResult:
    """
    Validate one normalized Signal.
    """

    result = ValidationResult()

    if not isinstance(
        signal,
        Signal,
    ):
        result.add_error(
            "invalid_signal_object",
            (
                "Signal must be a Signal "
                "instance."
            ),
        )

        return result

    if not non_empty_string(
        signal.signal_type
    ):
        result.add_error(
            "missing_signal_type",
            (
                "Signal type is required."
            ),
            field_name=(
                "signal.signal_type"
            ),
        )

    elif (
        signal.signal_type
        not in SIGNAL_TYPES
    ):
        result.add_warning(
            "unknown_signal_type",
            (
                "Signal type is not in the "
                "common SignalType enum."
            ),
            field_name=(
                "signal.signal_type"
            ),
            value=signal.signal_type,
        )

    if not non_empty_string(
        signal.provider
    ):
        result.add_error(
            "missing_signal_provider",
            (
                "Signal provider is required."
            ),
            field_name=(
                "signal.provider"
            ),
        )

    elif (
        signal.provider
        not in PROVIDERS
    ):
        result.add_warning(
            "unknown_signal_provider",
            (
                "Signal provider is not "
                "currently in Provider enum."
            ),
            field_name=(
                "signal.provider"
            ),
            value=signal.provider,
        )

    if not non_empty_string(
        signal.resource_id
    ):
        result.add_error(
            "missing_signal_resource",
            (
                "Signal resource_id "
                "is required."
            ),
            field_name=(
                "signal.resource_id"
            ),
        )

    if not timestamp_valid(
        signal.timestamp
    ):
        result.add_error(
            "invalid_signal_timestamp",
            (
                "Signal timestamp must be "
                "valid ISO-8601."
            ),
            field_name=(
                "signal.timestamp"
            ),
            value=signal.timestamp,
        )

    if not non_empty_string(
        signal.environment_id
    ):
        result.add_error(
            "missing_signal_environment",
            (
                "Signal environment_id "
                "is required."
            ),
            field_name=(
                "signal.environment_id"
            ),
        )

    if (
        not non_empty_string(
            signal.severity
        )
        or signal.severity
        not in SEVERITIES
    ):
        result.add_warning(
            "unknown_severity",
            (
                "Signal severity is not "
                "a common Severity value."
            ),
            field_name=(
                "signal.severity"
            ),
            value=signal.severity,
        )

    if (
        signal.signal_type
        == SignalType.METRIC.value
    ):
        if signal.value is None:
            result.add_warning(
                "metric_without_value",
                (
                    "Metric signal has no "
                    "measurement value."
                ),
                field_name=(
                    "signal.value"
                ),
            )

        elif isinstance(
            signal.value,
            float,
        ) and not math.isfinite(
            signal.value
        ):
            result.add_error(
                "invalid_metric_value",
                (
                    "Metric value must not "
                    "be NaN or infinity."
                ),
                field_name=(
                    "signal.value"
                ),
                value=signal.value,
            )

    if (
        signal.signal_type
        == SignalType.LOG.value
        and not signal.message
    ):
        result.add_warning(
            "log_without_message",
            (
                "Log signal does not have "
                "a message body."
            ),
            field_name=(
                "signal.message"
            ),
        )

    if (
        signal.signal_type
        == SignalType.EVENT.value
        and not signal.signal_name
    ):
        result.add_warning(
            "event_without_name",
            (
                "Event signal does not have "
                "a stable event name."
            ),
            field_name=(
                "signal.signal_name"
            ),
        )

    result.extend(
        validate_attributes(
            signal.attributes,
            path=(
                "signal.attributes"
            ),
        )
    )

    return result


def validate_relationship(
    relationship: Relationship,
) -> ValidationResult:
    """
    Validate one topology/resource relationship.
    """

    result = ValidationResult()

    if not isinstance(
        relationship,
        Relationship,
    ):
        result.add_error(
            "invalid_relationship_object",
            (
                "Relationship must be a "
                "Relationship instance."
            ),
        )

        return result

    if not non_empty_string(
        relationship.source_resource_id
    ):
        result.add_error(
            "missing_relationship_source",
            (
                "Relationship source "
                "resource ID is required."
            ),
            field_name=(
                "relationship."
                "source_resource_id"
            ),
        )

    if not non_empty_string(
        relationship.target_resource_id
    ):
        result.add_error(
            "missing_relationship_target",
            (
                "Relationship target "
                "resource ID is required."
            ),
            field_name=(
                "relationship."
                "target_resource_id"
            ),
        )

    if (
        relationship.source_resource_id
        == relationship.target_resource_id
    ):
        result.add_warning(
            "self_relationship",
            (
                "Relationship source and "
                "target are identical."
            ),
        )

    if not non_empty_string(
        relationship.relationship_type
    ):
        result.add_error(
            "missing_relationship_type",
            (
                "Relationship type "
                "is required."
            ),
            field_name=(
                "relationship."
                "relationship_type"
            ),
        )

    elif (
        relationship.relationship_type
        not in RELATIONSHIP_TYPES
    ):
        result.add_warning(
            "unknown_relationship_type",
            (
                "Relationship type is not "
                "in RelationshipType enum."
            ),
            field_name=(
                "relationship."
                "relationship_type"
            ),
            value=(
                relationship.
                relationship_type
            ),
        )

    if not non_empty_string(
        relationship.provider
    ):
        result.add_error(
            "missing_relationship_provider",
            (
                "Relationship provider "
                "is required."
            ),
            field_name=(
                "relationship.provider"
            ),
        )

    if not timestamp_valid(
        relationship.timestamp
    ):
        result.add_error(
            "invalid_relationship_timestamp",
            (
                "Relationship timestamp "
                "must be ISO-8601."
            ),
            field_name=(
                "relationship.timestamp"
            ),
            value=(
                relationship.timestamp
            ),
        )

    if (
        relationship.confidence
        is not None
    ):
        if not valid_number(
            relationship.confidence
        ):
            result.add_error(
                "invalid_relationship_confidence",
                (
                    "Relationship confidence "
                    "must be numeric."
                ),
                field_name=(
                    "relationship."
                    "confidence"
                ),
                value=(
                    relationship.
                    confidence
                ),
            )

        elif not (
            0.0
            <= float(
                relationship.confidence
            )
            <= 1.0
        ):
            result.add_error(
                "relationship_confidence_range",
                (
                    "Relationship confidence "
                    "must be between 0 and 1."
                ),
                field_name=(
                    "relationship."
                    "confidence"
                ),
                value=(
                    relationship.
                    confidence
                ),
            )

    result.extend(
        validate_attributes(
            relationship.evidence,
            path=(
                "relationship.evidence"
            ),
        )
    )

    return result


def validate_record(
    telemetry: TelemetryRecord,
) -> ValidationResult:
    """
    Validate a complete canonical telemetry
    record.

    This validates both individual structures and
    important cross-field consistency.
    """

    result = ValidationResult()

    if not isinstance(
        telemetry,
        TelemetryRecord,
    ):
        result.add_error(
            "invalid_telemetry_record",
            (
                "Object must be a "
                "TelemetryRecord."
            ),
        )

        return result

    result.extend(
        validate_resource(
            telemetry.resource
        )
    )

    result.extend(
        validate_signal(
            telemetry.signal
        )
    )

    for index, relationship in enumerate(
        telemetry.relationships
    ):
        relationship_result = (
            validate_relationship(
                relationship
            )
        )

        for issue in (
            relationship_result.issues
        ):
            field_name = (
                issue.field
            )

            if field_name:
                field_name = (
                    "relationships"
                    f"[{index}]."
                    f"{field_name}"
                )
            else:
                field_name = (
                    "relationships"
                    f"[{index}]"
                )

            result.issues.append(
                ValidationIssue(
                    level=issue.level,
                    code=issue.code,
                    message=issue.message,
                    field=field_name,
                    value=issue.value,
                )
            )

    # Cross-field checks
    if (
        telemetry.signal.provider
        != telemetry.resource.provider
    ):
        result.add_error(
            "provider_mismatch",
            (
                "Signal and resource provider "
                "must match before correlation."
            ),
            value={
                "resource_provider": (
                    telemetry.resource.provider
                ),
                "signal_provider": (
                    telemetry.signal.provider
                ),
            },
        )

    if (
        telemetry.signal.environment_id
        != telemetry.resource.environment_id
    ):
        result.add_error(
            "environment_mismatch",
            (
                "Signal and resource "
                "environment_id must match."
            ),
            value={
                "resource_environment": (
                    telemetry.resource.
                    environment_id
                ),
                "signal_environment": (
                    telemetry.signal.
                    environment_id
                ),
            },
        )

    expected_resource_id = (
        telemetry.resource.effective_id()
    )

    if (
        telemetry.signal.resource_id
        != expected_resource_id
    ):
        result.add_error(
            "resource_reference_mismatch",
            (
                "Signal resource_id does not "
                "reference the normalized "
                "resource identity."
            ),
            value={
                "expected": (
                    expected_resource_id
                ),
                "actual": (
                    telemetry.signal.
                    resource_id
                ),
            },
        )

    if not isinstance(
        telemetry.native,
        Mapping,
    ):
        result.add_error(
            "invalid_native_record",
            (
                "TelemetryRecord.native "
                "must be a mapping."
            ),
            field_name="native",
        )

    if not non_empty_string(
        telemetry.schema_version
    ):
        result.add_error(
            "missing_schema_version",
            (
                "schema_version is required."
            ),
            field_name=(
                "schema_version"
            ),
        )

    if not timestamp_valid(
        telemetry.normalized_at
    ):
        result.add_error(
            "invalid_normalized_timestamp",
            (
                "normalized_at must be "
                "valid ISO-8601."
            ),
            field_name=(
                "normalized_at"
            ),
            value=(
                telemetry.normalized_at
            ),
        )

    # record_id may not exist until finalize()
    # runs, so finalize before verifying it.
    try:
        telemetry.finalize()
    except Exception as exc:
        result.add_error(
            "record_finalize_failure",
            (
                "Unable to finalize "
                "telemetry record."
            ),
            value=str(exc),
        )

        return result

    if not non_empty_string(
        telemetry.record_id
    ):
        result.add_error(
            "missing_record_id",
            (
                "Finalized telemetry record "
                "must have a record_id."
            ),
            field_name="record_id",
        )

    return result


def validate_records(
    records: Iterable[
        TelemetryRecord
    ],
) -> List[
    tuple[
        TelemetryRecord,
        ValidationResult,
    ]
]:
    """
    Validate a batch without discarding anything.
    """

    output = []

    for telemetry in records:
        output.append(
            (
                telemetry,
                validate_record(
                    telemetry
                ),
            )
        )

    return output


def valid_records(
    records: Iterable[
        TelemetryRecord
    ],
) -> List[TelemetryRecord]:
    """
    Return only records without validation errors.

    Warnings are allowed.
    """

    output = []

    for telemetry in records:
        result = validate_record(
            telemetry
        )

        if result.valid:
            output.append(
                telemetry
            )

    return output


def invalid_records(
    records: Iterable[
        TelemetryRecord
    ],
) -> List[
    tuple[
        TelemetryRecord,
        ValidationResult,
    ]
]:
    """
    Return records containing at least one
    validation error.

    This is useful for a future quarantine
    pipeline.
    """

    output = []

    for telemetry in records:
        result = validate_record(
            telemetry
        )

        if not result.valid:
            output.append(
                (
                    telemetry,
                    result,
                )
            )

    return output


def validation_summary(
    results: Iterable[
        ValidationResult
    ],
) -> Dict[str, Any]:
    """
    Build a small aggregate suitable for runtime
    health and observability.
    """

    total = 0
    valid = 0
    invalid = 0
    errors = 0
    warnings = 0

    codes: Dict[
        str,
        int,
    ] = {}

    for result in results:
        total += 1

        if result.valid:
            valid += 1
        else:
            invalid += 1

        errors += len(
            result.errors
        )

        warnings += len(
            result.warnings
        )

        for issue in result.issues:
            codes[
                issue.code
            ] = (
                codes.get(
                    issue.code,
                    0,
                )
                + 1
            )

    return {
        "records_checked": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "error_count": errors,
        "warning_count": warnings,
        "issue_codes": codes,
    }


class TelemetryValidationError(
    ValueError
):
    """
    Raised by strict pipeline validation.
    """

    def __init__(
        self,
        result: ValidationResult,
    ) -> None:
        self.result = result

        messages = [
            (
                f"{issue.code}: "
                f"{issue.message}"
            )
            for issue in (
                result.errors
            )
        ]

        super().__init__(
            "; ".join(messages)
            or "Telemetry validation failed"
        )


def require_valid(
    telemetry: TelemetryRecord,
) -> TelemetryRecord:
    """
    Strict validation helper.

    The normalizer can use this when invalid
    telemetry should be rejected immediately.
    """

    result = validate_record(
        telemetry
    )

    if not result.valid:
        raise TelemetryValidationError(
            result
        )

    return telemetry
