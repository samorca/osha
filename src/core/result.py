from __future__ import annotations

"""
Generic bounded result envelopes shared by OSHA
subsystems.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
)

from src.core.exceptions import (
    OshaError,
    ValidationError,
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
# Generic result value
# ---------------------------------------------------------------

T = TypeVar("T")


# ---------------------------------------------------------------
# Result status
# ---------------------------------------------------------------


class ResultStatus(
    str,
    Enum,
):
    SUCCESS = "success"

    PARTIAL = "partial"

    BLOCKED = "blocked"

    FAILED = "failed"

    CANCELLED = "cancelled"

    SKIPPED = "skipped"


# ---------------------------------------------------------------
# Error information
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ErrorInfo:
    """
    Bounded error metadata.

    details are excluded from serialization by
    default because they may contain internal or
    sensitive subsystem information.
    """

    code: str

    message: str

    retryable: bool = False

    error_type: Optional[str] = None

    details: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Construct from exception
    # ---------------------------------------------------------

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        expose_message: bool = False,
        include_details: bool = False,
        fallback_message: str = ("Operation failed."),
    ) -> "ErrorInfo":
        """
        Convert an exception into bounded
        metadata.

        Exception messages are NOT exposed by
        default.
        """

        if isinstance(
            error,
            OshaError,
        ):
            return cls(
                code=error.code,
                message=(error.message if expose_message else fallback_message),
                retryable=(error.retryable),
                error_type=(type(error).__name__),
                details=(dict(error.details) if include_details else {}),
            ).finalize()

        return cls(
            code="unexpected_error",
            message=(str(error) if expose_message else fallback_message),
            retryable=False,
            error_type=(type(error).__name__),
        ).finalize()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def finalize(
        self,
    ) -> "ErrorInfo":
        required_string(
            self.code,
            name="error code",
            maximum_length=128,
        )

        required_string(
            self.message,
            name="error message",
            maximum_length=2048,
        )

        return self

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(
        self,
        *,
        include_details: bool = False,
    ) -> Dict[str, Any]:
        self.finalize()

        value: Dict[
            str,
            Any,
        ] = {
            "code": self.code,
            "message": (self.message),
            "retryable": (self.retryable),
            "error_type": (self.error_type),
        }

        if include_details and self.details:
            value["details"] = to_json_safe(self.details)

        return value


# ---------------------------------------------------------------
# Warning information
# ---------------------------------------------------------------


@dataclass(frozen=True)
class WarningInfo:
    """
    Non-fatal warning associated with an OSHA
    result.
    """

    code: str

    message: str

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "WarningInfo":
        required_string(
            self.code,
            name="warning code",
            maximum_length=128,
        )

        required_string(
            self.message,
            name="warning message",
            maximum_length=2048,
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "code": self.code,
            "message": (self.message),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Generic result
# ---------------------------------------------------------------


@dataclass
class Result(Generic[T]):
    """
    Generic OSHA result envelope.

    The value contains the immediate result when
    appropriate.

    reference_ids should be preferred when the
    authoritative result belongs to another
    subsystem.
    """

    status: str

    value: Optional[T] = None

    errors: List[ErrorInfo] = field(default_factory=list)

    warnings: List[WarningInfo] = field(default_factory=list)

    reference_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_now)

    # ---------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------

    @property
    def success(
        self,
    ) -> bool:
        return self.status in {
            ResultStatus.SUCCESS.value,
            ResultStatus.PARTIAL.value,
        }

    @property
    def failed(
        self,
    ) -> bool:
        return self.status == ResultStatus.FAILED.value

    @property
    def blocked(
        self,
    ) -> bool:
        return self.status == ResultStatus.BLOCKED.value

    @property
    def retryable(
        self,
    ) -> bool:
        """
        A result is retryable only when:

          at least one error exists
          AND
          every recorded error is retryable

        Higher-level policy can still refuse the
        retry.
        """

        return bool(self.errors) and all(error.retryable for error in self.errors)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def finalize(
        self,
    ) -> "Result[T]":
        allowed = {item.value for item in ResultStatus}

        if self.status not in allowed:
            raise ValidationError(f"Unknown result status: {self.status}")

        self.errors = [item.finalize() for item in self.errors]

        self.warnings = [item.finalize() for item in self.warnings]

        self.reference_ids = unique_strings(self.reference_ids)

        if self.status == ResultStatus.SUCCESS.value and self.errors:
            raise ValidationError("A successful result cannot contain errors.")

        if self.status == ResultStatus.FAILED.value and not self.errors:
            raise ValidationError("A failed result must contain at least one error.")

        if self.status == ResultStatus.BLOCKED.value and not self.errors:
            raise ValidationError(
                "A blocked result must contain at least one error explaining the block."
            )

        return self

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(
        self,
        *,
        include_value: bool = True,
        include_error_details: bool = False,
    ) -> Dict[str, Any]:
        self.finalize()

        output: Dict[
            str,
            Any,
        ] = {
            "status": self.status,
            "success": (self.success),
            "retryable": (self.retryable),
            "errors": [
                item.to_dict(include_details=(include_error_details))
                for item in self.errors
            ],
            "warnings": [item.to_dict() for item in self.warnings],
            "reference_ids": list(self.reference_ids),
            "attributes": (to_json_safe(self.attributes)),
            "created_at": (self.created_at),
        }

        if include_value:
            output["value"] = to_json_safe(self.value)

        return output

    # ---------------------------------------------------------
    # Successful result
    # ---------------------------------------------------------

    @classmethod
    def ok(
        cls,
        value: Optional[T] = None,
        *,
        reference_ids: Iterable[str] = (),
        warnings: Iterable[WarningInfo] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> "Result[T]":
        return cls(
            status=(ResultStatus.SUCCESS.value),
            value=value,
            warnings=list(warnings),
            reference_ids=list(reference_ids),
            attributes=dict(attributes or {}),
        ).finalize()

    # ---------------------------------------------------------
    # Partial result
    # ---------------------------------------------------------

    @classmethod
    def partial(
        cls,
        value: Optional[T] = None,
        *,
        errors: Iterable[ErrorInfo] = (),
        warnings: Iterable[WarningInfo] = (),
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> "Result[T]":
        return cls(
            status=(ResultStatus.PARTIAL.value),
            value=value,
            errors=list(errors),
            warnings=list(warnings),
            reference_ids=list(reference_ids),
            attributes=dict(attributes or {}),
        ).finalize()

    # ---------------------------------------------------------
    # Failure result
    # ---------------------------------------------------------

    @classmethod
    def fail(
        cls,
        error: ErrorInfo | BaseException,
        *,
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> "Result[T]":
        info = (
            error
            if isinstance(
                error,
                ErrorInfo,
            )
            else (ErrorInfo.from_exception(error))
        )

        return cls(
            status=(ResultStatus.FAILED.value),
            errors=[info],
            reference_ids=list(reference_ids),
            attributes=dict(attributes or {}),
        ).finalize()

    # ---------------------------------------------------------
    # Blocked result
    # ---------------------------------------------------------

    @classmethod
    def blocked_result(
        cls,
        error: ErrorInfo | BaseException,
        *,
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> "Result[T]":
        info = (
            error
            if isinstance(
                error,
                ErrorInfo,
            )
            else (ErrorInfo.from_exception(error))
        )

        return cls(
            status=(ResultStatus.BLOCKED.value),
            errors=[info],
            reference_ids=list(reference_ids),
            attributes=dict(attributes or {}),
        ).finalize()


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "ResultStatus",
    "ErrorInfo",
    "WarningInfo",
    "Result",
]
