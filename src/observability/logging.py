from __future__ import annotations

"""
Structured, bounded logging for OSHA application
components.
"""

import logging as pylogging

from collections import deque
from threading import RLock

from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.core.serialization import (
    DEFAULT_SENSITIVE_KEYS,
    to_json_safe,
)

from src.core.validation import (
    integer,
)

from src.observability.config import (
    LoggingConfig,
)

from src.observability.events import (
    SEVERITY_RANK,
    severity_at_least,
)

from src.observability.schema import (
    LogRecord,
    ObservabilityContext,
    Severity,
)


# ---------------------------------------------------------------
# Python logging levels
# ---------------------------------------------------------------

PYTHON_LEVELS = {
    Severity.DEBUG.value: (pylogging.DEBUG),
    Severity.INFO.value: (pylogging.INFO),
    Severity.WARNING.value: (pylogging.WARNING),
    Severity.ERROR.value: (pylogging.ERROR),
    Severity.CRITICAL.value: (pylogging.CRITICAL),
}


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class ObservabilityLoggingError(RuntimeError):
    pass


# ---------------------------------------------------------------
# Safe exception metadata
# ---------------------------------------------------------------


def safe_exception_metadata(
    error: BaseException,
) -> Dict[str, Any]:
    """
    Return bounded exception metadata.

    The arbitrary exception message is
    deliberately NOT copied because exception
    messages can contain credentials, connection
    strings, payloads, or other sensitive
    information.
    """

    return {
        "error_type": (type(error).__name__),
        "error_module": (type(error).__module__),
    }


# ---------------------------------------------------------------
# Structured log buffer
# ---------------------------------------------------------------


class LogBuffer:
    """
    Thread-safe bounded buffer for structured
    OSHA LogRecord objects.

    This is the application's in-process logging
    model. Durable logging/export happens later
    through observability exporters.
    """

    def __init__(
        self,
        *,
        config: Optional[LoggingConfig] = None,
        maximum_records: int = 5000,
        redact_sensitive: bool = True,
    ) -> None:
        self.config = config or LoggingConfig()

        self.config.validate()

        self.maximum_records = integer(
            maximum_records,
            name="maximum_records",
            minimum=1,
        )

        self.redact_sensitive = bool(redact_sensitive)

        self._records: Deque[LogRecord] = deque(maxlen=(self.maximum_records))

        self._lock = RLock()

        self._accepted = 0
        self._filtered = 0
        self._evicted = 0

    # ---------------------------------------------------------
    # Acceptance
    # ---------------------------------------------------------

    def accepts(
        self,
        severity: str,
    ) -> bool:
        if severity not in SEVERITY_RANK:
            raise (ObservabilityLoggingError(f"Unknown log severity: {severity}"))

        return bool(
            self.config.enabled
            and severity_at_least(
                severity,
                self.config.minimum_severity,
            )
        )

    # ---------------------------------------------------------
    # Record an existing LogRecord
    # ---------------------------------------------------------

    def record(
        self,
        record: LogRecord,
    ) -> bool:
        record.finalize()

        with self._lock:
            if not self.accepts(record.severity):
                self._filtered += 1

                return False

            # Re-sanitize attributes before
            # putting them into the log buffer.
            record.attributes = to_json_safe(
                record.attributes,
                redact_sensitive=(self.redact_sensitive),
                sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
            )

            was_full = len(self._records) == self._records.maxlen

            self._records.append(record)

            self._accepted += 1

            if was_full:
                self._evicted += 1

            return True

    # ---------------------------------------------------------
    # Generic log creation
    # ---------------------------------------------------------

    def log(
        self,
        *,
        message: str,
        context: ObservabilityContext,
        severity: str = (Severity.INFO.value),
        logger: str = "osha",
        error: Optional[BaseException] = None,
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[LogRecord]:
        """
        Construct and record one structured log.

        Applications should provide their own
        bounded, intentional message.

        When an exception is supplied, only its
        type/module are recorded by default.
        """

        if not self.accepts(severity):
            with self._lock:
                self._filtered += 1

            return None

        safe_attributes = to_json_safe(
            dict(attributes or {}),
            redact_sensitive=(self.redact_sensitive),
            sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
        )

        if error is not None:
            safe_attributes = {
                **safe_attributes,
                **safe_exception_metadata(error),
            }

        record = LogRecord(
            message=message,
            context=context,
            severity=severity,
            logger=logger,
            error_type=(type(error).__name__ if error is not None else None),
            reference_ids=list(reference_ids),
            attributes=(safe_attributes),
        ).finalize()

        if self.record(record):
            return record

        return None

    # ---------------------------------------------------------
    # Convenience log methods
    # ---------------------------------------------------------

    def debug(
        self,
        *,
        message: str,
        context: ObservabilityContext,
        **kwargs: Any,
    ) -> Optional[LogRecord]:
        return self.log(
            message=message,
            context=context,
            severity=(Severity.DEBUG.value),
            **kwargs,
        )

    def info(
        self,
        *,
        message: str,
        context: ObservabilityContext,
        **kwargs: Any,
    ) -> Optional[LogRecord]:
        return self.log(
            message=message,
            context=context,
            severity=(Severity.INFO.value),
            **kwargs,
        )

    def warning(
        self,
        *,
        message: str,
        context: ObservabilityContext,
        **kwargs: Any,
    ) -> Optional[LogRecord]:
        return self.log(
            message=message,
            context=context,
            severity=(Severity.WARNING.value),
            **kwargs,
        )

    def error(
        self,
        *,
        message: str,
        context: ObservabilityContext,
        error: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LogRecord]:
        return self.log(
            message=message,
            context=context,
            severity=(Severity.ERROR.value),
            error=error,
            **kwargs,
        )

    def critical(
        self,
        *,
        message: str,
        context: ObservabilityContext,
        error: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LogRecord]:
        return self.log(
            message=message,
            context=context,
            severity=(Severity.CRITICAL.value),
            error=error,
            **kwargs,
        )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    def list(
        self,
        *,
        minimum_severity: Optional[str] = None,
        service: Optional[str] = None,
        component: Optional[str] = None,
        logger: Optional[str] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[LogRecord]:
        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            records = list(self._records)

        if newest_first:
            records.reverse()

        output: List[LogRecord] = []

        for record in records:
            if minimum_severity is not None and not severity_at_least(
                record.severity,
                minimum_severity,
            ):
                continue

            if service is not None and record.context.service != service:
                continue

            if component is not None and record.context.component != component:
                continue

            if logger is not None and record.logger != logger:
                continue

            output.append(record)

            if limit is not None and len(output) >= limit:
                break

        return output

    # ---------------------------------------------------------
    # Drain
    # ---------------------------------------------------------

    def drain(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[LogRecord]:
        """
        Remove and return oldest buffered logs.

        Intended for future exporter use.
        """

        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            count = (
                len(self._records)
                if limit is None
                else min(
                    limit,
                    len(self._records),
                )
            )

            return [self._records.popleft() for _ in range(count)]

    # ---------------------------------------------------------
    # Clear
    # ---------------------------------------------------------

    def clear(
        self,
    ) -> int:
        with self._lock:
            count = len(self._records)

            self._records.clear()

            return count

    # ---------------------------------------------------------
    # Length
    # ---------------------------------------------------------

    def __len__(
        self,
    ) -> int:
        with self._lock:
            return len(self._records)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            by_severity: Dict[
                str,
                int,
            ] = {}

            for record in self._records:
                by_severity[record.severity] = (
                    by_severity.get(
                        record.severity,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (self.config.enabled),
                "minimum_severity": (self.config.minimum_severity),
                "structured": (self.config.structured),
                "include_context": (self.config.include_context),
                "redact_sensitive": (self.redact_sensitive),
                "buffered_records": (len(self._records)),
                "maximum_records": (self.maximum_records),
                "accepted_total": (self._accepted),
                "filtered_total": (self._filtered),
                "evicted_total": (self._evicted),
                "buffered_by_severity": (by_severity),
            }


# ---------------------------------------------------------------
# Python logging adapter
# ---------------------------------------------------------------


class PythonLoggingSink:
    """
    Optional adapter for emitting a structured
    OSHA LogRecord through Python's standard
    logging package.

    It does not configure file handlers, remote
    transports, or credentials.
    """

    def __init__(
        self,
        logger: Optional[pylogging.Logger] = None,
    ) -> None:
        self.logger = logger or pylogging.getLogger("osha")

    def emit(
        self,
        record: LogRecord,
    ) -> None:
        record.finalize()

        level = PYTHON_LEVELS.get(
            record.severity,
            pylogging.INFO,
        )

        self.logger.log(
            level,
            record.message,
            extra={
                "osha_context": (record.context.to_dict()),
                "osha_attributes": (to_json_safe(record.attributes)),
                "osha_reference_ids": (list(record.reference_ids)),
                "osha_error_type": (record.error_type),
            },
        )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "PYTHON_LEVELS",
    "ObservabilityLoggingError",
    "safe_exception_metadata",
    "LogBuffer",
    "PythonLoggingSink",
]
