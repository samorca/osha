from __future__ import annotations

"""
Accelerator selection and runtime management for OSHA.

AcceleratorManager composes:

    AcceleratorConfig
    AcceleratorDetector
    AcceleratorInventory
    AcceleratorRequirements
    AcceleratorBackend

to provide:

    capability-aware device selection
    explicit device selection
    backend-priority selection
    memory-aware ranking
    controlled CPU fallback
    runtime-device construction
    synchronization
    cache clearing
    release handling
    health reporting

The manager does not:

    train models
    load model artifacts
    move caller-owned tensors
    execute inference
    perform infrastructure remediation
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
)

from src.accelerators.config import (
    AcceleratorConfig,
    safe_config_summary,
)

from src.accelerators.detector import (
    AcceleratorDetectionResult,
    AcceleratorDetector,
)

from src.accelerators.errors import (
    AcceleratorCapabilityMissingError,
    AcceleratorConfigurationError,
    AcceleratorDeviceNotFoundError,
    AcceleratorDeviceUnavailableError,
    AcceleratorFallbackDisabledError,
    AcceleratorFallbackUnavailableError,
    AcceleratorInsufficientMemoryError,
    AcceleratorMixedPrecisionUnsupportedError,
    AcceleratorNoMatchingDeviceError,
    AcceleratorPrecisionUnsupportedError,
    AcceleratorRuntimeDeviceError,
    AcceleratorSelectionError,
    selection_failure_details,
)

from src.accelerators.schema import (
    AcceleratorCapability,
    AcceleratorDevice,
    AcceleratorHealthReport,
    AcceleratorInventory,
    AcceleratorKind,
    AcceleratorRequirements,
    AcceleratorSelection,
    AcceleratorState,
)


# ---------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------

class AcceleratorCandidate:
    """
    One evaluated selection candidate.

    A candidate retains its rejection reasons so
    selection failures remain explainable.
    """

    def __init__(
        self,
        *,
        device: AcceleratorDevice,
        eligible: bool,
        rejection_reasons: Iterable[
            str
        ] = (),
        score: Tuple[Any, ...] = (),
        fallback_candidate: bool = False,
    ) -> None:
        device.validate()

        self.device = device

        self.eligible = bool(
            eligible
        )

        self.rejection_reasons = tuple(
            dict.fromkeys(
                str(
                    reason
                ).strip()
                for reason
                in rejection_reasons
                if str(
                    reason
                ).strip()
            )
        )

        self.score = tuple(
            score
        )

        self.fallback_candidate = bool(
            fallback_candidate
        )

        if (
            self.eligible
            and self.rejection_reasons
        ):
            raise ValueError(
                "Eligible accelerator "
                "candidate cannot contain "
                "rejection reasons."
            )

        if (
            not self.eligible
            and not self.
            rejection_reasons
        ):
            raise ValueError(
                "Rejected accelerator "
                "candidate requires at "
                "least one reason."
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "device_id": (
                self.device.device_id
            ),

            "kind": self.device.kind,

            "runtime_device": (
                self.device.runtime_device
            ),

            "eligible": self.eligible,

            "rejection_reasons": list(
                self.rejection_reasons
            ),

            "score": list(
                self.score
            ),

            "fallback_candidate": (
                self.fallback_candidate
            ),
        }


# ---------------------------------------------------------------
# Selection attempt
# ---------------------------------------------------------------

class AcceleratorSelectionAttempt:
    """
    Complete evaluation of one selection request.
    """

    def __init__(
        self,
        *,
        requirements: AcceleratorRequirements,
        inventory: AcceleratorInventory,
        candidates: Iterable[
            AcceleratorCandidate
        ],
    ) -> None:
        requirements.validate()
        inventory.validate()

        self.requirements = requirements
        self.inventory = inventory
        self.candidates = tuple(
            candidates
        )

    @property
    def eligible_candidates(
        self,
    ) -> Tuple[
        AcceleratorCandidate,
        ...,
    ]:
        return tuple(
            candidate
            for candidate
            in self.candidates
            if candidate.eligible
        )

    @property
    def rejected_candidates(
        self,
    ) -> Tuple[
        AcceleratorCandidate,
        ...,
    ]:
        return tuple(
            candidate
            for candidate
            in self.candidates
            if not candidate.eligible
        )

    def rejection_map(
        self,
    ) -> Dict[
        str,
        Tuple[str, ...],
    ]:
        return {
            candidate.device.device_id: (
                candidate.
                rejection_reasons
            )
            for candidate
            in self.rejected_candidates
        }

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "requirements": (
                self.requirements.to_dict()
            ),

            "inventory": (
                self.inventory.to_dict()
            ),

            "candidate_count": len(
                self.candidates
            ),

            "eligible_count": len(
                self.eligible_candidates
            ),

            "rejected_count": len(
                self.rejected_candidates
            ),

            "candidates": [
                candidate.to_dict()
                for candidate
                in self.candidates
            ],
        }


# ---------------------------------------------------------------
# Backend-priority utilities
# ---------------------------------------------------------------

def backend_priority_index(
    kind: str,
    priority: Tuple[
        str,
        ...,
    ],
) -> int:
    """
    Return a stable backend-priority index.

    Unknown custom backends are placed after
    configured backends.
    """

    try:
        return priority.index(
            kind
        )

    except ValueError:
        return len(
            priority
        )


# ---------------------------------------------------------------
# Memory ranking utilities
# ---------------------------------------------------------------

def known_memory_rank(
    value: Optional[int],
) -> Tuple[int, int]:
    """
    Rank known memory above unknown memory.

    Larger known values rank higher when the
    overall candidate score is sorted ascending
    because we negate the byte value.
    """

    if value is None:
        return (
            1,
            0,
        )

    return (
        0,
        -int(
            value
        ),
    )


# ---------------------------------------------------------------
# Accelerator manager
# ---------------------------------------------------------------

class AcceleratorManager:
    """
    Select and operate accelerator devices.

    Selection order:

        explicit device_id
        explicit backend + index
        preferred backend
        configured auto backend priority
        optional CPU fallback

    Every successful selection returns an
    AcceleratorSelection with reason and fallback
    metadata.
    """

    def __init__(
        self,
        *,
        config: Optional[
            AcceleratorConfig
        ] = None,
        detector: Optional[
            AcceleratorDetector
        ] = None,
    ) -> None:
        self.config = (
            config
            or AcceleratorConfig()
        )

        self.config.validate()

        self.detector = (
            detector
            or AcceleratorDetector(
                config=self.config
            )
        )

        self._lock = RLock()

        self._last_detection: Optional[
            AcceleratorDetectionResult
        ] = None

        self._last_attempt: Optional[
            AcceleratorSelectionAttempt
        ] = None

        self._last_selection: Optional[
            AcceleratorSelection
        ] = None

        self._active_selections: Dict[
            str,
            AcceleratorSelection,
        ] = {}

    # ---------------------------------------------------------
    # Enabled guard
    # ---------------------------------------------------------

    def _require_enabled(
        self,
    ) -> None:
        if not self.config.enabled:
            raise AcceleratorConfigurationError(
                "Accelerator subsystem is "
                "disabled."
            )

    # ---------------------------------------------------------
    # Detection
    # ---------------------------------------------------------

    def detect(
        self,
        *,
        refresh: bool = True,
        fail_fast: Optional[
            bool
        ] = None,
    ) -> AcceleratorDetectionResult:
        self._require_enabled()

        if not refresh:
            with self._lock:
                cached = self._last_detection

            if cached is not None:
                return cached

        result = self.detector.detect(
            fail_fast=fail_fast
        )

        with self._lock:
            self._last_detection = (
                result
            )

        return result

    # ---------------------------------------------------------
    # Inventory
    # ---------------------------------------------------------

    def inventory(
        self,
        *,
        refresh: bool = True,
    ) -> AcceleratorInventory:
        return self.detect(
            refresh=refresh
        ).inventory

    # ---------------------------------------------------------
    # Backend lookup
    # ---------------------------------------------------------

    def backend(
        self,
        kind: str,
    ) -> AcceleratorBackend:
        return (
            self.detector.
            require_backend(
                kind
            )
        )

    # ---------------------------------------------------------
    # Requirements
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
        return self.config.requirements(
            training=training,
            inference=inference,
            minimum_total_memory_bytes=(
                minimum_total_memory_bytes
            ),
            minimum_free_memory_bytes=(
                minimum_free_memory_bytes
            ),
            preferred_kind=(
                preferred_kind
            ),
            device_index=device_index,
            device_id=device_id,
            require_accelerator=(
                require_accelerator
            ),
            allow_cpu_fallback=(
                allow_cpu_fallback
            ),
            mixed_precision=(
                mixed_precision
            ),
            required_capabilities=(
                required_capabilities
            ),
            required_precisions=(
                required_precisions
            ),
        )

    # ---------------------------------------------------------
    # Explicit candidate filtering
    # ---------------------------------------------------------

    @staticmethod
    def _explicit_identity_reasons(
        device: AcceleratorDevice,
        requirements: AcceleratorRequirements,
    ) -> List[str]:
        reasons: List[str] = []

        if (
            requirements.device_id
            is not None
            and device.device_id
            != requirements.device_id
        ):
            reasons.append(
                "device_id_mismatch"
            )

        if (
            requirements.device_index
            is not None
            and device.index
            != requirements.device_index
        ):
            reasons.append(
                "device_index_mismatch"
            )

        return reasons

    # ---------------------------------------------------------
    # Requirement matching
    # ---------------------------------------------------------

    @staticmethod
    def _workload_reasons(
        device: AcceleratorDevice,
        requirements: AcceleratorRequirements,
        *,
        ignore_kind: bool = False,
    ) -> List[str]:
        """
        Apply workload constraints.

        ignore_kind=True is used only when
        evaluating CPU as an explicit fallback
        for a non-CPU preferred backend.
        """

        reasons: List[str] = []

        if not device.available:
            reasons.append(
                "device_unavailable"
            )

        if (
            device.degraded
            and not requirements.
            allow_degraded
        ):
            reasons.append(
                "degraded_device_not_allowed"
            )

        if (
            not ignore_kind
            and requirements.
            preferred_kind
            != AcceleratorKind.AUTO.value
            and device.kind
            != requirements.
            preferred_kind
        ):
            reasons.append(
                "backend_kind_mismatch"
            )

        reasons.extend(
            AcceleratorManager.
            _explicit_identity_reasons(
                device,
                requirements,
            )
        )

        if (
            requirements.
            require_accelerator
            and not device.accelerator
        ):
            reasons.append(
                "hardware_accelerator_required"
            )

        if (
            requirements.
            minimum_total_memory_bytes
            is not None
        ):
            total = (
                device.memory.
                total_bytes
            )

            if (
                total is None
                or total
                < requirements.
                minimum_total_memory_bytes
            ):
                reasons.append(
                    "insufficient_total_memory"
                )

        if (
            requirements.
            minimum_free_memory_bytes
            is not None
        ):
            free = (
                device.memory.
                free_bytes
            )

            if (
                free is None
                or free
                < requirements.
                minimum_free_memory_bytes
            ):
                reasons.append(
                    "insufficient_free_memory"
                )

        for capability in (
            requirements.
            required_capabilities
        ):
            if not device.supports_capability(
                capability
            ):
                reasons.append(
                    "missing_capability:"
                    + capability
                )

        for precision in (
            requirements.
            required_precisions
        ):
            if not device.supports_precision(
                precision
            ):
                reasons.append(
                    "missing_precision:"
                    + precision
                )

        if (
            requirements.training
            and not device.
            supports_capability(
                AcceleratorCapability.
                TRAINING.value
            )
        ):
            reasons.append(
                "training_capability_missing"
            )

        if (
            requirements.inference
            and not device.
            supports_capability(
                AcceleratorCapability.
                INFERENCE.value
            )
        ):
            reasons.append(
                "inference_capability_missing"
            )

        if (
            requirements.mixed_precision
            and not device.
            supports_capability(
                AcceleratorCapability.
                MIXED_PRECISION.value
            )
        ):
            reasons.append(
                "mixed_precision_capability_missing"
            )

        return list(
            dict.fromkeys(
                reasons
            )
        )

    # ---------------------------------------------------------
    # Candidate score
    # ---------------------------------------------------------

    def _candidate_score(
        self,
        device: AcceleratorDevice,
        requirements: AcceleratorRequirements,
        *,
        fallback_candidate: bool,
    ) -> Tuple[Any, ...]:
        """
        Lower score wins.

        Ranking is deterministic and uses:

            fallback status
            explicit device match
            preferred backend match
            configured backend priority
            availability state
            free memory
            total memory
            device index
            device ID
        """

        explicit_id_penalty = (
            0
            if (
                requirements.device_id
                is not None
                and device.device_id
                == requirements.device_id
            )
            else (
                1
                   device_id is not None
                else 0
            )
        )

        explicit_index_penalty = (
            0
            if (
                requirements.device_index
                is not None
                and device.index
                == requirements.device_index
            )
            else (
                1
                if requirements.
                device_index
                is not None
                else 0
            )
        )

        preferred_kind_penalty = (
            0
            if (
                requirements.
                preferred_kind
                == AcceleratorKind.
                AUTO.value
                or device.kind
                == requirements.
                preferred_kind
            )
            else 1
        )

        backend_priority = (
            backend_priority_index(
                device.kind,
                self.config.
                discovery.
                backend_priority,
            )
        )

        degraded_penalty = (
            1
            if device.state
            == AcceleratorState.
            DEGRADED.value
            else 0
        )

        if (
            self.config.
            selection.
            prefer_highest_free_memory
        ):
            free_memory_rank = (
                known_memory_rank(
                    device.memory.
                    free_bytes
                )
            )

            total_memory_rank = (
                known_memory_rank(
                    device.memory.
                    total_bytes
                )
            )

        else:
            free_memory_rank = (
                0,
                0,
            )

            total_memory_rank = (
                0,
                0,
            )

        index_rank = (
            device.index
            if device.index
            is not None
            else -1
        )

        return (
            1
            if fallback_candidate
            else 0,

            explicit_id_penalty,

            explicit_index_penalty,

            preferred_kind_penalty,

            backend_priority,

            degraded_penalty,

            *free_memory_rank,

            *total_memory_rank,

            index_rank,

            device.device_id,
        )

    # ---------------------------------------------------------
    # Evaluate candidates
    # ---------------------------------------------------------

    def evaluate(
        self,
        requirements: AcceleratorRequirements,
        *,
        inventory: Optional[
            AcceleratorInventory
        ] = None,
        refresh: bool = True,
    ) -> AcceleratorSelectionAttempt:
        self._require_enabled()

        requirements.validate()

        source = (
            inventory
            or self.inventory(
                refresh=refresh
            )
        )

        source.validate()

        candidates: List[
            AcceleratorCandidate
        ] = []

        preferred_kind = (
            requirements.
            preferred_kind
        )

        fallback_allowed = (
            requirements.
            allow_cpu_fallback
            and not requirements.
            require_accelerator
        )

        for device in source.devices:
            fallback_candidate = bool(
                fallback_allowed
                and preferred_kind
                not in {
                    AcceleratorKind.
                    AUTO.value,
                    AcceleratorKind.
                    CPU.value,
                }
                and device.kind
                == AcceleratorKind.
                CPU.value
            )

            ignore_kind = (
                fallback_candidate
            )

            reasons = (
                self._workload_reasons(
                    device,
                    requirements,
                    ignore_kind=(
                        ignore_kind
                    ),
                )
            )

            # CPU fallback must not bypass explicit
            # device identity. An explicit cuda:1
            # request should not silently select CPU.
            if (
                fallback_candidate
                and (
                    requirements.device_id
                    is not None
                    or requirements.
                    device_index
                    is not None
                )
            ):
                reasons.append(
                    "explicit_device_does_not_allow_fallback"
                )

            reasons = list(
                dict.fromkeys(
                    reasons
                )
            )

            eligible = not reasons

            candidates.append(
                AcceleratorCandidate(
                    device=device,
                    eligible=eligible,
                    rejection_reasons=(
                        ()
                        if eligible
                        else reasons
                    ),
                    score=(
                        self.
                        _candidate_score(
                            device,
                            requirements,
                            fallback_candidate=(
                                fallback_candidate
                            ),
                        )
                    ),
                    fallback_candidate=(
                        fallback_candidate
                    ),
                )
            )

        attempt = (
            AcceleratorSelectionAttempt(
                requirements=requirements,
                inventory=source,
                candidates=candidates,
            )
        )

        with self._lock:
            self._last_attempt = attempt

        return attempt

    # ---------------------------------------------------------
    # Failure classification
    # ---------------------------------------------------------

    @staticmethod
    def _raise_selection_failure(
        attempt: AcceleratorSelectionAttempt,
    ) -> None:
        requirements = (
            attempt.requirements
        )

        rejection_map = (
            attempt.rejection_map()
        )

        all_reasons = {
            reason
            for reasons
            in rejection_map.values()
            for reason
            in reasons
        }

        details = (
            selection_failure_details(
                preferred_kind=(
                    requirements.
                    preferred_kind
                ),
                device_id=(
                    requirements.device_id
                ),
                device_index=(
                    requirements.
                    device_index
                ),
                candidates=(
                    rejection_map
                ),
                fallback_allowed=(
                    requirements.
                    allow_cpu_fallback
                ),
            )
        )

        if requirements.device_id:
            if (
                attempt.inventory.get(
                    requirements.device_id
                )
                is None
            ):
                raise (
                    AcceleratorDeviceNotFoundError(
                        "Explicit accelerator "
                        "device was not found.",
                        details=details,
                    )
                )

        if (
            requirements.
            preferred_kind
            not in {
                AcceleratorKind.
                AUTO.value,
                AcceleratorKind.
                CPU.value,
            }
            and not requirements.
            allow_cpu_fallback
        ):
            preferred_devices = (
                attempt.inventory.
                by_kind(
                    requirements.
                    preferred_kind,
                    available_only=True,
                )
            )

            if not preferred_devices:
                raise (
                    AcceleratorFallbackDisabledError(
                        "Preferred accelerator "
                        "is unavailable and CPU "
                        "fallback is disabled.",
                        details=details,
                    )
                )

        memory_reasons = {
            "insufficient_total_memory",
            "insufficient_free_memory",
        }

        if (
            all_reasons
            and all(
                reason in memory_reasons
                for reason
                in all_reasons
            )
        ):
            raise (
                AcceleratorInsufficientMemoryError(
                    "No accelerator device "
                    "satisfies the workload "
                    "memory requirements.",
                    details=details,
                )
            )

        if any(
            reason.startswith(
                "missing_precision:"
            )
            for reason
            in all_reasons
        ):
            raise (
                AcceleratorPrecisionUnsupportedError(
                    "No accelerator device "
                    "supports the requested "
                    "precision.",
                    details=details,
                )
            )

        if (
            "mixed_precision_capability_missing"
            in all_reasons
        ):
            raise (
                AcceleratorMixedPrecisionUnsupportedError(
                    "No accelerator device "
                    "supports the requested "
                    "mixed-precision policy.",
                    details=details,
                )
            )

        if any(
            reason.startswith(
                "missing_capability:"
            )
            or reason
            in {
                "training_capability_missing",
                "inference_capability_missing",
            }
            for reason
            in all_reasons
        ):
            raise (
                AcceleratorCapabilityMissingError(
                    "No accelerator device "
                    "supports all required "
                    "capabilities.",
                    details=details,
                )
            )

        if (
            requirements.
            allow_cpu_fallback
            and requirements.
            preferred_kind
            not in {
                AcceleratorKind.
                AUTO.value,
                AcceleratorKind.
                CPU.value,
            }
            and not attempt.inventory.
            by_kind(
                AcceleratorKind.
                CPU.value,
                available_only=True,
            )
        ):
            raise (
                AcceleratorFallbackUnavailableError(
                    "Preferred accelerator "
                    "is unavailable and no "
                    "usable CPU fallback was "
                    "detected.",
                    details=details,
                )
            )

        raise AcceleratorNoMatchingDeviceError(
            "No accelerator device "
            "satisfies the workload "
            "requirements.",
            details=details,
        )

    # ---------------------------------------------------------
    # Selection reason
    # ---------------------------------------------------------

    @staticmethod
    def _selection_reason(
        candidate: AcceleratorCandidate,
        requirements: AcceleratorRequirements,
    ) -> str:
        device = candidate.device

        if requirements.device_id:
            return (
                "Explicit device identity "
                "satisfies all workload "
                "requirements."
            )

        if (
            requirements.device_index
            is not None
            and requirements.
            preferred_kind
            != AcceleratorKind.
            AUTO.value
        ):
            return (
                "Explicit backend device "
                "index satisfies all "
                "workload requirements."
            )

        if candidate.fallback_candidate:
            return (
                "Preferred accelerator "
                "could not satisfy the "
                "workload; CPU fallback "
                "was selected."
            )

        if (
            requirements.
            preferred_kind
            == AcceleratorKind.
            AUTO.value
        ):
            return (
                "Automatic accelerator "
                "selection chose the "
                "highest-ranked eligible "
                "device."
            )

        return (
            "Preferred accelerator "
            "backend satisfies all "
            "workload requirements."
        )

    # ---------------------------------------------------------
    # Select
    # ---------------------------------------------------------

    def select(
        self,
        requirements: Optional[
            AcceleratorRequirements
        ] = None,
        *,
        inventory: Optional[
            AcceleratorInventory
        ] = None,
        refresh: bool = True,
        training: bool = False,
        inference: bool = False,
    ) -> AcceleratorSelection:
        """
        Select one accelerator device.

        If requirements is omitted, requirements
        are built from AcceleratorConfig plus the
        training/inference flags.
        """

        self._require_enabled()

        resolved_requirements = (
            requirements
            or self.requirements(
                training=training,
                inference=inference,
            )
        )

        resolved_requirements.validate()

        attempt = self.evaluate(
            resolved_requirements,
            inventory=inventory,
            refresh=refresh,
        )

        eligible = sorted(
            attempt.eligible_candidates,
            key=lambda candidate: (
                candidate.score
            ),
        )

        if not eligible:
            self._raise_selection_failure(
                attempt
            )

        selected = eligible[
            0
        ]

        selection = (
            AcceleratorSelection(
                device=selected.device,

                requirements=(
                    resolved_requirements
                ),

                reason=(
                    self.
                    _selection_reason(
                        selected,
                        resolved_requirements,
                    )
                ),

                fallback_used=(
                    selected.
                    fallback_candidate
                ),

                candidates_considered=tuple(
                    candidate.
                    device.device_id
                    for candidate
                    in attempt.candidates
                ),

                rejected_candidates=(
                    attempt.rejection_map()
                ),

                attributes={
                    "candidate_score": list(
                        selected.score
                    ),

                    "backend_priority": list(
                        self.config.
                        discovery.
                        backend_priority
                    ),

                    "prefer_highest_free_memory": (
                        self.config.
                        selection.
                        prefer_highest_free_memory
                    ),
                },
            )
        )

        selection.validate()

        with self._lock:
            self._last_selection = (
                selection
            )

            self._active_selections[
                selection.device_id
            ] = selection

        return selection

    # ---------------------------------------------------------
    # Convenience training selection
    # ---------------------------------------------------------

    def select_for_training(
        self,
        *,
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
        required_precisions: Tuple[
            str,
            ...,
        ] = (),
        refresh: bool = True,
    ) -> AcceleratorSelection:
        requirements = (
            self.requirements(
                training=True,
                minimum_total_memory_bytes=(
                    minimum_total_memory_bytes
                ),
                minimum_free_memory_bytes=(
                    minimum_free_memory_bytes
                ),
                preferred_kind=(
                    preferred_kind
                ),
                device_index=(
                    device_index
                ),
                device_id=device_id,
                require_accelerator=(
                    require_accelerator
                ),
                allow_cpu_fallback=(
                    allow_cpu_fallback
                ),
                mixed_precision=(
                    mixed_precision
                ),
                required_precisions=(
                    required_precisions
                ),
            )
        )

        return self.select(
            requirements,
            refresh=refresh,
        )

    # ---------------------------------------------------------
    # Convenience inference selection
    # ---------------------------------------------------------

    def select_for_inference(
        self,
        *,
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
        required_precisions: Tuple[
            str,
            ...,
        ] = (),
        refresh: bool = True,
    ) -> AcceleratorSelection:
        requirements = (
            self.requirements(
                inference=True,
                minimum_total_memory_bytes=(
                    minimum_total_memory_bytes
                ),
                minimum_free_memory_bytes=(
                    minimum_free_memory_bytes
                ),
                preferred_kind=(
                    preferred_kind
                ),
                device_index=(
                    device_index
                ),
                device_id=device_id,
                require_accelerator=(
                    require_accelerator
                ),
                allow_cpu_fallback=(
                    allow_cpu_fallback
                ),
                mixed_precision=(
                    mixed_precision
                ),
                required_precisions=(
                    required_precisions
                ),
            )
        )

        return self.select(
            requirements,
            refresh=refresh,
        )

    # ---------------------------------------------------------
    # Resolve selection
    # ---------------------------------------------------------

    def _selection(
        self,
        selection: Optional[
            AcceleratorSelection
        ],
    ) -> AcceleratorSelection:
        resolved = (
            selection
            or self.last_selection
        )

        if resolved is None:
            raise AcceleratorSelectionError(
                "No accelerator device has "
                "been selected."
            )

        resolved.validate()

        return resolved

    # ---------------------------------------------------------
    # Runtime device
    # ---------------------------------------------------------

    def runtime_device(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ) -> Any:
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        try:
            return backend.runtime_device(
                resolved.device
            )

        except Exception as exc:
            if isinstance(
                exc,
                AcceleratorRuntimeDeviceError,
            ):
                raise

            raise AcceleratorRuntimeDeviceError(
                "Selected accelerator "
                "runtime device could not "
                "be constructed.",
                details={
                    "device_id": (
                        resolved.device_id
                    ),
                    "kind": (
                        resolved.kind
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            ) from exc

    # ---------------------------------------------------------
    # Refresh selected device
    # ---------------------------------------------------------

    def refresh_device(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ) -> AcceleratorDevice:
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        return backend.refresh(
            resolved.device
        )

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def memory(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ):
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        return backend.memory(
            resolved.device
        )

    # ---------------------------------------------------------
    # Synchronize
    # ---------------------------------------------------------

    def synchronize(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ) -> None:
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        backend.synchronize(
            resolved.device
        )

    # ---------------------------------------------------------
    # Clear cache
    # ---------------------------------------------------------

    def clear_cache(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ) -> None:
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        backend.clear_cache(
            resolved.device
        )

    # ---------------------------------------------------------
    # Allocation failure
    # ---------------------------------------------------------

    def allocation_failure(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ) -> None:
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        backend.allocation_failure(
            resolved.device
        )

    # ---------------------------------------------------------
    # Release
    # ---------------------------------------------------------

    def release(
        self,
        selection: Optional[
            AcceleratorSelection
        ] = None,
    ) -> None:
        resolved = self._selection(
            selection
        )

        backend = self.backend(
            resolved.kind
        )

        backend.release(
            resolved.device
        )

        with self._lock:
            self._active_selections.pop(
                resolved.device_id,
                None,
            )

            if (
                self._last_selection
                is not None
                and self._last_selection.
                device_id
                == resolved.device_id
            ):
                self._last_selection = (
                    None
                )

    # ---------------------------------------------------------
    # Release all
    # ---------------------------------------------------------

    def release_all(
        self,
    ) -> List[str]:
        with self._lock:
            selections = list(
                self._active_selections.
                values()
            )

        released: List[str] = []

        for selection in selections:
            self.release(
                selection
            )

            released.append(
                selection.device_id
            )

        return released

    # ---------------------------------------------------------
    # Last properties
    # ---------------------------------------------------------

    @property
    def last_detection(
        self,
    ) -> Optional[
        AcceleratorDetectionResult
    ]:
        with self._lock:
            return self._last_detection

    @property
    def last_attempt(
        self,
    ) -> Optional[
        AcceleratorSelectionAttempt
    ]:
        with self._lock:
            return self._last_attempt

    @property
    def last_selection(
        self,
    ) -> Optional[
        AcceleratorSelection
    ]:
        with self._lock:
            return self._last_selection

    # ---------------------------------------------------------
    # Active selections
    # ---------------------------------------------------------

    def active_selections(
        self,
    ) -> Tuple[
        AcceleratorSelection,
        ...,
    ]:
        with self._lock:
            return tuple(
                self._active_selections.
                values()
            )

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health(
        self,
        *,
        deep: Optional[
            bool
        ] = None,
        refresh_inventory: bool = True,
    ) -> AcceleratorHealthReport:
        return self.detector.health(
            deep=deep,
            refresh_inventory=(
                refresh_inventory
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        *,
        refresh_inventory: bool = False,
    ) -> Dict[str, Any]:
        inventory = None

        if refresh_inventory:
            inventory = self.inventory(
                refresh=True
            )

        elif self.last_detection is not None:
            inventory = (
                self.last_detection.
                inventory
            )

        last_selection = (
            self.last_selection
        )

        last_attempt = (
            self.last_attempt
        )

        return {
            "config": (
                safe_config_summary(
                    self.config
                )
            ),

            "inventory": (
                inventory.to_dict()
                if inventory is not None
                else None
            ),

            "last_selection": (
                last_selection.to_dict()
                if last_selection
                is not None
                else None
            ),

            "last_attempt": (
                last_attempt.to_dict()
                if last_attempt
                is not None
                else None
            ),

            "active_selection_count": len(
                self.active_selections()
            ),

            "active_selections": [
                selection.to_dict()
                for selection
                in self.active_selections()
            ],

            "training_executed": (
                False
            ),

            "inference_executed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Factory
# ---------------------------------------------------------------

def create_accelerator_manager(
    *,
    config: Optional[
        AcceleratorConfig
    ] = None,
    detector: Optional[
        AcceleratorDetector
    ] = None,
) -> AcceleratorManager:
    return AcceleratorManager(
        config=config,
        detector=detector,
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AcceleratorCandidate",
    "AcceleratorSelectionAttempt",

    "backend_priority_index",
    "known_memory_rank",

    "AcceleratorManager",

    "create_accelerator_manager",
]
