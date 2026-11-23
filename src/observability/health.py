from __future__ import annotations

"""
Internal OSHA component health/readiness probing.
"""

from dataclasses import (
    dataclass,
    field,
)

from threading import RLock

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
)

from src.core.serialization import (
    DEFAULT_SENSITIVE_KEYS,
    to_json_safe,
)

from src.observability.config import (
    HealthConfig,
)

from src.observability.schema import (
    HealthObservation,
    HealthStatus,
    ObservabilityContext,
)


# ---------------------------------------------------------------
# Probe contract
# ---------------------------------------------------------------

HealthProbe = Callable[
    [],
    Any,
]


# ---------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ProbeResult:
    """
    Normalized result from one internal health
    probe.
    """

    status: str

    message: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "ProbeResult":
        if (
            self.status
            not in {
                item.value
                for item
                in HealthStatus
            }
        ):
            raise ValueError(
                "Unknown health status: "
                f"{self.status}"
            )

        return self


# ---------------------------------------------------------------
# Registered probe
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RegisteredProbe:
    """
    Definition of one OSHA internal health
    probe.
    """

    name: str

    probe: HealthProbe

    context: ObservabilityContext

    required: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------
# Aggregate health report
# ---------------------------------------------------------------

@dataclass
class HealthReport:
    status: str

    observations: List[
        HealthObservation
    ] = field(
        default_factory=list
    )

    required_failures: List[
        str
    ] = field(
        default_factory=list
    )

    optional_failures: List[
        str
    ] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "status": self.status,

            "required_failures": (
                sorted(
                    set(
                        self.
                        required_failures
                    )
                )
            ),

            "optional_failures": (
                sorted(
                    set(
                        self.
                        optional_failures
                    )
                )
            ),

            "observations": [
                item.to_dict()
                for item
                in self.observations
            ],
        }


# ---------------------------------------------------------------
# Probe result normalization
# ---------------------------------------------------------------

def normalize_probe_result(
    value: Any,
) -> ProbeResult:
    """
    Accepted probe outputs:

      ProbeResult
      bool
      health-status string
      mapping

    Examples:

      True

      "healthy"

      {
          "status": "degraded",
          "message": "...",
          "attributes": {...}
      }
    """

    if isinstance(
        value,
        ProbeResult,
    ):
        return value.finalize()

    if isinstance(
        value,
        bool,
    ):
        return ProbeResult(
            status=(
                HealthStatus.
                HEALTHY.value
                if value
                else HealthStatus.
                UNHEALTHY.value
            )
        )

    if isinstance(
        value,
        Mapping,
    ):
        return ProbeResult(
            status=str(
                value.get(
                    "status",
                    HealthStatus.
                    UNKNOWN.value,
                )
            ),
            message=(
                value.get(
                    "message"
                )
            ),
            attributes=dict(
                value.get(
                    "attributes",
                    {},
                )
            ),
        ).finalize()

    if (
        isinstance(
            value,
            str,
        )
        and value
        in {
            item.value
            for item
            in HealthStatus
        }
    ):
        return ProbeResult(
            status=value
        )

    raise TypeError(
        "Health probe must return "
        "bool, status string, mapping, "
        "or ProbeResult"
    )


# ---------------------------------------------------------------
# Health registry
# ---------------------------------------------------------------

class HealthRegistry:
    """
    Register and safely execute health probes for
    OSHA internal application components.

    Probe failures are converted into bounded
    unhealthy observations rather than exposing
    arbitrary exception messages.
    """

    def __init__(
        self,
        *,
        config: Optional[
            HealthConfig
        ] = None,
        redact_sensitive: bool = True,
    ) -> None:
        self.config = (
            config
            or HealthConfig()
        )

        self.config.validate()

        self.redact_sensitive = bool(
            redact_sensitive
        )

        self._probes: Dict[
            str,
            RegisteredProbe,
        ] = {}

        self._observations: List[
            HealthObservation
        ] = []

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        probe: HealthProbe,
        context: ObservabilityContext,
        required: bool = True,
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        normalized = (
            name.strip()
        )

        if not normalized:
            raise ValueError(
                "Health probe name "
                "is required"
            )

        context.finalize()

        with self._lock:
            if (
                normalized
                in self._probes
            ):
                raise ValueError(
                    "Health probe already "
                    "registered: "
                    f"{normalized}"
                )

            self._probes[
                normalized
            ] = RegisteredProbe(
                name=normalized,
                probe=probe,
                context=context,
                required=required,
                attributes=dict(
                    attributes
                    or {}
                ),
            )

    # ---------------------------------------------------------
    # Unregister
    # ---------------------------------------------------------

    def unregister(
        self,
        name: str,
    ) -> bool:
        with self._lock:
            return (
                self._probes.pop(
                    name,
                    None,
                )
                is not None
            )

    # ---------------------------------------------------------
    # Names
    # ---------------------------------------------------------

    def names(
        self,
    ) -> List[str]:
        with self._lock:
            return sorted(
                self._probes
            )

    # ---------------------------------------------------------
    # Execute one registered probe
    # ---------------------------------------------------------

    def _observe(
        self,
        registered: RegisteredProbe,
    ) -> HealthObservation:
        try:
            result = (
                normalize_probe_result(
                    registered.probe()
                )
            )

            attributes = {
                **registered.attributes,
                **result.attributes,
            }

        except Exception as exc:
            result = ProbeResult(
                status=(
                    HealthStatus.
                    UNHEALTHY.value
                ),
                message=(
                    "Health probe failed."
                ),
                attributes={
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_module": (
                        type(exc).__module__
                    ),
                },
            )

            attributes = {
                **registered.attributes,
                **result.attributes,
            }

        return (
            HealthObservation(
                component=(
                    registered.name
                ),
                status=result.status,
                context=(
                    registered.context
                ),
                message=result.message,
                required=(
                    registered.required
                ),
                attributes=(
                    to_json_safe(
                        attributes,
                        redact_sensitive=(
                            self.
                            redact_sensitive
                        ),
                        sensitive_keys=(
                            DEFAULT_SENSITIVE_KEYS
                        ),
                    )
                ),
            ).finalize()
        )

    # ---------------------------------------------------------
    # Check one
    # ---------------------------------------------------------

    def check(
        self,
        name: str,
    ) -> HealthObservation:
        with self._lock:
            registered = (
                self._probes.get(
                    name
                )
            )

        if registered is None:
            raise KeyError(
                "Unknown health probe: "
                f"{name}"
            )

        observation = (
            self._observe(
                registered
            )
        )

        with self._lock:
            self._append_observation(
                observation
            )

        return observation

    # ---------------------------------------------------------
    # Check all
    # ---------------------------------------------------------

    def check_all(
        self,
    ) -> HealthReport:
        """
        Execute all registered probes.

        Aggregate rules:

          required issue -> UNHEALTHY
          optional issue -> DEGRADED
          all observed healthy -> HEALTHY
          no conclusive observations -> UNKNOWN
        """

        if not self.config.enabled:
            return HealthReport(
                status=(
                    HealthStatus.
                    UNKNOWN.value
                )
            )

        with self._lock:
            probes = [
                self._probes[
                    name
                ]
                for name
                in sorted(
                    self._probes
                )
            ]

        observations = [
            self._observe(
                probe
            )
            for probe
            in probes
        ]

        required_failures: List[
            str
        ] = []

        optional_failures: List[
            str
        ] = []

        for observation in (
            observations
        ):
            if (
                observation.status
                in {
                    HealthStatus.
                    UNHEALTHY.value,

                    HealthStatus.
                    UNKNOWN.value,
                }
            ):
                target = (
                    required_failures
                    if observation.required
                    else optional_failures
                )

                target.append(
                    observation.component
                )

            elif (
                observation.status
                == HealthStatus.
                DEGRADED.value
            ):
                target = (
                    required_failures
                    if observation.required
                    else optional_failures
                )

                target.append(
                    observation.component
                )

        if required_failures:
            status = (
                HealthStatus.
                UNHEALTHY.value
            )

        elif optional_failures:
            status = (
                HealthStatus.
                DEGRADED.value
            )

        elif (
            observations
            and all(
                item.status
                == HealthStatus.
                HEALTHY.value
                for item
                in observations
            )
        ):
            status = (
                HealthStatus.
                HEALTHY.value
            )

        else:
            status = (
                HealthStatus.
                UNKNOWN.value
            )

        with self._lock:
            for observation in (
                observations
            ):
                self._append_observation(
                    observation
                )

        return HealthReport(
            status=status,
            observations=observations,
            required_failures=(
                required_failures
            ),
            optional_failures=(
                optional_failures
            ),
        )

    # ---------------------------------------------------------
    # Bounded observation buffer
    # ---------------------------------------------------------

    def _append_observation(
        self,
        observation: HealthObservation,
    ) -> None:
        self._observations.append(
            observation
        )

        maximum = (
            self.config.
            maximum_buffered_observations
        )

        if (
            len(
                self._observations
            )
            > maximum
        ):
            remove_count = (
                len(
                    self._observations
                )
                - maximum
            )

            del self._observations[
                :remove_count
            ]

    # ---------------------------------------------------------
    # Observations
    # ---------------------------------------------------------

    def observations(
        self,
    ) -> List[
        HealthObservation
    ]:
        with self._lock:
            return list(
                self._observations
            )

    # ---------------------------------------------------------
    # Drain
    # ---------------------------------------------------------

    def drain(
        self,
    ) -> List[
        HealthObservation
    ]:
        with self._lock:
            values = list(
                self._observations
            )

            self._observations.clear()

            return values

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            statuses: Dict[
                str,
                int,
            ] = {}

            for observation in (
                self._observations
            ):
                statuses[
                    observation.status
                ] = (
                    statuses.get(
                        observation.status,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (
                    self.config.enabled
                ),

                "registered_probes": (
                    len(
                        self._probes
                    )
                ),

                "required_probes": (
                    sum(
                        1
                        for item
                        in self._probes.
                        values()
                        if item.required
                    )
                ),

                "optional_probes": (
                    sum(
                        1
                        for item
                        in self._probes.
                        values()
                        if not item.required
                    )
                ),

                "buffered_observations": (
                    len(
                        self._observations
                    )
                ),

                "maximum_buffered_observations": (
                    self.config.
                    maximum_buffered_observations
                ),

                "buffered_by_status": (
                    statuses
                ),

                "scope": (
                    "osha_application_components"
                ),

                "external_infrastructure_health_claimed": (
                    False
                ),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "HealthProbe",

    "ProbeResult",
    "RegisteredProbe",
    "HealthReport",

    "normalize_probe_result",

    "HealthRegistry",
]
