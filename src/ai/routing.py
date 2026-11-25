from __future__ import annotations

"""
Model routing for OSHA AI inference.

Routing selects exactly one registered model for
an InferenceRequest.

Selection precedence:

    explicit family + model + version
        ↓
    explicit family + model + latest version
        ↓
    configured family fallback model
        ↓
    unique/latest model for family

Routing does not load or execute models.
"""

import re

from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.errors import (
    AIAmbiguousRouteError,
    AIFamilyNotAllowedError,
    AIModelRouteNotFoundError,
    AIModelStateError,
    AIVersionNotFoundError,
)

from src.ai.registry import (
    AIModelRegistry,
    AIModelRegistryEntry,
)

from src.ai.schema import (
    AIModelDescriptor,
    AIModelFamily,
    AIModelState,
    InferenceRequest,
)


# ---------------------------------------------------------------
# Route result
# ---------------------------------------------------------------


@dataclass(frozen=True)
class AIRoute:
    """
    Result of resolving one inference request to
    one registered model.

    The route references a descriptor, not an
    already-loaded runtime model.
    """

    request_id: str

    descriptor: AIModelDescriptor

    reason: str

    fallback_used: bool = False

    def validate(
        self,
    ) -> None:
        if not str(self.request_id).strip():
            raise AIModelStateError("AI route requires request_id.")

        self.descriptor.finalize()

        if not str(self.reason).strip():
            raise AIModelStateError("AI route requires a selection reason.")

    @property
    def model_id(
        self,
    ) -> str:
        return self.descriptor.identity.model_id

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "request_id": (self.request_id),
            "model_id": (self.model_id),
            "reason": (self.reason),
            "fallback_used": (self.fallback_used),
            "descriptor": (self.descriptor.to_dict()),
        }


# ---------------------------------------------------------------
# Version ordering
# ---------------------------------------------------------------

_VERSION_TOKEN = re.compile(r"\d+|[A-Za-z]+")


def version_key(
    version: str,
) -> Tuple[
    Tuple[
        int,
        Any,
    ],
    ...,
]:
    """
    Build a deterministic natural-sort key.

    Examples:

        1
        2
        10

    are ordered numerically rather than
    lexicographically.

    This is not intended to implement every
    semantic-version rule. It provides stable,
    predictable ordering for OSHA model versions.
    """

    text = str(version).strip()

    if not text:
        return (
            (
                1,
                "",
            ),
        )

    tokens = _VERSION_TOKEN.findall(text)

    if not tokens:
        return (
            (
                1,
                text.lower(),
            ),
        )

    output: List[
        Tuple[
            int,
            Any,
        ]
    ] = []

    for token in tokens:
        if token.isdigit():
            output.append(
                (
                    0,
                    int(token),
                )
            )

        else:
            output.append(
                (
                    1,
                    token.lower(),
                )
            )

    return tuple(output)


# ---------------------------------------------------------------
# Routable states
# ---------------------------------------------------------------


def entry_routable(
    entry: AIModelRegistryEntry,
) -> bool:
    """
    A registered/ready/unloaded model can be
    routed.

    FAILED or unavailable descriptors are not
    selected.
    """

    return entry.descriptor.state in {
        AIModelState.REGISTERED.value,
        AIModelState.READY.value,
        AIModelState.UNLOADED.value,
        AIModelState.DEGRADED.value,
    }


# ---------------------------------------------------------------
# Select latest entry
# ---------------------------------------------------------------


def latest_entry(
    entries: List[AIModelRegistryEntry],
) -> AIModelRegistryEntry:
    if not entries:
        raise AIModelRouteNotFoundError("No AI models are available for routing.")

    return max(
        entries,
        key=lambda item: (
            version_key(item.descriptor.identity.version),
            item.descriptor.identity.model_id,
        ),
    )


# ---------------------------------------------------------------
# Router
# ---------------------------------------------------------------


class AIModelRouter:
    """
    Resolve an inference request against the
    model registry.

    This class selects a descriptor only.
    AIModelRegistry.load() owns runtime loading.
    """

    def __init__(
        self,
        *,
        config: Optional[AIConfig] = None,
        registry: Optional[AIModelRegistry] = None,
    ) -> None:
        self.config = config or AIConfig()

        self.config.validate()

        self.registry = (
            registry
            if registry is not None
            else AIModelRegistry(config=self.config)
        )

    # ---------------------------------------------------------
    # Family guard
    # ---------------------------------------------------------

    def _family(
        self,
        request: InferenceRequest,
    ) -> str:
        request.finalize()

        family = str(request.family).strip().lower()

        if family not in {item.value for item in AIModelFamily}:
            raise AIFamilyNotAllowedError(
                "Inference request has an unsupported model family.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                },
            )

        if not self.config.family_allowed(family):
            raise AIFamilyNotAllowedError(
                "Requested AI model family is disabled.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                },
            )

        return family

    # ---------------------------------------------------------
    # Matching entries
    # ---------------------------------------------------------

    def _entries(
        self,
        *,
        family: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[AIModelRegistryEntry]:
        entries = self.registry.find(
            family=family,
            model_name=model_name,
            version=version,
        )

        return [entry for entry in entries if entry_routable(entry)]

    # ---------------------------------------------------------
    # Configured fallback name
    # ---------------------------------------------------------

    def _fallback_model_name(
        self,
        family: str,
    ) -> Optional[str]:
        if family == AIModelFamily.EVENT.value:
            return self.config.routing.fallback_event_model

        if family == AIModelFamily.ANOMALY.value:
            return self.config.routing.fallback_anomaly_model

        return None

    # ---------------------------------------------------------
    # Exact model/version
    # ---------------------------------------------------------

    def _exact(
        self,
        *,
        request: InferenceRequest,
        family: str,
        model_name: str,
        version: str,
    ) -> AIRoute:
        entries = self._entries(
            family=family,
            model_name=model_name,
            version=version,
        )

        if not entries:
            raise AIVersionNotFoundError(
                "Requested AI model version is not registered or routable.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                    "model_name": (model_name),
                    "version": version,
                },
            )

        if len(entries) != 1:
            raise AIAmbiguousRouteError(
                "Multiple registered AI models match the exact route.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                    "model_name": (model_name),
                    "version": version,
                    "match_count": len(entries),
                },
            )

        return AIRoute(
            request_id=str(request.request_id),
            descriptor=(entries[0].descriptor),
            reason=("explicit_model_version"),
            fallback_used=False,
        )

    # ---------------------------------------------------------
    # Named model latest version
    # ---------------------------------------------------------

    def _named_latest(
        self,
        *,
        request: InferenceRequest,
        family: str,
        model_name: str,
        fallback_used: bool = False,
    ) -> AIRoute:
        entries = self._entries(
            family=family,
            model_name=model_name,
        )

        if not entries:
            raise AIModelRouteNotFoundError(
                "Requested AI model has no registered routable version.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                    "model_name": (model_name),
                },
            )

        if (
            len(entries) > 1
            and not self.config.routing.use_latest_version_when_unspecified
        ):
            raise AIAmbiguousRouteError(
                "Multiple model versions "
                "are available and latest "
                "version routing is "
                "disabled.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                    "model_name": (model_name),
                    "versions": [
                        entry.descriptor.identity.version for entry in entries
                    ],
                },
            )

        entry = latest_entry(entries) if len(entries) > 1 else entries[0]

        return AIRoute(
            request_id=str(request.request_id),
            descriptor=(entry.descriptor),
            reason=(
                "configured_fallback_latest"
                if fallback_used
                else "explicit_model_latest"
            ),
            fallback_used=(fallback_used),
        )

    # ---------------------------------------------------------
    # Family route
    # ---------------------------------------------------------

    def _family_route(
        self,
        *,
        request: InferenceRequest,
        family: str,
    ) -> AIRoute:
        entries = self._entries(family=family)

        if not entries:
            raise AIModelRouteNotFoundError(
                "No routable AI model is registered for the requested family.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                },
            )

        model_names = sorted(
            {entry.descriptor.identity.model_name for entry in entries}
        )

        if len(model_names) == 1:
            return self._named_latest(
                request=request,
                family=family,
                model_name=(model_names[0]),
            )

        if self.config.routing.allow_fallback_model:
            fallback_name = self._fallback_model_name(family)

            if fallback_name:
                return self._named_latest(
                    request=request,
                    family=family,
                    model_name=(fallback_name),
                    fallback_used=True,
                )

        raise AIAmbiguousRouteError(
            "Multiple model names are registered for the requested family.",
            details={
                "request_id": (request.request_id),
                "family": family,
                "models": model_names,
            },
        )

    # ---------------------------------------------------------
    # Route
    # ---------------------------------------------------------

    def route(
        self,
        request: InferenceRequest,
    ) -> AIRoute:
        """
        Resolve one request to exactly one model.

        No model is loaded or executed here.
        """

        request.finalize()

        family = self._family(request)

        model_name = (
            str(request.model_name).strip() if request.model_name is not None else None
        )

        version = (
            str(request.model_version).strip()
            if request.model_version is not None
            else None
        )

        if version and not model_name:
            raise AIModelRouteNotFoundError(
                "A model version cannot be routed without a model name.",
                details={
                    "request_id": (request.request_id),
                    "family": family,
                    "version": version,
                },
            )

        # -----------------------------------------------------
        # Exact requested version
        # -----------------------------------------------------

        if model_name and version:
            route = self._exact(
                request=request,
                family=family,
                model_name=model_name,
                version=version,
            )

            route.validate()

            return route

        # -----------------------------------------------------
        # Named model, latest version
        # -----------------------------------------------------

        if model_name:
            route = self._named_latest(
                request=request,
                family=family,
                model_name=model_name,
            )

            route.validate()

            return route

        # -----------------------------------------------------
        # Configured fallback model
        # -----------------------------------------------------

        if self.config.routing.allow_fallback_model:
            fallback_name = self._fallback_model_name(family)

            if fallback_name:
                fallback_entries = self._entries(
                    family=family,
                    model_name=(fallback_name),
                )

                if fallback_entries:
                    route = self._named_latest(
                        request=request,
                        family=family,
                        model_name=(fallback_name),
                        fallback_used=True,
                    )

                    route.validate()

                    return route

        # -----------------------------------------------------
        # Family-only routing
        # -----------------------------------------------------

        route = self._family_route(
            request=request,
            family=family,
        )

        route.validate()

        return route

    # ---------------------------------------------------------
    # Can route
    # ---------------------------------------------------------

    def can_route(
        self,
        request: InferenceRequest,
    ) -> bool:
        try:
            self.route(request)

        except (
            AIFamilyNotAllowedError,
            AIModelRouteNotFoundError,
            AIVersionNotFoundError,
            AIAmbiguousRouteError,
        ):
            return False

        return True

    # ---------------------------------------------------------
    # Explain family routing
    # ---------------------------------------------------------

    def available_routes(
        self,
        *,
        family: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        entries = self.registry.find(family=family)

        routes: List[Dict[str, Any]] = []

        for entry in entries:
            if not entry_routable(entry):
                continue

            identity = entry.descriptor.identity

            routes.append(
                {
                    "model_id": (identity.model_id),
                    "family": (identity.family),
                    "model_name": (identity.model_name),
                    "version": (identity.version),
                    "state": (entry.descriptor.state),
                    "loaded": (entry.loaded),
                    "ready": (entry.ready),
                }
            )

        return sorted(
            routes,
            key=lambda item: (
                item["family"],
                item["model_name"],
                version_key(item["version"]),
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        routes = self.available_routes()

        by_family: Dict[
            str,
            int,
        ] = {}

        for route in routes:
            family = str(route["family"])

            by_family[family] = (
                by_family.get(
                    family,
                    0,
                )
                + 1
            )

        return {
            "route_count": len(routes),
            "by_family": (by_family),
            "use_latest_version_when_unspecified": (
                self.config.routing.use_latest_version_when_unspecified
            ),
            "allow_fallback_model": (self.config.routing.allow_fallback_model),
            "fallback_event_model": (self.config.routing.fallback_event_model),
            "fallback_anomaly_model": (self.config.routing.fallback_anomaly_model),
            "routing_loads_models": (False),
            "routing_executes_inference": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AIRoute",
    "version_key",
    "entry_routable",
    "latest_entry",
    "AIModelRouter",
]
