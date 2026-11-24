from __future__ import annotations

"""
Top-level AI-serving facade for OSHA.

The AIPipeline coordinates:

    promoted-model discovery
    model registration
    lazy/eager model loading
    model routing
    event inference
    anomaly inference
    batch inference
    AI-serving health
    model refresh
    runtime unload/shutdown

Architecture:

    models/
        -> AIModelLoader
        -> AIModelRegistry
        -> AIModelRouter
        -> AIInferenceEngine
        -> InferenceResult
        -> intelligence/

This module does not:

    train models
    create checkpoints
    promote candidate artifacts
    interpret incidents
    authorize remediation
    execute remediation
"""

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.ai.config import (
    AIConfig,
    safe_config_summary,
)

from src.ai.errors import (
    AIConfigurationError,
    AIInferenceRequestError,
)

from src.ai.health import (
    AIHealth,
    AIHealthChecker,
)

from src.ai.inference import (
    AIInferenceEngine,
    InferenceExecution,
)

from src.ai.loader import (
    AIModelDiscoveryResult,
    AIModelLoader,
)

from src.ai.registry import (
    AIModelRegistry,
    AIModelRegistryEntry,
)

from src.ai.routing import (
    AIRoute,
    AIModelRouter,
)

from src.ai.schema import (
    AIModelDescriptor,
    AIModelFamily,
    BatchInferenceRequest,
    BatchInferenceResult,
    InferenceProvenance,
    InferenceRequest,
    InferenceResult,
)


# ---------------------------------------------------------------
# AI pipeline
# ---------------------------------------------------------------


class AIPipeline:
    """
    Public facade for OSHA AI model serving.

    The pipeline composes:

        loader
        registry
        router
        inference engine
        health checker

    Models are lazy-loaded by default according to
    AIConfig.
    """

    def __init__(
        self,
        *,
        config: Optional[AIConfig] = None,
        registry: Optional[AIModelRegistry] = None,
        loader: Optional[AIModelLoader] = None,
        router: Optional[AIModelRouter] = None,
        inference: Optional[AIInferenceEngine] = None,
        health: Optional[AIHealthChecker] = None,
    ) -> None:
        self.config = config or AIConfig()

        self.config.validate()

        self.registry = registry or AIModelRegistry(config=self.config)

        self.loader = loader or AIModelLoader(
            config=self.config,
            registry=self.registry,
        )

        self.router = router or AIModelRouter(
            config=self.config,
            registry=self.registry,
        )

        self.inference = inference or AIInferenceEngine(
            config=self.config,
            registry=self.registry,
            router=self.router,
        )

        self.health = health or AIHealthChecker(
            config=self.config,
            registry=self.registry,
            loader=self.loader,
        )

        self._started = False

    # ---------------------------------------------------------
    # Guards
    # ---------------------------------------------------------

    def _require_enabled(
        self,
    ) -> None:
        if not self.config.enabled:
            raise AIConfigurationError("AI serving is disabled.")

    def _require_family(
        self,
        family: str,
    ) -> str:
        self._require_enabled()

        normalized = str(family).strip().lower()

        if not self.config.family_allowed(normalized):
            raise AIConfigurationError(
                "AI model family is not enabled.",
                details={
                    "family": normalized,
                },
            )

        return normalized

    # ---------------------------------------------------------
    # Runtime directories
    # ---------------------------------------------------------

    def initialize_directories(
        self,
    ) -> Dict[str, str]:
        """
        Create AI-owned runtime directories.

        models/ is intentionally not created here.
        It belongs to the model-promotion
        lifecycle.
        """

        self._require_enabled()

        paths = self.config.paths.resolved()

        try:
            Path(paths["cache"]).mkdir(
                parents=True,
                exist_ok=True,
            )

            Path(paths["logs"]).mkdir(
                parents=True,
                exist_ok=True,
            )

        except OSError as exc:
            raise AIConfigurationError(
                "AI runtime directories could not be initialized.",
                details={
                    "error_type": (type(exc).__name__),
                },
            ) from exc

        return {
            "models": (paths["models"]),
            "cache": (paths["cache"]),
            "logs": (paths["logs"]),
        }

    # ---------------------------------------------------------
    # Model discovery
    # ---------------------------------------------------------

    def discover_models(
        self,
        *,
        fail_fast: bool = False,
    ) -> AIModelDiscoveryResult:
        self._require_enabled()

        return self.loader.discover(fail_fast=fail_fast)

    # ---------------------------------------------------------
    # Discover + register
    # ---------------------------------------------------------

    def register_promoted_models(
        self,
        *,
        fail_fast: bool = False,
        replace: bool = False,
        eager_load: Optional[bool] = None,
    ) -> AIModelDiscoveryResult:
        self._require_enabled()

        (
            result,
            _entries,
        ) = self.loader.discover_and_register(
            fail_fast=fail_fast,
            replace=replace,
            eager_load=eager_load,
        )

        return result

    # ---------------------------------------------------------
    # Explicit descriptor registration
    # ---------------------------------------------------------

    def register_model(
        self,
        descriptor: AIModelDescriptor,
        *,
        replace: bool = False,
        load: bool = False,
    ) -> AIModelRegistryEntry:
        self._require_enabled()

        descriptor.finalize()

        return self.registry.register(
            descriptor,
            replace=replace,
            create_adapter=load,
            load=load,
        )

    # ---------------------------------------------------------
    # Register one promoted artifact
    # ---------------------------------------------------------

    def register_artifact(
        self,
        path: str | Path,
        *,
        replace: bool = False,
        load: bool = False,
    ) -> AIModelRegistryEntry:
        self._require_enabled()

        return self.loader.register_artifact(
            path,
            replace=replace,
            load=load,
        )

    # ---------------------------------------------------------
    # Refresh discovery
    # ---------------------------------------------------------

    def refresh(
        self,
        *,
        fail_fast: bool = False,
        replace: bool = False,
    ) -> AIModelDiscoveryResult:
        """
        Rescan models/ for newly promoted model
        artifacts.

        Existing identities are preserved unless
        replacement is explicitly requested and
        enabled in AIConfig.
        """

        self._require_enabled()

        return self.loader.refresh(
            fail_fast=fail_fast,
            replace=replace,
        )

    # ---------------------------------------------------------
    # Model inventory
    # ---------------------------------------------------------

    def models(
        self,
        *,
        family: Optional[str] = None,
    ) -> List[AIModelDescriptor]:
        self._require_enabled()

        normalized_family = None

        if family is not None:
            normalized_family = self._require_family(family)

        return self.registry.descriptors(family=normalized_family)

    # ---------------------------------------------------------
    # Route inventory
    # ---------------------------------------------------------

    def routes(
        self,
        *,
        family: Optional[str] = None,
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        self._require_enabled()

        normalized_family = None

        if family is not None:
            normalized_family = self._require_family(family)

        return self.router.available_routes(family=(normalized_family))

    # ---------------------------------------------------------
    # Route one request
    # ---------------------------------------------------------

    def route(
        self,
        request: InferenceRequest,
    ) -> AIRoute:
        self._require_enabled()

        request.finalize()

        return self.router.route(request)

    # ---------------------------------------------------------
    # Explicit load
    # ---------------------------------------------------------

    def load_model(
        self,
        model_id: str,
    ):
        self._require_enabled()

        return self.registry.load(model_id)

    # ---------------------------------------------------------
    # Explicit unload
    # ---------------------------------------------------------

    def unload_model(
        self,
        model_id: str,
        *,
        drop_adapter: bool = False,
    ) -> None:
        self._require_enabled()

        self.registry.unload(
            model_id,
            drop_adapter=(drop_adapter),
        )

    # ---------------------------------------------------------
    # Unload all
    # ---------------------------------------------------------

    def unload_all(
        self,
        *,
        drop_adapters: bool = False,
    ) -> List[str]:
        self._require_enabled()

        return self.registry.unload_all(drop_adapters=(drop_adapters))

    # ---------------------------------------------------------
    # Remove model
    # ---------------------------------------------------------

    def remove_model(
        self,
        model_id: str,
        *,
        unload: bool = True,
    ) -> AIModelDescriptor:
        self._require_enabled()

        return self.registry.remove(
            model_id,
            unload=unload,
        )

    # ---------------------------------------------------------
    # Generic strict inference
    # ---------------------------------------------------------

    def infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        self._require_enabled()

        return self.inference.infer(request)

    # ---------------------------------------------------------
    # Detailed inference
    # ---------------------------------------------------------

    def execute(
        self,
        request: InferenceRequest,
    ) -> InferenceExecution:
        self._require_enabled()

        return self.inference.execute(request)

    # ---------------------------------------------------------
    # Safe inference
    # ---------------------------------------------------------

    def infer_safe(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        self._require_enabled()

        return self.inference.infer_safe(request)

    # ---------------------------------------------------------
    # Event request builder
    # ---------------------------------------------------------

    def event_request(
        self,
        *,
        content: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        provenance: Optional[InferenceProvenance] = None,
        options: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        payload: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> InferenceRequest:
        self._require_family(AIModelFamily.EVENT.value)

        text = str(content).strip()

        if not text:
            raise AIInferenceRequestError("Event inference content cannot be empty.")

        body = dict(payload or {})

        body["content"] = text

        return InferenceRequest(
            family=(AIModelFamily.EVENT.value),
            payload=body,
            model_name=(model_name),
            model_version=(model_version),
            provenance=(provenance or InferenceProvenance()),
            options=dict(options or {}),
        ).finalize()

    # ---------------------------------------------------------
    # Event inference
    # ---------------------------------------------------------

    def infer_event(
        self,
        *,
        content: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        provenance: Optional[InferenceProvenance] = None,
        options: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        payload: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        safe: bool = False,
    ) -> InferenceResult:
        """
        Convenience event-understanding inference.
        """

        request = self.event_request(
            content=content,
            model_name=model_name,
            model_version=(model_version),
            provenance=provenance,
            options=options,
            payload=payload,
        )

        if safe:
            return self.infer_safe(request)

        return self.infer(request)

    # ---------------------------------------------------------
    # Anomaly request builder
    # ---------------------------------------------------------

    def anomaly_request(
        self,
        *,
        features: Iterable[float],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        provenance: Optional[InferenceProvenance] = None,
        threshold: Optional[float] = None,
        options: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        payload: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> InferenceRequest:
        self._require_family(AIModelFamily.ANOMALY.value)

        values = list(features)

        if not values:
            raise AIInferenceRequestError("Anomaly feature vector cannot be empty.")

        body = dict(payload or {})

        body["features"] = values

        request_options = dict(options or {})

        if threshold is not None:
            request_options["threshold"] = float(threshold)

        return InferenceRequest(
            family=(AIModelFamily.ANOMALY.value),
            payload=body,
            model_name=(model_name),
            model_version=(model_version),
            provenance=(provenance or InferenceProvenance()),
            options=(request_options),
        ).finalize()

    # ---------------------------------------------------------
    # Anomaly inference
    # ---------------------------------------------------------

    def infer_anomaly(
        self,
        *,
        features: Iterable[float],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        provenance: Optional[InferenceProvenance] = None,
        threshold: Optional[float] = None,
        options: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        payload: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        safe: bool = False,
    ) -> InferenceResult:
        """
        Convenience anomaly inference.
        """

        request = self.anomaly_request(
            features=features,
            model_name=model_name,
            model_version=(model_version),
            provenance=provenance,
            threshold=threshold,
            options=options,
            payload=payload,
        )

        if safe:
            return self.infer_safe(request)

        return self.infer(request)

    # ---------------------------------------------------------
    # Explicit batch
    # ---------------------------------------------------------

    def infer_batch(
        self,
        batch: BatchInferenceRequest,
        *,
        fail_fast: bool = False,
        require_same_family: bool = False,
    ) -> BatchInferenceResult:
        self._require_enabled()

        return self.inference.infer_batch(
            batch,
            fail_fast=fail_fast,
            require_same_family=(require_same_family),
        )

    # ---------------------------------------------------------
    # Convenience batch
    # ---------------------------------------------------------

    def infer_many(
        self,
        requests: Iterable[InferenceRequest],
        *,
        fail_fast: bool = False,
        require_same_family: bool = False,
    ) -> BatchInferenceResult:
        self._require_enabled()

        return self.inference.infer_many(
            requests,
            fail_fast=fail_fast,
            require_same_family=(require_same_family),
        )

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health_check(
        self,
        *,
        deep: bool = False,
    ) -> AIHealth:
        return self.health.check(deep=deep)

    # ---------------------------------------------------------
    # Ready
    # ---------------------------------------------------------

    def ready(
        self,
        *,
        deep: bool = False,
    ) -> bool:
        return self.health.ready(deep=deep)

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
        *,
        discover_models: bool = True,
        fail_fast: bool = False,
        eager_load: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Initialize AI model serving.

        The default startup behavior is:

            initialize cache/log directories
            discover promoted models
            register descriptors
            keep models lazy-loaded

        eager_load may override the configured
        lazy-loading policy.
        """

        if not self.config.enabled:
            self._started = False

            return {
                "enabled": False,
                "started": False,
                "registered_models": 0,
                "loaded_models": 0,
                "inference_executed": (False),
            }

        paths = self.initialize_directories()

        discovery = None

        if discover_models:
            discovery = self.register_promoted_models(
                fail_fast=fail_fast,
                replace=False,
                eager_load=(eager_load),
            )

        self._started = True

        health = self.health_check(deep=False)

        return {
            "enabled": True,
            "started": True,
            "paths": paths,
            "discovery": (discovery.to_dict() if discovery is not None else None),
            "registered_models": len(self.registry),
            "loaded_models": (self.registry.loaded_count()),
            "health": (health.to_dict()),
            "inference_executed": (False),
            "training_executed": (False),
            "artifact_promotion_executed": (False),
            "remediation_executed": (False),
        }

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(
        self,
        *,
        drop_adapters: bool = True,
    ) -> Dict[str, Any]:
        """
        Release currently loaded runtime models.

        Registered descriptors are retained in
        this AIPipeline instance.
        """

        unloaded = self.registry.unload_all(drop_adapters=(drop_adapters))

        self._started = False

        return {
            "stopped": True,
            "unloaded_models": (unloaded),
            "unloaded_model_count": len(unloaded),
            "registered_models": len(self.registry),
            "inference_cancelled": (False),
            "training_executed": (False),
            "remediation_executed": (False),
        }

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "AIPipeline":
        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.stop()

        return False

    # ---------------------------------------------------------
    # Started status
    # ---------------------------------------------------------

    @property
    def started(
        self,
    ) -> bool:
        return self._started

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        *,
        include_health: bool = True,
    ) -> Dict[str, Any]:
        result: Dict[
            str,
            Any,
        ] = {
            "started": (self._started),
            "config": (safe_config_summary(self.config)),
            "loader": (self.loader.summary()),
            "registry": (self.registry.summary()),
            "routing": (self.router.summary()),
            "inference": (self.inference.summary()),
            "public_operations": {
                "event_inference": True,
                "anomaly_inference": True,
                "batch_inference": (self.config.inference.enable_batch_inference),
                "model_discovery": True,
                "model_refresh": True,
                "explicit_model_load": (True),
                "explicit_model_unload": (True),
            },
            "training_supported_here": (False),
            "checkpoint_creation_supported_here": (False),
            "artifact_promotion_supported_here": (False),
            "incident_interpretation_supported_here": (False),
            "decision_authorization_supported_here": (False),
            "remediation_supported_here": (False),
        }

        if include_health:
            result["health"] = self.health.summary(deep=False)

        return result


# ---------------------------------------------------------------
# Convenience construction
# ---------------------------------------------------------------


def create_ai_pipeline(
    *,
    config: Optional[AIConfig] = None,
    start: bool = False,
    fail_fast: bool = False,
    eager_load: Optional[bool] = None,
) -> AIPipeline:
    """
    Construct an AIPipeline and optionally start
    model discovery/registration immediately.
    """

    pipeline = AIPipeline(config=config)

    if start:
        pipeline.start(
            fail_fast=fail_fast,
            eager_load=eager_load,
        )

    return pipeline


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AIPipeline",
    "create_ai_pipeline",
]
