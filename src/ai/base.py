from __future__ import annotations

"""
Base model-adapter contracts for OSHA AI serving.

The AI runtime should not call PyTorch event or
anomaly models directly.

Instead:

    InferenceRequest
        -> AIModelAdapter
        -> preprocessing
        -> model execution
        -> structured InferenceResult

Concrete adapters may have very different
preprocessing while exposing one common runtime
contract.
"""

from abc import (
    ABC,
    abstractmethod,
)

from threading import (
    RLock,
)

from typing import (
    Any,
    Dict,
    Optional,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.errors import (
    AIInferenceRequestError,
    AIModelStateError,
    AIModelUnavailableError,
)

from src.ai.schema import (
    AIModelDescriptor,
    AIModelFamily,
    AIModelState,
    InferenceRequest,
    InferenceResult,
    InferenceState,
)


# ---------------------------------------------------------------
# Base AI model adapter
# ---------------------------------------------------------------

class AIModelAdapter(
    ABC
):
    """
    Base interface for a runtime model adapter.

    Concrete implementations own:

        family-specific preprocessing
        framework-specific model invocation
        output decoding
        resource cleanup

    They do not own:

        training
        model promotion
        infrastructure decisions
        remediation
    """

    family: str

    def __init__(
        self,
        descriptor: AIModelDescriptor,
        *,
        config: Optional[
            AIConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or AIConfig()
        )

        self.config.validate()

        self.descriptor = descriptor
        self.descriptor.finalize()

        self._model: Optional[
            Any
        ] = None

        self._lock = RLock()

        if (
            self.descriptor.
            identity.family
            != self.family
        ):
            raise AIModelStateError(
                "Model descriptor family "
                "does not match adapter "
                "family.",
                details={
                    "descriptor_family": (
                        self.descriptor.
                        identity.family
                    ),
                    "adapter_family": (
                        self.family
                    ),
                },
            )

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    @property
    def model_id(
        self,
    ) -> str:
        return (
            self.descriptor.
            identity.model_id
        )

    @property
    def model(
        self,
    ) -> Optional[Any]:
        return self._model

    @property
    def state(
        self,
    ) -> str:
        return self.descriptor.state

    # ---------------------------------------------------------
    # Loaded/ready status
    # ---------------------------------------------------------

    @property
    def loaded(
        self,
    ) -> bool:
        return (
            self._model is not None
            and self.descriptor.state
            == AIModelState.READY.value
        )

    @property
    def ready(
        self,
    ) -> bool:
        return self.loaded

    # ---------------------------------------------------------
    # State mutation
    # ---------------------------------------------------------

    def set_state(
        self,
        state: str,
        *,
        error: Optional[
            BaseException
        ] = None,
    ) -> None:
        if (
            state
            not in {
                item.value
                for item
                in AIModelState
            }
        ):
            raise AIModelStateError(
                "Unknown AI model state.",
                details={
                    "state": state,
                    "model_id": (
                        self.model_id
                    ),
                },
            )

        self.descriptor.state = state

        self.descriptor.error_type = (
            type(error).__name__
            if error is not None
            else None
        )

    # ---------------------------------------------------------
    # Request validation
    # ---------------------------------------------------------

    def validate_request(
        self,
        request: InferenceRequest,
    ) -> None:
        """
        Validate common request requirements.

        Family-specific adapters should call this
        before applying their own input rules.
        """

        request.finalize()

        if (
            request.family
            != self.family
        ):
            raise AIInferenceRequestError(
                "Inference request family "
                "does not match the model "
                "adapter.",
                details={
                    "request_id": (
                        request.request_id
                    ),
                    "request_family": (
                        request.family
                    ),
                    "adapter_family": (
                        self.family
                    ),
                    "model_id": (
                        self.model_id
                    ),
                },
            )

        if (
            request.model_name
            is not None
            and request.model_name
            != self.descriptor.
            identity.model_name
        ):
            raise AIInferenceRequestError(
                "Inference request model "
                "name does not match the "
                "selected model.",
                details={
                    "request_id": (
                        request.request_id
                    ),
                    "model_id": (
                        self.model_id
                    ),
                },
            )

        if (
            request.model_version
            is not None
            and request.model_version
            != self.descriptor.
            identity.version
        ):
            raise AIInferenceRequestError(
                "Inference request model "
                "version does not match "
                "the selected model.",
                details={
                    "request_id": (
                        request.request_id
                    ),
                    "model_id": (
                        self.model_id
                    ),
                },
            )

    # ---------------------------------------------------------
    # Ready guard
    # ---------------------------------------------------------

    def require_ready(
        self,
    ) -> None:
        if not self.ready:
            raise AIModelUnavailableError(
                details={
                    "model_id": (
                        self.model_id
                    ),
                    "state": (
                        self.descriptor.state
                    ),
                },
            )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(
        self,
    ) -> "AIModelAdapter":
        """
        Load the concrete runtime model.

        Idempotent when already ready.
        """

        with self._lock:
            if self.ready:
                return self

            self.set_state(
                AIModelState.LOADING.value
            )

            try:
                model = (
                    self._load_model()
                )

                if model is None:
                    raise AIModelStateError(
                        "Model adapter loader "
                        "returned no runtime "
                        "model.",
                        details={
                            "model_id": (
                                self.model_id
                            ),
                        },
                    )

                self._model = model

                self._after_load()

                self.set_state(
                    AIModelState.READY.value
                )

            except Exception as exc:
                self._model = None

                self.set_state(
                    AIModelState.FAILED.value,
                    error=exc,
                )

                raise

        return self

    # ---------------------------------------------------------
    # Unload
    # ---------------------------------------------------------

    def unload(
        self,
    ) -> None:
        """
        Release runtime model resources.

        Concrete adapters may override
        _release_model() for framework-specific
        cleanup.
        """

        with self._lock:
            model = self._model

            try:
                if model is not None:
                    self._release_model(
                        model
                    )

            finally:
                self._model = None

                self.descriptor.loaded_at = (
                    None
                )

                self.set_state(
                    AIModelState.
                    UNLOADED.value
                )

    # ---------------------------------------------------------
    # Inference entrypoint
    # ---------------------------------------------------------

    def infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """
        Execute one family-specific inference.

        Routing and request timeout enforcement
        belong to higher runtime layers.
        """

        request.finalize()

        self.validate_request(
            request
        )

        self.require_ready()

        result = self._infer(
            request
        )

        if not isinstance(
            result,
            InferenceResult,
        ):
            raise AIModelStateError(
                "AI model adapter did not "
                "return InferenceResult.",
                details={
                    "model_id": (
                        self.model_id
                    ),
                    "request_id": (
                        request.request_id
                    ),
                },
            )

        result.finalize()

        if (
            result.request_id
            != request.request_id
        ):
            raise AIModelStateError(
                "Inference result request "
                "identity does not match "
                "the input request.",
                details={
                    "model_id": (
                        self.model_id
                    ),
                },
            )

        if (
            result.model.model_id
            != self.descriptor.
            identity.model_id
        ):
            raise AIModelStateError(
                "Inference result model "
                "identity does not match "
                "the executing adapter."
            )

        if (
            result.state
            != InferenceState.
            COMPLETED.value
        ):
            raise AIModelStateError(
                "Successful adapter "
                "execution must return a "
                "completed inference "
                "result.",
                details={
                    "request_id": (
                        request.request_id
                    ),
                    "state": (
                        result.state
                    ),
                },
            )

        return result

    # ---------------------------------------------------------
    # Framework lifecycle hooks
    # ---------------------------------------------------------

    @abstractmethod
    def _load_model(
        self,
    ) -> Any:
        """
        Construct and return the framework-specific
        runtime model.
        """

        raise NotImplementedError

    def _after_load(
        self,
    ) -> None:
        """
        Optional post-load hook.

        Implementations can restore vocabularies,
        feature schemas, label maps, thresholds,
        or other runtime assets here.
        """

    def _release_model(
        self,
        model: Any,
    ) -> None:
        """
        Optional framework-specific cleanup.
        """

    # ---------------------------------------------------------
    # Inference implementation
    # ---------------------------------------------------------

    @abstractmethod
    def _infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """
        Execute family-specific inference.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Adapter summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "model_id": (
                self.model_id
            ),

            "family": self.family,

            "model_name": (
                self.descriptor.
                identity.model_name
            ),

            "version": (
                self.descriptor.
                identity.version
            ),

            "state": (
                self.descriptor.state
            ),

            "loaded": self.loaded,

            "ready": self.ready,

            "device": (
                self.descriptor.device
            ),

            "capabilities": list(
                self.descriptor.
                capabilities
            ),

            "training_supported": (
                False
            ),

            "remediation_supported": (
                False
            ),
        }


# ---------------------------------------------------------------
# Event adapter base
# ---------------------------------------------------------------

class EventModelAdapter(
    AIModelAdapter
):
    """
    Base adapter for event-understanding models.
    """

    family = (
        AIModelFamily.EVENT.value
    )

    def validate_request(
        self,
        request: InferenceRequest,
    ) -> None:
        super().validate_request(
            request
        )

        payload = request.payload

        if not isinstance(
            payload,
            dict,
        ):
            raise AIInferenceRequestError(
                "Event inference payload "
                "must be an object."
            )


# ---------------------------------------------------------------
# Anomaly adapter base
# ---------------------------------------------------------------

class AnomalyModelAdapter(
    AIModelAdapter
):
    """
    Base adapter for anomaly-detection models.
    """

    family = (
        AIModelFamily.ANOMALY.value
    )

    def validate_request(
        self,
        request: InferenceRequest,
    ) -> None:
        super().validate_request(
            request
        )

        payload = request.payload

        if not isinstance(
            payload,
            dict,
        ):
            raise AIInferenceRequestError(
                "Anomaly inference "
                "payload must be an "
                "object."
            )


# ---------------------------------------------------------------
# Adapter type check
# ---------------------------------------------------------------

def require_adapter_family(
    adapter: AIModelAdapter,
    family: str,
) -> AIModelAdapter:
    """
    Validate a runtime adapter against an expected
    model family.
    """

    expected = str(
        family
    ).strip().lower()

    if adapter.family != expected:
        raise AIModelStateError(
            "AI model adapter family "
            "does not match the expected "
            "family.",
            details={
                "adapter_family": (
                    adapter.family
                ),
                "expected_family": (
                    expected
                ),
                "model_id": (
                    adapter.model_id
                ),
            },
        )

    return adapter


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AIModelAdapter",

    "EventModelAdapter",
    "AnomalyModelAdapter",

    "require_adapter_family",
]
