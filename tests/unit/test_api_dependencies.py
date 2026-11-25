from src.ai.config import (
    AIConfig,
)
from src.ai.registry import (
    AIModelRegistry,
)
from src.api.config import (
    APIConfig,
)
from src.api.dependencies import (
    create_api_dependencies,
)
from src.api.services.inference import (
    InferenceService,
)


def test_create_api_dependencies_registers_route_services() -> None:
    dependencies = create_api_dependencies(
        config=APIConfig(),
    )

    assert set(dependencies.services) == {
        "accelerators",
        "decisions",
        "incidents",
        "inference",
        "models",
        "remediation",
    }


def test_create_api_dependencies_preserves_shared_empty_model_registry() -> None:
    ai_config = AIConfig()
    registry = AIModelRegistry(config=ai_config)

    dependencies = create_api_dependencies(
        config=APIConfig(),
        ai_config=ai_config,
        model_registry=registry,
    )
    pipeline = dependencies.ai_pipeline

    assert dependencies.model_registry is registry
    assert dependencies.model_loader.registry is registry
    assert pipeline.registry is registry
    assert pipeline.loader is dependencies.model_loader
    assert pipeline.router.registry is registry
    assert pipeline.inference.registry is registry
    assert pipeline.health.registry is registry
    assert pipeline.health.loader is dependencies.model_loader


def test_inference_service_uses_pipeline_health_check() -> None:
    class Pipeline:
        health = object()

        @staticmethod
        def health_check(*, deep: bool = False) -> dict[str, object]:
            return {
                "status": "healthy",
                "ready": True,
                "deep": deep,
            }

    service = InferenceService(
        config=APIConfig(),
        pipeline=Pipeline(),
    )

    assert service.health() == {
        "status": "healthy",
        "ready": True,
        "deep": False,
    }
