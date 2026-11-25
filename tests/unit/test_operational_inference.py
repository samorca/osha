import asyncio

import pytest

from src.ai.schema import (
    AIModelIdentity,
    InferenceProvenance,
    InferenceResult,
    Prediction,
)
from src.api.config import APIConfig
from src.api.services.inference import (
    InferenceService,
    InferenceOperationContext,
    anomaly_domain_request,
    event_domain_request,
    operational_domain_request,
    provenance_response,
)
from src.api.schema import (
    AnomalyInferenceRequest,
    EventInferenceRequest,
    OperationalInferenceRequest,
)


@pytest.mark.parametrize(
    "family",
    [
        "incident_root_cause",
        "decision_policy_risk",
        "remediation_recommendation",
    ],
)
def test_operational_request_translates_to_domain_request(family):
    request = OperationalInferenceRequest.model_validate(
        {
            "family": family,
            "input": {"confidence": 0.8, "severity": "high"},
            "top_k": 2,
            "provenance": {
                "source": "test",
                "source_id": "record-1",
            },
        }
    )

    domain_request = operational_domain_request(
        request,
        request_id="request-1",
        context=InferenceOperationContext(
            tenant_id="tenant-1",
            correlation_id="correlation-1",
        ),
    )

    assert domain_request.family == family
    assert domain_request.payload["input"] == request.input
    assert domain_request.options == {"top_k": 2}
    assert isinstance(domain_request.provenance, InferenceProvenance)
    assert domain_request.provenance.source_ids == ["record-1"]
    assert domain_request.provenance.attributes["tenant_id"] == "tenant-1"
    assert domain_request.provenance.correlation_id == "correlation-1"

    response_provenance = provenance_response(domain_request.provenance)
    assert response_provenance.source == "test"
    assert response_provenance.source_id == "record-1"


@pytest.mark.parametrize(
    "payload",
    [
        {"family": "event", "input": {"x": 1}},
        {"family": "decision_policy_risk", "input": {}},
        {"family": "remediation_recommendation", "input": {"x": 1}, "top_k": 9},
    ],
)
def test_operational_request_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        OperationalInferenceRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("api_request", "translate"),
    [
        (EventInferenceRequest(content="service restarted"), event_domain_request),
        (AnomalyInferenceRequest(features=[0.1, 0.2]), anomaly_domain_request),
    ],
)
def test_existing_inference_requests_use_domain_provenance(api_request, translate):
    domain_request = translate(
        api_request,
        request_id="request-1",
    )

    assert isinstance(domain_request.provenance, InferenceProvenance)
    domain_request.finalize()
    assert provenance_response(InferenceProvenance()) is None


def test_remediation_inference_response_is_advisory_only():
    class StubPipeline:
        def infer(self, request):
            return InferenceResult(
                request_id=request.request_id,
                model=AIModelIdentity(
                    model_name="remediation-ranker",
                    family=request.family,
                    version="v1",
                ),
                state="completed",
                predictions=[
                    Prediction(
                        label="recommendation_score",
                        score=0.5,
                        attributes={
                            "ranking_only": True,
                            "authorizes_action": False,
                        },
                    )
                ],
                provenance=request.provenance,
            )

    service = InferenceService(
        config=APIConfig(),
        pipeline=StubPipeline(),
    )
    request = OperationalInferenceRequest.model_validate(
        {
            "family": "remediation_recommendation",
            "input": {"confidence": 0.8},
        }
    )

    response = asyncio.run(service.infer_operational(request))

    assert response.attributes["advisory_only"] is True
    assert response.predictions[0].attributes["authorizes_action"] is False
