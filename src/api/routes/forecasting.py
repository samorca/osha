from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.api.routes import typed_service
from src.api.schema import (
    CapacityForecastRequest,
    CapacityForecastResponse,
)
from src.api.services.forecasting import (
    CAPACITY_FORECAST_SERVICE_NAME,
    CapacityForecastAPIService,
)


router = APIRouter(
    prefix="/forecasting",
    tags=["Forecasting"],
)


def get_capacity_forecasting_service(
    request: Request,
) -> CapacityForecastAPIService:
    return typed_service(
        request,
        CAPACITY_FORECAST_SERVICE_NAME,
    )


CapacityForecastingServiceDependency = Annotated[
    CapacityForecastAPIService,
    Depends(get_capacity_forecasting_service),
]


@router.post(
    "/capacity",
    response_model=CapacityForecastResponse,
    summary="Forecast infrastructure capacity",
    operation_id="forecast_capacity",
)
def forecast_capacity(
    payload: CapacityForecastRequest,
    service: CapacityForecastingServiceDependency,
) -> CapacityForecastResponse:
    return service.forecast(payload)


__all__ = [
    "router",
    "get_capacity_forecasting_service",
    "forecast_capacity",
]
