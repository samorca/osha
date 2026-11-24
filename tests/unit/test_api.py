from __future__ import annotations

import importlib
import pkgutil

from collections import Counter

from typing import (
    Any,
    Iterable,
)

import pytest

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.api.app import app


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def api_routes() -> list[APIRoute]:
    return [
        route
        for route in app.routes
        if isinstance(
            route,
            APIRoute,
        )
    ]


def routee_map() -> dict[
    str,
    set[str],
\]:
    result: dict[
       tr,
        set[str],
    ] = {}

    for route in api_routes():
        result.setdefault(
            route.path,
            set(),
        ).update(
            route.methods
            or set()
        )

    return result


def openapi_paths() -> dict[
    str,
    Any,
]:
    schema = app.openapi()

    paths = schema.get(
        "paths",
        {}
    )

    assert isinstance(
        paths,
        dict,
    )

    return paths


def find_route(
    path: str,
) -> APIRoute:
    matches = [
        route
        for route in api_routes()
        if route.path == path
    ]

    assert matches, (
        f"API route is not registered: {path}"
    )

    return matches[0]


# -------------------------------------------------------------------
# Package
# -------------------------------------------------------------------


def test_api_package_imports() -> None:
    module = importlib.import_module(
        "src.api"
    )

    assert module is not None


def test_application_imports() -> None:
    module = importlib.import_module(
        "src.api.app"
    )

    assert module is not None

    assert getattr(
        module,
        "app",
        None,
    ) is app


def test_application_is_fastapi() -> None:
    assert isinstance(
        app,
        FastAPI,
    )


# -------------------------------------------------------------------
# Major API package boundaries
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    (
        "src.api.config",
        "src.api.dependencies",
        "src.api.errors",
        "src.api.middleware",
        "src.api.schema",
    ),
)
def test_api_core_modules_import(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


def test_api_routes_package_imports() -> None:
    module = importlib.import_module(
        "src.api.routes"
    )

    assert module is not None


def test_api_services_package_imports() -> None:
    module = importlib.import_module(
        "src.api.services"
    )

    assert module is not None


# -------------------------------------------------------------------
# Import every implemented API submodule
# -------------------------------------------------------------------


def test_all_api_modules_import() -> None:
    import src.api as api_package

    failures: list[
        tuple[
            str,
            str,
        ]
    ] = []


    for module_info in pkgutil.walk_packages(
        api_package.__path__,
        prefix=(
            api_package.__name__
            + "."
        ),
    ):
        name = module_info.name

        try:
            importlib.import_module(
                name
            )

        except Exception as exc:
            failures.append(
                (
                    name,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )


    assert not failures, (
        "API submodule import failures: "
        + "; ".join(
            (
                f"{name} -> {error}"
            )
            for name, error in failures
        )
    )


# -------------------------------------------------------------------
# Application metadata
# -------------------------------------------------------------------


def test_application_has_title() -> None:
    assert isinstance(
        app.title,
        str,
    )

    assert app.title.strip()


def test_application_has_version_string() -> None:
    assert isinstance(
        app.version,
        str,
    )


def test_application_has_registered_routes() -> None:
    assert api_routes()


# -------------------------------------------------------------------
# OpenAPI
# -------------------------------------------------------------------


def test_openapi_schema_is_mapping() -> None:
    schema = app.openapi()

    assert isinstance(
        schema,
        dict,
    )


def test_openapi_schema_declares_version() -> None:
    schema = app.openapi()

    assert "openapi" in schema

    assert isinstance(
        schema["openapi"],
        str,
    )


def test_openapi_schema_has_info() -> None:
    schema = app.openapi()

    assert "info" in schema

    assert isinstance(
        schema["info"],
        dict,
    )


def test_openapi_schema_has_paths() -> None:
    paths = openapi_paths()

    assert paths


# -------------------------------------------------------------------
# Health route contracts
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    (
        "/health",
        "/health/live",
        "/health/ready",
    ),
)
def test_health_routes_exist(
    path: str,
) -> None:
    routes = route_map()

    assert path in routes


@pytest.mark.parametrize(
    "path",
    (
        "/health",
        "/health/live",
        "/health/ready",
    ),
)
def test_health_routes_are_get(
    path: str,
) -> None:
    routes = route_map()

    assert "GET" in routes[
        path
    ]


# -------------------------------------------------------------------
# Accelerator API
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/accelerators",
        (
            "/api/v1/accelerators/"
            "{device_id}"
        ),
        (
            "/api/v1/accelerators/"
            "{device_id}/metrics"
        ),
    ),
)
def test_accelerator_read_routes_exist(
    path: str,
) -> None:
    assert path in route_map()


def test_accelerator_inventory_is_get() -> None:
    assert (
        "GET"
        in route_map()[
            "/api/v1/accelerators"
        ]
    )


def test_accelerator_selection_is_not_get() -> None:
    path = (
        "/api/v1/accelerators/select"
    )

    assert path in route_map()

    assert (
        "GET"
        not in route_map()[
            path
        ]
    )


def test_accelerator_optimization_is_not_get() -> None:
    path = (
        "/api/v1/accelerators/"
        "optimize"
    )

    assert path in route_map()

    assert (
        "GET"
        not in route_map()[
            path
        ]
    )


# -------------------------------------------------------------------
# Model API
# -------------------------------------------------------------------


def test_model_list_route_exists() -> None:
    routes = route_map()

    assert (
        "/api/v1/models"
        in routes
    )

    assert (
        "GET"
        in routes[
            "/api/v1/models"
        ]
    )


def test_model_detail_route_exists() -> None:
    path = (
        "/api/v1/models/{model_id}"
    )

    assert path in route_map()

    assert (
        "GET"
        in route_map()[
            path
        ]
    )


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/models/refresh",
        (
            "/api/v1/models/"
            "{model_id}/load"
        ),
        (
            "/api/v1/models/"
            "{model_id}/unload"
        ),
    ),
)
def test_model_mutation_routes_are_not_get(
    path: str,
) -> None:
    routes = route_map()

    assert path in routes

    assert "GET" not in routes[
        path
    ]


# -------------------------------------------------------------------
# Inference API
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/inference/events",
        "/api/v1/inference/anomalies",
        "/api/v1/inference/batch",
    ),
)
def test_inference_routes_exist(
    path: str,
) -> None:
    routes = route_map()

    assert path in routes


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/inference/events",
        "/api/v1/inference/anomalies",
        "/api/v1/inference/batch",
    ),
)
def test_inference_routes_are_not_get(
    path: str,
) -> None:
    routes = route_map()

    assert "GET" not in routes[
        path
    ]


# -------------------------------------------------------------------
# Incident API
# -------------------------------------------------------------------


def test_incident_list_route_exists() -> None:
    path = "/api/v1/incidents"

    assert path in route_map()


def test_incident_detail_route_exists() -> None:
    path = (
        "/api/v1/incidents/"
        "{incident_id}"
    )

    assert path in route_map()


def test_incident_status_route_exists() -> None:
    path = (
        "/api/v1/incidents/"
        "{incident_id}/status"
    )

    assert path in route_map()


# -------------------------------------------------------------------
# Decision API
# -------------------------------------------------------------------


def test_decision_list_route_exists() -> None:
    path = "/api/v1/decisions"

    assert path in route_map()


def test_decision_detail_route_exists() -> None:
    path = (
        "/api/v1/decisions/"
        "{decision_id}"
    )

    assert path in route_map()


def test_decision_evaluate_is_not_get() -> None:
    path = (
        "/api/v1/decisions/evaluate"
    )

    assert path in route_map()

    assert "GET" not in route_map()[
        path
    ]


# -------------------------------------------------------------------
# Remediation API
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/remediation/plans",
        (
            "/api/v1/remediation/plans/"
            "{plan_id}"
        ),
    ),
)
def test_remediation_read_routes_exist(
    path: str,
) -> None:
    assert path in route_map()


@pytest.mark.parametrize(
    "path",
    (
        (
            "/api/v1/remediation/plans/"
            "{plan_id}/approve"
        ),
        (
            "/api/v1/remediation/plans/"
            "{plan_id}/execute"
        ),
    ),
)
def test_remediation_mutation_routes_are_not_get(
    path: str,
) -> None:
    routes = route_map()

    assert path in routes

    assert "GET" not in routes[
        path
    ]


# -------------------------------------------------------------------
# Route uniqueness
# -------------------------------------------------------------------


def test_no_duplicate_method_path_pairs() -> None:
    pairs: list[
        tuple[
            str,
            str,
        ]
    ] = []


    for route in api_routes():
        for method in (
            route.methods
            or set()
        ):
            pairs.append(
                (
                    method,
                    route.path,
                )
            )


    counts = Counter(
        pairs
    )


    duplicates = {
        pair: count
        for pair, count
        in counts.items()
        if count > 1
    }


    assert not duplicates, (
        "Duplicate API method/path pairs: "
        f"{duplicates}"
    )


# -------------------------------------------------------------------
# Route metadata
# -------------------------------------------------------------------


def test_all_api_routes_have_names() -> None:
    unnamed = [
        route.path
        for route in api_routes()
        if not str(
            route.name
            or ""
        ).strip()
    ]

    assert not unnamed, (
        "Unnamed API routes: "
        f"{unnamed}"
    )


def test_all_versioned_routes_are_in_openapi() -> None:
    paths = openapi_paths()

    missing = [
        route.path
        for route in api_routes()
        if (
            route.path.startswith(
                "/api/v1/"
            )
            and route.include_in_schema
            and route.path
            not in paths
        )
    ]

    assert not missing, (
        "Versioned routes missing "
        f"from OpenAPI: {missing}"
    )


# -------------------------------------------------------------------
# HTTP-level tests
# -------------------------------------------------------------------


@pytest.fixture
def client() -> Iterable[
    TestClient
]:
    """
    TestClient exercises the ASGI application
    directly without starting Uvicorn.

    Dependency overrides are cleared before and
    after each test so one API test cannot pollute
    another.
    """

    app.dependency_overrides.clear()

    with TestClient(
        app,
        raise_server_exceptions=True,
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_openapi_http_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    document = response.json()

    assert isinstance(
        document,
        dict,
    )

    assert "openapi" in document
    assert "paths" in document


def test_docs_endpoint_is_not_server_error(
    client: TestClient,
) -> None:
    response = client.get(
        "/docs"
    )

    assert response.status_code < 500


# -------------------------------------------------------------------
# Liveness
# -------------------------------------------------------------------


def test_liveness_http_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/health/live"
    )

    assert response.status_code == 200

    body = response.json()

    assert isinstance(
        body,
        dict,
    )


# -------------------------------------------------------------------
# Readiness
# -------------------------------------------------------------------


def test_readiness_endpoint_does_not_crash(
    client: TestClient,
) -> None:
    """
    Readiness may legitimately be non-200 when
    dependencies are not ready in the unit-test
    environment.

    What this test requires is that the endpoint
    behaves as an HTTP health contract rather than
    raising an unhandled server exception.
    """

    response = client.get(
        "/health/ready"
    )

    assert response.status_code in {
        200,
        503,
    }

    body = response.json()

    assert isinstance(
        body,
        dict,
    )


# -------------------------------------------------------------------
# Full health
# -------------------------------------------------------------------


def test_health_endpoint_does_not_crash(
    client: TestClient,
) -> None:
    response = client.get(
        "/health"
    )

    assert response.status_code < 500

    body = response.json()

    assert isinstance(
        body,
        dict,
    )


# -------------------------------------------------------------------
# Request ID middleware
# -------------------------------------------------------------------


def test_request_id_can_be_sent(
    client: TestClient,
) -> None:
    response = client.get(
        "/health/live",
        headers={
            "X-Request-ID": (
                "pytest-request-id"
            )
        },
    )

    assert response.status_code == 200


# -------------------------------------------------------------------
# Unknown endpoint
# -------------------------------------------------------------------


def test_unknown_endpoint_returns_not_found(
    client: TestClient,
) -> None:
    response = client.get(
        "/__osha_pytest_missing__"
    )

    assert response.status_code == 404


# -------------------------------------------------------------------
# OpenAPI route-method consistency
# -------------------------------------------------------------------


def test_openapi_methods_match_registered_routes() -> None:
    paths = openapi_paths()

    ignored_methods = {
        "HEAD",
        "OPTIONS",
    }


    for route in api_routes():
        if not route.include_in_schema:
            continue

        assert route.path in paths

        openapi_methods = {
            method.upper()
            for method
            in paths[
                route.path
            ].keys()
        }


        registered_methods = {
            method.upper()
            for method
            in (
                route.methods
                or set()
            )
            if method.upper()
            not in ignored_methods
        }


        assert registered_methods.issubset(
            openapi_methods
        ), (
            f"Route {route.path} has "
            f"registered methods "
            f"{registered_methods} "
            "but OpenAPI exposes "
            f"{openapi_methods}"
        )
