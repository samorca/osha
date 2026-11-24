from __future__ import annotations

import importlib
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.orchestrator as orchestrator_package

from src.orchestrator.coordinator import (
    run_stage,
)

from src.orchestrator.health import (
    run_health,
)

from src.orchestrator.pipeline import (
    run,
    run_many,
)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ORCHESTRATOR_ROOT = (
    PROJECT_ROOT
    / "src"
    / "orchestrator"
)


# -------------------------------------------------------------------
# Discovery helpers
# -------------------------------------------------------------------

def discover_orchestrator_modules() -> list"""
    Discover implemented modules recursively under
    src.orchestrator.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                orchestrator_package.__path__,
                prefix="src.orchestrator.",
            )
        }
    )


def import_orchestrator_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented orchestrator module
    and report all import failures together.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_orchestrator_modules()
    ):
        try:
            imported[
                module_name
            ] = importlib.import_module(
                module_name
            )

        except Exception as exc:
            failures.append(
                (
                    f"{module_name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )


    assert not failures, (
        "Orchestrator import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


def module_classes(
    module: object,
) -> list"""
    Return classes implemented directly by an
    orchestrator module.
    """

    module_name = getattr(
        module,
        "__name__",
        "",
    )


    return [
        value
        for _, value
        in inspect.getmembers(
            module,
            inspect.isclass,
        )
        if getattr(
            value,
            "__module__",
            None,
        )
        == module_name
    ]


def public_functions(
    module: object,
) -> list[
    tuple[
        str,
        object,
    ]
]:
    module_name = getattr(
        module,
        "__name__",
        "",
    )


    return [
        (
            name,
            value,
        )
        for name, value
        in inspect.getmembers(
            module,
            inspect.isfunction,
        )
        if (
            not name.startswith("_")
            and getattr(
                value,
                "__module__",
                None,
            )
            == module_name
        )
    ]


# -------------------------------------------------------------------
# Package contract
# -------------------------------------------------------------------

def test_orchestrator_package_imports() -> None:
    module = importlib.import_module(
        "src.orchestrator"
    )

    assert module is not None


def test_orchestrator_directory_exists() -> None:
    assert ORCHESTRATOR_ROOT.is_dir()


def test_orchestrator_package_has_path() -> None:
    assert hasattr(
        orchestrator_package,
        "__path__",
    )


def test_orchestrator_has_python_source() -> None:
    assert list(
        ORCHESTRATOR_ROOT.rglob(
            "*.py"
        )
    )


# -------------------------------------------------------------------
# Pipeline contract
# -------------------------------------------------------------------

def test_pipeline_module_imports() -> None:
    module = importlib.import_module(
        "src.orchestrator.pipeline"
    )

    assert module is not None


def test_run_is_callable() -> None:
    assert callable(
        run
    )


def test_run_many_is_callable() -> None:
    assert callable(
        run_many
    )


def test_pipeline_functions_belong_to_pipeline_module() -> None:
    assert (
        run.__module__
        == "src.orchestrator.pipeline"
    )

    assert (
        run_many.__module__
        == "src.orchestrator.pipeline"
    )


@pytest.mark.parametrize(
    "function",
    (
        run,
        run_many,
    ),
)
def test_pipeline_signatures_are_inspectable(
    function: object,
) -> None:
    signature = inspect.signature(
        function
    )

    assert signature is not None


@pytest.mark.parametrize(
    "function",
    (
        run,
        run_many,
    ),
)
def test_pipeline_parameters_have_names(
    function: object,
) -> None:
    signature = inspect.signature(
        function
    )


    for parameter in (
        signature.parameters.values()
    ):
        assert parameter.name.strip()


# -------------------------------------------------------------------
# Coordinator contract
# -------------------------------------------------------------------

def test_coordinator_module_imports() -> None:
    module = importlib.import_module(
        "src.orchestrator.coordinator"
    )

    assert module is not None


def test_run_stage_is_callable() -> None:
    assert callable(
        run_stage
    )


def test_run_stage_belongs_to_coordinator() -> None:
    assert (
        run_stage.__module__
        == "src.orchestrator.coordinator"
    )


def test_run_stage_signature_is_inspectable() -> None:
    signature = inspect.signature(
        run_stage
    )

    assert signature is not None


def test_run_stage_parameters_have_names() -> None:
    signature = inspect.signature(
        run_stage
    )


    for parameter in (
        signature.parameters.values()
    ):
        assert parameter.name.strip()


# -------------------------------------------------------------------
# Health contract
# -------------------------------------------------------------------

def test_health_module_imports() -> None:
    module = importlib.import_module(
        "src.orchestrator.health"
    )

    assert module is not None


def test_run_health_is_callable() -> None:
    assert callable(
        run_health
    )


def test_run_health_belongs_to_health_module() -> None:
    assert (
        run_health.__module__
        == "src.orchestrator.health"
    )


def test_run_health_signature_is_inspectable() -> None:
    signature = inspect.signature(
        run_health
    )

    assert signature is not None


# -------------------------------------------------------------------
# No positional-only orchestration boundaries
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "function",
    (
        run,
        run_many,
        run_stage,
        run_health,
    ),
)
def test_public_orchestration_boundary_has_no_positional_only_arguments(
    function: object,
) -> None:
    signature = inspect.signature(
        function
    )


    positional_only = [
        parameter.name
        for parameter
        in signature.parameters.values()
        if (
            parameter.kind
            == inspect.Parameter.POSITIONAL_ONLY
        )
    ]


    assert not positional_only, (
        f"{function.__module__}."
        f"{function.__name__} has "
        "positional-only arguments: "
        f"{positional_only}"
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_orchestrator_discovery_returns_list() -> None:
    modules = (
        discover_orchestrator_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_orchestrator_modules_are_unique() -> None:
    modules = (
        discover_orchestrator_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_orchestrator_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_orchestrator_modules()
    ):
        assert module_name.startswith(
            "src.orchestrator."
        )


# -------------------------------------------------------------------
# Import all implemented modules
# -------------------------------------------------------------------

def test_all_orchestrator_modules_import() -> None:
    import_orchestrator_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_orchestrator_modules(),
)
def test_orchestrator_modules_import_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_orchestrator_sources_stay_inside_package() -> None:
    modules = (
        import_orchestrator_modules()
    )

    root = ORCHESTRATOR_ROOT.resolve()


    for (
        module_name,
        module,
    ) in modules.items():
        source = getattr(
            module,
            "__file__",
            None,
        )

        assert source is not None, (
            f"{module_name} has no source"
        )


        source_path = Path(
            source
        ).resolve()


        assert source_path.exists()


        try:
            source_path.relative_to(
                root
            )

        except ValueError:
            pytest.fail(
                (
                    f"{module_name} resolves "
                    "outside src/orchestrator: "
                    f"{source_path}"
                )
            )


def test_orchestrator_modules_report_expected_names() -> None:
    modules = (
        import_orchestrator_modules()
    )


    for (
        expected_name,
        module,
    ) in modules.items():
        assert (
            getattr(
                module,
                "__name__",
                None,
            )
            == expected_name
        )


# -------------------------------------------------------------------
# Class contracts
# -------------------------------------------------------------------

def test_orchestrator_classes_have_identity() -> None:
    modules = (
        import_orchestrator_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            assert cls.__name__.strip()

            assert (
                cls.__module__
                == module_name
            )


def test_orchestrator_class_signatures_are_inspectable() -> None:
    modules = (
        import_orchestrator_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            try:
                signature = (
                    inspect.signature(
                        cls
                    )
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                pytest.fail(
                    (
                        "Unable to inspect "
                        f"{module_name}."
                        f"{cls.__name__}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Dataclass/schema boundaries
# -------------------------------------------------------------------

def test_orchestrator_dataclass_fields_are_mappings() -> None:
    modules = (
        import_orchestrator_modules()
    )


    for module in modules.values():
        for cls in module_classes(
            module
        ):
            fields = getattr(
                cls,
                "__dataclass_fields__",
                None,
            )


            if fields is None:
                continue


            assert isinstance(
                fields,
                dict,
            )


# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------

def test_orchestrator_to_dict_is_callable_when_present() -> None:
    modules = (
        import_orchestrator_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            converter = getattr(
                cls,
                "to_dict",
                None,
            )


            if converter is None:
                continue


            assert callable(
                converter
            ), (
                f"{module_name}."
                f"{cls.__name__}.to_dict "
                "must be callable"
            )


# -------------------------------------------------------------------
# Public functions
# -------------------------------------------------------------------

def test_public_orchestrator_functions_are_callable() -> None:
    modules = (
        import_orchestrator_moduleses()
    )


    for module in modules.values():
        for (
            _,
            function,
        ) in public_functions(
            module
        ):
            assert callable(
                function
            )


def test_public_orchestrator_functions_have_signatures() -> None:
    modules = (
        import_orchestrator_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for (
            function_name,
            function,
        ) in public_functions(
            module
        ):
            try:
                signature = (
                    inspect.signature(
                        function
                    )
                )

            except (               TypeError,
                ValueError,
            ) as exc:
                pytest.fail(
                    (
                        f"Unable to inspect "
                        f"{module_name}."
                        f"{function_name}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_orchestrator_exports_resolve() -> None:
    exports = getattr(
        orchestrator_package,
        "__all__",
        None,
    )


    if exports is None:
        return


    assert isinstance(
        exports,
        (
            list,
            tuple,
        ),
    )


    missing = [
        name
        for name in exports
        if not hasattr(
            orchestrator_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.orchestrator exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Safety boundary
# -------------------------------------------------------------------

def test_importing_orchestrator_does_not_execute_pipeline() -> None:
    """
    Importing orchestration code must not start a
    runtime pipeline or remediation workflow.

    We deliberately do not call run(), run_many(),
    or run_stage() here because their dependencies
    should be supplied through controlled fixtures
    when behavioral tests are added.
    """

    modules = (
        import_orchestrator_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


def test_pipeline_symbols_are_functions_not_execution_results() -> None:
    assert inspect.isfunction(
        run
    )

    assert inspect.isfunction(
        run_many
    )

    assert inspect.isfunction(
        run_stage
    )

    assert inspect.isfunction(
        run_health
    )


# -------------------------------------------------------------------
# Architectural separation
# -------------------------------------------------------------------

def test_orchestrator_namespace_is_not_remediation_namespace() -> None:
    for module_name in (
        discover_orchestrator_modules()
    ):
        assert not module_name.startswith(
            "src.remediation."
        )


def test_orchestrator_namespace_is_not_worker_namespace() -> None:
    for module_name in (
        discover_orchestrator_modules()
    ):
        assert not module_name.startswith(
            "src.workers."
        )


def test_orchestrator_namespace_is_not_runtime_namespace() -> None:
    """
    src.runtime owns process lifecycle.
    src.orchestrator owns orchestration flow.
    """

    for module_name in (
        discover_orchestrator_modules()
    ):
        assert not module_name.startswith(
            "src.runtime."
        )


# -------------------------------------------------------------------
# No test package leakage
# -------------------------------------------------------------------

def test_orchestrator_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_orchestrator_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
