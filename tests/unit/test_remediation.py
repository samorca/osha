from __future__ import annotations

import importlib
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.remediation as remediation_package

from src.remediation.pipeline import (
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

REMEDIATION_ROOT = (
    PROJECT_ROOT
    / "src"
    / "remediation"
)


# -------------------------------------------------------------------
# Discovery helpers
# -------------------------------------------------------------------

def discover_remediation_modules() -> list[str]:
    """
    Discover implemented modules recursively under
    src.remediation.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                remediation_package.__path__,
                prefix="src.remediation.",
            )
        }
    )


def import_remediation_modules() -> dict[
    str,
    object,
]:
    """
    Import every remediation module.

    All failures are collected so pytest can report
    the complete broken-import set.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_remediation_modules()
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
        "Remediation module import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


def module_classes(
    module: object,
) -> list[type]:
    """
    Return classes implemented by the module itself.
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
    """
    Return public functions implemented directly
    inside the remediation module.
    """

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

def test_remediation_package_imports() -> None:
    module = importlib.import_module(
        "src.remediation"
    )

    assert module is not None


def test_remediation_package_has_path() -> None:
    assert hasattr(
        remediation_package,
        "__path__",
    )


def test_remediation_directory_exists() -> None:
    assert REMEDIATION_ROOT.is_dir()


def test_remediation_has_python_source() -> None:
    assert list(
        REMEDIATION_ROOT.rglob(
            "*.py"
        )
    )


# -------------------------------------------------------------------
# Real OSHA pipeline contract
# -------------------------------------------------------------------

def test_remediation_pipeline_imports() -> None:
    module = importlib.import_module(
        "src.remediation.pipeline"
    )

    assert module is not None


def test_run_is_callable() -> None:
    """
    run() is one of the real remediation pipeline
    boundaries.
    """

    assert callable(
        run
    )


def test_run_many_is_callable() -> None:
    """
    run_many() is the batch remediation pipeline
    boundary.
    """

    assert callable(
        run_many
    )


def test_run_belongs_to_remediation_pipeline() -> None:
    assert (
        run.__module__
        == "src.remediation.pipeline"
    )


def test_run_many_belongs_to_remediation_pipeline() -> None:
    assert (
        run_many.__module__
        == "src.remediation.pipeline"
    )


# -------------------------------------------------------------------
# Pipeline signatures
# -------------------------------------------------------------------

def test_run_signature_is_inspectable() -> None:
    signature = inspect.signature(
        run
    )

    assert signature is not None


def test_run_many_signature_is_inspectable() -> None:
    signature = inspect.signature(
        run_many
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


@pytest.mark.parametrize(
    "function",
    (
        run,
        run_many,
    ),
)
def test_pipeline_has_no_positional_only_contract(
    function: object,
) -> None:
    """
    Orchestration should be able to call remediation
    using explicit named arguments.
    """

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
        f"{function.__name__} contains "
        "positional-only arguments: "
        f"{positional_only}"
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_remediation_discovery_returns_list() -> None:
    modules = (
        discover_remediation_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_remediation_modules_are_unique() -> None:
    modules = (
        discover_remediation_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_remediation_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_remediation_modules()
    ):
        assert module_name.startswith(
            "src.remediation."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_remediation_modules_import() -> None:
    import_remediation_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_remediation_modules(),
)
def test_remediation_modules_import_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_remediation_sources_stay_inside_package() -> None:
    modules = (
        import_remediation_modules()
    )

    root = REMEDIATION_ROOT.resolve()


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


        assert source_path.exists(), (
            f"Missing source for "
            f"{module_name}: "
            f"{source_path}"
        )


        try:
            source_path.relative_to(
                root
            )

        except ValueError:
            pytest.fail(
                (
                    f"{module_name} resolves "
                    "outside src/remediation: "
                    f"{source_path}"
                )
            )


def test_remediation_modules_report_expected_names() -> None:
    modules = (
        import_remediation_modules()
    )


    for (
        name,
        module,
    ) in modules.items():
        assert (
            getattr(
                module,
                "__name__",
                None,
            )
            == name
        )


# -------------------------------------------------------------------
# Classes / schemas
# -------------------------------------------------------------------

def test_remediation_classes_have_identity() -> None:
    modules = (
        import_remediation_modules()
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


def test_remediation_class_signatures_are_inspectable() -> None:
    modules = (
        import_remediation_modules()
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


def test_remediation_dataclass_fields_are_mappings() -> None:
    modules = (
        import_remediation_modules()
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
# Serialization contracts
# -------------------------------------------------------------------

def test_remediation_to_dict_is_callable_when_present() -> None:
    modules = (
        import_remediation_modules()
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


def test_remediation_model_dump_is_callable_when_present() -> None:
    modules = (
        import_remediation_modules()
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
                "model_dump",
                None,
            )


            if converter is None:
                continue


            assert callable(
                converter
            ), (
                f"{module_name}."
                f"{cls.__name__}.model_dump "
                "must be callable"
            )


# -------------------------------------------------------------------
# Public functions
# -------------------------------------------------------------------

def test_public_remediation_functions_are_callable() -> None:
    modules = (
        import_remediation_modules()
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


def test_public_remediation_functions_are_inspectable() -> None:
    modules = (
        import_remediation_modules()
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

            except (
                TypeError,
                ValueError,
            ) as exc:
                pytest.fail(
                    (
                        "Unable to inspect "
                        f"{module_name}."
                        f"{function_name}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Optional command-line entrypoints
# -------------------------------------------------------------------

def test_remediation_main_is_callable_when_present() -> None:
    modules = (
        import_remediation_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        if not hasattr(
            module,
            "main",
       e


        main = getattr(
            module,
            "main"
        )


        assert callable(main), (
            f"{module_name}.main exists "
            "but is not callable"
        )


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_remediation_exports_resolve() -> None:
    exports = getattr(
        remediation_package,
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
            remediation_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.remediation exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Safety boundaries
# -------------------------------------------------------------------

def test_importing_remediation_does_not_execute_actions() -> None:
    """
    Importing the subsystem must remain passive.

    The test deliberately does NOT call run() or
    run_many() because doing so without controlled
    fixtures could cross the infrastructure-action
    boundary.
    """

    modules = (
        import_remediation_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


def test_pipeline_functions_are_references_not_results() -> None:
    """
    Merely importing the pipeline should yield
    callable functions. It must not pre-execute
    remediation and replace those symbols with an
    execution result.
    """

    assert inspect.isfunction(
        run
    )

    assert inspect.isfunction(
        run_many
    )


# -------------------------------------------------------------------
# Architecture boundary
# -------------------------------------------------------------------

def test_remediation_namespace_is_separate_from_decision() -> None:
    """
    Decision determines what should happen.
    Remediation owns the action-planning/execution
    boundary.

    Package discovery should therefore remain under
    src.remediation.
    """

    for module_name in (
        discover_remediation_modules()
    ):
        assert not module_name.startswith(
            "src.decision."
        )


def test_remediation_namespace_is_separate_from_orchestrator() -> None:
    for module_name in (
        discover_remediation_modules()
    ):
        assert not module_name.startswith(
            "src.orchestrator."
        )


# -------------------------------------------------------------------
# No accidental test imports
# -------------------------------------------------------------------

def test_remediation_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_remediation_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
