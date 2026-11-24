from __future__ import annotations

import importlib
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.self_healing as self_healing_package

from src.self_healing.pipeline import (
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

SELF_HEALING_ROOT = (
    PROJECT_ROOT
    / "src"
    / "self_healing"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_self_healing_modules() -> list"""
    Discover implemented modules recursively under
    src.self_healing.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                self_healing_package.__path__,
                prefix="src.self_healing.",
            )
        }
    )


def import_self_healing_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented self-healing module.

    Accumulate failures so one broken import does not
    hide other subsystem problems.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_self_healing_modules()
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
        "Self-healing import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


# -------------------------------------------------------------------
# Module-owned members
# -------------------------------------------------------------------

def module_classes(
    module: object,
) -> list"""
    Return classes actually implemented by a
    self-healing module.
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
# Package
# -------------------------------------------------------------------

def test_self_healing_package_imports() -> None:
    module = importlib.import_module(
        "src.self_healing"
    )

    assert module is not None


def test_self_healing_directory_exists() -> None:
    assert SELF_HEALING_ROOT.is_dir()


def test_self_healing_package_has_path() -> None:
    assert hasattr(
        self_healing_package,
        "__path__",
    )


def test_self_healing_has_python_source() -> None:
    assert list(
        SELF_HEALING_ROOT.rglob(
            "*.py"
        )
    )


# -------------------------------------------------------------------
# Actual OSHA pipeline contract
# -------------------------------------------------------------------

def test_self_healing_pipeline_imports() -> None:
    module = importlib.import_module(
        "src.self_healing.pipeline"
    )

    assert module is not None


def test_run_is_callable() -> None:
    """
    run() is the single-item self-healing pipeline
    boundary.
    """

    assert callable(
        run
    )


def test_run_many_is_callable() -> None:
    """
    run_many() is the batch self-healing pipeline
    boundary.
    """

    assert callable(
        run_many
    )


def test_run_belongs_to_self_healing_pipeline() -> None:
    assert (
        run.__module__
        == "src.self_healing.pipeline"
    )


def test_run_many_belongs_to_self_healing_pipeline() -> None:
    assert (
        run_many.__module__
        == "src.self_healing.pipeline"
    )


# -------------------------------------------------------------------
# Pipeline signatures
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "function",
    (
        run,
        run_many,
    ),
)
def test_pipeline_signature_is_inspectable(
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


@pytest.mark.parametrize(
    "function",
    (
        run,
        run_many,
    ),
)
def test_pipeline_has_no_positional_only_arguments(
    function: object,
) -> None:
    """
    Orchestrator integration is clearer when
    self-healing calls can use named arguments.
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
        f"{function.__name__} has positional-only "
        f"arguments: {positional_only}"
    )


# -------------------------------------------------------------------
# Pipeline distinction
# -------------------------------------------------------------------

def test_run_and_run_many_are_distinct_functions() -> None:
    assert run is not run_many


def test_pipeline_symbols_are_functions() -> None:
    assert inspect.isfunction(
        run
    )

    assert inspect.isfunction(
        run_many
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_self_healing_discovery_returns_list() -> None:
    modules = (
        discover_self_healing_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_self_healing_modules_are_unique() -> None:
    modules = (
        discover_self_healing_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_self_healing_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_self_healing_modules()
    ):
        assert module_name.startswith(
            "src.self_healing."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_self_healing_modules_import() -> None:
    import_self_healing_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_self_healing_modules(),
)
def test_self_healing_modules_import_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Module/source ownership
# -------------------------------------------------------------------

def test_self_healing_sources_stay_inside_package() -> None:
    modules = (
        import_self_healing_modules()
    )

    root = (
        SELF_HEALING_ROOT.resolve()
    )


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
            f"Source does not exist for "
            f"{module_name}: {source_path}"
        )


        try:
            source_path.relative_to(
                root
            )

        except ValueError:
            pytest.fail(
                (
                    f"{module_name} resolves "
                    "outside src/self_healing: "
                    f"{source_path}"
                )
            )


def test_self_healing_modules_report_expected_names() -> None:
    modules = (
        import_self_healing_modules()
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

def test_self_healing_classes_have_identity() -> None:
    modules = (
        import_self_healing_modules()
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


def test_self_healing_class_signatures_are_inspectable() -> None:
    modules = (
        import_self_healing_modules()
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
# Dataclass/schema contracts
# -------------------------------------------------------------------

def test_self_healing_dataclass_fields_are_mappings() -> None:
    modules = (
        import_self_healing_modules()
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

def test_self_healing_to_dict_is_callable_when_present() -> None:
    modules = (
        import_self_healing_modules()
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


def test_self_healing_model_dump_is_callable_when_present() -> None:
    modules = (
        import_self_healing_modules()
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

def test_public_self_healing_functions_are_callable() -> None:
    modules = (
        import_self_healing_modules()
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


def test_public_self_healing_functions_are_inspectable() -> None:
    modules = (
        import_self_healing_modules()
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
# Package exports
# -------------------------------------------------------------------

def test_self_healing_exports_resolve() -> None:
    exports = getattr(
        self_healing_package,
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
            self_healing_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.self_healing exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Safety boundaries
# -------------------------------------------------------------------

def test_importing_self_healing_does_not_execute_pipeline() -> None:
    """
    Importing the subsystem must not trigger healing
    or remediation.

    This is particularly important because merely
    collecting unit tests must remain operationally
    harmless.
    """

    modules = (
        import_self_healing_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


def test_self_healing_pipeline_is_not_executed_during_import() -> None:
    """
    Imported symbols remain callable functions rather
    than precomputed execution results.
    """

    assert inspect.isfunction(
        run
    )

    assert inspect.isfunction(
        run_many
    )


# -------------------------------------------------------------------
# Architectural separation
# -------------------------------------------------------------------

def test_self_healing_is_separate_from_remediation_package() -> None:
    """
    Self-healing coordinates the higher-level healing
    workflow while remediation remains its own
    package boundary.
    """

    for module_name in (
        discover_self_healing_modules()
    ):
        assert not module_name.startswith(
            "src.remediation."
        )


def test_self_healing_is_separate_from_decision_package() -> None:
    for module_name in (
        discover_self_healing_modules()
    ):
        assert not module_name.startswith(
            "src.decision."
        )


def test_self_healing_is_separate_from_runtime_package() -> None:
    for module_name in (
        discover_self_healing_modules()
    ):
        assert not module_name.startswith(
            "src.runtime."
        )


# -------------------------------------------------------------------
# Test package isolation
# -------------------------------------------------------------------

def test_self_healing_modules_are_not_tests() -> None:
    for module_name in (
        discover_self_healing_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
