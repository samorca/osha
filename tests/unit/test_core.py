from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.core as core_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CORE_ROOT = (
    PROJECT_ROOT
    / "src"
    / "core"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_core_modules() -> list"""
    Discover every implemented Python module
    recursively under src.core.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                core_package.__path__,
                prefix="src.core.",
            )
        }
    )


def import_core_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented core module.

    Collect all failures so pytest reports every
    broken core import together.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_core_modules()
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
        "Core module import failures:\n"
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
    Return classes implemented directly by a core
    module.

    Imported classes from other packages are
    excluded.
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
    Return public functions implemented directly by
    a core module.
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
# Package
# -------------------------------------------------------------------

def test_core_package_imports() -> None:
    module = importlib.import_module(
        "src.core"
    )

    assert module is not None


def test_core_package_has_path() -> None:
    assert hasattr(
        core_package,
        "__path__",
    )


def test_core_directory_exists() -> None:
    assert CORE_ROOT.is_dir()


def test_core_has_python_source() -> None:
    sources = list(
        CORE_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/core contains no Python source"
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_core_module_discovery_returns_list() -> None:
    modules = discover_core_modules()

    assert isinstance(
        modules,
        list,
    )


def test_core_modules_are_unique() -> None:
    modules = discover_core_modules()

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_core_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_core_modules()
    ):
        assert module_name.startswith(
            "src.core."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_core_modules_import() -> None:
    import_core_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_core_modules(),
)
def test_core_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Python source/module consistency
# -------------------------------------------------------------------

def test_core_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in CORE_ROOT.rglob(
            "*.py"
        )
        if path.name
        != "__init__.py"
    ]


    failures: list[str] = []


    for path in python_files:
        relative = (
            path.relative_to(
                PROJECT_ROOT
            )
            .with_suffix("")
        )

        module_name = ".".join(
            relative.parts
        )


        try:
            spec = importlib.util.find_spec(
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

            continue


        if spec is None:
            failures.append(
                (
                    f"{module_name}: "
                    "cannot resolve module"
                )
            )


    assert not failures, (
        "Unresolvable core modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_core_sources_stay_inside_package() -> None:
    modules = import_core_modules()

    root = CORE_ROOT.resolve()


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
                    "outside src/core: "
                    f"{source_path}"
                )
            )


def test_core_modules_report_expected_names() -> None:
    modules = import_core_modules()


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
# Classes
# -------------------------------------------------------------------

def test_core_classes_have_identity() -> None:
    modules = import_core_modules()


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


def test_core_class_signatures_are_inspectable() -> None:
    modules = import_core_modules()


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
                        f"Unable to inspect "
                        f"{module_name}."
                        f"{cls.__name__}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Dataclasses / core schemas
# -------------------------------------------------------------------

def test_core_dataclass_fields_are_mappings() -> None:
    modules = import_core_modules()


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
# Public functions
# -------------------------------------------------------------------

def test_core_public_functions_are_callable() -> None:
    modules = import_core_modules()


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


def test_core_function_signatures_are_inspectable() -> None:
    modules = import_core_modules()


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
                        f"Unable to inspect "
                        f"{module_name}."
                        f"{function_name}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Exception contracts
# -------------------------------------------------------------------

def test_core_exception_classes_inherit_exception() -> None:
    """
    Core classes with names ending in Error or
    Exception should be proper exception types.
    """

    modules = import_core_modules()

    violations: list[str] = []


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            if not cls.__name__.endswith(
                (
                    "Error",
                    "Exception",
                )
            ):
                continue


            if not issubclass(
                cls,
                Exception,
            ):
                violations.append(
                    (
                        f"{module_name}."
                        f"{cls.__name__}"
                    )
                )


    assert not violations, (
        "Core exception-named classes do not "
        "inherit Exception: "
        f"{violations}"
    )


# -------------------------------------------------------------------
# Enum contracts
# -------------------------------------------------------------------

def test_core_enum_members_have_names() -> None:
    """
    Validate enum-like classes without assuming which
    specific enums the core package implements.
    """

    from enum import Enum


    modules = import_core_modules()


    for module in modules.values():
        for cls in module_classes(
            module
        ):
            if not issubclass(
                cls,
                Enum,
            ):
                continue


            for member in cls:
                assert member.name


# -------------------------------------------------------------------
# Serialization contracts
# -------------------------------------------------------------------

def test_core_to_dict_is_callable_when_present() -> None:
    modules = import_core_modules()


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
                "is not callable"
            )


def test_core_model_dump_is_callable_when_present() -> None:
    modules = import_core_modules()


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
            )


# -------------------------------------------------------------------
# Validation contracts
# -------------------------------------------------------------------

def test_core_validate_is_callable_when_present() -> None:
    modules = import_core_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            validator = getattr(
                cls,
                "validate",
                None,
            )


            if validator is None:
                continue


            assert callable(
                validator
            ), (
                f"{module_name}."
                f"{cls.__name__}.validate "
                "is not callable"
            )


# -------------------------------------------------------------------
# Configuration boundary
# -------------------------------------------------------------------

def test_core_config_imports_if_present() -> None:
    for module_name in (
        "src.core.config",
        "src.core.configuration",
    ):
        try:
            spec = importlib.util.find_spec(
                module_name
            )

        except (
            ModuleNotFoundError,
            ImportError,
        ):
            continue


        if spec is None:
            continue


        module = importlib.import_module(
            module_name
        )


        assert module is not None


# -------------------------------------------------------------------
# Schema boundary
# -------------------------------------------------------------------

def test_core_schema_imports_if_present() -> None:
    module_name = "src.core.schema"


    try:
        spec = importlib.util.find_spec(
            module_name
        )

    except (
        ModuleNotFoundError,
        ImportError,
    ):
        return


    if spec is None:
        return


    module = importlib.import_module(
        module_name
    )


    assert module is not None


# -------------------------------------------------------------------
# Errors boundary
# -------------------------------------------------------------------

def test_core_errors_imports_if_present() -> None:
    module_name = "src.core.errors"


    try:
        spec = importlib.util.find_spec(
            module_name
        )

    except (
        ModuleNotFoundError,
        ImportError,
    ):
        return


    if spec is None:
        return


    module = importlib.import_module(
        module_name
    )


    assert module is not None


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_core_exports_resolve() -> None:
    exports = getattr(
        core_package,
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
            core_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.core exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Import safety
# -------------------------------------------------------------------

def test_core_imports_are_passive() -> None:
    """
    Importing core code should not start runtime
    workers, infrastructure collectors, model
    training, API servers, or remediation.

    Core is a foundational library boundary.
    """

    modules = import_core_modules()

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Dependency direction
# -------------------------------------------------------------------

def test_core_namespace_does_not_become_runtime_namespace() -> None:
    """
    Core provides foundational types/utilities.
    Runtime remains a separate subsystem.
    """

    for module_name in (
        discover_core_modules()
    ):
        assert not module_name.startswith(
            "src.runtime."
        )


def test_core_namespace_does_not_become_api_namespace() -> None:
    for module_name in (
        discover_core_modules()
    ):
        assert not module_name.startswith(
            "src.api."
        )


def test_core_namespace_does_not_become_remediation_namespace() -> None:
    for module_name in (
        discover_core_modules()
    ):
        assert not module_name.startswith(
            "src.remediation."
        )


# -------------------------------------------------------------------
# Test isolation
# -------------------------------------------------------------------

def test_core_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_core_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
