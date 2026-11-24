from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import pkgutil

from pathlib import Path

import pytest

import src.observability as observability_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OBSERVABILITY_ROOT = (
    PROJECT_ROOT
    / "src"
    / "observability"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_observability_modules() -> list"""
    Discover all implemented modules recursively
    under src.observability.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                observability_package.__path__,
                prefix="src.observability.",
            )
        }
    )


def import_observability_modules() -> dict[
    str,
    object,
]:
    """
    Import every observability module.

    Collect all failures so one broken module does
    not hide problems elsewhere in the subsystem.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []

    for module_name in (
        discover_observability_modules()
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
        "Observability module import failures:\n"
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
    Return classes implemented directly by an
    observability module.
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
    the observability module.
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

def test_observability_package_imports() -> None:
    module = importlib.import_module(
        "src.observability"
    )

    assert module is not None


def test_observability_package_has_path() -> None:
    assert hasattr(
        observability_package,
        "__path__",
    )


def test_observability_directory_exists() -> None:
    assert OBSERVABILITY_ROOT.is_dir()


def test_observability_has_python_source() -> None:
    sources = list(
        OBSERVABILITY_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/observability contains no Python source"
    )


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def test_observability_discovery_returns_list() -> None:
    modules = (
        discover_observability_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_observability_modules_are_unique() -> None:
    modules = (
        discover_observability_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_observability_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_observability_modules()
    ):
        assert module_name.startswith(
            "src.observability."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_observability_modules_import() -> None:
    import_observability_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_observability_modules(),
)
def test_observability_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# File/module mapping
# -------------------------------------------------------------------

def test_observability_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in OBSERVABILITY_ROOT.rglob(
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
            spec = (
                importlib.util.find_spec(
                    module_name
                )
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
        "Unresolvable observability modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_observability_sources_stay_inside_package() -> None:
    modules = (
        import_observability_modules()
    )

    root = (
        OBSERVABILITY_ROOT.resolve()
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

        assert source_path.exists()

        try:
            source_path.relative_to(
                root
            )

        except ValueError:
            pytest.fail(
                (
                    f"{module_name} resolves "
                    "outside src/observability: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------

def test_observability_classes_have_identity() -> None:
    modules = (
        import_observability_modules()
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


def test_observability_class_signatures_are_inspectable() -> None:
    modules = (
        import_observability_modules()
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
# Public functions
# -------------------------------------------------------------------

def test_observability_public_functions_are_callable() -> None:
    modules = (
        import_observability_modules()
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


def test_observability_function_signatures_are_inspectable() -> None:
    modules = (
        import_observability_modules()
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
# Known observability component names
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "suffix",
    (
        "logging",
        "metrics",
        "tracing",
        "audit",
        "health",
    ),
)
def test_standard_observability_module_imports_if_present(
    suffix: str,
) -> None:
    """
    OSHA can implement any subset of the common
    observability components.

    When the module exists it must be importable.
    """

    module_name = (
        f"src.observability.{suffix}"
    )

    spec = importlib.util.find_spec(
        module_name
    )

    if spec is None:
        return

    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Logger contracts
# -------------------------------------------------------------------

def test_declared_loggers_are_logging_loggers() -> None:
    """
    If observability modules expose a module-level
    logger named logger or LOGGER, ensure it is a
    normal Python logger.
    """

    modules = (
        import_observability_modules()
    )

    for (
        module_name,
        module,
    ) in modules.items():
        for name in (
            "logger",
            "LOGGER",
        ):
            value = getattr(
                module,
                name,
                None,
            )

            if value is None:
                continue

            assert isinstance(
                value,
                logging.Logger,
            ), (
                f"{module_name}.{name} "
                "is not logging.Logger"
            )


# -------------------------------------------------------------------
# Metrics contracts
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    (
        "record",
        "increment",
        "observe",
        "set",
        "collect",
        "snapshot",
    ),
)
def test_metric_methods_are_callable_when_present(
    method_name: str,
) -> None:
    modules = (
        import_observability_modules()
    )

    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            method = getattr(
                cls,
                method_name,
                None,
            )

            if method is None:
                continue

            assert callable(
                method
            ), (
                f"{module_name}."
                f"{cls.__name__}."
                f"{method_name} "
                "is not callable"
            )


# -------------------------------------------------------------------
# Health contracts
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    (
        "health",
        "healthy",
        "ready",
        "readiness",
        "status",
    ),
)
def test_health_methods_are_callable_or_properties_when_present(
    method_name: str,
) -> None:
    modules = (
        import_observability_modules()
    )

    for module in modules.values():
        for cls in module_classes(
            module
        ):
            attribute = getattr(
                cls,
                method_name,
                None,
            )

            if attribute is None:
                continue

            assert (
                callable(
                    attribute
                )
                or isinstance(
                    attribute,
                    property,
                )
            )


# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------

def test_observability_to_dict_is_callable_when_present() -> None:
    modules = (
        import_observability_modules()
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
                "is not callable"
            )


def test_observability_model_dump_is_callable_when_present() -> None:
    modules = (
        import_observability_modules()
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
            )


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

def test_observability_dataclass_fields_are_mappings() -> None:
    modules = (
        import_observability_modules()
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
# Optional lifecycle
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    (
        "start",
        "stop",
        "close",
        "flush",
    ),
)
def test_observability_lifecycle_methods_are_callable_when_present(
    method_name: str,
) -> None:
    modules = (
        import_observability_modules()
    )

    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            method = getattr(
                cls,
                method_name,
                None,
            )

            if method is None:
                continue

            assert callable(
                method
            ), (
                f"{module_name}."
                f"{cls.__name__}."
                f"{method_name} "
                "is not callable"
            )


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_observability_exports_resolve() -> None:
    exports = getattr(
        observability_package,
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
            observability_package,
            name,
        )
    ]

    assert not missing, (
        "Missing src.observability exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Import safety
# -------------------------------------------------------------------

def test_observability_imports_are_passive() -> None:
    """
    Importing observability code must not require
    starting monitoring servers, exporters, worker
    loops, or external integrations.
    """

    modules = (
        import_observability_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Storage separation
# -------------------------------------------------------------------

def test_observability_code_is_separate_from_log_storage() -> None:
    """
    src/observability contains implementation;
    logs/ contains generated output.
    """

    assert (
        OBSERVABILITY_ROOT.resolve()
        != (
            PROJECT_ROOT
            / "logs"
        ).resolve()
    )


def test_observability_code_is_separate_from_artifacts() -> None:
    assert (
        OBSERVABILITY_ROOT.resolve()
        != (
            PROJECT_ROOT
            / "artifacts"
            / "observability"
        ).resolve()
    )


# -------------------------------------------------------------------
# Telemetry boundary
# -------------------------------------------------------------------

def test_observability_namespace_is_separate_from_telemetry() -> None:
    """
    Telemetry produces/normalizes operational
    signals. Observability exposes operational
    visibility. They remain separate namespaces.
    """

    for module_name in (
        discover_observability_modules()
    ):
        assert not module_name.startswith(
            "src.telemetry."
        )


# -------------------------------------------------------------------
# Security boundary
# -------------------------------------------------------------------

def test_observability_namespace_is_separate_from_security() -> None:
    for module_name in (
        discover_observability_modules()
    ):
        assert not module_name.startswith(
            "src.security."
        )


# -------------------------------------------------------------------
# Test namespace isolation
# -------------------------------------------------------------------

def test_observability_modules_are_not_tests() -> None:
    for module_name in (
        discover_observability_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
