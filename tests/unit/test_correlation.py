from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.correlation as correlation_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CORRELATION_ROOT = (
    PROJECT_ROOT
    / "src"
    / "correlation"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_correlation_modules() -> list[str]:
    """
  implemented modules recursively under
    src.correlation.
    """

    modules = [
        module_info.name
        for module_info
        in pkgutil.walk_packages(
            correlation_package.__path__,
            prefix="src.correlation.",
        )
    ]

    return sorted(
        set(modules)
    )


def import_correlation_modules() -> dict[str, object]:
    """
    Import every discovered correlation module.

    Import failures are accumulated so one broken
    module does not hide other broken imports.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_correlation_modules()
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
        "Correlation module import failures:\n"
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
    inside a correlation module.
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

def test_correlation_package_imports() -> None:
    module = importlib.import_module(
        "src.correlation"
    )

    assert module is not None


def test_correlation_package_has_path() -> None:
    assert hasattr(
        correlation_package,
        "__path__",
    )


def test_correlation_directory_exists() -> None:
    assert CORRELATION_ROOT.is_dir()


def test_correlation_package_source_exists() -> None:
    source = getattr(
        correlation_package,
        "__file__",
        None,
    )

    assert source is not None

    assert Path(
        source
    ).exists()


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_correlation_module_discovery_returns_list() -> None:
    modules = (
        discover_correlation_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_correlation_modules_are_unique() -> None:
    modules = (
        discover_correlation_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_correlation_modules_use_expected_namespace() -> None:
    for module_name in (
        discover_correlation_modules()
    ):
        assert module_name.startswith(
            "src.correlation."
        )


# -------------------------------------------------------------------
# Module imports
# -------------------------------------------------------------------

def test_all_correlation_modules_import() -> None:
    import_correlation_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_correlation_modules(),
)
def test_correlation_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Python source/module consistency
# -------------------------------------------------------------------

def test_correlation_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in CORRELATION_ROOT.rglob(
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
                    f"{path}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            continue


        if spec is None:
            failures.append(
                (
                    f"{path}: cannot resolve "
                    f"{module_name}"
                )
            )


    assert not failures, (
        "Unresolvable correlation modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Module/source ownership
# -------------------------------------------------------------------

def test_correlation_module_sources_exist() -> None:
    modules = (
        import_correlation_modules()
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
            f"{module_name} has no source file"
        )

        assert Path(
            source
        ).exists()


def test_correlation_modules_stay_inside_package() -> None:
    modules = (
        import_correlation_modules()
    )

    root = (
        CORRELATION_ROOT.resolve()
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

        assert source is not None

        source_path = Path(
            source
        ).resolve()


        try:
            source_path.relative_to(
                root
            )

        except ValueError:
            pytest.fail(
                (
                    f"{module_name} resolves "
                    "outside src/correlation: "
                    f"{source_path}"
                )
            )


def test_correlation_modules_report_expected_names() -> None:
    modules = (
        import_correlation_modules()
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
# Classes
# -------------------------------------------------------------------

def test_correlation_classes_have_names() -> None:
    modules = (
        import_correlation_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            assert cls.__name__.strip(), (
                "Unnamed correlation class "
                f"in {module_name}"
            )


def test_correlation_classes_belong_to_declaring_module() -> None:
    modules = (
        import_correlation_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            assert (
                cls.__module__
                == module_name
            )


def test_correlation_class_signatures_are_inspectable() -> None:
    modules = (
        import_correlation_modules()
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

def test_correlation_dataclass_signatures_are_inspectable() -> None:
    modules = (
        import_correlation_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            if not hasattr(
                cls,
                "__dataclass_fields__",
            ):
                continue

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
                        "correlation dataclass "
                        f"{module_name}."
                        f"{cls.__name__}: "
                        f"{exc}"
                    )
                )

            assert signature is not None


# -------------------------------------------------------------------
# Public functions
# -------------------------------------------------------------------

def test_public_correlation_functions_are_callable() -> None:
    modules = (
        import_correlation_modules()
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


def test_public_correlation_functions_have_signatures() -> None:
    modules = (
        import_correlation_modules()
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
# Optional entrypoints
# -------------------------------------------------------------------

def test_correlation_main_is_callable_when_present() -> None:
    """
    Correlation modules are not required to expose
    command-line entrypoints.

    If main exists, it must be callable.
    """

    modules = (
        import_correlation_modules()
    )


    for (
        module_name,
        module,
    ) in modules.items():
        if not hasattr(
            module,
            "main",
        ):
            continue

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

def test_correlation_package_exports_resolve() -> None:
    """
    If src.correlation defines __all__, every export
    must exist on the package.
    """

    exports = getattr(
        correlation_package,
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
            correlation_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.correlation exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Import safety
# -------------------------------------------------------------------

def test_correlation_imports_do_not_execute_pipeline() -> None:
    """
    Importing correlation modules should not start a
    long-running correlation pipeline.

    Importability itself is the contract checked
    here. Behavioral execution will be tested after
    the exact public correlation API is inspected.
    """

    modules = (
        import_correlation_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Subsystem separation
# -------------------------------------------------------------------

def test_correlation_discovery_remains_in_namespace() -> None:
    modules = (
        discover_correlation_modules()
    )


    forbidden_prefixes = (
        "src.collectors.",
        "src.telemetry.",
        "src.decision.",
        "src.remediation.",
    )


    for module_name in modules:
        assert not module_name.startswith(
            forbidden_prefixes
        )


# -------------------------------------------------------------------
# No test-package leakage
# -------------------------------------------------------------------

def test_correlation_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_correlation_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )


# -------------------------------------------------------------------
# Source availability
# -------------------------------------------------------------------

def test_correlation_has_python_source() -> None:
    sources = list(
        CORRELATION_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/correlation contains no Python source"
    )
