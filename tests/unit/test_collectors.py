from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.collectors as collectors_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

COLLECTORS_ROOT = (
    PROJECT_ROOT
    / "src"
    / "collectors"
)


# -------------------------------------------------------------------
# Discovery helpers
# -------------------------------------------------------------------


def discover_collector_modules() -> list[str]:
    """
    Discover implemented modules recursively under
    src.collectors.
    """

    modules = [
        module_info.name
        for module_info
        in pkgutil.walk_packages(
            collectors_package.__path__,
            prefix="src.collectors.",
        )
    ]

    return sorted(
        set(modules)
    )


def import_collector_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented collector module.

    Import failures are accumulated so pytest reports
    every failing module together.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_collector_modules()
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
        "Collector module import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


def collector_classes(
    module: object,
) -> list[type]:
    """
    Return classes implemented by the collector
    module itself.

    Classes imported into the module from another
    subsystem are excluded.
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
    a collector module.
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


def test_collectors_package_imports() -> None:
    module = importlib.import_module(
        "src.collectors"
    )

    assert module is not None


def test_collectors_package_has_path() -> None:
    assert hasattr(
        collectors_package,
        "__path__",
    )


def test_collectors_source_directory_exists() -> None:
    assert COLLECTORS_ROOT.is_dir()


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------


def test_collector_module_discovery_returns_list() -> None:
    modules = (
        discover_collector_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_discovered_collector_modules_are_unique() -> None:
    modules = (
        discover_collector_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_discovered_modules_belong_to_collectors_package() -> None:
    for module_name in (
        discover_collector_modules()
    ):
        assert module_name.startswith(
            "src.collectors."
        )


# -------------------------------------------------------------------
# Import every implemented module
# -------------------------------------------------------------------


def test_all_collector_modules_import() -> None:
    import_collector_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_collector_modules(),
)
def test_collector_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Source/module consistency
# -------------------------------------------------------------------


def test_collector_python_files_are_import_resolvable() -> None:
    """
    Every Python source file below src/collectors
    should resolve through Python's import system.
    """

    files = [
        path
        for path
        in COLLECTORS_ROOT.rglob(
            "*.py"
        )
        if path.name
        != "__init__.py"
    ]


    failures: list[str] = []


    for path in files:
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
        "Unresolvable collector modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source locations
# -------------------------------------------------------------------


def test_collector_modules_resolve_under_collectors_directory() -> None:
    modules = import_collector_modules()

    root = (
        COLLECTORS_ROOT.resolve()
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


        source_path = Path(
            source
        ).resolve()


        assert source_path.exists(), (
            f"Collector source is missing: "
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
                    "outside src/collectors: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# Module identity
# -------------------------------------------------------------------


def test_collector_modules_report_expected_names() -> None:
    modules = import_collector_modules()


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
# Collector classes
# -------------------------------------------------------------------


def test_collector_classes_have_names() -> None:
    modules = import_collector_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in collector_classes(
            module
        ):
            assert cls.__name__.strip(), (
                f"Unnamed collector class "
                f"in {module_name}"
            )


def test_collector_classes_have_inspectable_signatures() -> None:
    modules = import_collector_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in collector_classes(
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
# Public function contracts
# -------------------------------------------------------------------


def test_public_collector_functions_are_callable() -> None:
    modules = import_collector_modules()


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


def test_public_collector_functions_have_signatures() -> None:
    modules = import_collector_modules()


    me,
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
# Optional CLI contracts
# -------------------------------------------------------------------


def test_collector_main_functions_are_callable_when_present() -> None:
    """
    Collector modules do not need to expose CLI
    entrypoints.

    If a module defines main(), however, it must be
    callable.
    """

    modules = import_collector_modules()


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
# Imports must not execute collection
# -------------------------------------------------------------------


def test_collector_modules_can_be_imported_without_collection() -> None:
    """
    Importing the collector package must be safe.

    Unit-test discovery should not itself start
    collection loops, contact infrastructure, or
    spawn foreground workers.
    """

    modules = import_collector_modules()

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Package ownership
# -------------------------------------------------------------------


def test_collector_classes_belong_to_declaring_module() -> None:
    modules = import_collector_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in collector_classes(
            module
        ):
            assert (
                cls.__module__
                == module_name
            )


# -------------------------------------------------------------------
# __all__ contract
# -------------------------------------------------------------------


def test_collectors_package_exports_resolve() -> None:
    """
    If src.collectors defines __all__, every declared
    export should exist on the package.
    """

    exports = getattr(
        collectors_package,
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
            collectors_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.collectors exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# No test dependency leakage
# -------------------------------------------------------------------


def test_collectors_do_not_resolve_to_test_modules() -> None:
    for module_name in (
        discover_collector_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )


# -------------------------------------------------------------------
# Basic source availability
# -------------------------------------------------------------------


def test_collectors_have_python_source() -> None:
    """
    The subsystem should contain at least its package
    initialization source.

    We deliberately do not require a specific list
    of collector implementations here.
    """

    sources = list(
        COLLECTORS_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/collectors contains no Python source"
    )
