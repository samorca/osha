from __future__ import annotations

import importlib
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.workers as workers_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

WORKERS_ROOT = (
    PROJECT_ROOT
    / "src"
    / "workers"
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def discover_worker_modules() -> list[str]:
    """
    Discover every importable module directly or
    recursively under src.workers.
    """

    modules = [
        module_info.name
        for module_info in pkgutil.walk_packages(
            workers_package.__path__,
            prefix="src.workers.",
        )
    ]

    return sorted(
        set(modules)
    )


def import_worker_modules() -> dict[
    str,
    object,
]:
    """
    Import every discovered worker module.

    Any failure is reported with the exact module
    name rather than being hidden by a generic
    package-import assertion.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_worker_modules()
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
        "Worker module import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


def worker_classes(
    module: object,
) -> list[
    type
]:
    """
    Return classes defined by the module itself.

    Imported classes from other packages are not
    treated as worker implementations belonging to
    this module.
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


# -------------------------------------------------------------------
# Package
# -------------------------------------------------------------------


def test_workers_package_imports() -> None:
    module = importlib.import_module(
        "src.workers"
    )

    assert module is not None


def test_workers_package_has_path() -> None:
    assert hasattr(
        workers_package,
        "__path__",
    )


def test_workers_source_directory_exists() -> None:
    assert WORKERS_ROOT.is_dir()


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------


def test_worker_module_discovery_returns_list() -> None:
    modules = discover_worker_modules()

    assert isinstance(
        modules,
        list,
    )


def test_discovered_worker_modules_belong_to_workers_package() -> None:
    for module_name in (
        discover_worker_modules()
    ):
        assert module_name.startswith(
            "src.workers."
        )


def test_worker_module_names_are_unique() -> None:
    modules = discover_worker_modules()

    assert len(
        modules
    ) == len(
        set(modules)
    )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------


def test_all_worker_modules_import() -> None:
    import_worker_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_worker_modules(),
)
def test_worker_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Source/module consistency
# -------------------------------------------------------------------


def test_python_worker_files_are_discoverable() -> None:
    """
    Every normal .py module under src/workers
    should correspond to the src.workers package.

    __init__.py is package metadata and is excluded.
    """

    python_files = [
        path
        for path in WORKERS_ROOT.rglob(
            "*.py"
        )
        if path.name
        != "__init__.py"
    ]


    for path in python_files:
        relative = path.relative_to(
            PROJECT_ROOT
        ).with_suffix(
            ""
        )

        module_name = ".".join(
            relative.parts
        )

        spec = importlib.util.find_spec(
            module_name
        )

        assert spec is not None, (
            "Worker source file cannot be "
            "resolved as a Python module: "
            f"{path}"
        )


# -------------------------------------------------------------------
# Worker classes
# -------------------------------------------------------------------


def test_worker_classes_have_names() -> None:
    modules = import_worker_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in worker_classes(
            module
        ):
            assert cls.__name__.strip(), (
                "Unnamed class found in "
                f"{module_name}"
            )


def test_worker_classes_belong_to_their_modules() -> None:
    modules = import_worker_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        for cls in worker_classes(
            module
        ):
            assert (
                cls.__module__
                == module_name
            )


# -------------------------------------------------------------------
# main() CLI contracts
# -------------------------------------------------------------------


def test_worker_main_functions_are_callable() -> None:
    """
    main() is optional for library worker modules.

    When a module exposes main, however, it must be
    callable because scripts/runtime/worker.sh uses
    that as its executable-entrypoint contract.
    """

    modules = import_worker_modules()


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


def test_common_worker_entrypoint_if_present() -> None:
    """
    scripts/runtime/worker.sh attempts the conventional
    automatic entrypoints:

        src.workers.main
        src.workers.worker

    They are not required here because your worker
    package may use explicitly named worker modules.

    If present, they must expose main().
    """

    for module_name in (
        "src.workers.main",
        "src.workers.worker",
    ):
        spec = importlib.util.find_spec(
            module_name
        )

        if spec is None:
            continue

        module = importlib.import_module(
            module_name
        )

        main = getattr(
            module,
            "main",
            None,
        )

        assert callable(main), (
            f"{module_name} exists but "
            "does not expose callable main()"
        )


# -------------------------------------------------------------------
# Main signatures
# -------------------------------------------------------------------


def test_worker_main_signatures_are_inspectable() -> None:
    modules = import_worker_modules()


    for (
        module_name,
        module,
    ) in modules.items():
        main = getattr(
            module,
            "main",
            None,
        )

        if not callable(main):
            continue

        try:
            signature = (
                inspect.signature(
                    main
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            pytest.fail(
                (
                    "Unable to inspect "
                    f"{module_name}.main: "
                    f"{exc}"
                )
            )

        assert signature is not None


# -------------------------------------------------------------------
# Avoid accidental runtime-main duplication
# -------------------------------------------------------------------


def test_workers_package_does_not_resolve_runtime_main() -> None:
    """
    The application runtime and individual workers
    are separate process boundaries.

    This protects the architectural correction we
    made to scripts/runtime/worker.sh.
    """

    modules = discover_worker_modules()

    assert (
        "src.runtime.main"
        not in modules
    )


# -------------------------------------------------------------------
# Package-owned module identity
# -------------------------------------------------------------------


def test_imported_worker_modules_report_correct_names() -> None:
    modules = import_worker_modules()


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
# Module source paths
# -------------------------------------------------------------------


def test_worker_modules_have_source_files() -> None:
    modules = import_worker_modules()


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
            f"{module_name} has no source path"
        )

        source_path = Path(
            source
        ).resolve()

        assert source_path.exists(), (
            f"{module_name} source does not exist: "
          _path}"
        )


def test_worker_sources_stay_under_workers_package() -> None:
    modules = import_worker_modules()

    workers_root = (
        WORKERS_ROOT.resolve()
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
                workers_root
            )

        except ValueError:
            pytest.fail(
                (
                    f"{module_name} resolves "
                    "outside src/workers: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# No test-time worker execution
# -------------------------------------------------------------------


def test_worker_imports_do_not_require_cli_execution() -> None:
    """
    Worker modules must remain importable without
    invoking main().

    This test intentionally does not run worker
    loops. Unit tests should not start polling,
    remediation, telemetry, or long-running worker
    processes merely by importing a module.
    """

    modules = import_worker_modules()

    assert isinstance(
        modules,
        dict,
    )
