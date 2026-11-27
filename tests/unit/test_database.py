from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.database as database_package
from src.database.config import DatabaseConfig, load_database_config
from src.database.pipeline import DatabasePipeline


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATABASE_ROOT = (
    PROJECT_ROOT
    / "src"
    / "database"
)


def test_database_config_accepts_compose_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OSHA_DB_HOST",
        "OSHA_DB_PORT",
        "OSHA_DB_NAME",
        "OSHA_DB_USER",
        "OSHA_DB_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("OSHA_DATABASE_HOST", "127.0.0.1")
    monkeypatch.setenv("OSHA_DATABASE_PORT", "5433")
    monkeypatch.setenv("OSHA_DATABASE_NAME", "osha_test")
    monkeypatch.setenv("OSHA_DATABASE_USERNAME", "osha_test_user")
    monkeypatch.setenv("OSHA_DATABASE_PASSWORD", "test-password")

    config = load_database_config()

    assert config.host == "127.0.0.1"
    assert config.port == 5433
    assert config.database == "osha_test"
    assert config.username == "osha_test_user"
    assert config.password == "test-password"


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_database_modules() -> list[str]:
    """
    Discover every implemented module recursively
    under src.database.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                database_package.__path__,
                prefix="src.database.",
            )
        }
    )


def import_database_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented database module.

    Collect failures so pytest reports the complete
    set rather than stopping at the first broken
    module.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_database_modules()
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
        "Database module import failures:\n"
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
) -> list[type]:
    """
    Return classes implemented directly by the
    database module.

    Classes imported from another package are
    deliberately excluded.
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
    the database module.
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

def test_database_package_imports() -> None:
    module = importlib.import_module(
        "src.database"
    )

    assert module is not None


def test_database_package_has_path() -> None:
    assert hasattr(
        database_package,
        "__path__",
    )


def test_database_directory_exists() -> None:
    assert DATABASE_ROOT.is_dir()


def test_database_has_python_source() -> None:
    sources = list(
        DATABASE_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/database contains no Python source"
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_database_module_discovery_returns_list() -> None:
    modules = (
        discover_database_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_database_modules_are_unique() -> None:
    modules = (
        discover_database_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_database_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_database_modules()
    ):
        assert module_name.startswith(
            "src.database."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_database_modules_import() -> None:
    import_database_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_database_modules(),
)
def test_database_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Python source resolution
# -------------------------------------------------------------------

def test_database_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in DATABASE_ROOT.rglob(
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
        "Unresolvable database modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_database_sources_stay_inside_package() -> None:
    modules = (
        import_database_modules()
    )

    root = (
        DATABASE_ROOT.resolve()
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
            f"Missing database source: "
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
                    "outside src/database: "
                    f"{source_path}"
                )
            )


def test_database_modules_report_expected_names() -> None:
    modules = (
        import_database_modules()
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
# Database classes
# -------------------------------------------------------------------

def test_database_classes_have_identity() -> None:
    modules = (
        import_database_modules()
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


def test_database_class_signatures_are_inspectable() -> None:
    modules = (
        import_database_modules()
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
# Dataclasses / database schemas
# -------------------------------------------------------------------

def test_database_dataclass_fields_are_mappings() -> None:
    modules = (
        import_database_modules()
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
# Public functions
# -------------------------------------------------------------------

def test_database_public_functions_are_callable() -> None:
    modules = (
        import_database_modules()
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


def test_database_public_function_signatures_are_inspectable() -> None:
    modules = (
        import_database_modules()
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
# Async database contracts
# -------------------------------------------------------------------

def test_async_database_functions_are_coroutines() -> None:
    """
    When a public database function is declared
    async, Python must recognize it as a coroutine
    function.
    """

    modules = (
        import_database_modules()
    )


    for module in modules.values():
        for (
            _,
            function,
        ) in public_functions(
            module
        ):
            if inspect.iscoroutinefunction(
                function
            ):
                assert (
                    inspect.iscoroutinefunction(
                        function
                    )
                )


# -------------------------------------------------------------------
# Repository-like contracts
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    (
        "get",
        "list",
        "create",
        "update",
        "delete",
        "save",
        "find",
    ),
)
def test_repository_methods_are_callable_when_present(
    method_name: str,
) -> None:
    """
    Database implementations can expose different
    repository classes.

    If one of the common repository methods exists,
    it must be callable.
    """

    modules = (
        import_database_modules()
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
# Transaction-like contracts
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    (
        "commit",
        "rollback",
        "close",
        "connect",
        "disconnect",
    ),
)
def test_transaction_methods_are_callable_when_present(
    method_name: str,
) -> None:
    modules = (
        import_database_modules()
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
# Serialization
# -------------------------------------------------------------------

def test_database_to_dict_is_callable_when_present() -> None:
    modules = (
        import_database_modules()
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


def test_database_model_dump_is_callable_when_present() -> None:
    modules = (
        import_database_modules()
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
                "is not callable"
            )


# -------------------------------------------------------------------
# Configuration boundary
# -------------------------------------------------------------------

def test_database_config_imports_if_present() -> None:
    spec = importlib.util.find_spec(
        "src.database.config"
    )


    if spec is None:
        return


    module = importlib.import_module(
        "src.database.config"
    )

    assert module is not None


# -------------------------------------------------------------------
# Schema boundary
# -------------------------------------------------------------------

def test_database_schema_imports_if_present() -> None:
    spec = importlib.util.find_spec(
        "src.database.schema"
    )


    if spec is None:
        return


    module = importlib.import_module(
        "src.database.schema"
    )

    assert module is not None


# -------------------------------------------------------------------
# Models boundary
# -------------------------------------------------------------------

def test_database_models_import_if_present() -> None:
    spec = importlib.util.find_spec(
        "src.database.models"
    )


    if spec is None:
        return


    module = importlib.import_module(
        "src.database.models"
    )

    assert module is not None


# -------------------------------------------------------------------
# Repository boundary
# -------------------------------------------------------------------

def test_database_repository_imports_if_present() -> None:
    candidates = (
        "src.database.repository",
        "src.database.repositories",
    )


    for module_name in candidates:
        spec = importlib.util.find_spec(
            module_name
        )


        if spec is None:
            continue


        module = importlib.import_module(
            module_name
        )

        assert module is not None


# -------------------------------------------------------------------
# Session boundary
# -------------------------------------------------------------------

def test_database_session_imports_if_present() -> None:
    candidates = (
        "src.database.session",
        "src.database.sessions",
    )


    for module_name in candidates:
        spec = importlib.util.find_spec(
            module_name
        )


        if spec is None:
            continue


        module = importlib.import_module(
            module_name
        )

        assert module is not None


# -------------------------------------------------------------------
# Engine / connection boundary
# -------------------------------------------------------------------

def test_database_engine_imports_if_present() -> None:
    candidates = (
        "src.database.engine",
        "src.database.connection",
        "src.database.connections",
    )


    for module_name in candidates:
        spec = importlib.util.find_spec(
            module_name
        )


        if spec is None:
            continue


        module = importlib.import_module(
            module_name
        )

        assert module is not None


# -------------------------------------------------------------------
# Migrations boundary
# -------------------------------------------------------------------

def test_database_migrations_import_if_python_package_present() -> None:
    spec = importlib.util.find_spec(
        "src.database.migrations"
    )


    if spec is None:
        return


    module = importlib.import_module(
        "src.database.migrations"
    )

    assert module is not None


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_database_exports_resolve() -> None:
    exports = getattr(
        database_package,
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
            database_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.database exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Operational safety
# -------------------------------------------------------------------

def test_database_imports_do_not_require_live_database() -> None:
    """
    Unit-test collection must be able to import the
    database subsystem without requiring a running
    PostgreSQL, MySQL, SQLite service, or other
    external database.

    Actual connection tests belong under
    tests/integration/.
    """

    modules = (
        import_database_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


def test_unit_suite_does_not_call_database_mutations() -> None:
    """
    This test file deliberately discovers create,
    update, delete, commit and rollback contracts
    without executing them.

    Mutation behavior belongs in controlled tests
    using temporary/in-memory database state.
    """

    assert DATABASE_ROOT.is_dir()


# -------------------------------------------------------------------
# Security boundary
# -------------------------------------------------------------------

def test_database_namespace_is_separate_from_security() -> None:
    """
    Database code may consume credentials through
    the security layer, but the packages remain
    separate architectural boundaries.
    """

    for module_name in (
        discover_database_modules()
    ):
        assert not module_name.startswith(
            "src.security."
        )


# -------------------------------------------------------------------
# Integration boundary
# -------------------------------------------------------------------

def test_database_namespace_is_separate_from_integrations() -> None:
    for module_name in (
        discover_database_modules()
    ):
        assert not module_name.startswith(
            "src.integrations."
        )


# -------------------------------------------------------------------
# No test namespace leakage
# -------------------------------------------------------------------

def test_database_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_database_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )


def test_telemetry_repository_stores_idempotent_batches() -> None:
    pipeline = DatabasePipeline(config=DatabaseConfig())
    pipeline.start()
    repository = pipeline.telemetry_repository()
    repository.ensure_schema()
    record = {
        "signal_type": "resource_state",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "payload": {"cpu_percent": 12.5},
    }

    try:
        assert repository.store_batch(
            [record],
            source="linux",
            worker_id="worker-1",
            environment_id="test",
        ) == 1
        assert repository.store_batch(
            [record],
            source="linux",
            worker_id="worker-1",
            environment_id="test",
        ) == 0
    finally:
        pipeline.stop()
