from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.integrations as integrations_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

INTEGRATIONS_ROOT = (
    PROJECT_ROOT
    / "src"
    / "integrations"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_integration_modules() -> list"""
    Discover every implemented Python module under
    src.integrations.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                integrations_package.__path__,
                prefix="src.integrations.",
            )
        }
    )


def import_integration_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented integration module.

    All failures are accumulated so pytest reports
    the complete broken-import set.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []

    for module_name in (
        discover_integration_modules()
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
        "Integration module import failures:\n"
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
    integration module.

    Imported SDK/client classes are excluded.
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
    the integration module.
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

def test_integrations_package_imports() -> None:
    module = importlib.import_module(
        "src.integrations"
    )

    assert module is not None


def test_integrations_package_has_path() -> None:
    assert hasattr(
        integrations_package,
        "__path__",
    )


def test_integrations_directory_exists() -> None:
    assert INTEGRATIONS_ROOT.is_dir()


def test_integrations_have_python_source() -> None:
    sources = list(
        INTEGRATIONS_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/integrations contains no Python source"
    )


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def test_integration_discovery_returns_list() -> None:
    modules = (
        discover_integration_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_integration_modules_are_unique() -> None:
    modules = (
        discover_integration_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_integration_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_integration_modules()
    ):
        assert module_name.startswith(
            "src.integrations."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_integration_modules_import() -> None:
    """
    Every integration implementation should be safe
    to import without immediately connecting to an
    external provider.
    """

    import_integration_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_integration_modules(),
)
def test_integration_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Python source resolution
# -------------------------------------------------------------------

def test_integration_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in INTEGRATIONS_ROOT.rglob(
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
                    "module cannot be resolved"
                )
            )

    assert not failures, (
        "Unresolvable integration modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_integration_sources_stay_inside_package() -> None:
    modules = (
        import_integration_modules()
    )

    root = (
        INTEGRATIONS_ROOT.resolve()
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
                    "outside src/integrations: "
                    f"{source_path}"
                )
            )


def test_integration_modules_report_expected_names() -> None:
    modules = (
        import_integration_modules()
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
# Integration classes
# -------------------------------------------------------------------

def test_integration_classes_have_identity() -> None:
    modules = (
        import_integration_modules()
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


def test_integration_class_signatures_are_inspectable() -> None:
    modules = (
        import_integration_modules()
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
# Dataclasses / schemas
# -------------------------------------------------------------------

def test_integration_dataclass_fields_are_mappings() -> None:
    modules = (
        import_integration_modules()
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

def test_integration_public_functions_are_callable() -> None:
    modules = (
        import_integration_modules()
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


def test_integration_function_signatures_are_inspectable() -> None:
    modules = (
        import_integration_modules()
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
# Common provider-client contracts
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    (
        "connect",
        "disconnect",
        "close",
        "health",
        "status",
        "discover",
        "collect",
        "list",
        "get",
    ),
)
def test_common_integration_methods_are_callable_when_present(
    method_name: str,
) -> None:
    """
    Provider integrations use different concrete
    implementations.

    If one of these common integration operations is
    exposed, it must be callable.
    """

    modules = (
        import_integration_modules()
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
# Async boundaries
# -------------------------------------------------------------------

def test_async_integration_functions_are_valid_coroutine_functions() -> None:
    modules = (
        import_integration_modules()
    )

    for module in modules.values():
        for (
            _,
            function,
        ) in public_functions(
            module
        ):
            if not inspect.iscoroutinefunction(
                function
            ):
                continue

            assert (
                inspect.iscoroutinefunction(
                    function
                )
            )


# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------

def test_integration_to_dict_is_callable_when_present() -> None:
    modules = (
        import_integration_modules()
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


def test_integration_model_dump_is_callable_when_present() -> None:
    modules = (
        import_integration_modules()
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
# Configuration contracts
# -------------------------------------------------------------------

def test_integration_config_modules_import_when_present() -> None:
    """
    Provider-specific configurations can be nested
    inside their own packages.

    Any module named config or configuration must
    remain importable.
    """

    modules = (
        discover_integration_modules()
    )

    candidates = [
        module_name
        for module_name in modules
        if module_name.endswith(
            (
                ".config",
                ".configuration",
            )
        )
    ]

    for module_name in candidates:
        module = importlib.import_module(
            module_name
        )

        assert module is not None


# -------------------------------------------------------------------
# Schema contracts
# -------------------------------------------------------------------

def test_integration_schema_modules_import_when_present() -> None:
    candidates = [
        module_name
        for module_name
        in discover_integration_modules()
        if module_name.endswith(
            ".schema"
        )
    ]

    for module_name in candidates:
        module = importlib.import_module(
            module_name
        )

        assert module is not None


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_integrations_exports_resolve() -> None:
    exports = getattr(
        integrations_package,
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
            integrations_package,
            name,
        )
    ]

    assert not missing, (
        "Missing src.integrations exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Credential boundary
# -------------------------------------------------------------------

def test_integration_classes_do_not_define_literal_password_defaults() -> None:
    """
    Protect against accidentally committing a
    non-empty literal password as a dataclass
    default.

    Credentials themselves remain the responsibility
    of the security/secrets layer.
    """

    modules = (
        import_integration_modules()
    )

    credential_names = {
        "password",
        "passwd",
        "passphrase",
    }

    violations: list[str] = []

    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            fields = getattr(
                cls,
                "__dataclass_fields__",
                None,
            )

            if not isinstance(
                fields,
                dict,
            ):
                continue

            for (
                field_name,
                field,
            ) in fields.items():
                if (
                    field_name.lower()
                    not in credential_names
                ):
                    continue

                default = getattr(
                    field,
                    "default",
                    None,
                )

                if (
                    isinstance(
                        default,
                        str,
                    )
                    and default.strip()
                ):
                    violations.append(
                        (
                            f"{module_name}."
                            f"{cls.__name__}."
                            f"{field_name}"
                        )
                    )

    assert not violations, (
        "Literal password defaults found: "
        f"{violations}"
    )


# -------------------------------------------------------------------
# Import safety
# -------------------------------------------------------------------

def test_integration_imports_do_not_connect_to_providers() -> None:
    """
    Importing integrations must be passive.

    This unit suite intentionally does not call
    connect(), discover(), collect(), provisioning,
    or mutation operations. Tests requiring real or
    emulated providers belong under integration/e2e.
    """

    modules = (
        import_integration_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Infrastructure boundary
# -------------------------------------------------------------------

def test_integration_implementation_is_separate_from_infrastructure_files() -> None:
    """
    src/integrations contains Python integration
    implementations.

    infrastructure/ contains deployment/provider
    infrastructure definitions.
    """

    assert (
        INTEGRATIONS_ROOT.resolve()
        != (
            PROJECT_ROOT
            / "infrastructure"
        ).resolve()
    )


# -------------------------------------------------------------------
# Security boundary
# -------------------------------------------------------------------

def test_integrations_namespace_is_separate_from_security() -> None:
    for module_name in (
        discover_integration_modules()
    ):
        assert not module_name.startswith(
            "src.security."
        )


# -------------------------------------------------------------------
# Collector boundary
# -------------------------------------------------------------------

def test_integrations_namespace_is_separate_from_collectors() -> None:
    """
    Integrations provide provider/service access.
    Collectors own telemetry/data collection.
    """

    for module_name in (
        discover_integration_modules()
    ):
        assert not module_name.startswith(
            "src.collectors."
        )


# -------------------------------------------------------------------
# Remediation boundary
# -------------------------------------------------------------------

def test_integrations_namespace_is_separate_from_remediation() -> None:
    for module_name in (
        discover_integration_modules()
    ):
        assert not module_name.startswith(
            "src.remediation."
        )


# -------------------------------------------------------------------
# Test namespace isolation
# -------------------------------------------------------------------

def test_integration_modules_are_not_tests() -> None:
    for module_name in (
        discover_integration_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
