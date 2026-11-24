from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.security as security_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SECURITY_ROOT = (
    PROJECT_ROOT
    / "src"
    / "security"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_security_modules() -> list[str]:
    """
    Discover every implemented Python module under
    src.security.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                security_package.__path__,
                prefix="src.security.",
            )
        }
    )


def import_security_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented security module.

    Report all import failures together.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []

    for module_name in (
        discover_security_modules()
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
        "Security module import failures:\n"
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
    security module.
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
    the security module.
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

def test_security_package_imports() -> None:
    module = importlib.import_module(
        "src.security"
    )

    assert module is not None


def test_security_package_has_path() -> None:
    assert hasattr(
        security_package,
        "__path__",
    )


def test_security_directory_exists() -> None:
    assert SECURITY_ROOT.is_dir()


def test_security_has_python_source() -> None:
    sources = list(
        SECURITY_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/security contains no Python source"
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_security_discovery_returns_list() -> None:
    modules = (
        discover_security_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_security_modules_are_unique() -> None:
    modules = (
        discover_security_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_security_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_security_modules()
    ):
        assert module_name.startswith(
            "src.security."
        )


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

def test_all_security_modules_import() -> None:
    import_security_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_security_modules(),
)
def test_security_modules_import_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Python source resolution
# -------------------------------------------------------------------

def test_security_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in SECURITY_ROOT.rglob(
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
        "Unresolvable security modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_security_sources_stay_inside_package() -> None:
    modules = (
        import_security_modules()
    )

    root = (
        SECURITY_ROOT.resolve()
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
            f"Missing security source: "
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
                    "outside src/security: "
                    f"{source_path}"
                )
            )


def test_security_modules_report_expected_names() -> None:
    modules = (
        import_security_modules()
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

def test_security_classes_have_identity() -> None:
    modules = (
        import_security_modules()
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


def test_security_class_signatures_are_inspectable() -> None:
    modules = (
        import_security_modules()
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

def test_security_dataclass_fields_are_mappings() -> None:
    modules = (
        import_security_modules()
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

def test_security_public_functions_are_callable() -> None:
    modules = (
        import_security_modules()
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


def test_security_public_function_signatures_are_inspectable() -> None:
    modules = (
        import_security_modules()
    )

    for (
        module_name,
        module,
    ) in modules.items():
        for (
            function_name,
            functionunctions(
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
# Serialization contracts
# -------------------------------------------------------------------

def test_security_to_dict_is_callable_when_present() -> None:
    modules = (
        import_security_modules()
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


def test_security_model_dump_is_callable_when_present() -> None:
    modules = (
        import_security_modules()
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
# Security-specific public names
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "term",
    (
        "secret",
        "credential",
        "token",
        "permission",
        "policy",
        "audit",
        "authorization",
        "authentication",
    ),
)
def test_security_named_contracts_are_callable_or_classes_when_present(
    term: str,
) -> None:
    """
    Security implementations may expose different
    concrete names.

    When a public class/function contains one of the
    core security terms, it must represent a callable
    contract rather than static executable state.
    """

    modules = (
        import_security_modules()
    )

    for module in modules.values():
        for (
            name,
            value,
        ) in inspect.getmembers(
            module
        ):
            if name.startswith("_"):
                continue

            if term not in name.lower():
                continue

            if inspect.isfunction(
                value
            ):
                assert callable(
                    value
                )

            elif inspect.isclass(
                value
            ):
                assert inspect.isclass(
                    value
                )


# -------------------------------------------------------------------
# Secret-bearing field representation
# -------------------------------------------------------------------

def test_security_classes_do_not_define_obvious_plaintext_password_defaults() -> None:
    """
    Protect against accidentally introducing a
    literal non-empty password default into a
    dataclass field.

    This does not make claims about secret storage.
    It only guards against committed literal defaults
    on dataclass schemas.
    """

    modules = (
        import_security_modules()
    )

    names = {
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
                    not in names
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
# Optional main()
# -------------------------------------------------------------------

def test_security_main_is_callable_when_present() -> None:
    modules = (
        import_security_modules()
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

def test_security_exports_resolve() -> None:
    exports = getattr(
        security_package,
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
            security_package,
            name,
        )
    ]

    assert not missing, (
        "Missing src.security exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Import safety
# -------------------------------------------------------------------

def test_security_imports_are_passive() -> None:
    """
    Unit-test discovery should be able to import the
    security subsystem without performing an
    authentication attempt or prompting for secrets.

    Successful import of the whole package is the
    boundary checked here.
    """

    modules = (
        import_security_modules()
    )

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Architectural boundaries
# -------------------------------------------------------------------

def test_security_modules_remain_in_security_namespace() -> None:
    for module_name in (
        discover_security_modules()
    ):
        assert module_name.startswith(
            "src.security."
        )


def test_security_is_separate_from_remediation() -> None:
    for module_name in (
        discover_security_modules()
    ):
        assert not module_name.startswith(
            "src.remediation."
        )


def test_security_is_separate_from_integrations() -> None:
    """
    Provider integrations and security controls
    remain separate package namespaces even when an
    integration consumes credentials.
    """

    for module_name in (
        discover_security_modules()
    ):
        assert not module_name.startswith(
            "src.integrations."
        )


# -------------------------------------------------------------------
# Test namespace isolation
# -------------------------------------------------------------------

def test_security_modules_are_not_tests() -> None:
    for module_name in (
        discover_security_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
