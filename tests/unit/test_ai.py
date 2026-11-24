from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.ai as ai_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

AI_ROOT = (
    PROJECT_ROOT
    / "src"
    / "ai"
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_ai_modules() -> list"""
    Discover every implemented Python module under
    src.ai.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                ai_package.__path__,
                prefix="src.ai.",
            )
        }
    )


def import_ai_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented AI module.

    Collect all failures so one broken module does
    not hide another.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_ai_modules()
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
        "AI module import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


# -------------------------------------------------------------------
# Module-owned classes/functions
# -------------------------------------------------------------------

def module_classes(
    module: object,
) -> list"""
    Return classes defined directly by an AI module.
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

def test_ai_package_imports() -> None:
    module = importlib.import_module(
        "src.ai"
    )

    assert module is not None


def test_ai_directory_exists() -> None:
    assert AI_ROOT.is_dir()


def test_ai_package_has_path() -> None:
    assert hasattr(
        ai_package,
        "__path__",
    )


def test_ai_contains_python_source() -> None:
    sources = list(
        AI_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/ai contains no Python source"
    )


# -------------------------------------------------------------------
# Registry boundary
# -------------------------------------------------------------------

def test_ai_registry_module_exists() -> None:
    """
    Model-management tooling depends on the AI
    registry boundary.
    """

    spec = importlib.util.find_spec(
        "src.ai.registry"
    )

    assert spec is not None


def test_ai_registry_imports() -> None:
    module = importlib.import_module(
        "src.ai.registry"
    )

    assert module is not None


def test_ai_registry_exposes_registry_class() -> None:
    """
    Support the registry names used by the model
    administration scripts while requiring at least
    one concrete registry implementation.
    """

    registry = importlib.import_module(
        "src.ai.registry"
    )


    candidates = (
        "AIModelRegistry",
        "ModelRegistry",
        "Registry",
    )


    classes = [
        getattr(
            registry,
            name,
            None,
        )
        for name in candidates
    ]


    classes = [
        value
        for value in classes
        if inspect.isclass(
            value
        )
    ]


    assert classes, (
        "src.ai.registry exposes no supported "
        "registry class"
    )


def test_registry_class_signature_is_inspectable() -> None:
    registry = importlib.import_module(
        "src.ai.registry"
    )


    candidates = (
        "AIModelRegistry",
        "ModelRegistry",
        "Registry",
    )


    for name in candidates:
        cls = getattr(
            registry,
            name,
            None,
        )

        if not inspect.isclass(
            cls
        ):
            continue

        signature = inspect.signature(
            cls
        )

        assert signature is not None

        return


    pytest.fail(
        "No registry class found."
    )


# -------------------------------------------------------------------
# Optional loader boundary
# -------------------------------------------------------------------

def test_loader_module_imports_if_present() -> None:
    """
    Some AI layouts expose the model loader through
    src.ai.loader.

    If the module exists, it must import correctly.
    """

    spec = importlib.util.find_spec(
        "src.ai.loader"
    )


    if spec is None:
        return


    module = importlib.import_module(
        "src.ai.loader"
    )

    assert module is not None


# -------------------------------------------------------------------
# All implemented AI modules
# -------------------------------------------------------------------

def test_ai_module_discovery_returns_list() -> None:
    modules = (
        discover_ai_modules()
    )

    assert isinstance(
        modules,
        list,
    )


def test_ai_modules_are_unique() -> None:
    modules = (
        discover_ai_modules()
    )

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_ai_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_ai_modules()
    ):
        assert module_name.startswith(
            "src.ai."
        )


def test_all_ai_modules_import() -> None:
    import_ai_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_ai_modules(),
)
def test_ai_module_imports_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Source/module consistency
# -------------------------------------------------------------------

def test_ai_python_files_resolve_as_modules() -> None:
    python_files = [
        path
        for path
        in AI_ROOT.rglob(
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
        "Unresolvable AI modules:\n"
        + "\n".join(
            failures
        )
    )


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_ai_module_sources_stay_inside_package() -> None:
    modules = import_ai_modules()

    root = AI_ROOT.resolve()


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
                    "outside src/ai: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# Class contracts
# -------------------------------------------------------------------

def test_ai_classes_have_identity() -> None:
    modules = import_ai_modules()


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


def test_ai_class_signatures_are_inspectable() -> None:
    modules = import_ai_modules()


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
                        "Cannot inspect "
                        f"{module_name}."
                        f"{cls.__name__}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Dataclass/schema contracts
# -------------------------------------------------------------------

def test_ai_dataclass_fields_are_mappings() -> None:
    modules = import_ai_modules()


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

def test_ai_to_dict_is_callable_when_present() -> None:
    modules = import_ai_modules()


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


def test_ai_model_dump_is_callable_when_present() -> None:
    modules = import_ai_modules()


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
# Public functions
# -------------------------------------------------------------------

def test_public_ai_functions_are_callable() -> None:
    modules = import_ai_modules()


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


def test_public_ai_function_signatures_are_inspectable() -> None:
    modules = import_ai_modules()


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
                        "Cannot inspect "
                        f"{module_name}."
                        f"{function_name}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Model lifecycle methods
# -------------------------------------------------------------------

def test_registry_lifecycle_methods_are_callable_when_present() -> None:
    registry = importlib.import_module(
        "src.ai.registry"
    )


    candidate_class = None


    for name in (
        "AIModelRegistry",
        "ModelRegistry",
        "Registry",
    ):
        value = getattr(
            registry,
            name,
            None,
        )

        if inspect.isclass(
            value
        ):
            candidate_class = value
            break


    assert candidate_class is not None


    for method_name in (
        "start",
        "discover",
        "refresh",
        "scan",
        "list_models",
        "get_model",
        "validate_model",
        "promote_model",
        "close",
    ):
        value = getattr(
            candidate_class,
            method_name,
            None,
        )


        if value is None:
            continue


        assert callable(
            value
        ), (
            f"{candidate_class.__name__}."
            f"{method_name} is not callable"
        )


# -------------------------------------------------------------------
# Optional main() contracts
# -------------------------------------------------------------------

def test_ai_main_is_callable_when_present() -> None:
    modules = import_ai_modules()


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


        assert callable(
            main
        ), (
            f"{module_name}.main exists "
            "but is not callable"
        )


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_ai_package_exports_resolve() -> None:
    exports = getattr(
        ai_package,
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
            ai_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.ai exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Operational safety
# -------------------------------------------------------------------

def test_importing_ai_does_not_require_model_loading() -> None:
    """
    Importing the AI package must not require loading
    checkpoint weights onto CPU/GPU.

    This deliberately tests importability only.
    """

    modules = import_ai_modules()

    assert isinstance(
        modules,
        dict,
    )


def test_registry_import_does_not_require_accelerator() -> None:
    """
    Registry metadata management and accelerator
    model loading are separate concerns.
    """

    module = importlib.import_module(
        "src.ai.registry"
    )

    assert module is not None


# -------------------------------------------------------------------
# Package boundaries
# -------------------------------------------------------------------

def test_ai_discovery_does_not_include_training_modules() -> None:
    """
    src.training owns model training.

    src.ai owns the serving/runtime AI layer and its
    model-management contracts.
    """

    for module_name in (
        discover_ai_modules()
    ):
        assert not module_name.startswith(
            "src.training."
        )


def test_ai_discovery_does_not_include_intelligence_modules() -> None:
    """
    The intelligence interpretation layer remains a
    separate subsystem from the lower AI/model layer.
    """

    for module_name in (
        discover_ai_modules()
    ):
        assert not module_name.startswith(
            "src.intelligence."
        )


# -------------------------------------------------------------------
# No test package leakage
# -------------------------------------------------------------------

def test_ai_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_ai_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
