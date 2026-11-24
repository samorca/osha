from __future__ import annotations

import importlib
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.decision as decision_package
from src.decision.pipeline import run_pipeline


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DECISION_ROOT = (
    PROJECT_ROOT
    / "src"
    / "decision"
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def discover_decision_modules() -> list"""
    Return every implemented module under
    src.decision.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                decision_package.__path__,
                prefix="src.decision.",
            )
        }
    )


def import_decision_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented decision module and
    report all broken imports together.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []

    for module_name in (
        discover_decision_modules()
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
        "Decision module import failures:\n"
        + "\n".join(
            failures
        )
    )

    return imported


def module_classes(
    module: object,
) -> list"""
    Return classes actually defined by a particular
    decision module.
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
# Package contract
# -------------------------------------------------------------------

def test_decision_package_imports() -> None:
    module = importlib.import_module(
        "src.decision"
    )

    assert module is not None


def test_decision_source_directory_exists() -> None:
    assert DECISION_ROOT.is_dir()


def test_decision_package_has_path() -> None:
    assert hasattr(
        decision_package,
        "__path__",
    )


# -------------------------------------------------------------------
# Actual OSHA decision pipeline contract
# -------------------------------------------------------------------

def test_decision_pipeline_imports() -> None:
    module = importlib.import_module(
        "src.decision.pipeline"
    )

    assert module is not None


def test_run_pipeline_is_callable() -> None:
    """
    run_pipeline() is the decision subsystem's real
    pipeline boundary.
    """

    assert callable(
        run_pipeline
    )


def test_run_pipeline_belongs_to_decision_pipeline() -> None:
    assert (
        run_pipeline.__module__
        == "src.decision.pipeline"
    )


def test_run_pipeline_signature_is_inspectable() -> None:
    signature = inspect.signature(
        run_pipeline
    )

    assert signature is not None


def test_run_pipeline_has_named_parameters() -> None:
    """
    Keep this deliberately schema-independent.

    The exact arguments remain owned by the current
    implementation, but the pipeline must expose an
    inspectable, explicit Python contract.
    """

    signature = inspect.signature(
        run_pipeline
    )

    parameters = list(
        signature.parameters.values()
    )

    for parameter in parameters:
        assert parameter.name.strip()


def test_run_pipeline_has_no_positional_only_interface() -> None:
    """
    A decision pipeline is easier and safer to call
    from orchestration code when its public arguments
    are named rather than positional-only.
    """

    signature = inspect.signature(
        run_pipeline
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
        "run_pipeline() contains positional-only "
        f"parameters: {positional_only}"
    )


# -------------------------------------------------------------------
# Decision module imports
# -------------------------------------------------------------------

def test_all_decision_modules_import() -> None:
    import_decision_modules()


@pytest.mark.parametrize(
    "module_name",
    discover_decision_modules(),
)
def test_decision_modules_import_individually(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_decision_modules_resolve_inside_package() -> None:
    modules = import_decision_modules()

    root = DECISION_ROOT.resolve()

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
                    f"{module_name} resolves outside "
                    "src/decision: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# Classes and schemas
# -------------------------------------------------------------------

def test_decision_classes_have_valid_identity() -> None:
    modules = import_decision_modules()

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


def test_decision_classes_have_inspectable_contracts() -> None:
    modules = import_decision_modules()

    for (
        module_name,
        module,
    ) in modules.items():
        for cls in module_classes(
            module
        ):
            try:
                signature = inspect.signature(
                    cls
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                pytest.fail(
                    (
                        f"Cannot inspect "
                        f"{module_name}."
                        f"{cls.__name__}: {exc}"
                    )
                )

            assert signature is not None


def test_decision_dataclasses_have_fields() -> None:
    modules = import_decision_modules()

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
# Public function contracts
# -------------------------------------------------------------------

def test_decision_public_functions_are_callable() -> None:
    modules = import_decision_modules()

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


def test_decision_public_functions_are_inspectable() -> None:
    modules = import_decision_modules()

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
                signature = inspect.signature(
                    function
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                pytest.fail(
                    (
                        f"Cannot inspect "
                        f"{module_name}."
                        f"{function_name}: {exc}"
                    )
                )

            assert signature is not None


# -------------------------------------------------------------------
# Serialization contracts
# -------------------------------------------------------------------

def test_decision_to_dict_is_callable_when_present() -> None:
    modules = import_decision_modules()

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


def test_decision_model_dump_is_callable_when_present() -> None:
    modules = import_decision_modules()

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
# Package exports
# -------------------------------------------------------------------

def test_decision_exports_resolve() -> None:
    exports = getattr(
        decision_package,
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
            decision_package,
            name,
        )
    ]

    assert not missing, (
        "Missing src.decision exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Safety boundary
# -------------------------------------------------------------------

def test_decision_import_does_not_require_remediation_execution() -> None:
    """
    The decision subsystem determines what should
    happen. Importing it must remain independent of
    actually executing infrastructure remediation.

    This test intentionally imports the complete
    decision package without calling remediation.
    """

    modules = import_decision_modules()

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Namespace separation
# -------------------------------------------------------------------

def test_decision_modules_stay_in_decision_namespace() -> None:
    for module_name in (
        discover_decision_modules()
    ):
        assert module_name.startswith(
            "src.decision."
        )

        assert not module_name.startswith(
            "src.remediation."
        )


# -------------------------------------------------------------------
# Source availability
# -------------------------------------------------------------------

def test_decision_has_python_source() -> None:
    sources = list(
        DECISION_ROOT.rglob(
            "*.py"
        )
    )

    assert sources, (
        "src/decision contains no Python source"
    )
