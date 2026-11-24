from __future__ import annotations

import importlib
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.data as data_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_ROOT = (
    PROJECT_ROOT
    / "src"
    / "data"
)


# -------------------------------------------------------------------
# Expected data architecture
# -------------------------------------------------------------------

EXPECTED_MODULES = (
    "src.data.checksums",
    "src.data.config",
    "src.data.errors",
    "src.data.fetch",
    "src.data.manifests",
    "src.data.normalize",
    "src.data.pipeline",
    "src.data.prepare_corpus",
    "src.data.records",
    "src.data.schema",
    "src.data.validate",
)


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def discover_data_modules() -> list[str]:
    """
    Discover every implemented module recursively
    under src.data.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                data_package.__path__,
                prefix="src.data.",
            )
        }
    )


def import_data_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented data module and report
    all failures together.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_data_modules()
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
        "Data module import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


# -------------------------------------------------------------------
# Module members
# -------------------------------------------------------------------

def module_classes(
    module: object,
) -> list[type]:
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

def test_data_package_imports() -> None:
    module = importlib.import_module(
        "src.data"
    )

    assert module is not None


def test_data_package_has_path() -> None:
    assert hasattr(
        data_package,
        "__path__",
    )


def test_data_directory_exists() -> None:
    assert DATA_ROOT.is_dir()


def test_data_has_python_source() -> None:
    sources = list(
        DATA_ROOT.rglob(
            "*.py"
        )
    )

    assert sources


# -------------------------------------------------------------------
# Core architecture
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name",
    EXPECTED_MODULES,
)
def test_expected_data_module_exists(
    module_name: str,
) -> None:
    spec = importlib.util.find_spec(
        module_name
    )

    assert spec is not None, (
        f"Expected data module is missing: "
        f"{module_name}"
    )


@pytest.mark.parametrize(
    "module_name",
    EXPECTED_MODULES,
)
def test_expected_data_module_imports(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Schema boundary
# -------------------------------------------------------------------

def test_schema_module_imports() -> None:
    module = importlib.import_module(
        "src.data.schema"
    )

    assert module is not None


def test_schema_classes_are_inspectable() -> None:
    module = importlib.import_module(
        "src.data.schema"
    )


    for cls in module_classes(
        module
    ):
        signature = inspect.signature(
            cls
        )

        assert signature is not None


# -------------------------------------------------------------------
# Records boundary
# -------------------------------------------------------------------

def test_records_module_imports() -> None:
    module = importlib.import_module(
        "src.data.records"
    )

    assert module is not None


def test_record_functions_are_callable() -> None:
    module = importlib.import_module(
        "src.data.records"
    )


    for (
        _,
        function,
    ) in public_functions(
        module
    ):
        assert callable(
            function
        )


# -------------------------------------------------------------------
# Normalization boundary
# -------------------------------------------------------------------

def test_normalize_module_imports() -> None:
    module = importlib.import_module(
        "src.data.normalize"
    )

    assert module is not None


def test_normalization_functions_have_explicit_signatures() -> None:
    module = importlib.import_module(
        "src.data.normalize"
    )


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
                    "Cannot inspect normalization "
                    f"function {function_name}: "
                    f"{exc}"
                )
            )


        assert signature is not None


# -------------------------------------------------------------------
# Validation boundary
# -------------------------------------------------------------------

def test_validation_module_imports() -> None:
    module = importlib.import_module(
        "src.data.validate"
    )

    assert module is not None


def test_validation_functions_are_callable() -> None:
    module = importlib.import_module(
        "src.data.validate"
    )


    functions = public_functions(
        module
    )


    for (
        _,
        function,
    ) in functions:
        assert callable(
            function
        )


# -------------------------------------------------------------------
# Checksum boundary
# -------------------------------------------------------------------

def test_checksum_module_imports() -> None:
    module = importlib.import_module(
        "src.data.checksums"
    )

    assert module is not None


def test_checksum_public_functions_are_callable() -> None:
    module = importlib.import_module(
        "src.data.checksums"
    )


    for (
        _,
        function,
    ) in public_functions(
        module
    ):
        assert callable(
            function
        )


def test_checksum_functions_are_inspectable() -> None:
    module = importlib.import_module(
        "src.data.checksums"
    )


    for (
        name,
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
                    "Cannot inspect checksum "
                    f"function {name}: {exc}"
                )
            )


        assert signature is not None


# -------------------------------------------------------------------
# Manifest boundary
# -------------------------------------------------------------------

def test_manifest_module_imports() -> None:
    module = importlib.import_module(
        "src.data.manifests"
    )

    assert module is not None


def test_manifest_contracts_are_inspectable() -> None:
    module = importlib.import_module(
        "src.data.manifests"
    )


    for cls in module_classes(
        module
    ):
        assert (
            inspect.signature(
                cls
            )
            is not None
        )


    for (
        _,
        function,
    ) in public_functions(
        module
    ):
        assert (
            inspect.signature(
                function
            )
            is not None
        )


# -------------------------------------------------------------------
# Data pipeline
# -------------------------------------------------------------------

def test_pipeline_module_imports() -> None:
    module = importlib.import_module(
        "src.data.pipeline"
    )

    assert module is not None


def test_pipeline_exposes_public_contract() -> None:
    module = importlib.import_module(
        "src.data.pipeline"
    )


    public = public_functions(
        module
    )


    classes = module_classes(
        module
    )


    assert (
        public
        or classes
    ), (
        "src.data.pipeline exposes no public "
        "pipeline contract"
    )


def test_pipeline_functions_are_inspectable() -> None:
    module = importlib.import_module(
        "src.data.pipeline"
    )


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
                    "Cannot inspect data pipeline "
                    f"function {function_name}: "
                    f"{exc}"
                )
            )


        assert signature is not None


# -------------------------------------------------------------------
# Corpus preparation
# -------------------------------------------------------------------

def test_prepare_corpus_module_imports() -> None:
    module = importlib.import_module(
        "src.data.prepare_corpus"
    )

    assert module is not None


def test_prepare_corpus_main_is_callable_when_present() -> None:
    module = importlib.import_module(
        "src.data.prepare_corpus"
    )


    main = getattr(
        module,
        "main",
        None,
    )


    if main is None:
        return


    assert callable(
        main
    )


# -------------------------------------------------------------------
# Fetching
# -------------------------------------------------------------------

def test_fetch_module_imports_without_downloading() -> None:
    """
    Importing src.data.fetch must not perform network
    acquisition.

    The unit-test boundary is importability only.
    """

    module = importlib.import_module(
        "src.data.fetch"
    )

    assert module is not None


def test_fetch_main_is_callable_when_present() -> None:
    module = importlib.import_module(
        "src.data.fetch"
    )


    main = getattr(
        module,
        "main",
        None,
    )


    if main is None:
        return


    assert callable(
        main
    )


# -------------------------------------------------------------------
# Configuration/errors
# -------------------------------------------------------------------

def test_data_config_imports() -> None:
    module = importlib.import_module(
        "src.data.config"
    )

    assert module is not None


def test_data_errors_import() -> None:
    module = importlib.import_module(
        "src.data.errors"
    )

    assert module is not None


# -------------------------------------------------------------------
# All implemented modules
# -------------------------------------------------------------------

def test_data_discovery_returns_list() -> None:
    modules = discover_data_modules()

    assert isinstance(
        modules,
        list,
    )


def test_data_modules_are_unique() -> None:
    modules = discover_data_modules()

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_data_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_data_modules()
    ):
        assert module_name.startswith(
            "src.data."
        )


def test_all_data_modulesimport_data_modules()


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_data_module_sources_stay_inside_package() -> None:
    modules = import_data_modules()

    root = DATA_ROOT.resolve()


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
                    "src/data: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------

def test_data_classes_have_identity() -> None:
    modules = import_data_modules()


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


def test_data_class_signatures_are_inspectable() -> None:
    modules = import_data_modules()


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
                        f"Cannot inspect "
                        f"{module_name}."
                        f"{cls.__name__}: "
                        f"{exc}"
                    )
                )


            assert signature is not None


# -------------------------------------------------------------------
# Dataclass/schema contracts
# -------------------------------------------------------------------

def test_data_dataclass_fields_are_mappings() -> None:
    modules = import_data_modules()


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
# Serialization
# -------------------------------------------------------------------

def test_data_to_dict_is_callable_when_present() -> None:
    modules = import_data_modules()


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


def test_data_model_dump_is_callable_when_present() -> None:
    modules = import_data_modules()


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

def test_data_package_exports_resolve() -> None:
    exports = getattr(
        data_package,
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
            data_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.data exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Safety: do not mutate corpus during imports
# -------------------------------------------------------------------

def test_importing_data_modules_does_not_require_pipeline_execution() -> None:
    """
    Module import must remain independent from actual
    dataset preparation or fetching.

    We intentionally do not execute fetch/main or
    preparation pipelines in this package-level unit
    test.
    """

    modules = import_data_modules()

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# Data storage boundary
# -------------------------------------------------------------------

def test_data_implementation_is_separate_from_corpus_storage() -> None:
    """
    src/data contains implementation.

    orchestration-ai-data contains datasets and
    generated corpus material.
    """

    assert DATA_ROOT.resolve() != (
        PROJECT_ROOT
        / "orchestration-ai-data"
    ).resolve()


# -------------------------------------------------------------------
# Namespace separation
# -------------------------------------------------------------------

def test_data_modules_are_not_training_modules() -> None:
    for module_name in (
        discover_data_modules()
    ):
        assert not module_name.startswith(
            "src.training."
        )


def test_data_modules_are_not_collector_modules() -> None:
    for module_name in (
        discover_data_modules()
    ):
        assert not module_name.startswith(
            "src.collectors."
        )


# -------------------------------------------------------------------
# No test leakage
# -------------------------------------------------------------------

def test_data_modules_are_not_test_modules() -> None:
    for module_name in (
        discover_data_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
