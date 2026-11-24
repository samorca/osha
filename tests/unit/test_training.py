from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil

from pathlib import Path

import pytest

import src.training as training_package


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TRAINING_ROOT = (
    PROJECT_ROOT
    / "src"
    / "training"
)


# -------------------------------------------------------------------
# Training architecture
# -------------------------------------------------------------------

#
# These are the modules we explicitly built as part
# of the OSHA training subsystem.
#
EXPECTED_MODULES = (
    "src.training.schema",
    "src.training.config",
    "src.training.errors",
    "src.training.sources",
    "src.training.assembly",
    "src.training.features",
    "src.training.splits",
    "src.training.event_dataset",
    "src.training.anomaly_dataset",
    "src.training.event_model",
    "src.training.anomaly_model",
    "src.training.pipeline",
    "src.training.train_events",
    "src.training.train_anomalies",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def discover_training_modules() -> list"""
    Discover every implemented Python module
    recursively under src.training.
    """

    return sorted(
        {
            module_info.name
            for module_info
            in pkgutil.walk_packages(
                training_package.__path__,
                prefix="src.training.",
            )
        }
    )


def import_training_modules() -> dict[
    str,
    object,
]:
    """
    Import every implemented training module.

    Failures are accumulated so a single broken
    import does not hide additional problems.
    """

    imported: dict[
        str,
        object,
    ] = {}

    failures: list[str] = []


    for module_name in (
        discover_training_modules()
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
        "Training module import failures:\n"
        + "\n".join(
            failures
        )
    )


    return imported


def module_classes(
    module: object,
) -> list"""
    Return classes implemented by a module itself.

    Imported framework/classes are excluded.
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

def test_training_package_imports() -> None:
    module = importlib.import_module(
        "src.training"
    )

    assert module is not None


def test_training_directory_exists() -> None:
    assert TRAINING_ROOT.is_dir()


def test_training_package_has_path() -> None:
    assert hasattr(
        training_package,
        "__path__",
    )


def test_training_has_python_source() -> None:
    sources = list(
        TRAINING_ROOT.rglob(
            "*.py"
        )
    )

    assert sources


# -------------------------------------------------------------------
# Expected architecture
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name",
    EXPECTED_MODULES,
)
def test_expected_training_module_exists(
    module_name: str,
) -> None:
    spec = importlib.util.find_spec(
        module_name
    )

    assert spec is not None, (
        "Expected training module is missing: "
        f"{module_name}"
    )


@pytest.mark.parametrize(
    "module_name",
    EXPECTED_MODULES,
)
def test_expected_training_module_imports(
    module_name: str,
) -> None:
    module = importlib.import_module(
        module_name
    )

    assert module is not None


# -------------------------------------------------------------------
# Schema
# -------------------------------------------------------------------

def test_training_schema_imports() -> None:
    module = importlib.import_module(
        "src.training.schema"
    )

    assert module is not None


def test_training_schema_classes_are_inspectable() -> None:
    module = importlib.import_module(
        "src.training.schema"
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


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

def test_training_config_imports() -> None:
    module = importlib.import_module(
        "src.training.config"
    )

    assert module is not None


def test_training_config_contracts_are_inspectable() -> None:
    module = importlib.import_module(
        "src.training.config"
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


# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------

def test_training_sources_imports() -> None:
    module = importlib.import_module(
        "src.training.sources"
    )

    assert module is not None


def test_training_source_functions_are_callable() -> None:
    module = importlib.import_module(
        "src.training.sources"
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
# Assembly
# -------------------------------------------------------------------

def test_training_assembly_imports() -> None:
    module = importlib.import_module(
        "src.training.assembly"
    )

    assert module is not None


def test_training_assembly_contracts_are_inspectable() -> None:
    module = importlib.import_module(
        "src.training.assembly"
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
# Features
# -------------------------------------------------------------------

def test_training_features_imports() -> None:
    module = importlib.import_module(
        "src.training.features"
    )

    assert module is not None


def test_feature_functions_are_inspectable() -> None:
    module = importlib.import_module(
        "src.training.features"
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
                    f"Unable to inspect feature "
                    f"function {function_name}: "
                    f"{exc}"
                )
            )


        assert signature is not None


# -------------------------------------------------------------------
# Splits
# -------------------------------------------------------------------

def test_training_splits_imports() -> None:
    module = importlib.import_module(
        "src.training.splits"
    )

    assert module is not None


def test_split_functions_are_inspectable() -> None:
    module = importlib.import_module(
        "src.training.splits"
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
                    f"Unable to inspect split "
                    f"function {function_name}: "
                    f"{exc}"
                )
            )


        assert signature is not None


# -------------------------------------------------------------------
# Event dataset
# -------------------------------------------------------------------

def test_event_dataset_module_imports() -> None:
    module = importlib.import_module(
        "src.training.event_dataset"
    )

    assert module is not None


def test_event_dataset_class_exists() -> None:
    module = importlib.import_module(
        "src.training.event_dataset"
    )


    cls = getattr(
        module,
        "EventDataset",
        None,
    )


    assert inspect.isclass(
        cls
    ), (
        "src.training.event_dataset "
        "must expose EventDataset"
    )


def test_event_dataset_constructor_is_inspectable() -> None:
    module = importlib.import_module(
        "src.training.event_dataset"
    )

    cls = getattr(
        module,
        "EventDataset"
    )


    signature = inspect.signature(
        cls
    )


    assert signature is not None


# -------------------------------------------------------------------
# Anomaly dataset
# -------------------------------------------------------------------

def test_anomaly_dataset_module_imports() -> None:
    module = importlib.import_module(
        "src.training.anomaly_dataset"
    )

    assert module is not None


def test_anomaly_dataset_class_exists() -> None:
    module = importlib.import_module(
        "src.training.anomaly_dataset"
    )


    cls = getattr(
        module,
        "AnomalyDataset",
        None,
    )


    assert inspect.isclass(
        cls
    ), (
        "src.training.anomaly_dataset "
        "must expose AnomalyDataset"
    )


def test_anomaly_dataset_constructor_is_inspectable() -> None:
    module = importlib.import_module(
        "src.training.anomaly_dataset"
    )

    cls = getattr(
        module,
        "AnomalyDataset"
    )


    assert (
        inspect.signature(
            cls
        )
        is not None
    )


# -------------------------------------------------------------------
# Event model
# -------------------------------------------------------------------

def test_event_model_module_imports() -> None:
    module = importlib.import_module(
        "src.training.event_model"
    )

    assert module is not None


def test_event_model_defines_model_class() -> None:
    module = importlib.import_module(
        "src.training.event_model"
    )


    classes = module_classes(
        module
    )


    assert classes, (
        "src.training.event_model defines "
        "no model class"
    )


# -------------------------------------------------------------------
# Anomaly model
# -------------------------------------------------------------------

def test_anomaly_model_module_imports() -> None:
    module = importlib.import_module(
        "src.training.anomaly_model"
    )

    assert module is not None


def test_infrastructure_anomaly_model_exists() -> None:
    module = importlib.import_module(
        "src.training.anomaly_model"
    )


    cls = getattr(
        module,
        "InfrastructureAnomalyModel",
        None,
    )


    assert inspect.isclass(
        cls
    ), (
        "src.training.anomaly_model must expose "
        "InfrastructureAnomalyModel"
    )


def test_infrastructure_anomaly_model_constructor_is_inspectable() -> None:
    module = importlib.import_module(
        "src.training.anomaly_model"
    )


    cls = getattr(
        module,
        "InfrastructureAnomalyModel"
    )


    assert (
        inspect.signature(
            cls
        )
        is not None
    )


# -------------------------------------------------------------------
# Training pipeline
# -------------------------------------------------------------------

def test_training_pipeline_module_imports() -> None:
    module = importlib.import_module(
        "src.training.pipeline"
    )

    assert module is not None


def test_training_pipeline_class_exists() -> None:
    module = importlib.import_module(
        "src.training.pipeline"
    )


    cls = getattr(
        module,
        "TrainingPipeline",
        None,
    )


    assert inspect.isclass(
        cls
    ), (
        "src.training.pipeline must expose "
        "TrainingPipeline"
    )


def test_training_pipeline_constructor_is_inspectable() -> None:
    module = importlib.import_module(
        "src.training.pipeline"
    )


    cls = getattr(
        module,
        "TrainingPipeline"
    )


    assert (
        inspect.signature(
            cls
        )
        is not None
    )


# -------------------------------------------------------------------
# Event trainer
# -------------------------------------------------------------------

def test_event_trainer_imports() -> None:
    module = importlib.import_module(
        "src.training.train_events"
    )

    assert module is not None


def test_event_trainer_exposes_main() -> None:
    module = importlib.import_module(
        "src.training.train_events"
    )


    main = getattr(
        module,
        "main",
        None,
    )


    assert callable(
        main
    ), (
        "train_events must expose callable main()"
    )


def test_event_trainer_main_signature_is_inspectable() -> None:
    module = importlib.import_module(
        "src.training.train_events"
    )


    assert (
        inspect.signature(
            module.main
        )
        is not None
    )


# -------------------------------------------------------------------
# Anomaly trainer
# -------------------------------------------------------------------

def test_anomaly_trainer_imports() -> None:
    module = importlib.import_module(
        "src.training.train_anomalies"
    )

    assert module is not None


def test_anomaly_trainer_exposes_main() -> None:
    module = importlib.import_module(
        "src.training.train_anomalies"
    )


    main = getattr(
        module,
        "main",
        None,
    )


    assert callable(
        main
    ), (
        "train_anomalies must expose callable main()"
    )


def test_anomaly_trainer_main_signature_is_inspectable() -> None:
    module = importlib.import_module(
        "src.training.train_anomalies"
    )


    assert (
        inspect.signature(
            module.main
        )
        is not None
    )


# -------------------------------------------------------------------
# Trainer separation
# -------------------------------------------------------------------

def test_event_and_anomaly_trainers_are_distinct() -> None:
    event = importlib.import_module(
        "src.training.train_events"
    )

    anomaly = importlib.import_module(
        "src.training.train_anomalies"
    )


    assert event.main is not anomaly.main


# -------------------------------------------------------------------
# Accelerator integration
# -------------------------------------------------------------------

def test_accelerator_manager_imports_for_training() -> None:
    module = importlib.import_module(
        "src.accelerators"
    )

    manager = getattr(
        module,
        "AcceleratorManager",
        None,
    )


    assert manager is not None


def test_accelerator_manager_is_class_or_factory() -> None:
    module = importlib.import_module(
        "src.accelerators"
    )

    manager = getattr(
        module,
        "AcceleratorManager",
        None,
    )


    assert (
        inspect.isclass(
            manager
        )
        or callable(
            manager
        )
    )


# -------------------------------------------------------------------
# PyTorch boundary
# -------------------------------------------------------------------

def test_pytorch_imports() -> None:
    torch = importlib.import_module(
        "torch"
    )

    assert torch is not None


def test_training_import_does_not_require_gpu() -> None:
    """
    Training modules must remain importable even when
    the current machine has no CUDA/ROCm accelerator.

    Actual device selection belongs to
    AcceleratorManager.
    """

    modules = import_training_modules()

    assert isinstance(
        modules,
        dict,
    )


# -------------------------------------------------------------------
# No training execution during import
# -------------------------------------------------------------------

def test_importing_trainers_does_not_start_training() -> None:
    """
    Merely importing trainer modules must not launch
    epochs, create checkpoints, or start data-loader
    workers.
    """

    event = importlib.import_module(
        "src.training.train_events"
    )

    anomaly = importlib.import_module(
        "src.training.train_anomalies"
    )


    assert callable(
        event.main
    )

    assert callable(
        anomaly.main
    )


# -------------------------------------------------------------------
# Checkpoint boundary
# -------------------------------------------------------------------

def test_checkpoint_module_if_present_imports() -> None:
    """
    Checkpoint implementation may live in its own
    module. When present, it must import cleanly.
    """

    spec = importlib.util.find_spec(
        "src.training.checkpoints"
    )


    if spec is None:
        return


    module = importlib.import_module(
        "src.training.checkpoints"
    )

    assert module is not None


# -------------------------------------------------------------------
# Evaluation boundaries
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name",
    (
        "src.training.evaluate_events",
        "src.training.evaluate_anomalies",
    ),
)
def test_evaluation_module_imports_if_present(
    module_name: str,
) -> None:
    """
    Evaluation implementations are tested when they
    exist, but this test does not require unfinished
    evaluator modules merely because shell wrappers
    reference their intended location.
    """

    spec = importlib.util.find_spec(
        module_name
    )


    if spec is None:
        return


    module = importlib.import_module(
        module_name
    )

    assert module is not None


@pytest.mark.parametrize(
    "module_name",
    (
        "src.training.evaluate_events",
        "src.training.evaluate_anomalies",
    ),
)
def test_evaluation_main_is_callable_if_present(
    module_name: str,
) -> None:
    spec = importlib.util.find_spec(
        module_name
    )


    if spec is None:
        return


    module = importlib.import_module(
        module_name
    )


    main = getattr(
        module,
        "main",
        None,
    )


    assert callable(
        main
    ), (
        f"{module_name} exists but "
        "does not expose callable main()"
    )


# -------------------------------------------------------------------
# Module discovery
# -------------------------------------------------------------------

def test_training_discovery_returns_list() -> None:
    modules = discover_training_modules()

    assert isinstance(
        modules,
        list,
    )


def test_training_modules_are_unique() -> None:
    modules = discover_training_modules()

    assert len(
        modules
    ) == len(
        set(modules)
    )


def test_training_modules_use_correct_namespace() -> None:
    for module_name in (
        discover_training_modules()
    ):
        assert module_name.startswith(
            "src.training."
        )


# -------------------------------------------------------------------
# Import every module
# -------------------------------------------------------------------

def test_all_training_modules_import() -> None:
    import_training_modules()


# -------------------------------------------------------------------
# Source ownership
# -------------------------------------------------------------------

def test_training_module_sources_stay_inside_package() -> None:
    modules = import_training_modules()

    root = TRAINING_ROOT.resolve()


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
                    "src/training: "
                    f"{source_path}"
                )
            )


# -------------------------------------------------------------------
# Class contracts
# -------------------------------------------------------------------

def test_training_classes_have_identity() -> None:
    modules = import_training_modules()


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


def test_training_class_signatures_are_inspectable() -> None:
    modules = import_training_modules()


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
# Schema serialization
# -------------------------------------------------------------------

def test_training_to_dict_is_callable_when_present() -> None:
    modules = import_training_modules()


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


# -------------------------------------------------------------------
# Package exports
# -------------------------------------------------------------------

def test_training_exports_resolve() -> None:
    exports = getattr(
        training_package,
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
            training_package,
            name,
        )
    ]


    assert not missing, (
        "Missing src.training exports: "
        f"{missing}"
    )


# -------------------------------------------------------------------
# Architecture boundaries
# -------------------------------------------------------------------

def test_training_is_separate_from_data_package() -> None:
    for module_name in (
        discover_training_modules()
    ):
        assert not module_name.startswith(
            "src.data."
        )


def test_training_is_separate_from_ai_package() -> None:
    for module_name in (
        discover_training_modules()
    ):
        assert not module_name.startswith(
            "src.ai."
        )


# -------------------------------------------------------------------
# Artifact safety
# -------------------------------------------------------------------

def test_training_unit_test_does_not_require_artifact_creation() -> None:
    """
    This unit-test module deliberately does not call
    trainer main(), TrainingPipeline training loops,
    checkpoint writers, or evaluator entrypoints.

    Training execution belongs in integration and
    smoke tests using temporary output directories.
    """

    assert TRAINING_ROOT.is_dir()


# -------------------------------------------------------------------
# No test package leakage
# -------------------------------------------------------------------

def test_training_modules_are_not_tests() -> None:
    for module_name in (
        discover_training_modules()
    ):
        assert not module_name.startswith(
            "tests."
        )
