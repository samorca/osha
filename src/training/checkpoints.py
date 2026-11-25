from __future__ import annotations

"""
Checkpoint management for OSHA training.

Responsibilities:

    atomic checkpoint writes
    SHA-256 integrity verification
    checkpoint metadata persistence
    checkpoint registration and lookup
    best-checkpoint selection
    retention of recent checkpoints
    PyTorch checkpoint loading

A checkpoint is not automatically a promoted
production model. Promotion belongs in
src/training/artifacts.py.
"""

import json
import math
import os
import shutil
import tempfile

from pathlib import Path
from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.data.checksums import (
    sha256_file,
)

from src.training.config import (
    CheckpointConfig,
    TrainingConfig,
)

from src.training.errors import (
    CheckpointConflictError,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointLoadError,
    CheckpointNotFoundError,
    CheckpointSaveError,
)

from src.training.schema import (
    CheckpointState,
    TrainingCheckpoint,
    TrainingRun,
)


# ---------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------

try:
    import torch

except ImportError:
    torch = None


# ---------------------------------------------------------------
# Checkpoint filenames
# ---------------------------------------------------------------

CHECKPOINT_FILE_NAME = "checkpoint.pt"

METADATA_FILE_NAME = "checkpoint.json"


# ---------------------------------------------------------------
# Filesystem-safe name
# ---------------------------------------------------------------


def safe_checkpoint_name(
    value: str,
) -> str:
    output: List[str] = []

    for character in str(value):
        if character.isalnum() or character in {
            "-",
            "_",
            ".",
        }:
            output.append(character)

        else:
            output.append("_")

    normalized = "".join(output).strip("._")

    if not normalized:
        raise CheckpointError("Checkpoint identity cannot produce a safe path.")

    return normalized


# ---------------------------------------------------------------
# Controlled path resolution
# ---------------------------------------------------------------


def _inside(
    root: Path,
    path: Path,
) -> bool:
    try:
        path.resolve().relative_to(root.resolve())

        return True

    except ValueError:
        return False


def _require_inside(
    root: Path,
    path: Path,
) -> Path:
    resolved = path.expanduser().resolve()

    if not _inside(
        root,
        resolved,
    ):
        raise CheckpointError(
            "Checkpoint path escapes the configured checkpoint directory."
        )

    return resolved


# ---------------------------------------------------------------
# Checkpoint directory
# ---------------------------------------------------------------


def checkpoint_directory(
    root: str | Path,
    *,
    task: str,
    model_name: str,
    run_id: str,
    checkpoint_id: str,
) -> Path:
    root_path = Path(root).expanduser().resolve()

    return (
        root_path
        / safe_checkpoint_name(task)
        / safe_checkpoint_name(model_name)
        / safe_checkpoint_name(run_id)
        / safe_checkpoint_name(checkpoint_id)
    )


# ---------------------------------------------------------------
# Metadata serialization
# ---------------------------------------------------------------


def checkpoint_to_json(
    checkpoint: TrainingCheckpoint,
) -> str:
    checkpoint.finalize()

    try:
        return json.dumps(
            checkpoint.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise CheckpointSaveError(
            "Checkpoint metadata could not be serialized.",
            details={
                "checkpoint_id": (checkpoint.checkpoint_id),
                "error_type": (type(exc).__name__),
            },
        ) from exc


# ---------------------------------------------------------------
# Metadata parser
# ---------------------------------------------------------------


def checkpoint_from_dict(
    value: Mapping[
        str,
        Any,
    ],
) -> TrainingCheckpoint:
    try:
        checkpoint = TrainingCheckpoint(
            run_id=str(value["run_id"]),
            model_name=str(value["model_name"]),
            path=str(value["path"]),
            state=str(
                value.get(
                    "state",
                    CheckpointState.SAVED.value,
                )
            ),
            checkpoint_id=(
                str(value["checkpoint_id"])
                if value.get("checkpoint_id") is not None
                else None
            ),
            epoch=(int(value["epoch"]) if value.get("epoch") is not None else None),
            step=(int(value["step"]) if value.get("step") is not None else None),
            score=(float(value["score"]) if value.get("score") is not None else None),
            score_name=(
                str(value["score_name"])
                if value.get("score_name") is not None
                else None
            ),
            checksum=(
                str(value["checksum"]) if value.get("checksum") is not None else None
            ),
            attributes=dict(
                value.get(
                    "attributes",
                    {},
                )
            ),
            created_at=str(value.get("created_at", "")),
        )

        return checkpoint.finalize()

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise CheckpointLoadError(
            "Checkpoint metadata is invalid.",
            details={
                "error_type": (type(exc).__name__),
            },
        ) from exc


# ---------------------------------------------------------------
# Metadata load
# ---------------------------------------------------------------


def load_checkpoint_metadata(
    path: str | Path,
) -> TrainingCheckpoint:
    source = Path(path).expanduser()

    if source.is_dir():
        source = source / METADATA_FILE_NAME

    if not source.exists():
        raise CheckpointNotFoundError(
            "Checkpoint metadata was not found.",
            details={
                "path": str(source),
            },
        )

    if not source.is_file():
        raise CheckpointLoadError("Checkpoint metadata path is not a regular file.")

    try:
        value = json.loads(source.read_text(encoding="utf-8"))

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise CheckpointLoadError(
            "Checkpoint metadata could not be read.",
            details={
                "path": str(source),
                "error_type": (type(exc).__name__),
            },
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise CheckpointLoadError("Checkpoint metadata root must be an object.")

    return checkpoint_from_dict(value)


# ---------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------


class CheckpointRegistry:
    """
    Thread-safe in-memory checkpoint registry.
    """

    def __init__(
        self,
    ) -> None:
        self._checkpoints: Dict[
            str,
            TrainingCheckpoint,
        ] = {}

        self._lock = RLock()

    def register(
        self,
        checkpoint: TrainingCheckpoint,
        *,
        replace: bool = False,
    ) -> TrainingCheckpoint:
        checkpoint.finalize()

        checkpoint_id = str(checkpoint.checkpoint_id)

        with self._lock:
            if checkpoint_id in self._checkpoints and not replace:
                raise CheckpointConflictError(
                    "Checkpoint is already registered.",
                    details={
                        "checkpoint_id": (checkpoint_id),
                    },
                )

            self._checkpoints[checkpoint_id] = checkpoint

        return checkpoint

    def get(
        self,
        checkpoint_id: str,
    ) -> Optional[TrainingCheckpoint]:
        with self._lock:
            return self._checkpoints.get(str(checkpoint_id))

    def require(
        self,
        checkpoint_id: str,
    ) -> TrainingCheckpoint:
        checkpoint = self.get(checkpoint_id)

        if checkpoint is None:
            raise CheckpointNotFoundError(
                "Checkpoint is not registered.",
                details={
                    "checkpoint_id": (checkpoint_id),
                },
            )

        return checkpoint

    def remove(
        self,
        checkpoint_id: str,
    ) -> TrainingCheckpoint:
        with self._lock:
            checkpoint = self._checkpoints.pop(
                str(checkpoint_id),
                None,
            )

        if checkpoint is None:
            raise CheckpointNotFoundError("Checkpoint is not registered.")

        return checkpoint

    def list(
        self,
        *,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[TrainingCheckpoint]:
        with self._lock:
            values = list(self._checkpoints.values())

        output: List[TrainingCheckpoint] = []

        for checkpoint in values:
            if run_id is not None and checkpoint.run_id != run_id:
                continue

            if model_name is not None and checkpoint.model_name != model_name:
                continue

            if state is not None and checkpoint.state != state:
                continue

            output.append(checkpoint)

        return sorted(
            output,
            key=lambda item: (
                item.created_at,
                item.checkpoint_id or "",
            ),
        )

    def summary(
        self,
    ) -> Dict[str, Any]:
        values = self.list()

        by_state: Dict[
            str,
            int,
        ] = {}

        for checkpoint in values:
            by_state[checkpoint.state] = (
                by_state.get(
                    checkpoint.state,
                    0,
                )
                + 1
            )

        return {
            "checkpoint_count": len(values),
            "by_state": (by_state),
        }


# ---------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------


class CheckpointManager:
    """
    Save, load, verify, select, and retain
    training checkpoints.

    PyTorch model/optimizer state is stored in
    checkpoint.pt. Provider-neutral metadata is
    stored separately in checkpoint.json.
    """

    def __init__(
        self,
        *,
        config: Optional[TrainingConfig] = None,
        registry: Optional[CheckpointRegistry] = None,
    ) -> None:
        self.config = config or TrainingConfig()

        self.config.validate()

        self.registry = registry or CheckpointRegistry()

    # ---------------------------------------------------------
    # Root
    # ---------------------------------------------------------

    @property
    def root_directory(
        self,
    ) -> Path:
        return Path(self.config.paths.resolved()["checkpoints"])

    # ---------------------------------------------------------
    # Torch guard
    # ---------------------------------------------------------

    @staticmethod
    def _require_torch(
        self=None,
    ) -> None:
        if torch is None:
            raise CheckpointError("PyTorch is not installed.")

    # ---------------------------------------------------------
    # Build state payload
    # ---------------------------------------------------------

    def build_payload(
        self,
        *,
        run: TrainingRun,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        score: Optional[float] = None,
        score_name: Optional[str] = None,
        vocabulary: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        feature_schema: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        model_config: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        extra: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Dict[str, Any]:
        self._require_torch()

        run.finalize()

        state_dict = getattr(
            model,
            "state_dict",
            None,
        )

        if not callable(state_dict):
            raise CheckpointSaveError("Model does not expose state_dict().")

        if score is not None and not math.isfinite(float(score)):
            raise CheckpointSaveError("Checkpoint score must be finite.")

        payload: Dict[
            str,
            Any,
        ] = {
            "schema_version": "1",
            "run": run.to_dict(),
            "model_state_dict": (state_dict()),
            "epoch": epoch,
            "step": step,
            "score": score,
            "score_name": (score_name),
            "model_config": dict(model_config or {}),
            "vocabulary": dict(vocabulary or {}),
            "labels": dict(labels or {}),
            "feature_schema": dict(feature_schema or {}),
            "extra": dict(extra or {}),
        }

        if optimizer is not None and self.config.checkpoint.save_optimizer_state:
            optimizer_state = getattr(
                optimizer,
                "state_dict",
                None,
            )

            if callable(optimizer_state):
                payload["optimizer_state_dict"] = optimizer_state()

        if scheduler is not None:
            scheduler_state = getattr(
                scheduler,
                "state_dict",
                None,
            )

            if callable(scheduler_state):
                payload["scheduler_state_dict"] = scheduler_state()

        if scaler is not None:
            scaler_state = getattr(
                scaler,
                "state_dict",
                None,
            )

            if callable(scaler_state):
                payload["scaler_state_dict"] = scaler_state()

        return payload

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(
        self,
        *,
        run: TrainingRun,
        model: Any,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        score: Optional[float] = None,
        score_name: Optional[str] = None,
        vocabulary: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        feature_schema: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        model_config: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        extra: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        checkpoint_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> TrainingCheckpoint:
        """
        Save one checkpoint through a staging
        directory, then publish it by rename.
        """

        self._require_torch()

        run.finalize()

        resolved_model_name = model_name or run.model_name

        if not resolved_model_name:
            raise CheckpointSaveError("Checkpoint requires a model name.")

        resolved_task = task or run.task

        provisional = TrainingCheckpoint(
            run_id=str(run.run_id),
            model_name=(resolved_model_name),
            path="pending",
            state=(CheckpointState.CREATED.value),
            checkpoint_id=(checkpoint_id),
            epoch=epoch,
            step=step,
            score=score,
            score_name=(score_name or self.config.checkpoint.monitor_metric),
            attributes={
                "task": (resolved_task),
            },
        ).finalize()

        destination = checkpoint_directory(
            self.root_directory,
            task=resolved_task,
            model_name=(resolved_model_name),
            run_id=str(run.run_id),
            checkpoint_id=str(provisional.checkpoint_id),
        )

        if destination.exists():
            if not overwrite:
                raise CheckpointConflictError(
                    "Checkpoint destination already exists.",
                    details={
                        "checkpoint_id": (provisional.checkpoint_id),
                        "destination": (str(destination)),
                    },
                )

            shutil.rmtree(destination)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=(
                    "." + safe_checkpoint_name(str(provisional.checkpoint_id)) + "-"
                ),
                suffix=".saving",
                dir=str(destination.parent),
            )
        )

        checkpoint_path = staging / CHECKPOINT_FILE_NAME

        try:
            payload = self.build_payload(
                run=run,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=step,
                score=score,
                score_name=(provisional.score_name),
                vocabulary=vocabulary,
                labels=labels,
                feature_schema=(feature_schema),
                model_config=(model_config),
                extra=extra,
            )

            torch.save(
                payload,
                checkpoint_path,
            )

            checksum = sha256_file(checkpoint_path)

            final_checkpoint = TrainingCheckpoint(
                run_id=(provisional.run_id),
                model_name=(provisional.model_name),
                path=str(destination / CHECKPOINT_FILE_NAME),
                state=(CheckpointState.SAVED.value),
                checkpoint_id=(provisional.checkpoint_id),
                epoch=epoch,
                step=step,
                score=score,
                score_name=(provisional.score_name),
                checksum=checksum,
                attributes={
                    **provisional.attributes,
                    "metadata_path": str(destination / METADATA_FILE_NAME),
                    "optimizer_state_saved": (
                        optimizer is not None
                        and self.config.checkpoint.save_optimizer_state
                    ),
                },
                created_at=(provisional.created_at),
            ).finalize()

            metadata_path = staging / METADATA_FILE_NAME

            metadata_path.write_text(
                checkpoint_to_json(final_checkpoint) + "\n",
                encoding="utf-8",
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            staging.replace(destination)

        except (
            CheckpointError,
            CheckpointSaveError,
        ):
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

            raise

        except Exception as exc:
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

            raise CheckpointSaveError(
                "Checkpoint could not be saved.",
                details={
                    "run_id": (run.run_id),
                    "model_name": (resolved_model_name),
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

        self.registry.register(
            final_checkpoint,
            replace=overwrite,
        )

        if final_checkpoint.checkpoint_id not in run.checkpoint_ids:
            run.checkpoint_ids.append(str(final_checkpoint.checkpoint_id))

        return final_checkpoint

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    def verify(
        self,
        checkpoint: TrainingCheckpoint,
    ) -> bool:
        checkpoint.finalize()

        path = _require_inside(
            self.root_directory,
            Path(checkpoint.path),
        )

        if not path.exists():
            raise CheckpointNotFoundError(
                "Checkpoint state file was not found.",
                details={
                    "checkpoint_id": (checkpoint.checkpoint_id),
                    "path": str(path),
                },
            )

        if not path.is_file():
            raise CheckpointIntegrityError(
                "Checkpoint state path is not a regular file."
            )

        if not checkpoint.checksum:
            raise CheckpointIntegrityError(
                "Checkpoint has no recorded checksum.",
                details={
                    "checkpoint_id": (checkpoint.checkpoint_id),
                },
            )

        actual = sha256_file(path)

        valid = actual.lower() == checkpoint.checksum.lower()

        if not valid:
            raise CheckpointIntegrityError(
                "Checkpoint checksum verification failed.",
                details={
                    "checkpoint_id": (checkpoint.checkpoint_id),
                    "expected_checksum": (checkpoint.checksum),
                    "actual_checksum": (actual),
                },
            )

        return True

    # ---------------------------------------------------------
    # Load raw payload
    # ---------------------------------------------------------

    def load(
        self,
        checkpoint: (TrainingCheckpoint | str),
        *,
        map_location: Any = "cpu",
        verify: bool = True,
        weights_only: bool = False,
    ) -> Dict[str, Any]:
        self._require_torch()

        if isinstance(
            checkpoint,
            str,
        ):
            checkpoint_value = self.registry.require(checkpoint)

        else:
            checkpoint_value = checkpoint

        checkpoint_value.finalize()

        if verify:
            self.verify(checkpoint_value)

        path = Path(checkpoint_value.path)

        try:
            try:
                payload = torch.load(
                    path,
                    map_location=(map_location),
                    weights_only=(weights_only),
                )

            except TypeError:
                # Compatibility with older
                # PyTorch versions that do not
                # expose weights_only.
                payload = torch.load(
                    path,
                    map_location=(map_location),
                )

        except Exception as exc:
            raise CheckpointLoadError(
                "Checkpoint state could not be loaded.",
                details={
                    "checkpoint_id": (checkpoint_value.checkpoint_id),
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

        if not isinstance(
            payload,
            Mapping,
        ):
            raise CheckpointLoadError("Checkpoint payload root must be an object.")

        return dict(payload)

    # ---------------------------------------------------------
    # Restore objects
    # ---------------------------------------------------------

    def restore(
        self,
        checkpoint: (TrainingCheckpoint | str),
        *,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        map_location: Any = "cpu",
        strict: bool = True,
        verify: bool = True,
    ) -> Dict[str, Any]:
        payload = self.load(
            checkpoint,
            map_location=(map_location),
            verify=verify,
            weights_only=False,
        )

        model_state = payload.get("model_state_dict")

        if model_state is None:
            raise CheckpointLoadError("Checkpoint contains no model state.")

        load_model = getattr(
            model,
            "load_state_dict",
            None,
        )

        if not callable(load_model):
            raise CheckpointLoadError("Model does not expose load_state_dict().")

        load_model(
            model_state,
            strict=strict,
        )

        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])

        return payload

    # ---------------------------------------------------------
    # Score comparison
    # ---------------------------------------------------------

    def _better(
        self,
        candidate: TrainingCheckpoint,
        current: TrainingCheckpoint,
    ) -> bool:
        if candidate.score is None:
            return False

        if current.score is None:
            return True

        mode = self.config.checkpoint.mode

        if mode == "min":
            return candidate.score < current.score

        return candidate.score > current.score

    # ---------------------------------------------------------
    # Best checkpoint
    # ---------------------------------------------------------

    def best(
        self,
        *,
        run_id: str,
        score_name: Optional[str] = None,
        mark: bool = False,
    ) -> TrainingCheckpoint:
        metric = score_name or self.config.checkpoint.monitor_metric

        candidates = [
            checkpoint
            for checkpoint in self.registry.list(run_id=run_id)
            if (checkpoint.score is not None and checkpoint.score_name == metric)
        ]

        if not candidates:
            raise CheckpointNotFoundError(
                "No scored checkpoint was found for the run.",
                details={
                    "run_id": run_id,
                    "score_name": metric,
                },
            )

        selected = candidates[0]

        for candidate in candidates[1:]:
            if self._better(
                candidate,
                selected,
            ):
                selected = candidate

        if mark:
            selected.state = CheckpointState.BEST.value

            self._rewrite_metadata(selected)

        return selected

    # ---------------------------------------------------------
    # Rewrite metadata
    # ---------------------------------------------------------

    def _rewrite_metadata(
        self,
        checkpoint: TrainingCheckpoint,
    ) -> None:
        checkpoint.finalize()

        checkpoint_path = Path(checkpoint.path)

        metadata_path = checkpoint_path.parent / METADATA_FILE_NAME

        temporary = metadata_path.with_name(metadata_path.name + ".writing")

        try:
            temporary.write_text(
                checkpoint_to_json(checkpoint) + "\n",
                encoding="utf-8",
            )

            temporary.replace(metadata_path)

        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

            raise CheckpointSaveError(
                "Checkpoint metadata could not be updated.",
                details={
                    "checkpoint_id": (checkpoint.checkpoint_id),
                    "error_type": (type(exc).__name__),
                },
            ) from exc

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(
        self,
        checkpoint: (TrainingCheckpoint | str),
    ) -> None:
        if isinstance(
            checkpoint,
            str,
        ):
            value = self.registry.require(checkpoint)

        else:
            value = checkpoint

        value.finalize()

        path = _require_inside(
            self.root_directory,
            Path(value.path).parent,
        )

        try:
            shutil.rmtree(path)

        except FileNotFoundError:
            pass

        except OSError as exc:
            raise CheckpointError(
                "Checkpoint directory could not be removed.",
                details={
                    "checkpoint_id": (value.checkpoint_id),
                    "error_type": (type(exc).__name__),
                },
            ) from exc

        if value.checkpoint_id:
            try:
                self.registry.remove(value.checkpoint_id)
            except CheckpointNotFoundError:
                pass

    # ---------------------------------------------------------
    # Retention
    # ---------------------------------------------------------

    def apply_retention(
        self,
        *,
        run_id: str,
    ) -> List[str]:
        """
        Keep the configured number of latest
        checkpoints plus the selected best
        checkpoint when keep_best is enabled.
        """

        checkpoints = self.registry.list(run_id=run_id)

        if len(checkpoints) <= self.config.checkpoint.keep_last:
            return []

        ordered = sorted(
            checkpoints,
            key=lambda item: (
                item.epoch if item.epoch is not None else -1,
                item.step if item.step is not None else -1,
                item.created_at,
            ),
            reverse=True,
        )

        retained_ids = {
            str(checkpoint.checkpoint_id)
            for checkpoint in ordered[: self.config.checkpoint.keep_last]
        }

        if self.config.checkpoint.keep_best:
            try:
                best = self.best(run_id=run_id)

                retained_ids.add(str(best.checkpoint_id))

            except CheckpointNotFoundError:
                pass

        removed: List[str] = []

        for checkpoint in ordered:
            checkpoint_id = str(checkpoint.checkpoint_id)

            if checkpoint_id in retained_ids:
                continue

            self.delete(checkpoint)

            removed.append(checkpoint_id)

        return removed

    # ---------------------------------------------------------
    # Discover existing checkpoints
    # ---------------------------------------------------------

    def discover(
        self,
    ) -> List[TrainingCheckpoint]:
        root = self.root_directory

        if not root.exists():
            return []

        discovered: List[TrainingCheckpoint] = []

        for metadata_path in sorted(
            root.rglob(METADATA_FILE_NAME),
            key=lambda item: str(item),
        ):
            try:
                metadata_value = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                metadata_value = None

            if (
                isinstance(metadata_value, Mapping)
                and metadata_value.get("model_type")
                == "osha.capacity.autoregressive-ridge"
            ):
                continue

            checkpoint = load_checkpoint_metadata(metadata_path)

            checkpoint_path = metadata_path.parent / CHECKPOINT_FILE_NAME

            checkpoint.path = str(checkpoint_path)

            checkpoint.finalize()

            self.registry.register(
                checkpoint,
                replace=True,
            )

            discovered.append(checkpoint)

        return discovered

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "root_directory": str(self.root_directory),
            "enabled": (self.config.checkpoint.enabled),
            "save_every_epochs": (self.config.checkpoint.save_every_epochs),
            "keep_last": (self.config.checkpoint.keep_last),
            "keep_best": (self.config.checkpoint.keep_best),
            "monitor_metric": (self.config.checkpoint.monitor_metric),
            "mode": (self.config.checkpoint.mode),
            "save_optimizer_state": (self.config.checkpoint.save_optimizer_state),
            "registry": (self.registry.summary()),
            "checkpoint_is_promoted_model": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "CHECKPOINT_FILE_NAME",
    "METADATA_FILE_NAME",
    "safe_checkpoint_name",
    "checkpoint_directory",
    "checkpoint_to_json",
    "checkpoint_from_dict",
    "load_checkpoint_metadata",
    "CheckpointRegistry",
    "CheckpointManager",
]
