from __fu*ure__ import annotations

"""
Trai* the OSHA structured anomaly model*

Hardware discovery and device se*ection are
delegated to:

    src.*ccelerators.AcceleratorManager

Th*s module owns the anomaly-specific*PyTorch
optimization loop.

Input *ecords must contain a numeric feat*re
vector under one of:

    featu*es
    feature_vector
    values
 *  vector
    x

and a binary label*under one of:

    label
    targe*
    anomaly_label
    is_anomaly
*   anomaly
"""

import argparse
im*ort hashlib
import json
import mat*
import shutil

from pathlib impor* Path

from typing import (
    An*,
    Dict,
    Iterable,
    List*
    Mapping,
    Optional,
    Tu*le,
)

import torch

from torch im*ort nn
from torch.optim import Ada*W
from torch.utils.data import Dat*Loader

from src.accelerators impo*t (
    AcceleratorConfig,
    Acc*leratorManager,
    AcceleratorPre*ision,
    AcceleratorSelection,
 *  AcceleratorSelectionConfig,
)

f*om src.training.anomaly_dataset im*ort (
    ANOMALY_LABEL,
    IGNOR*_INDEX,
    NORMAL_LABEL,
    Anom*lyDataset,
    anomaly_dataset_fro*_records,
    load_anomaly_records*
)

from src.training.anomaly_mode* import (
    InfrastructureAnomal*Model,
)

from src.training.config*import (
    AcceleratorConfig as *rainingAcceleratorConfig,
    Chec*pointConfig,
    OptimizationConfi*,
    ReproducibilityConfig,
    T*ainingConfig,
    TrainingPathConf*g,
)

from src.training.errors imp*rt (
    EmptyTrainingDatasetError*
    TrainingNumericalError,
    T*ainingRunError,
)

from src.traini*g.pipeline import (
    TrainingPi*eline,
)

from src.training.schema*import (
    EvaluationSplit,
    *rainingState,
    TrainingTask,
)
*
# -------------------------------*-------------------------------
# *LI
# -----------------------------*---------------------------------
*def parse_args() -> argparse.Names*ace:
    parser = argparse.Argumen*Parser(
        description=(
    *       "Train the OSHA structured *
            "infrastructure anoma*y model."
        )
    )

    par*er.add_argument(
        "--data",*        default=(
            "orc*estration-ai-data/"
            "p*ocessed/anomalies.jsonl"
        )*
    )

    parser.add_argument(
 *      "--output",
        default=*
            "artifacts/models/"
 *          "anomaly-detection-v1"
 *      ),
    )

    parser.add_arg*ment(
        "--model-name",
    *   default=(
            "infrastr*cture-anomaly-model"
        ),
  * )

    parser.add_argument(
     *  "--dataset-id",
        default=*
            "osha-infrastructure-*nomalies"
        ),
    )

    pa*ser.add_argument(
        "--epoch*",
        type=int,
        defau*t=20,
    )

    parser.add_argume*t(
        "--batch-size",
       *type=int,
        default=64,
    *

    parser.add_argument(
       *"--learning-rate",
        type=fl*at,
        default=1e-3,
    )

 *  parser.add_argument(
        "--*eight-decay",
        type=float,
*       default=0.0,
    )

    par*er.add_argument(
        "--hidden*dim",
        type=int,
        de*ault=256,
    )

    parser.add_ar*ument(
        "--representation-d*m",
        type=int,
        defa*lt=128,
    )

    parser.add_argu*ent(
        "--dropout",
        *ype=float,
        default=0.20,
 *  )

    parser.add_argument(
    *   "--gradient-clip-norm",
       *type=float,
        default=1.0,
 *  )

    parser.add_argument(
    *   "--threshold",
        type=flo*t,
        default=0.50,
    )

  * parser.add_argument(
        "--t*ain-ratio",
        type=float,
  *     default=0.80,
    )

    pars*r.add_argument(
        "--validat*on-ratio",
        type=float,
   *    default=0.10,
    )

    parse*.add_argument(
        "--max-reco*ds",
        type=int,
        def*ult=None,
    )

    parser.add_ar*ument(
        "--num-workers",
  *     type=int,
        default=2,
*   )

    parser.add_argument(
   *    "--seed",
        type=int,
  *     default=42,
    )

    parser*add_argument(
        "--accelerat*r",
        choices=(
            *auto",
            "cuda",
       *    "rocm",
            "cpu",
   *    ),
        default="auto",
   *)

    parser.add_argument(
      * "--device-index",
        type=in*,
        default=None,
    )

   *parser.add_argument(
        "--no*cpu-fallback",
        action="sto*e_true",
    )

    parser.add_arg*ment(
        "--mixed-precision",*        action="store_true",
    )*
    parser.add_argument(
        *--minimum-free-memory-gb",
       *type=float,
        default=None,
*   )

    parser.add_argument(
   *    "--pin-memory",
        action*"store_true",
    )

    parser.ad*_argument(
        "--keep-last",
*       type=int,
        default=5*
    )

    parser.add_argument(
 *      "--disable-class-weight",
  *     action="store_true",
    )

 *  return parser.parse_args()


# -*----------------------------------*--------------------------
# Argum*nt validation
# ------------------*----------------------------------*---------

def validate_args(
    *rgs: argparse.Namespace,
) -> None*
    if args.epochs < 1:
        r*ise ValueError(
            "--epo*hs must be >= 1"
        )

    if*args.batch_size < 1:
        raise*ValueError(
            "--batch-s*ze must be >= 1"
        )

    if*args.learning_rate <= 0:
        r*ise ValueError(
            "--lea*ning-rate must be > 0"
        )

*   if args.weight_decay < 0:
     *  raise ValueError(
            "-*weight-decay must be >= 0"
       *)

    if args.hidden_dim < 1:
   *    raise ValueError(
            *--hidden-dim must be >= 1"
       *)

    if args.representation_dim * 1:
        raise ValueError(
    *       "--representation-dim must *e >= 1"
        )

    if not 0.0 *= args.dropout < 1.0:
        rais* ValueError(
            "--dropou* must be in [0, 1)"
        )

   *if args.gradient_clip_norm <= 0:
 *      raise ValueError(
          * "--gradient-clip-norm must be > 0*
        )

    if not 0.0 <= args*threshold <= 1.0:
        raise Va*ueError(
            "--threshold *ust be in [0, 1]"
        )

    i* not 0.0 < args.train_ratio < 1.0:*        raise ValueError(
        *   "--train-ratio must be between * and 1"
        )

    if not 0.0 *= args.validation_ratio < 1.0:
   *    raise ValueError(
            *--validation-ratio must be in [0, 1)"
        )

    if (
        arg*.train_ratio
        + args.valida*ion_ratio
        >= 1.0
    ):
  *     raise ValueError(
           *"--train-ratio plus "
            *--validation-ratio must be less th*n 1"
        )

    if args.num_wo*kers < 0:
        raise ValueError*
            "--num-workers must b* >= 0"
        )

    if args.keep*last < 1:
        raise ValueError*
            "--keep-last must be *= 1"
        )

    if (
        a*gs.max_records is not None
       *and args.max_records < 1
    ):
  *     raise ValueError(
           *"--max-records must be >= 1"
     *  )

    if (
        args.minimum*free_memory_gb is not None
       *and args.minimum_free_memory_gb < *
    ):
        raise ValueError(
*           "--minimum-free-memory-*b must be >= 0"
        )

    if *
        args.mixed_precision
    *   and args.accelerator == "cpu"
 *  ):
        raise ValueError(
   *        "--mixed-precision cannot *e used "
            "with an expl*citly selected CPU"
        )


# *----------------------------------*---------------------------
# Trai*ing configuration
# --------------*----------------------------------*-------------

def build_training_*onfig(
    args: argparse.Namespac*,
) -> TrainingConfig:
    config * TrainingConfig(
        enabled=T*ue,

        allowed_tasks=(
     *      TrainingTask.EVENT.value,
  *         TrainingTask.ANOMALY.valu*,
        ),

        paths=Traini*gPathConfig(),

        reproducib*lity=ReproducibilityConfig(
      *     random_seed=args.seed,
      *     deterministic=True,
         *  benchmark=False,
        ),

   *    optimization=OptimizationConfi*(
            batch_size=args.batc*_size,
            epochs=args.epo*hs,
            learning_rate=(
  *             args.learning_rate
  *         ),
            weight_dec*y=(
                args.weight_de*ay
            ),
            grad*ent_clip_norm=(
                ar*s.gradient_clip_norm
            )*
            accumulation_steps=1,*        ),

        accelerator=Tr*iningAcceleratorConfig(
          * preferred=args.accelerator,
     *      allow_fallback_to_cpu=(
    *           not args.no_cpu_fallbac*
            ),
            mixed_*recision=(
                args.mi*ed_precision
            ),
      *     device_index=(
              * args.device_index
            ),
*       ),

        checkpoint=Chec*pointConfig(
            enabled=T*ue,
            save_every_epochs=*,
            keep_last=args.keep_*ast,
            keep_best=True,
 *          monitor_metric=(
       *        "validation_loss"
        *   ),
            mode="min",
    *       save_optimizer_state=True,
*       ),

        attributes={
  *         "trainer": "train_anomali*s",
            "accelerator_provi*er": (
                "src.accele*ators"
            ),
        },
 *  )

    config.validate()

    re*urn config


# -------------------*----------------------------------*--------
# Accelerator configurati*n
# ------------------------------*--------------------------------

*ef build_accelerator_manager(
    *rgs: argparse.Namespace,
) -> Acce*eratorManager:
    config = Accele*atorConfig(
        selection=Acce*eratorSelectionConfig(
           *preferred_kind=(
                a*gs.accelerator
            ),
    *       device_index=(
            *   args.device_index
            )*
            allow_cpu_fallback=(
*               not args.no_cpu_fal*back
            ),
            al*ow_degraded_devices=True,
        *   require_accelerator=(
         *      args.no_cpu_fallback
       *        and args.accelerator
     *          in {
                   *"cuda",
                    "rocm"*
                }
            ),
*           prefer_highest_free_mem*ry=True,
        ),
        attrib*tes={
            "consumer": (
  *             "training.train_anoma*ies"
            ),
        },
   *)

    config.validate()

    retu*n AcceleratorManager(
        conf*g=config
    )


def minimum_free_*emory_bytes(
    args: argparse.Na*espace,
) -> Optional[int]:
    if*args.minimum_free_memory_gb is Non*:
        return None

    return *nt(
        args.minimum_free_memo*y_gb
        * 1024**3
    )


def***lect_accelerator(
    manager: A***leratorManager,
    args: argpar***Namespace,
) -> AcceleratorSelec***n:
    required_precisions = ()
***  if args.mixed_precision:
     ***required_precisions = (
        *** AcceleratorPrecision.
         ***FLOAT16.value,
        )

    re***n manager.select_for_training(
 ***    preferred_kind=args.accelera***,
        device_index=args.devi***index,
        minimum_free_memo***bytes=(
            minimum_free***mory_bytes(
                args***          )
        ),
        r***ire_accelerator=(
            ar***no_cpu_fallback
            and ***s.accelerator
            in {
 ***            "cuda",
            *** "rocm",
            }
        )***       allow_cpu_fallback=(
    ***     not args.no_cpu_fallback
  ***   ),
        mixed_precision=(
***         args.mixed_precision
  ***   ),
        required_precision***
            required_precisions***      ),
        refresh=True,
 ***)


# --------------------------***--------------------------------***# Deterministic record splitting***--------------------------------***----------------------------

de***nomaly_record_key(
    record: M***ing[
        str,
        Any,
    ],
) -> str:
    for field in (
***     "group_id",
        "incide***id",
        "sequence_id",
    *** "session_id",
        "trace_id***        "window_id",
        "re***d_id",
        "example_id",
   ***
        value = record.get(
   ***      field
        )

        i***alue is None:
            contin***
        text = str(
           ***lue
        ).strip()

        i***ext:
            return (
      ***       field
                + "***                + text
         ***)

    serialized = json.dumps(
***     dict(record),
        sort_***s=True,
        ensure_ascii=Fal***
        separators=(
          ***,",
            ":",
        ),
*** )

    return (
        "record***        + hashlib.sha256(
      ***   serialized.encode(
          ***   "utf-8"
            )
       ***hexdigest()
    )


def split_re***ds(
    records: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
*** train_ratio: float,
    validat***_ratio: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    train: List[
        Dict[str, Any]
    ] = []

    validati*** List[
        Dict[str, Any]
    ] = []

    test: List[
        Dict[str, Any]
    ] = []

    train***toff = int(
        train_ratio ***0000
    )

    validation_cutof*** int(
        (
            trai***atio
            + validation_ra***
        )
        * 10000
    )***   for source_record in records:***      record = dict(
           ***urce_record
        )

        d***st = hashlib.sha256(
           ***omaly_record_key(
              ***ecord
            ).encode(
    ***         "utf-8"
            )
 ***    ).hexdigest()

        bucke*** (
            int(
            *** digest[:8],
                16,***          )
            % 10000
***     )

        if bucket < trai***utoff:
            train.append(***              record
           ***
        elif bucket < validatio***utoff:
            validation.ap***d(
                record
      ***   )

        else:
            ***t.append(
                record***          )

    if not train:
 ***    raise EmptyTrainingDatasetEr***(
            "Anomaly split pro***ed an "
            "empty train*** partition."
        )

    retu***(
        train,
        validat***,
        test,
    )


# ------***--------------------------------***-------------------
# DataLoader***--------------------------------***----------------------------

de***uild_loader(
    dataset: Option***
        AnomalyDataset
    ],
 ****,
    batch_size: int,
    shuf***: bool,
    num_workers: int,
  ***in_memory: bool,
) -> Optional[DataLoader]:
    if dataset is None:***      return None

    return Da***oader(
        dataset,
        ***ch_size=batch_size,
        shuf***=shuffle,
        num_workers=nu***orkers,
        pin_memory=pin_m***ry,
        persistent_workers=(***          num_workers > 0
      ***,
    )


# --------------------***--------------------------------***-----
# Class weighting
# ------***--------------------------------***-------------------

def positiv***lass_weight(
    dataset: Anomal***taset,
) -> float:
    normal_co*** = sum(
        1
        for sa***e
        in dataset.samples
        if sample.label
        == NORMAL_LABEL
    )

    anomaly_count = sum(
        1
        for sample
        in dataset.samples
        if sample.label
        == ANOMALY_LABEL
    )

    if anomaly_count == 0:
        raise EmptyTrainingDatasetError(
            "Training data contains no "
            "anomaly examples."
        )

    if normal_count == 0:
        raise EmptyTrainingDatasetError(
            "Training data contains no "
            "normal examples."
        )

    return (
        float(normal_count)
        / float(anomaly_count)
    )


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def confusion_counts(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
) -> Dict[str, int]:
    valid = (
        targets != IGNORE_INDEX
    )

    count = int(
        valid.sum().item()
    )

    if count == 0:
        return {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "count": 0,
        }

    predictions = (
        probabilities[valid]
        >= threshold
    ).to(
        torch.long
    )

    expected = targets[valid].to(
        torch.long
    )

    return {
        "true_positive": int(
            (
                (
                    predictions
                    == ANOMALY_LABEL
                )
                & (
                    expected
                    == ANOMALY_LABEL
                )
            ).sum().item()
        ),

        "true_negative": int(
            (
                (
                    predictions
                    == NORMAL_LABEL
                )
                & (
                    expected
                    == NORMAL_LABEL
                )
            ).sum().item()
        ),

        "false_positive": int(
            (
                (
                    predictions
                    == ANOMALY_LABEL
                )
                & (
                    expected
                    == NORMAL_LABEL
                )
            ).sum().item()
        ),

        "false_negative": int(
            (
                (
                    predictions
                    == NORMAL_LABEL
                )
                & (
                    expected
                    == ANOMALY_LABEL
                )
            ).sum().item()
        ),

        "count": count,
    }


def finalize_metrics(
    *,
    total_loss: float,
    batch_count: int,
    example_count: int,
    confusion: Mapping[
        str,
        int,
    ],
    threshold: float,
) -> Dict[str, Any]:
    tp = int(
        confusion["true_positive"]
    )

    tn = int(
        confusion["true_negative"]
    )

    fp = int(
        confusion["false_positive"]
    )

    fn = int(
        confusion["false_negative"]
    )

    count = int(
        confusion["count"]
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else None
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision is not None
            and recall is not None
            and precision + recall > 0
        )
        else None
    )

    return {
        "loss": (
            total_loss / batch_count
            if batch_count
            else None
        ),

        "batch_count": batch_count,

        "example_count": (
            example_count
        ),

        "evaluated_count": count,

        "threshold": threshold,

        "accuracy": (
            (tp + tn) / count
            if count
            else None
        ),

        "precision": precision,

        "recall": recall,

        "specificity": (
            tn / (tn + fp)
            if tn + fp
            else None
        ),

        "f1": f1,

        "false_positive_rate": (
            fp / (fp + tn)
            if fp + tn
            else None
        ),

        "false_negative_rate": (
            fn / (fn + tp)
            if fn + tp
            else None
        ),

        "true_positive": tp,

        "true_negative": tn,

        "false_positive": fp,

        "false_negative": fn,
    }


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate(
    model: InfrastructureAnomalyModel,
    loader: Optional[DataLoader],
    *,
    device: Any,
    criterion: nn.Module,
    threshold: float,
) -> Dict[str, Any]:
    if loader is None:
        return finalize_metrics(
            total_loss=0.0,
            batch_count=0,
            example_count=0,
            confusion={
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
                "count": 0,
            },
            threshold=threshold,
        )

    model.eval()

    total_loss = 0.0
    batch_count = 0
    example_count = 0

    confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "count": 0,
    }

    with torch.inference_mode():
        for batch in loader:
            features = batch[
                "features"
            ].to(
                device,
                non_blocking=True,
            )

            targets = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            valid = (
                targets != IGNORE_INDEX
            )

            if int(
                valid.sum().item()
            ) == 0:
                continue

            output = model(
                features
            )

            logits = output[
                "anomaly_logits"
            ]

            loss = criterion(
                logits[valid],
                targets[valid].float(),
            )

            if not torch.isfinite(
                loss
            ):
                raise TrainingNumericalError(
                    "Anomaly evaluation "
                    "produced a non-finite loss."
                )

            total_loss += float(
                loss.item()
            )

            batch_count += 1

            example_count += int(
                features.shape[0]
            )

            values = confusion_counts(
                output[
                    "anomaly_probability"
                ],
                targets,
                threshold=threshold,
            )

            for key in confusion:
                confusion[key] += (
                    values[key]
                )

    return finalize_metrics(
        total_loss=total_loss,
        batch_count=batch_count,
        example_count=example_count,
        confusion=confusion,
        threshold=threshold,
    )


# ---------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------

def write_json_atomic(
    path: Path,
    value: Mapping[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".writing"
    )

    try:
        temporary.write_text(
            json.dumps(
                dict(value),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        raise


def copy_checkpoint_file(
    source: str | Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name + ".copying"
    )

    shutil.copy2(
        source,
        temporary,
    )

    temporary.replace(
        destination
    )


# ---------------------------------------------------------------
# Register evaluation
# ---------------------------------------------------------------

def register_evaluation(
    pipeline: TrainingPipeline,
    run: Any,
    *,
    split: str,
    metrics: Mapping[
        str,
        Any,
    ],
):
    excluded = {
        "example_count",
        "evaluated_count",
        "batch_count",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "threshold",
    }

    numeric = {
        str(name): float(value)
        for name, value
        in metrics.items()
        if (
            value is not None
            and isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            and name not in excluded
            and math.isfinite(
                float(value)
            )
        )
    }

    if not numeric:
        return None

    return pipeline.record_evaluation(
        run=run,
        split=split,
        metrics=numeric,
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    training_config = (
        build_training_config(
            args
        )
    )

    pipeline = TrainingPipeline(
        config=training_config
    )

    pipeline.start(
        discover_existing=True
    )

    pipeline.apply_reproducibility()

    accelerator_manager = (
        build_accelerator_manager(
            args
        )
    )

    accelerator_selection = (
        select_accelerator(
            accelerator_manager,
            args,
        )
    )

    device = (
        accelerator_manager.
        runtime_device(
            accelerator_selection
        )
    )

    output_directory = Path(
        args.output
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    run = None

    try:
        records = load_anomaly_records(
            args.data,
            max_records=(
                args.max_records
            ),
            require_label=True,
        )

        (
            train_records,
            validation_records,
            test_records,
        ) = split_records(
            records,
            train_ratio=(
                args.train_ratio
            ),
            validation_ratio=(
                args.validation_ratio
            ),
        )

        train_dataset = (
            anomaly_dataset_from_records(
                train_records
            )
        )

        validation_dataset = (
            anomaly_dataset_from_records(
                validation_records
            )
            if validation_records
            else None
        )

        test_dataset = (
            anomaly_dataset_from_records(
                test_records
            )
            if test_records
            else None
        )

        feature_dimension = (
            train_dataset.
            feature_dimension
        )

        for name, dataset in (
            (
                "validation",
                validation_dataset,
            ),
            (
                "test",
                test_dataset,
            ),
        ):
            if (
                dataset is not None
                and dataset.
                feature_dimension
                != feature_dimension
            ):
                raise TrainingRunError(
                    "Anomaly dataset feature "
                    "dimension mismatch.",
                    details={
                        "split": name,
                        "train_dimension": (
                            feature_dimension
                        ),
                        "split_dimension": (
                            dataset.
                            feature_dimension
                        ),
                    },
                )

        pin_memory = bool(
            args.pin_memory
            and accelerator_selection.
            device.accelerator
        )

        train_loader = build_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=(
                args.num_workers
            ),
            pin_memory=pin_memory,
        )

        validation_loader = build_loader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=(
                args.num_workers
            ),
            pin_memory=pin_memory,
        )

        test_loader = build_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=(
                args.num_workers
            ),
            pin_memory=pin_memory,
        )

        model = InfrastructureAnomalyModel(
            input_dim=feature_dimension,
            hidden_dim=args.hidden_dim,
            representation_dim=(
                args.representation_dim
            ),
            dropout=args.dropout,
        ).to(
            device
        )

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=(
                args.weight_decay
            ),
        )

        class_weight = (
            1.0
            if args.disable_class_weight
            else positive_class_weight(
                train_dataset
            )
        )

        positive_weight = torch.tensor(
            class_weight,
            dtype=torch.float32,
            device=device,
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=(
                    positive_weight
                )
            )
        )

        use_amp = bool(
            args.mixed_precision
            and accelerator_selection.
            device.accelerator
        )

        scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=use_amp,
        )

        model_config = {
            **model.model_config(),

            "temporary_threshold": (
                args.threshold
            ),

            "positive_class_weight": (
                class_weight
            ),

            "feature_source": (
                "legacy_numeric_jsonl"
            ),
        }

        run = pipeline.create_run(
            task=(
                TrainingTask.ANOMALY.value
            ),
            dataset_id=(
                args.dataset_id
            ),
            model_name=(
                args.model_name
            ),
            accelerator=(
                accelerator_selection.
                runtime_device
            ),
            attributes={
                "data_path": str(
                    Path(
                        args.data
                    ).expanduser().
                    resolve()
                ),
                "output_directory": str(
                    output_directory
                ),
                "record_count": len(
                    records
                ),
                "feature_dimension": (
                    feature_dimension
                ),
                "accelerator_selection": (
                    accelerator_selection.
                    to_dict()
                ),
            },
        )

        pipeline.transition_run(
            run,
            TrainingState.READY.value,
        )

        pipeline.transition_run(
            run,
            TrainingState.TRAINING.value,
        )

        print(
            "Accelerator:",
            accelerator_selection.
            device_id,
        )

        print(
            "Runtime device:",
            device,
        )

        print(
            "Selection reason:",
            accelerator_selection.
            reason,
        )

        print(
            "CPU fallback used:",
            accelerator_selection.
            fallback_used,
        )

        print(
            "Feature dimension:",
            feature_dimension,
        )

        print(
            "Positive class weight:",
            class_weight,
        )

        history = []
        best_validation_loss = (
            math.inf
        )
        global_step = 0

        autocast_device = (
            "cuda"
            if accelerator_selection.
            device.accelerator
            else "cpu"
        )

        autocast_dtype = (
            torch.float16
            if autocast_device == "cuda"
            else torch.bfloat16
        )

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            model.train()

            total_loss = 0.0
            batch_count = 0

            for batch_number, batch in (
                enumerate(
                    train_loader,
                    start=1,
                )
            ):
                optimizer.zero_grad(
                    set_to_none=True
                )

                features = batch[
                    "features"
                ].to(
                    device,
                    non_blocking=True,
                )

                targets = batch[
                    "target"
                ].to(
                    device,
                    non_blocking=True,
                )

                valid = (
                    targets != IGNORE_INDEX
                )

                if int(
                    valid.sum().item()
                ) == 0:
                    continue

                with torch.autocast(
                    device_type=(
                        autocast_device
                    ),
                    dtype=(
                        autocast_dtype
                    ),
                    enabled=use_amp,
                ):
                    output = model(
                        features
                    )

                    loss = criterion(
                        output[
                            "anomaly_logits"
                        ][valid],
                        targets[
                            valid
                        ].float(),
                    )

                if not torch.isfinite(
                    loss
                ):
                    raise TrainingNumericalError(
                        "Anomaly training "
                        "produced non-finite loss.",
                        details={
                            "epoch": epoch,
                            "batch": (
                                batch_number
                            ),
                        },
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        args.
                        gradient_clip_norm
                    ),
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                total_loss += float(
                    loss.item()
                )

                batch_count += 1
                global_step += 1

            if batch_count == 0:
                raise TrainingRunError(
                    "Training epoch processed "
                    "no usable anomaly batches."
                )

            train_loss = (
                total_loss
                / batch_count
            )

            pipeline.add_metric(
                run,
                name="loss",
                value=train_loss,
                epoch=epoch,
                step=global_step,
                split=(
                    EvaluationSplit.
                    TRAIN.value
                ),
            )

            validation_metrics = evaluate(
                model,
                validation_loader,
                device=device,
                criterion=criterion,
                threshold=args.threshold,
            )

            validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )

            checkpoint_score = (
                float(
                    validation_loss
                )
                if validation_loss
                is not None
                else train_loss
            )

            history.append(
                {
                    "epoch": epoch,
                    "global_step": (
                        global_step
                    ),
                    "train_loss": (
                        train_loss
                    ),
                    **{
                        (
                            "validation_"
                            + name
                        ): value
                        for name, value
                        in (
                            validation_metrics.
                            items()
                        )
                    },
                }
            )

            checkpoint = (
                pipeline.save_checkpoint(
                    run=run,
                    model=model,
                    optimizer=optimizer,
                    scaler=(
                        scaler
                        if use_amp
                        else None
                    ),
                    epoch=epoch,
                    step=global_step,
                    score=(
                        checkpoint_score
                    ),
                    score_name=(
                        "validation_loss"
                    ),
                    model_config=(
                        model_config
                    ),
                    extra={
                        "history": list(
                            history
                        ),
                        "temporary_threshold": (
                            args.threshold
                        ),
                        "accelerator": (
                            accelerator_selection.
                            to_dict()
                        ),
                        "mixed_precision": (
                            use_amp
                        ),
                    },
                )
            )

            copy_checkpoint_file(
                checkpoint.path,
                output_directory
                / "last.pt",
            )

            if (
                checkpoint_score
                < best_validation_loss
            ):
                best_validation_loss = (
                    checkpoint_score
                )

                copy_checkpoint_file(
                    checkpoint.path,
                    output_directory
                    / "best.pt",
                )

                print(
                    f"Epoch {epoch}: "
                    "saved new best checkpoint."
                )

            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.6f} "
                f"validation_loss="
                f"{validation_loss}"
            )

        best_checkpoint = (
            pipeline.best_checkpoint(
                run_id=str(
                    run.run_id
                ),
                mark=True,
            )
        )

        pipeline.transition_run(
            run,
            TrainingState.
            EVALUATING.value,
        )

        pipeline.checkpoints.restore(
            best_checkpoint,
            model=model,
            map_location=device,
            strict=True,
            verify=True,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device=device,
            criterion=criterion,
            threshold=args.threshold,
        )

        test_metrics = evaluate(
            model,
            test_loader,
            device=device,
            criterion=criterion,
            threshold=args.threshold,
        )

        validation_evaluation = (
            register_evaluation(
                pipeline,
                run,
                split=(
                    EvaluationSplit.
                    VALIDATION.value
                ),
                metrics=(
                    validation_metrics
                ),
            )
            if validation_loader
            is not None
            else None
        )

        test_evaluation = (
            register_evaluation(
                pipeline,
                run,
                split=(
                    EvaluationSplit.
                    TEST.value
                ),
                metrics=test_metrics,
            )
            if test_loader
            is not None
            else None
        )

        pipeline.transition_run(
            run,
            TrainingState.
            COMPLETED.value,
        )

        report = {
            "schema_version": "1",
            "run": run.to_dict(),
            "data": str(
                Path(
                    args.data
                ).expanduser().resolve()
            ),
            "accelerator": (
                accelerator_selection.
                to_dict()
            ),
            "runtime_device": str(
                device
            ),
            "mixed_precision": (
                use_amp
            ),
            "records": {
                "total": len(records),
                "train": len(
                    train_records
                ),
                "validation": len(
                    validation_records
                ),
                "test": len(
                    test_records
                ),
            },
            "datasets": {
                "train": (
                    train_dataset.summary()
                ),
                "validation": (
                    validation_dataset.
                    summary()
                    if validation_dataset
                    is not None
                    else None
                ),
                "test": (
                    test_dataset.summary()
                    if test_dataset
                    is not None
                    else None
                ),
            },
            "model_config": (
                model_config
            ),
            "class_weighting": {
                "enabled": (
                    not args.
                    disable_class_weight
                ),
                "positive_class_weight": (
                    class_weight
                ),
            },
            "history": history,
            "best_checkpoint": (
                best_checkpoint.to_dict()
            ),
            "validation": (
                validation_metrics
            ),
            "test": test_metrics,
            "evaluations": {
                "validation": (
                    validation_evaluation.
                    to_dict()
                    if validation_evaluation
                    is not None
                    else None
                ),
                "test": (
                    test_evaluation.
                    to_dict()
                    if test_evaluation
                    is not None
                    else None
                ),
            },
            "threshold": {
                "temporary_value": (
                    args.threshold
                ),
                "calibrated": False,
                "embedded_in_model": (
                    False
                ),
            },
            "automatic_promotion": (
                False
            ),
        }

        report_path = (
            output_directory
            / "training_report.json"
        )

        write_json_atomic(
            report_path,
            report,
        )

        print(
            "\nAnomaly training complete."
        )

        print(
            "Best checkpoint:",
            best_checkpoint.path,
        )

        print(
            "Report:",
            report_path,
        )

    except Exception as exc:
        if (
            run is not None
            and run.state
            not in {
                TrainingState.
                COMPLETED.value,
                TrainingState.
                FAILED.value,
                TrainingState.
                CANCELLED.value,
            }
        ):
            pipeline.transition_run(
                run,
                TrainingState.
                FAILED.value,
                error=exc,
            )

        raise

    finally:
        try:
            accelerator_manager.release(
                accelerator_selection
            )

        except Exception as release_error:
            print(
                "Warning: accelerator "
                "release failed:",
                type(
                    release_error
                ).__name__,
            )


if __name__ == "__main__":
    main()
