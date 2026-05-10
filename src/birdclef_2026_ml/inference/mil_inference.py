from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from birdclef_2026_ml.configs import artifact_stem, load_experiment_config
from birdclef_2026_ml.models.artifacts import DualOneVsRestArtifacts, OneVsRestArtifacts
from birdclef_2026_ml.models.mil import (
    assign_bag_ids,
    collapse_bag_labels,
    collapse_bag_multilabel_targets,
    bag_order
)
from birdclef_2026_ml.models.one_vs_rest import (
    predict_proba_dual_one_vs_rest,
    predict_proba_one_vs_rest,
)
from birdclef_2026_ml.paths import load_project_paths
from birdclef_2026_ml.processing.memmap_dataset import (
    load_memmap_dataset,
    load_ovr_inference_memmap,
)

AGGREGATION_METHODS: tuple[str, ...] = ("max", "mean", "noisy_or")


def aggregate_probabilities_by_bag(
        proba: np.ndarray,
        bag_ids: np.ndarray,
        method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window-level probabilities into bag-level probabilities."""
    proba = np.asarray(proba, dtype=float)
    bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
    if proba.ndim != 2:
        raise ValueError("proba must have shape (n_samples, n_classes)")
    if bag_ids.shape[0] != proba.shape[0]:
        raise ValueError("bag_ids must have same length as proba")

    method = str(method)
    if method not in AGGREGATION_METHODS:
        supported = ", ".join(AGGREGATION_METHODS)
        raise ValueError(f"Unsupported aggregation method='{method}'. Supported: {supported}")

    ordered_bag_ids, _ = bag_order(bag_ids)
    out = np.zeros((ordered_bag_ids.shape[0], proba.shape[1]), dtype=float)

    for i, bag_id in enumerate(ordered_bag_ids):
        print(f"{i}/{len(ordered_bag_ids)}")
        bag_proba = proba[bag_ids == bag_id]
        if bag_proba.shape[0] == 0:
            continue
        if method == "max":
            out[i] = np.max(bag_proba, axis=0)
        elif method == "mean":
            out[i] = np.mean(bag_proba, axis=0)
        else:
            out[i] = 1.0 - np.prod(1.0 - bag_proba, axis=0)

    return out, ordered_bag_ids


def _predict_labels_from_proba(proba: np.ndarray, label_encoder) -> np.ndarray:
    pred_ids = np.argmax(proba, axis=1)
    return label_encoder.inverse_transform(pred_ids)


def _resolve_bag_ids(
        run_name: str,
        bags_meta: np.ndarray | None,
        *,
        soundscape: bool,
) -> np.ndarray:
    if bags_meta is None:
        raise ValueError("MIL inference requires bag metadata in the dataset")

    bag_meta_arr = np.asarray(bags_meta)
    if soundscape:
        if bag_meta_arr.ndim >= 2 and bag_meta_arr.shape[1] >= 1:
            return np.asarray(bag_meta_arr[:, 0], dtype=int)
        if bag_meta_arr.ndim == 1:
            return np.asarray(bag_meta_arr, dtype=int)
        raise ValueError("Soundscape bag metadata must be 1D or 2D with bag ids in column 0")

    if bag_meta_arr.ndim != 1:
        raise ValueError("Train MIL bag metadata must be 1D window ids")
    return assign_bag_ids(bag_meta_arr, run_name)


def _collapse_true_labels(
    y: np.ndarray,
    bag_ids: np.ndarray,
    artifacts,
    *,
    soundscape: bool,
) -> dict[str, np.ndarray] | np.ndarray:
    y = np.asarray(y)

    if isinstance(artifacts, DualOneVsRestArtifacts):
        if soundscape:
            if y.ndim != 3:
                raise ValueError("Soundscape dual-target y must be 3D")
            n_class = len(artifacts.class_name.label_encoder.classes_)
            n_primary = len(artifacts.primary_label.label_encoder.classes_)
            class_true = collapse_bag_multilabel_targets(bag_ids, y[:, 0, :n_class])
            primary_true = collapse_bag_multilabel_targets(bag_ids, y[:, 1, :n_primary])
        else:
            if y.ndim < 2:
                raise ValueError("Train dual-target y must have 2 columns")
            class_true = collapse_bag_labels(bag_ids, y[:, 0])
            primary_true = collapse_bag_labels(bag_ids, y[:, 1])
        return {
            "class_name": class_true,
            "primary_label": primary_true,
        }

    if soundscape:
        if y.ndim == 3:
            n_classes = len(artifacts.label_encoder.classes_)
            return collapse_bag_multilabel_targets(bag_ids, y[:, 1, :n_classes])
        if y.ndim == 2:
            return collapse_bag_multilabel_targets(bag_ids, y)
        raise ValueError("Soundscape single-target y must be 2D or 3D")

    if y.ndim == 2:
        target = y[:, 1] if y.shape[1] > 1 else y[:, 0]
        return collapse_bag_labels(bag_ids, target)
    return collapse_bag_labels(bag_ids, y)


def _predict_mil_from_artifacts(
        x,
        artifacts,
        *,
        bag_ids: np.ndarray,
        aggregation: str,
) -> tuple[dict[str, np.ndarray] | np.ndarray, dict[str, np.ndarray] | np.ndarray, np.ndarray]:
    if isinstance(artifacts, DualOneVsRestArtifacts):
        probas = predict_proba_dual_one_vs_rest(artifacts, x)
        class_proba, ordered_bag_ids = aggregate_probabilities_by_bag(
            probas["class_name"],
            bag_ids,
            method=aggregation,
        )
        primary_proba, _ = aggregate_probabilities_by_bag(
            probas["primary_label"],
            bag_ids,
            method=aggregation,
        )
        preds = {
            "class_name": _predict_labels_from_proba(class_proba, artifacts.class_name.label_encoder),
            "primary_label": _predict_labels_from_proba(primary_proba, artifacts.primary_label.label_encoder),
        }
        return {
            "class_name": class_proba,
            "primary_label": primary_proba,
        }, preds, ordered_bag_ids

    if isinstance(artifacts, OneVsRestArtifacts):
        proba = predict_proba_one_vs_rest(artifacts, x)
        bag_proba, ordered_bag_ids = aggregate_probabilities_by_bag(
            proba,
            bag_ids,
            method=aggregation,
        )
        preds = _predict_labels_from_proba(bag_proba, artifacts.label_encoder)
        return bag_proba, preds, ordered_bag_ids

    raise TypeError("Inference supports OneVsRestArtifacts or DualOneVsRestArtifacts")


def _ovr_output_stem(artifact_filename: str, *, soundscape: bool) -> str:
    stem = Path(artifact_filename).stem
    return f"{stem}_soundscape" if soundscape else stem


def _load_ovr_window_probas(
    inference_dir: Path,
    artifact_filename: str,
    artifacts,
    *,
    soundscape: bool,
) -> dict[str, np.ndarray] | np.ndarray:
    output_stem = _ovr_output_stem(artifact_filename, soundscape=soundscape)
    memmaps = load_ovr_inference_memmap(
        inference_dir,
        output_stem,
        dual=isinstance(artifacts, DualOneVsRestArtifacts),
    )
    if isinstance(artifacts, DualOneVsRestArtifacts):
        if memmaps.class_name is None or memmaps.primary_label is None:
            raise FileNotFoundError("Missing dual-target memmap probabilities")
        return {
            "class_name": memmaps.class_name,
            "primary_label": memmaps.primary_label,
        }
    if memmaps.proba is None:
        raise FileNotFoundError("Missing single-target memmap probabilities")
    return memmaps.proba


def _validate_proba_rows(proba: np.ndarray, bag_ids: np.ndarray, name: str) -> None:
    if proba.shape[0] != bag_ids.shape[0]:
        raise ValueError(f"{name} rows must match bag_ids length")


def _aggregate_mil_from_probas(
    probas,
    artifacts,
    *,
    bag_ids: np.ndarray,
    aggregation: str,
) -> tuple[dict[str, np.ndarray] | np.ndarray, dict[str, np.ndarray] | np.ndarray, np.ndarray]:
    if isinstance(artifacts, DualOneVsRestArtifacts):
        class_proba = np.asarray(probas["class_name"], dtype=float)
        primary_proba = np.asarray(probas["primary_label"], dtype=float)
        _validate_proba_rows(class_proba, bag_ids, "class_name_proba")
        _validate_proba_rows(primary_proba, bag_ids, "primary_label_proba")

        class_bag_proba, ordered_bag_ids = aggregate_probabilities_by_bag(
            class_proba,
            bag_ids,
            method=aggregation,
        )
        primary_bag_proba, _ = aggregate_probabilities_by_bag(
            primary_proba,
            bag_ids,
            method=aggregation,
        )
        preds = {
            "class_name": _predict_labels_from_proba(class_bag_proba, artifacts.class_name.label_encoder),
            "primary_label": _predict_labels_from_proba(primary_bag_proba, artifacts.primary_label.label_encoder),
        }
        return {
            "class_name": class_bag_proba,
            "primary_label": primary_bag_proba,
        }, preds, ordered_bag_ids

    proba = np.asarray(probas, dtype=float)
    _validate_proba_rows(proba, bag_ids, "proba")
    bag_proba, ordered_bag_ids = aggregate_probabilities_by_bag(
        proba,
        bag_ids,
        method=aggregation,
    )
    preds = _predict_labels_from_proba(bag_proba, artifacts.label_encoder)
    return bag_proba, preds, ordered_bag_ids


def _remap_val_idx(
    val_idx: np.ndarray | None,
    bag_ids: np.ndarray,
    ordered_bag_ids: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if val_idx is None:
        return None, None
    val_idx = np.asarray(val_idx, dtype=int)
    if val_idx.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    val_bag_ids, _ = bag_order(bag_ids[val_idx])
    bag_idx_by_id = {int(bid): idx for idx, bid in enumerate(ordered_bag_ids)}
    val_bag_idx = np.asarray(
        [bag_idx_by_id[int(bid)] for bid in val_bag_ids],
        dtype=int,
    )
    return val_bag_ids, val_bag_idx


def run_mil_inference(
        X_or_run_name,
        artifacts_or_experiment,
        *,
        class_name_proba: np.ndarray | None = None,
        primary_label_proba: np.ndarray | None = None,
        proba: np.ndarray | None = None,
        bag_ids: np.ndarray | None = None,
        y: np.ndarray | None = None,
        val_idx: np.ndarray | None = None,
        model_filename: str | None = None,
        reduced: bool = True,
        soundscape: bool = True,
        aggregation: str = "mean",
):
    if isinstance(artifacts_or_experiment, (DualOneVsRestArtifacts, OneVsRestArtifacts)):
        if bag_ids is None or y is None:
            raise ValueError("bag_ids and y must be provided when passing artifacts directly")
        bag_ids = np.asarray(bag_ids, dtype=int)
        y = np.asarray(y)

        if isinstance(artifacts_or_experiment, DualOneVsRestArtifacts):
            if class_name_proba is not None and primary_label_proba is not None:
                probas, preds, ordered_bag_ids = _aggregate_mil_from_probas(
                    {
                        "class_name": class_name_proba,
                        "primary_label": primary_label_proba,
                    },
                    artifacts_or_experiment,
                    bag_ids=bag_ids,
                    aggregation=aggregation,
                )
            else:
                probas, preds, ordered_bag_ids = _predict_mil_from_artifacts(
                    X_or_run_name,
                    artifacts_or_experiment,
                    bag_ids=bag_ids,
                    aggregation=aggregation,
                )
        else:
            if proba is not None:
                probas, preds, ordered_bag_ids = _aggregate_mil_from_probas(
                    proba,
                    artifacts_or_experiment,
                    bag_ids=bag_ids,
                    aggregation=aggregation,
                )
            else:
                probas, preds, ordered_bag_ids = _predict_mil_from_artifacts(
                    X_or_run_name,
                    artifacts_or_experiment,
                    bag_ids=bag_ids,
                    aggregation=aggregation,
                )

        y_true = _collapse_true_labels(y, bag_ids, artifacts_or_experiment, soundscape=soundscape)
        val_bag_ids, val_bag_idx = _remap_val_idx(val_idx, bag_ids, ordered_bag_ids)
        return {
            "probas": probas,
            "preds": preds,
            "y_true": y_true,
            "ordered_bag_ids": ordered_bag_ids,
            "val_bag_ids": val_bag_ids,
            "val_bag_idx": val_bag_idx,
        }

    run_name = str(X_or_run_name)
    experiment_name = str(artifacts_or_experiment)
    paths = load_project_paths()
    experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    inference_dir = experiment_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    stem = artifact_stem(experiment_cfg)
    artifact_filename = model_filename or f"{stem}_ovr.joblib"
    model_path = experiment_dir / artifact_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Saved OVR artifacts not found: {model_path}")

    dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=soundscape)
    artifacts = joblib.load(model_path)
    bag_ids_full = _resolve_bag_ids(run_name, dataset.bags_meta, soundscape=soundscape)
    y = np.asarray(dataset.y)
    bag_ids = np.asarray(bag_ids_full, dtype=int)
    output_stem = f"{Path(artifact_filename).stem}_mil_{aggregation}"

    if isinstance(artifacts, DualOneVsRestArtifacts):
        if class_name_proba is None or primary_label_proba is None:
            loaded = _load_ovr_window_probas(
                inference_dir,
                artifact_filename,
                artifacts,
                soundscape=soundscape,
            )
            if class_name_proba is None:
                class_name_proba = loaded["class_name"]
            if primary_label_proba is None:
                primary_label_proba = loaded["primary_label"]
        window_probas = {
            "class_name": class_name_proba,
            "primary_label": primary_label_proba,
        }
    else:
        if proba is None:
            proba = _load_ovr_window_probas(
                inference_dir,
                artifact_filename,
                artifacts,
                soundscape=soundscape,
            )
        window_probas = proba

    probas, preds, ordered_bag_ids = _aggregate_mil_from_probas(
        window_probas,
        artifacts,
        bag_ids=bag_ids,
        aggregation=aggregation,
    )

    np.save(inference_dir / f"{output_stem}_bag_ids.npy", ordered_bag_ids, allow_pickle=True)

    val_bag_ids, val_bag_idx = _remap_val_idx(val_idx, bag_ids, ordered_bag_ids)
    if val_bag_ids is not None:
        np.save(inference_dir / f"{output_stem}_val_bag_ids.npy", val_bag_ids, allow_pickle=True)
    if val_bag_idx is not None:
        np.save(inference_dir / f"{output_stem}_val_bag_idx.npy", val_bag_idx, allow_pickle=True)

    y_true = _collapse_true_labels(y, bag_ids, artifacts, soundscape=soundscape)

    if isinstance(artifacts, DualOneVsRestArtifacts):
        np.save(
            inference_dir / f"{output_stem}_class_name_probas.npy",
            probas["class_name"],
            allow_pickle=True,
        )
        np.save(
            inference_dir / f"{output_stem}_primary_label_probas.npy",
            probas["primary_label"],
            allow_pickle=True,
        )
        np.save(
            inference_dir / f"{output_stem}_class_name_preds.npy",
            preds["class_name"],
            allow_pickle=True,
        )
        np.save(
            inference_dir / f"{output_stem}_primary_label_preds.npy",
            preds["primary_label"],
            allow_pickle=True,
        )
        np.save(
            inference_dir / f"{output_stem}_class_name_true.npy",
            y_true["class_name"],
            allow_pickle=True,
        )
        np.save(
            inference_dir / f"{output_stem}_primary_label_true.npy",
            y_true["primary_label"],
            allow_pickle=True,
        )
    else:
        np.save(inference_dir / f"{output_stem}_probas.npy", probas, allow_pickle=True)
        np.save(inference_dir / f"{output_stem}_preds.npy", preds, allow_pickle=True)
        np.save(inference_dir / f"{output_stem}_true.npy", y_true, allow_pickle=True)

    return {
        "probas": probas,
        "preds": preds,
        "y_true": y_true,
        "ordered_bag_ids": ordered_bag_ids,
        "val_bag_ids": val_bag_ids,
        "val_bag_idx": val_bag_idx,
    }
