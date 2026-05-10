from pathlib import Path

import joblib
import numpy as np

from birdclef_2026_ml.configs import load_experiment_config, artifact_stem
from birdclef_2026_ml.models.artifacts import DualOneVsRestArtifacts, OneVsRestArtifacts
from birdclef_2026_ml.models.one_vs_rest import (
    predict_dual_one_vs_rest,
    predict_one_vs_rest,
    predict_proba_dual_one_vs_rest_batched,
    predict_proba_dual_one_vs_rest,
    predict_proba_one_vs_rest_batched,
    predict_proba_one_vs_rest,
)
from birdclef_2026_ml.paths import load_project_paths
from birdclef_2026_ml.processing.memmap_dataset import (
    load_memmap_dataset,
    save_memmap_array,
)


def predict_from_artifacts(
    X,
    artifacts,
):
    probas, preds = None, None
    if isinstance(artifacts, DualOneVsRestArtifacts):
        probas = predict_proba_dual_one_vs_rest(artifacts, X)
        preds = predict_dual_one_vs_rest(artifacts, X)

    elif isinstance(artifacts, OneVsRestArtifacts):
        probas = predict_proba_one_vs_rest(artifacts, X)
        preds = predict_one_vs_rest(artifacts, X)
    else:
        raise TypeError("Inference supports OneVsRestArtifacts or DualOneVsRestArtifacts")

    return probas, preds


def predict_from_artifacts_batched(
    X,
    artifacts,
    *,
    batch_size: int,
):
    probas, preds = None, None
    if isinstance(artifacts, DualOneVsRestArtifacts):
        probas = predict_proba_dual_one_vs_rest_batched(
            artifacts, X, batch_size=batch_size
        )
        preds = {
            "class_name": artifacts.class_name.label_encoder.inverse_transform(
                np.argmax(probas["class_name"], axis=1)
            ),
            "primary_label": artifacts.primary_label.label_encoder.inverse_transform(
                np.argmax(probas["primary_label"], axis=1)
            ),
        }

    elif isinstance(artifacts, OneVsRestArtifacts):
        probas = predict_proba_one_vs_rest_batched(artifacts, X, batch_size=batch_size)
        pred_ids = np.argmax(probas, axis=1)
        preds = artifacts.label_encoder.inverse_transform(pred_ids)
    else:
        raise TypeError("Inference supports OneVsRestArtifacts or DualOneVsRestArtifacts")

    return probas, preds


def run_ovr_inference(
    run_name: str,
    experiment_name: str,
    *,
    model_filename: str | None = None,
    reduced: bool = True,
    soundscape: bool = True,
    batch_size: int = 65536,
):
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
    output_stem = Path(artifact_filename).stem
    if soundscape:
        output_stem = f"{output_stem}_soundscape"

    probas, preds = predict_from_artifacts_batched(
        dataset.X, artifacts, batch_size=batch_size
    )

    if isinstance(artifacts, DualOneVsRestArtifacts):
        save_memmap_array(
            inference_dir,
            f"{output_stem}_class_name_probas",
            probas["class_name"],
        )
        save_memmap_array(
            inference_dir,
            f"{output_stem}_primary_label_probas",
            probas["primary_label"],
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
    else:
        save_memmap_array(
            inference_dir,
            f"{output_stem}_probas",
            probas,
        )
        np.save(inference_dir / f"{output_stem}_preds.npy", preds, allow_pickle=True)

    return inference_dir
