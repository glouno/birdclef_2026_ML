import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from birdclef_2026_ml.audio import apply_spectral_gating
from birdclef_2026_ml.configs import (
    PipelineConfig,
    SpectralGatingConfig,
    load_pipeline_config
)
from birdclef_2026_ml.feature_engineering import (
    build_profile,
    save_features_from_audio_dir,
    save_pooled_features_from_mel_dir,
    save_profiles,
)
from birdclef_2026_ml.paths import ProjectPaths, load_project_paths
from birdclef_2026_ml.processing.audio_utils import load_config
from birdclef_2026_ml.processing.feature_dataset_builder import (
    build_memmap_from_chunks,
    reduce_feature_memmap,
)
from birdclef_2026_ml.processing.memmap_dataset import load_memmap_dataset
from birdclef_2026_ml.processing.preprocess import preprocess_datasets_for_models
from birdclef_2026_ml.training.model_training import (
    calibrate_and_save_ovr_models_chunks,
    tune_and_save_ovr_thresholds,
    train_and_save_ovr_models_chunks,
)


DATASET_CHOICES = ("train", "soundscapes")
STAGE_CHOICES = ("raw", "clean")
FEATURE_KIND_CHOICES = ("mel", "pooled")


def _dataset_frame(paths: ProjectPaths, dataset: str) -> Path:
    return paths.dataset_frame(dataset)


def _audio_dir(paths: ProjectPaths, dataset: str, stage: str) -> Path:
    try:
        return paths.audio_dir(dataset, stage)
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset/stage pair: {dataset}/{stage}") from exc


def _feature_dir(paths: ProjectPaths, dataset: str, kind: str) -> Path:
    try:
        return paths.feature_dir(dataset, kind)
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset/kind pair: {dataset}/{kind}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="birdclef", description="BirdCLEF data utilities")
    parser.add_argument(
        "--project-config",
        type=str,
        default=None,
        help="Path to project YAML config. Defaults to configs/project.yaml or BIRDCLEF_CONFIG.",
    )
    subparsers = parser.add_subparsers(dest="command")

    parser_fit_ovr_models_chunks = subparsers.add_parser(
        "train-ovr-models-chunks",
        help="Train OVR models for one saved run.",
    )
    parser_fit_ovr_models_chunks.add_argument("--run-name", type=str, required=True)
    parser_fit_ovr_models_chunks.add_argument("--experiment", type=str, required=True)

    parser_calibrate_ovr_models_chunks = subparsers.add_parser(
        "calibrate-ovr-models-chunks",
        help="Calibrate pretrained OVR models for one saved run and experiment.",
    )
    parser_calibrate_ovr_models_chunks.add_argument("--run-name", type=str, required=True)
    parser_calibrate_ovr_models_chunks.add_argument("--experiment", type=str, required=True)
    parser_calibrate_ovr_models_chunks.add_argument("--calibration", type=str, required=True)

    parser_tune_ovr_thresholds = subparsers.add_parser(
        "tune-ovr-thresholds",
        help="Tune per-class decision thresholds for saved OVR artifacts.",
    )
    parser_tune_ovr_thresholds.add_argument("--run-name", type=str, required=True)
    parser_tune_ovr_thresholds.add_argument("--experiment", type=str, required=True)
    parser_tune_ovr_thresholds.add_argument("--score", type=str, default="macro_f1")
    parser_tune_ovr_thresholds.add_argument("--model-filename", type=str, default=None)
    parser_tune_ovr_thresholds.add_argument(
        "--target-name",
        type=str,
        default="class_name",
        choices=("class_name", "primary_label"),
    )
    parser_tune_ovr_thresholds.add_argument("--max-rounds", type=int, default=2)

    parser_build_feature_matrices = subparsers.add_parser(
        "build-feature-matrices",
        help="Build feature matrices from audio + cached mel or pooled features.",
    )
    parser_build_feature_matrices.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser_build_feature_matrices.add_argument("--audio-stage", type=str, default="clean", choices=STAGE_CHOICES)
    parser_build_feature_matrices.add_argument("--feature-kind", type=str, default="mel", choices=FEATURE_KIND_CHOICES)
    parser_build_feature_matrices.add_argument("--run-name", type=str, required=True)
    parser_build_feature_matrices.add_argument("--profiles-path", type=str, default=None)
    parser_build_feature_matrices.add_argument("--species-ids-path", type=str, default=None)

    parser_reduce_feature_matrices = subparsers.add_parser(
        "reduce-feature-matrices",
        help="Reduce saved feature matrices for one run.",
    )
    parser_reduce_feature_matrices.add_argument("--run-name", type=str, required=True)
    parser_reduce_feature_matrices.add_argument("--target-mel-bins", type=int, default=32)

    parser_profiles = subparsers.add_parser(
        "build-profiles",
        help="Build one species profile per label from one run.",
    )
    parser_profiles.add_argument("--run-name", type=str, required=True)
    parser_profiles.add_argument(
        "--pooling-method",
        type=str,
        default="median",
        choices=["median", "trimmed_mean", "mean"],
    )
    parser_profiles.add_argument("--trim-ratio", type=float, default=0.1)

    parser_spectral_gating = subparsers.add_parser(
        "spectral-gating",
        help="Apply spectral gating to dataset audio, save cleaned files in interim.",
    )
    parser_spectral_gating.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser_spectral_gating.add_argument("--config-path", type=str, default=None)
    parser_spectral_gating.add_argument("--input-stage", type=str, default="raw", choices=STAGE_CHOICES)
    parser_spectral_gating.add_argument("--output-stage", type=str, default="clean", choices=STAGE_CHOICES)
    parser_spectral_gating.add_argument("--filename-col", type=str, default="filename")

    subparsers.add_parser(
        "preprocess-datasets-for-models",
        help="Preprocess train and soundscape metadata for modeling.",
    )

    parser_mel = subparsers.add_parser(
        "extract-all-features",
        help="Extract raw log-mel spectrograms for one dataset.",
    )
    parser_mel.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser_mel.add_argument("--audio-stage", type=str, default="clean", choices=STAGE_CHOICES)

    # parser_pool_mel = subparsers.add_parser(
    #     "pool-mel-features",
    #     help="Pool precomputed mel spectrograms into tabular features.",
    # )
    # parser_pool_mel.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    # parser_pool_mel.add_argument("--input-kind", type=str, default="mel", choices=("mel",))
    # parser_pool_mel.add_argument("--output-kind", type=str, default="pooled", choices=("pooled",))

    return parser


def _run_preprocess_datasets_for_models(paths: ProjectPaths):
    train = pd.read_csv(paths.metadata_train)
    soundscapes = pd.read_csv(paths.metadata_soundscapes)
    taxonomy = pd.read_csv(paths.taxonomy)

    train_proc, ss_proc, le_pl, le_cn, pl_to_cn_arr = preprocess_datasets_for_models(train, soundscapes, taxonomy)

    paths.train_processed.parent.mkdir(parents=True, exist_ok=True)
    paths.soundscapes_processed.parent.mkdir(parents=True, exist_ok=True)
    paths.label_encoders_dir.mkdir(parents=True, exist_ok=True)

    train_proc.to_parquet(paths.train_processed, index=False)
    ss_proc.to_parquet(paths.soundscapes_processed, index=False)
    joblib.dump(le_pl, paths.label_encoders_dir / "primary_label.joblib")
    joblib.dump(le_cn, paths.label_encoders_dir / "class_name.joblib")
    np.save(paths.primary_to_class / "primary_to_class.npy", pl_to_cn_arr)


def _run_build_profiles(args, paths: ProjectPaths):
    dataset = load_memmap_dataset(args.run_name)
    species_ids, profiles = build_profile(
        dataset.X,
        dataset.y,
        dataset.feature_names,
        pooling_method=args.pooling_method,
        trim_ratio=args.trim_ratio,
    )

    paths.profiles_dir.mkdir(parents=True, exist_ok=True)
    profiles_path, species_ids_path = save_profiles(paths.profiles_dir, species_ids, profiles)
    print(f"Profiles saved to: {profiles_path}")
    print(f"Species ids saved to: {species_ids_path}")


def _run_spectral_gating(args, paths: ProjectPaths):
    df = pd.read_parquet(_dataset_frame(paths, args.dataset))
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    cfg_dict = load_config(Path(args.config_path) if args.config_path else paths.spectral_gating_config)
    config = SpectralGatingConfig(**cfg_dict)

    apply_spectral_gating(
        df=df,
        config=config,
        input_root=_audio_dir(paths, args.dataset, args.input_stage),
        output_root=_audio_dir(paths, args.dataset, args.output_stage),
        filename_col=args.filename_col,
    )


def _run_extract_all_features(args, paths: ProjectPaths):
    pipeline_cfg = PipelineConfig()
    df = pd.read_parquet(_dataset_frame(paths, args.dataset))

    save_features_from_audio_dir(
        df=df,
        input_path=_audio_dir(paths, args.dataset, args.audio_stage),
        config_path=paths.feature_config,
        pipeline_cfg=pipeline_cfg,
        output_path=_feature_dir(paths, args.dataset, "mel"),
    )


# def _run_pool_mel_features(args, paths: ProjectPaths):
#     pipeline_cfg = PipelineConfig()
#     df = pd.read_parquet(_dataset_frame(paths, args.dataset))

#     save_pooled_features_from_mel_dir(
#         df=df,
#         input_mel_dir=_feature_dir(paths, args.dataset, args.input_kind),
#         output_path=_feature_dir(paths, args.dataset, args.output_kind),
#         pipeline_cfg=pipeline_cfg,
#     )


def _run_build_feature_matrices(args, paths: ProjectPaths):
    df = pd.read_parquet(_dataset_frame(paths, args.dataset))
    pipeline_cfg = load_pipeline_config(args.run_name)
    # Compose the path to the pipeline config YAML

    build_memmap_from_chunks(
        df=df,
        pipeline_cfg=pipeline_cfg,
        pathroot=_audio_dir(paths, args.dataset, args.audio_stage),
        filename_col="filename",
        features_pathroot=_feature_dir(paths, args.dataset, args.feature_kind),
        out_instances_path=paths.run_data_dir(args.run_name),
        soundscapes=args.dataset == "soundscapes",
        profiles_path=paths.profiles_dir / "species_profiles.npy",
        species_ids_path=paths.profiles_dir / "species_profile_ids.npy"
    )


def _run_reduce_feature_matrices(args, paths: ProjectPaths):
    run_path = paths.run_data_dir(args.run_name)
    X_reduced, reduced_names = reduce_feature_memmap(run_path=run_path, target_mel_bins=args.target_mel_bins)
    print(f"Reduced matrix saved in: {run_path}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Reduced feature count: {len(reduced_names)}")


def _run_train_ovr_models_chunks(args):
    train_and_save_ovr_models_chunks(args.run_name, args.experiment)


def _run_calibrate_ovr_models_chunks(args):
    calibrate_and_save_ovr_models_chunks(
        args.run_name,
        args.experiment,
        calibration_name=args.calibration,
    )


def _run_tune_ovr_thresholds(args):
    tune_and_save_ovr_thresholds(
        args.run_name,
        args.experiment,
        score_name=args.score,
        model_filename=args.model_filename,
        target_name=args.target_name,
        max_rounds=args.max_rounds,
    )


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    paths = load_project_paths(args.project_config)

    if args.command == "build-feature-matrices":
        _run_build_feature_matrices(args, paths)
        return 0
    if args.command == "reduce-feature-matrices":
        _run_reduce_feature_matrices(args, paths)
        return 0
    if args.command == "preprocess-datasets-for-models":
        _run_preprocess_datasets_for_models(paths)
        return 0
    if args.command == "spectral-gating":
        _run_spectral_gating(args, paths)
        print("Spectral gating completed.")
        return 0
    if args.command == "build-profiles":
        _run_build_profiles(args, paths)
        return 0
    if args.command == "extract-all-features":
        _run_extract_all_features(args, paths)
        return 0
    # if args.command == "pool-mel-features":
    #     _run_pool_mel_features(args, paths)
    #     return 0
    if args.command == "train-ovr-models-chunks":
        _run_train_ovr_models_chunks(args)
        return 0
    if args.command == "calibrate-ovr-models-chunks":
        _run_calibrate_ovr_models_chunks(args)
        return 0
    if args.command == "tune-ovr-thresholds":
        _run_tune_ovr_thresholds(args)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
