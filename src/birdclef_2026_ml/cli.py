import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from birdclef_2026_ml.audio import apply_spectral_gating, apply_silence_trimming
from birdclef_2026_ml.configs import (
    PipelineConfig,
    SpectralGatingConfig,
    load_pipeline_config,
    load_config
)
from birdclef_2026_ml.feature_engineering import (
    build_profiles,
    extract_save_mel_spectograms,
)
from birdclef_2026_ml.inference.ovr_inference import run_ovr_inference
from birdclef_2026_ml.paths import ProjectPaths, load_project_paths
from birdclef_2026_ml.processing.feature_dataset_builder import (
    build_feature_memmap_artifacts,
    build_soundscape_feature_memmap_artifacts,
    reduce_feature_memmap,
)
from birdclef_2026_ml.processing.preprocess import preprocess_datasets_for_models
from birdclef_2026_ml.training.ovr_training import (
    train_and_save_ovr_models_chunks,
)
from birdclef_2026_ml.training.ss_second_stage import (
    build_mil_second_stage_clean_audio_data,
    fit_second_stage_soundscapes,
    fit_mil_second_stage_clean_audio,
    train_mil_second_stage_clean_audio,
    run_mil_second_stage_clean_audio_soundscape_inference,
    fit_second_stage_soundscapes_with_mil_proba,
)


DATASET_CHOICES = ("train", "soundscapes")
STAGE_CHOICES = ("raw", "clean", "clean_trim")
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

    parser_mil_clean = subparsers.add_parser(
        "train-mil-second-stage-clean",
        help="Train MIL second-stage model on clean audio bags.",
    )
    parser_mil_clean.add_argument("--run-name", type=str, required=True)
    parser_mil_clean.add_argument("--experiment", type=str, required=True)
    parser_mil_clean.add_argument("--model-filename", type=str, default=None)
    parser_mil_clean.add_argument(
        "--full",
        action="store_true",
        help="Use full (non-reduced) matrices when running stage-1 inference.",
    )
    parser_mil_clean.add_argument(
        "--bag-batch-size",
        type=int,
        default=512,
        help="Bag batch size for building MIL features.",
    )

    parser_mil_clean_data = subparsers.add_parser(
        "build-mil-second-stage-clean-data",
        help="Build clean-audio MIL second-stage features and targets.",
    )
    parser_mil_clean_data.add_argument("--run-name", type=str, required=True)
    parser_mil_clean_data.add_argument("--experiment", type=str, required=True)
    parser_mil_clean_data.add_argument("--model-filename", type=str, default=None)
    parser_mil_clean_data.add_argument(
        "--full",
        action="store_true",
        help="Use full (non-reduced) matrices when building features.",
    )
    parser_mil_clean_data.add_argument(
        "--bag-batch-size",
        type=int,
        default=512,
        help="Bag batch size for building MIL features.",
    )
    parser_mil_clean_data.add_argument(
        "--soundscapes",
        action="store_true",
        help="Use soundscape MIL matrices and bag metadata.",
    )

    parser_mil_clean_train = subparsers.add_parser(
        "train-mil-second-stage-clean-model",
        help="Train MIL second-stage model from saved clean-audio features.",
    )
    parser_mil_clean_train.add_argument("--run-name", type=str, required=True)
    parser_mil_clean_train.add_argument("--experiment", type=str, required=True)
    parser_mil_clean_train.add_argument("--model-filename", type=str, default=None)

    parser_mil_clean_soundscape = subparsers.add_parser(
        "run-mil-second-stage-clean-soundscape",
        help="Run clean-audio second-stage models on soundscape bag features.",
    )
    parser_mil_clean_soundscape.add_argument("--run-name", type=str, required=True)
    parser_mil_clean_soundscape.add_argument("--experiment", type=str, required=True)
    parser_mil_clean_soundscape.add_argument("--model-filename", type=str, default=None)

    parser_soundscape_mil_context = subparsers.add_parser(
        "run-soundscape-mil-context-oof",
        help="Train soundscape OOF model on P(bag), bag stats, and context features.",
    )
    parser_soundscape_mil_context.add_argument("--run-name", type=str, required=True)
    parser_soundscape_mil_context.add_argument("--experiment", type=str, required=True)
    parser_soundscape_mil_context.add_argument("--model-filename", type=str, default=None)
    parser_soundscape_mil_context.add_argument("--n-splits", type=int, default=5)
    parser_soundscape_mil_context.add_argument("--random-state", type=int, default=42)
    parser_soundscape_mil_context.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffle in soundscape OOF splits.",
    )
    parser_soundscape_mil_context.add_argument(
        "--full",
        action="store_true",
        help="Use full (non-reduced) soundscape matrices.",
    )

    parser_soundscape_second_stage = subparsers.add_parser(
        "run-soundscape-second-stage",
        help="Train soundscape OOF model on stage-1 probas and context features.",
    )
    parser_soundscape_second_stage.add_argument("--run-name", type=str, required=True)
    parser_soundscape_second_stage.add_argument("--experiment", type=str, required=True)
    parser_soundscape_second_stage.add_argument("--model-filename", type=str, default=None)
    parser_soundscape_second_stage.add_argument("--n-splits", type=int, default=5)
    parser_soundscape_second_stage.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Batch size for stage-1 inference used in second-stage training.",
    )
    parser_soundscape_second_stage.add_argument(
        "--full",
        action="store_true",
        help="Use full (non-reduced) soundscape matrices.",
    )

    parser_run_ovr_inference = subparsers.add_parser(
        "run-ovr-inference",
        help="Run inference with saved OVR artifacts.",
    )
    parser_run_ovr_inference.add_argument("--run-name", type=str, required=True)
    parser_run_ovr_inference.add_argument("--experiment", type=str, required=True)
    parser_run_ovr_inference.add_argument("--model-filename", type=str, default=None)
    parser_run_ovr_inference.add_argument(
        "--soundscapes",
        action="store_true",
        help="Run inference on soundscape feature matrices.",
    )
    parser_run_ovr_inference.add_argument(
        "--full",
        action="store_true",
        help="Use full (non-reduced) feature matrices.",
    )

    parser_build_feature_matrices = subparsers.add_parser(
        "build-feature-matrices",
        help="Build feature matrices from audio + cached mel or pooled features.",
    )
    parser_build_feature_matrices.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser_build_feature_matrices.add_argument("--audio-stage", type=str, default="clean", choices=STAGE_CHOICES)
    parser_build_feature_matrices.add_argument("--feature-kind", type=str, default="mel", choices=FEATURE_KIND_CHOICES)
    parser_build_feature_matrices.add_argument("--run-name", type=str, required=True)
    parser_build_feature_matrices.add_argument("--soundscapes", action="store_true")

    parser_reduce_feature_matrices = subparsers.add_parser(
        "reduce-feature-matrices",
        help="Reduce saved feature matrices for one run.",
    )
    parser_reduce_feature_matrices.add_argument("--run-name", type=str, required=True)
    parser_reduce_feature_matrices.add_argument("--target-mel-bins", type=int, default=32)
    parser_reduce_feature_matrices.add_argument("--soundscapes", action="store_true")

    parser_profiles = subparsers.add_parser(
        "build-profiles",
        help="Build one species profile per label from one run.",
    )
    # parser_profiles.add_argument("--run-name", type=str, required=True)
    parser_profiles.add_argument("--input-kind", type=str, default="mel", choices=("mel",))
    parser_profiles.add_argument("--output-kind", type=str, default="pooled", choices=("pooled",))
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

    parser_trim_train_silence = subparsers.add_parser(
        "trim-train-silence",
        help="Trim leading/trailing silence from clean train audio.",
    )
    parser_trim_train_silence.add_argument(
        "--top-db",
        type=float,
        default=20.0,
        help="Silence threshold for librosa.effects.trim.",
    )
    parser_trim_train_silence.add_argument("--filename-col", type=str, default="filename")

    subparsers.add_parser(
        "preprocess-datasets-for-models",
        help="Preprocess train and soundscape metadata for modeling.",
    )

    parser_mel = subparsers.add_parser(
        "extract-mel-spectograms",
        help="Extract raw log-mel spectrograms for one dataset.",
    )
    parser_mel.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser_mel.add_argument("--audio-stage", type=str, required=True, choices=STAGE_CHOICES)

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
    df = pd.read_parquet(paths.train_processed)
    species_ids, profiles = build_profiles(
        df=df,
        input_mel_dir=_feature_dir(paths, "train", args.input_kind),
        output_path=_feature_dir(paths, "train", args.output_kind),
    )

    output_dir = paths.profiles_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "species_profiles.npy"
    species_ids_path = output_dir / "species_profile_ids.npy"
    np.save(profiles_path, profiles)
    np.save(species_ids_path, species_ids)

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
        **args,
    )


def _run_trim_train_silence(args, paths: ProjectPaths):
    df = pd.read_parquet(paths.train_processed)
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    duration_trim = apply_silence_trimming(
        df=df,
        input_root=paths.train_audio_clean_dir,
        output_root=paths.train_audio_clean_trim_dir,
        filename_col=args.filename_col,
        top_db=args.top_db,
    )

    df["duration_trim"] = duration_trim
    df.to_parquet(paths.train_processed, index=False)


def _run_extract_mel_spectograms(args, paths: ProjectPaths):
    pipeline_cfg = PipelineConfig()
    df = pd.read_parquet(_dataset_frame(paths, args.dataset))

    print("Mel spectogram", _audio_dir(paths, args.dataset, args.audio_stage))
    extract_save_mel_spectograms(
        df=df,
        input_path=_audio_dir(paths, args.dataset, args.audio_stage),
        config_path=paths.feature_config,
        pipeline_cfg=pipeline_cfg,
        output_path=_feature_dir(paths, args.dataset, "mel"),
    )


def _run_build_feature_matrices(args, paths: ProjectPaths):
    df = pd.read_parquet(_dataset_frame(paths, args.dataset))
    pipeline_cfg = load_pipeline_config(args.run_name)
    # Compose the path to the pipeline config YAML

    if args.dataset == "soundscapes":
        build_soundscape_feature_memmap_artifacts(
            df=df,
            pipeline_cfg=pipeline_cfg,
            pathroot=_audio_dir(paths, args.dataset, args.audio_stage),
            features_pathroot=_feature_dir(paths, args.dataset, args.feature_kind),
            out_instances_path=paths.run_data_dir(args.run_name) / "full",
            profiles_path=paths.profiles_dir / "species_profiles.npy",
            species_ids_path=paths.profiles_dir / "species_profile_ids.npy",
        )
        return

    build_feature_memmap_artifacts(
        df=df,
        pipeline_cfg=pipeline_cfg,
        pathroot=_audio_dir(paths, args.dataset, args.audio_stage),
        filename_col="filename",
        features_pathroot=_feature_dir(paths, args.dataset, args.feature_kind),
        out_instances_path=paths.run_data_dir(args.run_name) / "full",
        profiles_path=paths.profiles_dir / "species_profiles.npy",
        species_ids_path=paths.profiles_dir / "species_profile_ids.npy",
    )


def _run_reduce_feature_matrices(args, paths: ProjectPaths):
    run_path = paths.run_data_dir(args.run_name)
    X_reduced, reduced_names = reduce_feature_memmap(
        run_path=run_path, target_mel_bins=args.target_mel_bins, soundscapes=args.soundscapes)
    print(f"Reduced matrix saved in: {run_path}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Reduced feature count: {len(reduced_names)}")


def _run_train_ovr_models_chunks(args):
    train_and_save_ovr_models_chunks(args.run_name, args.experiment)


def _run_ovr_inference(args):
    inference_dir = run_ovr_inference(
        args.run_name,
        args.experiment,
        model_filename=args.model_filename,
        reduced=not args.full,
        soundscape=args.soundscapes,
    )
    print(f"Inference outputs saved to: {inference_dir}")


def _run_train_mil_second_stage_clean(args):
    output_dir = fit_mil_second_stage_clean_audio(
        args.run_name,
        args.experiment,
        reduced=not args.full,
        model_filename=args.model_filename,
        bag_batch_size=args.bag_batch_size,
    )
    print(f"MIL clean-audio second-stage outputs saved to: {output_dir}")


def _run_build_mil_second_stage_clean_data(args):
    output_dir = build_mil_second_stage_clean_audio_data(
        args.run_name,
        args.experiment,
        soundscape=args.soundscapes,
        reduced=not args.full,
        model_filename=args.model_filename,
        bag_batch_size=args.bag_batch_size,
    )
    print(f"MIL clean-audio features saved to: {output_dir}")


def _run_train_mil_second_stage_clean_model(args):
    output_dir = train_mil_second_stage_clean_audio(
        args.run_name,
        args.experiment,
        model_filename=args.model_filename,
    )
    print(f"MIL clean-audio model outputs saved to: {output_dir}")


def _run_mil_second_stage_clean_soundscape(args):
    output_path = run_mil_second_stage_clean_audio_soundscape_inference(
        args.run_name,
        args.experiment,
        model_filename=args.model_filename,
    )
    print(f"MIL clean-audio soundscape inference saved to: {output_path}")


def _run_soundscape_mil_context_oof(args, paths: ProjectPaths):
    soundscapes = pd.read_parquet(paths.soundscapes_processed)
    output_path = fit_second_stage_soundscapes_with_mil_proba(
        soundscapes,
        args.run_name,
        args.experiment,
        reduced=not args.full,
        model_filename=args.model_filename,
        n_splits=args.n_splits,
        shuffle=not args.no_shuffle,
        random_state=args.random_state,
    )
    print(f"Soundscape MIL context OOF saved to: {output_path}")


def _run_soundscape_second_stage(args, paths: ProjectPaths):
    soundscapes = pd.read_parquet(paths.soundscapes_processed)
    output_dir = fit_second_stage_soundscapes(
        soundscapes,
        args.run_name,
        args.experiment,
        reduced=not args.full,
        model_filename=args.model_filename,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
    )
    print(f"Soundscape second-stage outputs saved to: {output_dir}")


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    paths = load_project_paths(args.project_config)

    if args.command == "preprocess-datasets-for-models":
        _run_preprocess_datasets_for_models(paths)
        return 0
    if args.command == "spectral-gating":
        _run_spectral_gating(args, paths)
        print("Spectral gating completed.")
        return 0
    if args.command == "trim-train-silence":
        _run_trim_train_silence(args, paths)
        print("Train silence trimming completed.")
        return 0
    if args.command == "extract-mel-spectograms":
        _run_extract_mel_spectograms(args, paths)
        return 0
    if args.command == "build-profiles":
        _run_build_profiles(args, paths)
        return 0
    if args.command == "build-feature-matrices":
        _run_build_feature_matrices(args, paths)
        return 0
    if args.command == "reduce-feature-matrices":
        _run_reduce_feature_matrices(args, paths)
        return 0
    if args.command == "train-ovr-models-chunks":
        _run_train_ovr_models_chunks(args)
        return 0
    if args.command == "run-ovr-inference":
        _run_ovr_inference(args)
        return 0
    if args.command == "train-mil-second-stage-clean":
        _run_train_mil_second_stage_clean(args)
        return 0
    if args.command == "build-mil-second-stage-clean-data":
        _run_build_mil_second_stage_clean_data(args)
        return 0
    if args.command == "train-mil-second-stage-clean-model":
        _run_train_mil_second_stage_clean_model(args)
        return 0
    if args.command == "run-mil-second-stage-clean-soundscape":
        _run_mil_second_stage_clean_soundscape(args)
        return 0
    if args.command == "run-soundscape-mil-context-oof":
        _run_soundscape_mil_context_oof(args, paths)
        return 0
    if args.command == "run-soundscape-second-stage":
        _run_soundscape_second_stage(args, paths)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
