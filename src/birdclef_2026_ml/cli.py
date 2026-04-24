import argparse

from birdclef_2026_ml.processing.audio_utils import build_audio_path, load_audio, load_config
from birdclef_2026_ml.audio import apply_spectral_gating, build_profile
from birdclef_2026_ml.feature_engineering.configs import SpectralGatingConfig
from birdclef_2026_ml.processing.preprocess import (
    preprocess_train_for_models,
    preprocess_soundscapes_for_models
)
from birdclef_2026_ml.models.model_training import (
    save_train_val_mil_bags_streaming
)
from birdclef_2026_ml.paths import PATHS
import pandas as pd
import numpy as np
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    # Optionally add more arguments for configs if needed
    parser = argparse.ArgumentParser(prog="birdclef", description="BirdCLEF data utilities")
    subparsers = parser.add_subparsers(dest="command")

    parser_mil_bags = subparsers.add_parser(
        "train-val-mil-bags",
        help="Create MIL bags for train/val datasets and save all artifacts."
    )
    parser_mil_bags.add_argument(
        "--input-audio-dir",
        type=str,
        default="train_audio_spectral_gating_dir",
        help="Key in PATHS for input audio files directory."
    )
    parser_mil_bags.add_argument(
        "--n-splits",
        type=int,
        default=50,
        help="Number of splits"
    )

    parser_profiles = subparsers.add_parser(
        "build-profiles",
        help="Build all audio profiles and save to all_profiles.npy."
    )
    parser_profiles.add_argument(
        "--output-path-key",
        type=str,
        default="profiles",
        help="Key in PATHS for output profiles directory."
    )
    parser_profiles.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bands."
    )
    parser_profiles.add_argument(
        "--input-train-key",
        type=str,
        default="proc_train",
        help="Key in PATHS for processed train parquet."
    )
    parser_profiles.add_argument(
        "--spectral-gating-config-path",
        type=str,
        default=None,
        help="Path to config.json for spectral gating (optional, overrides default)."
    )

    parser_spectral_gating = subparsers.add_parser(
        "spectral-gating",
        help="Apply spectral gating to audio files and save cleaned audio."
    )
    parser_spectral_gating.add_argument("--df-input-path", type=str, required=True,
                                        help="Key in PATHS for input DataFrame (parquet)")
    parser_spectral_gating.add_argument("--config-path", type=str, required=True,
                                        help="Path to config.json file with all spectral gating parameters.")
    parser_spectral_gating.add_argument("--input-root", type=str, required=True,
                                        help="Key in PATHS for input audio root")
    parser_spectral_gating.add_argument("--output-root", type=str, required=True,
                                        help="Key in PATHS for output audio root")
    parser_spectral_gating.add_argument("--filename-col", type=str, default="filename",
                                        help="Column name for filenames in DataFrame")

    # def add_common_args(cmd_parser: argparse.ArgumentParser):
    #     cmd_parser.add_argument("--input", type=str, default=None, help="Input CSV path")
    #     cmd_parser.add_argument("--output", type=str, default=None, help="Output Parquet path")

    parser_train_models = subparsers.add_parser(
        "preprocess-train-for-models",
        help="Preprocess train.csv for modeling",
    )
    # add_common_args(parser_train_models)

    parser_soundscapes_models = subparsers.add_parser(
        "preprocess-soundscapes-for-models",
        help="Preprocess train_soundscapes_labels.csv for modeling",
    )
    # add_common_args(parser_soundscapes_models)

    return parser


def _run_preprocess_train_for_models(args) -> Path:
    input_path = PATHS["raw_train"]
    output_path = PATHS["proc_train"]

    df = pd.read_csv(input_path)
    out = preprocess_train_for_models(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return output_path


def _run_preprocess_soundscapes_for_models(args) -> Path:
    input_path = PATHS["raw_soundscapes"]
    output_path = PATHS["proc_soundscapes"]

    df = pd.read_csv(input_path)
    out = preprocess_soundscapes_for_models(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return output_path


def _run_build_profiles(args):
    # Load train DataFrame
    train = pd.read_parquet(PATHS[args.input_train_key])

    # Load config
    config_path = Path(args.spectral_gating_config_path)
    cfg_dict = load_config(config_path)
    spectral_gating_config = SpectralGatingConfig(**cfg_dict)

    profiles = []
    for idx in train.index:
        y_clean_path = build_audio_path(train, idx, pathroot="train_audio_spectral_gating_dir")
        y_clean = load_audio(y_clean_path, spectral_gating_config.sr)
        profile = build_profile(
            y_clean,
            sr=spectral_gating_config.sr,
            n_mels=args.n_mels,
            n_fft=spectral_gating_config.n_fft,
            hop_length=spectral_gating_config.hop_length
        )
        profiles.append(profile)

    profiles = np.array(profiles)
    output_path = PATHS[args.output_path_key] / "all_profiles.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, profiles)
    print(f"Profiles saved to: {output_path}")


def _run_spectral_gating(args):
    df_input_path = PATHS[args.df_input_path]
    df = pd.read_parquet(df_input_path)
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    cfg_dict = load_config(Path(args.config_path))
    config = SpectralGatingConfig(**cfg_dict)

    apply_spectral_gating(
        df,
        config,
        input_root=args.input_root,
        output_root=args.output_root,
        filename_col=args.filename_col,
    )


def _run_train_val_mil_bags(args):
    df = pd.read_parquet(PATHS["proc_train"])
    save_train_val_mil_bags_streaming(df,
                                      pathroot=args.input_audio_dir,
                                      n_splits=args.n_splits)


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train-val-mil-bags":
        _run_train_val_mil_bags(args)
        return 0
    elif args.command == "preprocess-train-for-models":
        output_path = _run_preprocess_train_for_models(args)
        print(f"Wrote: {output_path}")
        return 0
    elif args.command == "preprocess-soundscapes-for-models":
        output_path = _run_preprocess_soundscapes_for_models(args)
        print(f"Wrote: {output_path}")
        return 0
    elif args.command == "spectral-gating":
        _run_spectral_gating(args)
        print("Spectral gating completed.")
        return 0
    elif args.command == "build-profiles":
        _run_build_profiles(args)
        return 0
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2
