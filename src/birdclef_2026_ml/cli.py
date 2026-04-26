import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from birdclef_2026_ml.processing.audio_utils import build_audio_path, load_audio, load_config
from birdclef_2026_ml.audio import apply_spectral_gating, build_profile
from birdclef_2026_ml.configs import PipelineConfig, SpectralGatingConfig
from birdclef_2026_ml.feature_engineering import save_features_from_audio_dir
from birdclef_2026_ml.processing.preprocess import preprocess_datasets_for_models
from birdclef_2026_ml.processing.feature_dataset_builder import build_memmap_from_chunks
from birdclef_2026_ml.paths import PATHS, DATA_ROOT, MODELS_ROOT
from birdclef_2026_ml.processing.data_split import split_audio_train_val, split_soundscapes_train_val


def _build_parser() -> argparse.ArgumentParser:
    # Optionally add more arguments for configs if needed
    parser = argparse.ArgumentParser(prog="birdclef", description="BirdCLEF data utilities")
    subparsers = parser.add_subparsers(dest="command")

    parser_build_feature_matrices = subparsers.add_parser(
        "build-feature-matrices",
        help="Build features matrices for train/validation datasets."
    )
    parser_build_feature_matrices.add_argument(
        "--input-df",
        type=str,
        required=True,
        help="Key in PATHS for preprocessed input df"
    )
    parser_build_feature_matrices.add_argument(
        "--input-audio-dir",
        type=str,
        required=True,
        help="Key in PATHS for input audio files directory."
    )
    parser_build_feature_matrices.add_argument(
        "--input-features-dir",
        type=str,
        required=True,
        help="Key in PATHS for input audio extracted features (.joblib) directory."
    )
    parser_build_feature_matrices.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder inside paths.MODELS_ROOT where to save extracted matrices."
    )
    parser_build_feature_matrices.add_argument(
        "--soundscapes",
        action="store_true",
        help="If set, process soundscapes; otherwise, process clean train audio."
    )

    parser_mil_bags = subparsers.add_parser(
        "train-val-mil-bags",
        help="Create MIL bags for train/val datasets and save all artifacts."
    )
    parser_mil_bags.add_argument(
        "--input-audio-dir",
        type=str,
        required=True,
        help="Key in PATHS for input audio files directory."
    )
    parser_mil_bags.add_argument(
        "--n-splits",
        type=int,
        default=50,
        help="Number of splits"
    )
    parser_mil_bags.add_argument(
        "--features-pathroot",
        type=str,
        default=None,
        help="Optional key in PATHS to precomputed per-audio feature .joblib files.",
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

    subparsers.add_parser(
        "preprocess-datasets-for-models",
        help="Preprocess train.csv and soundscapes for modeling",
    )

    parser_mel = subparsers.add_parser(
        "extract-all-features",
        help="Extract all features for all audios in a PATHS directory and save .joblib files.",
    )
    parser_mel.add_argument(
        "--input-df",
        type=str,
        required=True,
        help="Key in PATHS for input audios dataframe.",
    )
    parser_mel.add_argument(
        "--input-audio-dir",
        type=str,
        required=True,
        help="Key in PATHS for input audios directory.",
    )
    parser_mel.add_argument(
        "--output-audio-dir",
        type=str,
        required=True,
        help="Optional key in PATHS for output directory. Defaults to data/processed/features.",
    )

    return parser


def _run_preprocess_datasets_for_models(args):
    train = pd.read_csv(PATHS["raw_train"])
    soundscapes = pd.read_csv(PATHS["raw_soundscapes"])
    taxonomy = pd.read_csv(PATHS["taxonomy"])

    train_proc, ss_proc, le_pl, le_cn = preprocess_datasets_for_models(train, soundscapes, taxonomy)

    # Ensure output directories exist
    for key in ("proc_train", "proc_soundscapes"):
        PATHS[key].parent.mkdir(parents=True, exist_ok=True)
    PATHS["label_encoders"].mkdir(parents=True, exist_ok=True)

    # Save processed data
    # train_proc.to_parquet(PATHS["proc_train"], index=False)
    ss_proc.to_parquet(PATHS["proc_soundscapes"], index=False)
    # Save label encoders
    label_encoders_path = Path(PATHS["label_encoders"])
    joblib.dump(le_pl, label_encoders_path / "primary_label.joblib")
    joblib.dump(le_cn, label_encoders_path / "class_name.joblib")


def _run_build_profiles(args):
    # Load train DataFrame
    train = pd.read_parquet(PATHS[args.input_train_key])

    # Load config
    config_path = Path(args.spectral_gating_config_path)
    cfg_dict = load_config(config_path)
    spectral_gating_config = SpectralGatingConfig(**cfg_dict)

    profiles = []
    for idx in train.index:
        y_clean_path = build_audio_path(train, idx, pathroot=PATHS["train_audio_spectral_gating_dir"])
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


# def _run_train_val_mil_bags(args):
#     df = pd.read_parquet(PATHS["proc_train"])
#     save_train_val_mil_bags_streaming(df,
#                                       pathroot=args.input_audio_dir,
#                                       n_splits=args.n_splits,
#                                       features_pathroot=args.features_pathroot)


def _run_extract_all_features(args):
    pipeline_cfg = PipelineConfig()
    df = pd.read_parquet(PATHS[args.input_df])

    n_saved = save_features_from_audio_dir(
        df=df,
        input_path=PATHS[args.input_audio_dir],
        config_path=DATA_ROOT / "configs",
        pipeline_cfg=pipeline_cfg,
        output_path=PATHS[args.output_audio_dir],
    )


def _run_build_feature_matrices(args):
    df = pd.read_parquet(PATHS[args.input_df])

    pipeline_cfg = PipelineConfig()
    pathroot = PATHS[args.input_audio_dir]
    features_pathroot = PATHS[args.input_features_dir]

    if args.soundscapes:
        df_train, df_val = split_soundscapes_train_val(df)
        subsets = {
            "train_soundscapes": df_train,
            "val_soundscapes": df_val,
        }
    else:
        df_train, df_val = split_audio_train_val(df)
        subsets = {
            "train": df_train,
            "val": df_val,
        }

    for name, df_subset in subsets.items():
        print(f"Started processing {name} subset")

        out_path = MODELS_ROOT / args.output_folder / name

        build_memmap_from_chunks(
            df_subset,
            pipeline_cfg=pipeline_cfg,
            pathroot=pathroot,
            filename_col="filename",
            features_pathroot=features_pathroot,
            out_instances_path=out_path,
            soundscapes=args.soundscapes,
        )

        print(f"Finished processing {name} subset")


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "build-feature-matrices":
        _run_build_feature_matrices(args)
        return 0
    elif args.command == "preprocess-datasets-for-models":
        _run_preprocess_datasets_for_models(args)
        return 0
    elif args.command == "spectral-gating":
        _run_spectral_gating(args)
        print("Spectral gating completed.")
        return 0
    elif args.command == "build-profiles":
        _run_build_profiles(args)
        return 0
    elif args.command == "extract-all-features":
        _run_extract_all_features(args)
        return 0
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2
