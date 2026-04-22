import argparse
from pathlib import Path

import pandas as pd

from birdclef_2026_ml.paths import PATHS
from birdclef_2026_ml.processing.preprocess import (
    preprocess_train_for_models,
    preprocess_soundscapes_for_models
)
from birdclef_2026_ml.constants import SAMPLE_RATE

from birdclef_2026_ml.audio import apply_spectral_gating


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="birdclef", description="BirdCLEF data utilities")
    subparsers = parser.add_subparsers(dest="command")

    parser_spectral_gating = subparsers.add_parser(
        "spectral-gating",
        help="Apply spectral gating to audio files and save cleaned audio."
    )
    parser_spectral_gating.add_argument("--df-input-path", type=str, required=True,
                                        help="Key in PATHS for input DataFrame (parquet)")
    parser_spectral_gating.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate")
    parser_spectral_gating.add_argument("--n-fft", type=int, default=1024, help="FFT window size")
    parser_spectral_gating.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT")
    parser_spectral_gating.add_argument("--noise-percentile", type=float,
                                        default=15.0, help="Noise percentile for gating")
    parser_spectral_gating.add_argument("--alpha", type=float, default=3.0, help="Alpha for gating strength")
    parser_spectral_gating.add_argument("--smooth-freq", type=int, default=3, help="Smoothing in frequency bins")
    parser_spectral_gating.add_argument("--smooth-time", type=int, default=3, help="Smoothing in time frames")
    parser_spectral_gating.add_argument("--eps", type=float, default=1e-8, help="Epsilon for numerical stability")
    parser_spectral_gating.add_argument(
        "--input-root", type=str, required=True, help="Key in PATHS for input audio root")
    parser_spectral_gating.add_argument(
        "--output-root", type=str, required=True, help="Key in PATHS for output audio root")
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


def _run_spectral_gating(args):
    df_input_path = PATHS[args.df_input_path]
    df = pd.read_parquet(df_input_path)
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    apply_spectral_gating(
        df,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        noise_percentile=args.noise_percentile,
        alpha=args.alpha,
        smooth_freq=args.smooth_freq,
        smooth_time=args.smooth_time,
        eps=args.eps,
        input_root=args.input_root,
        output_root=args.output_root,
        filename_col=args.filename_col,
    )


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "preprocess-train-for-models":
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
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2
