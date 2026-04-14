import argparse
from pathlib import Path

import pandas as pd

from birdclef_2026_ml.paths import PATHS
from birdclef_2026_ml.preprocess import (
    preprocess_train_for_models,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="birdclef", description="BirdCLEF data utilities")
    subparsers = parser.add_subparsers(dest="command")

    # def add_common_args(cmd_parser: argparse.ArgumentParser):
    #     cmd_parser.add_argument("--input", type=str, default=None, help="Input CSV path")
    #     cmd_parser.add_argument("--output", type=str, default=None, help="Output Parquet path")

    parser_train_models = subparsers.add_parser(
        "preprocess-train-for-models",
        help="Preprocess train.csv for modeling",
    )
    # add_common_args(parser_train_models)

    return parser


def _run_preprocess_train_for_models(args) -> Path:
    input_path = PATHS["raw_train"]
    output_path = PATHS["proc_train"]

    df = pd.read_csv(input_path)
    out = preprocess_train_for_models(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return output_path


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "preprocess-train-for-models":
        output_path = _run_preprocess_train_for_models(args)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(f"Wrote: {output_path}")
    return 0
