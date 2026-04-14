import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROC_DATA_ROOT = PROJECT_ROOT / "data" / "processed"


def _resolve_data_root(data_root):
    # Optional override via env var for machine-specific dataset location.
    root_value = data_root
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def get_paths(raw_data_root, proc_data_root):
    raw_root = _resolve_data_root(raw_data_root)
    proc_root = _resolve_data_root(proc_data_root)

    return {
        "project_root": PROJECT_ROOT,
        "raw_data_root": raw_root,
        "proc_data_root": proc_root,

        # classes sounds : metadata + audio
        "raw_train": raw_root / "train.csv",
        "proc_train": proc_root / "train.parquet",
        "train_audio_dir": raw_root / "train_audio",

        # soundscapes : metadata + audio
        "raw_soundscape_labels": raw_root / "train_soundscapes_labels.csv",
        "proc_soundscape_labels": proc_root / "train_soundscapes_labels.parquet",
        "train_soundscapes_dir": raw_root / "train_soundscapes",

        # hierarchy
        "taxonomy": raw_root / "taxonomy.csv",

        # test dataset
        "test_soundscapes_dir": raw_root / "test_soundscapes",
        "sample_submission": raw_root / "sample_submission.csv",
    }


PATHS = get_paths(RAW_DATA_ROOT, PROC_DATA_ROOT)
