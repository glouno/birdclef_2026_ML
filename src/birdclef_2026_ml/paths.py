import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROC_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODELS_ROOT = PROJECT_ROOT / "models"


def _resolve_data_root(data_root):
    root_value = data_root
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def get_path(pathroot):
    return PATHS[pathroot]


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
        "train_audio_spectral_gating_dir": proc_root / "train_audio_spectral_gating",

        # soundscapes : metadata + audio
        "raw_soundscapes": raw_root / "train_soundscapes_labels.csv",
        "proc_soundscapes": proc_root / "train_soundscapes_labels.parquet",
        "train_soundscapes_dir": raw_root / "train_soundscapes",
        "train_soundscapes_spectral_gating_dir": proc_root / "train_soundscapes_spectral_gating",

        # hierarchy
        "taxonomy": raw_root / "taxonomy.csv",

        # test dataset
        "test_soundscapes_dir": raw_root / "test_soundscapes",
        "sample_submission": raw_root / "sample_submission.csv",

        "models": MODELS_ROOT,
        "proc_train_matrix": proc_root / "matrices"
    }


PATHS = get_paths(RAW_DATA_ROOT, PROC_DATA_ROOT)
