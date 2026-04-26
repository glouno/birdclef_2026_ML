
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

_env_project_root = os.getenv("DATA_ROOT")
if _env_project_root:
    DATA_ROOT = Path(_env_project_root).expanduser().resolve()
else:
    DATA_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_ROOT = DATA_ROOT / "data" / "raw"
PROC_DATA_ROOT = DATA_ROOT / "data" / "processed"
CONFIGS_ROOT = DATA_ROOT / "data" / "configs"
MODELS_ROOT = DATA_ROOT / "models"


def _resolve_data_root(data_root):
    root_value = data_root
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (DATA_ROOT / root).resolve()
    return root


def get_path(pathroot):
    return PATHS[pathroot]


def get_paths(raw_data_root, proc_data_root):
    raw_root = _resolve_data_root(raw_data_root)
    proc_root = _resolve_data_root(proc_data_root)

    return {
        "project_root": DATA_ROOT,
        "raw_data_root": raw_root,
        "proc_data_root": proc_root,

        # classes sounds : metadata + audio
        "raw_train": raw_root / "train.csv",
        "proc_train": proc_root / "train.parquet",
        "train_audio_dir": raw_root / "train_audio",
        "train_audio_spectral_gating_dir": proc_root / "spectral_gating" / "train_audio",
        "train_audio_features_dir": proc_root / "features" / "train_audio",

        # soundscapes : metadata + audio
        "raw_soundscapes": raw_root / "train_soundscapes_labels.csv",
        "proc_soundscapes": proc_root / "train_soundscapes_labels.parquet",
        "train_soundscapes_dir": raw_root / "train_soundscapes",
        "train_soundscapes_spectral_gating_dir": proc_root / "spectral_gating" / "train_soundscapes",
        "train_soundscapes_features_dir": proc_root / "features" / "train_soundscapes",

        # hierarchy
        "taxonomy": raw_root / "taxonomy.csv",

        # test dataset
        "test_soundscapes_dir": raw_root / "test_soundscapes",
        "sample_submission": raw_root / "sample_submission.csv",

        # Everything models-related
        "models": MODELS_ROOT,
        "configs": CONFIGS_ROOT,
        "profiles": proc_root / "profiles",
        "label_encoders": proc_root / "label_encoders",
    }


PATHS = get_paths(RAW_DATA_ROOT, PROC_DATA_ROOT)
