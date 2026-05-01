import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    config_file: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    external_dir: Path
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path
    metadata_train: Path
    metadata_soundscapes: Path
    taxonomy: Path
    sample_submission: Path
    train_audio_dir: Path
    train_soundscapes_dir: Path
    test_soundscapes_dir: Path
    train_processed: Path
    soundscapes_processed: Path
    primary_to_class: Path
    spectral_gating_config: Path
    feature_config: Path
    train_audio_clean_dir: Path
    soundscapes_clean_dir: Path
    train_mel_dir: Path
    soundscapes_mel_dir: Path
    train_pooled_dir: Path
    soundscapes_pooled_dir: Path
    label_encoders_dir: Path
    profiles_dir: Path
    runs_dir: Path
    experiments_dir: Path

    def dataset_frame(self, dataset: str) -> Path:
        mapping = {
            "train": self.train_processed,
            "soundscapes": self.soundscapes_processed,
        }
        return mapping[dataset]

    def audio_dir(self, dataset: str, stage: str) -> Path:
        mapping = {
            ("train", "raw"): self.train_audio_dir,
            ("train", "clean"): self.train_audio_clean_dir,
            ("soundscapes", "raw"): self.train_soundscapes_dir,
            ("soundscapes", "clean"): self.soundscapes_clean_dir,
            ("test_soundscapes", "raw"): self.test_soundscapes_dir,
        }
        return mapping[(dataset, stage)]

    def feature_dir(self, dataset: str, kind: str) -> Path:
        mapping = {
            ("train", "mel"): self.train_mel_dir,
            ("train", "pooled"): self.train_pooled_dir,
            ("soundscapes", "mel"): self.soundscapes_mel_dir,
            ("soundscapes", "pooled"): self.soundscapes_pooled_dir,
        }
        return mapping[(dataset, kind)]

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    def run_data_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name / "data"

    def experiment_pipeline_path(self, run_name: str) -> Path:
        return self.experiments_dir / run_name / "pipeline.yaml"

    def experiment_config_path(self, run_name: str, experiment_name: str) -> Path:
        return self.experiments_dir / run_name / f"{experiment_name}.yaml"

    def experiment_dir(self, run_name: str, experiment_name: str) -> Path:
        return self.run_dir(run_name) / "experiments" / experiment_name


def _build_paths(config_data: dict[str, Any], *, project_root: Path, config_file: Path) -> ProjectPaths:
    storage = config_data["storage"]
    metadata = config_data["datasets"]["metadata"]
    audio = config_data["datasets"]["audio"]
    processed = config_data["datasets"]["processed"]
    configs = config_data["configs"]

    raw_dir = _resolve_path(storage["raw_dir"], project_root=project_root)
    interim_dir = _resolve_path(storage["interim_dir"], project_root=project_root)
    processed_dir = _resolve_path(storage["processed_dir"], project_root=project_root)
    external_dir = _resolve_path(storage["external_dir"], project_root=project_root)
    artifacts_dir = _resolve_path(storage["artifacts_dir"], project_root=project_root)
    models_dir = _resolve_path(storage["models_dir"], project_root=project_root)
    reports_dir = _resolve_path(storage["reports_dir"], project_root=project_root)

    return ProjectPaths(
        project_root=project_root,
        config_file=config_file,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        external_dir=external_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,

        metadata_train=raw_dir / metadata["train"],
        metadata_soundscapes=raw_dir / metadata["soundscapes"],
        taxonomy=raw_dir / metadata["taxonomy"],
        sample_submission=raw_dir / metadata["sample_submission"],
        train_audio_dir=raw_dir / audio["train"],
        train_soundscapes_dir=raw_dir / audio["soundscapes"],
        test_soundscapes_dir=raw_dir / audio["test_soundscapes"],

        train_processed=processed_dir / processed["train"],
        soundscapes_processed=processed_dir / processed["soundscapes"],
        label_encoders_dir=processed_dir / "label_encoders",
        profiles_dir=processed_dir / "features" / "profiles",
        primary_to_class=processed_dir / "metadata",

        train_audio_clean_dir=interim_dir / audio["train_clean"],
        soundscapes_clean_dir=interim_dir / audio["soundscapes_clean"],
        train_mel_dir=interim_dir / audio["train_mel"],
        soundscapes_mel_dir=interim_dir / audio["soundscapes_mel"],
        train_pooled_dir=interim_dir / audio["train_pooled"],
        soundscapes_pooled_dir=interim_dir / audio["soundscapes_pooled"],

        runs_dir=models_dir / "runs",
        experiments_dir=models_dir / "runs" / "experiments",

        spectral_gating_config=_resolve_path(configs["spectral_gating"], project_root=project_root),
        feature_config=_resolve_path(configs["features"], project_root=project_root),
    )


def load_project_paths(config_path: str | Path | None = None) -> ProjectPaths:
    project_root_env = os.getenv("DATA_ROOT")
    project_root = _resolve_path(project_root_env, project_root=_default_project_root()
                                 ) if project_root_env else _default_project_root()

    config_env = os.getenv("BIRDCLEF_CONFIG")
    config_file = _resolve_path(
        config_path or config_env or "configs/project.yaml",
        project_root=project_root,
    )

    with open(config_file, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    return _build_paths(config_data, project_root=project_root, config_file=config_file)
