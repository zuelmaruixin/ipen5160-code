from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


@dataclass
class ProjectPaths:
    project_root: Path
    external_data_dir: Path
    outputs_dir: Path
    data_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    tables_dir: Path
    models_dir: Path
    logs_dir: Path
    predictions_dir: Path
    reports_dir: Path
    colab_package_dir: Path
    notebooks_dir: Path


def load_project_config(config_path: Path | None = None) -> Dict[str, Any]:
    config_path = config_path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_project_paths(config: Dict[str, Any] | None = None) -> ProjectPaths:
    config = config or load_project_config()
    project_root = PROJECT_ROOT
    outputs_dir = project_root / config["paths"]["outputs_dir"]
    data_dir = project_root / "data"
    return ProjectPaths(
        project_root=project_root,
        external_data_dir=Path(config["paths"]["external_data_dir"]).expanduser(),
        outputs_dir=outputs_dir,
        data_dir=data_dir,
        interim_dir=data_dir / "interim",
        processed_dir=data_dir / "processed",
        figures_dir=outputs_dir / "figures",
        tables_dir=outputs_dir / "tables",
        models_dir=outputs_dir / "models",
        logs_dir=outputs_dir / "logs",
        predictions_dir=outputs_dir / "predictions",
        reports_dir=outputs_dir / "reports",
        colab_package_dir=outputs_dir / "colab_package",
        notebooks_dir=project_root / "notebooks",
    )


def ensure_project_directories(paths: ProjectPaths) -> None:
    directories = [
        paths.data_dir / "raw",
        paths.interim_dir,
        paths.processed_dir,
        paths.outputs_dir,
        paths.figures_dir,
        paths.tables_dir,
        paths.models_dir,
        paths.logs_dir,
        paths.predictions_dir,
        paths.reports_dir,
        paths.colab_package_dir,
        paths.notebooks_dir,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def load_schema_summary(paths: ProjectPaths) -> Dict[str, Any]:
    schema_path = paths.interim_dir / "schema_summary.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema summary not found: {schema_path}. Run scripts/01_inspect_data.py first."
        )
    with schema_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

