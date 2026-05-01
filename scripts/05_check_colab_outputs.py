from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_paths, load_project_config, load_schema_summary
from src.report_utils import update_run_status, write_markdown_report


def main() -> None:
    config = load_project_config()
    paths = get_project_paths(config)
    schema = load_schema_summary(paths)

    required_paths = [
        paths.models_dir / "final_model",
        paths.tables_dir / "model_metrics.csv",
        paths.tables_dir / "classification_report.csv",
        paths.predictions_dir / "model_predictions.csv",
        paths.figures_dir / "training_loss_curve.png",
        paths.figures_dir / "confusion_matrix.png",
        paths.logs_dir / "train_log.json",
    ]
    if schema["supports_rating_prediction"]:
        required_paths.extend(
            [
                paths.tables_dir / "rating_prediction_metrics.csv",
                paths.figures_dir / "rating_prediction_actual_vs_predicted.png",
            ]
        )

    missing = [path for path in required_paths if not path.exists()]
    lines = ["# Colab Output Check", ""]
    if missing:
        lines.extend(["## Missing Files", "", *[f"- `{path}`" for path in missing], ""])
        status = "Blocked"
        notes = f"Missing {len(missing)} required Colab output files."
    else:
        lines.extend(["All required Colab outputs are present.", ""])
        status = "Completed"
        notes = "All required Colab outputs found."

    write_markdown_report(paths.reports_dir / "colab_output_check.md", lines)
    update_run_status(PROJECT_ROOT / "RUN_STATUS.md", "05_check_colab_outputs", status, notes)
    print("\n".join(lines))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

