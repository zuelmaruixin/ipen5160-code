from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import run_evaluation
from src.report_utils import update_run_status


def main() -> None:
    outputs = run_evaluation()
    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "06_evaluate_model",
        "Completed",
        f"Evaluation tables generated: {', '.join(outputs.keys())}.",
    )
    print("Model evaluation complete.")


if __name__ == "__main__":
    main()

