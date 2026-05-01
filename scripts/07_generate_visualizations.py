from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualize import generate_visualizations
from src.report_utils import update_run_status


def main() -> None:
    outputs = generate_visualizations()
    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "07_generate_visualizations",
        "Completed",
        f"Generated visualizations: {', '.join(outputs.keys())}.",
    )
    print("Visualization generation complete.")


if __name__ == "__main__":
    main()

