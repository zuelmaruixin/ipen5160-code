from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_dataset import prepare_processed_datasets
from src.report_utils import update_run_status


def main() -> None:
    outputs = prepare_processed_datasets()
    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "02_prepare_data",
        "Completed",
        f"Processed datasets written. Main files: train={outputs['train'].name}, val={outputs['val'].name}, test={outputs['test'].name}.",
    )
    print("Processed datasets created in data/processed/")


if __name__ == "__main__":
    main()

