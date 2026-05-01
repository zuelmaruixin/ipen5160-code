from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inspect_data import run_data_inspection, schema_is_clear
from src.report_utils import update_run_status


def main() -> None:
    schema = run_data_inspection()
    if not schema_is_clear(schema):
        update_run_status(
            PROJECT_ROOT / "RUN_STATUS.md",
            "01_inspect_data",
            "Blocked",
            "Schema is unclear. Manual confirmation is required before data preparation.",
        )
        raise SystemExit(
            "Schema is unclear. Please review outputs/reports/data_schema_report.md and confirm the fields."
        )
    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "01_inspect_data",
        "Completed",
        "Schema report generated. Task selected: aspect-level sentiment classification with rating prediction.",
    )
    print("Schema report generated: outputs/reports/data_schema_report.md")


if __name__ == "__main__":
    main()

