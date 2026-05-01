from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


RUN_STEPS = [
    "00_check_environment",
    "01_inspect_data",
    "02_prepare_data",
    "03_run_eda",
    "04_prepare_colab_training",
    "05_check_colab_outputs",
    "06_evaluate_model",
    "07_generate_visualizations",
    "08_generate_report",
]


def timestamp_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def update_run_status(
    run_status_path: Path, step: str, status: str, notes: str
) -> None:
    rows = []
    if run_status_path.exists():
        rows = _parse_run_status(run_status_path)
    if not rows:
        rows = [
            {
                "Step": name,
                "Status": "Pending",
                "Updated At": "-",
                "Notes": "Not started",
            }
            for name in RUN_STEPS
        ]
    step_names = {row["Step"] for row in rows}
    for name in RUN_STEPS:
        if name not in step_names:
            rows.append(
                {
                    "Step": name,
                    "Status": "Pending",
                    "Updated At": "-",
                    "Notes": "Not started",
                }
            )
    for row in rows:
        if row["Step"] == step:
            row["Status"] = status
            row["Updated At"] = timestamp_string()
            row["Notes"] = notes
            break
    ordered_rows = sorted(rows, key=lambda item: RUN_STEPS.index(item["Step"]))
    df = pd.DataFrame(ordered_rows)
    markdown = "# RUN STATUS\n\n" + dataframe_to_markdown(df, max_rows=len(df)) + "\n"
    run_status_path.write_text(markdown, encoding="utf-8")


def write_markdown_report(path: Path, lines: Iterable[str]) -> None:
    content = "\n".join(lines).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_Empty dataframe_"
    preview = df.head(max_rows).copy()
    headers = [str(column) for column in preview.columns]
    divider = ["---"] * len(headers)
    rows = [headers, divider]
    for _, row in preview.iterrows():
        rows.append([_escape_markdown_cell(value) for value in row.tolist()])
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def _parse_run_status(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        return []
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _escape_markdown_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")
