from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.config import ensure_project_directories, get_project_paths, load_project_config
from src.parse_schema import infer_schema, read_documents, scan_data_directory, schema_summary_markdown
from src.report_utils import write_markdown_report


def run_data_inspection() -> Dict[str, Any]:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)

    scan_result = scan_data_directory(paths.external_data_dir)
    docs = read_documents(scan_result["doc_files"])
    schema = infer_schema(scan_result["data_files"], docs, config)

    schema_path = paths.interim_dir / "schema_summary.yaml"
    schema_path.write_text(
        yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report_lines = schema_summary_markdown(
        schema=schema,
        data_dir=paths.external_data_dir,
        project_title=config["project"]["title_en"],
    )
    report_path = paths.reports_dir / "data_schema_report.md"
    write_markdown_report(report_path, report_lines)
    return schema


def schema_is_clear(schema: Dict[str, Any]) -> bool:
    required_keys = [
        "review_id_column",
        "text_column",
        "rating_column",
        "aspect_columns",
        "recommended_task",
    ]
    return all(bool(schema.get(key)) for key in required_keys)

