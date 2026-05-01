from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_project_directories, get_project_paths, load_project_config
from src.parse_schema import scan_data_directory
from src.report_utils import update_run_status, write_markdown_report


DEPENDENCIES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("nbformat", "nbformat"),
    ("tabulate", "tabulate"),
    ("transformers", "transformers"),
    ("torch", "torch"),
]


def main() -> None:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)

    scan_result = scan_data_directory(paths.external_data_dir) if paths.external_data_dir.exists() else None
    writable_ok = _check_writable(paths.outputs_dir)
    dep_rows = []
    missing = []
    for module_name, package_name in DEPENDENCIES:
        try:
            importlib.import_module(module_name)
            dep_rows.append(f"- {package_name}: installed")
        except ImportError:
            dep_rows.append(f"- {package_name}: missing")
            missing.append(package_name)
    if _has_crf_dependency():
        dep_rows.append("- TorchCRF: installed")
    else:
        dep_rows.append("- TorchCRF: missing")
        missing.append("TorchCRF")

    report_lines = [
        "# Environment Check",
        "",
        f"- Python version: `{platform.python_version()}`",
        f"- Executable: `{sys.executable}`",
        f"- Current working directory: `{os.getcwd()}`",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- External data path exists: `{paths.external_data_dir.exists()}`",
        f"- External data path readable: `{os.access(paths.external_data_dir, os.R_OK)}`",
        f"- Outputs writable: `{writable_ok}`",
        "",
        "## Documentation Scan",
        "",
        f"- Markdown / README / TXT files found: `{len(scan_result['doc_files']) if scan_result else 0}`",
        f"- Data files found: `{len(scan_result['data_files']) if scan_result else 0}`",
        "",
        "## Dependency Check",
        "",
        *dep_rows,
        "",
    ]
    if missing:
        report_lines.extend(
            [
                "## Action Needed",
                "",
                "Missing dependencies detected. Run:",
                "",
                "```bash",
                "pip install -r requirements.txt",
                "```",
                "",
            ]
        )
        status = "Completed with warnings"
        notes = f"Missing dependencies: {', '.join(missing)}"
    else:
        status = "Completed"
        notes = "Environment, data path, and dependencies look ready."

    write_markdown_report(paths.reports_dir / "environment_check.md", report_lines)
    update_run_status(PROJECT_ROOT / "RUN_STATUS.md", "00_check_environment", status, notes)
    print("\n".join(report_lines))


def _check_writable(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    test_path = directory / ".write_test"
    try:
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink()
        return True
    except OSError:
        return False


def _has_crf_dependency() -> bool:
    for module_name in ("torchcrf", "TorchCRF"):
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            continue
    return False


if __name__ == "__main__":
    main()
