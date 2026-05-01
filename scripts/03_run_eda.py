from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import run_eda
from src.report_utils import update_run_status
from src.token_analysis import analyze_token_lengths


def main() -> None:
    run_eda()
    token_outputs = None
    token_note = "Token length analysis completed."
    try:
        token_outputs = analyze_token_lengths()
        token_note = (
            f"Token stats={token_outputs['distribution'].name}, "
            f"truncated examples={token_outputs['truncated_examples'].name}."
        )
    except OSError as exc:
        token_note = (
            "Token length analysis skipped locally because the tokenizer files "
            f"were unavailable in the local cache: {exc}"
        )
    _generate_eda_notebook(PROJECT_ROOT / "notebooks" / "01_eda.ipynb")
    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "03_run_eda",
        "Completed",
        f"EDA tables/figures generated. {token_note}",
    )
    print(
        "EDA completed. "
        f"{token_note} "
        "See outputs/tables, outputs/figures, and notebooks/01_eda.ipynb"
    )


def _generate_eda_notebook(output_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# ASAP EDA\n\nThis notebook previews the processed aspect-level dataset and the generated EDA outputs."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "project_root = Path.cwd().resolve().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "tables_dir = project_root / 'outputs' / 'tables'\n"
            "figures_dir = project_root / 'outputs' / 'figures'\n"
            "display(pd.read_csv(tables_dir / 'dataset_summary.csv'))\n"
            "display(pd.read_csv(tables_dir / 'token_length_distribution.csv').head(12))\n"
            "display(pd.read_csv(tables_dir / 'rating_distribution.csv'))\n"
            "display(pd.read_csv(tables_dir / 'aspect_distribution.csv').head(10))\n"
        ),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(tables_dir / 'truncated_examples.csv').head(20))\n"
        ),
        nbf.v4.new_code_cell(
            "for name in [\n"
            "    'rating_distribution.png',\n"
            "    'aspect_distribution.png',\n"
            "    'sentiment_distribution.png',\n"
            "    'aspect_negative_rate.png',\n"
            "    'text_length_distribution.png',\n"
            "]:\n"
            "    display(Image(filename=str(figures_dir / name)))\n"
        ),
    ]
    output_path.write_text(nbf.writes(nb), encoding="utf-8")


if __name__ == "__main__":
    main()
