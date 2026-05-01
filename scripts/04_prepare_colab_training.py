from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from pathlib import Path

import nbformat as nbf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_project_directories, get_project_paths, load_project_config, load_schema_summary
from src.report_utils import update_run_status


PAUSE_MESSAGE = """============================================================
PAUSE: GPU TRAINING REQUIRED
============================================================

本地数据处理、EDA 和 Colab 训练文件已经准备完成。

请将以下文件夹上传到 Google Colab：

outputs/colab_package/

然后打开并运行：

notebooks/02_colab_training.ipynb

训练完成后，请下载 Colab 中的 outputs 文件夹，并将以下文件放回本地对应路径：

outputs/models/final_model/
outputs/tables/model_metrics.csv
outputs/tables/classification_report.csv
outputs/predictions/model_predictions.csv
outputs/figures/training_loss_curve.png
outputs/figures/confusion_matrix.png
outputs/logs/train_log.json

如果有 rating prediction，请同时放回：

outputs/tables/rating_prediction_metrics.csv
outputs/figures/rating_prediction_actual_vs_predicted.png

放回本地后，运行：

python scripts/05_check_colab_outputs.py
python scripts/06_evaluate_model.py
python scripts/07_generate_visualizations.py
python scripts/08_generate_report.py

在 Colab 输出完整放回本地之前，不要继续深度模型结果分析。
============================================================"""


def main() -> None:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)
    schema = load_schema_summary(paths)
    experiment_configs = _write_length_experiment_configs(config)

    notebook_path = PROJECT_ROOT / "notebooks" / "02_colab_training.ipynb"
    _generate_colab_notebook(notebook_path, schema, experiment_configs)
    _build_colab_package(paths)
    _write_colab_readme(paths.colab_package_dir)

    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "04_prepare_colab_training",
        "Completed",
        "Colab notebook and upload package generated. Waiting for GPU training.",
    )
    print(PAUSE_MESSAGE)


def _generate_colab_notebook(output_path: Path, schema: dict, experiment_configs: dict[str, str]) -> None:
    model_name = "BERT + Aspect Sentiment Head + Rating Prediction Head"
    if schema["supports_token_level_sequence_labeling"]:
        model_name = "BERT + CRF + Rating Prediction Head"
    config_literal = "{\n" + ",\n".join(
        f"    {key!r}: Path({value!r})" for key, value in experiment_configs.items()
    ) + "\n}"
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Colab Training Notebook\n\n"
            f"Detected task: `{schema['recommended_task']}`\n\n"
            f"Selected training design: **{model_name}**"
        ),
        nbf.v4.new_code_cell(
            "!pip install PyYAML scikit-learn seaborn transformers accelerate TorchCRF"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n\n"
            "MANUAL_PROJECT_ROOT = None\n\n"
            "def is_project_root(path: Path) -> bool:\n"
            "    return (\n"
            "        path.is_dir()\n"
            "        and (path / 'configs' / 'project_config.yaml').exists()\n"
            "        and (path / 'src').exists()\n"
            "    )\n\n"
            "def candidate_roots(base: Path):\n"
            "    if not base.exists() or not base.is_dir():\n"
            "        return []\n"
            "    roots = [base]\n"
            "    level1 = [child for child in sorted(base.iterdir()) if child.is_dir()]\n"
            "    roots.extend(level1)\n"
            "    for child in level1:\n"
            "        try:\n"
            "            roots.extend(grand for grand in sorted(child.iterdir()) if grand.is_dir())\n"
            "        except PermissionError:\n"
            "            pass\n"
            "    return roots\n\n"
            "print('Locating uploaded project package...')\n"
            "project_root = None\n"
            "if MANUAL_PROJECT_ROOT:\n"
            "    manual_path = Path(MANUAL_PROJECT_ROOT)\n"
            "    print(f'Trying manual path: {manual_path}')\n"
            "    if is_project_root(manual_path):\n"
            "        project_root = manual_path\n"
            "    else:\n"
            "        raise FileNotFoundError(f'Manual path is not a valid project root: {manual_path}')\n\n"
            "search_bases = [\n"
            "    Path('/content'),\n"
            "    Path('/content/drive/MyDrive'),\n"
            "]\n"
            "preferred_candidates = [\n"
            "    Path('/content/colab_package'),\n"
            "    Path('/content/ipen5160_meituan_nlp'),\n"
            "    Path('/content/drive/MyDrive/colab_package'),\n"
            "    Path('/content/drive/MyDrive/ipen5160_meituan_nlp'),\n"
            "]\n\n"
            "if project_root is None:\n"
            "    print('Step 1/2: checking common upload locations...')\n"
            "    for candidate in preferred_candidates:\n"
            "        print(f'  Checking {candidate}')\n"
            "        if is_project_root(candidate):\n"
            "            project_root = candidate\n"
            "            print('  Found project root via common location check.')\n"
            "            break\n\n"
            "if project_root is None:\n"
            "    print('Step 2/2: scanning shallow directory levels under common bases...')\n"
            "    for base in search_bases:\n"
            "        print(f'  Base: {base}')\n"
            "        for candidate in candidate_roots(base):\n"
            "            print(f'    Trying {candidate}')\n"
            "            if is_project_root(candidate):\n"
            "                project_root = candidate\n"
            "                print('    Found project root during shallow scan.')\n"
            "                break\n"
            "        if project_root is not None:\n"
            "            break\n\n"
            "if project_root is None:\n"
            "    raise FileNotFoundError(\n"
            "        'Could not locate the uploaded project package. '\n"
            "        'Put `colab_package` under /content or /content/drive/MyDrive, '\n"
            "        'or set MANUAL_PROJECT_ROOT explicitly in this cell.'\n"
            "    )\n"
            "os.chdir(project_root)\n"
            "sys.path.insert(0, str(project_root))\n"
            "print('PROJECT_ROOT =', project_root)\n"
        ),
        nbf.v4.new_code_cell(
            "import yaml\n"
            "from pathlib import Path\n\n"
            "config = yaml.safe_load((Path('configs') / 'project_config.yaml').read_text(encoding='utf-8'))\n"
            "schema = yaml.safe_load((Path('data') / 'interim' / 'schema_summary.yaml').read_text(encoding='utf-8'))\n"
            "print('Encoder:', config['model']['encoder_name'])\n"
            "print('Task:', schema['recommended_task'])\n"
            "print('Supports CRF:', schema['supports_token_level_sequence_labeling'])\n"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from src.token_analysis import analyze_token_lengths\n\n"
            "token_outputs = analyze_token_lengths()\n"
            "display(pd.read_csv(token_outputs['distribution']).head(20))\n"
            "display(pd.read_csv(token_outputs['truncated_examples']).head(20))\n"
        ),
        nbf.v4.new_code_cell(
            "import yaml\n"
            "from pathlib import Path\n\n"
            f"AVAILABLE_CONFIGS = {config_literal}\n"
            "SELECTED_CONFIG_KEY = 'maxlen_256'\n"
            "SELECTED_CONFIG_PATH = AVAILABLE_CONFIGS[SELECTED_CONFIG_KEY]\n"
            "selected_config = yaml.safe_load(SELECTED_CONFIG_PATH.read_text(encoding='utf-8'))\n"
            "print('Selected config:', SELECTED_CONFIG_PATH)\n"
            "print('max_length =', selected_config['model']['max_length'])\n"
            "print('use_dynamic_padding =', selected_config['model'].get('use_dynamic_padding', True))\n"
            "print('pad_to_multiple_of =', selected_config['model'].get('pad_to_multiple_of'))\n"
            "print('Training order recommendation: run maxlen_256 first; only try 384 if macro-F1 does not improve enough and >256 truncation is material.')\n"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import shutil\n\n"
            "for path in [\n"
            "    Path('outputs/models/checkpoints'),\n"
            "    Path('outputs/models/final_model'),\n"
            "    Path('outputs/logs/train_log.json'),\n"
            "    Path('outputs/tables/model_metrics.csv'),\n"
            "    Path('outputs/tables/classification_report.csv'),\n"
            "    Path('outputs/tables/rating_prediction_metrics.csv'),\n"
            "    Path('outputs/tables/epoch_history.csv'),\n"
            "    Path('outputs/predictions/model_predictions.csv'),\n"
            "    Path('outputs/figures/training_loss_curve.png'),\n"
            "    Path('outputs/figures/confusion_matrix.png'),\n"
            "    Path('outputs/figures/rating_prediction_actual_vs_predicted.png'),\n"
            "]:\n"
            "    if path.is_dir():\n"
            "        shutil.rmtree(path, ignore_errors=True)\n"
            "    elif path.exists():\n"
            "        path.unlink()\n"
            "print('Old checkpoints and outputs cleared.')\n"
        ),
        nbf.v4.new_code_cell(
            "from src.train_colab import main\n\n"
            "main(config_path=SELECTED_CONFIG_PATH)"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import shutil\n\n"
            "exp_dir = Path('outputs/experiments') / SELECTED_CONFIG_KEY\n"
            "exp_dir.mkdir(parents=True, exist_ok=True)\n"
            "for sub in ['models', 'logs', 'tables', 'figures', 'predictions']:\n"
            "    src = Path('outputs') / sub\n"
            "    dst = exp_dir / sub\n"
            "    if dst.exists():\n"
            "        shutil.rmtree(dst)\n"
            "    shutil.copytree(src, dst)\n"
            "print('Archived outputs to', exp_dir)\n"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "for path in sorted(Path('outputs').rglob('*')):\n"
            "    print(path)\n"
        ),
    ]
    output_path.write_text(nbf.writes(nb), encoding="utf-8")


def _write_length_experiment_configs(base_config: dict) -> dict[str, str]:
    experiments_dir = PROJECT_ROOT / "configs" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    config_map: dict[str, str] = {}
    for max_length in (256, 384, 512):
        experiment_config = deepcopy(base_config)
        experiment_config["model"]["max_length"] = max_length
        output_path = experiments_dir / f"maxlen_{max_length}.yaml"
        output_path.write_text(
            yaml.safe_dump(experiment_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        config_map[f"maxlen_{max_length}"] = str(output_path.relative_to(PROJECT_ROOT))
    return config_map


def _build_colab_package(paths) -> None:
    package_dir = paths.colab_package_dir
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    items_to_copy = [
        paths.project_root / "configs",
        paths.project_root / "src",
        paths.project_root / "scripts",
        paths.project_root / "requirements.txt",
        paths.project_root / "notebooks" / "02_colab_training.ipynb",
        paths.project_root / "data" / "processed",
        paths.project_root / "data" / "interim",
    ]
    for item in items_to_copy:
        destination = package_dir / item.relative_to(paths.project_root)
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)

    for rel_path in [
        "outputs/models",
        "outputs/logs",
        "outputs/predictions",
        "outputs/tables",
        "outputs/figures",
    ]:
        (package_dir / rel_path).mkdir(parents=True, exist_ok=True)


def _write_colab_readme(package_dir: Path) -> None:
    readme = package_dir / "README_COLAB.md"
    readme.write_text(
        "# README for Colab\n\n"
        "1. Upload the whole `colab_package` directory to Colab or Google Drive.\n"
        "2. Recommended locations: `/content/colab_package` or `/content/drive/MyDrive/colab_package`.\n"
        "3. If the notebook cannot auto-detect the root, set `MANUAL_PROJECT_ROOT` in the notebook setup cell.\n"
        "4. Open `notebooks/02_colab_training.ipynb`.\n"
        "5. Run the token length analysis cell first. It will write `outputs/tables/token_length_distribution.csv` and `outputs/tables/truncated_examples.csv`.\n"
        "6. Keep `SELECTED_CONFIG_KEY = 'maxlen_256'` for the first run. Only switch to `maxlen_384` or `maxlen_512` after checking truncation ratios and macro-F1.\n"
        "7. After training, download the generated `outputs/` files and place them back into the local project.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
