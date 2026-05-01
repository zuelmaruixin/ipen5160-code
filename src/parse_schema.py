from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from src.report_utils import dataframe_to_markdown


ASPECT_DESCRIPTION_MAP = {
    "Location#Transportation": "交通便利程度",
    "Location#Downtown": "距离商圈或核心区域的便利程度",
    "Location#Easy_to_find": "门店是否容易找到",
    "Service#Queue": "排队等待体验",
    "Service#Hospitality": "服务员态度与热情",
    "Service#Parking": "停车便利性",
    "Service#Timely": "上菜与服务及时性",
    "Price#Level": "价格水平",
    "Price#Cost_effective": "性价比",
    "Price#Discount": "优惠力度",
    "Ambience#Decoration": "装修与环境氛围",
    "Ambience#Noise": "噪音控制",
    "Ambience#Space": "空间感与座位舒适度",
    "Ambience#Sanitary": "卫生与清洁程度",
    "Food#Portion": "菜品分量",
    "Food#Taste": "菜品口味",
    "Food#Appearance": "菜品卖相",
    "Food#Recommend": "是否值得推荐",
}


def scan_data_directory(data_dir: Path) -> Dict[str, Any]:
    all_files = sorted(path for path in data_dir.rglob("*") if path.is_file())
    doc_files = [
        path
        for path in all_files
        if path.suffix.lower() in {".md", ".txt"} or path.name.lower().startswith("readme")
    ]
    data_files = [
        path
        for path in all_files
        if path.suffix.lower() in {".csv", ".json", ".jsonl", ".xlsx", ".tsv", ".parquet"}
    ]
    return {
        "all_files": all_files,
        "doc_files": doc_files,
        "data_files": data_files,
    }


def read_documents(doc_files: Iterable[Path]) -> Dict[str, str]:
    docs = {}
    for path in doc_files:
        content = ""
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        docs[str(path)] = content
    return docs


def inspect_csv_file(path: Path, sample_rows: int = 3) -> Dict[str, Any]:
    df = pd.read_csv(path)
    return {
        "path": str(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "sample_records": df.head(sample_rows).to_dict(orient="records"),
    }


def infer_schema(
    data_files: list[Path],
    document_texts: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    csv_files = [path for path in data_files if path.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError("No CSV files were found in the provided data directory.")

    file_summaries = [inspect_csv_file(path) for path in csv_files]
    preferred_files = config["dataset"]["preferred_files"]
    selected_files = {}
    for split_name, file_name in preferred_files.items():
        matched = next((path for path in csv_files if path.name == file_name), None)
        if matched is not None:
            selected_files[split_name] = matched
    if set(selected_files) != {"train", "val", "test"}:
        ranked = sorted(
            [summary for summary in file_summaries if "_sample" not in Path(summary["path"]).name],
            key=lambda item: item["rows"],
            reverse=True,
        )
        if not ranked:
            raise ValueError("Could not identify full dataset files.")
        selected_files = {"train": Path(ranked[0]["path"])}
        if len(ranked) > 1:
            selected_files["val"] = Path(ranked[1]["path"])
        if len(ranked) > 2:
            selected_files["test"] = Path(ranked[2]["path"])

    train_df = pd.read_csv(selected_files["train"])
    id_col = _pick_column(train_df.columns, config["dataset"]["id_column_candidates"])
    text_col = _pick_column(train_df.columns, config["dataset"]["text_column_candidates"])
    rating_col = _pick_column(train_df.columns, config["dataset"]["rating_column_candidates"])
    excluded = {id_col, text_col, rating_col}
    aspect_cols = [column for column in train_df.columns if column not in excluded]
    label_values = sorted(
        {
            int(value)
            for column in aspect_cols
            for value in pd.Series(train_df[column]).dropna().unique().tolist()
        }
    )

    docs_blob = "\n".join(document_texts.values()).lower()
    supports_span = any(keyword in docs_blob for keyword in ["bio", "span", "token-level", "sequence labeling"])
    has_span_columns = any(
        keyword in column.lower()
        for column in train_df.columns
        for keyword in ["bio", "label", "token", "start", "end", "span"]
    )
    supports_crf = supports_span and has_span_columns

    star_distribution = Counter(train_df[rating_col].astype(int).tolist())
    total_reviews = 0
    for path in selected_files.values():
        total_reviews += len(pd.read_csv(path))

    schema = {
        "selected_files": {split: str(path) for split, path in selected_files.items()},
        "file_summaries": file_summaries,
        "document_texts": document_texts,
        "review_id_column": id_col,
        "text_column": text_col,
        "rating_column": rating_col,
        "aspect_columns": aspect_cols,
        "aspect_descriptions": {
            column: ASPECT_DESCRIPTION_MAP.get(column, column.replace("#", " / "))
            for column in aspect_cols
        },
        "label_mapping": {
            int(raw): label for raw, label in config["dataset"]["label_mapping"].items()
        },
        "label_values_detected": label_values,
        "supports_token_level_sequence_labeling": bool(supports_crf),
        "supports_rating_prediction": True,
        "supports_aspect_sentiment_classification": True,
        "recommended_task": (
            "bert_crf_with_rating"
            if supports_crf
            else "bert_aspect_sentiment_multitask_with_rating"
        ),
        "why_no_crf": (
            ""
            if supports_crf
            else "README only describes aspect category sentiment labels and star ratings; "
            "CSV files contain review-level text plus 18 aspect-category columns, but no BIO tags, "
            "token list, entity span, or start/end offsets."
        ),
        "total_reviews": total_reviews,
        "train_star_distribution": dict(sorted(star_distribution.items())),
        "documents_found": list(document_texts.keys()),
    }
    return schema


def schema_summary_markdown(
    schema: Dict[str, Any], data_dir: Path, project_title: str
) -> list[str]:
    selected_files = schema["selected_files"]
    file_table = pd.DataFrame(schema["file_summaries"])[["path", "rows", "columns"]]
    aspect_df = pd.DataFrame(
        {
            "aspect_name": schema["aspect_columns"],
            "aspect_description": [
                schema["aspect_descriptions"][name] for name in schema["aspect_columns"]
            ],
        }
    )
    document_list = "\n".join(f"- `{path}`" for path in schema["documents_found"]) or "- None"
    data_file_list = "\n".join(f"- `{path}`" for path in file_table["path"].tolist())

    lines = [
        f"# Data Schema Report - {project_title}",
        "",
        "## 1. Data Directory",
        "",
        f"- External data path: `{data_dir}`",
        f"- Documents found: {len(schema['documents_found'])}",
        f"- Data files found: {len(file_table)}",
        "",
        "### Documentation Files",
        "",
        document_list,
        "",
        "### Data Files",
        "",
        data_file_list,
        "",
        "## 2. File Summary",
        "",
        dataframe_to_markdown(file_table, max_rows=len(file_table)),
        "",
        "## 3. Selected Files",
        "",
        f"- Train: `{selected_files.get('train', '')}`",
        f"- Validation: `{selected_files.get('val', '')}`",
        f"- Test: `{selected_files.get('test', '')}`",
        "- Decision: use the full official split files instead of the `_sample.csv` files because the sample files only contain 100 rows each.",
        "",
        "## 4. Field Meaning",
        "",
        f"- Review ID field: `{schema['review_id_column']}`",
        f"- Review text field: `{schema['text_column']}`",
        f"- Rating field: `{schema['rating_column']}`",
        "- Aspect sentiment fields: the remaining 18 columns correspond to predefined restaurant aspects.",
        "",
        "### Aspect Columns",
        "",
        dataframe_to_markdown(aspect_df, max_rows=len(aspect_df)),
        "",
        "### Sentiment Label Meaning",
        "",
        "- `1 -> Positive`",
        "- `0 -> Neutral`",
        "- `-1 -> Negative`",
        "- `-2 -> Not-Mentioned`",
        "",
        "## 5. Sample Data",
        "",
    ]
    sample_file = next(
        summary
        for summary in schema["file_summaries"]
        if Path(summary["path"]).name == Path(selected_files["train"]).name
    )
    sample_df = pd.DataFrame(sample_file["sample_records"])
    lines.extend([dataframe_to_markdown(sample_df, max_rows=len(sample_df)), ""])
    lines.extend(
        [
            "## 6. Task Feasibility Decision",
            "",
            f"- Supports BERT+CRF sequence labeling: `{schema['supports_token_level_sequence_labeling']}`",
            f"- Supports aspect sentiment classification: `{schema['supports_aspect_sentiment_classification']}`",
            f"- Supports rating prediction: `{schema['supports_rating_prediction']}`",
            f"- Recommended task: `{schema['recommended_task']}`",
            "",
        ]
    )
    if schema["supports_token_level_sequence_labeling"]:
        lines.extend(
            [
                "The dataset includes token-level annotations, so sequence labeling is supported.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Why BERT+CRF Is Not Appropriate Here",
                "",
                schema["why_no_crf"],
                "",
            ]
        )
    lines.extend(
        [
            "## 7. Final Modeling Plan",
            "",
            "- Use `train.csv`, `dev.csv`, and `test.csv` as the official split.",
            "- Convert the original wide table into aspect-level long format.",
            "- Model each `(review_text, aspect_name, aspect_description)` pair as one sentiment classification instance.",
            "- Use a BERT encoder with a sentiment classification head as the main task.",
            "- Use review star rating prediction as an auxiliary regression head with total loss `sentiment_loss + alpha * rating_loss`.",
            "- Do not build BIO labels or CRF layers because the data does not provide span supervision.",
            "",
        ]
    )
    return lines


def _pick_column(columns: Iterable[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Could not find any of the expected columns: {candidates}")
