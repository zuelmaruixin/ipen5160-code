from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import ensure_project_directories, get_project_paths, load_project_config
from src.tokenization_utils import build_aspect_pair_text, load_tokenizer_with_local_fallback


TOKEN_THRESHOLDS = (256, 384, 512)


def analyze_token_lengths(config_path: str | Path | None = None) -> Dict[str, Path]:
    config = load_project_config(Path(config_path) if config_path is not None else None)
    paths = get_project_paths(config)
    ensure_project_directories(paths)

    tokenizer = load_tokenizer_with_local_fallback(config["model"]["encoder_name"])
    analysis_max_length = int(config["model"]["max_length"])

    split_frames: Dict[str, pd.DataFrame] = {}
    distribution_rows: List[Dict[str, object]] = []
    truncated_frames: List[pd.DataFrame] = []

    for split_name in ("train", "val", "test"):
        split_path = paths.processed_dir / f"{split_name}.csv"
        if not split_path.exists():
            continue
        frame = pd.read_csv(split_path)
        frame["token_length"] = _batch_compute_token_lengths(frame, tokenizer)
        split_frames[split_name] = frame

        distribution_rows.append(
            _summarize_lengths(
                frame["token_length"],
                split=split_name,
                sentiment_label="ALL",
                scope="overall",
            )
        )
        for sentiment_label, group in frame.groupby("sentiment_label"):
            distribution_rows.append(
                _summarize_lengths(
                    group["token_length"],
                    split=split_name,
                    sentiment_label=str(sentiment_label),
                    scope="by_sentiment",
                )
            )

        truncated_split = frame.loc[frame["token_length"] > analysis_max_length, [
            "review_id",
            "review_text",
            "aspect_name",
            "aspect_description",
            "sentiment_label",
            "rating",
            "token_length",
        ]].copy()
        if not truncated_split.empty:
            truncated_frames.append(truncated_split)

    distribution_df = pd.DataFrame(distribution_rows)
    distribution_path = paths.tables_dir / "token_length_distribution.csv"
    distribution_df.to_csv(distribution_path, index=False)

    truncated_examples_path = paths.tables_dir / "truncated_examples.csv"
    if truncated_frames:
        truncated_examples = (
            pd.concat(truncated_frames, ignore_index=True)
            .sort_values("token_length", ascending=False)
            .drop_duplicates(subset=["review_id", "aspect_name", "sentiment_label"])
            .head(500)
        )
    else:
        truncated_examples = pd.DataFrame(
            columns=[
                "review_id",
                "review_text",
                "aspect_name",
                "aspect_description",
                "sentiment_label",
                "rating",
                "token_length",
            ]
        )
    truncated_examples.to_csv(truncated_examples_path, index=False)

    return {
        "distribution": distribution_path,
        "truncated_examples": truncated_examples_path,
    }


def _batch_compute_token_lengths(
    frame: pd.DataFrame,
    tokenizer,
    batch_size: int = 1024,
) -> List[int]:
    lengths: List[int] = []
    total = len(frame)
    if total == 0:
        return lengths

    for start in tqdm(
        range(0, total, batch_size),
        desc="Token length scan",
        leave=False,
    ):
        batch = frame.iloc[start : start + batch_size]
        review_texts = batch["review_text"].astype(str).tolist()
        pair_texts = [
            build_aspect_pair_text(aspect_name, aspect_description)
            for aspect_name, aspect_description in zip(
                batch["aspect_name"].tolist(),
                batch["aspect_description"].tolist(),
            )
        ]
        encoded = tokenizer(
            review_texts,
            pair_texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return lengths


def _summarize_lengths(
    series: Iterable[int],
    *,
    split: str,
    sentiment_label: str,
    scope: str,
) -> Dict[str, object]:
    values = np.asarray(list(series), dtype=np.int32)
    result: Dict[str, object] = {
        "scope": scope,
        "split": split,
        "sentiment_label": sentiment_label,
        "sample_count": int(values.size),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "p75": float(np.percentile(values, 75)) if values.size else 0.0,
        "p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "p99": float(np.percentile(values, 99)) if values.size else 0.0,
        "max": int(values.max()) if values.size else 0,
    }
    for threshold in TOKEN_THRESHOLDS:
        ratio = float(np.mean(values > threshold)) if values.size else 0.0
        result[f"gt_{threshold}_ratio"] = ratio
    return result
