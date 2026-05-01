from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from src.config import ensure_project_directories, get_project_paths, load_project_config
from src.error_analysis import (
    build_confused_label_pairs,
    build_error_examples,
    build_low_confidence_examples,
    summarize_error_patterns,
)


LABEL_ORDER = ["Positive", "Neutral", "Negative", "Not-Mentioned"]


def run_evaluation() -> Dict[str, Path]:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)

    pred_df = pd.read_csv(paths.predictions_dir / "model_predictions.csv")
    test_df = pred_df[pred_df["split"] == "test"].copy()

    final_metrics = _build_final_metrics_summary(test_df)
    error_examples = build_error_examples(test_df)
    confused_pairs = build_confused_label_pairs(test_df)
    low_confidence = build_low_confidence_examples(test_df)

    pattern_summary, _ = summarize_error_patterns(test_df)
    final_metrics = pd.concat([final_metrics, pattern_summary], ignore_index=True)

    outputs = {
        "final_metrics_summary.csv": paths.tables_dir / "final_metrics_summary.csv",
        "error_examples.csv": paths.tables_dir / "error_examples.csv",
        "confused_label_pairs.csv": paths.tables_dir / "confused_label_pairs.csv",
        "low_confidence_examples.csv": paths.tables_dir / "low_confidence_examples.csv",
    }
    final_metrics.to_csv(outputs["final_metrics_summary.csv"], index=False)
    error_examples.to_csv(outputs["error_examples.csv"], index=False)
    confused_pairs.to_csv(outputs["confused_label_pairs.csv"], index=False)
    low_confidence.to_csv(outputs["low_confidence_examples.csv"], index=False)
    return outputs


def _build_final_metrics_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    sentiment_true = test_df["sentiment_true"]
    sentiment_pred = test_df["sentiment_pred"]
    rating_true = test_df["rating_true"].astype(float)
    rating_pred = test_df["rating_pred"].astype(float)
    metrics = [
        {"metric": "test_accuracy", "value": round(float(accuracy_score(sentiment_true, sentiment_pred)), 4)},
        {
            "metric": "test_macro_f1",
            "value": round(
                float(f1_score(sentiment_true, sentiment_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
                4,
            ),
        },
        {
            "metric": "test_weighted_f1",
            "value": round(
                float(f1_score(sentiment_true, sentiment_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)),
                4,
            ),
        },
        {"metric": "rating_mae", "value": round(float(mean_absolute_error(rating_true, rating_pred)), 4)},
        {"metric": "rating_rmse", "value": round(float(math.sqrt(mean_squared_error(rating_true, rating_pred))), 4)},
        {
            "metric": "rating_rounded_accuracy",
            "value": round(
                float(
                    accuracy_score(
                        rating_true.round().astype(int),
                        rating_pred.round().clip(1, 5).astype(int),
                    )
                ),
                4,
            ),
        },
    ]
    return pd.DataFrame(metrics)

