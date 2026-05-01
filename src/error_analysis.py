from __future__ import annotations

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


LABEL_ORDER = ["Positive", "Neutral", "Negative", "Not-Mentioned"]


def build_error_examples(pred_df: pd.DataFrame, top_k: int = 200) -> pd.DataFrame:
    errors = pred_df[pred_df["sentiment_true"] != pred_df["sentiment_pred"]].copy()
    errors["rating_abs_error"] = (errors["rating_true"] - errors["rating_pred"]).abs()
    errors = errors.sort_values(["confidence", "rating_abs_error"], ascending=[True, False])
    columns = [
        "review_id",
        "aspect_name",
        "aspect_description",
        "sentiment_true",
        "sentiment_pred",
        "confidence",
        "rating_true",
        "rating_pred",
        "text_length",
        "review_text",
    ]
    return errors[columns].head(top_k)


def build_confused_label_pairs(pred_df: pd.DataFrame) -> pd.DataFrame:
    cm = confusion_matrix(
        pred_df["sentiment_true"],
        pred_df["sentiment_pred"],
        labels=LABEL_ORDER,
    )
    rows = []
    for i, j in product(range(len(LABEL_ORDER)), range(len(LABEL_ORDER))):
        if i == j or cm[i, j] == 0:
            continue
        rows.append(
            {
                "true_label": LABEL_ORDER[i],
                "pred_label": LABEL_ORDER[j],
                "count": int(cm[i, j]),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def build_low_confidence_examples(pred_df: pd.DataFrame, quantile: float = 0.05) -> pd.DataFrame:
    threshold = pred_df["confidence"].quantile(quantile)
    low_conf = pred_df[pred_df["confidence"] <= threshold].copy()
    return low_conf.sort_values("confidence").head(200)


def summarize_error_patterns(pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, float]]:
    error_df = pred_df.copy()
    error_df["is_error"] = (error_df["sentiment_true"] != error_df["sentiment_pred"]).astype(int)
    short_threshold = error_df["text_length"].quantile(0.25)
    short_error_rate = error_df.loc[error_df["text_length"] <= short_threshold, "is_error"].mean()
    long_error_rate = error_df.loc[error_df["text_length"] > short_threshold, "is_error"].mean()

    not_mentioned_confusion = error_df[
        ((error_df["sentiment_true"] == "Not-Mentioned") & (error_df["sentiment_pred"] == "Neutral"))
        | ((error_df["sentiment_true"] == "Neutral") & (error_df["sentiment_pred"] == "Not-Mentioned"))
    ]
    alignment = error_df.assign(
        rating_abs_error=(error_df["rating_true"] - error_df["rating_pred"]).abs(),
        rating_close=((error_df["rating_true"] - error_df["rating_pred"]).abs() <= 0.5).astype(int),
    )
    summary_df = pd.DataFrame(
        [
            {"metric": "overall_error_rate", "value": round(float(error_df["is_error"].mean()), 4)},
            {"metric": "short_text_error_rate", "value": round(float(short_error_rate), 4)},
            {"metric": "long_text_error_rate", "value": round(float(long_error_rate), 4)},
            {
                "metric": "not_mentioned_neutral_confusion_rate",
                "value": round(float(len(not_mentioned_confusion) / len(error_df)), 4),
            },
            {
                "metric": "sentiment_accuracy_when_rating_close",
                "value": round(
                    float(
                        1.0
                        - alignment.loc[alignment["rating_close"] == 1, "is_error"].mean()
                    ),
                    4,
                ),
            },
            {
                "metric": "sentiment_accuracy_when_rating_far",
                "value": round(
                    float(
                        1.0
                        - alignment.loc[alignment["rating_close"] == 0, "is_error"].mean()
                    ),
                    4,
                ),
            },
        ]
    )
    stats = {
        "short_threshold": float(short_threshold),
        "overall_error_rate": float(error_df["is_error"].mean()),
        "rating_error_correlation": float(
            np.corrcoef(
                error_df["is_error"].astype(float),
                (error_df["rating_true"] - error_df["rating_pred"]).abs().astype(float),
            )[0, 1]
        )
        if len(error_df) > 1
        else 0.0,
    }
    return summary_df, stats

