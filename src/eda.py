from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ensure_project_directories, get_project_paths, load_project_config
from src.report_utils import dataframe_to_markdown, write_markdown_report


sns.set_theme(style="whitegrid")


def run_eda() -> Dict[str, Path]:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)

    review_df = pd.read_csv(paths.processed_dir / "review_level_all.csv")
    long_df = pd.read_csv(paths.processed_dir / "aspect_sentiment_long.csv")

    outputs = {}
    dataset_summary = _dataset_summary(review_df, long_df)
    rating_distribution = _rating_distribution(review_df)
    aspect_distribution = _aspect_distribution(long_df)
    sentiment_distribution = _sentiment_distribution(long_df)
    aspect_negative_rate = _aspect_negative_rate(long_df)
    text_length_summary = _text_length_summary(review_df)
    sample_reviews = _sample_reviews(review_df, long_df, config["eda"]["sample_review_count"])

    tables = {
        "dataset_summary.csv": dataset_summary,
        "rating_distribution.csv": rating_distribution,
        "aspect_distribution.csv": aspect_distribution,
        "sentiment_distribution.csv": sentiment_distribution,
        "aspect_negative_rate.csv": aspect_negative_rate,
        "text_length_summary.csv": text_length_summary,
        "sample_reviews.csv": sample_reviews,
    }
    for filename, table in tables.items():
        path = paths.tables_dir / filename
        table.to_csv(path, index=False)
        outputs[filename] = path

    _plot_rating_distribution(rating_distribution, paths.figures_dir / "rating_distribution.png")
    _plot_bar(aspect_distribution, "aspect_name", "mentioned_count", "Mentioned Count", paths.figures_dir / "aspect_distribution.png")
    _plot_bar(sentiment_distribution, "sentiment_label", "count", "Count", paths.figures_dir / "sentiment_distribution.png")
    _plot_bar(aspect_negative_rate, "aspect_name", "negative_rate", "Negative Rate", paths.figures_dir / "aspect_negative_rate.png")
    _plot_text_length_distribution(review_df, paths.figures_dir / "text_length_distribution.png")

    outputs.update(
        {
            "rating_distribution.png": paths.figures_dir / "rating_distribution.png",
            "aspect_distribution.png": paths.figures_dir / "aspect_distribution.png",
            "sentiment_distribution.png": paths.figures_dir / "sentiment_distribution.png",
            "aspect_negative_rate.png": paths.figures_dir / "aspect_negative_rate.png",
            "text_length_distribution.png": paths.figures_dir / "text_length_distribution.png",
        }
    )

    summary_lines = _eda_summary_markdown(
        dataset_summary=dataset_summary,
        rating_distribution=rating_distribution,
        aspect_distribution=aspect_distribution,
        sentiment_distribution=sentiment_distribution,
        aspect_negative_rate=aspect_negative_rate,
        text_length_summary=text_length_summary,
    )
    write_markdown_report(paths.reports_dir / "eda_summary.md", summary_lines)
    return outputs


def _dataset_summary(review_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    mentioned_ratio = long_df["is_mentioned"].mean()
    summary = [
        ("total_reviews", len(review_df)),
        ("total_aspect_rows", len(long_df)),
        ("total_aspects", long_df["aspect_name"].nunique()),
        ("mentioned_aspect_ratio", round(float(mentioned_ratio), 4)),
        ("avg_text_length", round(float(review_df["text_length"].mean()), 2)),
        ("median_text_length", round(float(review_df["text_length"].median()), 2)),
    ]
    return pd.DataFrame(summary, columns=["metric", "value"])


def _rating_distribution(review_df: pd.DataFrame) -> pd.DataFrame:
    counts = review_df["rating"].astype(int).value_counts().sort_index()
    return pd.DataFrame(
        {
            "rating": counts.index,
            "count": counts.values,
            "percentage": (counts.values / counts.sum()).round(4),
        }
    )


def _aspect_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    mentioned = long_df[long_df["is_mentioned"] == 1]
    counts = mentioned["aspect_name"].value_counts()
    total_reviews = long_df["review_id"].nunique()
    result = pd.DataFrame(
        {
            "aspect_name": counts.index,
            "mentioned_count": counts.values,
        }
    )
    result["mention_rate_per_review"] = (result["mentioned_count"] / total_reviews).round(4)
    return result


def _sentiment_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    counts = long_df["sentiment_label"].value_counts()
    ordered_labels = ["Positive", "Neutral", "Negative", "Not-Mentioned"]
    rows = []
    total = counts.sum()
    for label in ordered_labels:
        count = int(counts.get(label, 0))
        rows.append(
            {
                "sentiment_label": label,
                "count": count,
                "percentage": round(count / total, 4),
            }
        )
    return pd.DataFrame(rows)


def _aspect_negative_rate(long_df: pd.DataFrame) -> pd.DataFrame:
    mentioned = long_df[long_df["is_mentioned"] == 1].copy()
    grouped = mentioned.groupby("aspect_name")
    result = grouped["sentiment_label"].agg(
        mentioned_count="count",
        negative_count=lambda series: int((series == "Negative").sum()),
    ).reset_index()
    result["negative_rate"] = (
        result["negative_count"] / result["mentioned_count"]
    ).round(4)
    return result.sort_values("negative_rate", ascending=False).reset_index(drop=True)


def _text_length_summary(review_df: pd.DataFrame) -> pd.DataFrame:
    quantiles = review_df["text_length"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    return pd.DataFrame(
        [
            {"statistic": "min", "value": int(quantiles.loc[0.0])},
            {"statistic": "p25", "value": int(quantiles.loc[0.25])},
            {"statistic": "median", "value": int(quantiles.loc[0.5])},
            {"statistic": "mean", "value": round(float(review_df["text_length"].mean()), 2)},
            {"statistic": "p75", "value": int(quantiles.loc[0.75])},
            {"statistic": "max", "value": int(quantiles.loc[1.0])},
        ]
    )


def _sample_reviews(
    review_df: pd.DataFrame, long_df: pd.DataFrame, sample_count: int
) -> pd.DataFrame:
    sample_ids = review_df.sort_values("text_length", ascending=False).head(sample_count)["review_id"]
    samples = long_df[
        (long_df["review_id"].isin(sample_ids)) & (long_df["is_mentioned"] == 1)
    ].copy()
    grouped = samples.groupby("review_id").agg(
        rating=("rating", "first"),
        review_text=("review_text", "first"),
        aspects=("aspect_name", lambda series: ", ".join(sorted(set(series)))),
        sentiments=("sentiment_label", lambda series: ", ".join(series.astype(str).tolist())),
    )
    grouped = grouped.reset_index()
    return grouped.head(sample_count)


def _plot_rating_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="rating", y="count", color="#4C78A8")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    order = df.sort_values(y_col, ascending=False)[x_col]
    sns.barplot(data=df, x=x_col, y=y_col, order=order, color="#72B7B2")
    plt.title(output_path.stem.replace("_", " ").title())
    plt.xlabel("")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_text_length_distribution(review_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(review_df["text_length"], bins=40, color="#F58518")
    plt.title("Text Length Distribution")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _eda_summary_markdown(
    dataset_summary: pd.DataFrame,
    rating_distribution: pd.DataFrame,
    aspect_distribution: pd.DataFrame,
    sentiment_distribution: pd.DataFrame,
    aspect_negative_rate: pd.DataFrame,
    text_length_summary: pd.DataFrame,
) -> list[str]:
    not_mentioned_pct = float(
        sentiment_distribution.loc[
            sentiment_distribution["sentiment_label"] == "Not-Mentioned", "percentage"
        ].iloc[0]
    )
    top_aspects = aspect_distribution.head(5)
    hardest_risk = aspect_negative_rate.head(5)
    lines = [
        "# EDA Summary",
        "",
        "## Key Findings",
        "",
        f"- Dataset size: {int(dataset_summary.loc[dataset_summary['metric'] == 'total_reviews', 'value'].iloc[0])} reviews.",
        f"- Aspect rows after wide-to-long conversion: {int(dataset_summary.loc[dataset_summary['metric'] == 'total_aspect_rows', 'value'].iloc[0])}.",
        f"- `Not-Mentioned` share: {not_mentioned_pct:.2%}. This is high and should be handled explicitly instead of being ignored.",
        "- The dataset is suitable for aspect-level sentiment classification with a rating prediction auxiliary task.",
        "- It is not suitable for sequence labeling because there are no token-level spans or BIO tags.",
        "",
        "## Rating Distribution",
        "",
        dataframe_to_markdown(rating_distribution, max_rows=len(rating_distribution)),
        "",
        "## Most Frequently Mentioned Aspects",
        "",
        dataframe_to_markdown(top_aspects, max_rows=len(top_aspects)),
        "",
        "## Highest Negative-Rate Aspects",
        "",
        dataframe_to_markdown(hardest_risk, max_rows=len(hardest_risk)),
        "",
        "## Text Length Summary",
        "",
        dataframe_to_markdown(text_length_summary, max_rows=len(text_length_summary)),
        "",
    ]
    return lines
