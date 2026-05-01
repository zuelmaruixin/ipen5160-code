from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import ensure_project_directories, get_project_paths, load_project_config


sns.set_theme(style="whitegrid")
LABEL_ORDER = ["Positive", "Neutral", "Negative", "Not-Mentioned"]


def generate_visualizations() -> Dict[str, Path]:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)

    pred_df = pd.read_csv(paths.predictions_dir / "model_predictions.csv")
    test_df = pred_df[pred_df["split"] == "test"].copy()
    history = json.loads((paths.logs_dir / "train_log.json").read_text(encoding="utf-8"))["history"]

    outputs = {}
    outputs["confusion_matrix.png"] = _plot_confusion_matrix(test_df, paths.figures_dir / "confusion_matrix.png")
    outputs["training_loss_curve.png"] = _plot_training_curve(pd.DataFrame(history), paths.figures_dir / "training_loss_curve.png")
    outputs["rating_prediction_actual_vs_predicted.png"] = _plot_rating_scatter(test_df, paths.figures_dir / "rating_prediction_actual_vs_predicted.png")
    outputs["error_analysis_visualization.png"] = _plot_error_profile(test_df, paths.figures_dir / "error_analysis_visualization.png")
    return outputs


def _plot_confusion_matrix(pred_df: pd.DataFrame, output_path: Path) -> Path:
    cm = confusion_matrix(pred_df["sentiment_true"], pred_df["sentiment_pred"], labels=LABEL_ORDER)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
    plt.title("Sentiment Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_training_curve(history_df: pd.DataFrame, output_path: Path) -> Path:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=history_df, x="epoch", y="train_loss", marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_rating_scatter(pred_df: pd.DataFrame, output_path: Path) -> Path:
    sample_df = pred_df.sample(min(3000, len(pred_df)), random_state=42)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=sample_df, x="rating_true", y="rating_pred", s=18, alpha=0.5)
    plt.plot([1, 5], [1, 5], linestyle="--", color="black", linewidth=1)
    plt.title("Rating Prediction: Actual vs Predicted")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_error_profile(pred_df: pd.DataFrame, output_path: Path) -> Path:
    plot_df = pred_df.copy()
    plot_df["is_error"] = (plot_df["sentiment_true"] != plot_df["sentiment_pred"]).astype(int)
    bins = [0, 50, 100, 150, 200, 5000]
    labels = ["<=50", "51-100", "101-150", "151-200", ">200"]
    plot_df["text_length_bin"] = pd.cut(plot_df["text_length"], bins=bins, labels=labels, include_lowest=True)
    error_profile = plot_df.groupby("text_length_bin")["is_error"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=error_profile, x="text_length_bin", y="is_error", color="#E45756")
    plt.title("Sentiment Error Rate by Text Length")
    plt.xlabel("Text Length Bin")
    plt.ylabel("Error Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
