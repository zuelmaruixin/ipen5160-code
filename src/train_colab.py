from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import ensure_project_directories, get_project_paths, load_project_config, load_schema_summary
from src.model import AspectSentimentMultiTaskModel
from src.tokenization_utils import encode_aspect_example, load_tokenizer_with_local_fallback


LABEL_ORDER = ["Positive", "Neutral", "Negative", "Not-Mentioned"]


@dataclass
class BatchEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    sentiment_labels: torch.Tensor
    rating_labels: torch.Tensor


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        label_to_id: Dict[str, int],
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        encoded = encode_aspect_example(
            self.tokenizer,
            row["review_text"],
            row["aspect_name"],
            row["aspect_description"],
            padding="max_length",
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sentiment_labels": torch.tensor(
                self.label_to_id[str(row["sentiment_label"])], dtype=torch.long
            ),
            "rating_labels": torch.tensor(float(row["rating"]), dtype=torch.float32),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path: str | Path | None = None) -> None:
    config = load_project_config(Path(config_path) if config_path is not None else None)
    paths = get_project_paths(config)
    ensure_project_directories(paths)
    schema = load_schema_summary(paths)

    if schema["recommended_task"] != "bert_aspect_sentiment_multitask_with_rating":
        raise NotImplementedError(
            "This training script currently targets the aspect-level multitask setup for ASAP."
        )

    set_seed(config["model"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = load_tokenizer_with_local_fallback(config["model"]["encoder_name"])
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_df = pd.read_csv(paths.processed_dir / "train.csv")
    val_df = pd.read_csv(paths.processed_dir / "val.csv")
    test_df = pd.read_csv(paths.processed_dir / "test.csv")

    class_weights = _maybe_compute_class_weights(train_df["sentiment_label"], label_to_id)
    model = AspectSentimentMultiTaskModel(
        encoder_name=config["model"]["encoder_name"],
        num_labels=len(label_to_id),
        alpha=float(config["model"]["alpha"]),
        dropout=float(config["model"]["dropout"]),
        class_weights=class_weights.to(device) if class_weights is not None else None,
    ).to(device)

    train_loader_kwargs = _build_dataloader_kwargs(config, device, is_train=True)
    eval_loader_kwargs = _build_dataloader_kwargs(config, device, is_train=False)
    train_loader = DataLoader(
        AspectSentimentDataset(train_df, tokenizer, config["model"]["max_length"], label_to_id),
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        AspectSentimentDataset(val_df, tokenizer, config["model"]["max_length"], label_to_id),
        batch_size=int(config["model"].get("eval_batch_size", config["model"]["batch_size"])),
        shuffle=False,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        AspectSentimentDataset(test_df, tokenizer, config["model"]["max_length"], label_to_id),
        batch_size=int(config["model"].get("eval_batch_size", config["model"]["batch_size"])),
        shuffle=False,
        **eval_loader_kwargs,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    total_steps = len(train_loader) * int(config["model"]["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    autocast_dtype = _resolve_autocast_dtype(config, device)
    use_amp = autocast_dtype is not None
    use_grad_scaler = use_amp and device.type == "cuda" and autocast_dtype == torch.float16
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if use_grad_scaler
        else None
    )

    history: List[Dict[str, float]] = []
    best_macro_f1 = -1.0
    max_patience = int(config["model"]["early_stopping_patience"])
    patience_left = max_patience
    best_state = None
    checkpoint_dir = paths.models_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1

    if bool(config["model"].get("resume_if_available", True)):
        resume_state = _maybe_resume_training(
            checkpoint_path=checkpoint_dir / "latest_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if resume_state is not None:
            start_epoch = resume_state["start_epoch"]
            history = resume_state["history"]
            best_macro_f1 = resume_state["best_macro_f1"]
            patience_left = resume_state["patience_left"]
            print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, int(config["model"]["epochs"]) + 1):
        train_loss = _train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            use_grad_scaler=use_grad_scaler,
            epoch=epoch,
        )
        val_metrics = _evaluate_loader(
            model=model,
            data_loader=val_loader,
            device=device,
            id_to_label=id_to_label,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_mae": val_metrics["rating_mae"],
            }
        )
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_left = max_patience
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1

        _save_epoch_artifacts(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            config=config,
            schema=schema,
            checkpoint_dir=checkpoint_dir,
            logs_dir=paths.logs_dir,
            tables_dir=paths.tables_dir,
            epoch=epoch,
            best_macro_f1=best_macro_f1,
            patience_left=patience_left,
            improved=improved,
        )

        if patience_left <= 0:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    final_model_dir = paths.models_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(final_model_dir / "encoder")
    tokenizer.save_pretrained(final_model_dir / "tokenizer")
    torch.save(model.state_dict(), final_model_dir / "model_state.pt")

    val_outputs = _predict_dataframe(model, val_df, val_loader, device, id_to_label)
    test_outputs = _predict_dataframe(model, test_df, test_loader, device, id_to_label)
    combined_predictions = pd.concat([val_outputs.assign(split="val"), test_outputs.assign(split="test")], ignore_index=True)
    combined_predictions.to_csv(paths.predictions_dir / "model_predictions.csv", index=False)

    val_metrics = _summarize_predictions(val_outputs)
    test_metrics = _summarize_predictions(test_outputs)
    pd.DataFrame([{"split": "val", **val_metrics}, {"split": "test", **test_metrics}]).to_csv(
        paths.tables_dir / "model_metrics.csv", index=False
    )

    report_df = pd.DataFrame(
        classification_report(
            test_outputs["sentiment_true"],
            test_outputs["sentiment_pred"],
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().reset_index().rename(columns={"index": "label"})
    report_df.to_csv(paths.tables_dir / "classification_report.csv", index=False)

    rating_metrics = pd.DataFrame(
        [
            {
                "split": "val",
                "mae": val_metrics["rating_mae"],
                "rmse": val_metrics["rating_rmse"],
                "rounded_accuracy": val_metrics["rating_rounded_accuracy"],
            },
            {
                "split": "test",
                "mae": test_metrics["rating_mae"],
                "rmse": test_metrics["rating_rmse"],
                "rounded_accuracy": test_metrics["rating_rounded_accuracy"],
            },
        ]
    )
    rating_metrics.to_csv(paths.tables_dir / "rating_prediction_metrics.csv", index=False)

    _save_train_log(paths.logs_dir / "train_log.json", history, config, schema)
    _plot_training_curve(history, paths.figures_dir / "training_loss_curve.png")
    _plot_confusion_matrix(test_outputs, paths.figures_dir / "confusion_matrix.png")
    _plot_rating_scatter(test_outputs, paths.figures_dir / "rating_prediction_actual_vs_predicted.png")


def _maybe_compute_class_weights(series: pd.Series, label_to_id: Dict[str, int]) -> torch.Tensor | None:
    labels = series.map(label_to_id)
    counts = labels.value_counts()
    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio < 1.5:
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_to_id)),
        y=labels.values,
    )
    return torch.tensor(weights, dtype=torch.float32)


def _train_one_epoch(
    model: AspectSentimentMultiTaskModel,
    data_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
    autocast_dtype: Optional[torch.dtype],
    use_grad_scaler: bool,
    epoch: int,
) -> float:
    model.train()
    losses = []
    progress = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        autocast_kwargs = {"device_type": device.type, "enabled": use_amp}
        if use_amp and autocast_dtype is not None:
            autocast_kwargs["dtype"] = autocast_dtype
        with torch.amp.autocast(**autocast_kwargs):
            outputs = model(**batch)
            loss = outputs["loss"]
        optimizer_ran = True
        if use_grad_scaler and scaler is not None:
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer_ran = scale_after >= scale_before
        else:
            loss.backward()
            optimizer.step()
        if optimizer_ran:
            scheduler.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return float(np.mean(losses))


def _build_dataloader_kwargs(
    config: Dict[str, object],
    device: torch.device,
    is_train: bool,
) -> Dict[str, object]:
    num_workers = int(config["model"].get("num_workers", 0))
    pin_memory = bool(config["model"].get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(config["model"].get("persistent_workers", num_workers > 0))
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = int(config["model"].get("prefetch_factor", 2))
    return kwargs


def _resolve_autocast_dtype(
    config: Dict[str, object],
    device: torch.device,
) -> Optional[torch.dtype]:
    if not bool(config["model"].get("use_mixed_precision", True)):
        return None
    if device.type != "cuda":
        return None
    amp_dtype = str(config["model"].get("amp_dtype", "auto")).lower()
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "none":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _maybe_resume_training(
    checkpoint_path: Path,
    model: AspectSentimentMultiTaskModel,
    optimizer: AdamW,
    scheduler,
) -> Optional[Dict[str, object]]:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "history": checkpoint.get("history", []),
        "best_macro_f1": float(checkpoint.get("best_macro_f1", -1.0)),
        "patience_left": int(checkpoint.get("patience_left", 0)),
    }


def _save_epoch_artifacts(
    model: AspectSentimentMultiTaskModel,
    optimizer: AdamW,
    scheduler,
    history: List[Dict[str, float]],
    config: Dict[str, object],
    schema: Dict[str, object],
    checkpoint_dir: Path,
    logs_dir: Path,
    tables_dir: Path,
    epoch: int,
    best_macro_f1: float,
    patience_left: int,
    improved: bool,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_macro_f1": best_macro_f1,
        "patience_left": patience_left,
        "history": history,
    }
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(payload, latest_path)
    if bool(config["model"].get("save_checkpoint_each_epoch", True)):
        torch.save(payload, checkpoint_dir / f"epoch_{epoch}.pt")
    if improved:
        torch.save(payload, checkpoint_dir / "best_checkpoint.pt")
    _save_train_log(logs_dir / "train_log.json", history, config, schema)
    pd.DataFrame(history).to_csv(tables_dir / "epoch_history.csv", index=False)


@torch.no_grad()
def _evaluate_loader(
    model: AspectSentimentMultiTaskModel,
    data_loader: DataLoader,
    device: torch.device,
    id_to_label: Dict[int, str],
) -> Dict[str, float]:
    model.eval()
    all_true, all_pred = [], []
    all_rating_true, all_rating_pred = [], []
    for batch in data_loader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        pred_ids = probs.argmax(dim=-1).detach().cpu().numpy()
        true_ids = inputs["sentiment_labels"].detach().cpu().numpy()
        all_true.extend([id_to_label[idx] for idx in true_ids])
        all_pred.extend([id_to_label[idx] for idx in pred_ids])
        all_rating_true.extend(inputs["rating_labels"].detach().cpu().numpy().tolist())
        all_rating_pred.extend(outputs["rating_pred"].detach().cpu().numpy().tolist())
    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "rating_mae": mean_absolute_error(all_rating_true, all_rating_pred),
    }


@torch.no_grad()
def _predict_dataframe(
    model: AspectSentimentMultiTaskModel,
    dataframe: pd.DataFrame,
    data_loader: DataLoader,
    device: torch.device,
    id_to_label: Dict[int, str],
) -> pd.DataFrame:
    model.eval()
    prediction_rows = []
    cursor = 0
    for batch in data_loader:
        batch_size = batch["input_ids"].shape[0]
        batch_df = dataframe.iloc[cursor : cursor + batch_size].copy()
        cursor += batch_size

        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        pred_ids = probs.argmax(dim=-1).detach().cpu().numpy()
        confidences = probs.max(dim=-1).values.detach().cpu().numpy()
        batch_df["sentiment_true"] = batch_df["sentiment_label"]
        batch_df["sentiment_pred"] = [id_to_label[idx] for idx in pred_ids]
        batch_df["confidence"] = confidences
        batch_df["rating_true"] = batch_df["rating"].astype(float)
        batch_df["rating_pred"] = outputs["rating_pred"].detach().cpu().numpy()
        prediction_rows.append(batch_df)
    return pd.concat(prediction_rows, ignore_index=True)


def _summarize_predictions(df: pd.DataFrame) -> Dict[str, float]:
    sentiment_true = df["sentiment_true"]
    sentiment_pred = df["sentiment_pred"]
    rating_true = df["rating_true"].astype(float)
    rating_pred = df["rating_pred"].astype(float)
    return {
        "accuracy": accuracy_score(sentiment_true, sentiment_pred),
        "macro_f1": f1_score(sentiment_true, sentiment_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(sentiment_true, sentiment_pred, labels=LABEL_ORDER, average="weighted", zero_division=0),
        "rating_mae": mean_absolute_error(rating_true, rating_pred),
        "rating_rmse": math.sqrt(mean_squared_error(rating_true, rating_pred)),
        "rating_rounded_accuracy": accuracy_score(rating_true.round().astype(int), np.clip(np.rint(rating_pred), 1, 5).astype(int)),
    }


def _save_train_log(path: Path, history: List[Dict[str, float]], config: Dict[str, object], schema: Dict[str, object]) -> None:
    payload = {
        "history": history,
        "encoder_name": config["model"]["encoder_name"],
        "task": schema["recommended_task"],
        "label_order": LABEL_ORDER,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_training_curve(history: List[Dict[str, float]], output_path: Path) -> None:
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=history_df, x="epoch", y="train_loss", marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_confusion_matrix(pred_df: pd.DataFrame, output_path: Path) -> None:
    cm = confusion_matrix(pred_df["sentiment_true"], pred_df["sentiment_pred"], labels=LABEL_ORDER)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
    plt.title("Sentiment Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_rating_scatter(pred_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=pred_df.sample(min(3000, len(pred_df)), random_state=42), x="rating_true", y="rating_pred", s=18, alpha=0.5)
    plt.plot([1, 5], [1, 5], linestyle="--", color="black", linewidth=1)
    plt.title("Rating Prediction: Actual vs Predicted")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
