from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import ensure_project_directories, get_project_paths, load_project_config, load_schema_summary


def prepare_processed_datasets() -> Dict[str, Path]:
    config = load_project_config()
    paths = get_project_paths(config)
    ensure_project_directories(paths)
    schema = load_schema_summary(paths)

    splits = _load_or_create_splits(schema=schema, config=config)
    long_frames = []
    review_frames = []
    output_files: Dict[str, Path] = {}
    for split_name, frame in splits.items():
        review_df = _build_review_level_frame(frame, schema)
        long_df = _wide_to_long(frame, schema, split_name)
        review_frames.append(review_df.assign(split=split_name))
        long_frames.append(long_df)

        processed_name = "val.csv" if split_name == "val" else f"{split_name}.csv"
        review_name = "review_level_val.csv" if split_name == "val" else f"review_level_{split_name}.csv"
        long_path = paths.processed_dir / processed_name
        review_path = paths.processed_dir / review_name
        long_df.to_csv(long_path, index=False)
        review_df.to_csv(review_path, index=False)
        output_files[split_name] = long_path
        output_files[f"review_level_{split_name}"] = review_path

    combined_long = pd.concat(long_frames, ignore_index=True)
    combined_reviews = pd.concat(review_frames, ignore_index=True)
    combined_long.to_csv(paths.processed_dir / "aspect_sentiment_long.csv", index=False)
    combined_reviews.to_csv(paths.processed_dir / "review_level_all.csv", index=False)
    return output_files


def _load_or_create_splits(schema: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    selected_files = schema["selected_files"]
    if {"train", "val", "test"}.issubset(selected_files):
        return {
            split_name: pd.read_csv(Path(file_path))
            for split_name, file_path in selected_files.items()
        }

    frames = [pd.read_csv(Path(file_path)) for file_path in selected_files.values()]
    full_df = pd.concat(frames, ignore_index=True)
    review_id_col = schema["review_id_column"]
    train_ratio = config["dataset"]["split_ratios"]["train"]
    val_ratio = config["dataset"]["split_ratios"]["val"]
    random_state = config["dataset"]["random_state"]

    splitter_1 = GroupShuffleSplit(
        n_splits=1, train_size=train_ratio, random_state=random_state
    )
    train_idx, temp_idx = next(
        splitter_1.split(full_df, groups=full_df[review_id_col])
    )
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    temp_df = full_df.iloc[temp_idx].reset_index(drop=True)

    adjusted_val_ratio = val_ratio / (1.0 - train_ratio)
    splitter_2 = GroupShuffleSplit(
        n_splits=1, train_size=adjusted_val_ratio, random_state=random_state
    )
    val_idx, test_idx = next(
        splitter_2.split(temp_df, groups=temp_df[review_id_col])
    )
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return {"train": train_df, "val": val_df, "test": test_df}


def _build_review_level_frame(frame: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    review_df = frame[[schema["review_id_column"], schema["text_column"], schema["rating_column"]]].copy()
    review_df.columns = ["review_id", "review_text", "rating"]
    review_df["text_length"] = review_df["review_text"].astype(str).str.len()
    return review_df


def _wide_to_long(
    frame: pd.DataFrame,
    schema: Dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    label_mapping = {int(key): value for key, value in schema["label_mapping"].items()}
    id_col = schema["review_id_column"]
    text_col = schema["text_column"]
    rating_col = schema["rating_column"]
    aspect_cols = schema["aspect_columns"]

    long_df = frame.melt(
        id_vars=[id_col, text_col, rating_col],
        value_vars=aspect_cols,
        var_name="aspect_name",
        value_name="sentiment_label_raw",
    )
    long_df = long_df.rename(
        columns={
            id_col: "review_id",
            text_col: "review_text",
            rating_col: "rating",
        }
    )
    long_df["sentiment_label_raw"] = long_df["sentiment_label_raw"].astype(int)
    long_df["aspect_description"] = long_df["aspect_name"].map(schema["aspect_descriptions"])
    long_df["sentiment_label"] = long_df["sentiment_label_raw"].map(label_mapping)
    long_df["is_mentioned"] = (long_df["sentiment_label_raw"] != -2).astype(int)
    long_df["text_length"] = long_df["review_text"].astype(str).str.len()
    long_df["split"] = split_name
    ordered_columns = [
        "review_id",
        "review_text",
        "rating",
        "aspect_name",
        "aspect_description",
        "sentiment_label_raw",
        "sentiment_label",
        "is_mentioned",
        "text_length",
        "split",
    ]
    return long_df[ordered_columns]

