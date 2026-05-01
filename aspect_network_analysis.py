from __future__ import annotations

import math
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import seaborn as sns


LABEL_NEGATIVE = "Negative"
LABEL_NOT_MENTIONED = "Not-Mentioned"
OUTPUT_ROOT = Path("/Users/maruixin/Documents/New project/aspect_network_outputs")
PREDICTION_PATH = Path("/Users/maruixin/Downloads/model_predictions.csv")
SPLIT_FILTER: tuple[str, ...] | None = ("val", "test")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tables_dir = OUTPUT_ROOT / "tables"
    figures_dir = OUTPUT_ROOT / "figures"
    reports_dir = OUTPUT_ROOT / "reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(PREDICTION_PATH)
    pred_df = prepare_prediction_frame(pred_df, split_filter=SPLIT_FILTER)

    review_count = pred_df["review_id"].nunique()
    aspect_order = sorted(pred_df["aspect_name"].dropna().astype(str).unique().tolist())
    mentioned_groups = group_aspects(pred_df, sentiment_value=None)
    negative_groups = group_aspects(pred_df, sentiment_value=LABEL_NEGATIVE)

    cooccurrence_matrix = build_pair_matrix(mentioned_groups, aspect_order)
    conegative_matrix = build_pair_matrix(negative_groups, aspect_order)
    risk_table = build_conditional_negative_risk(negative_groups, aspect_order, review_count)
    edge_table = build_network_edges(conegative_matrix, risk_table, min_support=10)
    node_table = build_node_summary(pred_df, mentioned_groups, negative_groups, review_count)
    length_by_sentiment = build_length_summary_by_sentiment(pred_df)

    cooccurrence_matrix.to_csv(tables_dir / "aspect_cooccurrence_matrix.csv")
    conegative_matrix.to_csv(tables_dir / "aspect_conegative_matrix.csv")
    risk_table.to_csv(tables_dir / "aspect_conditional_negative_risk.csv", index=False)
    edge_table.to_csv(tables_dir / "aspect_network_edges.csv", index=False)
    node_table.to_csv(tables_dir / "aspect_network_nodes.csv", index=False)
    length_by_sentiment.to_csv(tables_dir / "token_length_by_sentiment.csv", index=False)

    plot_heatmap(
        cooccurrence_matrix,
        "Aspect Co-occurrence (Predicted Mentioned)",
        figures_dir / "aspect_cooccurrence_heatmap.png",
    )
    plot_heatmap(
        conegative_matrix,
        "Aspect Co-Negative Matrix",
        figures_dir / "aspect_conegative_heatmap.png",
    )
    plot_aspect_network(
        edge_table,
        node_table,
        figures_dir / "aspect_conegative_network.png",
    )
    write_summary(
        reports_dir / "aspect_network_summary.md",
        node_table=node_table,
        edge_table=edge_table,
        risk_table=risk_table,
        review_count=review_count,
        prediction_path=PREDICTION_PATH,
    )


def prepare_prediction_frame(
    pred_df: pd.DataFrame,
    split_filter: tuple[str, ...] | None,
) -> pd.DataFrame:
    frame = pred_df.copy()
    if split_filter is not None and "split" in frame.columns:
        frame = frame[frame["split"].astype(str).isin(set(split_filter))].copy()
        if frame.empty:
            frame = pred_df.copy()
    frame["aspect_name"] = frame["aspect_name"].astype(str)
    frame["sentiment_pred"] = frame["sentiment_pred"].astype(str)
    if "confidence" in frame.columns:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    if "text_length" in frame.columns:
        frame["text_length"] = pd.to_numeric(frame["text_length"], errors="coerce")
    return frame


def group_aspects(pred_df: pd.DataFrame, sentiment_value: str | None) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for review_id, group in pred_df.groupby("review_id"):
        if sentiment_value is None:
            subset = group[group["sentiment_pred"] != LABEL_NOT_MENTIONED]
        else:
            subset = group[group["sentiment_pred"] == sentiment_value]
        grouped[str(review_id)] = set(subset["aspect_name"].astype(str).tolist())
    return grouped


def build_pair_matrix(grouped: dict[str, set[str]], aspect_order: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=aspect_order, columns=aspect_order, dtype=int)
    for aspects in grouped.values():
        for aspect in aspects:
            matrix.loc[aspect, aspect] += 1
        for left, right in combinations(sorted(aspects), 2):
            matrix.loc[left, right] += 1
            matrix.loc[right, left] += 1
    return matrix


def build_conditional_negative_risk(
    negative_groups: dict[str, set[str]],
    aspect_order: list[str],
    review_count: int,
) -> pd.DataFrame:
    negative_counts = {aspect: 0 for aspect in aspect_order}
    pair_counts = {(left, right): 0 for left, right in product(aspect_order, aspect_order) if left != right}

    for aspects in negative_groups.values():
        for aspect in aspects:
            negative_counts[aspect] += 1
        for left, right in product(aspects, aspects):
            if left == right:
                continue
            pair_counts[(left, right)] += 1

    rows: list[dict[str, float | int | str]] = []
    for left, right in product(aspect_order, aspect_order):
        if left == right:
            continue
        support_left = negative_counts[left]
        support_right = negative_counts[right]
        support_pair = pair_counts[(left, right)]
        conditional = support_pair / support_left if support_left else 0.0
        base_rate = support_right / review_count if review_count else 0.0
        lift = conditional / base_rate if base_rate else 0.0
        rows.append(
            {
                "aspect_a": left,
                "aspect_b": right,
                "support_a_negative": support_left,
                "support_b_negative": support_right,
                "support_ab_negative": support_pair,
                "p_b_negative_given_a_negative": conditional,
                "base_b_negative_rate": base_rate,
                "lift": lift,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["support_ab_negative", "p_b_negative_given_a_negative", "lift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_network_edges(
    conegative_matrix: pd.DataFrame,
    risk_table: pd.DataFrame,
    min_support: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for left, right in combinations(conegative_matrix.index.tolist(), 2):
        co_negative = int(conegative_matrix.loc[left, right])
        if co_negative < min_support:
            continue
        neg_left = int(conegative_matrix.loc[left, left])
        neg_right = int(conegative_matrix.loc[right, right])
        union = neg_left + neg_right - co_negative
        jaccard = co_negative / union if union else 0.0
        cond_lr = lookup_risk(risk_table, left, right, "p_b_negative_given_a_negative")
        cond_rl = lookup_risk(risk_table, right, left, "p_b_negative_given_a_negative")
        lift_lr = lookup_risk(risk_table, left, right, "lift")
        lift_rl = lookup_risk(risk_table, right, left, "lift")
        rows.append(
            {
                "source": left,
                "target": right,
                "co_negative_count": co_negative,
                "negative_count_source": neg_left,
                "negative_count_target": neg_right,
                "jaccard": jaccard,
                "conditional_negative_mean": (cond_lr + cond_rl) / 2.0,
                "lift_mean": (lift_lr + lift_rl) / 2.0,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "co_negative_count",
                "negative_count_source",
                "negative_count_target",
                "jaccard",
                "conditional_negative_mean",
                "lift_mean",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["co_negative_count", "conditional_negative_mean", "lift_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def lookup_risk(risk_table: pd.DataFrame, left: str, right: str, column: str) -> float:
    matched = risk_table[(risk_table["aspect_a"] == left) & (risk_table["aspect_b"] == right)]
    if matched.empty:
        return 0.0
    return float(matched.iloc[0][column])


def build_node_summary(
    pred_df: pd.DataFrame,
    mentioned_groups: dict[str, set[str]],
    negative_groups: dict[str, set[str]],
    review_count: int,
) -> pd.DataFrame:
    aspects = sorted(pred_df["aspect_name"].unique().tolist())
    mentioned_counts = {
        aspect: int(sum(aspect in aspects_set for aspects_set in mentioned_groups.values()))
        for aspect in aspects
    }
    negative_counts = {
        aspect: int(sum(aspect in aspects_set for aspects_set in negative_groups.values()))
        for aspect in aspects
    }
    rows = []
    for aspect in aspects:
        mentioned_count = mentioned_counts.get(aspect, 0)
        negative_count = negative_counts.get(aspect, 0)
        rows.append(
            {
                "aspect_name": aspect,
                "mentioned_review_count": mentioned_count,
                "negative_review_count": negative_count,
                "mentioned_review_ratio": mentioned_count / review_count if review_count else 0.0,
                "negative_review_ratio": negative_count / review_count if review_count else 0.0,
                "negative_rate_given_mentioned": negative_count / mentioned_count if mentioned_count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["negative_review_count", "mentioned_review_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_length_summary_by_sentiment(pred_df: pd.DataFrame) -> pd.DataFrame:
    if "text_length" not in pred_df.columns:
        return pd.DataFrame(columns=["sentiment_pred", "mean", "median", "p90", "p95", "max"])
    rows = []
    for sentiment, group in pred_df.groupby("sentiment_pred"):
        lengths = group["text_length"].dropna()
        if lengths.empty:
            continue
        rows.append(
            {
                "sentiment_pred": sentiment,
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "p90": float(lengths.quantile(0.90)),
                "p95": float(lengths.quantile(0.95)),
                "max": float(lengths.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def plot_heatmap(matrix_df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(11, 9))
    sns.heatmap(matrix_df, cmap="YlOrRd", square=True)
    plt.title(title)
    plt.xlabel("Aspect")
    plt.ylabel("Aspect")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_aspect_network(edge_df: pd.DataFrame, node_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 13), facecolor="white")
    ax.set_facecolor("#FCFCFD")
    ax.set_axis_off()

    node_df = node_df.copy()
    node_df["category"] = node_df["aspect_name"].map(get_aspect_category)
    aspects = node_df["aspect_name"].tolist()
    positions = build_grouped_layout(aspects)
    display_edges = select_display_edges(edge_df, top_k=24)

    category_colors = {
        "Food": "#E67E22",
        "Service": "#2E86DE",
        "Price": "#27AE60",
        "Ambience": "#8E44AD",
        "Location": "#C0392B",
    }

    if not display_edges.empty:
        max_edge = max(float(display_edges["co_negative_count"].max()), 1.0)
        min_lift = float(display_edges["lift_mean"].min())
        max_lift = float(display_edges["lift_mean"].max())
        lift_span = max(max_lift - min_lift, 1e-6)
        cmap = matplotlib.colormaps["YlOrRd"]

        for idx, row in display_edges.iterrows():
            source = row["source"]
            target = row["target"]
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            weight_scale = float(row["co_negative_count"]) / max_edge
            lift_scale = (float(row["lift_mean"]) - min_lift) / lift_span
            width = 1.2 + 6.5 * weight_scale
            color = cmap(0.28 + 0.68 * lift_scale)
            rad = 0.18 if idx % 2 == 0 else -0.18
            edge = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                linewidth=width,
                color=color,
                alpha=0.18 + 0.62 * min(1.0, weight_scale + 0.15),
                zorder=1,
            )
            ax.add_patch(edge)

    max_negative = max(float(node_df["negative_review_count"].max()), 1.0) if not node_df.empty else 1.0
    for _, row in node_df.iterrows():
        aspect = row["aspect_name"]
        category = row["category"]
        x, y = positions[aspect]
        size = 900 + 2800 * (float(row["negative_review_count"]) / max_negative)
        fill = category_colors.get(category, "#7F8C8D")

        ax.scatter([x], [y], s=size * 1.16, c="#F7F8FA", edgecolors="none", zorder=2)
        ax.scatter(
            [x],
            [y],
            s=size,
            c=fill,
            edgecolors="white",
            linewidths=2.8,
            alpha=0.95,
            zorder=3,
        )

        label_x, label_y = scale_point(x, y, 1.16)
        ha = "left" if label_x >= 0 else "right"
        ax.plot([x, label_x], [y, label_y], color="#C7CCD5", linewidth=0.9, alpha=0.8, zorder=2)
        ax.text(
            label_x,
            label_y,
            format_aspect_label(aspect),
            ha=ha,
            va="center",
            fontsize=10,
            color="#1F2937",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
        )

    draw_category_headers(ax, positions, category_colors)
    add_network_legend(ax, category_colors)

    ax.set_title("Aspect Co-Negative Network", fontsize=18, fontweight="bold", pad=20)
    ax.text(
        0.5,
        1.06,
        "Node size = negative review count | Edge width = co-negative count | Edge color = lift",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_circular_layout(aspects: list[str]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    total = max(len(aspects), 1)
    for idx, aspect in enumerate(aspects):
        angle = 2.0 * math.pi * idx / total
        positions[aspect] = (math.cos(angle), math.sin(angle))
    return positions


def build_grouped_layout(aspects: list[str]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    grouped: dict[str, list[str]] = {}
    for aspect in sorted(aspects):
        grouped.setdefault(get_aspect_category(aspect), []).append(aspect)

    category_order = [category for category in ["Food", "Service", "Price", "Ambience", "Location"] if category in grouped]
    sector_total = max(len(category_order), 1)
    radius = 1.0
    for idx, category in enumerate(category_order):
        category_aspects = grouped[category]
        center_angle = math.pi / 2 - idx * (2.0 * math.pi / sector_total)
        spread = 0.65 if len(category_aspects) > 1 else 0.0
        if len(category_aspects) == 1:
            angles = [center_angle]
        else:
            start = center_angle + spread / 2
            step = spread / (len(category_aspects) - 1)
            angles = [start - i * step for i in range(len(category_aspects))]
        for aspect, angle in zip(category_aspects, angles):
            positions[aspect] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def select_display_edges(edge_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if edge_df.empty:
        return edge_df.copy()
    ranked = edge_df.copy()
    ranked = ranked.sort_values(
        ["co_negative_count", "conditional_negative_mean", "lift_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    cutoff = ranked["co_negative_count"].quantile(0.72)
    filtered = ranked[ranked["co_negative_count"] >= cutoff].copy()
    if len(filtered) < min(top_k, len(ranked)):
        filtered = ranked.head(top_k).copy()
    return filtered.head(top_k).reset_index(drop=True)


def get_aspect_category(aspect_name: str) -> str:
    return str(aspect_name).split("#", 1)[0]


def format_aspect_label(aspect_name: str) -> str:
    left, right = str(aspect_name).split("#", 1)
    right = right.replace("_", " ")
    return f"{left}\n{right}"


def scale_point(x: float, y: float, scale: float) -> tuple[float, float]:
    return x * scale, y * scale


def draw_category_headers(ax: plt.Axes, positions: dict[str, tuple[float, float]], category_colors: dict[str, str]) -> None:
    by_category: dict[str, list[tuple[float, float]]] = {}
    for aspect, (x, y) in positions.items():
        by_category.setdefault(get_aspect_category(aspect), []).append((x, y))

    for category, coords in by_category.items():
        mean_x = sum(x for x, _ in coords) / len(coords)
        mean_y = sum(y for _, y in coords) / len(coords)
        header_x, header_y = scale_point(mean_x, mean_y, 1.42)
        ax.text(
            header_x,
            header_y,
            category,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=category_colors.get(category, "#374151"),
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            zorder=5,
        )


def add_network_legend(ax: plt.Axes, category_colors: dict[str, str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=category,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            markersize=10,
        )
        for category, color in category_colors.items()
    ]
    legend = ax.legend(
        handles=handles,
        title="Aspect group",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        title_fontsize=10.5,
    )
    ax.add_artist(legend)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Empty dataframe_"
    headers = [str(column) for column in df.columns]
    divider = ["---"] * len(headers)
    rows = [headers, divider]
    for _, row in df.iterrows():
        rows.append([escape_markdown_cell(value) for value in row.tolist()])
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def escape_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def write_summary(
    output_path: Path,
    *,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    risk_table: pd.DataFrame,
    review_count: int,
    prediction_path: Path,
) -> None:
    top_negative = node_table.head(5)[
        ["aspect_name", "negative_review_count", "negative_rate_given_mentioned"]
    ]
    strongest_edges = edge_table.head(8)[
        ["source", "target", "co_negative_count", "conditional_negative_mean", "lift_mean"]
    ]
    directional = risk_table[risk_table["support_ab_negative"] >= 10].head(8)[
        ["aspect_a", "aspect_b", "support_ab_negative", "p_b_negative_given_a_negative", "lift"]
    ]
    lines = [
        "# Aspect Network Summary",
        "",
        f"- Prediction source: `{prediction_path}`",
        f"- Split filter: `{SPLIT_FILTER if SPLIT_FILTER is not None else 'all splits'}`",
        f"- Review count used: {review_count}",
        f"- Node count: {len(node_table)}",
        f"- Edge count retained: {len(edge_table)}",
        "",
        "## Top Negative Aspects",
        "",
        dataframe_to_markdown(top_negative),
        "",
        "## Strongest Co-Negative Links",
        "",
        dataframe_to_markdown(strongest_edges) if not strongest_edges.empty else "_No edges passed support threshold._",
        "",
        "## Strongest Directional Negative Risks",
        "",
        dataframe_to_markdown(directional) if not directional.empty else "_No directional links passed support threshold._",
        "",
        "## Interpretation",
        "",
        "这份网络更适合解释问题联动和共负面模式，不应直接解释成因果或时间转移。`co_negative_count` 回答哪些问题容易一起被抱怨，`p_b_negative_given_a_negative` 与 `lift` 回答当 A 负面时，B 是否更容易一起出问题。",
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
