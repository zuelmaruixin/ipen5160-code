from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_paths, load_project_config, load_schema_summary
from src.report_utils import dataframe_to_markdown, update_run_status, write_markdown_report


def main() -> None:
    config = load_project_config()
    paths = get_project_paths(config)
    schema = load_schema_summary(paths)

    dataset_summary = pd.read_csv(paths.tables_dir / "dataset_summary.csv")
    rating_distribution = pd.read_csv(paths.tables_dir / "rating_distribution.csv")
    aspect_distribution = pd.read_csv(paths.tables_dir / "aspect_distribution.csv")
    sentiment_distribution = pd.read_csv(paths.tables_dir / "sentiment_distribution.csv")
    aspect_negative_rate = pd.read_csv(paths.tables_dir / "aspect_negative_rate.csv")
    model_metrics = pd.read_csv(paths.tables_dir / "model_metrics.csv")
    classification_report = pd.read_csv(paths.tables_dir / "classification_report.csv")
    final_metrics = pd.read_csv(paths.tables_dir / "final_metrics_summary.csv")
    confused_pairs = pd.read_csv(paths.tables_dir / "confused_label_pairs.csv")
    low_conf = pd.read_csv(paths.tables_dir / "low_confidence_examples.csv")

    lines = _build_report_lines(
        config=config,
        schema=schema,
        dataset_summary=dataset_summary,
        rating_distribution=rating_distribution,
        aspect_distribution=aspect_distribution,
        sentiment_distribution=sentiment_distribution,
        aspect_negative_rate=aspect_negative_rate,
        model_metrics=model_metrics,
        classification_report=classification_report,
        final_metrics=final_metrics,
        confused_pairs=confused_pairs,
        low_conf=low_conf,
    )
    write_markdown_report(paths.reports_dir / "final_report_draft.md", lines)
    update_run_status(
        PROJECT_ROOT / "RUN_STATUS.md",
        "08_generate_report",
        "Completed",
        "Final report draft generated.",
    )
    print("Final report draft created at outputs/reports/final_report_draft.md")


def _build_report_lines(**kwargs) -> list[str]:
    config = kwargs["config"]
    schema = kwargs["schema"]
    dataset_summary = kwargs["dataset_summary"]
    rating_distribution = kwargs["rating_distribution"]
    aspect_distribution = kwargs["aspect_distribution"]
    sentiment_distribution = kwargs["sentiment_distribution"]
    aspect_negative_rate = kwargs["aspect_negative_rate"]
    model_metrics = kwargs["model_metrics"]
    classification_report = kwargs["classification_report"]
    final_metrics = kwargs["final_metrics"]
    confused_pairs = kwargs["confused_pairs"]
    low_conf = kwargs["low_conf"]

    total_reviews = int(dataset_summary.loc[dataset_summary["metric"] == "total_reviews", "value"].iloc[0])
    total_aspect_rows = int(dataset_summary.loc[dataset_summary["metric"] == "total_aspect_rows", "value"].iloc[0])
    not_mentioned_pct = float(sentiment_distribution.loc[sentiment_distribution["sentiment_label"] == "Not-Mentioned", "percentage"].iloc[0])
    top_rating = int(rating_distribution.sort_values("count", ascending=False).iloc[0]["rating"])
    top_aspects = aspect_distribution.head(5)[["aspect_name", "mentioned_count"]]
    top_negative = aspect_negative_rate.head(5)[["aspect_name", "negative_rate"]]
    test_metrics = model_metrics[model_metrics["split"] == "test"].iloc[0].to_dict()
    top_confused = confused_pairs.head(5)
    low_conf_examples = low_conf.head(5)[["review_id", "aspect_name", "sentiment_true", "sentiment_pred", "confidence"]]
    weighted_f1 = final_metrics.loc[final_metrics["metric"] == "test_weighted_f1", "value"].iloc[0]
    macro_f1 = final_metrics.loc[final_metrics["metric"] == "test_macro_f1", "value"].iloc[0]
    rating_mae = final_metrics.loc[final_metrics["metric"] == "rating_mae", "value"].iloc[0]
    short_text_error = final_metrics.loc[final_metrics["metric"] == "short_text_error_rate", "value"].iloc[0]
    long_text_error = final_metrics.loc[final_metrics["metric"] == "long_text_error_rate", "value"].iloc[0]
    rating_close_acc = final_metrics.loc[final_metrics["metric"] == "sentiment_accuracy_when_rating_close", "value"].iloc[0]
    rating_far_acc = final_metrics.loc[final_metrics["metric"] == "sentiment_accuracy_when_rating_far", "value"].iloc[0]

    model_description = (
        "BERT + CRF sequence labeling + rating prediction"
        if schema["supports_token_level_sequence_labeling"]
        else "BERT encoder + aspect sentiment classification head + rating prediction head"
    )
    why_not_crf = schema.get("why_no_crf", "")

    return [
        "# Fine-Grained Sentiment Analysis and Rating Prediction for Meituan-Dianping Restaurant Reviews",
        "",
        "## 1. Executive Summary",
        "",
        f"本项目基于 ASAP 餐饮评论数据集，围绕 {total_reviews} 条点评与 {total_aspect_rows} 条 aspect-level 样本，构建了细粒度情感分析与评分预测流程。项目重点不是机械复刻论文结构，而是先读取 README 与真实 schema，再据此确定任务形式。由于该数据集提供的是 18 个 aspect category 的情感标签与 1-5 星评分，而不是 token/span 标注，因此最终采用 {model_description} 的多任务建模方案，用于支持商户经营诊断与平台决策分析。",
        "",
        "## 2. Problem Formulation",
        "",
        "总体评分只能反映评论的整体倾向，无法定位问题到底来自口味、服务、环境还是价格。细粒度情感分析将一条评论拆到 aspect 层面，可以帮助商户识别具体短板，也能帮助平台构建品牌仪表盘、服务质量监测和高风险负面反馈发现机制。本项目将 aspect sentiment classification 作为主任务，将 rating prediction 作为辅助任务，用多任务学习吸收整体满意度信号。",
        "",
        "## 3. Data and Schema Understanding",
        "",
        f"外部数据目录为 `{config['paths']['external_data_dir']}`。README 明确说明标签含义为 `1=Positive, 0=Neutral, -1=Negative, -2=Not-Mentioned`，评分范围为 1-5 星。实际 CSV 为 review-level 宽表结构，字段包括评论 ID、评论文本、星级评分以及 18 个 aspect category 列。由于没有 BIO、token、start/end 或 span 标注，本项目不构造伪造的序列标注任务，也不强行使用 CRF。",
        "",
        "## 4. Exploratory Data Analysis",
        "",
        f"数据整体规模为 {total_reviews} 条评论。出现最多的星级为 {top_rating} 星。`Not-Mentioned` 占比约为 {not_mentioned_pct:.2%}，说明绝大多数评论不会同时覆盖全部 18 个方面，因此显式建模未提及类别是必要的。最常被提及的方面和负面率最高的方面如下：",
        "",
        dataframe_to_markdown(top_aspects, max_rows=len(top_aspects)),
        "",
        dataframe_to_markdown(top_negative, max_rows=len(top_negative)),
        "",
        "这一分布也意味着类别不平衡客观存在，训练中需要考虑 class weights，并在结果解读时重点关注 `Not-Mentioned` 与 `Neutral` 的界限。",
        "",
        "## 5. Model Design",
        "",
        f"最终模型采用 {model_description}。在 aspect-level 设置下，输入形式为 `tokenizer(review_text, aspect_name + aspect_description, truncation='only_first')`，也就是把评论正文作为第一段、aspect 信息作为第二段，并优先截断评论正文而保留 aspect 语义。主任务输出单个 aspect 的情感标签，辅助任务输出评论星级。总损失函数为 `total_loss = sentiment_loss + alpha * rating_loss`，其中 `alpha` 默认设置为 0.2。这个设计既保留了 aspect 语义，又把整体评分作为弱监督信号引入。{why_not_crf}",
        "",
        "## 6. Results",
        "",
        f"测试集 sentiment accuracy 为 {test_metrics['accuracy']:.4f}，macro F1 为 {macro_f1:.4f}，weighted F1 为 {weighted_f1:.4f}。rating prediction 的 MAE 为 {rating_mae:.4f}。测试集 classification report 见下表摘要：",
        "",
        dataframe_to_markdown(classification_report.head(8), max_rows=8),
        "",
        "## 7. Error Analysis",
        "",
        f"短文本错误率为 {short_text_error:.4f}，长文本错误率为 {long_text_error:.4f}。这说明过短评论往往信息不足，更容易造成 aspect 判断歧义。最常见的标签混淆对如下：",
        "",
        dataframe_to_markdown(top_confused, max_rows=len(top_confused)),
        "",
        f"从 rating 辅助信号看，当 rating 预测较准确时，sentiment accuracy 为 {rating_close_acc:.4f}；当 rating 预测偏差较大时，sentiment accuracy 为 {rating_far_acc:.4f}。这可以支持“整体满意度与 aspect 极性存在关联”的判断，但由于这里没有单任务对照实验，不能把这一差异直接解释为严格的因果提升证据。低置信度样本建议进入人工复核队列：",
        "",
        dataframe_to_markdown(low_conf_examples, max_rows=len(low_conf_examples)),
        "",
        "## 8. Business Decision Implications",
        "",
        "该系统更适合作为 decision support，而不是自动裁决工具。对于商户，可用于定位高负面率方面，例如服务态度、排队体验或菜品口味；对于平台，可用于品牌仪表盘、区域门店对比和服务质量波动监测；对于评价管理，可把高风险负面 aspect 与低置信度样本优先推送人工复核。这样既能提高运营效率，也能控制误判带来的业务风险。",
        "",
        "## 9. Responsible Data Use and Limitations",
        "",
        "本项目是相关性和预测性建模，不是严格的因果推断。rating 与 aspect sentiment 的关系只能说明统计关联，不能直接解释为某个 aspect 导致总体评分变化。评论数据还存在选择偏差和主观表达差异，自动预测结果不能替代人工判断，尤其在商户考核和投诉处置场景中更需要谨慎使用。",
        "",
        "## 10. Conclusion",
        "",
        "本项目基于真实 schema 完成了从数据理解、任务构造、EDA、Colab 训练准备到结果分析的完整流程。结论很明确：对于 ASAP 这类 aspect-category 数据，合理方案不是伪造 BIO 标注，而是使用 aspect-level BERT 多任务模型，把细粒度情感分析与评分预测结合起来，为商户经营诊断提供更细、更稳的支持信号。",
        "",
    ]


if __name__ == "__main__":
    main()
