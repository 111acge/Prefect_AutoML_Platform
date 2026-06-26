# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""训练结果可视化服务。"""

import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Optional

from i18n import _, get_locale

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _fig_to_base64(fig: plt.Figure) -> str:
    """将 matplotlib figure 转为 base64 PNG。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


def plot_confusion_matrix(confusion_matrix: List[List[int]], labels: List[str]) -> str:
    """绘制混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        np.array(confusion_matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return _fig_to_base64(fig)


def plot_feature_importance(feature_importance: List[Dict[str, Any]], top_n: int = 15) -> str:
    """绘制特征重要性条形图。"""
    if not feature_importance:
        return ""

    df = pd.DataFrame(feature_importance)
    if "importance" not in df.columns:
        return ""

    df = df.sort_values("importance", ascending=True).tail(top_n)
    feature_col = "feature" if "feature" in df.columns else df.columns[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df[feature_col].astype(str), df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    return _fig_to_base64(fig)


def plot_roc_curve(fpr: List[float], tpr: List[float], auc: float) -> str:
    """绘制 ROC 曲线。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _fig_to_base64(fig)


def plot_pr_curve(recall: List[float], precision: List[float], auc: float) -> str:
    """绘制 PR 曲线。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    return _fig_to_base64(fig)


def plot_shap_summary(shap_values_path: Path, train_data_path: Optional[Path] = None) -> str:
    """绘制 SHAP summary plot。"""
    try:
        import joblib
        import shap

        shap_data = joblib.load(shap_values_path)
        shap_values = shap_data["shap_values"]
        feature_names = shap_data["feature_names"]
        features = shap_data.get("features")

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            features=features,
            feature_names=feature_names,
            show=False,
            plot_size=(8, 6),
        )
        ax.set_title("SHAP Feature Importance")
        return _fig_to_base64(fig)
    except Exception:
        return ""


def plot_residuals(y_true: List[float], y_pred: List[float]) -> str:
    """绘制回归残差图。"""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    residuals = y_true_arr - y_pred_arr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 预测 vs 真实值
    axes[0].scatter(y_pred_arr, residuals, alpha=0.6)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")

    # 残差分布
    axes[1].hist(residuals, bins=20, edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    return _fig_to_base64(fig)


def generate_report_plots(output_dir: Path, task_type: str) -> Dict[str, str]:
    """生成报告所需的所有图表，返回 {name: base64}。"""
    problem_type_mapping = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
        "regression": "regression",
    }
    problem_type = problem_type_mapping.get(task_type, task_type)

    plots = {}

    # 特征重要性
    importance_path = output_dir / "feature_importance.csv"
    if importance_path.exists():
        df = pd.read_csv(importance_path)
        plots["feature_importance"] = plot_feature_importance(df.head(20).to_dict("records"))

    # 混淆矩阵 / ROC / PR
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        extended = metrics.get("extended", {})

        if problem_type in ["binary", "multiclass"]:
            cm = extended.get("confusion_matrix")
            if cm:
                labels = extended.get("labels") or [str(i) for i in range(len(cm))]
                plots["confusion_matrix"] = plot_confusion_matrix(cm, labels)

            if problem_type == "binary":
                roc = extended.get("roc_curve")
                if roc:
                    plots["roc_curve"] = plot_roc_curve(
                        roc["fpr"], roc["tpr"], extended.get("auc_roc", 0)
                    )
                pr = extended.get("pr_curve")
                if pr:
                    plots["pr_curve"] = plot_pr_curve(
                        pr["recall"], pr["precision"], extended.get("auc_pr", 0)
                    )

        elif problem_type == "regression":
            # 残差图需要真实值和预测值，这里简化：不生成
            pass

    # SHAP summary
    shap_path = output_dir / "shap_values.joblib"
    if shap_path.exists():
        plots["shap_summary"] = plot_shap_summary(shap_path)

    return plots
