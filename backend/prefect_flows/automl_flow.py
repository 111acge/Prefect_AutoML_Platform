# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""Prefect AutoML Flow 定义。"""

import os

os.environ.setdefault("PREFECT_API_URL", "")

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from prefect.cache_policies import INPUTS, TASK_SOURCE

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import (
    settings,
    HIGH_CARDINALITY_CLASS_THRESHOLD,
    LARGE_DATASET_ROW_THRESHOLD,
    DEFAULT_TRAIN_EVAL_SAMPLE_SIZE,
)
from services.data_service import (
    load_dataframe,
    analyze_metadata,
    infer_target_column,
    infer_task_type,
)
from services.data_quality import assess_data_quality
from services.automl import AutoMLService
from services.preprocessing import clean_dataframe, engineer_features, train_val_test_split
from services.preprocessing_pipeline import DataPreprocessor, save_feature_columns
from services.sampling_service import build_sampling_strategy, apply_sampling
from services.training_strategy import build_strategy
from services.visualization import generate_report_plots
from services.cv_service import cross_validate_pipeline
from services.report_llm_service import generate_business_interpretation

logger = logging.getLogger(__name__)


def _file_cache_key(context, parameters):
    """基于文件路径与修改时间生成缓存键（使用 hash，避免超出 Prefect 存储 base path）。"""
    import hashlib

    path = Path(parameters["file_path"])
    mtime = path.stat().st_mtime if path.exists() else 0
    key = hashlib.sha256(f"{path.resolve()}:{mtime}".encode()).hexdigest()
    return key


@task(
    retries=2,
    retry_delay_seconds=5,
    cache_policy=INPUTS + TASK_SOURCE,
    cache_key_fn=_file_cache_key,
)
def load_data_task(file_path: str) -> pd.DataFrame:
    """加载数据任务（按文件路径+修改时间缓存）。"""
    df = load_dataframe(Path(file_path))
    logger.info(f"数据加载完成: {df.shape}")
    return df


@task
def validate_schema_task(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """校验目标列是否存在。"""
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在")
    return df


@task
def analyze_metadata_task(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """元数据分析任务。"""
    metadata = analyze_metadata(df, target_column)
    logger.info(f"元数据分析完成: {metadata['n_samples']} 样本, {metadata['n_features']} 特征")
    return metadata


@task
def clean_data_task(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """数据清洗任务。"""
    cleaned_df = clean_dataframe(df, target_column)
    logger.info(f"数据清洗完成: {cleaned_df.shape}")
    return cleaned_df


@task
def engineer_features_task(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """特征工程任务。"""
    featured_df = engineer_features(df, target_column)
    logger.info(f"特征工程完成: {featured_df.shape}")
    return featured_df


@task
def split_data_task(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "regression",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    rare_class_strategy: str = "auto",
) -> Dict[str, pd.DataFrame]:
    """严格划分训练集、验证集、测试集。

    验证集用于候选模型/配置选择，测试集仅用于最终 unbiased 评估。
    """
    result = train_val_test_split(
        df,
        target_column,
        task_type=task_type,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        rare_class_strategy=rare_class_strategy,
    )
    train_df, val_df, test_df = result["train"], result["val"], result["test"]
    logger.info(
        f"数据划分完成: 训练集 {train_df.shape}, 验证集 {val_df.shape}, 测试集 {test_df.shape}"
    )
    return {"train": train_df, "val": val_df, "test": test_df}


@task(cache_policy=INPUTS + TASK_SOURCE)
def build_strategy_task(
    metadata: Dict[str, Any],
    task_type: str,
    time_budget_minutes: float,
    preset: Optional[str] = None,
    primary_metric: Optional[str] = None,
    max_models: Optional[int] = None,
    candidate_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """根据数据元数据构建训练策略，并合并 Agent 推荐的候选配置。"""
    candidate_config = candidate_config or {}

    # Agent 推荐的候选参数覆盖用户/默认参数
    effective_time_budget = candidate_config.get("time_budget_minutes") or time_budget_minutes
    effective_preset = candidate_config.get("preset") or preset
    effective_primary_metric = candidate_config.get("primary_metric") or primary_metric
    effective_max_models = candidate_config.get("max_models") or max_models

    strategy = build_strategy(
        metadata=metadata,
        task_type=task_type,
        user_time_budget_minutes=effective_time_budget,
        user_preset=effective_preset,
        user_primary_metric=effective_primary_metric,
        user_max_models=effective_max_models,
    )
    strategy_dict = strategy.to_dict()

    # 合并候选配置中的预处理/验证/模型搜索空间覆盖
    if candidate_config.get("preprocessing"):
        strategy_dict.setdefault("preprocessing", {}).update(candidate_config["preprocessing"])
    if candidate_config.get("validation_strategy"):
        strategy_dict["validation_strategy"] = {
            **strategy_dict.get("validation_strategy", {}),
            **candidate_config["validation_strategy"],
        }
    if candidate_config.get("hyperparameters"):
        strategy_dict["hyperparameters"] = candidate_config["hyperparameters"]

    logger.info(f"训练策略: {strategy_dict}")
    return strategy_dict


@task
def build_sampling_strategy_task(
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> Dict[str, Any]:
    """根据训练集构建采样策略。"""
    strategy = build_sampling_strategy(train_df, target_column, task_type)
    logger.info(
        f"采样策略: method={strategy.method}, imbalance_ratio={strategy.imbalance_ratio}"
    )
    return strategy.to_dict()


@task
def apply_sampling_task(
    train_df: pd.DataFrame,
    target_column: str,
    sampling_strategy: Dict[str, Any],
) -> Dict[str, Any]:
    """应用采样策略，返回采样后的训练集和可选样本权重。"""
    from services.sampling_service import SamplingStrategy

    strategy = SamplingStrategy(
        method=sampling_strategy["method"],
        params=sampling_strategy.get("params", {}),
        rationale=sampling_strategy.get("rationale", []),
        imbalance_ratio=sampling_strategy.get("imbalance_ratio"),
    )
    sampled_df, sample_weight = apply_sampling(train_df, target_column, strategy)
    logger.info(
        f"采样完成: method={strategy.method}, "
        f"original={train_df.shape}, sampled={sampled_df.shape}"
    )
    return {
        "sampled_train_df": sampled_df,
        "sample_weight": sample_weight,
    }


@task(timeout_seconds=10800)
def train_model_task(
    train_data: pd.DataFrame,
    target_column: str,
    task_type: str,
    output_dir: str,
    time_limit: Optional[int],
    preset: str,
    primary_metric: Optional[str],
    seed: Optional[int] = None,
    max_models: int = 50,
    strategy: Optional[Dict[str, Any]] = None,
    sample_weight: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """训练模型任务。"""
    automl = AutoMLService(Path(output_dir))
    result = automl.train(
        train_data=train_data,
        target_column=target_column,
        task_type=task_type,
        time_limit=time_limit,
        preset=preset,
        primary_metric=primary_metric,
        seed=seed,
        max_models=max_models,
        strategy=strategy,
        sample_weight=sample_weight,
    )
    logger.info(f"模型训练完成: {result['primary_metric']}")
    return result


@task(timeout_seconds=3600)
def evaluate_model_task(
    test_data: pd.DataFrame,
    val_data: pd.DataFrame,
    train_data: pd.DataFrame,
    target_column: str,
    output_dir: str,
    cv_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """评估模型任务，生成验证集/测试集/训练集指标。

    验证集分数用于候选配置选择；测试集仅用于最终 unbiased 报告。
    """
    from autogluon.tabular import TabularPredictor

    model_path = Path(output_dir) / "autogluon_models"
    predictor = TabularPredictor.load(str(model_path))

    # 验证集评估（候选选择依据）
    val_performance = predictor.evaluate(val_data)
    metrics = {"val": {str(k): float(v) for k, v in val_performance.items()}}

    # 测试集评估（最终报告）
    test_performance = predictor.evaluate(test_data)
    metrics["final"] = {str(k): float(v) for k, v in test_performance.items()}

    # 扩展指标（测试集）
    X_test = test_data.drop(columns=[target_column])
    y_true = test_data[target_column]
    y_pred = predictor.predict(X_test)
    extended = _compute_extended_metrics(y_true, y_pred, X_test, predictor, predictor.problem_type)
    if extended:
        metrics["extended"] = extended

    # 二分类阈值调优（在验证集上搜索，应用于测试集报告）
    if predictor.problem_type == "binary":
        X_val = val_data.drop(columns=[target_column])
        y_val = val_data[target_column]
        threshold_info = _compute_optimal_threshold(y_val, X_val, predictor)
        metrics["threshold"] = threshold_info

    # 集成验证：检查 WeightedEnsemble 是否带来明显提升
    try:
        leaderboard = predictor.leaderboard(silent=True)
        metrics["ensemble_validation"] = _validate_ensemble(leaderboard, None)
    except Exception as e:
        logger.warning(f"集成验证失败: {e}")

    # 训练集参考指标
    if settings.train_eval_enabled:
        try:
            eval_train_data = train_data
            sample_size = settings.train_eval_sample_size
            n_classes = train_data[target_column].nunique(dropna=True)

            if sample_size == 0:
                logger.info("训练集评估已通过配置禁用（TRAIN_EVAL_SAMPLE_SIZE=0）")
                eval_train_data = None
            elif sample_size is not None and len(train_data) > sample_size:
                eval_train_data = train_data.sample(n=sample_size, random_state=42)
                logger.info(f"训练集评估按配置采样至 {len(eval_train_data)} 行")
            elif sample_size is None and (
                n_classes > HIGH_CARDINALITY_CLASS_THRESHOLD
                or len(train_data) > LARGE_DATASET_ROW_THRESHOLD
            ):
                sample_size = min(DEFAULT_TRAIN_EVAL_SAMPLE_SIZE, len(train_data))
                eval_train_data = train_data.sample(n=sample_size, random_state=42)
                logger.info(
                    f"训练集评估自动采样至 {len(eval_train_data)} 行"
                    f"（n_classes={n_classes}）"
                )

            if eval_train_data is not None:
                train_performance = predictor.evaluate(eval_train_data)
                metrics["train"] = {str(k): float(v) for k, v in train_performance.items()}
            else:
                metrics["train"] = {}
        except Exception as e:
            logger.warning(f"训练集评估失败: {e}")
            metrics["train"] = {}
    else:
        logger.warning("训练集评估已通过配置禁用")
        metrics["train"] = {}

    # 显式 CV 结果
    if cv_results:
        metrics["cv"] = cv_results

    # 保存指标到 JSON
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    logger.info(
        f"模型评估完成: val={metrics['val']}, test={metrics['final']}, "
        f"train={metrics.get('train', {})}, cv={metrics.get('cv', {})}"
    )
    return metrics


def _validate_ensemble(leaderboard: pd.DataFrame, primary_metric: Optional[str]) -> Dict[str, Any]:
    """检查集成是否优于单模型；若差距 <2% 则建议使用单模型。"""
    if leaderboard.empty or "model" not in leaderboard.columns:
        return {"ensemble_used": False, "reason": "leaderboard 为空"}

    score_col = "score_val"
    if score_col not in leaderboard.columns:
        score_col = leaderboard.columns[-1]

    sorted_lb = leaderboard.sort_values(score_col, ascending=False)
    top_model = sorted_lb.iloc[0]
    ensemble_used = str(top_model["model"]).startswith("WeightedEnsemble")

    # 找到最佳非集成模型
    non_ensemble = sorted_lb[~sorted_lb["model"].astype(str).str.startswith("WeightedEnsemble")]
    if non_ensemble.empty:
        return {"ensemble_used": ensemble_used, "reason": "无单模型可比较"}

    best_single_score = float(non_ensemble.iloc[0][score_col])
    top_score = float(top_model[score_col])
    improvement = (top_score - best_single_score) / max(abs(best_single_score), 1e-10)

    return {
        "ensemble_used": ensemble_used,
        "top_model": str(top_model["model"]),
        "best_single_model": str(non_ensemble.iloc[0]["model"]),
        "improvement_ratio": round(improvement, 6),
        "recommend_ensemble": ensemble_used and improvement > 0.02,
    }


def _compute_optimal_threshold(y_true, X_test, predictor) -> Dict[str, Any]:
    """在二分类测试集上搜索最优阈值（F1 最大）。"""
    from sklearn import metrics as sk_metrics
    import numpy as np

    pos_label = predictor.class_labels[1]
    y_proba = predictor.predict_proba(X_test)[pos_label].values
    y_true_binary = (y_true == pos_label).astype(int)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0.0
    best_threshold = 0.5
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = sk_metrics.f1_score(y_true_binary, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return {
        "default_threshold": 0.5,
        "optimal_threshold": best_threshold,
        "best_f1": round(best_f1, 6),
    }


def _compute_extended_metrics(
    y_true, y_pred, X_test, predictor, problem_type: str
) -> Dict[str, Any]:
    """计算扩展评估指标。"""
    from sklearn import metrics as sk_metrics
    import numpy as np

    extended: Dict[str, Any] = {}

    if problem_type in ["binary", "multiclass"]:
        # 混淆矩阵
        labels = predictor.class_labels
        extended["labels"] = [str(label) for label in labels]
        extended["confusion_matrix"] = sk_metrics.confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist()

        # Precision / Recall / F1
        average_options = ["macro", "weighted", "micro"]
        for avg in average_options:
            extended[f"precision_{avg}"] = float(
                sk_metrics.precision_score(y_true, y_pred, average=avg, zero_division=0)
            )
            extended[f"recall_{avg}"] = float(
                sk_metrics.recall_score(y_true, y_pred, average=avg, zero_division=0)
            )
            extended[f"f1_{avg}"] = float(
                sk_metrics.f1_score(y_true, y_pred, average=avg, zero_division=0)
            )

        # MCC / Kappa / Balanced Accuracy
        extended["mcc"] = float(sk_metrics.matthews_corrcoef(y_true, y_pred))
        extended["cohens_kappa"] = float(sk_metrics.cohen_kappa_score(y_true, y_pred))
        extended["balanced_accuracy"] = float(sk_metrics.balanced_accuracy_score(y_true, y_pred))

        # AUC-ROC / AUC-PR（二分类 + 多分类 OvR 宏平均）
        if problem_type in ["binary", "multiclass"]:
            try:
                y_proba = predictor.predict_proba(X_test)
                classes = predictor.class_labels

                if problem_type == "binary":
                    pos_label = classes[1]
                    extended["auc_roc"] = float(sk_metrics.roc_auc_score(y_true, y_proba[pos_label]))
                    extended["auc_pr"] = float(
                        sk_metrics.average_precision_score(y_true, y_proba[pos_label])
                    )
                    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_proba[pos_label], pos_label=pos_label)
                    extended["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                    precision_curve, recall_curve, _ = sk_metrics.precision_recall_curve(
                        y_true, y_proba[pos_label], pos_label=pos_label
                    )
                    extended["pr_curve"] = {
                        "precision": precision_curve.tolist(),
                        "recall": recall_curve.tolist(),
                    }
                else:
                    from sklearn.preprocessing import label_binarize

                    y_true_bin = label_binarize(y_true, classes=classes)
                    proba_arr = y_proba[classes].values

                    # ROC 宏平均
                    fpr_grid = np.linspace(0.0, 1.0, 100)
                    tpr_list = []
                    for i, cls in enumerate(classes):
                        fpr_i, tpr_i, _ = sk_metrics.roc_curve(y_true_bin[:, i], y_proba[cls])
                        tpr_list.append(np.interp(fpr_grid, fpr_i, tpr_i))
                    tpr_macro = np.mean(tpr_list, axis=0)
                    extended["auc_roc"] = float(
                        sk_metrics.roc_auc_score(y_true_bin, proba_arr, average="macro", multi_class="ovr")
                    )
                    extended["roc_curve"] = {"fpr": fpr_grid.tolist(), "tpr": tpr_macro.tolist()}

                    # PR 宏平均
                    recall_grid = np.linspace(0.0, 1.0, 100)
                    pr_list = []
                    for i, cls in enumerate(classes):
                        pr_i, rec_i, _ = sk_metrics.precision_recall_curve(y_true_bin[:, i], y_proba[cls])
                        pr_list.append(np.interp(recall_grid, rec_i[::-1], pr_i[::-1]))
                    pr_macro = np.mean(pr_list, axis=0)
                    extended["auc_pr"] = float(
                        sk_metrics.average_precision_score(y_true_bin, proba_arr, average="macro")
                    )
                    extended["pr_curve"] = {
                        "recall": recall_grid.tolist(),
                        "precision": pr_macro.tolist(),
                    }
            except Exception:
                pass

    elif problem_type == "regression":
        extended["mae"] = float(sk_metrics.mean_absolute_error(y_true, y_pred))
        extended["mse"] = float(sk_metrics.mean_squared_error(y_true, y_pred))
        extended["rmse"] = float(np.sqrt(extended["mse"]))
        extended["r2"] = float(sk_metrics.r2_score(y_true, y_pred))

        # MAPE / SMAPE
        y_true_arr = np.array(y_true, dtype=float)
        y_pred_arr = np.array(y_pred, dtype=float)
        mape = (
            np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(np.abs(y_true_arr), 1e-10))) * 100
        )
        extended["mape"] = float(mape)
        smape = (
            np.mean(
                2
                * np.abs(y_true_arr - y_pred_arr)
                / (np.abs(y_true_arr) + np.abs(y_pred_arr) + 1e-10)
            )
            * 100
        )
        extended["smape"] = float(smape)

        # 残差统计
        residuals = y_true_arr - y_pred_arr
        extended["residual_mean"] = float(np.mean(residuals))
        extended["residual_std"] = float(np.std(residuals))

    return extended


@task
def assess_data_quality_task(
    df: pd.DataFrame,
    target_column: str,
    output_dir: str,
) -> Dict[str, Any]:
    """评估数据质量并保存六维报告。"""
    quality = assess_data_quality(df, target_column)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    quality_path = output_path / "quality_report.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"数据质量报告保存: {quality_path}, overall_score={quality.get('overall_score')}")
    return quality


@task
def create_artifacts_task(
    output_dir: str,
    metadata: Dict[str, Any],
    strategy: Dict[str, Any],
    sampling_strategy: Dict[str, Any],
    quality: Optional[Dict[str, Any]] = None,
) -> None:
    """创建 Prefect Artifact，方便在 Prefect UI 中查看关键结果。"""
    output_path = Path(output_dir)

    # Leaderboard Artifact
    leaderboard_path = output_path / "leaderboard.csv"
    if leaderboard_path.exists():
        import pandas as pd

        leaderboard_df = pd.read_csv(leaderboard_path).head(25)
        create_table_artifact(
            key="leaderboard",
            table=leaderboard_df.to_dict(orient="records"),
            description="模型排行榜",
        )

    # 特征重要性 Artifact
    importance_path = output_path / "feature_importance.csv"
    if importance_path.exists():
        import pandas as pd

        importance_df = pd.read_csv(importance_path).head(20)
        create_table_artifact(
            key="feature-importance",
            table=importance_df.to_dict(orient="records"),
            description="Top 20 特征重要性",
        )

    # 策略 Markdown Artifact
    rationale = "\n".join(f"- {item}" for item in strategy.get("rationale", []))
    preprocessing = strategy.get("preprocessing", {})
    markdown = (
        f"# 训练策略\n\n"
        f"**数据规模**: {strategy.get('data_size_label')}\n\n"
        f"**Preset**: {strategy.get('preset')}\n\n"
        f"**主指标**: {strategy.get('primary_metric')}\n\n"
        f"**时间限制**: {strategy.get('time_limit_seconds') if strategy.get('time_limit_seconds') is not None else '无限制'}\n\n"
        f"**采样策略**: {sampling_strategy.get('method')}\n\n"
        f"**决策依据**:\n{rationale}\n\n"
        f"**预处理开关**:\n```json\n{json.dumps(preprocessing, ensure_ascii=False, indent=2)}\n```"
    )
    create_markdown_artifact(key="training-strategy", markdown=markdown)

    # 数据质量摘要 Artifact
    if quality:
        quality_summary = {
            "overall_score": quality.get("overall_score"),
            "n_rows": quality.get("n_rows"),
            "n_columns": quality.get("n_columns"),
            "warnings": quality.get("warnings", []),
        }
        create_markdown_artifact(
            key="data-quality-summary",
            markdown=f"```json\n{json.dumps(quality_summary, ensure_ascii=False, indent=2)}\n```",
        )


@task
def generate_report_task(
    run_id: str,
    output_dir: str,
    metadata: Dict[str, Any],
    strategy: Dict[str, Any],
    primary_metric: Optional[str],
    task_type: str,
    seed: Optional[int] = None,
    quality: Optional[Dict[str, Any]] = None,
    interpretation: Optional[Dict[str, Any]] = None,
) -> str:
    """生成 HTML 报告任务。"""
    output_path = Path(output_dir)

    # 读取数据质量报告（若未传入则尝试本地文件）
    if quality is None:
        quality_path = output_path / "quality_report.json"
        if quality_path.exists():
            with open(quality_path, "r", encoding="utf-8") as f:
                quality = json.load(f)
    if quality is None:
        quality = {}

    # 读取业务解读（若未传入则尝试本地文件）
    if interpretation is None:
        interpretation_path = output_path / "business_interpretation.json"
        if interpretation_path.exists():
            with open(interpretation_path, "r", encoding="utf-8") as f:
                interpretation = json.load(f)

    # 读取指标
    metrics_path = output_path / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f).get("final", {})

    # 读取排行榜
    leaderboard_path = output_path / "leaderboard.csv"
    leaderboard = []
    leaderboard_columns = []
    if leaderboard_path.exists():
        import pandas as pd

        leaderboard_df = pd.read_csv(leaderboard_path).head(25)
        leaderboard = leaderboard_df.to_dict(orient="records")
        leaderboard_columns = leaderboard_df.columns.tolist()

    # 读取特征重要性
    importance_path = output_path / "feature_importance.csv"
    feature_importance = []
    importance_columns = []
    if importance_path.exists():
        import pandas as pd

        importance_df = pd.read_csv(importance_path).head(20)
        feature_importance = importance_df.to_dict(orient="records")
        importance_columns = importance_df.columns.tolist()

    # 渲染模板
    template_dir = settings.project_root / "backend" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    # 生成可视化图表
    plots = generate_report_plots(output_path, task_type)

    # 读取 Permutation Importance
    perm_importance_path = output_path / "permutation_importance.csv"
    perm_importance = []
    perm_importance_columns = []
    if perm_importance_path.exists():
        import pandas as pd

        perm_df = pd.read_csv(perm_importance_path).head(20)
        perm_importance = perm_df.to_dict(orient="records")
        perm_importance_columns = perm_df.columns.tolist()

    template = env.get_template("report.html")
    html_content = template.render(
        run_id=run_id,
        status="completed",
        preset=strategy.get("preset"),
        strategy=strategy,
        primary_metric=primary_metric,
        seed=seed,
        metadata=metadata,
        metrics=metrics,
        leaderboard=leaderboard,
        leaderboard_columns=leaderboard_columns,
        feature_importance=feature_importance,
        importance_columns=importance_columns,
        perm_importance=perm_importance,
        perm_importance_columns=perm_importance_columns,
        plots=plots,
        quality=quality,
        metrics_full=metrics,
        interpretation=interpretation,
    )

    report_path = output_path / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML 报告生成完成: {report_path}")
    return str(report_path)


@task
def fit_preprocessor_task(
    train_df: pd.DataFrame,
    target_column: str,
    strategy: Dict[str, Any],
    cleaning_rules: Optional[Dict[str, Any]] = None,
) -> DataPreprocessor:
    """在训练集上拟合预处理 Pipeline。"""
    preprocessor = DataPreprocessor(
        target_column=target_column,
        strategy=strategy,
        cleaning_rules=cleaning_rules,
    )
    preprocessor.fit(train_df)
    logger.info(
        f"预处理器拟合完成: 对数变换列={preprocessor.log_transform_cols}, "
        f"最终特征数={len(preprocessor.feature_columns)}"
    )
    return preprocessor


@task
def transform_data_task(
    preprocessor: DataPreprocessor,
    df: pd.DataFrame,
    dataset_label: str = "data",
) -> pd.DataFrame:
    """使用已拟合的预处理器转换数据。"""
    transformed = preprocessor.transform(df)
    logger.info(f"{dataset_label} 转换完成: {transformed.shape}")
    return transformed


@task
def persist_preprocessor_task(
    preprocessor: DataPreprocessor,
    output_dir: str,
) -> Dict[str, str]:
    """保存预处理 Pipeline 与特征列清单。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_path / "preprocessing_pipeline.joblib"
    preprocessor.save(pipeline_path)
    save_feature_columns(output_dir, preprocessor.feature_columns)
    feature_columns_path = output_path / "feature_columns.json"

    logger.info(f"预处理器与特征列已保存: {pipeline_path}, {feature_columns_path}")
    return {
        "pipeline_path": str(pipeline_path),
        "feature_columns_path": str(feature_columns_path),
    }


@task
def cross_validate_task(
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    strategy: Dict[str, Any],
    cleaning_rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对完整预处理 + 基线模型 Pipeline 做显式交叉验证。"""
    cv_results = cross_validate_pipeline(
        train_df=train_df,
        target_column=target_column,
        task_type=task_type,
        strategy=strategy,
        cleaning_rules=cleaning_rules,
        n_folds=strategy.get("validation_strategy", {}).get("n_folds", 5),
        cv_type=strategy.get("validation_strategy", {}).get("cv_type"),
    )
    logger.info(
        f"交叉验证完成: cv_type={cv_results.get('cv_type')}, "
        f"mean={cv_results.get('cv_mean')}, std={cv_results.get('cv_std')}"
    )
    return cv_results


@task
async def generate_business_interpretation_task(
    output_dir: str,
    task_type: str,
    primary_metric: Optional[str],
    quality: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """生成 LLM 业务解读并保存到 JSON。"""
    output_path = Path(output_dir)

    # 读取测试集指标
    metrics_path = output_path / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # 读取特征重要性
    importance_path = output_path / "feature_importance.csv"
    feature_importance = []
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).head(10)
        feature_importance = importance_df.to_dict(orient="records")

    try:
        interpretation = await generate_business_interpretation(
            task_type=task_type,
            primary_metric=primary_metric,
            metrics=metrics,
            feature_importance=feature_importance,
            quality=quality,
            strategy=strategy,
        )
    except Exception as e:
        logger.warning(f"业务解读生成失败: {e}")
        return None

    interpretation_path = output_path / "business_interpretation.json"
    with open(interpretation_path, "w", encoding="utf-8") as f:
        json.dump(interpretation, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"业务解读已保存: {interpretation_path}")
    return interpretation


@flow(name="automl-end-to-end", log_prints=True)
def automl_pipeline(
    file_path: str,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    output_dir: str = "./automl_output",
    time_budget_minutes: Optional[float] = 10.0,
    preset: Optional[str] = None,
    primary_metric: Optional[str] = None,
    seed: Optional[int] = None,
    max_models: int = 50,
    cleaning_rules: Optional[Dict[str, Any]] = None,
    feature_engineering_enabled: bool = True,
    candidate_config: Optional[Dict[str, Any]] = None,
    rare_class_strategy: str = "auto",
) -> Dict[str, Any]:
    """
    端到端 AutoML Pipeline。

    Args:
        file_path: 数据集文件路径
        target_column: 目标列名（可选，为空时自动推断）
        task_type: 任务类型（可选，为空时自动推断）
        output_dir: 输出目录
        time_budget_minutes: 时间预算（分钟）
        preset: AutoGluon preset
        primary_metric: 主要评估指标
        candidate_config: Agent 推荐的候选配置（可选）
    """
    logger.info(f"启动 AutoML Pipeline: {file_path}")

    # 1. 加载数据
    df = load_data_task(file_path)

    # 1.5 自动推断目标列和任务类型（无人干预）
    if target_column is None or task_type is None:
        inferred_target, confidence = infer_target_column(df, hint=target_column)
        if inferred_target is None:
            raise ValueError("无法自动推断目标列，请显式指定 target_column")
        target_column = inferred_target
        logger.info(f"自动推断目标列: {target_column} (置信度 {confidence:.2f})")

    if task_type is None:
        task_type = infer_task_type(df[target_column])
        logger.info(f"自动推断任务类型: {task_type}")

    logger.info(f"使用目标列/任务: {target_column} / {task_type}")

    # 2. 校验 Schema
    df = validate_schema_task(df, target_column)

    # 3. 元数据分析（基于原始数据）
    metadata = analyze_metadata_task(df, target_column)

    # 3.5 数据质量六维评估（可选，失败不影响主流程）
    try:
        quality = assess_data_quality_task(df, target_column, output_dir)
    except Exception as e:
        logger.warning(f"数据质量评估失败，继续主流程: {e}")
        quality = {}

    # 4. 策略路由（数据驱动）
    strategy = build_strategy_task(
        metadata=metadata,
        task_type=task_type,
        time_budget_minutes=time_budget_minutes,
        preset=preset,
        primary_metric=primary_metric,
        max_models=max_models,
        candidate_config=candidate_config,
    )
    strategy["feature_engineering_enabled"] = (
        candidate_config.get("feature_engineering_enabled")
        if candidate_config and candidate_config.get("feature_engineering_enabled") is not None
        else feature_engineering_enabled
    )

    # 合并 Agent 推荐的清洗规则
    effective_cleaning_rules: Dict[str, Any] = cleaning_rules or {}
    if candidate_config and candidate_config.get("cleaning_rules"):
        effective_cleaning_rules = {
            **effective_cleaning_rules,
            **candidate_config["cleaning_rules"],
        }

    # 5. 划分训练集/验证集/测试集（必须在预处理前，防止数据泄露）
    split_result = split_data_task(
        df, target_column, task_type=task_type, rare_class_strategy=rare_class_strategy
    )
    train_df_raw = split_result["train"]
    val_df_raw = split_result["val"]
    test_df_raw = split_result["test"]

    # 5.5 显式交叉验证（可选，失败不影响主流程）
    try:
        cv_results = cross_validate_task(
            train_df=train_df_raw,
            target_column=target_column,
            task_type=task_type,
            strategy=strategy,
            cleaning_rules=effective_cleaning_rules,
        )
    except Exception as e:
        logger.warning(f"交叉验证失败，继续主流程: {e}")
        cv_results = {}

    # 6. 拟合并应用预处理 Pipeline（仅在训练集上 fit，再 transform 全量数据）
    preprocessor = fit_preprocessor_task(
        train_df=train_df_raw,
        target_column=target_column,
        strategy=strategy,
        cleaning_rules=effective_cleaning_rules,
    )
    train_df = transform_data_task(
        preprocessor=preprocessor,
        df=train_df_raw,
        dataset_label="训练集",
    )
    val_df = transform_data_task(
        preprocessor=preprocessor,
        df=val_df_raw,
        dataset_label="验证集",
    )
    test_df = transform_data_task(
        preprocessor=preprocessor,
        df=test_df_raw,
        dataset_label="测试集",
    )
    persist_preprocessor_task(
        preprocessor=preprocessor,
        output_dir=output_dir,
    )
    logger.info(
        f"预处理完成: 训练集 {train_df.shape}, 验证集 {val_df.shape}, 测试集 {test_df.shape}, "
        f"特征列={preprocessor.feature_columns}"
    )

    # 7. 条件采样（仅在训练集上）
    sampling_strategy = build_sampling_strategy_task(train_df, target_column, task_type)
    sampling_result = apply_sampling_task(train_df, target_column, sampling_strategy)
    sampled_train_df = sampling_result["sampled_train_df"]
    sample_weight = sampling_result["sample_weight"]

    # 8. 训练模型
    train_result = train_model_task(
        train_data=sampled_train_df,
        target_column=target_column,
        task_type=task_type,
        output_dir=output_dir,
        time_limit=strategy["time_limit_seconds"],
        preset=strategy["preset"],
        primary_metric=strategy["primary_metric"],
        seed=seed,
        max_models=strategy["max_models"],
        strategy=strategy,
        sample_weight=sample_weight,
    )

    # 9. 评估（验证集用于候选选择，测试集用于最终 unbiased 评估，训练集为参考）
    metrics = evaluate_model_task(
        test_df, val_df, train_df, target_column, output_dir, cv_results=cv_results
    )

    # 9.5 LLM 业务解读（可选，失败不影响主流程）
    interpretation = generate_business_interpretation_task(
        output_dir=output_dir,
        task_type=task_type,
        primary_metric=strategy.get("primary_metric"),
        quality=quality,
        strategy=strategy,
    )

    # 10. 创建 Prefect Artifact（可选）
    try:
        create_artifacts_task(
            output_dir=output_dir,
            metadata=metadata,
            strategy=strategy,
            sampling_strategy=sampling_strategy,
            quality=quality,
        )
    except Exception as e:
        logger.warning(f"Prefect Artifact 创建失败，继续主流程: {e}")

    # 11. 生成 HTML 报告（可选，失败不影响主流程）
    report_path = None
    try:
        strategy["sampling"] = sampling_strategy
        report_path = generate_report_task(
            run_id=Path(output_dir).name,
            output_dir=output_dir,
            metadata=metadata,
            strategy=strategy,
            primary_metric=strategy.get("primary_metric"),
            task_type=task_type,
            seed=seed,
            quality=quality,
            interpretation=interpretation,
        )
    except Exception as e:
        logger.warning(f"HTML 报告生成失败，继续主流程: {e}")

    return {
        "status": "completed",
        "metadata": metadata,
        "strategy": strategy,
        "train_result": train_result,
        "metrics": metrics,
        "report_path": report_path,
    }
