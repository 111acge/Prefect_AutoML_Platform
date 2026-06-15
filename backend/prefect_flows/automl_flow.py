"""Prefect AutoML Flow 定义。"""

import os

os.environ.setdefault("PREFECT_API_URL", "")

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from prefect import flow, task

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import settings
from services.data_service import load_dataframe, analyze_metadata
from services.automl import AutoMLService
from services.preprocessing import clean_dataframe, engineer_features, split_data
from services.preprocessing_pipeline import DataPreprocessor, save_feature_columns
from services.training_strategy import build_strategy
from services.visualization import generate_report_plots

logger = logging.getLogger(__name__)


@task(retries=2, retry_delay_seconds=5)
def load_data_task(file_path: str) -> pd.DataFrame:
    """加载数据任务。"""
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
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """划分训练集和测试集。"""
    result = split_data(
        df, target_column, task_type=task_type, test_size=test_size, random_state=random_state
    )
    train_df, test_df = result["train"], result["test"]
    logger.info(f"数据划分完成: 训练集 {train_df.shape}, 测试集 {test_df.shape}")
    return {"train": train_df, "test": test_df}


@task
def build_strategy_task(
    metadata: Dict[str, Any],
    task_type: str,
    time_budget_minutes: float,
    preset: Optional[str] = None,
    primary_metric: Optional[str] = None,
    max_models: Optional[int] = None,
) -> Dict[str, Any]:
    """根据数据元数据构建训练策略。"""
    strategy = build_strategy(
        metadata=metadata,
        task_type=task_type,
        user_time_budget_minutes=time_budget_minutes,
        user_preset=preset,
        user_primary_metric=primary_metric,
        user_max_models=max_models,
    )
    logger.info(f"训练策略: {strategy.to_dict()}")
    return strategy.to_dict()


@task
def train_model_task(
    train_data: pd.DataFrame,
    target_column: str,
    task_type: str,
    output_dir: str,
    time_limit: int,
    preset: str,
    primary_metric: Optional[str],
    seed: Optional[int] = None,
    max_models: int = 50,
    strategy: Optional[Dict[str, Any]] = None,
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
    )
    logger.info(f"模型训练完成: {result['primary_metric']}")
    return result


@task
def evaluate_model_task(
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    target_column: str,
    output_dir: str,
) -> Dict[str, float]:
    """评估模型任务，生成测试集标准指标、扩展指标和训练集参考指标。"""
    from autogluon.tabular import TabularPredictor

    model_path = Path(output_dir) / "autogluon_models"
    predictor = TabularPredictor.load(str(model_path))

    # 使用测试集评估，生成标准指标字典
    test_performance = predictor.evaluate(test_data)
    metrics = {"final": {str(k): float(v) for k, v in test_performance.items()}}

    # 扩展指标（测试集）
    X_test = test_data.drop(columns=[target_column])
    y_true = test_data[target_column]
    y_pred = predictor.predict(X_test)
    extended = _compute_extended_metrics(y_true, y_pred, X_test, predictor, predictor.problem_type)
    if extended:
        metrics["extended"] = extended

    # 训练集参考指标
    try:
        train_performance = predictor.evaluate(train_data)
        metrics["train"] = {str(k): float(v) for k, v in train_performance.items()}
    except Exception as e:
        logger.warning(f"训练集评估失败: {e}")
        metrics["train"] = {}

    # 保存指标到 JSON
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"模型评估完成: test={metrics['final']}, train={metrics.get('train', {})}")
    return metrics


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

        # AUC-ROC / AUC-PR（二分类）
        if problem_type == "binary":
            try:
                y_proba = predictor.predict_proba(X_test)
                pos_label = predictor.class_labels[1]
                extended["auc_roc"] = float(sk_metrics.roc_auc_score(y_true, y_proba[pos_label]))
                extended["auc_pr"] = float(
                    sk_metrics.average_precision_score(y_true, y_proba[pos_label])
                )

                # ROC / PR 曲线数据，用于报告图表
                fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_proba[pos_label], pos_label=pos_label)
                extended["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }
                precision_curve, recall_curve, _ = sk_metrics.precision_recall_curve(
                    y_true, y_proba[pos_label], pos_label=pos_label
                )
                extended["pr_curve"] = {
                    "precision": precision_curve.tolist(),
                    "recall": recall_curve.tolist(),
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
def generate_report_task(
    run_id: str,
    output_dir: str,
    metadata: Dict[str, Any],
    strategy: Dict[str, Any],
    primary_metric: Optional[str],
    task_type: str,
    seed: Optional[int] = None,
) -> str:
    """生成 HTML 报告任务。"""
    output_path = Path(output_dir)

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

        leaderboard_df = pd.read_csv(leaderboard_path).head(10)
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
        plots=plots,
    )

    report_path = output_path / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML 报告生成完成: {report_path}")
    return str(report_path)


@flow(name="automl-end-to-end", log_prints=True)
def automl_pipeline(
    file_path: str,
    target_column: str,
    task_type: str,
    output_dir: str,
    time_budget_minutes: float = 10.0,
    preset: Optional[str] = None,
    primary_metric: Optional[str] = None,
    seed: Optional[int] = None,
    max_models: int = 50,
) -> Dict[str, Any]:
    """
    端到端 AutoML Pipeline。

    Args:
        file_path: 数据集文件路径
        target_column: 目标列名
        task_type: 任务类型
        output_dir: 输出目录
        time_budget_minutes: 时间预算（分钟）
        preset: AutoGluon preset
        primary_metric: 主要评估指标
    """
    logger.info(f"启动 AutoML Pipeline: {file_path} -> {target_column}")

    # 1. 加载数据
    df = load_data_task(file_path)

    # 2. 校验 Schema
    df = validate_schema_task(df, target_column)

    # 3. 元数据分析（基于原始数据）
    metadata = analyze_metadata_task(df, target_column)

    # 4. 策略路由（数据驱动）
    strategy = build_strategy_task(
        metadata=metadata,
        task_type=task_type,
        time_budget_minutes=max(time_budget_minutes, 0.5),
        preset=preset,
        primary_metric=primary_metric,
        max_models=max_models,
    )

    # 5. 拟合并应用预处理 Pipeline（按策略执行）
    preprocessor = DataPreprocessor(target_column=target_column, strategy=strategy)
    df = preprocessor.fit_transform(df)
    preprocessor.save(Path(output_dir) / "preprocessing_pipeline.joblib")
    save_feature_columns(output_dir, preprocessor.feature_columns)
    logger.info(f"预处理完成: {df.shape}, 特征列={preprocessor.feature_columns}")

    # 6. 划分训练集/测试集
    split_result = split_data_task(df, target_column, task_type=task_type)
    train_df = split_result["train"]
    test_df = split_result["test"]

    # 7. 训练模型
    train_result = train_model_task(
        train_data=train_df,
        target_column=target_column,
        task_type=task_type,
        output_dir=output_dir,
        time_limit=strategy["time_limit_seconds"],
        preset=strategy["preset"],
        primary_metric=strategy["primary_metric"],
        seed=seed,
        max_models=strategy["max_models"],
        strategy=strategy,
    )

    # 8. 评估（测试集为主，训练集为参考）
    metrics = evaluate_model_task(test_df, train_df, target_column, output_dir)

    # 9. 生成 HTML 报告
    report_path = generate_report_task(
        run_id=Path(output_dir).name,
        output_dir=output_dir,
        metadata=metadata,
        strategy=strategy,
        primary_metric=strategy.get("primary_metric"),
        task_type=task_type,
        seed=seed,
    )

    return {
        "status": "completed",
        "metadata": metadata,
        "strategy": strategy,
        "train_result": train_result,
        "metrics": metrics,
        "report_path": report_path,
    }
