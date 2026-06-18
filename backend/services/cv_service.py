"""交叉验证策略服务。

根据任务类型与数据特征自动选择并执行显式 CV：
- 分类：StratifiedKFold
- 回归：KFold
- 时间序列：TimeSeriesSplit
- 分组数据：GroupKFold

CV 在原始训练数据上通过 sklearn Pipeline 完成，确保预处理步骤也在每折内 fit，
防止数据泄露。
"""

from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    GroupKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from services.preprocessing_pipeline import DataPreprocessor


def get_cv_splitter(
    task_type: str,
    n_samples: int,
    n_folds: int = 5,
    cv_type: Optional[str] = None,
) -> Any:
    """根据任务类型获取 CV 分割器。

    Args:
        task_type: binary_classification / multiclass_classification / regression
        n_samples: 样本数
        n_folds: 折数
        cv_type: 可选指定 cv / kfold / stratified / timeseries / group

    Returns:
        sklearn CV splitter
    """
    n_folds = max(2, min(n_folds, n_samples // 2))

    if cv_type == "timeseries" or cv_type == "time_series":
        return TimeSeriesSplit(n_splits=n_folds)
    if cv_type == "group":
        return GroupKFold(n_splits=n_folds)

    if task_type in ("binary_classification", "multiclass_classification"):
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return KFold(n_splits=n_folds, shuffle=True, random_state=42)


def _sklearn_scorer_name(primary_metric: str, task_type: str) -> str:
    """把内部指标名映射为 sklearn scoring 名称（越大越好）。"""
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "roc_auc": "roc_auc_ovr_weighted",
        "log_loss": "neg_log_loss",
        "root_mean_squared_error": "neg_root_mean_squared_error",
        "mean_squared_error": "neg_mean_squared_error",
        "mean_absolute_error": "neg_mean_absolute_error",
        "r2": "r2",
    }
    normalized = primary_metric.replace("-", "_").lower()
    if normalized in mapping:
        return mapping[normalized]

    if task_type in ("binary_classification", "multiclass_classification"):
        return "f1_weighted"
    return "neg_root_mean_squared_error"


def _build_baseline_estimator(task_type: str) -> BaseEstimator:
    """构造一个轻量基线模型用于 CV 评估。"""
    if task_type == "binary_classification":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if task_type == "multiclass_classification":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def cross_validate_pipeline(
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    strategy: Dict[str, Any],
    cleaning_rules: Optional[Dict[str, Any]] = None,
    n_folds: int = 5,
    cv_type: Optional[str] = None,
) -> Dict[str, Any]:
    """对完整预处理 + 基线模型 Pipeline 做交叉验证。

    Returns:
        {
            "cv_scores": List[float],
            "cv_mean": float,
            "cv_std": float,
            "cv_type": str,
            "n_folds": int,
            "primary_metric": str,
            "scorer": str,
        }
    """
    if target_column not in train_df.columns:
        return {}

    # 若未显式指定，从 strategy 中读取 CV 配置
    validation_strategy = strategy.get("validation_strategy", {}) if strategy else {}
    if n_folds == 5 and "n_folds" in validation_strategy:
        n_folds = int(validation_strategy["n_folds"])
    if cv_type is None and "cv_type" in validation_strategy:
        cv_type = validation_strategy["cv_type"]

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    n_samples = len(train_df)
    cv = get_cv_splitter(task_type, n_samples, n_folds=n_folds, cv_type=cv_type)

    primary_metric = strategy.get("primary_metric")
    if not primary_metric:
        primary_metric = "f1" if task_type in ("binary_classification", "multiclass_classification") else "root_mean_squared_error"

    scorer = _sklearn_scorer_name(primary_metric, task_type)

    preprocessor = DataPreprocessor(
        target_column=target_column,
        strategy=strategy,
        cleaning_rules=cleaning_rules or {},
    )
    model = _build_baseline_estimator(task_type)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        return {
            "cv_scores": [round(float(s), 6) for s in scores],
            "cv_mean": round(float(np.mean(scores)), 6),
            "cv_std": round(float(np.std(scores)), 6),
            "cv_type": type(cv).__name__,
            "n_folds": len(scores),
            "primary_metric": primary_metric,
            "scorer": scorer,
        }
    except Exception as e:
        return {
            "cv_scores": [],
            "cv_error": str(e),
            "cv_type": type(cv).__name__,
            "n_folds": n_folds,
            "primary_metric": primary_metric,
            "scorer": scorer,
        }
