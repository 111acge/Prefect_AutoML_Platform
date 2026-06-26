# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""条件采样服务。

根据数据特征自动选择并应用采样策略，仅用于处理类别不平衡。
所有采样/加权操作只在训练集上进行，验证集和测试集保持原始分布。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from i18n import _
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


@dataclass
class SamplingStrategy:
    """采样策略定义。"""

    method: str  # none / class_weight / random_over / smote / smotenc / adasyn / random_under / smote_enn
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    imbalance_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "rationale": self.rationale,
            "imbalance_ratio": self.imbalance_ratio,
        }


def _imbalance_ratio(y: pd.Series) -> float:
    """计算类别不平衡比例。"""
    counts = y.value_counts()
    if len(counts) < 2:
        return 1.0
    return float(counts.max() / counts.min())


def _safe_k_neighbors(y: pd.Series, default: int = 5) -> int:
    """根据最小类样本数计算安全的 k_neighbors，避免 SMOTE 崩溃。"""
    min_class_count = y.value_counts().min()
    # SMOTE 要求 k_neighbors < min_class_count
    safe_k = min(default, max(1, int(min_class_count) - 1))
    return safe_k


def build_sampling_strategy(
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> SamplingStrategy:
    """根据训练集特征构建采样策略。

    Args:
        train_df: 训练集（已清洗、已特征工程）。
        target_column: 目标列名。
        task_type: 任务类型。
        metadata: 可选元数据。

    Returns:
        SamplingStrategy
    """
    if task_type not in ("binary_classification", "multiclass_classification"):
        return SamplingStrategy(
            method="none",
            rationale=[_("strategy.not_classification_no_sampling")],
        )

    y = train_df[target_column]
    ratio = _imbalance_ratio(y)

    if ratio <= 1.5:
        return SamplingStrategy(
            method="none",
            imbalance_ratio=ratio,
            rationale=[_("strategy.balanced_no_sampling", ratio=ratio)],
        )

    rationale = [_("strategy.detected_imbalance", ratio=f"{ratio:.2f}")]

    # 轻度不平衡：优先用类别权重，不改动数据分布
    if ratio <= 3.0:
        return SamplingStrategy(
            method="class_weight",
            imbalance_ratio=ratio,
            rationale=rationale + [_("strategy.mild_imbalance_class_weight")],
        )

    # 中度及以上不平衡：考虑重采样
    feature_df = train_df.drop(columns=[target_column])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    n_samples = len(train_df)

    # 大数据量优先欠采样，避免训练时间爆炸
    if n_samples > 100_000:
        min_class_count = int(y.value_counts().min())
        # 若最小类样本数过少，RandomUnderSampler(auto) 会把所有类别欠采样到同样大小，
        # 在多分类场景下可能导致训练集仅剩 1 条/类，直接崩溃。
        if min_class_count < 100:
            return SamplingStrategy(
                method="class_weight",
                imbalance_ratio=ratio,
                rationale=rationale + [_("strategy.large_data_small_min_class", n_samples=n_samples, min_class=min_class_count)],
            )
        strategy = SamplingStrategy(
            method="random_under",
            params={"sampling_strategy": "auto", "random_state": 42},
            imbalance_ratio=ratio,
            rationale=rationale + [_("strategy.large_data_under_sample", n_samples=n_samples, min_class=min_class_count)],
        )
        return strategy

    # 无缺失值且全数值：可用 SMOTE / ADASYN
    has_missing = feature_df.isnull().any().any()
    safe_k = _safe_k_neighbors(y)
    if safe_k <= 0:
        return SamplingStrategy(
            method="random_over",
            params={"random_state": 42},
            imbalance_ratio=ratio,
            rationale=rationale + [_("strategy.smote_unsafe_oversample")],
        )

    if not has_missing and len(categorical_cols) == 0:
        if ratio <= 10.0:
            strategy = SamplingStrategy(
                method="smote",
                params={"k_neighbors": safe_k, "random_state": 42},
                imbalance_ratio=ratio,
                rationale=rationale + [_("strategy.numeric_no_missing_smote")],
            )
        else:
            strategy = SamplingStrategy(
                method="smote_enn",
                params={"k_neighbors": safe_k, "random_state": 42},
                imbalance_ratio=ratio,
                rationale=rationale + [_("strategy.severe_imbalance_smoteenn")],
            )
        return strategy

    # 含类别特征：使用 SMOTENC 或回退到 RandomOverSampler
    if categorical_cols and not has_missing:
        cat_indices = [feature_df.columns.get_loc(c) for c in categorical_cols]
        strategy = SamplingStrategy(
            method="smotenc",
            params={
                "categorical_features": cat_indices,
                "k_neighbors": safe_k,
                "random_state": 42,
            },
            imbalance_ratio=ratio,
            rationale=rationale + [_("strategy.categorical_smotenc", columns=categorical_cols)],
        )
        return strategy

    # 默认回退：RandomOverSampler 对缺失值和混合类型最宽容
    strategy = SamplingStrategy(
        method="random_over",
        params={"random_state": 42},
        imbalance_ratio=ratio,
        rationale=rationale + [_("strategy.mixed_random_over")],
    )
    return strategy


def compute_sample_weight_series(y: pd.Series, strategy: str = "balanced") -> pd.Series:
    """计算样本权重列。"""
    weights = compute_sample_weight(strategy, y)
    return pd.Series(weights, index=y.index, name="_sample_weight")


def apply_sampling(
    train_df: pd.DataFrame,
    target_column: str,
    strategy: SamplingStrategy,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """应用采样策略到训练集。

    Returns:
        (sampled_train_df, sample_weight_series)
        - 如果 strategy.method == "class_weight"，则不改变 train_df，返回样本权重。
        - 否则返回重采样后的 train_df，sample_weight 为 None。
        - method == "none" 时原样返回。
    """
    method = strategy.method
    sample_weight_col = "_sample_weight"

    if method == "none":
        return train_df, None

    if method == "class_weight":
        sample_weight = compute_sample_weight_series(train_df[target_column])
        train_df = train_df.copy()
        train_df[sample_weight_col] = sample_weight.values
        logger.info(_("sampling.applied_class_weight", shape=train_df.shape))
        return train_df, None

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    sampler = _build_sampler(method, strategy.params)
    if sampler is None:
        logger.warning(_("sampling.unknown_method", method=method))
        return train_df, None

    try:
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as e:
        logger.warning(_("sampling.failed_fallback", method=method, msg=e))
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

    # 重建 DataFrame
    resampled_df = pd.concat([X_res, y_res], axis=1)
    # 保持原始列顺序
    resampled_df = resampled_df[train_df.columns]

    logger.info(
        _(
            "sampling.applied",
            method=method,
            old_shape=train_df.shape,
            new_shape=resampled_df.shape,
            ratio=f"{_imbalance_ratio(resampled_df[target_column]):.2f}",
        )
    )
    return resampled_df, None


def _build_sampler(method: str, params: Dict[str, Any]):
    """根据方法名和参数构造 imbalanced-learn 采样器。"""
    mapping = {
        "random_over": RandomOverSampler,
        "smote": SMOTE,
        "smotenc": SMOTENC,
        "adasyn": ADASYN,
        "random_under": RandomUnderSampler,
        "smote_enn": SMOTEENN,
    }
    cls = mapping.get(method)
    if cls is None:
        return None
    return cls(**params)
