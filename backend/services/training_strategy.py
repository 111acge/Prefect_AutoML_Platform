"""数据驱动的训练策略生成服务。

根据数据集的元数据（样本量、特征数、字段类型、目标分布、缺失率等）
自动选择 AutoML 训练策略，包括 preset、超参、验证方式、样本权重、
预处理参数等。
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TrainingStrategy:
    """训练策略定义。"""

    data_size_label: str  # small / medium / large
    preset: str
    # None 表示不限制训练时间（无穷大）
    time_limit_seconds: Optional[int]
    max_models: int
    auto_stack: bool
    num_bag_folds: int
    num_stack_levels: int
    primary_metric: Optional[str]
    use_sample_weight: bool
    sample_weight_strategy: str
    validation_strategy: Dict[str, Any]
    preprocessing: Dict[str, Any]
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_strategy(
    metadata: Dict[str, Any],
    task_type: str,
    user_time_budget_minutes: Optional[float] = 10.0,
    user_preset: Optional[str] = None,
    user_primary_metric: Optional[str] = None,
    user_max_models: Optional[int] = None,
) -> TrainingStrategy:
    """基于数据元数据构建训练策略。

    Args:
        metadata: analyze_metadata 返回的字典。
        task_type: binary_classification / multiclass_classification / regression。
        user_time_budget_minutes: 用户设置的时间预算（分钟）。
        user_preset: 用户指定的 preset，非空时优先使用。
        user_primary_metric: 用户指定的主评估指标。
        user_max_models: 用户指定的最大模型数。

    Returns:
        TrainingStrategy 对象。
    """
    n_samples = int(metadata.get("n_samples", 1000))
    n_features = int(metadata.get("n_features", 10))
    memory_mb = float(metadata.get("memory_mb", 10.0))
    target_info = metadata.get("target_info") or {}
    field_types = metadata.get("field_types", {})
    missing_rates = metadata.get("missing_rates", {})

    rationale: List[str] = []

    # 1. 数据规模分级
    if n_samples < 1000:
        data_size_label = "small"
    elif n_samples < 100_000:
        data_size_label = "medium"
    else:
        data_size_label = "large"
    rationale.append(f"数据规模: {data_size_label} ({n_samples} 样本, {n_features} 特征)")

    # 2. 选择 preset
    preset = user_preset
    if preset in (None, "", "auto"):
        if data_size_label == "large" or n_features > 500 or memory_mb > 2048:
            preset = "medium_quality"
            rationale.append("大数据/高维场景，使用 medium_quality 控制训练时间")
        elif data_size_label == "small":
            # 小数据集用 best_quality 容易过拟合，且收益低
            preset = "good_quality"
            rationale.append("小数据集使用 good_quality，避免 best_quality 过度堆叠")
        else:
            preset = "best_quality"
            rationale.append("中等规模数据使用 best_quality")

    # 3. 时间预算（秒）
    if user_time_budget_minutes is None:
        # None 表示不限制训练时间（无穷大）
        time_limit_seconds = None
        rationale.append("训练时间: 无限制")
    else:
        user_time_seconds = max(user_time_budget_minutes, 0.5) * 60
        if data_size_label == "small":
            # 小数据集不需要太长，但保证 AutoGluon 能完成基础训练
            time_limit_seconds = int(min(user_time_seconds, max(60, user_time_seconds)))
        elif data_size_label == "medium":
            time_limit_seconds = int(min(user_time_seconds, max(120, user_time_seconds)))
        else:
            time_limit_seconds = int(user_time_seconds)
        # 至少给 30 秒，避免 fit 失败
        time_limit_seconds = max(time_limit_seconds, 30)
        rationale.append(f"训练时间限制: {time_limit_seconds}s")

    # 4. 模型复杂度（stacking / bagging）
    if user_max_models is not None:
        max_models = user_max_models
        rationale.append(f"使用用户指定 max_models={max_models}")
    elif data_size_label == "small":
        max_models = 10
        rationale.append("小数据集减少模型数量，降低过拟合风险")
    elif data_size_label == "large":
        max_models = 50
        rationale.append("大数据集使用更多模型提升泛化")
    else:
        max_models = 25

    if data_size_label == "small" or max_models <= 10:
        auto_stack = False
        num_bag_folds = 0
        num_stack_levels = 0
        rationale.append("小数据集关闭 stacking/bagging，防止过拟合")
    elif data_size_label == "large" and max_models >= 30:
        auto_stack = True
        num_bag_folds = 3
        num_stack_levels = 1
        rationale.append("大数据集启用轻量 stacking")
    else:
        auto_stack = False
        num_bag_folds = 0
        num_stack_levels = 0
        rationale.append("中等规模使用单阶段模型")

    # 5. 评估指标
    primary_metric = user_primary_metric
    if primary_metric is None:
        primary_metric = _auto_primary_metric(task_type, target_info)
        rationale.append(f"自动选择主评估指标: {primary_metric}")

    # 6. 样本权重（类别不平衡）
    imbalance_ratio = _imbalance_ratio(target_info)
    use_sample_weight = (
        task_type in ("binary_classification", "multiclass_classification")
        and imbalance_ratio is not None
        and imbalance_ratio > 1.5
    )
    if use_sample_weight:
        rationale.append(f"检测到类别不平衡 (ratio={imbalance_ratio:.2f})，启用 balanced sample_weight")
        sample_weight_strategy = "balanced"
    else:
        sample_weight_strategy = "none"

    # 7. 验证策略
    validation_strategy = _auto_validation_strategy(n_samples, target_info, task_type)
    if validation_strategy["name"] == "holdout":
        # 小数据集给更多验证样本，大数据集用默认 0.1
        if data_size_label == "small":
            validation_strategy["holdout_frac"] = 0.2
        elif data_size_label == "medium":
            validation_strategy["holdout_frac"] = 0.15
        else:
            validation_strategy["holdout_frac"] = 0.1
    if validation_strategy["name"] == "cv":
        rationale.append(
            f"验证策略: {validation_strategy['name']} "
            f"(cv_type={validation_strategy.get('cv_type')}, "
            f"n_folds={validation_strategy.get('n_folds')})"
        )
    else:
        rationale.append(
            f"验证策略: {validation_strategy['name']} "
            f"(holdout_frac={validation_strategy.get('holdout_frac', '-')})"
        )

    # 8. 预处理策略
    missing_rate_values = list(missing_rates.values()) if missing_rates else []
    high_missing = any(r > 0.3 for r in missing_rate_values)
    has_datetime = any(t == "datetime" for t in field_types.values())
    has_text = any(t == "text" for t in field_types.values())

    preprocessing = {
        "numeric_impute": "median",
        "categorical_impute": "mode",
        "log_transform": True,
        "log_skew_threshold": 1.0,
        "outlier_strategy": "iqr",
        "scaler_type": "auto",
        "rare_category_threshold": max(5, int(n_samples * 0.001)),
        "datetime_cyclical": has_datetime,
        "text_embeddings": False,  # 默认关闭，避免依赖和内存问题；可手动开启
        "high_cardinality_threshold": 50,
        "drop_high_missing": high_missing,
        "datetime_features": has_datetime,
        "text_features": has_text,
        # 缺失值分级策略
        "missing_row_drop_threshold": 0.05,   # <5% 删行（仅在训练时生效）
        "missing_impute_threshold": 0.30,     # 5-30% 填充
        "missing_drop_threshold": 0.50,       # >50% 丢弃或缺失指示列
        "missing_indicator": True,
        # 类别编码按基数选择
        "one_hot_threshold": 10,
        "target_encoding_threshold": 50,
        # 条件降维
        "correlation_threshold": 0.95,
        "low_variance_threshold": 0.0,
        "pca_variance_ratio": 0.95,
        "enable_pca": n_features > 100 or memory_mb > 1024,
    }
    if high_missing:
        rationale.append("存在高缺失率特征，训练后将给出缺失率报告")
    if has_datetime:
        rationale.append("检测到时间特征，将提取年月日、小时及周期特征")
    if has_text:
        rationale.append("检测到文本特征，默认交给 AutoGluon 自动处理；可手动开启 Embedding")

    return TrainingStrategy(
        data_size_label=data_size_label,
        preset=preset,
        time_limit_seconds=time_limit_seconds,
        max_models=max_models,
        auto_stack=auto_stack,
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        primary_metric=primary_metric,
        use_sample_weight=use_sample_weight,
        sample_weight_strategy=sample_weight_strategy,
        validation_strategy=validation_strategy,
        preprocessing=preprocessing,
        rationale=rationale,
    )


def _auto_primary_metric(task_type: str, target_info: Dict[str, Any]) -> str:
    """根据任务类型和目标分布自动选择指标。"""
    if task_type == "binary_classification":
        class_distribution = target_info.get("class_distribution", {})
        if class_distribution:
            counts = list(class_distribution.values())
            if len(counts) >= 2:
                ratio = max(counts) / max(min(counts), 1)
                if ratio > 3:
                    return "auc_pr"
        return "f1"
    elif task_type == "multiclass_classification":
        return "log_loss"
    elif task_type == "regression":
        return "root_mean_squared_error"
    return "accuracy"


def _imbalance_ratio(target_info: Dict[str, Any]) -> Optional[float]:
    """计算类别不平衡比例。"""
    class_distribution = target_info.get("class_distribution")
    if not class_distribution:
        return None
    counts = [v for v in class_distribution.values() if isinstance(v, (int, float)) and v > 0]
    if len(counts) < 2:
        return None
    return max(counts) / min(counts)


def _auto_validation_strategy(
    n_samples: int, target_info: Dict[str, Any], task_type: str
) -> Dict[str, Any]:
    """根据样本量选择验证策略。"""
    class_distribution = target_info.get("class_distribution", {})
    min_class_count = min(
        (v for v in class_distribution.values() if isinstance(v, (int, float)) and v > 0),
        default=n_samples,
    )

    if n_samples < 500:
        # 小样本用交叉验证
        if task_type in ("binary_classification", "multiclass_classification"):
            n_folds = min(5, int(min_class_count))
            n_folds = max(2, n_folds)
            cv_type = "stratified"
        else:
            n_folds = 5
            cv_type = "kfold"
        return {"name": "cv", "n_folds": n_folds, "cv_type": cv_type}

    return {"name": "holdout", "test_size": 0.2}
