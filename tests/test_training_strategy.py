"""训练策略服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.training_strategy import build_strategy


def _metadata(n_samples: int, n_features: int, **kwargs):
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "memory_mb": 1.0,
        "field_types": {},
        "missing_rates": {},
        "target_info": kwargs.get("target_info"),
    }


def test_small_classification_uses_best_quality_with_stacking():
    metadata = _metadata(150, 5)
    strategy = build_strategy(metadata, "multiclass_classification", user_time_budget_minutes=1.0)
    assert strategy.data_size_label == "small"
    assert strategy.preset == "best_quality"
    assert strategy.auto_stack is True
    assert strategy.num_stack_levels == 1
    assert strategy.num_bag_folds == 3
    assert strategy.max_models == 25
    assert strategy.primary_metric == "log_loss"


def test_large_data_uses_medium_quality():
    metadata = _metadata(200_000, 20)
    strategy = build_strategy(metadata, "binary_classification", user_time_budget_minutes=10.0)
    assert strategy.data_size_label == "large"
    assert strategy.preset == "medium_quality"


def test_user_preset_overrides():
    metadata = _metadata(150, 5)
    strategy = build_strategy(
        metadata, "multiclass_classification", user_preset="best_quality"
    )
    assert strategy.preset == "best_quality"


def test_user_max_models_overrides():
    metadata = _metadata(150, 5)
    strategy = build_strategy(
        metadata, "multiclass_classification", user_max_models=3
    )
    assert strategy.max_models == 3


def test_imbalance_triggers_sample_weight():
    metadata = _metadata(
        1000,
        5,
        target_info={"class_distribution": {"a": 900, "b": 100}},
    )
    strategy = build_strategy(metadata, "binary_classification")
    assert strategy.use_sample_weight is True
    assert strategy.sample_weight_strategy == "balanced"


def test_validation_strategy_for_small_data():
    metadata = _metadata(100, 5)
    strategy = build_strategy(metadata, "multiclass_classification")
    assert strategy.validation_strategy["name"] == "cv"


def test_validation_strategy_for_medium_data():
    metadata = _metadata(5000, 5)
    strategy = build_strategy(metadata, "multiclass_classification")
    assert strategy.validation_strategy["name"] == "holdout"
    assert "holdout_frac" in strategy.validation_strategy
