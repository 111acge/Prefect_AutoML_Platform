"""自动优化阈值测试。

验证系统在高类别数/大数据集场景下，无需人工配置即可自动触发
采样、跳过、模型复杂度限制等优化策略。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.fixture
def high_cardinality_data():
    """60 类多分类数据，共 300 行，确保能触发采样。"""
    return pd.DataFrame({
        "f1": list(range(300)),
        "target": [i % 60 for i in range(300)],
    })


def test_train_eval_auto_samples_for_high_cardinality(high_cardinality_data, tmp_path):
    """高类别数场景下训练集评估应自动采样。"""
    from prefect_flows.automl_flow import evaluate_model_task

    test_data = high_cardinality_data.sample(n=20, random_state=42)
    train_data = high_cardinality_data

    mock_predictor = MagicMock()
    mock_predictor.problem_type = "multiclass"
    mock_predictor.class_labels = list(range(60))
    mock_predictor.evaluate.return_value = {"accuracy": 0.9}
    mock_predictor.predict.return_value = pd.Series([0] * len(test_data))
    mock_predictor.eval_metric.name = "accuracy"
    mock_predictor.leaderboard.return_value = pd.DataFrame({"model": ["GBM"], "score_val": [0.9]})

    evaluate_records = []

    def _evaluate_side_effect(df):
        evaluate_records.append(len(df))
        return {"accuracy": 0.9}

    mock_predictor.evaluate.side_effect = _evaluate_side_effect

    output_dir = tmp_path / "mock_run"
    output_dir.mkdir()

    val_data = high_cardinality_data.sample(n=20, random_state=1)

    with patch("autogluon.tabular.TabularPredictor") as MockPredictor:
        MockPredictor.load.return_value = mock_predictor
        metrics = evaluate_model_task.fn(
            test_data=test_data,
            val_data=val_data,
            train_data=train_data,
            target_column="target",
            output_dir=str(output_dir),
        )

    assert "train" in metrics
    assert "val" in metrics
    # 训练集被自动采样至 200 行，不应等于全量 300
    assert 200 in evaluate_records
    assert 300 not in evaluate_records


def test_training_strategy_limits_gbm_for_high_cardinality():
    """高类别数多分类策略应限制 GBM/XGB 复杂度。"""
    from services.training_strategy import build_strategy

    metadata = {
        "n_samples": 800,
        "n_features": 5,
        "memory_mb": 10.0,
        "target_info": {
            "type": "categorical",
            "unique_values": 60,
            "class_distribution": {i: 10 for i in range(60)},
        },
        "field_types": {"f1": "numeric"},
        "missing_rates": {},
        "high_cardinality_columns": [],
    }

    strategy = build_strategy(metadata, "multiclass_classification", user_time_budget_minutes=10.0)

    assert strategy.hyperparameters is not None
    assert "GBM" in strategy.hyperparameters
    assert strategy.hyperparameters["GBM"].get("n_estimators") == 1000
    assert strategy.hyperparameters["XGB"].get("n_estimators") == 500
