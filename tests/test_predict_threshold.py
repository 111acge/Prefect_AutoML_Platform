"""预测阈值应用测试。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from routers.runs import _load_optimal_threshold, _predict_with_threshold


class _FakeBinaryPredictor:
    """伪二分类预测器。"""

    problem_type = "binary"
    class_labels = [0, 1]

    def predict(self, df):
        return pd.Series([0, 1, 0])

    def predict_proba(self, df):
        return pd.DataFrame({0: [0.6, 0.3, 0.4], 1: [0.4, 0.7, 0.6]})


def test_load_optimal_threshold(tmp_path):
    """从 metrics.json 中读取最优阈值。"""
    metrics = {
        "threshold": {
            "default_threshold": 0.5,
            "optimal_threshold": 0.72,
            "best_f1": 0.88,
        }
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(__import__("json").dumps(metrics), encoding="utf-8")

    assert _load_optimal_threshold(tmp_path) == pytest.approx(0.72)


def test_load_optimal_threshold_missing():
    """没有 metrics.json 时返回 None。"""
    assert _load_optimal_threshold(Path("/nonexistent")) is None


def test_predict_with_threshold_binary():
    """使用自定义阈值覆盖默认 0.5 预测。"""
    predictor = _FakeBinaryPredictor()
    df = pd.DataFrame({"x": [1, 2, 3]})

    # 阈值 0.5：prob[1] >= 0.5 -> [0, 1, 1]
    preds = _predict_with_threshold(predictor, df, 0.5)
    assert preds.tolist() == [0, 1, 1]

    # 阈值 0.65：prob[1] >= 0.65 -> [0, 1, 0]
    preds = _predict_with_threshold(predictor, df, 0.65)
    assert preds.tolist() == [0, 1, 0]


def test_predict_with_threshold_none():
    """阈值为 None 时回退到 predictor.predict。"""
    predictor = _FakeBinaryPredictor()
    df = pd.DataFrame({"x": [1, 2, 3]})
    preds = _predict_with_threshold(predictor, df, None)
    assert preds.tolist() == [0, 1, 0]
