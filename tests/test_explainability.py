"""可解释性服务测试。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.explainability import compute_permutation_importance


class _FakePredictor:
    """伪预测器：feature_1 重要，feature_2 无用。"""

    def __init__(self):
        self.eval_metric = type("M", (), {"name": "accuracy"})()

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X["f1"] > 0).astype(int)

    def evaluate(self, X):
        y_pred = self.predict(X)
        y_true = X["target"]
        acc = (y_pred == y_true).mean()
        return {"accuracy": float(acc)}


def test_permutation_importance_detects_important_feature(tmp_path):
    """Permutation Importance 应识别出真正重要的特征。"""
    predictor = _FakePredictor()
    data = pd.DataFrame({
        "f1": [1, -1, 1, -1, 1, -1, 1, -1],
        "f2": [0, 0, 1, 1, 2, 2, 3, 3],
        "target": [1, 0, 1, 0, 1, 0, 1, 0],
    })

    info = compute_permutation_importance(predictor, data, "target", tmp_path, n_repeats=3)

    assert "path" in info
    assert Path(info["path"]).exists()
    top = info["top_features"]
    assert "f1" in top
    # f1 的重要性应明显高于 f2
    assert top["f1"] > top["f2"]
