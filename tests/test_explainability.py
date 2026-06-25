# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""可解释性服务测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.explainability import (
    compute_permutation_importance,
    compute_shap_values,
    _encode_for_shap,
    _decode_for_prediction,
)


class _FakePredictor:
    """伪预测器：feature_1 重要，feature_2 无用。"""

    problem_type = "binary"
    class_labels = [0, 1]

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


def test_shap_encode_decode_categorical_strings():
    """SHAP 编码应能正确处理类别字符串，并可还原供预测器使用。"""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "cat": ["a", "b", "a"],
    })
    encoded, encoders = _encode_for_shap(df)

    # 数值列不变
    assert encoded["num"].tolist() == [1.0, 2.0, 3.0]
    # 类别列被编码为浮点数
    assert encoded["cat"].dtype.kind == "f"
    assert "cat" in encoders

    # 解码后能还原为原始字符串
    decoded = _decode_for_prediction(encoded, encoders)
    assert decoded["cat"].tolist() == ["a", "b", "a"]


def test_shap_encode_decode_with_shap_perturbation():
    """模拟 SHAP 扰动产生的非整数值，解码应能还原到最近的有效类别。"""
    df = pd.DataFrame({
        "cat": ["a", "b", "c"],
    })
    encoded, encoders = _encode_for_shap(df)

    # 模拟 SHAP 扰动后的值
    perturbed = encoded.copy()
    perturbed["cat"] = [0.2, 0.8, 2.3]
    decoded = _decode_for_prediction(perturbed, encoders)
    assert decoded["cat"].tolist() == ["a", "b", "c"]


def test_resolve_permutation_importance_params_auto_downgrades():
    """高类别数/大数据集下 Permutation Importance 参数应自动降级。"""
    from services.explainability import _resolve_permutation_importance_params

    repeats, sample_size = _resolve_permutation_importance_params(
        n_repeats=None, sample_size=None, n_classes=60, n_rows=600
    )
    assert repeats == 2
    assert sample_size == 500

    # 普通二分类保持默认
    repeats, sample_size = _resolve_permutation_importance_params(
        n_repeats=None, sample_size=None, n_classes=2, n_rows=100
    )
    assert repeats == 5
    assert sample_size is None


def test_permutation_importance_auto_downgrades_for_high_cardinality(monkeypatch, tmp_path):
    """高类别数场景下 Permutation Importance 应自动降低 n_repeats 并采样。"""
    from config import settings
    # 确保使用自动默认值
    monkeypatch.setattr(settings, "permutation_importance_max_repeats", None)
    monkeypatch.setattr(settings, "permutation_importance_sample_size", None)

    class _HighCardPredictor(_FakePredictor):
        problem_type = "multiclass"
        class_labels = list(range(60))

    predictor = _HighCardPredictor()
    data = pd.DataFrame({
        "f1": list(range(600)),
        "f2": list(range(600)),
        "target": [i % 60 for i in range(600)],
    })
    evaluate_calls = []
    original_evaluate = predictor.evaluate

    def _tracking_evaluate(df):
        evaluate_calls.append(len(df))
        return original_evaluate(df)

    predictor.evaluate = _tracking_evaluate

    info = compute_permutation_importance(predictor, data, "target", tmp_path)

    assert "path" in info
    # 自动：n_repeats=2, sample_size=500
    # baseline(1) + 2 features * 2 repeats = 5 次 evaluate
    assert len(evaluate_calls) == 5
    # 评估数据被采样到 500 行
    assert evaluate_calls[0] == 500


def test_permutation_importance_disabled_by_config(monkeypatch, tmp_path):
    """Permutation Importance 可通过配置禁用。"""
    from config import settings
    monkeypatch.setattr(settings, "permutation_importance_enabled", False)

    predictor = _FakePredictor()
    data = pd.DataFrame({
        "f1": [1, -1, 1, -1],
        "target": [1, 0, 1, 0],
    })
    info = compute_permutation_importance(predictor, data, "target", tmp_path)
    assert info == {}


def test_resolve_shap_sample_size_auto():
    """SHAP 采样大小应根据类别数自动决策。"""
    from services.explainability import _resolve_shap_sample_size

    assert _resolve_shap_sample_size(sample_size=None, n_classes=2) == 200
    assert _resolve_shap_sample_size(sample_size=None, n_classes=60) == 50
    assert _resolve_shap_sample_size(sample_size=None, n_classes=250) == 0
    # 显式参数优先
    assert _resolve_shap_sample_size(sample_size=10, n_classes=60) == 10


def test_shap_disabled_by_config(monkeypatch, tmp_path):
    """SHAP 可通过配置禁用。"""
    from config import settings
    monkeypatch.setattr(settings, "shap_enabled", False)

    class _MinimalPredictor:
        problem_type = "binary"
        class_labels = [0, 1]

    data = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "target": [0, 1, 0],
    })
    info = compute_shap_values(_MinimalPredictor(), data, "target", tmp_path)
    assert info == {}


def test_shap_auto_skips_for_extreme_cardinality(monkeypatch, tmp_path):
    """超高类别数场景下 SHAP 应自动跳过。"""
    from config import settings
    monkeypatch.setattr(settings, "shap_enabled", True)
    monkeypatch.setattr(settings, "shap_max_sample_size", None)

    class _ExtremeCardPredictor:
        problem_type = "multiclass"
        class_labels = list(range(250))

    data = pd.DataFrame({
        "f1": [float(i) for i in range(300)],
        "target": [i % 250 for i in range(300)],
    })
    info = compute_shap_values(_ExtremeCardPredictor(), data, "target", tmp_path)
    assert info == {}
