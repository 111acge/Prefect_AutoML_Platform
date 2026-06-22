"""AutoML 服务单元测试。"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.automl import AutoMLService


class _FakePredictor:
    """用于测试的伪 Predictor。"""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.problem_type = kwargs.get("problem_type", "binary")
        self.class_labels = [0, 1]
        self._best = "WeightedEnsemble"
        self._models = ["WeightedEnsemble", "LightGBM", "XGBoost"]

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def save(self):
        pass

    def leaderboard(self, silent=True):
        scores = [0.85 - i * 0.01 for i in range(len(self._models))]
        return pd.DataFrame(
            [{"model": m, "score_val": s} for m, s in zip(self._models, scores)]
        )

    def set_model_best(self, model):
        self._best = model

    def delete_models(self, models_to_keep=None, models_to_delete=None, models=None, dry_run=False):
        if models_to_delete is not None:
            self._models = [m for m in self._models if m not in models_to_delete]
        elif models is not None:
            self._models = [m for m in self._models if m not in models]
        elif models_to_keep is not None:
            if isinstance(models_to_keep, str):
                self._models = [m for m in self._models if m == models_to_keep]
            else:
                self._models = [m for m in self._models if m in models_to_keep]

    def model_names(self):
        return self._models

    def predict_proba(self, data):
        return pd.DataFrame({0: [0.2, 0.7], 1: [0.8, 0.3]})


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "run"


def test_ensemble_fallback_when_improvement_small(output_dir, monkeypatch):
    """集成提升不足 2% 时应回退到最优单模型。"""
    service = AutoMLService(output_dir)
    predictor = _FakePredictor()
    leaderboard = predictor.leaderboard()

    result = service._apply_ensemble_fallback(predictor, leaderboard)

    assert result["fallback_applied"] is True
    assert result["best_single_model"] == "LightGBM"
    assert predictor._best == "LightGBM"
    assert "WeightedEnsemble" not in predictor.model_names()


def test_ensemble_fallback_not_triggered_when_single_is_top(output_dir):
    """最优模型本身不是集成时不触发回退。"""
    service = AutoMLService(output_dir)
    predictor = _FakePredictor()
    # 手动把集成去掉，让单模型排第一
    predictor._models = ["LightGBM", "XGBoost"]
    leaderboard = predictor.leaderboard()

    result = service._apply_ensemble_fallback(predictor, leaderboard)

    assert result["fallback_applied"] is False
    assert result["ensemble_used"] is False


def test_validation_strategy_cv_passes_num_bag_folds(output_dir, monkeypatch):
    """策略为 cv 时，fit 应收到 num_bag_folds。"""
    captured = {}

    class _CapturePredictor(_FakePredictor):
        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

    def _fake_predictor(*args, **kwargs):
        return _CapturePredictor(**kwargs)

    monkeypatch.setattr("services.automl.TabularPredictor", _fake_predictor)
    monkeypatch.setattr("services.automl.compute_shap_values", lambda *a, **k: None)

    service = AutoMLService(output_dir)
    train_data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "b": ["x", "y"] * 5,
        "target": [0, 1] * 5,
    })
    strategy = {
        "preset": "medium_quality",
        "time_limit_seconds": 30,
        "max_models": 5,
        "primary_metric": "f1",
        "validation_strategy": {"name": "cv", "n_folds": 4},
    }

    service.train(
        train_data=train_data,
        target_column="target",
        task_type="binary_classification",
        time_limit=30,
        preset="medium_quality",
        strategy=strategy,
    )

    assert captured["fit_kwargs"]["num_bag_folds"] == 4
    assert "holdout_frac" not in captured["fit_kwargs"]


def test_validation_strategy_holdout_passes_holdout_frac(output_dir, monkeypatch):
    """策略为 holdout 时，fit 应收到 holdout_frac。"""
    captured = {}

    class _CapturePredictor(_FakePredictor):
        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

    def _fake_predictor(*args, **kwargs):
        return _CapturePredictor(**kwargs)

    monkeypatch.setattr("services.automl.TabularPredictor", _fake_predictor)
    monkeypatch.setattr("services.automl.compute_shap_values", lambda *a, **k: None)

    service = AutoMLService(output_dir)
    train_data = pd.DataFrame({
        "a": list(range(20)),
        "b": ["x", "y"] * 10,
        "target": [0, 1] * 10,
    })
    strategy = {
        "validation_strategy": {"name": "holdout", "holdout_frac": 0.2},
    }

    service.train(
        train_data=train_data,
        target_column="target",
        task_type="binary_classification",
        time_limit=30,
        strategy=strategy,
    )

    assert captured["fit_kwargs"]["holdout_frac"] == 0.2


def test_validation_strategy_cv_fallback_when_min_class_count_is_one(
    output_dir, monkeypatch
):
    """最小类只有 1 个样本时，CV bagging 应被禁用并回退到 holdout。"""
    captured = {}

    class _CapturePredictor(_FakePredictor):
        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

    def _fake_predictor(*args, **kwargs):
        return _CapturePredictor(**kwargs)

    monkeypatch.setattr("services.automl.TabularPredictor", _fake_predictor)
    monkeypatch.setattr("services.automl.compute_shap_values", lambda *a, **k: None)

    service = AutoMLService(output_dir)
    train_data = pd.DataFrame({
        "a": list(range(10)),
        "b": ["x", "y"] * 5,
        # 二分类，正类只有 1 个样本
        "target": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    strategy = {
        "validation_strategy": {"name": "cv", "n_folds": 5},
    }

    service.train(
        train_data=train_data,
        target_column="target",
        task_type="binary_classification",
        time_limit=30,
        strategy=strategy,
    )

    assert "num_bag_folds" not in captured["fit_kwargs"]
    assert captured["fit_kwargs"]["holdout_frac"] == 0.2


def test_validation_strategy_cv_folds_capped_by_min_class_count(
    output_dir, monkeypatch
):
    """CV 折数不应超过最小类样本数。"""
    captured = {}

    class _CapturePredictor(_FakePredictor):
        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

    def _fake_predictor(*args, **kwargs):
        return _CapturePredictor(**kwargs)

    monkeypatch.setattr("services.automl.TabularPredictor", _fake_predictor)
    monkeypatch.setattr("services.automl.compute_shap_values", lambda *a, **k: None)

    service = AutoMLService(output_dir)
    train_data = pd.DataFrame({
        "a": list(range(12)),
        "b": ["x", "y"] * 6,
        # 二分类，正类只有 3 个样本
        "target": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    strategy = {
        "validation_strategy": {"name": "cv", "n_folds": 10},
    }

    service.train(
        train_data=train_data,
        target_column="target",
        task_type="binary_classification",
        time_limit=30,
        strategy=strategy,
    )

    assert captured["fit_kwargs"]["num_bag_folds"] == 3
    assert "holdout_frac" not in captured["fit_kwargs"]
