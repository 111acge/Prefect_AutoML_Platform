"""交叉验证策略服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
import numpy as np

from services.cv_service import (
    get_cv_splitter,
    cross_validate_pipeline,
)


def test_classification_gets_stratified_kfold():
    splitter = get_cv_splitter("binary_classification", 100, n_folds=5)
    assert splitter.__class__.__name__ == "StratifiedKFold"


def test_regression_gets_kfold():
    splitter = get_cv_splitter("regression", 100, n_folds=5)
    assert splitter.__class__.__name__ == "KFold"


def test_timeseries_gets_time_series_split():
    splitter = get_cv_splitter("regression", 100, n_folds=5, cv_type="timeseries")
    assert splitter.__class__.__name__ == "TimeSeriesSplit"


def test_group_gets_group_kfold():
    splitter = get_cv_splitter("binary_classification", 100, n_folds=5, cv_type="group")
    assert splitter.__class__.__name__ == "GroupKFold"


def test_min_class_count_one_falls_back_to_kfold():
    """最小类只有 1 个样本时，StratifiedKFold 会崩溃，应回退到 KFold。"""
    splitter = get_cv_splitter(
        "binary_classification", 100, n_folds=5, min_class_count=1
    )
    assert splitter.__class__.__name__ == "KFold"


def test_n_folds_capped_by_min_class_count():
    """StratifiedKFold 的折数不能超过最小类样本数。"""
    splitter = get_cv_splitter(
        "binary_classification", 100, n_folds=10, min_class_count=3
    )
    assert splitter.__class__.__name__ == "StratifiedKFold"
    assert splitter.n_splits == 3


def test_cross_validate_binary_classification():
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "target": [0, 1] * 50,
    })
    strategy = {
        "primary_metric": "f1",
        "validation_strategy": {"name": "cv", "n_folds": 3, "cv_type": "stratified"},
        "preprocessing": {},
    }
    result = cross_validate_pipeline(df, "target", "binary_classification", strategy)

    assert "cv_scores" in result
    assert len(result["cv_scores"]) == 3
    assert "cv_mean" in result
    assert "cv_std" in result
    assert result["cv_type"] == "StratifiedKFold"


def test_cross_validate_regression():
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "target": np.random.randn(100),
    })
    strategy = {
        "primary_metric": "root_mean_squared_error",
        "validation_strategy": {"name": "cv", "n_folds": 3, "cv_type": "kfold"},
        "preprocessing": {},
    }
    result = cross_validate_pipeline(df, "target", "regression", strategy)

    assert "cv_scores" in result
    assert len(result["cv_scores"]) == 3
    assert result["cv_type"] == "KFold"


def test_cross_validate_missing_target_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    strategy = {"validation_strategy": {"name": "cv", "n_folds": 2}, "preprocessing": {}}
    result = cross_validate_pipeline(df, "target", "binary_classification", strategy)
    assert result == {}
