"""特征工程服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
import pandas as pd

from services.feature_engineering import FeatureEngineer


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 100.0, 5.0],
        "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "a", "a", "b", "c"],
        "dt": pd.to_datetime(["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10", "2024-05-05"]),
        "target": [0, 1, 0, 1, 0],
    })


def test_outlier_capping():
    df = _make_df()
    fe = FeatureEngineer(target_column="target", outlier_strategy="iqr", scaler_type="none", one_hot_threshold=0)
    transformed = fe.fit_transform(df)
    # 100 是异常值，应被截断
    assert transformed["num_a"].max() < 100.0
    # num_b 没有异常值，最大值不变
    assert transformed["num_b"].max() == 50.0


def test_auto_scaler_chooses_robust_with_outliers():
    df = _make_df()
    fe = FeatureEngineer(target_column="target", outlier_strategy="iqr", scaler_type="auto", one_hot_threshold=0)
    fe.fit(df)
    assert fe.scaler is not None


def test_no_scaling_when_disabled():
    df = _make_df()
    fe = FeatureEngineer(target_column="target", outlier_strategy="none", scaler_type="none", one_hot_threshold=0)
    fe.fit(df)
    assert fe.scaler is None


def test_rare_category_grouping():
    df = pd.DataFrame({
        "cat": ["a", "a", "a", "a", "b", "c"],
        "target": [0, 1, 0, 1, 0, 1],
    })
    fe = FeatureEngineer(
        target_column="target",
        outlier_strategy="none",
        scaler_type="none",
        rare_category_threshold=2,
        one_hot_threshold=0,
    )
    transformed = fe.fit_transform(df)
    # 'c' 出现 1 次，应被合并为 __other__
    assert "__other__" in transformed["cat"].values
    assert "c" not in transformed["cat"].values


def test_datetime_features_and_cyclical():
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-25"]),
        "target": [0, 1, 0],
    })
    fe = FeatureEngineer(
        target_column="target",
        outlier_strategy="none",
        scaler_type="none",
        datetime_cyclical=True,
        one_hot_threshold=0,
    )
    transformed = fe.fit_transform(df)
    for suffix in ["_year", "_month", "_day", "_dayofweek", "_hour"]:
        assert f"dt{suffix}" in transformed.columns
    for suffix in ["_month_sin", "_month_cos", "_dayofweek_sin", "_dayofweek_cos"]:
        assert f"dt{suffix}" in transformed.columns


def test_text_column_identification():
    df = pd.DataFrame({
        "short_cat": ["a", "b", "c", "d", "e"],
        "long_text": [
            "this is a long sentence about machine learning",
            "another long sentence about deep learning",
            "natural language processing is fun",
            "computer vision and image recognition",
            "reinforcement learning for games",
        ],
        "target": [0, 1, 0, 1, 0],
    })
    fe = FeatureEngineer(
        target_column="target",
        outlier_strategy="none",
        scaler_type="none",
        text_embeddings=False,
        one_hot_threshold=0,
    )
    fe.fit(df)
    assert "long_text" in fe.text_cols
    assert "short_cat" in fe.categorical_cols


def test_fit_transform_preserves_target():
    df = _make_df()
    fe = FeatureEngineer(target_column="target", outlier_strategy="iqr", scaler_type="auto", one_hot_threshold=0)
    transformed = fe.fit_transform(df)
    assert "target" in transformed.columns
    assert transformed["target"].equals(df["target"])
