# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""类别编码测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd

from services.feature_engineering import FeatureEngineer


def test_one_hot_encoding_low_cardinality():
    df = pd.DataFrame({
        "cat": ["a", "b", "a", "c"],
        "target": [0, 1, 0, 1],
    })
    fe = FeatureEngineer(
        target_column="target",
        outlier_strategy="none",
        scaler_type="none",
        one_hot_threshold=10,
        target_encoding_threshold=100,
    )
    transformed = fe.fit_transform(df)
    assert "cat_a" in transformed.columns
    assert "cat_b" in transformed.columns
    assert "cat_c" in transformed.columns
    assert "cat" not in transformed.columns


def test_target_encoding_high_cardinality():
    df = pd.DataFrame({
        "cat": [f"v{i}" for i in range(60)],
        "target": list(range(60)),
    })
    fe = FeatureEngineer(
        target_column="target",
        outlier_strategy="none",
        scaler_type="none",
        one_hot_threshold=5,
        target_encoding_threshold=50,
    )
    transformed = fe.fit_transform(df)
    assert "cat_te" in transformed.columns
    assert "cat" not in transformed.columns


def test_missing_indicator():
    df = pd.DataFrame({
        "num": [1.0, None, 3.0, 4.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    fe = FeatureEngineer(
        target_column="target",
        outlier_strategy="none",
        scaler_type="none",
        one_hot_threshold=0,
        missing_indicator=True,
        missing_indicator_threshold=0.05,
    )
    transformed = fe.fit_transform(df)
    assert "num_missing" in transformed.columns
    assert transformed["num_missing"].sum() == 1
