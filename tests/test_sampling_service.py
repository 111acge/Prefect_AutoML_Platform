# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""条件采样服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
import pytest

from services.sampling_service import (
    build_sampling_strategy,
    apply_sampling,
    compute_sample_weight_series,
)


def _make_df(imbalanced: bool = False):
    if imbalanced:
        return pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # ratio 7:3 -> 2.33
        })
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [0.0, 1.0, 0.0, 1.0, 0.0],
        "target": [0, 1, 0, 1, 0],  # ratio 3:2 -> 1.5
    })


def test_balanced_data_no_sampling():
    df = _make_df(imbalanced=False)
    strategy = build_sampling_strategy(df, "target", "binary_classification")
    assert strategy.method == "none"
    sampled, weight = apply_sampling(df, "target", strategy)
    assert sampled.shape == df.shape
    assert weight is None


def test_imbalanced_class_weight():
    df = _make_df(imbalanced=True)
    strategy = build_sampling_strategy(df, "target", "binary_classification")
    # ratio 7:3 -> 2.33, method should be class_weight
    assert strategy.method == "class_weight"
    sampled, weight = apply_sampling(df, "target", strategy)
    assert sampled.shape[0] == df.shape[0]
    assert sampled.shape[1] == df.shape[1] + 1
    assert weight is None
    assert "_sample_weight" in sampled.columns


def test_smote_numeric():
    df = pd.DataFrame({
        "a": list(range(20)),
        "b": list(range(20, 40)),
        "target": [0] * 17 + [1] * 3,  # ratio ~5.67
    })
    strategy = build_sampling_strategy(df, "target", "binary_classification")
    assert strategy.method == "smote"
    sampled, weight = apply_sampling(df, "target", strategy)
    assert weight is None
    assert sampled.shape[0] > df.shape[0]
    assert set(sampled.columns) == set(df.columns)


def test_categorical_fallback_to_random_over():
    df = pd.DataFrame({
        "a": list(range(20)),
        "c": ["x"] * 10 + ["y"] * 10,
        "target": [0] * 17 + [1] * 3,
    })
    strategy = build_sampling_strategy(df, "target", "binary_classification")
    # mixed types with missing? no missing -> smotenc selected, but if fails fallback
    assert strategy.method in ("smotenc", "random_over")
    sampled, weight = apply_sampling(df, "target", strategy)
    assert weight is None
    assert sampled.shape[0] > df.shape[0]


def test_compute_sample_weight():
    y = pd.Series([0, 0, 0, 1, 1])
    w = compute_sample_weight_series(y)
    assert len(w) == 5
    # 每个类别的总权重相等
    assert w[y == 0].sum() == w[y == 1].sum()


def test_regression_no_sampling():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "target": [1.5, 2.5, 3.5],
    })
    strategy = build_sampling_strategy(df, "target", "regression")
    assert strategy.method == "none"
