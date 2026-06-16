"""预处理防泄露测试。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.preprocessing_pipeline import DataPreprocessor
from services.preprocessing import split_data


def test_preprocessor_fit_on_train_only_for_target_encoding():
    """Target Encoding 必须在训练集上 fit，测试集只能使用训练集学到的均值。"""
    train_df = pd.DataFrame({
        "cat": ["a", "a", "b", "b", "c"],
        "target": [1.0, 2.0, 10.0, 11.0, 5.0],
    })
    test_df = pd.DataFrame({
        "cat": ["a", "b", "c", "d"],
        "target": [100.0, 200.0, 300.0, 400.0],
    })

    preprocessor = DataPreprocessor(
        target_column="target",
        strategy={
            "preprocessing": {
                "one_hot_threshold": 2,
                "target_encoding_threshold": 3,
                "rare_category_threshold": 0,
            }
        },
        cleaning_rules={},
    )
    preprocessor.fit(train_df)
    transformed_test = preprocessor.transform(test_df)

    # a 的训练集目标均值为 (1+2)/2 = 1.5
    # b 的训练集目标均值为 (10+11)/2 = 10.5
    # c 的训练集目标均值为 5.0
    # d 为未见类别，应使用全局均值 (1+2+10+11+5)/5 = 5.8
    expected = pd.Series([1.5, 10.5, 5.0, 5.8], name="cat_te")
    pd.testing.assert_series_equal(
        transformed_test["cat_te"].reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_preprocessor_transform_does_not_leak_test_distribution():
    """验证 transform 不会使用测试集的统计量（如缩放参数）。"""
    train_df = pd.DataFrame({
        "num": [0.0, 1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1, 0],
    })
    test_df = pd.DataFrame({
        "num": [100.0, 200.0, 300.0],
        "target": [0, 1, 0],
    })

    preprocessor = DataPreprocessor(
        target_column="target",
        strategy={"preprocessing": {"scaler_type": "standard"}},
        cleaning_rules={},
    )
    preprocessor.fit(train_df)
    transformed_train = preprocessor.transform(train_df)
    transformed_test = preprocessor.transform(test_df)

    # 训练集缩放应基于训练集均值 2.0 和标准差
    assert transformed_train["num"].mean() == pytest.approx(0.0, abs=1e-6)

    # 测试集的缩放参数来自训练集，因此测试集均值不一定为 0
    # 这里只需验证测试集没有被缩放到均值为 0（如果使用测试集统计量则会是 0）
    assert transformed_test["num"].mean() != pytest.approx(0.0, abs=1e-6)


def test_split_data_drops_rows_with_nan_target():
    """划分前必须丢弃目标列缺失的行，避免 stratify 报错。"""
    df = pd.DataFrame({
        "feat": list(range(11)),
        "target": ["a", "a", "a", "a", "b", "b", "b", "b", "a", "b", None],
    })
    result = split_data(df, target_column="target", task_type="multiclass_classification")
    train_df = result["train"]
    test_df = result["test"]

    assert len(train_df) + len(test_df) == 10
    assert not train_df["target"].isnull().any()
    assert not test_df["target"].isnull().any()
    assert set(train_df["target"].unique()).issubset({"a", "b"})


def test_split_data_drops_rare_classes():
    """划分前会过滤样本数 < 2 的稀有类别，避免 stratify 报错。"""
    targets = (
        ["a"] * 20
        + ["b"] * 20
        + ["c"] * 1
        + ["d"] * 1
        + ["e"] * 5
    )
    df = pd.DataFrame({
        "feat": list(range(len(targets))),
        "target": targets,
    })
    result = split_data(df, target_column="target", task_type="multiclass_classification")
    train_df = result["train"]
    test_df = result["test"]

    # c 和 d 只有 1 条，应被过滤
    combined = pd.concat([train_df, test_df])
    assert "c" not in combined["target"].values
    assert "d" not in combined["target"].values
    assert set(combined["target"].unique()) == {"a", "b", "e"}
    # stratify 应保证训练/测试集中都包含 a、b、e
    assert set(train_df["target"].unique()) == {"a", "b", "e"}
    assert set(test_df["target"].unique()).issubset({"a", "b", "e"})
