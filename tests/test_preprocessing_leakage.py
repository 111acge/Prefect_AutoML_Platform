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

    # 当前 Target Encoding 使用平滑全局均值（smoothing=10），
    # 训练集全局均值 global_mean = (1+2+10+11+5)/5 = 5.8
    # a: (1.5*2 + 5.8*10) / (2+10) = 5.083333333333333
    # b: (10.5*2 + 5.8*10) / (2+10) = 6.583333333333333
    # c: (5.0*1 + 5.8*10) / (1+10) = 5.7272727272727275
    # d 为未见类别，应使用全局均值 5.8
    expected = pd.Series(
        [5.083333333333333, 6.583333333333333, 5.7272727272727275, 5.8],
        name="cat_te",
    )
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


def test_transform_without_target_does_not_drop_duplicate_rows():
    """预测输入不含目标列时，不应按特征去重导致行数变化。"""
    train_df = pd.DataFrame({
        "cat": ["a", "a", "b", "b"],
        "num": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })
    preprocessor = DataPreprocessor(
        target_column="target",
        strategy={"preprocessing": {}},
        cleaning_rules={},
    )
    preprocessor.fit(train_df)

    # 测试集：特征行与训练集重复但目标未知
    test_df = pd.DataFrame({
        "cat": ["a", "a", "b"],
        "num": [1.0, 2.0, 3.0],
    })
    transformed = preprocessor.transform(test_df)
    assert len(transformed) == 3


def test_split_data_oversamples_rare_classes_by_default():
    """默认策略会对样本数 < 2 的稀有类别进行过采样，保留全部类别并完成 stratify。"""
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

    # 所有类别都应保留，且训练/测试集都包含所有类别
    combined = pd.concat([train_df, test_df])
    assert set(combined["target"].unique()) == {"a", "b", "c", "d", "e"}
    for cls in {"a", "b", "c", "d", "e"}:
        assert cls in train_df["target"].values
        assert cls in test_df["target"].values


def test_split_data_can_drop_rare_classes():
    """显式指定 rare_class_strategy='drop' 时保留旧行为：过滤稀有类别。"""
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
    result = split_data(
        df,
        target_column="target",
        task_type="multiclass_classification",
        rare_class_strategy="drop",
    )
    train_df = result["train"]
    test_df = result["test"]

    combined = pd.concat([train_df, test_df])
    assert "c" not in combined["target"].values
    assert "d" not in combined["target"].values
    assert set(combined["target"].unique()) == {"a", "b", "e"}
    assert set(train_df["target"].unique()) == {"a", "b", "e"}
    assert set(test_df["target"].unique()).issubset({"a", "b", "e"})


def test_split_data_handles_binary_singleton_class():
    """二分类任务中某一类只有 1 个样本时，仍应完成划分并保证两类都出现。"""
    df = pd.DataFrame({
        "feat": list(range(10)),
        "target": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    result = split_data(
        df,
        target_column="target",
        task_type="binary_classification",
    )
    train_df = result["train"]
    test_df = result["test"]

    assert {0, 1}.issubset(set(train_df["target"].unique()))
    assert {0, 1}.issubset(set(test_df["target"].unique()))
