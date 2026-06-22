"""目标列/任务类型自动推断与 Schema 鲁棒性测试。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.data_service import (
    analyze_metadata,
    infer_target_column,
    infer_task_type,
)
from services.schema_service import infer_field_type, infer_schema


def test_infer_target_by_name():
    """列名命中目标关键词时应被优先推断。"""
    df = pd.DataFrame({
        "id": range(100),
        "feature_a": range(100),
        "is_fraud": [0, 1] * 50,
    })
    col, conf = infer_target_column(df)
    assert col == "is_fraud"
    assert conf >= 0.7


def test_infer_target_by_position_when_no_semantic():
    """无目标关键词时，应选择唯一值比例适中的非 ID 列。"""
    df = pd.DataFrame({
        "uuid": range(100),
        "feat": range(100),
        "label": [0, 1, 2] * 33 + [0],
    })
    # 重命名最后一列为无意义名称，但 label 列名仍命中关键词
    df_renamed = df.rename(columns={"label": "zzz"})
    col, conf = infer_target_column(df_renamed)
    # zzz 是唯一值适中的列，且不是 ID/序号
    assert col == "zzz"


def test_infer_target_skips_id_column():
    """ID/序号列不应被选为目标。"""
    df = pd.DataFrame({
        "customer_id": range(100),
        "score": range(100),
    })
    col, conf = infer_target_column(df)
    assert col == "score"


def test_infer_task_type_binary():
    """二值目标推断为二分类。"""
    s = pd.Series([0, 1, 0, 1])
    assert infer_task_type(s) == "binary_classification"


def test_infer_task_type_multiclass_categorical():
    """类别型目标推断为多分类。"""
    s = pd.Series(["a", "b", "c", "a"])
    assert infer_task_type(s) == "multiclass_classification"


def test_infer_task_type_regression_numeric():
    """连续数值目标推断为回归。"""
    s = pd.Series([float(i) + 0.1 * i for i in range(12)])
    assert infer_task_type(s) == "regression"


def test_infer_task_type_ordinal_as_multiclass():
    """3-10 个唯一值的整数目标按多分类处理。"""
    s = pd.Series([1, 2, 3, 4, 5])
    assert infer_task_type(s) == "multiclass_classification"


def test_analyze_metadata_auto_suggestions():
    """未提供 target/task 时，analyze_metadata 应返回自动推断结果。"""
    df = pd.DataFrame({
        "id": range(100),
        "feat": range(100),
        "target": [0, 1, 2] * 33 + [0],
    })
    meta = analyze_metadata(df)
    assert meta["suggested_target_column"] == "target"
    assert meta["suggested_task_type"] == "multiclass_classification"
    assert meta["suggested_target_confidence"] >= 0.7


def test_schema_infer_datetime_string():
    """可解析的日期字符串列应被识别为 DATETIME。"""
    s = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01", None])
    assert infer_field_type(s) == "datetime"


def test_schema_infer_id_by_name_and_sequential():
    """名为 row_id 的单调整数列应判为 ID。"""
    s = pd.Series(range(100), name="row_id")
    assert infer_field_type(s, name="row_id") == "id"


def test_schema_infer_id_by_monotonicity():
    """即使无名，单调整数列也应判为 ID。"""
    s = pd.Series(range(1000))
    assert infer_field_type(s) == "id"


def test_schema_does_not_overclassify_high_card_numeric_as_id():
    """高基数数值列（如浮点）不应被判为 ID。"""
    s = pd.Series([i + 0.1 for i in range(200)])
    assert infer_field_type(s) == "numeric"


def test_schema_infer_categorical_adaptive_threshold():
    """低基数数值列应按自适应阈值判为 CATEGORICAL。"""
    # 60 行 3 个唯一值：cutoff = max(2, int(60*0.05)) = 3
    s = pd.Series([1, 2, 3] * 20)
    assert infer_field_type(s) == "categorical"


def test_schema_infer_text_high_cardinality():
    """半唯一且取值很多的字符串列判为 TEXT。"""
    s = pd.Series([f"long text description number {i}" for i in range(500)])
    assert infer_field_type(s) == "text"


def test_infer_schema_allowed_values_are_records_only():
    """Schema 中的 allowed_values 仅记录，不应导致未知类别校验失败。"""
    df_train = pd.DataFrame({
        "cat": ["a", "b", "a"],
        "num": [1.0, 2.0, 3.0],
    })
    schema = infer_schema(df_train)
    cat_field = next(f for f in schema.fields if f.name == "cat")
    assert set(cat_field.constraints.allowed_values) == {"a", "b"}
