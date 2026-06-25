# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""Schema 服务单元测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd

from services.schema_service import (
    FieldType,
    infer_field_type,
    infer_schema,
    validate_against_schema,
    align_to_schema,
)


def test_infer_numeric_field():
    """测试数值字段推断。"""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert infer_field_type(series) == FieldType.NUMERIC


def test_infer_categorical_field():
    """测试类别字段推断。"""
    series = pd.Series(["a", "b", "c", "a", "b"])
    assert infer_field_type(series) == FieldType.CATEGORICAL


def test_infer_binary_field():
    """测试二值字段推断。"""
    series = pd.Series([0, 1, 0, 1, 0])
    assert infer_field_type(series) == FieldType.BINARY


def test_infer_text_field():
    """测试文本字段推断。"""
    series = pd.Series([f"text_{i}" for i in range(150)])
    assert infer_field_type(series) == FieldType.TEXT


def test_infer_id_like_high_cardinality_field():
    """测试高基数 ID-like 字段推断。"""
    # 模拟单据号：大量唯一值但未达到文本阈值
    series = pd.Series([f"DOC{ i:08d}" for i in range(200_000)])
    assert infer_field_type(series) == FieldType.ID


def test_infer_datetime_field():
    """测试时间字段推断。"""
    series = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert infer_field_type(series) == FieldType.DATETIME


def test_infer_schema():
    """测试 Schema 推断。"""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
            "target": ["x", "y", "x"],
        }
    )
    schema = infer_schema(df, target_column="target")
    assert len(schema.fields) == 3
    assert schema.target_column == "target"

    field_map = {f.name: f for f in schema.fields}
    assert field_map["num"].field_type == FieldType.NUMERIC
    assert field_map["cat"].field_type == FieldType.CATEGORICAL
    assert field_map["target"].field_type == FieldType.BINARY


def test_validate_against_schema_missing_column():
    """测试缺少字段校验。"""
    df = pd.DataFrame({"a": [1, 2, 3]})
    schema = infer_schema(df)

    df_missing = pd.DataFrame({"b": [1, 2, 3]})
    errors = validate_against_schema(df_missing, schema)
    assert any("缺少必填字段" in e for e in errors)


def test_validate_against_schema_allows_unknown_category():
    """未知类别取值不再硬性报错，由预处理统一编码。"""
    df = pd.DataFrame({"cat": ["a", "b", "a"]})
    schema = infer_schema(df)

    df_new = pd.DataFrame({"cat": ["a", "c", "a"]})
    errors = validate_against_schema(df_new, schema)
    assert not any("未允许的取值" in e for e in errors)


def test_align_to_schema():
    """测试数据对齐。"""
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
    schema = infer_schema(df)

    # 模拟新数据，数值被读成字符串
    new_df = pd.DataFrame({"num": ["4", "5", "6"], "cat": ["d", "e", "f"]})
    aligned = align_to_schema(new_df, schema)
    assert aligned["num"].dtype.kind in "iuf"
    assert aligned["cat"].dtype == object or aligned["cat"].dtype.name == "object"


def test_align_to_schema_add_missing_column():
    """测试对齐时补充缺失列。"""
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    full_df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
    schema = infer_schema(full_df)

    aligned = align_to_schema(df, schema)
    assert "cat" in aligned.columns
