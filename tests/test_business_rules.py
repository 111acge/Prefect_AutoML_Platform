# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""业务规则提取测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
import pytest

from services.llm_intent_service import extract_business_rules, infer_schema_with_llm


@pytest.mark.asyncio
async def test_rule_based_extract_drop_column():
    df = pd.DataFrame({"id": [1, 2, 3], "age": [20, 30, 40], "target": [0, 1, 0]})
    rules = await extract_business_rules("删除 id 列", df_sample=df, provider="auto")
    assert "id" in rules["drop_columns"]


@pytest.mark.asyncio
async def test_rule_based_extract_value_constraint():
    df = pd.DataFrame({"年龄": [20, 30, 200], "target": [0, 1, 0]})
    rules = await extract_business_rules("年龄不能大于 120", df_sample=df, provider="auto")
    assert len(rules["value_constraints"]) == 1
    assert rules["value_constraints"][0]["column"] == "年龄"
    assert rules["value_constraints"][0]["max_value"] == 120.0


@pytest.mark.asyncio
async def test_rule_based_extract_impute():
    df = pd.DataFrame({"age": [20, None, 40], "target": [0, 1, 0]})
    rules = await extract_business_rules("缺失值用 0 填充", df_sample=df, provider="auto")
    assert rules["numeric_impute_strategy"] == "constant"
    assert rules["numeric_impute_constant"] == 0.0


@pytest.mark.asyncio
async def test_infer_schema_fallback():
    df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
    schema = await infer_schema_with_llm(df, provider="auto")
    names = {f["name"] for f in schema}
    assert "num" in names
    assert "cat" in names
