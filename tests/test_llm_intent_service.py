# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 意图解析服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
import pytest

from services.llm_intent_service import parse_intent, IntentConfig


# 当 openai SDK 未安装时，跳过需要模拟 LLM 调用的测试
try:
    import openai  # noqa: F401
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [3000.0, 5000.0, 8000.0, 12000.0, 15000.0],
        "defaulted": [0, 1, 0, 1, 0],
    })


@pytest.mark.asyncio
async def test_rule_based_parse_binary_classification():
    df = _make_sample_df()
    result = await parse_intent(
        "预测用户是否会违约，用 defaulted 作为目标列，二分类",
        df_sample=df,
        provider="auto",  # 无 API key，会降级
    )
    assert result.provider == "rule_engine"
    assert result.target_column == "defaulted"
    assert result.task_type == "binary_classification"
    assert result.primary_metric in ("f1", "accuracy")


@pytest.mark.asyncio
async def test_rule_based_parse_regression():
    df = pd.DataFrame({
        "sqft": [1000, 1500, 2000],
        "price": [200000.0, 300000.0, 400000.0],
    })
    result = await parse_intent("根据面积预测房价", df_sample=df, provider="auto")
    assert result.task_type == "regression"
    assert result.target_column == "price"
    assert result.primary_metric == "root_mean_squared_error"


@pytest.mark.asyncio
async def test_rule_based_infers_time_budget_and_max_models():
    df = _make_sample_df()
    result = await parse_intent(
        "用 income 预测 defaulted，预算 5 分钟，最多 10 个模型",
        df_sample=df,
        provider="auto",
    )
    assert result.time_budget_minutes == 5.0
    assert result.max_models == 10


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai SDK not installed")
@pytest.mark.asyncio
async def test_llm_parse_success(monkeypatch):
    async def _fake_call(provider, messages):
        return (
            '{"target_column": "defaulted", "task_type": "binary_classification", '
            '"primary_metric": "roc_auc", "time_budget_minutes": 5, "max_models": 8, '
            '"preset": "good_quality", "confidence": 0.9, "reasoning": "测试"}'
        )

    monkeypatch.setenv("KIMI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "services.llm_intent_service._call_provider",
        _fake_call,
    )

    df = _make_sample_df()
    result = await parse_intent("预测违约", df_sample=df, provider="kimi")
    assert result.provider == "kimi"
    assert result.target_column == "defaulted"
    assert result.task_type == "binary_classification"
    assert result.primary_metric == "roc_auc"
    assert result.time_budget_minutes == 5.0
    assert result.max_models == 8


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai SDK not installed")
@pytest.mark.asyncio
async def test_llm_parse_invalid_json_fallback_to_rule(monkeypatch):
    async def _fake_call(provider, messages):
        return "这不是 JSON"

    monkeypatch.setenv("KIMI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "services.llm_intent_service._call_provider",
        _fake_call,
    )

    df = _make_sample_df()
    result = await parse_intent("预测 defaulted", df_sample=df, provider="kimi")
    assert result.provider == "rule_engine"
    assert result.target_column == "defaulted"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai SDK not installed")
@pytest.mark.asyncio
async def test_llm_parse_missing_fields_filled_by_rule(monkeypatch):
    async def _fake_call(provider, messages):
        return '{"task_type": "binary_classification", "confidence": 0.7}'

    monkeypatch.setenv("KIMI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "services.llm_intent_service._call_provider",
        _fake_call,
    )

    df = _make_sample_df()
    result = await parse_intent("预测违约", df_sample=df, provider="kimi")
    assert result.provider == "kimi"
    assert result.task_type == "binary_classification"
    # 目标列由规则引擎补齐
    assert result.target_column == "defaulted"


@pytest.mark.asyncio
async def test_empty_query_returns_rule_fallback():
    result = await parse_intent("", df_sample=None, provider="auto")
    assert result.provider == "rule_engine"
    assert result.target_column is None
