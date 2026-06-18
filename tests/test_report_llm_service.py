"""LLM 业务解读服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from services.report_llm_service import (
    _rule_based_interpretation,
    generate_business_interpretation,
)


def test_rule_based_interpretation_returns_required_fields():
    result = _rule_based_interpretation(
        task_type="binary_classification",
        primary_metric="f1",
        metrics={"final": {"f1": 0.85}},
        feature_importance=[
            {"feature": "age", "importance": 0.5},
            {"feature": "income", "importance": 0.3},
        ],
        quality=None,
        strategy=None,
    )

    assert "business_summary" in result
    assert "key_insights" in result
    assert "feature_interpretations" in result
    assert "caveats" in result
    assert "recommendations" in result
    assert result["provider"] == "rule_template"
    assert len(result["feature_interpretations"]) == 2


@pytest.mark.asyncio
async def test_generate_business_interpretation_fallback_without_api_key():
    """未配置 API 密钥时应降级到规则模板。"""
    result = await generate_business_interpretation(
        task_type="binary_classification",
        primary_metric="f1",
        metrics={"final": {"f1": 0.85}},
        feature_importance=[{"feature": "age", "importance": 0.5}],
        quality=None,
        strategy=None,
    )

    assert result["provider"] == "rule_template"
    assert "business_summary" in result
