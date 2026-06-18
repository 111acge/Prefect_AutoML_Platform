"""LLM 业务解读报告服务。

根据训练结果、特征重要性、数据质量报告生成面向业务人员的自然语言解读。
所有 LLM 调用均为可选，失败时自动降级到规则模板。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from services.llm_intent_service import (
    _resolve_providers,
    _call_provider,
    _parse_llm_json,
)

logger = logging.getLogger(__name__)


def _build_business_interpretation_prompt(
    task_type: str,
    primary_metric: Optional[str],
    metrics: Dict[str, Any],
    feature_importance: List[Dict[str, Any]],
    quality: Optional[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """构造业务解读提示。"""
    system_prompt = (
        "你是 AutoML 平台的业务解读助手。请根据以下训练结果，为业务人员生成一份"
        "简洁、 actionable 的中文解读报告。返回严格 JSON（不要 Markdown 代码块），包含：\n"
        "- business_summary: 一段话概括模型表现和业务含义\n"
        "- key_insights: 列表，3-5 条关键发现\n"
        "- feature_interpretations: 列表，对 Top 5 特征分别说明其业务含义\n"
        "- caveats: 列表，使用模型时需要注意的限制或风险\n"
        "- recommendations: 列表，针对业务应用的下一步建议\n"
        "请用业务语言，避免过多技术术语。"
    )

    top_features = feature_importance[:5] if feature_importance else []
    quality_summary = {
        "overall_score": quality.get("overall_score") if quality else None,
        "n_rows": quality.get("n_rows") if quality else None,
        "n_features": quality.get("n_features") if quality else None,
        "warnings": quality.get("warnings", [])[:5] if quality else [],
    }
    strategy_summary = {
        "preset": strategy.get("preset") if strategy else None,
        "data_size_label": strategy.get("data_size_label") if strategy else None,
        "primary_metric": strategy.get("primary_metric") if strategy else None,
    }

    user_prompt = (
        f"任务类型: {task_type}\n"
        f"主评估指标: {primary_metric}\n"
        f"测试集指标: {json.dumps(metrics.get('final', {}), ensure_ascii=False, default=str)}\n"
        f"训练策略: {json.dumps(strategy_summary, ensure_ascii=False, default=str)}\n"
        f"数据质量摘要: {json.dumps(quality_summary, ensure_ascii=False, default=str)}\n"
        f"Top 5 特征重要性: {json.dumps(top_features, ensure_ascii=False, default=str)}\n"
        "请返回 JSON 格式的业务解读报告:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _rule_based_interpretation(
    task_type: str,
    primary_metric: Optional[str],
    metrics: Dict[str, Any],
    feature_importance: List[Dict[str, Any]],
    quality: Optional[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """LLM 失败时的规则模板兜底。"""
    final = metrics.get("final", {})
    metric_value = next(iter(final.values())) if final else None
    top_features = [f.get("feature", f.get("Unnamed: 0", "未知")) for f in feature_importance[:5]]

    if task_type in ("binary_classification", "multiclass_classification"):
        summary = (
            f"这是一个分类任务，主指标为 {primary_metric}。"
            f"模型在测试集上的表现为 {metric_value}，可作为业务决策参考。"
        )
    else:
        summary = (
            f"这是一个回归任务，主指标为 {primary_metric}。"
            f"模型在测试集上的表现为 {metric_value}，可用于预测目标值。"
        )

    warnings = quality.get("warnings", []) if quality else []
    caveats = []
    if warnings:
        caveats.append(f"数据质量提示: {warnings[0]}")
    caveats.append("模型表现基于历史数据，未来数据分布变化可能导致性能下降。")
    caveats.append("建议定期用新数据重新评估模型。")

    return {
        "business_summary": summary,
        "key_insights": [
            f"Top 重要特征包括: {', '.join(top_features)}",
            "模型已自动选择 preset 和验证策略，适合当前数据规模。",
            "建议在正式部署前进行 A/B 测试或小范围试运行。",
        ],
        "feature_interpretations": [
            {"feature": f, "interpretation": "该特征对模型预测有显著影响，建议关注其业务含义。"}
            for f in top_features
        ],
        "caveats": caveats,
        "recommendations": [
            "结合业务经验验证 Top 特征的合理性。",
            "监控模型上线后的实际效果。",
            "收集更多标注数据以持续优化。",
        ],
        "provider": "rule_template",
    }


async def generate_business_interpretation(
    task_type: str,
    primary_metric: Optional[str],
    metrics: Dict[str, Any],
    feature_importance: List[Dict[str, Any]],
    quality: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    provider: str = "auto",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """生成业务解读报告。

    LLM 不可用或失败时，自动降级到规则模板。
    """
    providers = _resolve_providers(provider)
    if not providers:
        logger.warning("未配置 LLM API 密钥，业务解读降级到规则模板")
        return _rule_based_interpretation(
            task_type, primary_metric, metrics, feature_importance, quality, strategy
        )

    messages = _build_business_interpretation_prompt(
        task_type, primary_metric, metrics, feature_importance, quality, strategy
    )

    for prov in providers:
        try:
            raw = await asyncio.wait_for(_call_provider(prov, messages), timeout=timeout)
            parsed = _parse_llm_json(raw)
            parsed["provider"] = prov
            # 确保必要字段存在
            for key in ["business_summary", "key_insights", "feature_interpretations", "caveats", "recommendations"]:
                if key not in parsed:
                    parsed[key] = []
            return parsed
        except asyncio.TimeoutError:
            logger.warning("LLM 业务解读 %s 超时", prov)
        except Exception as e:
            logger.warning("LLM 业务解读失败 (%s): %s", prov, e)

    logger.warning("所有 LLM 提供商业务解读失败，降级到规则模板")
    return _rule_based_interpretation(
        task_type, primary_metric, metrics, feature_importance, quality, strategy
    )
