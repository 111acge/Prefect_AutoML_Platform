# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 业务解读报告服务。

根据训练结果、特征重要性、数据质量报告生成面向业务人员的自然语言解读。
所有 LLM 调用均为可选，失败时自动降级到规则模板。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from i18n import _, get_locale
from services.llm_client import call_provider, resolve_providers
from services.llm_intent_service import _parse_llm_json

logger = logging.getLogger(__name__)


def _build_business_interpretation_prompt(
    task_type: str,
    primary_metric: Optional[str],
    metrics: Dict[str, Any],
    feature_importance: List[Dict[str, Any]],
    quality: Optional[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """构造业务解读提示（按当前语言切换）。"""
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

    system_prompt = _("llm_prompt.business_interpretation_system")
    user_prompt = _(
        "llm_prompt.business_interpretation_user",
        task_type=task_type,
        primary_metric=primary_metric,
        metrics=json.dumps(metrics.get("final", {}), ensure_ascii=False, default=str),
        strategy_summary=json.dumps(strategy_summary, ensure_ascii=False, default=str),
        quality_summary=json.dumps(quality_summary, ensure_ascii=False, default=str),
        top_features=json.dumps(top_features, ensure_ascii=False, default=str),
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
    top_features = [f.get("feature", f.get("Unnamed: 0", _("common.unknown"))) for f in feature_importance[:5]]

    if task_type in ("binary_classification", "multiclass_classification"):
        summary = _("report_llm.classification_summary", metric=primary_metric, value=metric_value)
    else:
        summary = _("report_llm.regression_summary", metric=primary_metric, value=metric_value)

    warnings = quality.get("warnings", []) if quality else []
    caveats = []
    if warnings:
        caveats.append(_("report_llm.data_quality_hint", warning=warnings[0]))
    caveats.append(_("report_llm.performance_may_degrade"))
    caveats.append(_("report_llm.re_evaluate_regularly"))

    return {
        "business_summary": summary,
        "key_insights": [
            _("report_llm.top_features", features=", ".join(top_features)),
            _("report_llm.auto_strategy_selected"),
            _("report_llm.ab_test_recommendation"),
        ],
        "feature_interpretations": [
            {"feature": f, "interpretation": _("report_llm.feature_significant")}
            for f in top_features
        ],
        "caveats": caveats,
        "recommendations": [
            _("report_llm.validate_features"),
            _("report_llm.monitor_effect"),
            _("report_llm.collect_more_data"),
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
    force_rule_based: bool = False,
    raise_on_failure: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """生成业务解读报告。

    LLM 不可用或失败时，自动降级到规则模板（除非 raise_on_failure=True）。

    Args:
        force_rule_based: 为 True 时直接返回规则模板（用于训练流程默认不自动调用 LLM，
                         避免用户未明确同意就将数据发送到第三方 LLM）。
        raise_on_failure: 为 True 时 LLM 调用失败抛出异常，不静默降级（用于用户主动触发的重新生成）。
    """
    if force_rule_based:
        logger.info(_("report_llm.rule_based_log"))
        return _rule_based_interpretation(
            task_type, primary_metric, metrics, feature_importance, quality, strategy
        )

    providers = resolve_providers(provider, api_key=api_key)
    if not providers:
        msg = _("report_llm.no_api_key")
        logger.warning("%s, falling back to rule-based interpretation", msg)
        if raise_on_failure:
            raise RuntimeError(msg)
        return _rule_based_interpretation(
            task_type, primary_metric, metrics, feature_importance, quality, strategy
        )

    messages = _build_business_interpretation_prompt(
        task_type, primary_metric, metrics, feature_importance, quality, strategy
    )

    last_error: Optional[Exception] = None
    for prov in providers:
        try:
            raw = await call_provider(
                prov,
                messages,
                max_tokens=None,
                temperature=0.0,
                timeout=None,
                api_key=api_key,
            )
            parsed = _parse_llm_json(raw)
            parsed["provider"] = prov
            # 确保必要字段存在
            for key in ["business_summary", "key_insights", "feature_interpretations", "caveats", "recommendations"]:
                if key not in parsed:
                    parsed[key] = []
            return parsed
        except asyncio.TimeoutError:
            logger.warning(_("report_llm.timeout", provider=prov))
            last_error = asyncio.TimeoutError(f"{prov} timeout")
        except Exception as e:
            logger.warning(_("report_llm.failed", provider=prov, msg=e))
            last_error = e

    msg = _("report_llm.all_failed", last_error=last_error)
    logger.warning(msg)
    if raise_on_failure:
        if last_error is not None:
            raise last_error
        raise RuntimeError(msg)
    return _rule_based_interpretation(
        task_type, primary_metric, metrics, feature_importance, quality, strategy
    )
