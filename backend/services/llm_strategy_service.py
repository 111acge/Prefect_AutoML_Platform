# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 元策略推荐器。

根据数据集形态（metadata + quality_report）让大模型推荐一组候选配置，
用于 Agent 搜索空间初始化或单次运行策略增强。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from i18n import _, get_locale
from services.llm_client import call_llm_for_json
from schemas import CandidateConfig, CleaningRules

logger = logging.getLogger(__name__)


def _default_candidates(task_type: str, primary_metric: Optional[str] = None) -> List[CandidateConfig]:
    """LLM 不可用时返回的经验默认候选。"""
    return [
        CandidateConfig(
            preset="medium_quality",
            max_models=50,
            time_budget_minutes=10,
            primary_metric=primary_metric,
            feature_engineering_enabled=True,
            reasoning=_("strategy.llm_unavailable_default"),
        ),
        CandidateConfig(
            preset="good_quality",
            max_models=80,
            time_budget_minutes=20,
            primary_metric=primary_metric,
            feature_engineering_enabled=True,
            reasoning=_("strategy.llm_unavailable_stronger"),
        ),
    ]


def _parse_cleaning_rules(raw: Optional[Dict[str, Any]]) -> Optional[CleaningRules]:
    if not raw:
        return None
    try:
        return CleaningRules(**raw)
    except ValidationError as e:
        logger.warning(_("llm.strategy.cleaning_rules_invalid", msg=e))
        return None


def _raw_to_candidate(raw: Dict[str, Any]) -> Optional[CandidateConfig]:
    """把 LLM 返回的单个候选 JSON 转为 CandidateConfig。"""
    try:
        params = {k: v for k, v in raw.items() if k != "reasoning"}
        if "cleaning_rules" in params:
            params["cleaning_rules"] = _parse_cleaning_rules(params["cleaning_rules"])
        candidate = CandidateConfig(**params)
        candidate.reasoning = raw.get("reasoning") or candidate.reasoning
        return candidate
    except ValidationError as e:
        logger.warning(_("llm.strategy.candidate_validation_failed", msg=e, raw=raw))
        return None


def _build_strategy_prompt(
    metadata: Dict[str, Any],
    quality: Optional[Dict[str, Any]],
    task_type: str,
    primary_metric: Optional[str] = None,
) -> List[Dict[str, str]]:
    """构造给 LLM 的策略推荐 Prompt。"""
    system_prompt = _("llm_prompt.strategy_system")
    primary_metric_text = primary_metric or _("llm_prompt.strategy_primary_metric_unspecified")
    user_prompt = _(
        "llm_prompt.strategy_user",
        task_type=task_type,
        primary_metric=primary_metric_text,
        metadata=json.dumps(metadata, ensure_ascii=False, default=str),
        quality=json.dumps(quality or {}, ensure_ascii=False, default=str),
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def recommend_candidates(
    metadata: Dict[str, Any],
    quality: Optional[Dict[str, Any]],
    task_type: str,
    primary_metric: Optional[str] = None,
    provider: str = "auto",
    max_candidates: int = 3,
) -> List[CandidateConfig]:
    """基于数据集形态推荐候选配置列表。

    LLM 失败或不可用时，返回经验默认候选。
    """
    try:
        messages = _build_strategy_prompt(metadata, quality, task_type, primary_metric)
        result = await call_llm_for_json(
            messages,
            provider=provider,
            max_tokens=4096,
            temperature=0.2,
            timeout=60.0,
            retries=1,
        )
        raw_candidates = result.get("candidates", [])
        if not isinstance(raw_candidates, list):
            logger.warning(_("llm.strategy.candidates_not_list"))
            return _default_candidates(task_type, primary_metric)

        candidates: List[CandidateConfig] = []
        for raw in raw_candidates[:max_candidates]:
            candidate = _raw_to_candidate(raw)
            if candidate:
                candidates.append(candidate)

        if not candidates:
            logger.warning(_("llm.strategy.no_valid_candidates"))
            return _default_candidates(task_type, primary_metric)

        return candidates
    except Exception as e:
        logger.warning(_("llm.strategy.recommend_failed", msg=e))
        return _default_candidates(task_type, primary_metric)
