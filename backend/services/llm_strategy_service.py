"""LLM 元策略推荐器。

根据数据集形态（metadata + quality_report）让大模型推荐一组候选配置，
用于 Agent 搜索空间初始化或单次运行策略增强。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

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
            reasoning="LLM 不可用，使用默认经验配置",
        ),
        CandidateConfig(
            preset="good_quality",
            max_models=80,
            time_budget_minutes=20,
            primary_metric=primary_metric,
            feature_engineering_enabled=True,
            reasoning="LLM 不可用，使用稍强的经验配置",
        ),
    ]


def _parse_cleaning_rules(raw: Optional[Dict[str, Any]]) -> Optional[CleaningRules]:
    if not raw:
        return None
    try:
        return CleaningRules(**raw)
    except ValidationError as e:
        logger.warning(f"LLM 推荐的 cleaning_rules 格式无效，忽略: {e}")
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
        logger.warning(f"LLM 候选配置校验失败: {e}; raw={raw}")
        return None


def _build_strategy_prompt(
    metadata: Dict[str, Any],
    quality: Optional[Dict[str, Any]],
    task_type: str,
    primary_metric: Optional[str] = None,
) -> List[Dict[str, str]]:
    """构造给 LLM 的策略推荐 Prompt。"""
    system_prompt = (
        "你是 AutoML 平台的元策略专家。请根据数据集元信息和质量报告，"
        "推荐 1~3 组候选训练配置（JSON 格式）。目标是让模型在验证集上的 primary_metric 尽可能好。\n\n"
        "输出严格 JSON，顶层字段为:\n"
        "- candidates: 候选配置列表（1~3 个）\n\n"
        "每个候选配置字段:\n"
        "- preset: AutoGluon preset，可选 auto / best_quality / high_quality / good_quality / medium_quality / fast_training\n"
        "- max_models: 整数 1~200\n"
        "- time_budget_minutes: 数字 >=0.1（分钟）\n"
        "- primary_metric: 可选 accuracy/f1/f1_macro/log_loss/roc_auc/auc_pr/root_mean_squared_error/r2 等\n"
        "- feature_engineering_enabled: true / false\n"
        "- cleaning_rules: 可选，包含 remove_duplicates(bool)、drop_rows_with_missing_target(bool)、"
        "numeric_impute_strategy('median'|'mean'|'constant')、categorical_impute_strategy('mode'|'constant')、"
        "drop_columns(字符串列表)、value_constraints(列表，每项 {column, min_value, max_value})\n"
        "- hyperparameters: 可选，AutoGluon 模型搜索空间覆盖，如 {'CAT': {'iterations': 500}}\n"
        "- validation_strategy: 可选，如 {'name': 'cv', 'n_folds': 5}\n"
        "- reasoning: 简短说明为什么推荐这组配置\n\n"
        "请只返回 JSON，不要 Markdown 代码块。"
    )

    user_prompt = (
        f"任务类型: {task_type}\n"
        f"用户指定主指标: {primary_metric or '未指定，请根据数据推荐'}\n"
        f"数据集元信息: {json.dumps(metadata, ensure_ascii=False, default=str)}\n"
        f"数据质量报告: {json.dumps(quality or {}, ensure_ascii=False, default=str)}\n"
        "请推荐候选配置:"
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
            logger.warning("LLM 策略推荐返回的 candidates 不是列表，使用默认候选")
            return _default_candidates(task_type, primary_metric)

        candidates: List[CandidateConfig] = []
        for raw in raw_candidates[:max_candidates]:
            candidate = _raw_to_candidate(raw)
            if candidate:
                candidates.append(candidate)

        if not candidates:
            logger.warning("LLM 未返回有效候选配置，使用默认候选")
            return _default_candidates(task_type, primary_metric)

        return candidates
    except Exception as e:
        logger.warning(f"LLM 元策略推荐失败，降级到默认候选: {e}")
        return _default_candidates(task_type, primary_metric)
