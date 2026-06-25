# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 意图理解服务：将自然语言解析为训练配置。

支持多 LLM 提供商（KIMI / DeepSeek / MiniMax 等 OpenAI 兼容端点），
并在 API 失败/超时/未配置时降级到规则引擎。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from config import settings

logger = logging.getLogger(__name__)

# 动态用户配置（支持前端配置 LLM provider/key/model）
from services.llm_settings_service import (
    get_active_api_key as _get_dynamic_api_key,
    get_active_model as _get_dynamic_model,
    get_active_provider as _get_dynamic_provider,
    get_provider_config as _get_dynamic_provider_config,
    SUPPORTED_PROVIDERS as _DYNAMIC_SUPPORTED_PROVIDERS,
)

# 可选依赖：openai SDK
try:
    from openai import AsyncOpenAI, APIError

    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - 仅用于类型/导入保护
    AsyncOpenAI = None  # type: ignore[misc, assignment]
    APIError = Exception  # type: ignore[misc, assignment]
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# 提供商配置
# ---------------------------------------------------------------------------

_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "env_key": "KIMI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M3",
        "env_key": "MINIMAX_API_KEY",
    },
    "openai": {
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "env_key": "GLM_API_KEY",
    },
}


def _provider_config(provider: str) -> Dict[str, Any]:
    """返回合并后的提供商配置（动态配置覆盖静态默认值）。"""
    cfg = dict(_PROVIDER_CONFIG.get(provider, {}))
    dynamic = _get_dynamic_provider_config(provider) or {}
    for key in ("base_url", "default_model"):
        if dynamic.get(key):
            cfg[key] = dynamic[key]
    return cfg

_TASK_TYPES = ("binary_classification", "multiclass_classification", "regression")
_METRICS = (
    "accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "precision",
    "recall",
    "roc_auc",
    "log_loss",
    "mcc",
    "root_mean_squared_error",
    "mean_squared_error",
    "mean_absolute_error",
    "r2",
    "auc_pr",
)
_PRESETS = ("auto", "best_quality", "high_quality", "good_quality", "medium_quality", "fast_training")


class IntentConfig(BaseModel):
    """解析后的训练意图配置。"""

    target_column: Optional[str] = None
    task_type: Optional[str] = None
    primary_metric: Optional[str] = None
    # None 表示不限制训练时间（无穷大）
    time_budget_minutes: Optional[float] = Field(default=None, ge=0.1)
    max_models: Optional[int] = Field(default=None, ge=1, le=200)
    preset: Optional[str] = "auto"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provider: Optional[str] = None
    reasoning: Optional[str] = None

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in _TASK_TYPES:
            raise ValueError(f"不支持的任务类型: {v}")
        return v

    @field_validator("primary_metric")
    @classmethod
    def _validate_metric(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in _METRICS:
            raise ValueError(f"不支持的评估指标: {v}")
        return v

    @field_validator("preset")
    @classmethod
    def _validate_preset(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "auto"
        v = v.strip().lower()
        if v not in _PRESETS:
            raise ValueError(f"不支持的 preset: {v}")
        return v


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


async def parse_intent(
    query: str,
    df_sample: Optional[pd.DataFrame] = None,
    provider: str = "auto",
    timeout: float = 30.0,
) -> IntentConfig:
    """解析用户自然语言意图。

    参数：
        query: 用户自然语言描述，例如"用 income 列预测是否违约，二分类，预算 5 分钟"。
        df_sample: 可选的数据集样例，用于规则降级与辅助 LLM 推断。
        provider: 指定 LLM 提供商；"auto" 会按配置优先级尝试。
        timeout: 单个提供商调用的超时时间（秒）。

    返回：
        IntentConfig，包含目标列、任务类型、指标等配置。
    """
    query = (query or "").strip()
    if not query:
        return _rule_based_parse("", df_sample)

    # 未安装 openai 库时直接降级
    if not _OPENAI_AVAILABLE:
        logger.warning("openai 库未安装，LLM 意图解析降级到规则引擎")
        return _rule_based_parse(query, df_sample)

    providers = _resolve_providers(provider)
    if not providers:
        logger.warning("未配置任何 LLM API 密钥，降级到规则引擎")
        return _rule_based_parse(query, df_sample)

    messages = _build_messages(query, df_sample)

    last_error: Optional[Exception] = None
    for prov in providers:
        try:
            raw = await asyncio.wait_for(
                _call_provider(prov, messages),
                timeout=timeout,
            )
            parsed = _parse_llm_json(raw)
            parsed["provider"] = prov
            parsed["confidence"] = parsed.get("confidence", 0.8)
            # 规则层兜底：若 LLM 未给出关键字段，用规则补齐
            merged = _merge_with_rule_fallback(parsed, query, df_sample)
            return IntentConfig(**merged)
        except asyncio.TimeoutError:
            logger.warning("LLM 提供商 %s 调用超时", prov)
            last_error = asyncio.TimeoutError(f"{prov} timeout")
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 提供商 %s 调用失败: %s", prov, e)
            last_error = e

    logger.warning("所有 LLM 提供商均失败，降级到规则引擎; 最后错误: %s", last_error)
    return _rule_based_parse(query, df_sample)


async def extract_business_rules(
    query: str,
    df_sample: Optional[pd.DataFrame] = None,
    provider: str = "auto",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """从自然语言中提取数据清洗规则。

    例如：
        - "年龄不能大于 120" -> value_constraints
        - "删除 ID 列" -> drop_columns
        - "缺失值用 0 填充" -> numeric_impute_strategy
    """
    query = (query or "").strip()
    if not query:
        return _rule_based_extract_rules(query, df_sample)

    if not _OPENAI_AVAILABLE:
        return _rule_based_extract_rules(query, df_sample)

    providers = _resolve_providers(provider)
    if not providers:
        return _rule_based_extract_rules(query, df_sample)

    messages = _build_rule_extraction_messages(query, df_sample)

    for prov in providers:
        try:
            raw = await asyncio.wait_for(_call_provider(prov, messages), timeout=timeout)
            parsed = _parse_llm_json(raw)
            # 用规则层校验/补齐
            rules = _normalize_rules(parsed)
            rules["provider"] = prov
            return rules
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 规则提取失败 (%s): %s", prov, e)

    return _rule_based_extract_rules(query, df_sample)


async def infer_schema_with_llm(
    df_sample: pd.DataFrame,
    provider: str = "auto",
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """LLM 辅助 Schema 推断，失败时返回规则推断的 Schema。"""
    if df_sample.empty:
        return []

    # 先用规则引擎兜底
    from services.schema_service import infer_schema

    rule_schema = infer_schema(df_sample).to_dict()["fields"]

    if not _OPENAI_AVAILABLE:
        return rule_schema

    providers = _resolve_providers(provider)
    if not providers:
        return rule_schema

    messages = _build_schema_inference_messages(df_sample)

    for prov in providers:
        try:
            raw = await asyncio.wait_for(_call_provider(prov, messages), timeout=timeout)
            parsed = _parse_llm_json(raw)
            fields = parsed.get("fields", [])
            if fields and isinstance(fields, list):
                # 仅当字段名存在时才采用，否则回退规则
                known_cols = set(df_sample.columns)
                valid = [f for f in fields if f.get("name") in known_cols]
                if valid:
                    return valid
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM Schema 推断失败 (%s): %s", prov, e)

    return rule_schema


# ---------------------------------------------------------------------------
# LLM 调用实现
# ---------------------------------------------------------------------------


def _resolve_providers(provider: str) -> List[str]:
    """根据 provider 参数与可用配置确定实际尝试的提供商列表。"""
    # 如果用户通过前端指定了 provider，优先只使用它
    dynamic_provider = _get_dynamic_provider()

    if provider and provider.lower() != "auto":
        prov = provider.lower()
        if prov not in _PROVIDER_CONFIG and prov not in _DYNAMIC_SUPPORTED_PROVIDERS:
            logger.warning("未知 LLM 提供商: %s，将尝试 auto", prov)
        elif _get_api_key(prov):
            return [prov]
        else:
            logger.warning("LLM 提供商 %s 未配置 API 密钥", prov)

    # 动态 provider 优先级最高
    if dynamic_provider and _get_api_key(dynamic_provider):
        return [dynamic_provider]

    # auto 优先级：KIMI -> DeepSeek -> MiniMax -> GLM -> OpenAI
    order = ["kimi", "deepseek", "minimax", "glm", "openai"]
    return [p for p in order if _get_api_key(p)]


def _get_api_key(provider: str) -> Optional[str]:
    """读取对应提供商的 API 密钥（优先动态配置，其次环境变量/pydantic settings）。"""
    if _get_dynamic_provider() == provider:
        dynamic_key = _get_dynamic_api_key()
        if dynamic_key:
            return dynamic_key

    cfg = _PROVIDER_CONFIG.get(provider, {})
    env_key = cfg.get("env_key")
    if env_key:
        value = os.environ.get(env_key) or getattr(settings, env_key.lower(), None)
        if value:
            return str(value)
    # openai 兜底
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None)
    return None


async def _call_provider(provider: str, messages: List[Dict[str, str]]) -> str:
    """调用指定 OpenAI 兼容端点。"""
    if not _OPENAI_AVAILABLE or AsyncOpenAI is None:
        raise RuntimeError("openai 库未安装，无法调用 LLM")

    cfg = _provider_config(provider)
    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"未配置 {provider} API 密钥")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if cfg.get("base_url"):
        client_kwargs["base_url"] = cfg["base_url"]

    client = AsyncOpenAI(**client_kwargs)
    # 动态模型覆盖：若当前激活 provider 匹配，优先使用用户指定的 model
    model = cfg.get("default_model")
    if _get_dynamic_provider() == provider:
        dynamic_model = _get_dynamic_model()
        if dynamic_model:
            model = dynamic_model
    if not model:
        model = getattr(settings, "default_llm_model", None) or cfg["default_model"]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _build_messages(query: str, df_sample: Optional[pd.DataFrame]) -> List[Dict[str, str]]:
    """构造 LLM 对话消息。"""
    system_prompt = (
        "你是一个 AutoML 平台的意图解析助手。用户会用自然语言描述他想训练的机器学习任务，"
        "你需要提取出以下字段并返回严格 JSON（不要 Markdown 代码块）：\n"
        "- target_column: 目标列名称（字符串，必须与数据集列名一致）\n"
        "- task_type: 任务类型，只能是 binary_classification / multiclass_classification / regression\n"
        "- primary_metric: 主要评估指标，可选值包括 accuracy, f1, f1_macro, f1_micro, precision, recall, "
        "roc_auc, log_loss, mcc, root_mean_squared_error, mean_squared_error, mean_absolute_error, r2, auc_pr\n"
        "- time_budget_minutes: 训练时间预算（分钟，0.1-180 之间的数字；null 表示不限制）\n"
        "- max_models: 最多尝试模型数（1-200 之间的整数）\n"
        "- preset: AutoGluon preset，可选 auto / best_quality / high_quality / good_quality / medium_quality / fast_training\n"
        "- confidence: 你对解析结果的置信度（0.0-1.0）\n"
        "- reasoning: 简短解释你的推断依据\n"
        "如果某个字段无法从用户描述中确定，使用 null。不要编造列名。"
    )

    sample_text = ""
    if df_sample is not None and not df_sample.empty:
        preview = df_sample.head(3).to_dict(orient="records")
        columns = df_sample.columns.tolist()
        sample_text = f"\n数据集列名: {columns}\n前 3 行样例: {json.dumps(preview, ensure_ascii=False, default=str)}"

    user_prompt = f"用户描述: {query}{sample_text}\n请返回 JSON:"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_rule_extraction_messages(query: str, df_sample: Optional[pd.DataFrame]) -> List[Dict[str, str]]:
    system_prompt = (
        "你是数据清洗规则提取助手。请从用户描述中提取结构化清洗规则并返回严格 JSON：\n"
        "- drop_columns: 要删除的列名列表\n"
        "- value_constraints: 列表，每项包含 column、min_value、max_value\n"
        "- numeric_impute_strategy: median / mean / constant\n"
        "- numeric_impute_constant: 数值常数\n"
        "- categorical_impute_strategy: mode / constant\n"
        "- categorical_impute_constant: 字符串常数\n"
        "如果某项不存在，使用 null 或空列表。不要 Markdown 代码块。"
    )
    sample_text = ""
    if df_sample is not None and not df_sample.empty:
        sample_text = f"\n数据集列名: {df_sample.columns.tolist()}"
    user_prompt = f"用户描述: {query}{sample_text}\n请返回 JSON:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_schema_inference_messages(df_sample: pd.DataFrame) -> List[Dict[str, str]]:
    system_prompt = (
        "你是 Schema 推断助手。请根据数据集样例推断每个字段的语义并返回严格 JSON：\n"
        "- fields: 列表，每项包含 name、field_type（numeric/categorical/binary/text/datetime/id）、"
        "role（feature/target/id/timestamp）、constraints（min_value/max_value/allowed_values 等）\n"
        "不要 Markdown 代码块。"
    )
    preview = df_sample.head(3).to_dict(orient="records")
    user_prompt = (
        f"列名: {df_sample.columns.tolist()}\n"
        f"前 3 行: {json.dumps(preview, ensure_ascii=False, default=str)}\n"
        "请返回 JSON:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """清洗并解析 LLM 返回的 JSON。"""
    text = raw.strip()
    # 去掉可能的 Markdown 代码围栏
    if text.startswith("```"):
        text = re.sub(r"^```[\w]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return json.loads(text)


def _normalize_rules(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """规范化并兜底 LLM 返回的清洗规则。"""
    rules: Dict[str, Any] = {
        "remove_duplicates": True,
        "drop_rows_with_missing_target": True,
        "numeric_impute_strategy": "median",
        "numeric_impute_constant": 0.0,
        "categorical_impute_strategy": "mode",
        "categorical_impute_constant": "missing",
        "drop_columns": [],
        "value_constraints": [],
    }
    if isinstance(parsed.get("drop_columns"), list):
        rules["drop_columns"] = parsed["drop_columns"]
    for c in parsed.get("value_constraints", []) or []:
        if isinstance(c, dict) and c.get("column"):
            rules["value_constraints"].append(
                {
                    "column": c["column"],
                    "min_value": c.get("min_value"),
                    "max_value": c.get("max_value"),
                }
            )
    if parsed.get("numeric_impute_strategy") in ("median", "mean", "constant"):
        rules["numeric_impute_strategy"] = parsed["numeric_impute_strategy"]
    if parsed.get("categorical_impute_strategy") in ("mode", "constant"):
        rules["categorical_impute_strategy"] = parsed["categorical_impute_strategy"]
    if "numeric_impute_constant" in parsed and parsed["numeric_impute_constant"] is not None:
        rules["numeric_impute_constant"] = float(parsed["numeric_impute_constant"])
    if "categorical_impute_constant" in parsed and parsed["categorical_impute_constant"] is not None:
        rules["categorical_impute_constant"] = str(parsed["categorical_impute_constant"])
    return rules


# ---------------------------------------------------------------------------
# 规则引擎（降级兜底）
# ---------------------------------------------------------------------------


def _rule_based_parse(query: str, df_sample: Optional[pd.DataFrame]) -> IntentConfig:
    """基于关键词与数据集样例推断训练意图。"""
    q = query.lower()

    # 1. 目标列推断
    target_column = _infer_target_column_from_query(query, df_sample)

    # 2. 任务类型推断
    task_type = _infer_task_type(q, target_column, df_sample)

    # 3. 评估指标推断
    primary_metric = _infer_metric(q, task_type)

    # 4. 时间预算与模型数
    time_budget_minutes = _extract_number(q, r"(\d+(?:\.\d+)?)\s*分钟", r"(\d+(?:\.\d+)?)\s*min")
    max_models = int(_extract_number(q, r"(\d+)\s*个模型", r"max_models[=: ]*(\d+)") or 0)

    # 5. preset
    preset = _infer_preset(q)

    reasoning = "规则引擎推断：基于关键词与数据集样例"
    confidence = 0.5

    return IntentConfig(
        target_column=target_column,
        task_type=task_type,
        primary_metric=primary_metric,
        time_budget_minutes=time_budget_minutes if time_budget_minutes else None,
        max_models=max_models if max_models else None,
        preset=preset,
        confidence=confidence,
        provider="rule_engine",
        reasoning=reasoning,
    )


def _rule_based_extract_rules(query: str, df_sample: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """基于规则从自然语言提取清洗规则。"""
    rules = {
        "remove_duplicates": True,
        "drop_rows_with_missing_target": True,
        "numeric_impute_strategy": "median",
        "numeric_impute_constant": 0.0,
        "categorical_impute_strategy": "mode",
        "categorical_impute_constant": "missing",
        "drop_columns": [],
        "value_constraints": [],
        "provider": "rule_engine",
    }

    columns = df_sample.columns.tolist() if df_sample is not None else []

    # 删除列：删除 xxx 列 / 去掉 xxx
    for match in re.finditer(r'''(?:删除|去掉|移除)\s*["']?([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fa5]+)["']?\s*列?''', query):
        col = match.group(1)
        if col in columns:
            rules["drop_columns"].append(col)

    # 取值约束：列名 不能 (>/<) 值，年龄不能大于 120
    constraint_patterns = [
        r'''([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fa5]+?)\s*(?:不能|必须|应)?\s*(?:大于|超过|>)\s*([\d\.]+)''',
        r'''([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fa5]+?)\s*(?:不能|必须|应)?\s*(?:小于|<)\s*([\d\.]+)''',
        r'''([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fa5]+?)\s*(?:在|范围)\s*([\d\.]+)\s*[-~到]\s*([\d\.]+)''',
    ]
    for pat in constraint_patterns:
        for match in re.finditer(pat, query):
            col = match.group(1)
            if col not in columns:
                continue
            constraint = {"column": col, "min_value": None, "max_value": None}
            if ">" in pat or "大于" in pat:
                constraint["max_value"] = float(match.group(2))
            elif "<" in pat or "小于" in pat:
                constraint["min_value"] = float(match.group(2))
            else:
                constraint["min_value"] = float(match.group(2))
                constraint["max_value"] = float(match.group(3))
            rules["value_constraints"].append(constraint)

    # 缺失值填充策略
    if "均值" in query or "mean" in query.lower():
        rules["numeric_impute_strategy"] = "mean"
    if "中位数" in query or "median" in query.lower():
        rules["numeric_impute_strategy"] = "median"
    if re.search(r"用\s*([\d\.]+)\s*填充", query):
        rules["numeric_impute_strategy"] = "constant"
        rules["numeric_impute_constant"] = float(re.search(r"用\s*([\d\.]+)\s*填充", query).group(1))

    return rules


def _infer_target_column_from_query(query: str, df_sample: Optional[pd.DataFrame]) -> Optional[str]:
    """从查询中提取目标列，未提取到时使用启发式规则。"""
    if df_sample is None or df_sample.empty:
        return None

    columns = df_sample.columns.tolist()
    lower_cols = {c.lower(): c for c in columns}

    # 先尝试从引号中匹配列名
    quoted = re.findall(r'["\']([^"\']+)["\']', query)
    for token in quoted:
        if token in columns:
            return token
        if token.lower() in lower_cols:
            return lower_cols[token.lower()]

    # 关键词模式：...列/字段...作为目标/预测/标签
    patterns = [
        r'''(?:目标|标签|预测|target|label|y)[是为]?\s*["']?\s*([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fa5]+)''',
        r'''(?:预测|判断|分类|估计)\s*["']?([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fa5]+)["']?''',
    ]
    for pat in patterns:
        match = re.search(pat, query)
        if match:
            token = match.group(1).strip()
            if token in columns:
                return token
            if token.lower() in lower_cols:
                return lower_cols[token.lower()]

    # 启发式：找名为 target/label/churn/default 等列，否则取最后一列
    for candidate in ["target", "label", "y", "churn", "default", "outcome", "class"]:
        if candidate in lower_cols:
            return lower_cols[candidate]

    return columns[-1]


def _infer_task_type(query: str, target_column: Optional[str], df_sample: Optional[pd.DataFrame]) -> Optional[str]:
    """推断任务类型。"""
    q = query.lower()
    if "回归" in query or "regression" in q:
        return "regression"
    if "二分类" in query or "binary" in q:
        return "binary_classification"
    if "多分类" in query or "multiclass" in q or "多类" in q:
        return "multiclass_classification"

    if df_sample is None or target_column is None or target_column not in df_sample.columns:
        return None

    y = df_sample[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        unique = y.nunique()
        if unique == 2:
            return "binary_classification"
        if unique >= 3:
            return "multiclass_classification"
        return None

    unique = y.nunique()
    if unique <= 10 and (unique / max(len(y), 1)) < 0.05:
        if unique == 2:
            return "binary_classification"
        return "multiclass_classification"
    return "regression"


def _infer_metric(query: str, task_type: Optional[str]) -> Optional[str]:
    """根据查询与任务类型推断主要评估指标。"""
    q = query.lower()
    metric_map = {
        "准确率": "accuracy",
        "accuracy": "accuracy",
        "f1": "f1",
        "auc": "roc_auc",
        "roc_auc": "roc_auc",
        "roc-auc": "roc_auc",
        "auc-pr": "auc_pr",
        "auc_pr": "auc_pr",
        "精确率": "precision",
        "precision": "precision",
        "召回率": "recall",
        "recall": "recall",
        "rmse": "root_mean_squared_error",
        "mse": "mean_squared_error",
        "mae": "mean_absolute_error",
        "r2": "r2",
        "logloss": "log_loss",
        "mcc": "mcc",
    }
    for keyword, metric in metric_map.items():
        if keyword in q:
            return metric

    if task_type == "binary_classification":
        return "f1"
    if task_type == "multiclass_classification":
        return "f1_macro"
    if task_type == "regression":
        return "root_mean_squared_error"
    return None


def _infer_preset(query: str) -> str:
    """推断 preset 偏好。"""
    q = query.lower()
    if "最快" in query or "fast" in q:
        return "fast_training"
    if "最高质量" in query or "best" in q:
        return "best_quality"
    if "高质量" in query or "high" in q:
        return "high_quality"
    if "中等" in query or "medium" in q:
        return "medium_quality"
    return "auto"


def _extract_number(query: str, *patterns: str) -> Optional[float]:
    """从查询中提取数字。"""
    for pat in patterns:
        match = re.search(pat, query)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _merge_with_rule_fallback(parsed: Dict[str, Any], query: str, df_sample: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """用规则引擎补全 LLM 未给出的字段。"""
    fallback = _rule_based_parse(query, df_sample).model_dump()
    merged = dict(fallback)
    for key in [
        "target_column",
        "task_type",
        "primary_metric",
        "time_budget_minutes",
        "max_models",
        "preset",
    ]:
        if parsed.get(key) not in (None, "", 0):
            merged[key] = parsed[key]
    merged["confidence"] = parsed.get("confidence", fallback["confidence"])
    merged["reasoning"] = parsed.get("reasoning") or fallback.get("reasoning")
    if parsed.get("provider"):
        merged["provider"] = parsed["provider"]
    return merged
