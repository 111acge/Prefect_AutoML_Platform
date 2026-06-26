# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""通用 LLM 客户端。

为 Agent 系统（元策略推荐器、错误诊断、特征生成等）提供统一的 LLM 调用入口：
- 多提供商自动 fallback
- 结构化 JSON 输出（带简单修复）
- 可配置 retry / timeout / max_tokens
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from config import settings
from i18n import _
from services.llm_settings_service import (
    get_active_model as _get_dynamic_model,
    get_active_provider as _get_dynamic_provider,
    get_provider_config as _get_provider_config,
    SUPPORTED_PROVIDERS,
)

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIError

    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[misc, assignment]
    APIError = Exception  # type: ignore[misc, assignment]
    _OPENAI_AVAILABLE = False


PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
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
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "env_key": "GLM_API_KEY",
    },
    "openai": {
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}


def _get_api_key(provider: str) -> Optional[str]:
    """读取对应提供商的 API 密钥（仅来自环境变量 / .env，服务器不保存 key）。"""
    cfg = PROVIDER_CONFIG.get(provider, {})
    env_key = cfg.get("env_key")
    if env_key:
        value = os.environ.get(env_key) or getattr(settings, env_key.lower(), None)
        if value:
            return str(value)
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None)
    return None


def resolve_providers(provider: str = "auto", api_key: Optional[str] = None) -> List[str]:
    """根据 provider 参数与可用 key 确定实际尝试的提供商列表。

    Args:
        provider: 指定提供商或 auto。
        api_key: 临时传入的 API Key；传入时优先使用，不再检查环境变量。
    """
    if api_key:
        if provider and provider.lower() != "auto":
            prov = provider.lower()
            if prov in PROVIDER_CONFIG:
                return [prov]
            logger.warning(_("llm.unknown_provider_log", provider=prov))
            return []
        # provider=auto 时，使用当前已配置的提供商（由用户在前端设置）
        active = _get_dynamic_provider()
        if active and active in PROVIDER_CONFIG:
            return [active]
        logger.warning(_("llm.no_active_provider_for_ephemeral_key"))
        return []

    if provider and provider.lower() != "auto":
        prov = provider.lower()
        if prov not in PROVIDER_CONFIG:
            logger.warning(_("llm.unknown_provider_log", provider=prov))
        elif _get_api_key(prov):
            return [prov]
        else:
            logger.warning(_("llm.provider_not_configured_log", provider=prov))

    order = ["kimi", "deepseek", "minimax", "glm", "openai"]
    return [p for p in order if _get_api_key(p)]


def _extract_json(text: str) -> Optional[str]:
    """从 LLM 返回文本中提取 JSON 片段（支持 Markdown 代码块）。"""
    text = text.strip()
    # 先尝试整个文本
    if text.startswith("{") and text.endswith("}"):
        return text
    # 代码块
    code_block = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1)
    # 第一个 { ... } 块
    match = re.search(r"({.*})", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


async def call_provider(
    provider: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = 4096,
    temperature: float = 0.0,
    timeout: Optional[float] = 60.0,
    api_key: Optional[str] = None,
) -> str:
    """调用指定 OpenAI 兼容端点并返回原始文本。

    Args:
        api_key: 临时传入的 API Key；为空时从环境变量读取。
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError(_("llm.openai_not_installed"))

    cfg = PROVIDER_CONFIG[provider]
    effective_api_key = (api_key or _get_api_key(provider) or "").strip()
    if not effective_api_key:
        raise RuntimeError(_("llm.provider_not_configured", provider=provider))
    try:
        effective_api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(_("llm.api_key_non_ascii")) from exc

    client_kwargs: Dict[str, Any] = {"api_key": effective_api_key}
    if cfg.get("base_url"):
        client_kwargs["base_url"] = cfg["base_url"]

    client = AsyncOpenAI(**client_kwargs)
    # 优先使用用户配置的模型；其次使用环境变量覆盖；最后使用默认模型
    model = _get_dynamic_model() if _get_dynamic_provider() == provider else None
    if not model:
        model = getattr(settings, "default_llm_model", None) or cfg["default_model"]

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens

    if timeout is None:
        response = await client.chat.completions.create(**create_kwargs)
    else:
        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=timeout,
        )
    return response.choices[0].message.content or ""


async def call_llm_for_json(
    messages: List[Dict[str, str]],
    *,
    provider: str = "auto",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: float = 60.0,
    retries: int = 2,
) -> Dict[str, Any]:
    """调用 LLM 并解析返回为 JSON 字典。

    若一个提供商失败，自动尝试下一个；解析失败时会重试同一提供商。
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError(_("llm.openai_not_installed"))

    providers = resolve_providers(provider)
    if not providers:
        raise RuntimeError(_("llm.no_provider_configured"))

    last_error: Optional[Exception] = None
    for prov in providers:
        for attempt in range(retries + 1):
            try:
                raw = await call_provider(
                    prov,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                json_str = _extract_json(raw)
                if json_str is None:
                    raise ValueError(_("llm.json_extraction_failed", raw=raw[:200]))
                return json.loads(json_str)
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning(
                    _("llm.call_parse_failed", prov=prov, attempt=attempt + 1, msg=e)
                )
                if attempt < retries:
                    await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(_("llm.all_providers_failed", last_error=last_error))
