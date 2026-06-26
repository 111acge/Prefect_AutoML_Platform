# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 运行时配置管理。

服务器端不持久化保存 API Key。API Key 应通过环境变量 / .env 注入，
服务器仅持久化保存：提供商选择、默认模型。前端传入的 API Key 仅用于
一次性验证，不会被写入数据库或内存缓存。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from sqlalchemy import select

from config import settings
from i18n import _
from database import AsyncSessionLocal
from models import Setting

logger = logging.getLogger(__name__)

# 支持的 LLM 提供商（与前端保持一致）
SUPPORTED_PROVIDERS = ["kimi", "deepseek", "minimax", "glm"]

# 提供商默认配置
PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
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
}

# 内存缓存（启动时从数据库加载，不包含 api_key）
_llm_config_cache: Dict[str, Any] = {
    "provider": None,
    "model": None,
}


def _get_env_api_key(provider: str) -> Optional[str]:
    """读取环境变量或 .env 中配置的 API Key。"""
    cfg = PROVIDER_DEFAULTS.get(provider, {})
    env_key = cfg.get("env_key")
    if not env_key:
        return None
    value = os.environ.get(env_key) or getattr(settings, env_key.lower(), None)
    return str(value) if value else None


async def load_llm_config() -> Dict[str, Any]:
    """从数据库加载 LLM 配置到内存缓存（不含 API Key）。"""
    global _llm_config_cache
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Setting).where(Setting.key.like("llm_%")))
        rows = result.scalars().all()
        db_values = {row.key: row.value for row in rows}

    provider = db_values.get("llm_provider")
    if not provider:
        # 向后兼容：如果环境变量配置了某个 key，则使用该 provider
        for p in SUPPORTED_PROVIDERS:
            if _get_env_api_key(p):
                provider = p
                break

    model = None
    if provider and provider in SUPPORTED_PROVIDERS:
        model = db_values.get(f"llm_{provider}_model")
        if not model:
            model = PROVIDER_DEFAULTS[provider]["default_model"]

    _llm_config_cache = {
        "provider": provider,
        "model": model,
    }
    logger.info(_("llm.config_loaded", provider=provider, model=model))
    return {"provider": provider, "model": model}


async def save_llm_config(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """保存 LLM 配置到数据库（不保存 API Key）。

    Args:
        provider: 提供商，必须是 SUPPORTED_PROVIDERS 之一
        api_key: 仅用于一次性验证，不会被持久化
        model: 模型名称，为空时使用默认模型
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(_("llm.unsupported_provider", provider=provider))

    model = (model or "").strip() or PROVIDER_DEFAULTS[provider]["default_model"]

    async with AsyncSessionLocal() as db:
        keys_to_save = {
            "llm_provider": provider,
            f"llm_{provider}_model": model,
        }
        for key, value in keys_to_save.items():
            result = await db.execute(select(Setting).where(Setting.key == key))
            setting = result.scalar_one_or_none()
            if setting is None:
                setting = Setting(key=key, value=value)
                db.add(setting)
            else:
                setting.value = value
        await db.commit()

    global _llm_config_cache
    _llm_config_cache = {
        "provider": provider,
        "model": model,
    }
    return {"provider": provider, "model": model}


async def clear_llm_config() -> None:
    """清空 LLM 配置（主要用于测试或重置）。"""
    global _llm_config_cache
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Setting).where(Setting.key.like("llm_%")))
        for row in result.scalars().all():
            await db.delete(row)
        await db.commit()
    _llm_config_cache = {"provider": None, "model": None}


def get_llm_config() -> Dict[str, Any]:
    """获取当前内存缓存的 LLM 配置（不含 API Key）。"""
    return _llm_config_cache.copy()


def get_provider_config(provider: str) -> Optional[Dict[str, Any]]:
    """获取指定提供商的默认配置。"""
    return PROVIDER_DEFAULTS.get(provider)


def get_active_api_key() -> Optional[str]:
    """获取当前生效的 API Key（仅来自环境变量 / .env）。"""
    provider = get_active_provider()
    if not provider:
        return None
    return _get_env_api_key(provider)


def get_active_provider() -> Optional[str]:
    """获取当前生效的 LLM 提供商。"""
    return get_llm_config().get("provider")


def get_active_model() -> Optional[str]:
    """获取当前生效的模型名称。"""
    cfg = get_llm_config()
    provider = cfg.get("provider")
    if not provider:
        return None
    return cfg.get("model") or PROVIDER_DEFAULTS.get(provider, {}).get("default_model")


async def init_llm_config() -> None:
    """应用启动时初始化 LLM 配置。"""
    await load_llm_config()
