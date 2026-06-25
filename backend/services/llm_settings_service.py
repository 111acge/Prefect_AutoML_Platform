"""LLM 运行时配置管理。

支持将 LLM 提供商选择、API Key、默认模型持久化到数据库，
并在内存中缓存，供 llm_client / llm_intent_service 动态读取。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import select

from config import settings
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

# 内存缓存（启动时从数据库加载）
_llm_config_cache: Dict[str, Any] = {
    "provider": None,
    "api_key": None,
    "model": None,
}


def _cache_key(provider: str, suffix: str) -> str:
    """生成配置在 Setting 表中的 key。"""
    return f"llm_{provider}_{suffix}"


def _get_env_api_key(provider: str) -> Optional[str]:
    """读取环境变量或 .env 中配置的 API Key（向后兼容）。"""
    cfg = PROVIDER_DEFAULTS.get(provider, {})
    env_key = cfg.get("env_key")
    if not env_key:
        return None
    value = settings.config.get(env_key.lower()) if hasattr(settings, "config") else None
    if not value:
        value = getattr(settings, env_key.lower(), None)
    return str(value) if value else None


async def load_llm_config() -> Dict[str, Any]:
    """从数据库加载 LLM 配置到内存缓存。"""
    global _llm_config_cache
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Setting).where(Setting.key.like("llm_%")))
        rows = result.scalars().all()
        db_values = {row.key: row.value for row in rows}

    # 找到已配置的 provider
    provider = db_values.get("llm_provider")
    if not provider:
        # 向后兼容：如果环境变量配置了某个 key，则使用该 provider
        for p in SUPPORTED_PROVIDERS:
            if _get_env_api_key(p):
                provider = p
                break

    api_key = None
    model = None
    if provider and provider in SUPPORTED_PROVIDERS:
        api_key = db_values.get(_cache_key(provider, "api_key"))
        model = db_values.get(_cache_key(provider, "model"))
        if not api_key:
            api_key = _get_env_api_key(provider)
        if not model:
            model = PROVIDER_DEFAULTS[provider]["default_model"]

    _llm_config_cache = {
        "provider": provider,
        "api_key": api_key,
        "model": model,
    }
    logger.info("LLM 配置已加载: provider=%s, model=%s", provider, model)
    return _llm_config_cache.copy()


async def save_llm_config(provider: str, api_key: str, model: Optional[str] = None) -> Dict[str, Any]:
    """保存 LLM 配置到数据库并更新缓存。

    Args:
        provider: 提供商，必须是 SUPPORTED_PROVIDERS 之一
        api_key: API 密钥
        model: 模型名称，为空时使用默认模型
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
    if not api_key or not api_key.strip():
        raise ValueError("API Key 不能为空")

    model = (model or "").strip() or PROVIDER_DEFAULTS[provider]["default_model"]

    async with AsyncSessionLocal() as db:
        keys_to_save = {
            "llm_provider": provider,
            _cache_key(provider, "api_key"): api_key.strip(),
            _cache_key(provider, "model"): model,
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
        "api_key": api_key.strip(),
        "model": model,
    }
    return _llm_config_cache.copy()


async def clear_llm_config() -> None:
    """清空 LLM 配置（主要用于测试或重置）。"""
    global _llm_config_cache
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Setting).where(Setting.key.like("llm_%")))
        for row in result.scalars().all():
            await db.delete(row)
        await db.commit()
    _llm_config_cache = {"provider": None, "api_key": None, "model": None}


def get_llm_config() -> Dict[str, Any]:
    """获取当前内存缓存的 LLM 配置。"""
    return _llm_config_cache.copy()


def get_provider_config(provider: str) -> Optional[Dict[str, Any]]:
    """获取指定提供商的默认配置。"""
    return PROVIDER_DEFAULTS.get(provider)


def get_active_api_key() -> Optional[str]:
    """获取当前生效的 API Key（优先动态配置，其次环境变量）。"""
    cfg = get_llm_config()
    provider = cfg.get("provider")
    if not provider:
        return None
    api_key = cfg.get("api_key")
    if api_key:
        return api_key
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
    """服务启动时调用，加载 LLM 配置到内存。"""
    await load_llm_config()
