# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 配置管理 API。

服务器不持久化保存 API Key。API Key 必须通过环境变量 / .env 注入。
前端传入的 API Key 仅用于一次性验证，不会被写入数据库或内存缓存。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.llm_settings_service import (
    SUPPORTED_PROVIDERS,
    get_active_api_key,
    get_llm_config,
    get_provider_config,
    save_llm_config,
    PROVIDER_DEFAULTS,
)

router = APIRouter(tags=["settings"])


class LLMConfigRequest(BaseModel):
    """保存 LLM 配置请求。

    api_key 仅用于一次性验证，服务器不会保留。
    """

    provider: str = Field(..., pattern="^(kimi|deepseek|minimax|glm)$")
    api_key: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)


class LLMConfigResponse(BaseModel):
    """LLM 配置响应（不包含 API Key 任何信息）。"""

    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_configured: bool = False
    supported_providers: list[str] = []


class ProviderInfoResponse(BaseModel):
    """提供商信息响应。"""

    providers: Dict[str, Dict[str, Any]]


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_settings():
    """获取当前 LLM 配置。API Key 是否配置仅通过环境变量判断。"""
    cfg = get_llm_config()
    return LLMConfigResponse(
        provider=cfg.get("provider"),
        model=cfg.get("model"),
        api_key_configured=get_active_api_key() is not None,
        supported_providers=SUPPORTED_PROVIDERS,
    )


@router.post("/llm", response_model=LLMConfigResponse)
async def save_llm_settings(request: LLMConfigRequest):
    """保存 LLM 配置（仅保存 provider / model，不保存 API Key）。"""
    try:
        await save_llm_config(
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cfg = get_llm_config()
    return LLMConfigResponse(
        provider=cfg.get("provider"),
        model=cfg.get("model"),
        api_key_configured=get_active_api_key() is not None,
        supported_providers=SUPPORTED_PROVIDERS,
    )


@router.get("/llm/providers", response_model=ProviderInfoResponse)
async def list_llm_providers():
    """列出支持的 LLM 提供商及其默认配置。"""
    return ProviderInfoResponse(providers=PROVIDER_DEFAULTS)
