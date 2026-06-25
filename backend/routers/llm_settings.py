"""LLM 配置管理 API。

支持前端动态配置 LLM 提供商、API Key 和模型。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.llm_settings_service import (
    SUPPORTED_PROVIDERS,
    get_llm_config,
    get_provider_config,
    save_llm_config,
    PROVIDER_DEFAULTS,
)

router = APIRouter(tags=["settings"])


class LLMConfigRequest(BaseModel):
    """保存 LLM 配置请求。"""

    provider: str = Field(..., pattern="^(kimi|deepseek|minimax|glm)$")
    api_key: str = Field(..., min_length=1)
    model: Optional[str] = Field(default=None)


class LLMConfigResponse(BaseModel):
    """LLM 配置响应（不包含 API Key 完整值）。"""

    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_masked: Optional[str] = None
    supported_providers: list[str] = []


class ProviderInfoResponse(BaseModel):
    """提供商信息响应。"""

    providers: Dict[str, Dict[str, Any]]


def _mask_api_key(key: Optional[str]) -> Optional[str]:
    """对 API Key 做掩码展示。"""
    if not key:
        return None
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "****" + key[-4:]


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_settings():
    """获取当前 LLM 配置（API Key 已掩码）。"""
    cfg = get_llm_config()
    return LLMConfigResponse(
        provider=cfg.get("provider"),
        model=cfg.get("model"),
        api_key_masked=_mask_api_key(cfg.get("api_key")),
        supported_providers=SUPPORTED_PROVIDERS,
    )


@router.post("/llm", response_model=LLMConfigResponse)
async def save_llm_settings(request: LLMConfigRequest):
    """保存 LLM 配置。"""
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
        api_key_masked=_mask_api_key(cfg.get("api_key")),
        supported_providers=SUPPORTED_PROVIDERS,
    )


@router.get("/llm/providers", response_model=ProviderInfoResponse)
async def list_llm_providers():
    """列出支持的 LLM 提供商及其默认配置。"""
    return ProviderInfoResponse(providers=PROVIDER_DEFAULTS)
