# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""意图解析 API 路由。"""

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Dataset
from schemas import CleaningRules
from services.data_service import load_dataframe
from services.llm_intent_service import (
    parse_intent,
    extract_business_rules,
    infer_schema_with_llm,
    IntentConfig,
)

router = APIRouter(tags=["intent"])


class IntentParseRequest(BaseModel):
    """意图解析请求。"""

    query: str
    dataset_id: Optional[str] = None
    provider: Optional[str] = "auto"


class RuleExtractionRequest(BaseModel):
    """规则提取请求。"""

    query: str
    dataset_id: Optional[str] = None
    provider: Optional[str] = "auto"


class SchemaInferenceRequest(BaseModel):
    """Schema 推断请求。"""

    dataset_id: str
    provider: Optional[str] = "auto"


@router.post("/parse", response_model=IntentConfig)
async def parse_intent_endpoint(
    request: IntentParseRequest,
    db: AsyncSession = Depends(get_db),
):
    """将自然语言描述解析为训练配置。

    - 若提供 dataset_id，会读取数据集样例辅助解析。
    - 支持 provider 指定 LLM 提供商；未配置/失败时自动降级到规则引擎。
    """
    df_sample: Optional[pd.DataFrame] = None
    if request.dataset_id:
        result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(status_code=404, detail="数据集不存在")
        if dataset.file_path:
            try:
                df_sample = load_dataframe(dataset.file_path).head(5)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"读取数据集样例失败: {str(e)}",
                )

    parsed = await parse_intent(
        query=request.query,
        df_sample=df_sample,
        provider=request.provider or "auto",
    )
    return parsed


@router.post("/rules", response_model=CleaningRules)
async def extract_rules_endpoint(
    request: RuleExtractionRequest,
    db: AsyncSession = Depends(get_db),
):
    """从自然语言中提取数据清洗规则。

    例如："年龄不能大于 120，删除 ID 列，缺失值用 0 填充"。
    """
    df_sample: Optional[pd.DataFrame] = None
    if request.dataset_id:
        result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(status_code=404, detail="数据集不存在")
        if dataset.file_path:
            try:
                df_sample = load_dataframe(dataset.file_path).head(5)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"读取数据集样例失败: {str(e)}")

    rules = await extract_business_rules(
        query=request.query,
        df_sample=df_sample,
        provider=request.provider or "auto",
    )
    return CleaningRules(**rules)


@router.post("/schema", response_model=List[Dict[str, Any]])
async def infer_schema_endpoint(
    request: SchemaInferenceRequest,
    db: AsyncSession = Depends(get_db),
):
    """LLM 辅助 Schema 推断，失败时返回规则推断结果。"""
    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if not dataset.file_path:
        raise HTTPException(status_code=500, detail="数据集没有文件路径")

    try:
        df_sample = load_dataframe(dataset.file_path).head(10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取数据集失败: {str(e)}")

    return await infer_schema_with_llm(
        df_sample=df_sample,
        provider=request.provider or "auto",
    )
