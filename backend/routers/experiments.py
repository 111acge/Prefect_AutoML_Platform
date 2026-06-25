# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""实验（Experiment）API 路由。"""

import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Dataset, Experiment, Trial
from schemas import ExperimentCreate, ExperimentResponse, TrialResponse
from services.data_service import infer_target_column, infer_task_type
from services.search_agent import run_experiment

router = APIRouter(tags=["experiments"])


def _resolve_target_task(dataset: Dataset, request: ExperimentCreate):
    """解析/推断目标列与任务类型。"""
    target_column = request.target_column or dataset.target_column
    task_type = request.task_type or dataset.task_type

    schema_info = dataset.schema_info or {}
    if not target_column:
        target_column = schema_info.get("suggested_target_column")
    if not task_type:
        task_type = schema_info.get("suggested_task_type")

    return target_column, task_type


@router.post("", response_model=ExperimentResponse, status_code=202)
async def create_experiment(
    request: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """创建并启动一个 LLM 驱动的多候选搜索实验。

    实验在后台运行，可通过 GET /experiments/{id} 查看状态。
    """
    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    target_column, task_type = _resolve_target_task(dataset, request)
    if not target_column:
        raise HTTPException(status_code=400, detail="无法确定目标列")
    if not task_type:
        raise HTTPException(status_code=400, detail="无法确定任务类型")

    # 创建实验记录
    experiment = Experiment(
        dataset_id=request.dataset_id,
        search_config={
            "target_column": target_column,
            "task_type": task_type,
            "primary_metric": request.primary_metric,
            "max_iterations": request.max_iterations,
            "trials_per_iteration": request.trials_per_iteration,
            "time_budget_minutes": request.time_budget_minutes,
            "rare_class_strategy": request.rare_class_strategy,
        },
        status="running",
    )
    db.add(experiment)
    await db.flush()
    await db.commit()
    await db.refresh(experiment)

    # 后台启动搜索
    asyncio.create_task(
        run_experiment(
            dataset_id=request.dataset_id,
            target_column=target_column,
            task_type=task_type,
            primary_metric=request.primary_metric,
            max_iterations=request.max_iterations,
            trials_per_iteration=request.trials_per_iteration,
            time_budget_minutes=request.time_budget_minutes,
            rare_class_strategy=request.rare_class_strategy,
            experiment_id=experiment.id,
        )
    )

    return experiment


@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    """列出所有实验。"""
    result = await db.execute(select(Experiment).order_by(Experiment.created_at.desc()))
    return result.scalars().all()


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """获取实验详情。"""
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")
    return experiment


@router.get("/{experiment_id}/trials", response_model=List[TrialResponse])
async def list_trials(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """获取实验的所有 Trial。"""
    result = await db.execute(
        select(Trial).where(Trial.experiment_id == experiment_id).order_by(Trial.created_at.asc())
    )
    return result.scalars().all()


@router.get("/{experiment_id}/best-run")
async def get_best_run(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """返回实验最佳 Run 的 ID。"""
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")
    if not experiment.best_run_id:
        raise HTTPException(status_code=400, detail="实验尚未产生最佳 Run")
    return {"experiment_id": experiment_id, "best_run_id": experiment.best_run_id}
