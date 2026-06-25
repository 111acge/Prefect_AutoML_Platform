# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""数据库 ORM 模型。"""

import uuid
from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def generate_uuid() -> str:
    """生成 UUID 字符串。"""
    return str(uuid.uuid4())


class Dataset(Base):
    """数据集表。"""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_column: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    task_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    schema_info: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class Experiment(Base):
    """实验表：一次 LLM 驱动的多候选搜索。"""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="running")
    search_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    best_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Trial(Base):
    """实验 Trial 表：记录一次候选运行及其结果。"""

    __tablename__ = "trials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    candidate_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    val_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    primary_metric: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Run(Base):
    """训练任务表。"""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), nullable=False)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    # 默认值由 Pydantic 模式（RunCreate.time_budget_minutes=10）控制，数据库不设置 default，
    # 以便显式传入 None 时能够保存为 NULL（表示训练时间不限制）。
    time_budget_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    primary_metric: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    output_dir: Mapped[str] = mapped_column(String(512), nullable=False)
    prefect_flow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Metric(Base):
    """评估指标表。"""

    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class RunStep(Base):
    """训练任务原子步骤表。"""

    __tablename__ = "run_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    step_name: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_manifest: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_manifest: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class Setting(Base):
    """系统配置表，用于存储 LLM API Key 等运行时配置。"""

    __tablename__ = "settings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
