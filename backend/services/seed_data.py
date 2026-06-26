# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""默认数据集初始化服务。"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import AsyncSessionLocal
from models import Dataset
from services.data_service import load_dataframe, analyze_metadata


DEFAULT_DATASETS: List[Dict[str, Any]] = [
    {
        "name": "iris",
        "file": "iris.csv",
        "target_column": "target",
        "task_type": "multiclass_classification",
    },
    {
        "name": "temperature",
        "file": "temperature.csv",
        "target_column": "temperature",
        "task_type": "regression",
    },
]


def get_default_dataset_path(file_name: str) -> Path:
    """获取默认数据集路径。"""
    return settings.data_dir / "default" / file_name


def get_default_dataset_names() -> List[str]:
    """获取所有默认数据集名称。"""
    return [d["name"] for d in DEFAULT_DATASETS]


def get_default_dataset_config(name: str) -> Optional[Dict[str, Any]]:
    """根据名称获取默认数据集配置。"""
    for config in DEFAULT_DATASETS:
        if config["name"] == name:
            return config
    return None


def get_default_dataset_path_by_name(name: str) -> Optional[Path]:
    """根据名称获取默认数据集文件路径。"""
    config = get_default_dataset_config(name)
    if config is None:
        return None
    return get_default_dataset_path(config["file"])


async def ensure_default_dataset(
    name: str = "iris",
    db: Optional[AsyncSession] = None,
) -> Optional[Dataset]:
    """确保指定默认数据集已加载到数据库。

    Args:
        name: 默认数据集名称，如 "iris" 或 "temperature"。
        db: 可选的数据库会话。

    Returns:
        已存在或新创建的 Dataset 对象；如果文件不存在则返回 None。
    """
    config = get_default_dataset_config(name)
    if config is None:
        return None

    dataset_path = get_default_dataset_path(config["file"])
    if not dataset_path.exists():
        return None

    own_session = db is None
    if own_session:
        db = AsyncSessionLocal()

    try:
        result = await db.execute(select(Dataset).where(Dataset.name == config["name"]))
        dataset = result.scalar_one_or_none()
        if dataset:
            return dataset

        df = load_dataframe(dataset_path)
        metadata = analyze_metadata(df, target_column=config["target_column"])

        dataset = Dataset(
            name=config["name"],
            file_path=str(dataset_path),
            file_size_bytes=dataset_path.stat().st_size,
            row_count=len(df),
            column_count=len(df.columns),
            target_column=config["target_column"],
            task_type=config["task_type"],
            schema_info=metadata,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset
    except Exception:
        await db.rollback()
        raise
    finally:
        if own_session:
            await db.close()


async def ensure_all_default_datasets(db: Optional[AsyncSession] = None) -> List[Dataset]:
    """确保所有默认数据集都已加载到数据库。"""
    results: List[Dataset] = []
    own_session = db is None
    if own_session:
        db = AsyncSessionLocal()
    try:
        for config in DEFAULT_DATASETS:
            dataset = await ensure_default_dataset(config["name"], db=db)
            if dataset is not None:
                results.append(dataset)
        return results
    finally:
        if own_session:
            await db.close()
