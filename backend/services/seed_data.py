# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""默认数据集初始化服务。"""

from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import AsyncSessionLocal
from models import Dataset
from services.data_service import load_dataframe, analyze_metadata

DEFAULT_DATASET_NAME = "iris"
DEFAULT_DATASET_FILE = "iris.csv"
DEFAULT_TARGET_COLUMN = "target"


def get_default_dataset_path() -> Path:
    """获取默认数据集路径。"""
    return settings.data_dir / "default" / DEFAULT_DATASET_FILE


async def ensure_default_dataset(db: Optional[AsyncSession] = None) -> Optional[Dataset]:
    """确保默认数据集已加载到数据库。"""
    dataset_path = get_default_dataset_path()
    if not dataset_path.exists():
        return None

    own_session = db is None
    if own_session:
        db = AsyncSessionLocal()

    try:
        result = await db.execute(select(Dataset).where(Dataset.name == DEFAULT_DATASET_NAME))
        dataset = result.scalar_one_or_none()
        if dataset:
            return dataset

        df = load_dataframe(dataset_path)
        metadata = analyze_metadata(df, target_column=DEFAULT_TARGET_COLUMN)

        dataset = Dataset(
            name=DEFAULT_DATASET_NAME,
            file_path=str(dataset_path),
            file_size_bytes=dataset_path.stat().st_size,
            row_count=len(df),
            column_count=len(df.columns),
            target_column=DEFAULT_TARGET_COLUMN,
            task_type="multiclass_classification",
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
