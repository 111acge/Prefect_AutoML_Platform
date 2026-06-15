"""数据集 API 路由。"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Dataset
from schemas import (
    DatasetResponse,
    DatasetPreview,
    DatasetUpdate,
    DatasetSchemaResponse,
    SchemaValidationResponse,
)
from services.storage import storage_service
from services.data_quality import assess_data_quality
from services.data_service import load_dataframe, analyze_metadata, preview_dataframe
from services.schema_service import (
    build_schema_from_file,
    validate_against_schema,
    SchemaValidationError,
)

router = APIRouter(tags=["datasets"])


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """上传数据集。"""
    try:
        # 创建数据集记录
        dataset = Dataset(name=name)
        db.add(dataset)
        await db.flush()

        # 保存文件
        file_path = await storage_service.save_upload(dataset.id, file)

        # 加载并分析
        df = load_dataframe(file_path)
        metadata = analyze_metadata(df)
        quality = assess_data_quality(df, target_column="")

        # 更新记录
        dataset.file_path = str(file_path)
        dataset.file_size_bytes = file_path.stat().st_size
        dataset.row_count = len(df)
        dataset.column_count = len(df.columns)
        dataset.schema_info = {**metadata, "quality": quality}

        await db.commit()
        await db.refresh(dataset)

        return dataset

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """列出所有数据集。"""
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """获取数据集详情。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """预览数据集。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    try:
        df = load_dataframe(dataset.file_path)
        return preview_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    request: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
):
    """设置数据集目标列和任务类型，并重新生成 Schema 信息。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    df = load_dataframe(dataset.file_path)
    if request.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"目标列 '{request.target_column}' 不存在于数据集中",
        )

    # 校验任务类型与目标列是否匹配
    from services.data_service import infer_field_type

    target_type = infer_field_type(df[request.target_column])
    unique_count = df[request.target_column].nunique()
    is_numeric = target_type == "numeric"

    if request.task_type == "binary_classification" and unique_count != 2:
        raise HTTPException(
            status_code=400,
            detail=f"二分类任务要求目标列恰好有 2 个唯一值，当前有 {unique_count} 个",
        )
    if request.task_type == "multiclass_classification" and unique_count < 3:
        raise HTTPException(
            status_code=400,
            detail=f"多分类任务要求目标列至少有 3 个唯一值，当前有 {unique_count} 个",
        )
    if request.task_type == "regression" and not is_numeric:
        raise HTTPException(status_code=400, detail="回归任务要求目标列为数值类型")

    dataset.target_column = request.target_column
    dataset.task_type = request.task_type

    # 重新生成 schema_info
    metadata = analyze_metadata(df, target_column=request.target_column)
    quality = assess_data_quality(df, target_column=request.target_column)
    dataset.schema_info = {**metadata, "quality": quality}

    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get("/{dataset_id}/schema", response_model=DatasetSchemaResponse)
async def get_dataset_schema(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """获取数据集 Schema。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    try:
        schema = build_schema_from_file(dataset.file_path, target_column=dataset.target_column)
        return schema.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/validate", response_model=SchemaValidationResponse)
async def validate_dataset_schema(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """校验当前数据集是否符合其 Schema。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    try:
        schema = build_schema_from_file(dataset.file_path, target_column=dataset.target_column)
        df = load_dataframe(dataset.file_path)
        errors = validate_against_schema(df, schema)
        return SchemaValidationResponse(valid=len(errors) == 0, errors=errors)
    except SchemaValidationError as e:
        return SchemaValidationResponse(valid=False, errors=e.errors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/quality")
async def get_dataset_quality(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """获取数据集质量报告。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    try:
        df = load_dataframe(dataset.file_path)
        target_column = dataset.target_column or ""
        return assess_data_quality(df, target_column)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """删除数据集。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    try:
        storage_service.delete_dataset_files(dataset_id)
        await db.delete(dataset)
        await db.commit()
        return {"success": True, "message": "数据集已删除"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
