"""数据集 API 路由。"""

import asyncio
from typing import List, Optional
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
    CleaningRules,
    DatasetQualityResponse,
    DatasetConnectRequest,
)
from services.storage import storage_service
from services.data_quality import assess_data_quality
from services.data_service import load_dataframe, analyze_metadata, preview_dataframe
from services.db_connection_service import (
    load_from_sql,
    build_connection_display_name,
)
from services.schema_service import (
    build_schema_from_file,
    validate_against_schema,
    SchemaValidationError,
)

router = APIRouter(tags=["datasets"])


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    name: str = Form(...),
    target_column: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """上传数据集。

    target_column / task_type 可选；未提供时系统自动推断并写入 schema_info。
    """
    try:
        # 创建数据集记录
        dataset = Dataset(name=name)
        db.add(dataset)
        await db.flush()

        # 保存文件
        file_path = await storage_service.save_upload(dataset.id, file)

        # 加载并分析
        df = load_dataframe(file_path)
        metadata = analyze_metadata(df, target_column=target_column, task_type=task_type)

        # 自动填充目标列/任务类型
        effective_target = target_column or metadata.get("suggested_target_column")
        effective_task = task_type or metadata.get("suggested_task_type")
        if effective_target:
            dataset.target_column = effective_target
        if effective_task:
            dataset.task_type = effective_task

        quality = assess_data_quality(df, target_column=effective_target or "")

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
    """设置数据集目标列和任务类型，并重新生成 Schema 信息。

    当字段为空时，系统会自动推断。
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    df = load_dataframe(dataset.file_path)

    # 若未提供，从当前数据自动推断
    target_column = request.target_column
    task_type = request.task_type
    if target_column is None or task_type is None:
        inferred_meta = analyze_metadata(df, target_column=target_column, task_type=task_type)
        if target_column is None:
            target_column = inferred_meta.get("suggested_target_column")
        if task_type is None:
            task_type = inferred_meta.get("suggested_task_type")

    if not target_column or target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"目标列 '{target_column}' 不存在或无法自动推断",
        )

    # 校验任务类型与目标列是否匹配
    from services.data_service import infer_field_type

    target_type = infer_field_type(df[target_column])
    unique_count = df[target_column].nunique()
    is_numeric = target_type == "numeric"

    if task_type == "binary_classification" and unique_count != 2:
        raise HTTPException(
            status_code=400,
            detail=f"二分类任务要求目标列恰好有 2 个唯一值，当前有 {unique_count} 个",
        )
    if task_type == "multiclass_classification" and unique_count < 3:
        raise HTTPException(
            status_code=400,
            detail=f"多分类任务要求目标列至少有 3 个唯一值，当前有 {unique_count} 个",
        )
    if task_type == "regression" and not is_numeric:
        raise HTTPException(status_code=400, detail="回归任务要求目标列为数值类型")

    dataset.target_column = target_column
    dataset.task_type = task_type

    # 重新生成 schema_info
    metadata = analyze_metadata(df, target_column=target_column, task_type=task_type)
    quality = assess_data_quality(df, target_column=target_column)
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


@router.get("/{dataset_id}/quality", response_model=DatasetQualityResponse)
async def get_dataset_quality(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """获取数据集质量报告（六维模型）。"""
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


@router.get("/{dataset_id}/cleaning-rules", response_model=CleaningRules)
async def get_cleaning_rules(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """获取数据集清洗规则。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    rules = (dataset.schema_info or {}).get("cleaning_rules", {})
    return CleaningRules(**rules)


@router.put("/{dataset_id}/cleaning-rules", response_model=CleaningRules)
async def update_cleaning_rules(
    dataset_id: str,
    request: CleaningRules,
    db: AsyncSession = Depends(get_db),
):
    """更新数据集清洗规则。"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    schema_info = dataset.schema_info or {}
    schema_info["cleaning_rules"] = request.model_dump()
    dataset.schema_info = schema_info

    await db.commit()
    await db.refresh(dataset)
    return request


@router.post("/connect", response_model=DatasetResponse)
async def connect_database(
    request: DatasetConnectRequest,
    db: AsyncSession = Depends(get_db),
):
    """通过数据库连接导入数据集。"""
    try:
        df = await asyncio.to_thread(
            load_from_sql,
            request.connection_type,
            request.connection_params,
            request.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库查询失败: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="查询结果为空")

    try:
        dataset = Dataset(name=request.name or build_connection_display_name(request.connection_type, request.connection_params))
        db.add(dataset)
        await db.flush()

        file_path = storage_service.upload_dir / f"{dataset.id}.csv"
        df.to_csv(file_path, index=False)

        metadata = analyze_metadata(df)
        quality = assess_data_quality(df, target_column="")

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
