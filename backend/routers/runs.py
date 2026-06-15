"""训练任务 API 路由。"""

import asyncio
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from models import Dataset, Run
from schemas import RunCreate, RunResponse, RunResult, PredictionRequest, PredictionResponse
from services.data_service import load_dataframe
from services.storage import storage_service
from services.training_executor import training_executor
from services.schema_service import build_schema_from_file, validate_against_schema

router = APIRouter(tags=["runs"])


def _get_output_dir(run_id: str) -> Path:
    """获取任务输出目录。"""
    output_dir = settings.report_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@router.post("", response_model=RunResponse)
async def create_run(
    request: RunCreate,
    db: AsyncSession = Depends(get_db),
):
    """启动训练任务。"""
    # 检查数据集是否存在
    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    # 校验任务类型与目标列是否匹配
    try:
        await asyncio.to_thread(
            _validate_task_type,
            dataset.file_path,
            request.target_column,
            request.task_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Schema 校验
    try:
        schema = await asyncio.to_thread(
            build_schema_from_file,
            dataset.file_path,
            request.target_column,
        )
        errors = await asyncio.to_thread(
            validate_against_schema,
            load_dataframe(dataset.file_path),
            schema,
        )
        if errors:
            raise HTTPException(
                status_code=400,
                detail=f"Schema 校验失败: {'; '.join(errors)}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema 校验异常: {str(e)}")

    try:
        output_dir = _get_output_dir("temp")
        run = Run(
            dataset_id=request.dataset_id,
            status="pending",
            time_budget_minutes=request.time_budget_minutes,
            primary_metric=request.primary_metric,
            output_dir=str(output_dir),
            config={
                "target_column": request.target_column,
                "task_type": request.task_type,
                "preset": request.preset,
                "max_models": request.max_models,
                "seed": request.seed,
            },
        )
        db.add(run)
        await db.flush()

        # 重新设置输出目录
        output_dir = _get_output_dir(run.id)
        run.output_dir = str(output_dir)
        await db.commit()
        await db.refresh(run)

        # 提交到异步训练执行器，避免阻塞 FastAPI worker
        await training_executor.submit(
            run_id=run.id,
            file_path=dataset.file_path,
            target_column=request.target_column,
            task_type=request.task_type,
            output_dir=str(output_dir),
            time_budget_minutes=request.time_budget_minutes,
            preset=request.preset,
            primary_metric=request.primary_metric,
            seed=request.seed,
            max_models=request.max_models,
        )

        return run

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


def _validate_task_type(file_path: str, target_column: str, task_type: str) -> None:
    """校验任务类型与目标列是否匹配。"""
    import pandas as pd

    df = load_dataframe(file_path)
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在")

    y = df[target_column]
    unique_count = y.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(y)

    if task_type == "binary_classification" and unique_count != 2:
        raise ValueError(f"二分类任务要求目标列恰好有 2 个唯一值，当前有 {unique_count} 个")
    if task_type == "multiclass_classification" and unique_count < 3:
        raise ValueError(f"多分类任务要求目标列至少有 3 个唯一值，当前有 {unique_count} 个")
    if task_type == "regression" and not is_numeric:
        raise ValueError("回归任务要求目标列为数值类型")


@router.get("", response_model=List[RunResponse])
async def list_runs(db: AsyncSession = Depends(get_db)):
    """列出训练任务。"""
    result = await db.execute(select(Run).order_by(Run.created_at.desc()))
    runs = result.scalars().all()
    return runs


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """获取任务详情。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    return run


@router.get("/{run_id}/results", response_model=RunResult)
async def get_run_results(run_id: str, db: AsyncSession = Depends(get_db)):
    """获取训练结果。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    try:
        output_dir = Path(run.output_dir)

        # 读取指标
        metrics = {}
        extended_metrics = None
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                final = metrics_data.get("final", {})
                metrics = {
                    str(k): float(v) for k, v in final.items() if isinstance(v, (int, float))
                }
                extended_metrics = metrics_data.get("extended")

        # 读取排行榜
        leaderboard = []
        leaderboard_path = output_dir / "leaderboard.csv"
        if leaderboard_path.exists():
            import pandas as pd

            leaderboard = pd.read_csv(leaderboard_path).head(10).to_dict(orient="records")

        # 读取特征重要性
        feature_importance = []
        importance_path = output_dir / "feature_importance.csv"
        if importance_path.exists():
            import pandas as pd

            feature_importance = pd.read_csv(importance_path).head(20).to_dict(orient="records")

        return RunResult(
            run_id=run_id,
            status=run.status,
            metrics=metrics,
            extended_metrics=extended_metrics,
            leaderboard=leaderboard,
            feature_importance=feature_importance,
            model_path=(
                str(output_dir / "autogluon_models")
                if (output_dir / "autogluon_models").exists()
                else None
            ),
            report_path=(
                str(output_dir / "report.html") if (output_dir / "report.html").exists() else None
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/predict", response_model=PredictionResponse)
async def predict(run_id: str, request: PredictionRequest, db: AsyncSession = Depends(get_db)):
    """使用训练好的模型预测。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    try:
        import pandas as pd
        from services.automl import load_predictor
        from services.preprocessing_pipeline import (
            DataPreprocessor,
            load_feature_columns,
            validate_prediction_input,
        )

        output_dir = Path(run.output_dir)
        model_path = output_dir / "autogluon_models"
        predictor = load_predictor(model_path)

        df = pd.DataFrame(request.data)

        # 加载并应用预处理 Pipeline
        preprocessor_path = output_dir / "preprocessing_pipeline.joblib"
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor.load(preprocessor_path)
            errors = validate_prediction_input(preprocessor, df)
            if errors:
                raise HTTPException(status_code=400, detail="; ".join(errors))
            df = preprocessor.transform(df)

        # 确保特征列顺序与训练时一致
        feature_columns_path = output_dir / "feature_columns.json"
        if feature_columns_path.exists():
            feature_columns = load_feature_columns(output_dir)
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise HTTPException(
                    status_code=400, detail=f"输入缺少特征列: {sorted(missing_cols)}"
                )
            # 只使用训练时的特征列（AutoGluon 需要一致的列）
            df = df[feature_columns]

        preds = predictor.predict(df)

        response = PredictionResponse(predictions=preds.tolist())

        if predictor.problem_type in ["binary", "multiclass"]:
            try:
                probs = predictor.predict_proba(df)
                response.probabilities = probs.to_dict(orient="records")
            except Exception:
                pass

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/report")
async def download_report(run_id: str, db: AsyncSession = Depends(get_db)):
    """下载 HTML 报告。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    report_path = Path(run.output_dir) / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")

    return FileResponse(
        str(report_path),
        media_type="text/html",
        filename=f"report_{run_id}.html",
    )


@router.get("/{run_id}/model")
async def download_model(run_id: str, db: AsyncSession = Depends(get_db)):
    """下载训练好的模型（zip 压缩包）。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    output_dir = Path(run.output_dir)
    model_dir = output_dir / "autogluon_models"
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    import zipfile

    zip_path = output_dir / "model.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # AutoGluon 模型
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(model_dir))
        # 预处理 Pipeline
        pipeline_path = output_dir / "preprocessing_pipeline.joblib"
        if pipeline_path.exists():
            zf.write(pipeline_path, "preprocessing_pipeline.joblib")
        # 特征列清单
        feature_columns_path = output_dir / "feature_columns.json"
        if feature_columns_path.exists():
            zf.write(feature_columns_path, "feature_columns.json")

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"model_{run_id}.zip",
    )


@router.get("/{run_id}/logs")
async def get_run_logs(run_id: str, db: AsyncSession = Depends(get_db)):
    """获取训练日志。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    log_path = Path(run.output_dir) / "training.log"
    if not log_path.exists():
        return PlainTextResponse("日志尚未生成\n")

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)


@router.delete("/{run_id}")
async def delete_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """删除训练任务。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    try:
        storage_service.delete_run_files(run_id)
        await db.delete(run)
        await db.commit()
        return {"success": True, "message": "任务已删除"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
