"""训练任务 API 路由。"""

import asyncio
import hashlib
import json
import logging
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from models import Dataset, Run, RunStep

logger = logging.getLogger(__name__)
from schemas import (
    RunCreate,
    RunResponse,
    RunResult,
    RunStepResponse,
    StepExecutionRequest,
    PredictionRequest,
    PredictionResponse,
    ExplainRequest,
    ExplainResponse,
    RunCompareRequest,
    RunCompareResponse,
    RunCompareItem,
)
from services.data_service import load_dataframe
from services.storage import storage_service
from services.step_manifest import StepManifest
from services.step_runner import STEP_ORDER, initialize_run_steps
from services.training_executor import training_executor
from services.schema_service import build_schema_from_file, validate_against_schema

router = APIRouter(tags=["runs"])


def _get_output_dir(run_id: str) -> Path:
    """获取任务输出目录。"""
    output_dir = settings.report_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _get_id_columns_to_drop(schema_info: dict | None) -> list[str]:
    """从 schema_info 中提取 ID-like 列，训练前自动剔除。"""
    if not schema_info:
        return []
    field_types = schema_info.get("field_types", {})
    return [col for col, ft in field_types.items() if ft == "id"]


def _load_optimal_threshold(output_dir: Path) -> Optional[float]:
    """从 metrics.json 中加载二分类最优阈值。"""
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        threshold_info = metrics.get("threshold", {})
        optimal = threshold_info.get("optimal_threshold")
        if optimal is not None and isinstance(optimal, (int, float)):
            return float(optimal)
    except Exception:
        pass
    return None


def _predict_with_threshold(
    predictor, df: pd.DataFrame, threshold: Optional[float]
) -> pd.Series:
    """预测，支持使用自定义二分类阈值。"""
    if predictor.problem_type != "binary" or threshold is None:
        return predictor.predict(df)

    try:
        proba = predictor.predict_proba(df)
        pos_label = predictor.class_labels[1]
        neg_label = predictor.class_labels[0]
        preds = (proba[pos_label] >= threshold).astype(int)
        # 映射回原始标签
        label_map = {0: neg_label, 1: pos_label}
        preds = preds.map(label_map)
        preds.index = df.index
        return preds
    except Exception:
        return predictor.predict(df)


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

    # 自动填充目标列和任务类型
    target_column = request.target_column or dataset.target_column
    task_type = request.task_type or dataset.task_type

    # 若数据集上仍未设置，尝试从 schema_info 推断
    schema_info = dataset.schema_info or {}
    if not target_column:
        target_column = schema_info.get("suggested_target_column")
    if not task_type:
        task_type = schema_info.get("suggested_task_type")

    if not target_column:
        raise HTTPException(status_code=400, detail="无法确定目标列，请在上传数据集时指定或更新数据集")
    if not task_type:
        raise HTTPException(status_code=400, detail="无法确定任务类型，请在上传数据集时指定或更新数据集")

    # 校验任务类型与目标列是否匹配
    try:
        await asyncio.to_thread(
            _validate_task_type,
            dataset.file_path,
            target_column,
            task_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Schema 校验
    try:
        schema = await asyncio.to_thread(
            build_schema_from_file,
            dataset.file_path,
            target_column,
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
            experiment_id=request.experiment_id,
            status="pending",
            time_budget_minutes=request.time_budget_minutes,
            primary_metric=request.primary_metric,
            output_dir=str(output_dir),
            config={
                "target_column": target_column,
                "task_type": task_type,
                "preset": request.preset,
                "max_models": request.max_models,
                "seed": request.seed,
                "feature_engineering_enabled": request.feature_engineering_enabled,
                "experiment_id": request.experiment_id,
                "candidate_config": request.candidate_config.model_dump() if request.candidate_config else None,
            },
        )
        db.add(run)
        await db.flush()

        # 重新设置输出目录
        output_dir = _get_output_dir(run.id)
        run.output_dir = str(output_dir)

        # 生成配置快照
        # 构建一个与 request 等价的快照对象用于保存
        snapshot_request = RunCreate(
            dataset_id=request.dataset_id,
            target_column=target_column,
            task_type=task_type,
            primary_metric=request.primary_metric,
            time_budget_minutes=request.time_budget_minutes,
            max_models=request.max_models,
            preset=request.preset,
            seed=request.seed,
            feature_engineering_enabled=request.feature_engineering_enabled,
            experiment_id=request.experiment_id,
            candidate_config=request.candidate_config,
        )
        snapshot = _build_config_snapshot(dataset, snapshot_request)
        snapshot_path = output_dir / "config_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        run.config = {**run.config, "snapshot": snapshot}

        await db.commit()
        await db.refresh(run)

        # 读取数据集配置的清洗规则，并自动加入 ID-like 列剔除
        cleaning_rules = (dataset.schema_info or {}).get("cleaning_rules") or {}
        id_columns = _get_id_columns_to_drop(dataset.schema_info)
        if id_columns:
            existing_drops = set(cleaning_rules.get("drop_columns", []))
            existing_drops.update(id_columns)
            cleaning_rules["drop_columns"] = list(existing_drops)
            logger.info(f"运行 {run.id} 自动剔除 ID-like 列: {id_columns}")

        # 初始化运行上下文（step 模式需要；auto 模式由 ingest 步骤创建）
        run_context = {
            "target_column": target_column,
            "task_type": task_type,
            "seed": request.seed,
            "file_path": dataset.file_path,
        }

        if request.mode == "step":
            # 仅创建草稿与步骤记录，不启动训练
            await initialize_run_steps(run.id, str(output_dir), run_context)
            run.status = "pending"
            await db.commit()
            return run

        # 自动模式：顺序执行所有步骤
        training_executor.submit_sync(
            run_id=run.id,
            file_path=dataset.file_path,
            target_column=target_column,
            task_type=task_type,
            output_dir=str(output_dir),
            time_budget_minutes=request.time_budget_minutes,
            preset=request.preset,
            primary_metric=request.primary_metric,
            seed=request.seed,
            max_models=request.max_models,
            cleaning_rules=cleaning_rules,
            feature_engineering_enabled=request.feature_engineering_enabled,
            candidate_config=(
                request.candidate_config.model_dump() if request.candidate_config else None
            ),
            step="all",
        )

        return run

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


def _build_config_snapshot(dataset: Dataset, request: RunCreate) -> Dict[str, Any]:
    """构建训练任务配置快照，保证可复现。"""
    file_hash = ""
    if dataset.file_path:
        try:
            with open(dataset.file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            pass

    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "dataset_file_hash": file_hash,
        "dataset_schema_info": dataset.schema_info,
        "target_column": request.target_column,
        "task_type": request.task_type,
        "preset": request.preset,
        "max_models": request.max_models,
        "primary_metric": request.primary_metric,
        "seed": request.seed,
        "time_budget_minutes": request.time_budget_minutes,
        "dataset_file_path": dataset.file_path,
    }


def _validate_task_type(file_path: str, target_column: str, task_type: str) -> None:
    """校验任务类型与目标列是否匹配，并拦截退化数据集。"""
    import pandas as pd

    df = load_dataframe(file_path)
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在")

    # 退化数据集校验
    feature_cols = [c for c in df.columns if c != target_column]
    if len(df) < 10:
        raise ValueError(f"样本数过少，至少需要 10 行，当前 {len(df)} 行")
    if len(feature_cols) < 1:
        raise ValueError("至少需要 1 个特征列")

    y = df[target_column]

    if y.isnull().all():
        raise ValueError("目标列全部为缺失值，无法训练")

    y_dropped = y.dropna()
    unique_count = y_dropped.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(y)

    if unique_count <= 1:
        raise ValueError(f"目标列唯一值数量不足，当前 {unique_count} 个")

    if task_type == "binary_classification" and unique_count != 2:
        raise ValueError(f"二分类任务要求目标列恰好有 2 个唯一值，当前有 {unique_count} 个")
    if task_type == "multiclass_classification" and unique_count < 2:
        raise ValueError(f"多分类任务要求目标列至少有 2 个唯一值，当前有 {unique_count} 个")
    if task_type == "regression" and not is_numeric:
        raise ValueError("回归任务要求目标列为数值类型")

    # 分类任务中若最小类样本数极少，不再直接拒绝训练。
    # 下游 split_data / AutoGluon 会通过 rare_class_strategy 自动过采样或限制折数。
    if task_type in ("binary_classification", "multiclass_classification"):
        min_class_count = y_dropped.value_counts().min()
        if min_class_count < 2:
            logger.warning(
                f"目标列存在样本数 < 2 的类别（最小类={min_class_count}），"
                f"将启用稀有类别自动处理策略（默认过采样到 2 条）。"
            )


@router.get("", response_model=List[RunResponse])
async def list_runs(db: AsyncSession = Depends(get_db)):
    """列出训练任务。"""
    result = await db.execute(select(Run).order_by(Run.created_at.desc()))
    runs = result.scalars().all()
    return runs


@router.post("/compare", response_model=RunCompareResponse)
async def compare_runs(request: RunCompareRequest, db: AsyncSession = Depends(get_db)):
    """对比多个已完成的训练任务。"""
    result = await db.execute(select(Run).where(Run.id.in_(request.run_ids)))
    runs = result.scalars().all()
    run_map = {run.id: run for run in runs}

    missing = [rid for rid in request.run_ids if rid not in run_map]
    if missing:
        raise HTTPException(status_code=404, detail=f"任务不存在: {', '.join(missing)}")

    not_completed = [rid for rid in request.run_ids if run_map[rid].status != "completed"]
    if not_completed:
        raise HTTPException(status_code=400, detail=f"以下任务尚未完成: {', '.join(not_completed)}")

    items: List[RunCompareItem] = []
    metric_name = None
    for rid in request.run_ids:
        run = run_map[rid]
        output_dir = Path(run.output_dir)

        metrics: Dict[str, float] = {}
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                metrics = {
                    str(k): float(v) for k, v in metrics_data.get("final", {}).items()
                    if isinstance(v, (int, float))
                }
            except (json.JSONDecodeError, OSError):
                pass

        best_model = None
        best_score = None
        leaderboard_path = output_dir / "leaderboard.csv"
        if leaderboard_path.exists():
            try:
                import pandas as pd
                lb = pd.read_csv(leaderboard_path)
                if not lb.empty:
                    best_row = lb.iloc[0]
                    best_model = str(best_row.get("model"))
                    best_score = (
                        float(best_row.get("score_val"))
                        if pd.notna(best_row.get("score_val"))
                        else None
                    )
            except Exception:
                pass

        feature_count = None
        feature_columns_path = output_dir / "feature_columns.json"
        if feature_columns_path.exists():
            try:
                with open(feature_columns_path, "r", encoding="utf-8") as f:
                    feature_count = len(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass

        snapshot = (run.config or {}).get("snapshot", {})
        dataset_name = snapshot.get("dataset_name") or run.dataset_id

        if metric_name is None and run.primary_metric:
            metric_name = run.primary_metric

        items.append(
            RunCompareItem(
                run_id=rid,
                dataset_name=dataset_name,
                status=run.status,
                primary_metric=run.primary_metric,
                metrics=metrics,
                best_model=best_model,
                best_model_score=best_score,
                feature_count=feature_count,
            )
        )

    best_run_id = None
    if metric_name:
        scored = [
            (item.run_id, item.metrics.get(metric_name))
            for item in items
            if item.metrics.get(metric_name) is not None
        ]
        if scored:
            best_run_id = max(scored, key=lambda x: x[1])[0]

    return RunCompareResponse(runs=items, metric_name=metric_name, best_run_id=best_run_id)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """获取任务详情。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    return run


@router.get("/{run_id}/steps", response_model=List[RunStepResponse])
async def list_run_steps(run_id: str, db: AsyncSession = Depends(get_db)):
    """获取任务的所有原子步骤状态。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    result = await db.execute(
        select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.sequence)
    )
    steps = result.scalars().all()
    return steps


@router.post("/{run_id}/steps/{step_name}", response_model=RunStepResponse)
async def execute_run_step(
    run_id: str,
    step_name: str,
    db: AsyncSession = Depends(get_db),
):
    """执行单个步骤。"""
    if step_name not in STEP_ORDER:
        raise HTTPException(status_code=400, detail=f"未知步骤: {step_name}")

    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    step_result = await db.execute(
        select(RunStep).where(RunStep.run_id == run_id, RunStep.step_name == step_name)
    )
    step = step_result.scalar_one_or_none()
    if step is not None and step.status == "running":
        raise HTTPException(status_code=409, detail=f"步骤 {step_name} 正在执行中")

    # 提交单步任务到执行器
    config = run.config or {}
    snapshot = config.get("snapshot", {})
    cleaning_rules = snapshot.get("cleaning_rules") or {}
    id_columns = _get_id_columns_to_drop(snapshot.get("dataset_schema_info"))
    if id_columns:
        existing_drops = set(cleaning_rules.get("drop_columns", []))
        existing_drops.update(id_columns)
        cleaning_rules["drop_columns"] = list(existing_drops)

    # 从数据集记录或快照中解析文件路径
    dataset_result = await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))
    dataset = dataset_result.scalar_one_or_none()
    file_path = (dataset.file_path if dataset else None) or snapshot.get("dataset_file_path")

    training_executor.submit_step_sync(
        run_id=run.id,
        file_path=file_path,
        target_column=snapshot.get("target_column"),
        task_type=snapshot.get("task_type"),
        output_dir=run.output_dir,
        time_budget_minutes=run.time_budget_minutes,
        preset=snapshot.get("preset"),
        primary_metric=run.primary_metric,
        seed=snapshot.get("seed"),
        max_models=snapshot.get("max_models", 50),
        cleaning_rules=cleaning_rules,
        feature_engineering_enabled=snapshot.get("feature_engineering_enabled", True),
        candidate_config=config.get("candidate_config"),
        step=step_name,
    )

    # 立即返回步骤记录（状态会在执行完成后异步更新）
    if step is None:
        step = RunStep(
            run_id=run_id,
            step_name=step_name,
            status="pending",
            sequence=STEP_ORDER.index(step_name),
        )
        db.add(step)
        await db.commit()
        await db.refresh(step)
    return step


@router.post("/{run_id}/continue", response_model=RunStepResponse)
async def continue_run(
    run_id: str,
    request: StepExecutionRequest,
    db: AsyncSession = Depends(get_db),
):
    """继续执行下一个待完成的步骤。

    如果请求体指定 step_name，则执行该步骤；否则自动找到第一个 pending 步骤。
    """
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    if request.step_name:
        target_step = request.step_name
        if target_step not in STEP_ORDER:
            raise HTTPException(status_code=400, detail=f"未知步骤: {target_step}")
    else:
        result = await db.execute(
            select(RunStep)
            .where(RunStep.run_id == run_id)
            .order_by(RunStep.sequence)
        )
        steps = result.scalars().all()
        target_step = None
        for step in steps:
            if step.status in ("pending", "failed"):
                target_step = step.step_name
                break
        if target_step is None:
            raise HTTPException(status_code=400, detail="所有步骤已完成")

    return await execute_run_step(run_id, target_step, db)


@router.get("/{run_id}/artifacts/{artifact_name}")
async def get_artifact(run_id: str, artifact_name: str, db: AsyncSession = Depends(get_db)):
    """获取中间产物。

    artifact_name 支持：
    - JSON 产物：metadata, quality_report, strategy, cv_results, metrics, interpretation
    - 文件产物：report, preprocessor, feature_columns, raw, train_raw, val_raw, test_raw,
      train_transformed, val_transformed, test_transformed, sampled_train,
      leaderboard, feature_importance, permutation_importance
    """
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    manifest = StepManifest(run.output_dir)
    json_artifacts = {
        "metadata": manifest.metadata_path,
        "quality_report": manifest.quality_report_path,
        "strategy": manifest.strategy_path,
        "cv_results": manifest.cv_results_path,
        "metrics": manifest.metrics_path,
        "interpretation": manifest.interpretation_path,
    }
    file_artifacts = {
        "report": manifest.report_path,
        "preprocessor": manifest.preprocessor_path,
        "feature_columns": manifest.feature_columns_path,
        "raw": manifest.raw_data_path,
        "train_raw": manifest.train_raw_path,
        "val_raw": manifest.val_raw_path,
        "test_raw": manifest.test_raw_path,
        "train_transformed": manifest.train_transformed_path,
        "val_transformed": manifest.val_transformed_path,
        "test_transformed": manifest.test_transformed_path,
        "sampled_train": manifest.sampled_train_path,
        "leaderboard": manifest.leaderboard_path,
        "feature_importance": manifest.feature_importance_path,
        "permutation_importance": manifest.permutation_importance_path,
    }

    if artifact_name in json_artifacts:
        path = json_artifacts[artifact_name]
        if not path.exists():
            raise HTTPException(status_code=404, detail="产物不存在")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail=f"读取产物失败: {e}")
        return data

    if artifact_name in file_artifacts:
        path = file_artifacts[artifact_name]
        if not path.exists():
            raise HTTPException(status_code=404, detail="产物不存在")
        return FileResponse(
            str(path),
            filename=path.name,
            media_type="application/octet-stream",
        )

    raise HTTPException(status_code=400, detail=f"未知产物名称: {artifact_name}")


@router.get("/{run_id}/events")
async def run_events(run_id: str, db: AsyncSession = Depends(get_db)):
    """Server-Sent Events：实时推送训练任务状态变化。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")

    q: queue.Queue = queue.Queue(maxsize=10)
    training_executor.subscribe_status(run_id, q)

    async def event_stream():
        try:
            # 先推送当前状态
            yield f"data: {json.dumps({'status': run.status, 'error_message': run.error_message}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    data = await asyncio.to_thread(q.get, timeout=25)
                except queue.Empty:
                    yield ":keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                if data.get("status") in ("completed", "failed"):
                    break
        finally:
            training_executor.unsubscribe_status(run_id, q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


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
        train_metrics = None
        cv_results = None
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                final = metrics_data.get("final", {})
                metrics = {
                    str(k): float(v) for k, v in final.items() if isinstance(v, (int, float))
                }
                extended_metrics = metrics_data.get("extended")
                train_metrics = {
                    str(k): float(v)
                    for k, v in metrics_data.get("train", {}).items()
                    if isinstance(v, (int, float))
                }
                cv_results = metrics_data.get("cv")

        # 读取详细错误信息
        error_details = None
        error_path = output_dir / "error.json"
        if error_path.exists():
            try:
                with open(error_path, "r", encoding="utf-8") as f:
                    error_details = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # 读取排行榜
        leaderboard = []
        leaderboard_path = output_dir / "leaderboard.csv"
        if leaderboard_path.exists():
            import pandas as pd

            leaderboard = pd.read_csv(leaderboard_path).head(25).to_dict(orient="records")

        # 读取特征重要性
        feature_importance = []
        importance_path = output_dir / "feature_importance.csv"
        if importance_path.exists():
            import pandas as pd

            feature_importance = pd.read_csv(importance_path).head(20).to_dict(orient="records")

        # 读取 Permutation Importance（如果存在）
        permutation_importance = None
        perm_importance_path = output_dir / "permutation_importance.csv"
        if perm_importance_path.exists():
            import pandas as pd

            permutation_importance = (
                pd.read_csv(perm_importance_path).head(20).to_dict(orient="records")
            )

        # 读取业务解读
        business_interpretation = None
        interpretation_path = output_dir / "business_interpretation.json"
        if interpretation_path.exists():
            try:
                with open(interpretation_path, "r", encoding="utf-8") as f:
                    business_interpretation = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        return RunResult(
            run_id=run_id,
            status=run.status,
            error_message=run.error_message,
            error_details=error_details,
            metrics=metrics,
            extended_metrics=extended_metrics,
            train_metrics=train_metrics if train_metrics else None,
            cv_results=cv_results,
            leaderboard=leaderboard,
            feature_importance=feature_importance,
            permutation_importance=permutation_importance,
            model_path=(
                str(output_dir / "autogluon_models")
                if (output_dir / "autogluon_models").exists()
                else None
            ),
            report_path=(
                str(output_dir / "report.html") if (output_dir / "report.html").exists() else None
            ),
            business_interpretation=business_interpretation,
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

        threshold = _load_optimal_threshold(output_dir)
        preds = _predict_with_threshold(predictor, df, threshold)

        response = PredictionResponse(predictions=preds.tolist(), threshold=threshold)

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


@router.post("/{run_id}/predict/batch")
async def batch_predict(
    run_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """批量预测：上传 CSV，返回 predictions.csv。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    try:
        from services.automl import load_predictor
        from services.preprocessing_pipeline import (
            DataPreprocessor,
            load_feature_columns,
            validate_prediction_input,
        )

        output_dir = Path(run.output_dir)
        model_path = output_dir / "autogluon_models"
        predictor = load_predictor(model_path)

        # 读取上传的 CSV
        content = await file.read()
        import io

        df = pd.read_csv(io.BytesIO(content))

        preprocessor_path = output_dir / "preprocessing_pipeline.joblib"
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor.load(preprocessor_path)
            errors = validate_prediction_input(preprocessor, df)
            if errors:
                raise HTTPException(status_code=400, detail="; ".join(errors))
            df = preprocessor.transform(df)

        feature_columns_path = output_dir / "feature_columns.json"
        if feature_columns_path.exists():
            feature_columns = load_feature_columns(output_dir)
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise HTTPException(
                    status_code=400, detail=f"输入缺少特征列: {sorted(missing_cols)}"
                )
            df = df[feature_columns]

        threshold = _load_optimal_threshold(output_dir)
        preds = _predict_with_threshold(predictor, df, threshold)
        result_df = pd.DataFrame({"prediction": preds})

        if predictor.problem_type in ["binary", "multiclass"]:
            try:
                probs = predictor.predict_proba(df)
                for col in probs.columns:
                    result_df[f"prob_{col}"] = probs[col].values
            except Exception:
                pass

        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{run_id}.csv"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/explain", response_model=ExplainResponse)
async def explain_sample(
    run_id: str,
    request: ExplainRequest,
    db: AsyncSession = Depends(get_db),
):
    """对单条样本进行 TreeSHAP / SHAP 解释。"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    if len(request.data) != 1:
        raise HTTPException(status_code=400, detail="每次只能解释一条样本")

    try:
        from services.automl import load_predictor
        from services.explainability import explain_single_sample
        from services.preprocessing_pipeline import (
            DataPreprocessor,
            load_feature_columns,
            validate_prediction_input,
        )

        output_dir = Path(run.output_dir)
        model_path = output_dir / "autogluon_models"
        predictor = load_predictor(model_path)

        df = pd.DataFrame(request.data)

        preprocessor_path = output_dir / "preprocessing_pipeline.joblib"
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor.load(preprocessor_path)
            errors = validate_prediction_input(preprocessor, df)
            if errors:
                raise HTTPException(status_code=400, detail="; ".join(errors))
            df = preprocessor.transform(df)

        feature_columns_path = output_dir / "feature_columns.json"
        if feature_columns_path.exists():
            feature_columns = load_feature_columns(output_dir)
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise HTTPException(
                    status_code=400, detail=f"输入缺少特征列: {sorted(missing_cols)}"
                )
            df = df[feature_columns]

        explanation = explain_single_sample(predictor, df)
        return ExplainResponse(**explanation)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/report")
async def download_report(
    run_id: str,
    download: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """获取或下载 HTML 报告。

    Args:
        download: 为 True 时返回 attachment 强制下载；
                  为 False 时返回 inline，可用于 iframe 预览。
    """
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
        filename=f"report_{run_id}.html" if download else None,
        content_disposition_type="attachment" if download else "inline",
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

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
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
