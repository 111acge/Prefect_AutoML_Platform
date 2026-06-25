# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""原子步骤执行器。

将数据科学流程拆分为独立、可重试、可观测的步骤，每个步骤通过文件产物传递状态，
支持从任意已完成步骤继续执行。
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import pandas as pd

from config import settings
from database import AsyncSessionLocal
from models import RunStep
from schemas import CleaningRules
from services.automl import AutoMLService
from services.cv_service import cross_validate_pipeline
from services.data_quality import assess_data_quality
from services.data_service import (
    analyze_metadata,
    infer_target_column,
    infer_task_type,
    load_dataframe,
)
from services.preprocessing import train_val_test_split
from services.preprocessing_pipeline import DataPreprocessor
from services.report_llm_service import generate_business_interpretation
from services.sampling_service import SamplingStrategy, build_sampling_strategy, apply_sampling
from services.step_manifest import StepManifest
from services.training_strategy import build_strategy
from services.visualization import generate_report_plots
from sqlalchemy import select

logger = logging.getLogger(__name__)


# 标准步骤顺序（sequence 从 0 开始）
STEP_ORDER: List[str] = [
    "ingest",
    "analyze",
    "quality",
    "strategy",
    "split",
    "cross_validate",
    "fit_preprocessor",
    "transform",
    "sample",
    "train",
    "evaluate",
    "interpret",
    "report",
]

# 可选步骤：失败不影响继续执行
OPTIONAL_STEPS = {"quality", "cross_validate", "interpret", "report"}


async def initialize_run_steps(
    run_id: str,
    output_dir: str | Path,
    context: Dict[str, Any],
) -> None:
    """为 step 模式创建运行上下文与所有待执行步骤记录。"""
    manifest = StepManifest(output_dir)
    manifest.save_run_context(context)

    async with AsyncSessionLocal() as db:
        for idx, step_name in enumerate(STEP_ORDER):
            result = await db.execute(
                select(RunStep).where(
                    RunStep.run_id == run_id,
                    RunStep.step_name == step_name,
                )
            )
            if result.scalar_one_or_none() is None:
                db.add(
                    RunStep(
                        run_id=run_id,
                        step_name=step_name,
                        status="pending",
                        sequence=idx,
                    )
                )
        await db.commit()


def _step_sequence(step_name: str) -> int:
    """获取步骤在流水线中的顺序。"""
    try:
        return STEP_ORDER.index(step_name)
    except ValueError as e:
        raise ValueError(f"未知步骤: {step_name}") from e


class StepRunner:
    """单步执行器。

    每个 step 都是幂等的：若 output_manifest 已存在且状态为 completed，则直接跳过。
    """

    def __init__(self, run_id: str, output_dir: str | Path):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.manifest = StepManifest(self.output_dir)

    # ------------------------------------------------------------------
    # 数据库操作（复用 AsyncSessionLocal，子进程中通过 asyncio.run 调用）
    # ------------------------------------------------------------------
    async def _get_or_create_step(self, step_name: str) -> RunStep:
        """获取或创建 RunStep 记录。"""
        async with AsyncSessionLocal() as db:
            sequence = _step_sequence(step_name)
            result = await db.execute(
                select(RunStep).where(
                    RunStep.run_id == self.run_id,
                    RunStep.step_name == step_name,
                )
            )
            step = result.scalar_one_or_none()
            if step is None:
                step = RunStep(
                    run_id=self.run_id,
                    step_name=step_name,
                    status="pending",
                    sequence=sequence,
                )
                db.add(step)
                await db.commit()
                await db.refresh(step)
            return step

    async def _update_step(
        self,
        step_name: str,
        status: str,
        error_message: Optional[str] = None,
        input_manifest: Optional[Dict[str, Any]] = None,
        output_manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新步骤状态。"""
        now = datetime.now(UTC)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(RunStep).where(
                    RunStep.run_id == self.run_id,
                    RunStep.step_name == step_name,
                )
            )
            step = result.scalar_one_or_none()
            if step is None:
                step = RunStep(
                    run_id=self.run_id,
                    step_name=step_name,
                    status=status,
                    sequence=_step_sequence(step_name),
                )
                db.add(step)
            else:
                step.status = status

            if status == "running" and step.started_at is None:
                step.started_at = now
            if status in ("completed", "failed"):
                step.completed_at = now
            if error_message is not None:
                step.error_message = error_message
            if input_manifest is not None:
                step.input_manifest = input_manifest
            if output_manifest is not None:
                step.output_manifest = output_manifest

            await db.commit()

    async def _prerequisite_completed(self, step_name: str) -> bool:
        """检查某前置步骤是否已完成。"""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(RunStep).where(
                    RunStep.run_id == self.run_id,
                    RunStep.step_name == step_name,
                )
            )
            step = result.scalar_one_or_none()
            return step is not None and step.status == "completed"

    async def _check_prerequisites(self, step_name: str) -> None:
        """校验前置步骤是否完成。"""
        idx = _step_sequence(step_name)
        for prev_step in STEP_ORDER[:idx]:
            if prev_step in OPTIONAL_STEPS:
                # 可选步骤不强制完成
                continue
            if not await self._prerequisite_completed(prev_step):
                raise RuntimeError(
                    f"步骤 {step_name} 依赖的前置步骤 {prev_step} 尚未完成"
                )

    # ------------------------------------------------------------------
    # 步骤实现
    # ------------------------------------------------------------------
    def _run_context(self) -> Dict[str, Any]:
        """读取运行上下文。"""
        return self.manifest.load_run_context()

    def _config_snapshot(self) -> Dict[str, Any]:
        """读取配置快照。"""
        return self.manifest.load_config_snapshot()

    def _write_error(self, step_name: str, message: str) -> None:
        """将步骤错误写入 error.json，兼容现有 executor 读取逻辑。"""
        error_path = self.output_dir / "error.json"
        try:
            self.manifest.save_json(
                error_path,
                {"step": step_name, "error_message": message},
            )
        except Exception:
            pass

    # ---------- ingest ----------
    async def _run_ingest(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """加载数据、校验目标列、推断目标列/任务类型、保存原始数据。"""
        config = self._config_snapshot()
        file_path = config.get("dataset_file_path") or config.get("file_path")
        if not file_path:
            # 尝试从 dataset_id 推导（需要 dataset 记录）
            file_path = await self._resolve_dataset_path()

        target_column = ctx.get("target_column")
        task_type = ctx.get("task_type")

        df = load_dataframe(file_path)

        # 自动推断
        if target_column is None or task_type is None:
            inferred_target, confidence = infer_target_column(df, hint=target_column)
            if inferred_target is None:
                raise ValueError("无法自动推断目标列，请显式指定 target_column")
            if target_column is None:
                target_column = inferred_target
                logger.info(f"自动推断目标列: {target_column} (置信度 {confidence:.2f})")
            if task_type is None:
                task_type = infer_task_type(df[target_column])
                logger.info(f"自动推断任务类型: {task_type}")

        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在")

        # 保存运行上下文（覆盖未提供的字段）
        ctx.update(
            {
                "target_column": target_column,
                "task_type": task_type,
                "file_path": str(file_path),
                "n_samples": int(df.shape[0]),
                "n_features": int(df.shape[1] - 1),
            }
        )
        self.manifest.save_run_context(ctx)

        # 保存原始数据
        df.to_parquet(self.manifest.raw_data_path, index=False)

        return {
            "raw_data": str(self.manifest.raw_data_path),
            "target_column": target_column,
            "task_type": task_type,
            "shape": list(df.shape),
        }

    async def _resolve_dataset_path(self) -> str:
        """根据 run_id 对应的数据集解析文件路径。"""
        from models import Run, Dataset

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Run).where(Run.id == self.run_id))
            run = result.scalar_one_or_none()
            if run is None:
                raise ValueError(f"Run {self.run_id} 不存在")
            result = await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))
            dataset = result.scalar_one_or_none()
            if dataset is None or not dataset.file_path:
                raise ValueError(f"数据集不存在或缺少文件路径: {run.dataset_id}")
            return dataset.file_path

    # ---------- analyze ----------
    async def _run_analyze(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """分析元数据。"""
        df = pd.read_parquet(self.manifest.raw_data_path)
        metadata = analyze_metadata(df, ctx["target_column"], ctx["task_type"])
        self.manifest.save_json(self.manifest.metadata_path, metadata)
        return {"metadata": str(self.manifest.metadata_path)}

    # ---------- quality ----------
    async def _run_quality(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """六维数据质量评估。"""
        df = pd.read_parquet(self.manifest.raw_data_path)
        quality = assess_data_quality(df, ctx["target_column"])
        self.manifest.save_json(self.manifest.quality_report_path, quality)
        return {"quality_report": str(self.manifest.quality_report_path)}

    # ---------- strategy ----------
    async def _run_strategy(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """构建训练策略。"""
        config = self._config_snapshot()
        metadata = self.manifest.load_json(self.manifest.metadata_path, {})

        candidate_config = config.get("candidate_config") or {}
        effective_time_budget = (
            candidate_config.get("time_budget_minutes")
            or config.get("time_budget_minutes")
            or 10.0
        )
        effective_preset = candidate_config.get("preset") or config.get("preset")
        effective_primary_metric = (
            candidate_config.get("primary_metric") or config.get("primary_metric")
        )
        effective_max_models = (
            candidate_config.get("max_models") or config.get("max_models") or 50
        )

        strategy_obj = build_strategy(
            metadata=metadata,
            task_type=ctx["task_type"],
            user_time_budget_minutes=effective_time_budget,
            user_preset=effective_preset,
            user_primary_metric=effective_primary_metric,
            user_max_models=effective_max_models,
        )
        strategy = strategy_obj.to_dict()

        # 合并候选配置
        if candidate_config.get("preprocessing"):
            strategy.setdefault("preprocessing", {}).update(candidate_config["preprocessing"])
        if candidate_config.get("validation_strategy"):
            strategy["validation_strategy"] = {
                **strategy.get("validation_strategy", {}),
                **candidate_config["validation_strategy"],
            }
        if candidate_config.get("hyperparameters"):
            strategy["hyperparameters"] = candidate_config["hyperparameters"]

        strategy["feature_engineering_enabled"] = (
            candidate_config.get("feature_engineering_enabled")
            if candidate_config.get("feature_engineering_enabled") is not None
            else config.get("feature_engineering_enabled", True)
        )

        self.manifest.save_json(self.manifest.strategy_path, strategy)
        return {"strategy": str(self.manifest.strategy_path)}

    # ---------- split ----------
    async def _run_split(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """划分训练/验证/测试集（必须在预处理前）。"""
        df = pd.read_parquet(self.manifest.raw_data_path)
        strategy = self.manifest.load_json(self.manifest.strategy_path, {})
        val_size = strategy.get("validation_strategy", {}).get("val_size", 0.15)
        test_size = strategy.get("validation_strategy", {}).get("test_size", 0.15)
        seed = ctx.get("seed") or 42

        result = train_val_test_split(
            df,
            ctx["target_column"],
            task_type=ctx["task_type"],
            val_size=val_size,
            test_size=test_size,
            random_state=seed,
            rare_class_strategy=ctx.get("rare_class_strategy", "auto"),
        )
        train_df, val_df, test_df = result["train"], result["val"], result["test"]

        train_df.to_parquet(self.manifest.train_raw_path, index=False)
        val_df.to_parquet(self.manifest.val_raw_path, index=False)
        test_df.to_parquet(self.manifest.test_raw_path, index=False)

        return {
            "train_raw": str(self.manifest.train_raw_path),
            "val_raw": str(self.manifest.val_raw_path),
            "test_raw": str(self.manifest.test_raw_path),
            "shapes": {
                "train": list(train_df.shape),
                "val": list(val_df.shape),
                "test": list(test_df.shape),
            },
        }

    # ---------- cross_validate ----------
    async def _run_cross_validate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """显式交叉验证。"""
        train_df = pd.read_parquet(self.manifest.train_raw_path)
        strategy = self.manifest.load_json(self.manifest.strategy_path, {})
        config = self._config_snapshot()
        cleaning_rules = self._effective_cleaning_rules(config)

        cv_results = cross_validate_pipeline(
            train_df=train_df,
            target_column=ctx["target_column"],
            task_type=ctx["task_type"],
            strategy=strategy,
            cleaning_rules=cleaning_rules,
            n_folds=strategy.get("validation_strategy", {}).get("n_folds", 5),
            cv_type=strategy.get("validation_strategy", {}).get("cv_type"),
        )
        self.manifest.save_json(self.manifest.cv_results_path, cv_results)
        return {"cv_results": str(self.manifest.cv_results_path)}

    # ---------- fit_preprocessor ----------
    async def _run_fit_preprocessor(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """在训练集上拟合预处理器并保存。"""
        train_df = pd.read_parquet(self.manifest.train_raw_path)
        strategy = self.manifest.load_json(self.manifest.strategy_path, {})
        config = self._config_snapshot()
        cleaning_rules = self._effective_cleaning_rules(config)

        preprocessor = DataPreprocessor(
            target_column=ctx["target_column"],
            strategy=strategy,
            cleaning_rules=cleaning_rules,
        )
        preprocessor.fit(train_df)
        preprocessor.save(self.manifest.preprocessor_path)

        # 保存特征列
        from services.preprocessing_pipeline import save_feature_columns

        save_feature_columns(self.output_dir, preprocessor.feature_columns)

        return {
            "preprocessor": str(self.manifest.preprocessor_path),
            "feature_columns": str(self.manifest.feature_columns_path),
            "n_features": len(preprocessor.feature_columns),
        }

    # ---------- transform ----------
    async def _run_transform(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """使用已拟合的预处理器转换训练/验证/测试集。"""
        preprocessor = DataPreprocessor.load(self.manifest.preprocessor_path)

        train_df = preprocessor.transform(pd.read_parquet(self.manifest.train_raw_path))
        val_df = preprocessor.transform(pd.read_parquet(self.manifest.val_raw_path))
        test_df = preprocessor.transform(pd.read_parquet(self.manifest.test_raw_path))

        train_df.to_parquet(self.manifest.train_transformed_path, index=False)
        val_df.to_parquet(self.manifest.val_transformed_path, index=False)
        test_df.to_parquet(self.manifest.test_transformed_path, index=False)

        return {
            "train_transformed": str(self.manifest.train_transformed_path),
            "val_transformed": str(self.manifest.val_transformed_path),
            "test_transformed": str(self.manifest.test_transformed_path),
            "shapes": {
                "train": list(train_df.shape),
                "val": list(val_df.shape),
                "test": list(test_df.shape),
            },
        }

    # ---------- sample ----------
    async def _run_sample(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """构建并应用采样策略（仅在训练集）。"""
        train_df = pd.read_parquet(self.manifest.train_transformed_path)
        strategy_dict = build_sampling_strategy(
            train_df, ctx["target_column"], ctx["task_type"]
        ).to_dict()
        self.manifest.save_json(self.manifest.sampling_strategy_path, strategy_dict)

        strategy = SamplingStrategy(
            method=strategy_dict["method"],
            params=strategy_dict.get("params", {}),
            rationale=strategy_dict.get("rationale", []),
            imbalance_ratio=strategy_dict.get("imbalance_ratio"),
        )
        sampled_df, sample_weight = apply_sampling(train_df, ctx["target_column"], strategy)

        sampled_df.to_parquet(self.manifest.sampled_train_path, index=False)
        if sample_weight is not None:
            pd.Series(sample_weight, name="sample_weight").to_frame().to_parquet(
                self.manifest.sample_weight_path, index=False
            )

        return {
            "sampled_train": str(self.manifest.sampled_train_path),
            "sample_weight": (
                str(self.manifest.sample_weight_path)
                if self.manifest.sample_weight_path.exists()
                else None
            ),
            "original_shape": list(train_df.shape),
            "sampled_shape": list(sampled_df.shape),
        }

    # ---------- train ----------
    async def _run_train(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """训练 AutoGluon 模型。"""
        sampled_train = pd.read_parquet(self.manifest.sampled_train_path)
        strategy = self.manifest.load_json(self.manifest.strategy_path, {})

        sample_weight = None
        if self.manifest.sample_weight_path.exists():
            sample_weight = pd.read_parquet(self.manifest.sample_weight_path)[
                "sample_weight"
            ]

        automl = AutoMLService(self.output_dir)
        result = automl.train(
            train_data=sampled_train,
            target_column=ctx["target_column"],
            task_type=ctx["task_type"],
            time_limit=strategy["time_limit_seconds"],
            preset=strategy["preset"],
            primary_metric=strategy["primary_metric"],
            seed=ctx.get("seed"),
            max_models=strategy["max_models"],
            strategy=strategy,
            sample_weight=sample_weight,
        )
        return {"train_result": result}

    # ---------- evaluate ----------
    async def _run_evaluate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """评估模型并保存指标。"""
        from prefect_flows.automl_flow import (
            _compute_extended_metrics,
            _compute_optimal_threshold,
            _validate_ensemble,
        )
        from autogluon.tabular import TabularPredictor

        train_df = pd.read_parquet(self.manifest.train_transformed_path)
        val_df = pd.read_parquet(self.manifest.val_transformed_path)
        test_df = pd.read_parquet(self.manifest.test_transformed_path)
        target_column = ctx["target_column"]

        predictor = TabularPredictor.load(str(self.manifest.model_dir))

        # 验证集评估
        val_performance = predictor.evaluate(val_df)
        metrics = {"val": {str(k): float(v) for k, v in val_performance.items()}}

        # 测试集评估
        test_performance = predictor.evaluate(test_df)
        metrics["final"] = {str(k): float(v) for k, v in test_performance.items()}

        # 扩展指标
        X_test = test_df.drop(columns=[target_column])
        y_true = test_df[target_column]
        y_pred = predictor.predict(X_test)
        extended = _compute_extended_metrics(y_true, y_pred, X_test, predictor, predictor.problem_type)
        if extended:
            metrics["extended"] = extended

        # 阈值调优
        if predictor.problem_type == "binary":
            X_val = val_df.drop(columns=[target_column])
            y_val = val_df[target_column]
            metrics["threshold"] = _compute_optimal_threshold(y_val, X_val, predictor)

        # 集成验证
        try:
            leaderboard = predictor.leaderboard(silent=True)
            metrics["ensemble_validation"] = _validate_ensemble(leaderboard, None)
        except Exception as e:
            logger.warning(f"集成验证失败: {e}")

        # 训练集参考指标
        metrics["train"] = {}
        if settings.train_eval_enabled:
            try:
                from config import (
                    HIGH_CARDINALITY_CLASS_THRESHOLD,
                    LARGE_DATASET_ROW_THRESHOLD,
                    DEFAULT_TRAIN_EVAL_SAMPLE_SIZE,
                )

                eval_train_data = train_df
                sample_size = settings.train_eval_sample_size
                n_classes = train_df[target_column].nunique(dropna=True)

                if sample_size == 0:
                    eval_train_data = None
                elif sample_size is not None and len(train_df) > sample_size:
                    eval_train_data = train_df.sample(n=sample_size, random_state=42)
                elif sample_size is None and (
                    n_classes > HIGH_CARDINALITY_CLASS_THRESHOLD
                    or len(train_df) > LARGE_DATASET_ROW_THRESHOLD
                ):
                    sample_size = min(DEFAULT_TRAIN_EVAL_SAMPLE_SIZE, len(train_df))
                    eval_train_data = train_df.sample(n=sample_size, random_state=42)

                if eval_train_data is not None:
                    train_performance = predictor.evaluate(eval_train_data)
                    metrics["train"] = {str(k): float(v) for k, v in train_performance.items()}
            except Exception as e:
                logger.warning(f"训练集评估失败: {e}")

        # CV 结果
        if self.manifest.cv_results_path.exists():
            metrics["cv"] = self.manifest.load_json(self.manifest.cv_results_path, {})

        self.manifest.save_json(self.manifest.metrics_path, metrics)
        return {"metrics": str(self.manifest.metrics_path)}

    # ---------- interpret ----------
    async def _run_interpret(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """生成 LLM 业务解读。"""
        strategy = self.manifest.load_json(self.manifest.strategy_path, {})
        quality = self.manifest.load_json(self.manifest.quality_report_path, {})
        metrics = self.manifest.load_json(self.manifest.metrics_path, {})

        feature_importance = []
        if self.manifest.feature_importance_path.exists():
            feature_importance = (
                pd.read_csv(self.manifest.feature_importance_path).head(10).to_dict(orient="records")
            )

        # 训练流程默认使用规则模板生成业务解读；LLM 版本需用户在页面上主动触发，
        # 并在触发前展示数据外传免责声明。
        interpretation = await generate_business_interpretation(
            task_type=ctx["task_type"],
            primary_metric=strategy.get("primary_metric"),
            metrics=metrics,
            feature_importance=feature_importance,
            quality=quality,
            strategy=strategy,
            force_rule_based=True,
        )
        self.manifest.save_json(self.manifest.interpretation_path, interpretation)
        return {"interpretation": str(self.manifest.interpretation_path)}

    # ---------- report ----------
    async def _run_report(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """生成 HTML 报告。"""
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        strategy = self.manifest.load_json(self.manifest.strategy_path, {})
        sampling_strategy = self.manifest.load_json(self.manifest.sampling_strategy_path, {})
        strategy["sampling"] = sampling_strategy
        metadata = self.manifest.load_json(self.manifest.metadata_path, {})
        quality = self.manifest.load_json(self.manifest.quality_report_path, {})
        interpretation = self.manifest.load_json(self.manifest.interpretation_path)
        metrics = self.manifest.load_json(self.manifest.metrics_path, {})

        plots = generate_report_plots(self.output_dir, ctx["task_type"])

        template_dir = settings.project_root / "backend" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template("report.html")

        leaderboard = []
        leaderboard_columns = []
        if self.manifest.leaderboard_path.exists():
            lb = pd.read_csv(self.manifest.leaderboard_path).head(25)
            leaderboard = lb.to_dict(orient="records")
            leaderboard_columns = lb.columns.tolist()

        feature_importance = []
        importance_columns = []
        if self.manifest.feature_importance_path.exists():
            fi = pd.read_csv(self.manifest.feature_importance_path).head(20)
            feature_importance = fi.to_dict(orient="records")
            importance_columns = fi.columns.tolist()

        perm_importance = []
        perm_importance_columns = []
        if self.manifest.permutation_importance_path.exists():
            pi = pd.read_csv(self.manifest.permutation_importance_path).head(20)
            perm_importance = pi.to_dict(orient="records")
            perm_importance_columns = pi.columns.tolist()

        html_content = template.render(
            run_id=self.run_id,
            status="completed",
            preset=strategy.get("preset"),
            strategy=strategy,
            primary_metric=strategy.get("primary_metric"),
            seed=ctx.get("seed"),
            metadata=metadata,
            metrics=metrics.get("final", {}),
            leaderboard=leaderboard,
            leaderboard_columns=leaderboard_columns,
            feature_importance=feature_importance,
            importance_columns=importance_columns,
            perm_importance=perm_importance,
            perm_importance_columns=perm_importance_columns,
            plots=plots,
            quality=quality,
            metrics_full=metrics,
            interpretation=interpretation,
        )

        with open(self.manifest.report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return {"report": str(self.manifest.report_path)}

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------
    def _effective_cleaning_rules(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """合并用户清洗规则与候选配置中的清洗规则。"""
        cleaning_rules = config.get("cleaning_rules") or {}
        candidate_config = config.get("candidate_config") or {}
        if candidate_config.get("cleaning_rules"):
            cleaning_rules = {**cleaning_rules, **candidate_config["cleaning_rules"]}
        return cleaning_rules

    async def run_step(self, step_name: str) -> Dict[str, Any]:
        """执行单个步骤并返回 output_manifest。"""
        if step_name not in STEP_ORDER:
            raise ValueError(f"未知步骤: {step_name}")

        # 幂等检查
        step_record = await self._get_or_create_step(step_name)
        if step_record.status == "completed":
            logger.info(f"步骤 {step_name} 已完成，跳过执行")
            return step_record.output_manifest or {}

        await self._check_prerequisites(step_name)

        # ingest 步骤负责创建运行上下文，因此允许此时 run_context 不存在
        if step_name == "ingest":
            ctx = self._config_snapshot()
            ctx = {
                "target_column": ctx.get("target_column"),
                "task_type": ctx.get("task_type"),
                "seed": ctx.get("seed"),
                "file_path": ctx.get("dataset_file_path"),
                "rare_class_strategy": ctx.get("rare_class_strategy", "auto"),
            }
        else:
            ctx = self._run_context()

        input_manifest = {
            "target_column": ctx.get("target_column"),
            "task_type": ctx.get("task_type"),
            "seed": ctx.get("seed"),
        }

        await self._update_step(step_name, "running", input_manifest=input_manifest)
        logger.info(f"开始执行步骤: {step_name}")

        try:
            handler: Callable[[Dict[str, Any]], Any] = getattr(self, f"_run_{step_name}")
            output_manifest = await handler(ctx)
            await self._update_step(
                step_name,
                "completed",
                input_manifest=input_manifest,
                output_manifest=output_manifest,
            )
            logger.info(f"步骤 {step_name} 执行完成")
            return output_manifest
        except Exception as e:
            error_message = str(e)
            logger.exception(f"步骤 {step_name} 执行失败: {error_message}")
            self._write_error(step_name, error_message)
            await self._update_step(
                step_name,
                "failed",
                error_message=error_message,
                input_manifest=input_manifest,
            )
            if step_name in OPTIONAL_STEPS:
                # 可选步骤失败不抛出，返回空 manifest 以便继续
                return {}
            raise

    async def run_all(self) -> Dict[str, Any]:
        """顺序执行所有步骤（一键训练模式）。"""
        results: Dict[str, Any] = {}
        for step_name in STEP_ORDER:
            try:
                results[step_name] = await self.run_step(step_name)
            except Exception:
                if step_name in OPTIONAL_STEPS:
                    results[step_name] = {"skipped": True, "reason": "optional step failed"}
                    continue
                raise
        return results
