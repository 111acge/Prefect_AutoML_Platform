# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""LLM 驱动的 AutoML 搜索 Agent。

在严格隔离 train/val/test 的前提下，迭代地提出候选配置、提交训练、
读取验证集指标、反思并生成下一批候选，最终返回验证集上最好的 Run。

实现上通过调用内部 /api/runs 端点复用现有训练执行逻辑，避免与 routers 层循环依赖。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from database import AsyncSessionLocal
from models import Dataset, Experiment, Trial
from schemas import CandidateConfig, ExperimentCreate, CleaningRules
from i18n import _, get_locale
from services.llm_client import call_llm_for_json
from services.llm_strategy_service import recommend_candidates

logger = logging.getLogger(__name__)


def _cap_candidate_budgets(
    candidates: List[CandidateConfig],
    time_budget_minutes: Optional[float],
) -> None:
    """将候选时间预算限制在实验总预算内。"""
    if time_budget_minutes is None or not candidates:
        return
    per_candidate_budget = max(0.1, time_budget_minutes / len(candidates))
    for candidate in candidates:
        if candidate.time_budget_minutes is None or candidate.time_budget_minutes > per_candidate_budget:
            candidate.time_budget_minutes = per_candidate_budget


def _is_higher_better(metric: str) -> bool:
    """判断指标是否越大越好。"""
    lower_better = {
        "log_loss",
        "root_mean_squared_error",
        "mean_squared_error",
        "mean_absolute_error",
        "rmse",
        "mse",
        "mae",
    }
    return metric.lower() not in lower_better


def _read_val_score(output_dir: str, metric: str) -> Optional[float]:
    """从 metrics.json 中读取验证集分数。"""
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val_metrics = data.get("val", {})
        value = val_metrics.get(metric)
        if value is None:
            value = val_metrics.get("score_val") or next(iter(val_metrics.values()), None)
        return float(value) if value is not None else None
    except Exception as e:
        logger.warning(_("llm.search.read_val_score_failed", path=output_dir, msg=e))
        return None


def _read_test_score(output_dir: str, metric: str) -> Optional[float]:
    """从 metrics.json 中读取测试集分数（仅用于最终报告）。"""
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        final_metrics = data.get("final", {})
        value = final_metrics.get(metric)
        if value is None:
            value = final_metrics.get("score_val") or next(iter(final_metrics.values()), None)
        return float(value) if value is not None else None
    except Exception as e:
        logger.warning(_("llm.search.read_test_score_failed", path=output_dir, msg=e))
        return None


def _get_internal_client() -> Any:
    """延迟导入并构造内部 HTTP 客户端，避免模块级循环依赖。"""
    from httpx import AsyncClient, ASGITransport
    from main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _submit_candidate_via_api(
    experiment_id: str,
    dataset_id: str,
    target_column: str,
    task_type: str,
    candidate: CandidateConfig,
    rare_class_strategy: Optional[str] = "auto",
) -> tuple[str, str, str]:
    """通过内部 API 提交候选，返回 (run_id, output_dir, status)。"""
    client = _get_internal_client()
    payload = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "task_type": task_type,
        "primary_metric": candidate.primary_metric,
        "time_budget_minutes": candidate.time_budget_minutes,
        "max_models": candidate.max_models,
        "preset": candidate.preset,
        "seed": candidate.seed,
        "feature_engineering_enabled": candidate.feature_engineering_enabled,
        "experiment_id": experiment_id,
        "candidate_config": candidate.model_dump(),
        "rare_class_strategy": candidate.rare_class_strategy or rare_class_strategy,
    }
    async with client:
        response = await client.post("/api/runs", json=payload)
        response.raise_for_status()
        run_data = response.json()
        run_id = run_data["id"]
        output_dir = run_data["output_dir"]

        # 轮询状态
        for _ in range(7200):  # 最多 4 小时
            await asyncio.sleep(2)
            status_resp = await client.get(f"/api/runs/{run_id}")
            status_resp.raise_for_status()
            status = status_resp.json()["status"]
            if status in ("completed", "failed"):
                return run_id, output_dir, status

        return run_id, output_dir, "timeout"


async def _create_trial(
    experiment_id: str,
    candidate: CandidateConfig,
    run_id: Optional[str] = None,
) -> Trial:
    async with AsyncSessionLocal() as db:
        trial = Trial(
            experiment_id=experiment_id,
            run_id=run_id,
            candidate_params=candidate.model_dump(),
            primary_metric=candidate.primary_metric,
            status="pending",
        )
        db.add(trial)
        await db.commit()
        await db.refresh(trial)
        return trial


async def _update_trial(trial_id: str, run_id: str, output_dir: str, status: str) -> Optional[Trial]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Trial).where(Trial.id == trial_id))
        trial = result.scalar_one_or_none()
        if trial is None:
            logger.warning(_("llm.search.trial_not_found", trial_id=trial_id))
            return None
        trial.run_id = run_id
        trial.status = status
        if status == "completed" and trial.primary_metric:
            trial.val_score = _read_val_score(output_dir, trial.primary_metric)
            trial.test_score = _read_test_score(output_dir, trial.primary_metric)
        trial.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(trial)
        return trial


def _build_reflection_prompt(
    task_type: str,
    primary_metric: str,
    history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system_prompt = _("llm_prompt.search_agent_system")
    metric_direction = (
        _("llm_prompt.metric_direction_higher_better")
        if _is_higher_better(primary_metric)
        else _("llm_prompt.metric_direction_lower_better")
    )
    user_prompt = _(
        "llm_prompt.search_agent_user",
        task_type=task_type,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        history=json.dumps(history, ensure_ascii=False, default=str),
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _propose_next_candidates(
    task_type: str,
    primary_metric: str,
    history: List[Dict[str, Any]],
    provider: str = "auto",
) -> List[CandidateConfig]:
    try:
        messages = _build_reflection_prompt(task_type, primary_metric, history)
        result = await call_llm_for_json(
            messages,
            provider=provider,
            max_tokens=4096,
            temperature=0.2,
            timeout=60.0,
            retries=1,
        )
        raw_candidates = result.get("candidates", [])
        candidates: List[CandidateConfig] = []
        for raw in raw_candidates[:3]:
            try:
                if "cleaning_rules" in raw:
                    raw["cleaning_rules"] = (
                        CleaningRules(**raw["cleaning_rules"])
                        if raw["cleaning_rules"]
                        else None
                    )
                candidates.append(CandidateConfig(**raw))
            except Exception as e:
                logger.warning(_("llm.search.reflection_parse_failed", msg=e, raw=raw))
        return candidates
    except Exception as e:
        logger.warning(_("llm.search.reflection_failed", msg=e))
        return []


async def run_experiment(
    dataset_id: str,
    target_column: str,
    task_type: str,
    primary_metric: Optional[str] = None,
    max_iterations: int = 5,
    trials_per_iteration: int = 2,
    time_budget_minutes: Optional[float] = None,
    rare_class_strategy: Optional[str] = "auto",
    provider: str = "auto",
    experiment_id: Optional[str] = None,
) -> Experiment:
    """运行一次 LLM 驱动的多候选搜索实验。

    若传入 experiment_id，则复用该实验记录；否则新建。
    """
    from services.data_service import load_dataframe, analyze_metadata
    from services.data_quality import assess_data_quality

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if dataset is None:
            raise ValueError(_("dataset.not_found"))

        if experiment_id:
            exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            experiment = exp_result.scalar_one_or_none()
            if experiment is None:
                raise ValueError(_("experiment.not_found"))
        else:
            experiment = Experiment(
                dataset_id=dataset_id,
                search_config={
                    "target_column": target_column,
                    "task_type": task_type,
                    "primary_metric": primary_metric,
                    "max_iterations": max_iterations,
                    "trials_per_iteration": trials_per_iteration,
                    "time_budget_minutes": time_budget_minutes,
                    "rare_class_strategy": rare_class_strategy,
                },
            )
            db.add(experiment)
            await db.flush()
            await db.commit()
            await db.refresh(experiment)

    # 元数据与质量报告
    df = load_dataframe(dataset.file_path)
    metadata = analyze_metadata(df, target_column=target_column, task_type=task_type)
    quality = assess_data_quality(df, target_column=target_column)

    if primary_metric is None:
        primary_metric = (
            "root_mean_squared_error" if task_type == "regression" else "accuracy"
        )

    # 第一轮候选
    candidates = await recommend_candidates(
        metadata=metadata,
        quality=quality,
        task_type=task_type,
        primary_metric=primary_metric,
        provider=provider,
        max_candidates=trials_per_iteration,
    )

    _cap_candidate_budgets(candidates, time_budget_minutes)

    history: List[Dict[str, Any]] = []
    best_run_id: Optional[str] = None
    best_val_score: Optional[float] = None
    no_improvement_count = 0

    for iteration in range(max_iterations):
        if not candidates:
            logger.info(_("llm.search.no_candidates_stop", iteration=iteration + 1))
            break

        logger.info(
            _(
                "llm.search.iteration_info",
                experiment_id=experiment.id,
                iteration=iteration + 1,
                max_iterations=max_iterations,
                n_candidates=len(candidates),
            )
        )

        # 并发提交候选
        tasks = [
            _submit_candidate_via_api(
                experiment.id,
                dataset_id,
                target_column,
                task_type,
                candidate,
                rare_class_strategy=rare_class_strategy,
            )
            for candidate in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        round_history: List[Dict[str, Any]] = []
        for candidate, res in zip(candidates, results):
            trial = await _create_trial(experiment.id, candidate)
            if isinstance(res, Exception):
                logger.warning(_("llm.search.candidate_run_exception", msg=res))
                async with AsyncSessionLocal() as db:
                    result = await db.execute(select(Trial).where(Trial.id == trial.id))
                    trial = result.scalar_one_or_none()
                    if trial:
                        trial.status = "failed"
                        trial.error_message = str(res)
                        trial.completed_at = datetime.now(UTC)
                        await db.commit()
                round_history.append(
                    {"candidate": candidate.model_dump(), "status": "failed", "error": str(res)}
                )
                continue

            run_id, output_dir, status = res
            trial = await _update_trial(trial.id, run_id, output_dir, status)
            entry = {
                "candidate": candidate.model_dump(),
                "status": status,
                "val_score": trial.val_score,
                "test_score": trial.test_score,
            }
            round_history.append(entry)

            if status == "completed" and trial.val_score is not None:
                improved = False
                if best_val_score is None:
                    improved = True
                elif _is_higher_better(primary_metric):
                    improved = trial.val_score > best_val_score
                else:
                    improved = trial.val_score < best_val_score

                if improved:
                    best_val_score = trial.val_score
                    best_run_id = run_id
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

        history.extend(round_history)

        if no_improvement_count >= 2 * trials_per_iteration:
            logger.info(_("llm.search.no_improvement_stop", count=no_improvement_count))
            break

        if iteration < max_iterations - 1:
            candidates = await _propose_next_candidates(
                task_type=task_type,
                primary_metric=primary_metric,
                history=history,
                provider=provider,
            )
            _cap_candidate_budgets(candidates, time_budget_minutes)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Experiment).where(Experiment.id == experiment.id))
        exp = result.scalar_one()
        exp.status = "completed" if best_run_id else "failed"
        exp.best_run_id = best_run_id
        exp.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(exp)

    return exp
