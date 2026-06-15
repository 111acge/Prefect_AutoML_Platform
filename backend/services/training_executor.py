"""训练任务执行器。

将 AutoML 训练流程放到独立子进程中异步执行，避免阻塞 FastAPI 事件循环，
同时通过信号量控制并发训练任务数量。
"""

import asyncio
import json
import os
import sys
from asyncio import Semaphore
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from models import Run


@dataclass
class TrainingJob:
    """训练任务运行时信息。"""

    run_id: str
    file_path: str
    target_column: str
    task_type: str
    output_dir: Path
    time_budget_minutes: float
    preset: str
    primary_metric: str | None
    seed: int | None
    max_models: int
    process: asyncio.subprocess.Process | None = None
    status: str = "pending"
    error_message: str | None = None
    callbacks: list[Callable[[str, str | None], Awaitable[None]]] = field(default_factory=list)


class TrainingExecutor:
    """异步训练任务执行器。"""

    def __init__(self, max_concurrent_jobs: int = 2):
        """初始化执行器。

        Args:
            max_concurrent_jobs: 最大并发训练任务数，默认 2。
        """
        self._semaphore = Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, TrainingJob] = {}

    def get_job(self, run_id: str) -> TrainingJob | None:
        """获取任务状态。"""
        return self._jobs.get(run_id)

    def list_jobs(self) -> list[TrainingJob]:
        """列出所有追踪中的任务。"""
        return list(self._jobs.values())

    async def submit(self, **kwargs: Any) -> TrainingJob:
        """提交训练任务。"""
        run_id = kwargs["run_id"]
        job = TrainingJob(
            run_id=run_id,
            file_path=kwargs["file_path"],
            target_column=kwargs["target_column"],
            task_type=kwargs["task_type"],
            output_dir=Path(kwargs["output_dir"]),
            time_budget_minutes=kwargs["time_budget_minutes"],
            preset=kwargs["preset"],
            primary_metric=kwargs.get("primary_metric"),
            seed=kwargs.get("seed"),
            max_models=kwargs.get("max_models", 50),
        )
        self._jobs[run_id] = job

        # 使用 create_task 让任务在后台运行，避免阻塞接口返回
        asyncio.create_task(self._run_with_semaphore(job))
        return job

    async def _run_with_semaphore(self, job: TrainingJob) -> None:
        """在信号量控制下执行任务。"""
        async with self._semaphore:
            await self._execute(job)

    async def _update_run(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
        set_completed: bool = False,
    ) -> None:
        """异步更新任务状态。"""
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Run).where(Run.id == run_id))
            run = result.scalar_one_or_none()
            if run is None:
                return
            run.status = status
            if error_message is not None:
                run.error_message = error_message
            if set_completed:
                run.completed_at = datetime.now(UTC)
            await db.commit()

    async def _save_metrics(self, run_id: str, output_dir: Path) -> None:
        """将 metrics.json 中的指标保存到数据库。"""
        from models import Metric

        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            return

        async with AsyncSessionLocal() as db:
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return

            all_metrics: dict[str, Any] = {}
            all_metrics.update(metrics_data.get("final", {}))
            all_metrics.update(metrics_data.get("extended", {}))

            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (int, float)):
                    db.add(
                        Metric(
                            run_id=run_id,
                            metric_name=str(metric_name),
                            metric_value=float(metric_value),
                        )
                    )
            await db.commit()

    async def _execute(self, job: TrainingJob) -> None:
        """执行训练子进程。"""
        await self._update_run(job.run_id, "running")
        job.status = "running"

        job.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = job.output_dir / "training.log"

        script_path = settings.project_root / "scripts" / "run_flow.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--file-path",
            job.file_path,
            "--target-column",
            job.target_column,
            "--task-type",
            job.task_type,
            "--output-dir",
            str(job.output_dir),
            "--time-budget-minutes",
            str(job.time_budget_minutes),
            "--preset",
            job.preset,
        ]
        if job.primary_metric:
            cmd.extend(["--primary-metric", job.primary_metric])
        if job.seed is not None:
            cmd.extend(["--seed", str(job.seed)])
        cmd.extend(["--max-models", str(job.max_models)])

        env = os.environ.copy()
        env["PREFECT_API_URL"] = ""

        try:
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(f"[{datetime.now(UTC).isoformat()}] 启动训练任务: {job.run_id}\n")
            log_file.flush()

            job.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

            # 等待子进程结束，设置超时为预算时间 + 5 分钟缓冲
            timeout = job.time_budget_minutes * 60 + 300
            returncode = await asyncio.wait_for(job.process.wait(), timeout=timeout)

            if returncode != 0:
                log_file.write(
                    f"\n[{datetime.now(UTC).isoformat()}] 训练失败 (exit {returncode})\n"
                )
                log_file.flush()
                raise RuntimeError(f"Flow 执行失败 (exit {returncode})，详见 training.log")

            log_file.write(f"\n[{datetime.now(UTC).isoformat()}] 训练完成\n")
            log_file.flush()

            await self._save_metrics(job.run_id, job.output_dir)
            await self._update_run(job.run_id, "completed", set_completed=True)
            job.status = "completed"

        except asyncio.TimeoutError:
            if job.process is not None and job.process.returncode is None:
                job.process.kill()
                await job.process.wait()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now(UTC).isoformat()}] ERROR: 训练超时\n")
            await self._update_run(job.run_id, "failed", "训练超时", set_completed=True)
            job.status = "failed"
            job.error_message = "训练超时"

        except Exception as e:
            if job.process is not None and job.process.returncode is None:
                job.process.kill()
                await job.process.wait()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now(UTC).isoformat()}] ERROR: {str(e)}\n")
            await self._update_run(job.run_id, "failed", str(e), set_completed=True)
            job.status = "failed"
            job.error_message = str(e)

        finally:
            if "log_file" in locals():
                log_file.close()
            for callback in job.callbacks:
                try:
                    await callback(job.status, job.error_message)
                except Exception:
                    pass


# 全局单例
training_executor = TrainingExecutor(max_concurrent_jobs=2)
