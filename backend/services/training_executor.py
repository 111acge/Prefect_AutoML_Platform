"""训练任务执行器。

将 AutoML 训练流程放到独立子进程中异步执行，避免阻塞 FastAPI 事件循环，
同时通过信号量控制并发训练任务数量。
"""

import asyncio
import json
import os
import queue
import sys
import threading
from asyncio import Semaphore
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

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
    # None 表示训练时间不限制（无穷大）
    time_budget_minutes: float | None
    preset: str
    primary_metric: str | None
    seed: int | None
    max_models: int
    cleaning_rules: Dict[str, Any] | None = None
    feature_engineering_enabled: bool = True
    candidate_config: Dict[str, Any] | None = None
    rare_class_strategy: str = "auto"
    # None 表示端到端执行；否则为单步名称
    step: str | None = None
    process: asyncio.subprocess.Process | None = None
    status: str = "pending"
    error_message: str | None = None
    callbacks: list[Callable[[str, str | None], Awaitable[None]]] = field(default_factory=list)


class TrainingExecutor:
    """异步训练任务执行器。"""

    # 即使 time_budget_minutes 为 None，也设置硬超时，防止子进程无限挂起
    DEFAULT_HARD_TIMEOUT_MINUTES = 240

    def __init__(self, max_concurrent_jobs: int = 2):
        """初始化执行器。

        Args:
            max_concurrent_jobs: 最大并发训练任务数，默认 2。
        """
        self._semaphore = Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, TrainingJob] = {}
        # SSE 状态订阅：run_id -> set(queue.Queue)
        self._status_subscribers: dict[str, set[queue.Queue]] = {}

        # 使用独立后台事件循环运行训练任务，避免被 FastAPI/TestClient 的请求生命周期取消
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """后台线程运行的独立事件循环。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def shutdown(self, timeout: float = 30.0) -> None:
        """优雅关闭后台事件循环与线程。"""
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if not self._loop.is_closed():
            self._loop.close()

    def get_job(self, run_id: str) -> TrainingJob | None:
        """获取任务状态。"""
        return self._jobs.get(run_id)

    def subscribe_status(self, run_id: str, q: queue.Queue) -> None:
        """订阅指定 run 的状态变化（SSE 用）。"""
        self._status_subscribers.setdefault(run_id, set()).add(q)

    def unsubscribe_status(self, run_id: str, q: queue.Queue) -> None:
        """取消订阅。"""
        subs = self._status_subscribers.get(run_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._status_subscribers.pop(run_id, None)

    def _notify_status(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
        step: str | None = None,
    ) -> None:
        """通知所有订阅者状态变化，可选携带当前步骤信息。"""
        subs = self._status_subscribers.get(run_id)
        if not subs:
            return
        data = {"status": status}
        if error_message is not None:
            data["error_message"] = error_message
        if step is not None:
            data["step"] = step
        for q in list(subs):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

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
            cleaning_rules=kwargs.get("cleaning_rules"),
            feature_engineering_enabled=kwargs.get("feature_engineering_enabled", True),
            candidate_config=kwargs.get("candidate_config"),
            rare_class_strategy=kwargs.get("rare_class_strategy", "auto"),
            step=kwargs.get("step"),
        )
        self._jobs[run_id] = job

        # 使用 create_task 让任务在后台运行，避免阻塞接口返回
        asyncio.create_task(self._run_with_semaphore(job))
        return job

    def submit_sync(self, **kwargs: Any) -> TrainingJob:
        """在独立后台事件循环中提交训练任务（线程安全）。"""
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
            cleaning_rules=kwargs.get("cleaning_rules"),
            feature_engineering_enabled=kwargs.get("feature_engineering_enabled", True),
            candidate_config=kwargs.get("candidate_config"),
            rare_class_strategy=kwargs.get("rare_class_strategy", "auto"),
            step=kwargs.get("step"),
        )
        self._jobs[run_id] = job

        asyncio.run_coroutine_threadsafe(
            self._run_with_semaphore(job), self._loop
        )
        return job

    def submit_step_sync(self, **kwargs: Any) -> TrainingJob:
        """在独立后台事件循环中提交单步执行任务（线程安全）。"""
        return self.submit_sync(**kwargs)

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
        step: str | None = None,
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

        # 通知 SSE 订阅者
        self._notify_status(run_id, status, error_message, step=step)

    def _read_error_message(self, output_dir: Path) -> str | None:
        """读取子进程写入的 error.json，获取真实失败原因。"""
        error_path = output_dir / "error.json"
        if not error_path.exists():
            return None
        try:
            data = json.loads(error_path.read_text(encoding="utf-8"))
            return data.get("error_message") or data.get("message")
        except (json.JSONDecodeError, OSError):
            return None

    def _has_partial_model(self, output_dir: Path) -> bool:
        """检查是否存在已保存的部分模型（用于全局超时后 Best-so-far 恢复）。"""
        model_dir = output_dir / "autogluon_models"
        if not model_dir.exists():
            return False
        # 至少包含一个模型文件/目录
        return any(model_dir.iterdir())

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

        # 完全原子化执行：使用 run_step.py
        script_path = settings.project_root / "scripts" / "run_step.py"
        step_arg = job.step if job.step is not None else "all"
        cmd = [
            sys.executable,
            str(script_path),
            "--run-id",
            job.run_id,
            "--output-dir",
            str(job.output_dir),
            "--step",
            step_arg,
        ]

        env = os.environ.copy()
        env["PREFECT_API_URL"] = ""
        # 强制子进程 stdout/stderr 使用 UTF-8，避免 Windows 重定向时写入 cp936 导致日志解码失败
        env["PYTHONIOENCODING"] = "utf-8"

        async def _pump_logs(reader: asyncio.StreamReader, writer) -> None:
            """实时过滤并写入子进程 stdout，丢弃 Prefect 内部噪音。"""
            noisy_patterns = (
                "EventsWorker failed",
                "Service 'EventsWorker' failed",
                "GlobalEventLoopThread | prefect._internal.concurrency",
                "Service .* failed with",
            )
            while True:
                try:
                    line = await reader.readline()
                except Exception:
                    break
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                if any(pattern in text for pattern in noisy_patterns):
                    continue
                writer.write(text)
                writer.flush()

        try:
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(f"[{datetime.now(UTC).isoformat()}] 启动训练任务: {job.run_id}\n")
            log_file.flush()

            job.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

            # 同时泵送日志并等待子进程结束
            # 当 time_budget_minutes 为 None 时使用硬超时，避免子进程无限挂起
            pump_task = asyncio.create_task(_pump_logs(job.process.stdout, log_file))
            if job.time_budget_minutes is None:
                timeout = self.DEFAULT_HARD_TIMEOUT_MINUTES * 60
            else:
                timeout = job.time_budget_minutes * 60 + 60
            returncode = await asyncio.wait_for(job.process.wait(), timeout=timeout)
            await pump_task

            if returncode != 0:
                error_message = (
                    self._read_error_message(job.output_dir)
                    or f"Flow 执行失败 (exit {returncode})，详见 training.log"
                )
                log_file.write(
                    f"\n[{datetime.now(UTC).isoformat()}] 训练失败 (exit {returncode}): {error_message}\n"
                )
                log_file.flush()
                raise RuntimeError(error_message)

            log_file.write(f"\n[{datetime.now(UTC).isoformat()}] 训练完成\n")
            log_file.flush()

            await self._save_metrics(job.run_id, job.output_dir)
            await self._update_run(job.run_id, "completed", set_completed=True)
            job.status = "completed"

        except asyncio.TimeoutError:
            if job.process is not None and job.process.returncode is None:
                job.process.kill()
                await job.process.wait()

            # 全局超时后尝试 Best-so-far 恢复：若 AutoGluon 已保存部分模型，则视为完成
            if self._has_partial_model(job.output_dir):
                warning = (
                    "训练在全局超时时完成，已返回 AutoGluon 当前最优模型。"
                    "部分产物（如报告、扩展指标）可能不完整。"
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.now(UTC).isoformat()}] {warning}\n")
                await self._save_metrics(job.run_id, job.output_dir)
                await self._update_run(job.run_id, "completed", warning, set_completed=True)
                job.status = "completed"
                job.error_message = warning
            else:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.now(UTC).isoformat()}] ERROR: 训练超时，且未找到可用模型\n")
                await self._update_run(job.run_id, "failed", "训练超时，且未找到可用模型", set_completed=True)
                job.status = "failed"
                job.error_message = "训练超时，且未找到可用模型"

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
            if "pump_task" in locals() and not pump_task.done():
                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
            if "log_file" in locals():
                log_file.close()
            for callback in job.callbacks:
                try:
                    await callback(job.status, job.error_message)
                except Exception:
                    pass


# 全局单例
training_executor = TrainingExecutor(max_concurrent_jobs=2)
