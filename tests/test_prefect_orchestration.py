# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""Prefect 编排集成测试。"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# 使用独立的测试数据库
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test_prefect.db.sqlite"

import pandas as pd
import pytest

from config import settings
from database import engine
from models import Base


@pytest.fixture(autouse=True)
def setup_database():
    """每个测试前重置数据库。"""
    async def _reset():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_reset())
    yield

    async def _drop():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    asyncio.run(_drop())


# ---------------------------------------------------------------------------
# 本地 Flow 执行测试（不依赖 Prefect Server）
# ---------------------------------------------------------------------------

def test_automl_pipeline_local_execution(monkeypatch):
    """验证 automl_pipeline 可在本地（ ephemeral 模式）完成一次端到端训练。"""
    # 本地 Flow 测试使用 ephemeral Prefect 模式，不依赖 Prefect Server
    monkeypatch.setenv("PREFECT_API_URL", "")
    from prefect_flows.automl_flow import automl_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        # 保证训练/验证/测试集均包含两类，避免 ROC AUC 计算失败
        df = pd.DataFrame({
            "a": list(range(40)),
            "b": list(range(1, 41)),
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
        })
        df.to_csv(csv_path, index=False)

        result = automl_pipeline(
            file_path=str(csv_path),
            target_column="target",
            task_type="binary_classification",
            output_dir=tmpdir,
            time_budget_minutes=1.0,
            preset="medium_quality",
            primary_metric="accuracy",
            seed=42,
            max_models=5,
        )

        assert result["status"] == "completed"
        assert "metrics" in result
        assert (Path(tmpdir) / "metrics.json").exists()
        assert (Path(tmpdir) / "report.html").exists()
        assert (Path(tmpdir) / "training.log").exists()


# ---------------------------------------------------------------------------
# TrainingExecutor Prefect 集成测试
# ---------------------------------------------------------------------------

def _prefect_server_reachable() -> bool:
    """检查 Prefect Server 是否可达。"""
    try:
        from prefect.client.orchestration import get_client

        async def _check() -> bool:
            async with get_client() as client:
                await client.hello()
            return True

        return asyncio.run(_check())
    except Exception:
        return False


def test_build_flow_parameters():
    """验证 TrainingExecutor 能正确构造 Flow 参数。"""
    from services.training_executor import TrainingExecutor, TrainingJob

    executor = TrainingExecutor(max_concurrent_jobs=1)
    job = TrainingJob(
        run_id="test-run",
        file_path="/tmp/data.csv",
        target_column="target",
        task_type="binary_classification",
        output_dir=Path("/tmp/output"),
        time_budget_minutes=1.0,
        preset="medium_quality",
        primary_metric="accuracy",
        seed=42,
        max_models=10,
        cleaning_rules={"drop_columns": ["id"]},
        feature_engineering_enabled=False,
        candidate_config={"max_models": 5},
        rare_class_strategy="copy",
    )
    params = executor._build_flow_parameters(job)

    assert params["target_column"] == "target"
    assert params["task_type"] == "binary_classification"
    assert params["time_budget_minutes"] == 1.0
    assert params["preset"] == "medium_quality"
    assert params["max_models"] == 10
    assert params["feature_engineering_enabled"] is False
    assert params["rare_class_strategy"] == "copy"
    assert Path(params["file_path"]).is_absolute()
    assert Path(params["output_dir"]).is_absolute()


@pytest.mark.skipif(not _prefect_server_reachable(), reason="Prefect Server 未启动")
def test_prefect_server_orchestration():
    """验证 TrainingExecutor 能通过 Prefect Server 提交并完成 Flow Run。"""
    import uuid

    # 确保 TrainingExecutor 连接到本地 Prefect Server
    os.environ["PREFECT_API_URL"] = settings.prefect_api_url or "http://localhost:4200/api"

    from services.training_executor import TrainingExecutor

    executor = TrainingExecutor(max_concurrent_jobs=1)
    run_id = f"test-prefect-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        # 保证训练/验证/测试集均包含两类，避免 ROC AUC 计算失败
        df = pd.DataFrame({
            "a": list(range(40)),
            "b": list(range(1, 41)),
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
        })
        df.to_csv(csv_path, index=False)

        executor.submit_sync(
            run_id=run_id,
            file_path=str(csv_path),
            target_column="target",
            task_type="binary_classification",
            output_dir=tmpdir,
            time_budget_minutes=1.0,
            preset="medium_quality",
            primary_metric="accuracy",
            seed=42,
            max_models=5,
        )

        # 最多等待 5 分钟
        deadline = time.time() + 300
        while time.time() < deadline:
            job = executor.get_job(run_id)
            if job and job.status in ("completed", "failed"):
                break
            time.sleep(2)

        job = executor.get_job(run_id)
        assert job is not None
        assert job.status == "completed", f"Flow Run 失败: {job.error_message}"
        assert job.prefect_flow_run_id is not None
        assert (Path(tmpdir) / "metrics.json").exists()
        assert (Path(tmpdir) / "training.log").exists()

    executor.shutdown()
