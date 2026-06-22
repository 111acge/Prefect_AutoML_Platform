"""实验（Experiment / Agent 搜索）API 测试。"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db.sqlite"

import pytest
from fastapi.testclient import TestClient

from config import settings
from database import engine
from main import app
from models import Base
from services.seed_data import ensure_default_dataset

settings.data_dir.mkdir(parents=True, exist_ok=True)

client = TestClient(app)


def _async_run(coro):
    """运行异步协程。"""
    import asyncio

    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def setup_database():
    """每个测试前重置数据库并加载默认数据集。"""

    async def _reset():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        await ensure_default_dataset()

    _async_run(_reset())
    yield

    async def _drop():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    _async_run(_drop())


def test_create_experiment_starts_background_search(monkeypatch):
    """创建实验应持久化记录并后台启动搜索任务。"""
    spy_call = {}

    def mock_run_experiment(**kwargs):
        spy_call.update(kwargs)
        return SimpleNamespace(
            id=kwargs.get("experiment_id"),
            status="running",
            best_run_id=None,
        )

    monkeypatch.setattr("routers.experiments.run_experiment", mock_run_experiment)

    created_tasks = []

    def fake_create_task(coro):
        created_tasks.append(coro)
        return coro

    monkeypatch.setattr("routers.experiments.asyncio.create_task", fake_create_task)

    datasets_resp = client.get("/api/datasets")
    assert datasets_resp.status_code == 200
    dataset_id = datasets_resp.json()[0]["id"]

    response = client.post(
        "/api/experiments",
        json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "task_type": "multiclass_classification",
            "primary_metric": "accuracy",
            "max_iterations": 2,
            "trials_per_iteration": 1,
            "time_budget_minutes": 0.5,
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["dataset_id"] == dataset_id
    assert data["status"] == "running"
    assert "id" in data

    assert spy_call["dataset_id"] == dataset_id
    assert spy_call["experiment_id"] == data["id"]
    assert spy_call["time_budget_minutes"] == 0.5
    assert spy_call["max_iterations"] == 2
    assert spy_call["trials_per_iteration"] == 1


def test_list_and_get_experiments():
    """实验列表与详情查询。"""
    datasets_resp = client.get("/api/datasets")
    dataset_id = datasets_resp.json()[0]["id"]

    create_resp = client.post(
        "/api/experiments",
        json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "task_type": "multiclass_classification",
            "max_iterations": 1,
            "trials_per_iteration": 1,
            "time_budget_minutes": 0.1,
        },
    )
    assert create_resp.status_code == 202
    experiment_id = create_resp.json()["id"]

    list_resp = client.get("/api/experiments")
    assert list_resp.status_code == 200
    assert any(e["id"] == experiment_id for e in list_resp.json())

    get_resp = client.get(f"/api/experiments/{experiment_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == experiment_id

    trials_resp = client.get(f"/api/experiments/{experiment_id}/trials")
    assert trials_resp.status_code == 200
    assert isinstance(trials_resp.json(), list)
