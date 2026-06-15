"""API 测试。"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# 使用独立的测试数据库，避免影响生产数据
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db.sqlite"

import pytest
from fastapi.testclient import TestClient

from config import settings
from database import engine
from main import app
from models import Base

# 确保测试数据库目录存在
settings.data_dir.mkdir(parents=True, exist_ok=True)

client = TestClient(app)


def _async_run(coro):
    """运行异步协程。"""
    return asyncio.run(coro)


@pytest.fixture
def default_dataset():
    """创建默认数据集记录。"""
    from services.seed_data import ensure_default_dataset

    return _async_run(ensure_default_dataset())


@pytest.fixture(autouse=True)
def setup_database():
    """每个测试前重置数据库并加载默认数据集。"""
    from services.seed_data import ensure_default_dataset

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


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Prefect AutoML Platform API" in response.json()["message"]


def test_default_dataset_loaded():
    """测试默认数据集是否已自动加载。"""
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "iris"
    assert data[0]["target_column"] == "target"


def test_upload_dataset():
    """测试数据集上传。"""
    csv_content = "a,b,target\n1,2,0\n3,4,1\n5,6,0\n"
    response = client.post(
        "/api/datasets/upload",
        data={"name": "test_data"},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_data"
    assert data["row_count"] == 3
    assert data["column_count"] == 3


def test_create_run(default_dataset):
    """测试创建训练任务。"""
    response = client.post(
        "/api/runs",
        json={
            "dataset_id": default_dataset.id,
            "target_column": "target",
            "task_type": "multiclass_classification",
            "time_budget_minutes": 0.1,
            "preset": "medium_quality",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["dataset_id"] == default_dataset.id


def test_get_run_not_found():
    """测试获取不存在的任务。"""
    response = client.get("/api/runs/non-existent-id")
    assert response.status_code == 404


def test_predict_before_completion(default_dataset, monkeypatch):
    """测试任务未完成时预测应失败。

    TestClient 会等待后台任务执行完毕，因此需要 mock 掉后台执行，
    才能验证 pending 状态下的预测拦截。
    """
    from fastapi import BackgroundTasks

    monkeypatch.setattr(BackgroundTasks, "add_task", lambda *args, **kwargs: None)

    response = client.post(
        "/api/runs",
        json={
            "dataset_id": default_dataset.id,
            "target_column": "target",
            "task_type": "multiclass_classification",
            "time_budget_minutes": 0.1,
            "preset": "medium_quality",
        },
    )
    run_id = response.json()["id"]

    response = client.post(
        f"/api/runs/{run_id}/predict",
        json={"data": [{"a": 1, "b": 2}]},
    )
    assert response.status_code == 400
    assert "尚未完成" in response.json()["detail"]


@pytest.mark.slow
def test_end_to_end_training(default_dataset):
    """端到端训练测试（耗时较长，默认跳过）。"""
    response = client.post(
        "/api/runs",
        json={
            "dataset_id": default_dataset.id,
            "target_column": "target",
            "task_type": "multiclass_classification",
            "time_budget_minutes": 1,
            "preset": "medium_quality",
        },
    )
    run_id = response.json()["id"]

    # 等待训练完成
    for _ in range(60):
        import time

        time.sleep(2)
        response = client.get(f"/api/runs/{run_id}")
        if response.json()["status"] in ("completed", "failed"):
            break

    assert response.json()["status"] == "completed"

    # 测试结果接口
    response = client.get(f"/api/runs/{run_id}/results")
    assert response.status_code == 200
    results = response.json()
    assert len(results["metrics"]) > 0
    assert len(results["leaderboard"]) > 0

    # 测试日志接口
    response = client.get(f"/api/runs/{run_id}/logs")
    assert response.status_code == 200
    assert "启动训练任务" in response.text

    # 测试预测接口
    response = client.post(
        f"/api/runs/{run_id}/predict",
        json={
            "data": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        },
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1

    # 测试报告下载并验证报告包含可视化图表
    response = client.get(f"/api/runs/{run_id}/report")
    assert response.status_code == 200
    report_html = response.text
    assert "<img" in report_html
    assert "特征重要性" in report_html
    assert "混淆矩阵" in report_html

    # 测试扩展指标已返回
    response = client.get(f"/api/runs/{run_id}/results")
    assert response.status_code == 200
    assert "extended_metrics" in response.json()
    assert response.json()["extended_metrics"].get("confusion_matrix") is not None

    # 测试模型下载
    response = client.get(f"/api/runs/{run_id}/model")
    assert response.status_code == 200
