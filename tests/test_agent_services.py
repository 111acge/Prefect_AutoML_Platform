"""Agent 相关服务单元/集成测试。"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db.sqlite"

import pytest

from config import settings
from database import engine
from models import Base
from schemas import CandidateConfig
from services.llm_client import resolve_providers
from services.llm_strategy_service import recommend_candidates
from services import search_agent
from services.seed_data import ensure_default_dataset

settings.data_dir.mkdir(parents=True, exist_ok=True)


def _async_run(coro):
    """运行异步协程。"""
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


async def _recommend_candidates_no_key():
    """无 API key 时元策略推荐器应返回默认候选。"""
    candidates = await recommend_candidates(
        metadata={
            "n_samples": 100,
            "n_features": 5,
            "target_info": {"type": "categorical", "unique_values": 2},
        },
        quality={"overall_score": 0.8},
        task_type="binary_classification",
        primary_metric="accuracy",
        provider="auto",
    )
    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    assert all(isinstance(c, CandidateConfig) for c in candidates)
    assert all(c.primary_metric == "accuracy" for c in candidates)


def test_recommend_candidates_fallback_no_key():
    _async_run(_recommend_candidates_no_key())


def test_resolve_providers_empty_without_key(monkeypatch):
    """未配置任何 key 时 resolve_providers 返回空列表。"""
    monkeypatch.setenv("KIMI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("MINIMAX_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # 清除 pydantic settings 中可能缓存的值
    from config import settings as cfg

    for key in ["kimi_api_key", "deepseek_api_key", "minimax_api_key", "openai_api_key"]:
        if hasattr(cfg, key):
            monkeypatch.setattr(cfg, key, None)

    providers = resolve_providers("auto")
    assert providers == []


def test_run_experiment_search_loop(tmp_path, monkeypatch):
    """验证 Agent 搜索循环能正确提交候选、读取指标并选出最佳 run。"""

    async def _run():
        dataset = await ensure_default_dataset()

        async def mock_submit(
            experiment_id: str,
            dataset_id: str,
            target_column: str,
            task_type: str,
            candidate: CandidateConfig,
            **kwargs,
        ):
            output_dir = tmp_path / "run_output"
            output_dir.mkdir(exist_ok=True)
            metrics = {
                "val": {"accuracy": 0.85},
                "final": {"accuracy": 0.84},
            }
            (output_dir / "metrics.json").write_text(json.dumps(metrics))
            return ("run-1", str(output_dir), "completed")

        monkeypatch.setattr(search_agent, "_submit_candidate_via_api", mock_submit)

        async def mock_recommend(*args, **kwargs):
            return [
                CandidateConfig(
                    preset="medium_quality",
                    max_models=2,
                    time_budget_minutes=0.1,
                    primary_metric="accuracy",
                )
            ]

        monkeypatch.setattr(search_agent, "recommend_candidates", mock_recommend)

        experiment = await search_agent.run_experiment(
            dataset_id=dataset.id,
            target_column="target",
            task_type="multiclass_classification",
            primary_metric="accuracy",
            max_iterations=1,
            trials_per_iteration=1,
            time_budget_minutes=0.2,
        )

        assert experiment.status == "completed"
        assert experiment.best_run_id == "run-1"

        from sqlalchemy import select
        from database import AsyncSessionLocal
        from models import Trial

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Trial).where(Trial.experiment_id == experiment.id)
            )
            trials = result.scalars().all()

        assert len(trials) == 1
        trial = trials[0]
        assert trial.status == "completed"
        assert trial.val_score == pytest.approx(0.85)
        assert trial.test_score == pytest.approx(0.84)
        assert trial.run_id == "run-1"

    _async_run(_run())


def test_run_experiment_infers_regression_metric(tmp_path, monkeypatch):
    """未指定主指标时，回归任务应默认使用 root_mean_squared_error。"""

    async def _run():
        dataset = await ensure_default_dataset()

        async def mock_submit(
            experiment_id: str,
            dataset_id: str,
            target_column: str,
            task_type: str,
            candidate: CandidateConfig,
            **kwargs,
        ):
            output_dir = tmp_path / "run_reg"
            output_dir.mkdir(exist_ok=True)
            metrics = {
                "val": {"root_mean_squared_error": 0.5},
                "final": {"root_mean_squared_error": 0.6},
            }
            (output_dir / "metrics.json").write_text(json.dumps(metrics))
            return ("run-reg", str(output_dir), "completed")

        monkeypatch.setattr(search_agent, "_submit_candidate_via_api", mock_submit)

        recommend_calls = []

        async def mock_recommend(*args, **kwargs):
            recommend_calls.append(kwargs)
            return [
                CandidateConfig(
                    time_budget_minutes=0.1,
                    primary_metric="root_mean_squared_error",
                )
            ]

        monkeypatch.setattr(search_agent, "recommend_candidates", mock_recommend)

        experiment = await search_agent.run_experiment(
            dataset_id=dataset.id,
            target_column="target",
            task_type="regression",
            primary_metric=None,
            max_iterations=1,
            trials_per_iteration=1,
            time_budget_minutes=0.2,
        )

        assert experiment.status == "completed"
        assert experiment.best_run_id == "run-reg"
        assert recommend_calls[0]["primary_metric"] == "root_mean_squared_error"

    _async_run(_run())
