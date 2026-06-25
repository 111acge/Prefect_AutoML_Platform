# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""FastAPI 应用入口。"""

import os

os.environ.setdefault("PREFECT_API_URL", "")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import datasets, runs, intent, experiments, llm_settings
from services.llm_settings_service import init_llm_config
from services.seed_data import ensure_default_dataset
from services.training_executor import training_executor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    # 启动时初始化数据库
    await init_db()
    # 加载 LLM 用户配置
    await init_llm_config()
    # 加载默认数据集
    await ensure_default_dataset()
    yield
    # 关闭时停止训练执行器后台循环
    training_executor.shutdown()


app = FastAPI(
    title="Prefect AutoML Platform API",
    description="端到端全自动机器学习平台后端 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(datasets.router, prefix="/api/datasets")
app.include_router(runs.router, prefix="/api/runs")
app.include_router(intent.router, prefix="/api/intent")
app.include_router(experiments.router, prefix="/api/experiments")
app.include_router(llm_settings.router, prefix="/api/settings")


@app.get("/health")
async def health_check():
    """健康检查。"""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """根路径。"""
    return {
        "message": "Prefect AutoML Platform API",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
