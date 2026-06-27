# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""FastAPI 应用入口。"""

from config import configure_prefect_api_url, get_gpu_summary

configure_prefect_api_url()

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from i18n import _
from i18n.dependencies import get_locale
from routers import datasets, runs, intent, experiments, llm_settings
from services.llm_settings_service import init_llm_config
from services.seed_data import ensure_all_default_datasets
from services.training_executor import training_executor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    # 打印 GPU 状态，便于排查训练加速是否生效
    logger.info("GPU status: %s", get_gpu_summary())
    # 启动时初始化数据库
    await init_db()
    # 加载 LLM 用户配置
    await init_llm_config()
    # 加载默认数据集
    await ensure_all_default_datasets()
    yield
    # 关闭时停止训练执行器后台循环
    training_executor.shutdown()


app = FastAPI(
    title="Prefect AutoML Platform API",
    description=_("app.description"),
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

# 注册路由，注入 locale 依赖
app.include_router(datasets.router, prefix="/api/datasets", dependencies=[Depends(get_locale)])
app.include_router(runs.router, prefix="/api/runs", dependencies=[Depends(get_locale)])
app.include_router(intent.router, prefix="/api/intent", dependencies=[Depends(get_locale)])
app.include_router(experiments.router, prefix="/api/experiments", dependencies=[Depends(get_locale)])
app.include_router(llm_settings.router, prefix="/api/settings", dependencies=[Depends(get_locale)])


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
