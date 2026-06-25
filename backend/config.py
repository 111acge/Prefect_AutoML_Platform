# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""应用配置管理。"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。"""

    # 基础路径
    project_root: Path = Path(__file__).parent.parent.resolve()
    data_dir: Path = project_root / "data"
    upload_dir: Path = data_dir / "uploads"
    model_dir: Path = data_dir / "models"
    report_dir: Path = data_dir / "reports"

    # 数据库
    database_url: str = f"sqlite+aiosqlite:///{data_dir / 'db.sqlite'}"

    # 上传限制
    max_upload_size_mb: int = 100

    # AutoML 默认参数
    default_time_budget_minutes: int = 10
    default_max_models: int = 50

    # Prefect
    prefect_home: Path = project_root / ".prefect"

    # LLM 意图解析（可选，失败自动降级）
    llm_provider: str = "auto"  # auto / kimi / deepseek / minimax / glm / openai
    kimi_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    minimax_api_key: Optional[str] = None
    glm_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_llm_model: Optional[str] = None

    # 性能/稳定性优化开关（默认自动，根据数据特征决策；.env 中显式设置可覆盖）
    train_eval_enabled: bool = True
    train_eval_sample_size: Optional[int] = None  # None=自动；0=跳过训练集评估；N=采样N行
    shap_enabled: bool = True
    shap_max_sample_size: Optional[int] = None  # None=自动；0=跳过SHAP；N=最大采样N行
    permutation_importance_enabled: bool = True
    permutation_importance_max_repeats: Optional[int] = None  # None=自动；0=跳过；N=重复N次
    permutation_importance_sample_size: Optional[int] = None  # None=自动；0=不采样；N=采样N行
    data_quality_max_rows: Optional[int] = None  # None=自动；0=强制全量；N=最多N行

    model_config = SettingsConfigDict(
        env_file=str(project_root / ".env"),
        env_file_encoding="utf-8",
    )


settings = Settings()

# 性能优化自动阈值常量
HIGH_CARDINALITY_CLASS_THRESHOLD = 50
EXTREME_CARDINALITY_CLASS_THRESHOLD = 200
LARGE_DATASET_ROW_THRESHOLD = 100_000
DEFAULT_DATA_QUALITY_SAMPLE_SIZE = 50_000
DEFAULT_TRAIN_EVAL_SAMPLE_SIZE = 200
DEFAULT_SHAP_SAMPLE_SIZE_HIGH_CARD = 50
DEFAULT_SHAP_SAMPLE_SIZE_NORMAL = 200
DEFAULT_PERM_IMPORTANCE_REPEATS_HIGH_CARD = 2
DEFAULT_PERM_IMPORTANCE_REPEATS_NORMAL = 5
DEFAULT_PERM_IMPORTANCE_SAMPLE_SIZE = 500

# 确保数据目录存在
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.report_dir.mkdir(parents=True, exist_ok=True)
settings.prefect_home.mkdir(parents=True, exist_ok=True)
