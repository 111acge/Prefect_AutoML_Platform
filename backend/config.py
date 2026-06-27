# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""应用配置管理。"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def get_gpu_info() -> Dict[str, Any]:
    """检测 GPU 可用性与基本信息。

    优先尊重环境变量显式设置：
    - USE_GPU=false / 0 / no / off：强制关闭 GPU
    - CUDA_VISIBLE_DEVICES=""：强制关闭 GPU
    """
    info: Dict[str, Any] = {
        "available": False,
        "count": 0,
        "name": None,
        "cuda_version": None,
        "torch_cuda_available": False,
    }

    env_use_gpu = os.getenv("USE_GPU", "").lower()
    if env_use_gpu in ("0", "false", "no", "off"):
        return info

    env_cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if env_cuda_devices is not None and env_cuda_devices.strip() == "":
        return info

    try:
        import torch

        if torch.cuda.is_available():
            info["torch_cuda_available"] = True
            info["available"] = True
            info["count"] = torch.cuda.device_count()
            if info["count"] > 0:
                info["name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        # torch 未安装或检测失败时保持关闭
        pass

    return info


def is_gpu_available() -> bool:
    """返回当前环境是否可用 GPU。"""
    return get_gpu_info()["available"]


def get_gpu_summary() -> str:
    """返回用于日志的 GPU 状态字符串。"""
    info = get_gpu_info()
    if not info["available"]:
        return "GPU not available (CPU mode)"
    parts = [f"{info['count']} GPU(s) available"]
    if info["name"]:
        parts.append(info["name"])
    if info["cuda_version"]:
        parts.append(f"CUDA {info['cuda_version']}")
    return ", ".join(parts)


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
    prefect_api_url: Optional[str] = "http://localhost:4200/api"
    prefect_enabled: bool = True
    prefect_flow_name: str = "automl-end-to-end"
    prefect_deployment_name: str = "automl-end-to-end"

    # 国际化默认语言
    default_locale: str = "zh-CN"  # zh-CN / en

    # LLM 意图解析（可选，失败自动降级）
    llm_provider: str = "auto"  # auto / kimi / deepseek / minimax / glm / openai
    kimi_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    minimax_api_key: Optional[str] = None
    glm_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_llm_model: Optional[str] = None

    # GPU 加速（默认自动检测 CUDA；可通过 USE_GPU=false 显式关闭）
    use_gpu: bool = is_gpu_available()
    num_gpus: int = 1

    @field_validator("num_gpus", mode="after")
    @classmethod
    def _validate_num_gpus(cls, v: int, info) -> int:
        """GPU 未启用时强制 num_gpus=0。"""
        if not info.data.get("use_gpu"):
            return 0
        return max(1, v)

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


def configure_prefect_api_url() -> None:
    """将配置中的 Prefect API URL 写回环境变量，确保 prefect 库使用同一地址。"""
    if settings.prefect_api_url is not None:
        os.environ["PREFECT_API_URL"] = settings.prefect_api_url





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

def get_recommended_num_cpus(reserved: int = 1) -> int:
    """根据实际 CPU 核心数动态返回推荐的并行核心数。

    Args:
        reserved: 保留给系统/其他进程的核心数，默认 1。

    Returns:
        至少为 1 的可用核心数。
    """
    cpus = os.cpu_count() or 1
    return max(1, cpus - reserved)


# 确保数据目录存在
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.report_dir.mkdir(parents=True, exist_ok=True)
settings.prefect_home.mkdir(parents=True, exist_ok=True)
