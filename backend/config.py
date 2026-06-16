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
    llm_provider: str = "auto"  # auto / kimi / deepseek / minimax / openai
    kimi_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    minimax_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_llm_model: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=str(project_root / ".env"),
        env_file_encoding="utf-8",
    )


settings = Settings()

# 确保数据目录存在
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.report_dir.mkdir(parents=True, exist_ok=True)
settings.prefect_home.mkdir(parents=True, exist_ok=True)
