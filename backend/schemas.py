"""Pydantic 数据模型。"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class DatasetCreate(BaseModel):
    """创建数据集请求。"""

    name: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None


class DatasetUpdate(BaseModel):
    """更新数据集请求（目标列 / 任务类型）。"""

    target_column: str
    task_type: str = Field(
        ..., pattern="^(binary_classification|multiclass_classification|regression)$"
    )


class SchemaField(BaseModel):
    """Schema 字段定义。"""

    name: str
    field_type: str
    nullable: bool = True
    constraints: Dict[str, Any] = Field(default_factory=dict)


class DatasetSchemaResponse(BaseModel):
    """数据集 Schema 响应。"""

    fields: List[SchemaField]
    target_column: Optional[str] = None


class SchemaValidationResponse(BaseModel):
    """Schema 校验结果响应。"""

    valid: bool
    errors: List[str] = Field(default_factory=list)


class DatasetResponse(BaseModel):
    """数据集响应。"""

    id: str
    name: str
    file_path: str
    file_size_bytes: int
    row_count: Optional[int]
    column_count: Optional[int]
    target_column: Optional[str]
    task_type: Optional[str]
    schema_info: Optional[Dict[str, Any]]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetPreview(BaseModel):
    """数据集预览。"""

    columns: List[str]
    rows: List[List[Any]]
    dtypes: Dict[str, str]
    shape: tuple


class RunCreate(BaseModel):
    """启动训练任务请求。"""

    dataset_id: str
    target_column: str
    task_type: str = Field(
        ..., pattern="^(binary_classification|multiclass_classification|regression)$"
    )
    primary_metric: Optional[str] = None
    time_budget_minutes: float = Field(default=10, ge=0.1, le=180)
    max_models: int = Field(default=50, ge=1, le=200)
    preset: Optional[str] = Field(default="auto")
    seed: Optional[int] = Field(default=None, ge=0)


class RunResponse(BaseModel):
    """训练任务响应。"""

    id: str
    dataset_id: str
    status: str
    time_budget_minutes: float
    primary_metric: Optional[str]
    output_dir: str
    prefect_flow_run_id: Optional[str]
    error_message: Optional[str]
    config: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class RunResult(BaseModel):
    """训练结果。"""

    run_id: str
    status: str
    metrics: Dict[str, float]
    extended_metrics: Optional[Dict[str, Any]] = None
    train_metrics: Optional[Dict[str, float]] = None
    leaderboard: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]
    model_path: Optional[str]
    report_path: Optional[str]


class PredictionRequest(BaseModel):
    """预测请求。"""

    data: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """预测响应。"""

    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
