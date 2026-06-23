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
    """更新数据集请求（目标列 / 任务类型）。

    当字段为空时，系统会根据数据自动推断。
    """

    target_column: Optional[str] = None
    task_type: Optional[str] = Field(
        default=None,
        pattern="^(binary_classification|multiclass_classification|regression)$",
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


class ValueConstraint(BaseModel):
    """列取值约束。"""

    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class CleaningRules(BaseModel):
    """数据清洗规则配置。"""

    remove_duplicates: bool = True
    drop_rows_with_missing_target: bool = True
    numeric_impute_strategy: str = "median"  # median / mean / constant
    numeric_impute_constant: Optional[float] = 0.0
    categorical_impute_strategy: str = "mode"  # mode / constant
    categorical_impute_constant: Optional[str] = "missing"
    drop_columns: List[str] = Field(default_factory=list)
    value_constraints: List[ValueConstraint] = Field(default_factory=list)


class CandidateConfig(BaseModel):
    """单次候选运行的配置参数（Agent 搜索空间）。"""

    preset: Optional[str] = Field(default="auto")
    max_models: Optional[int] = Field(default=None, ge=1, le=200)
    time_budget_minutes: Optional[float] = Field(default=None, ge=0.1)
    primary_metric: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0)
    feature_engineering_enabled: Optional[bool] = None
    cleaning_rules: Optional[CleaningRules] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_strategy: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


class RunCreate(BaseModel):
    """启动训练任务请求。

    target_column / task_type 为空时，从数据集 schema_info 自动推断。
    mode 控制创建后是否立即执行：auto（一键训练，默认），step（仅创建草稿，等待单步触发）。
    """

    dataset_id: str
    target_column: Optional[str] = None
    task_type: Optional[str] = Field(
        default=None,
        pattern="^(binary_classification|multiclass_classification|regression)$",
    )
    primary_metric: Optional[str] = None
    # None 表示不限制训练时间（无穷大）
    time_budget_minutes: Optional[float] = Field(default=10, ge=0.1)
    max_models: int = Field(default=50, ge=1, le=200)
    preset: Optional[str] = Field(default="auto")
    seed: Optional[int] = Field(default=None, ge=0)
    feature_engineering_enabled: bool = Field(default=True)
    experiment_id: Optional[str] = None
    candidate_config: Optional[CandidateConfig] = None
    mode: str = Field(default="auto", pattern="^(auto|step)$")


class ExperimentCreate(BaseModel):
    """创建实验请求。"""

    dataset_id: str
    target_column: Optional[str] = None
    task_type: Optional[str] = Field(
        default=None,
        pattern="^(binary_classification|multiclass_classification|regression)$",
    )
    primary_metric: Optional[str] = None
    max_iterations: int = Field(default=5, ge=1, le=20)
    trials_per_iteration: int = Field(default=2, ge=1, le=5)
    time_budget_minutes: Optional[float] = Field(default=10, ge=0.1)


class ExperimentResponse(BaseModel):
    """实验响应。"""

    id: str
    dataset_id: str
    status: str
    search_config: Optional[Dict[str, Any]]
    best_run_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class TrialResponse(BaseModel):
    """Trial 响应。"""

    id: str
    experiment_id: str
    run_id: Optional[str]
    candidate_params: Optional[Dict[str, Any]]
    val_score: Optional[float]
    test_score: Optional[float]
    primary_metric: Optional[str]
    status: str
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class RunResponse(BaseModel):
    """训练任务响应。"""

    id: str
    dataset_id: str
    experiment_id: Optional[str]
    status: str
    time_budget_minutes: Optional[float]
    primary_metric: Optional[str]
    output_dir: str
    prefect_flow_run_id: Optional[str]
    error_message: Optional[str]
    config: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class RunStepResponse(BaseModel):
    """训练任务原子步骤响应。"""

    id: str
    run_id: str
    step_name: str
    status: str
    sequence: int
    input_manifest: Optional[Dict[str, Any]] = None
    output_manifest: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StepExecutionRequest(BaseModel):
    """执行单个步骤请求。"""

    step_name: Optional[str] = None


class RunResult(BaseModel):
    """训练结果。"""

    run_id: str
    status: str
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float]
    extended_metrics: Optional[Dict[str, Any]] = None
    train_metrics: Optional[Dict[str, float]] = None
    cv_results: Optional[Dict[str, Any]] = None
    leaderboard: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]
    permutation_importance: Optional[List[Dict[str, Any]]] = None
    model_path: Optional[str]
    report_path: Optional[str]
    business_interpretation: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    """预测请求。"""

    data: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """预测响应。"""

    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    threshold: Optional[float] = None


class ExplainRequest(BaseModel):
    """单样本解释请求。"""

    data: List[Dict[str, Any]]


class ExplainResponse(BaseModel):
    """单样本解释响应。"""

    base_value: float
    prediction: Any
    problem_type: str
    features: List[Dict[str, Any]]


class DatasetQualityResponse(BaseModel):
    """数据集质量报告响应。"""

    n_rows: int
    n_columns: int
    n_features: int
    overall_score: float
    completeness: Dict[str, Any]
    consistency: Dict[str, Any]
    accuracy: Dict[str, Any]
    timeliness: Dict[str, Any]
    uniqueness: Dict[str, Any]
    validity: Dict[str, Any]
    target_info: Dict[str, Any]
    warnings: List[str]


class DatasetConnectRequest(BaseModel):
    """数据库连接上传请求。"""

    connection_type: str = Field(
        ..., pattern="^(mysql|postgresql|clickhouse|sqlite)$"
    )
    connection_params: Dict[str, Any]
    query: str
    name: Optional[str] = None


class RunCompareRequest(BaseModel):
    """跨 Run 模型对比请求。"""

    run_ids: List[str] = Field(..., min_length=2, max_length=10)


class RunCompareItem(BaseModel):
    """单个 Run 的对比信息。"""

    run_id: str
    dataset_name: Optional[str] = None
    status: str
    primary_metric: Optional[str] = None
    metrics: Dict[str, float]
    best_model: Optional[str] = None
    best_model_score: Optional[float] = None
    feature_count: Optional[int] = None


class RunCompareResponse(BaseModel):
    """跨 Run 模型对比响应。"""

    runs: List[RunCompareItem]
    metric_name: Optional[str] = None
    best_run_id: Optional[str] = None
