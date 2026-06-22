"""Schema 推断、校验与对齐服务。

提供从 DataFrame 推断 Schema、校验数据是否符合 Schema、
以及将数据对齐到 Schema 的能力。
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import warnings

import pandas as pd


class FieldType(str, Enum):
    """支持的字段类型。"""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    DATETIME = "datetime"
    ID = "id"


@dataclass
class FieldConstraint:
    """字段约束。"""

    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    allowed_values: list[Any] | None = None
    regex: str | None = None


@dataclass
class FieldSchema:
    """字段 Schema 定义。"""

    name: str
    field_type: FieldType
    nullable: bool = True
    constraints: FieldConstraint = field(default_factory=FieldConstraint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field_type": self.field_type.value,
            "nullable": self.nullable,
            "constraints": {
                k: v
                for k, v in [
                    ("min_value", self.constraints.min_value),
                    ("max_value", self.constraints.max_value),
                    ("min_length", self.constraints.min_length),
                    ("max_length", self.constraints.max_length),
                    ("allowed_values", self.constraints.allowed_values),
                    ("regex", self.constraints.regex),
                ]
                if v is not None
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldSchema":
        constraints = FieldConstraint(**data.get("constraints", {}))
        return cls(
            name=data["name"],
            field_type=FieldType(data["field_type"]),
            nullable=data.get("nullable", True),
            constraints=constraints,
        )


@dataclass
class DatasetSchema:
    """数据集 Schema 定义。"""

    fields: list[FieldSchema]
    target_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [f.to_dict() for f in self.fields],
            "target_column": self.target_column,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSchema":
        return cls(
            fields=[FieldSchema.from_dict(f) for f in data["fields"]],
            target_column=data.get("target_column"),
        )

    def get_field(self, name: str) -> FieldSchema | None:
        """按名称获取字段定义。"""
        for f in self.fields:
            if f.name == name:
                return f
        return None


def _looks_like_id(name: str) -> bool:
    """根据列名语义判断是否像 ID/标识列。"""
    lower = name.lower()
    id_patterns = {
        "id", "idx", "index", "key", "uuid", "serial", "code", "no", "num",
        "编号", "序号", "编码", "单号", "流水号", "标识", "主键", "键",
    }
    return any(p in lower for p in id_patterns)


def _is_datetime_string(series: pd.Series, threshold: float = 0.7) -> bool:
    """判断 object 列是否可以被解析为日期。"""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce")
    parsed_ratio = parsed.notna().mean()
    return parsed_ratio >= threshold


def infer_field_type(series: pd.Series, name: str = "") -> FieldType:
    """推断字段类型。

    对任意数据集鲁棒：考虑样本规模、列名语义、日期字符串解析等。
    """
    nunique = series.nunique(dropna=True)
    total = len(series)
    unique_ratio = nunique / max(total, 1)

    if pd.api.types.is_datetime64_any_dtype(series):
        return FieldType.DATETIME

    # object/string 列尝试识别日期
    if _is_datetime_string(series):
        return FieldType.DATETIME

    if pd.api.types.is_numeric_dtype(series):
        # 整数且几乎唯一：需同时满足列名像 ID 或呈现单调性才判为 ID
        if pd.api.types.is_integer_dtype(series) and unique_ratio > 0.9:
            if _looks_like_id(name) or _is_sequential(series):
                return FieldType.ID
        # 数值但唯一值极少 → 类别/二值；阈值随样本量自适应
        if nunique == 2:
            return FieldType.BINARY
        # 小样本（<100）时允许最多 10 个唯一值；大样本时按 5% 比例
        categorical_cutoff = min(10, max(2, int(total * 0.05)))
        if nunique <= categorical_cutoff:
            return FieldType.CATEGORICAL
        return FieldType.NUMERIC

    # 非数值类型
    if nunique == 2:
        return FieldType.BINARY

    # 高基数类别列：数量极大或几乎唯一，视为 ID（如单据号、流水号）
    # 需结合列名语义，避免误伤合法高基数类别（如门店编码）
    if nunique > 100_000 or (unique_ratio > 0.95 and nunique > 10_000):
        if _looks_like_id(name) or unique_ratio > 0.99:
            return FieldType.ID

    # 文本：半唯一且唯一值较多
    if unique_ratio > 0.5 and nunique > 100:
        return FieldType.TEXT

    return FieldType.CATEGORICAL


def _is_sequential(series: pd.Series) -> bool:
    """判断整数列是否近似单调（如 1,2,3,... 的行号）。"""
    try:
        sorted_vals = pd.to_numeric(series.dropna(), errors="coerce").sort_values().reset_index(drop=True)
        if len(sorted_vals) <= 1:
            return False
        diffs = sorted_vals.diff().dropna()
        # 单调递增且步长为 1 的比例高
        return float((diffs == 1).mean()) > 0.95
    except Exception:
        return False


def infer_schema(df: pd.DataFrame, target_column: str | None = None) -> DatasetSchema:
    """从 DataFrame 推断 Schema。"""
    fields = []
    for col in df.columns:
        series = df[col]
        field_type = infer_field_type(series, name=col)

        constraints = FieldConstraint()
        if field_type in (FieldType.NUMERIC, FieldType.DATETIME):
            constraints.min_value = float(series.min()) if pd.notna(series.min()) else None
            constraints.max_value = float(series.max()) if pd.notna(series.max()) else None
        elif field_type in (FieldType.CATEGORICAL, FieldType.BINARY):
            # allowed_values 仅作记录，不用于硬性校验，避免未知类别导致失败
            constraints.allowed_values = sorted(
                series.dropna().unique().tolist(),
                key=lambda x: str(x),
            )[:1000]
        elif field_type == FieldType.TEXT:
            lengths = series.dropna().astype(str).str.len()
            constraints.min_length = int(lengths.min()) if len(lengths) > 0 else None
            constraints.max_length = int(lengths.max()) if len(lengths) > 0 else None

        nullable = bool(series.isnull().any())

        fields.append(
            FieldSchema(
                name=str(col),
                field_type=field_type,
                nullable=nullable,
                constraints=constraints,
            )
        )

    return DatasetSchema(fields=fields, target_column=target_column)


class SchemaValidationError(Exception):
    """Schema 校验错误。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_against_schema(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """校验 DataFrame 是否符合 Schema，返回错误列表。"""
    errors = []

    # 检查必填字段是否存在
    schema_field_names = {f.name for f in schema.fields}
    missing = schema_field_names - set(df.columns)
    if missing:
        errors.append(f"缺少必填字段: {sorted(missing)}")

    for field_schema in schema.fields:
        if field_schema.name not in df.columns:
            continue

        series = df[field_schema.name]

        # 检查非空约束
        if not field_schema.nullable and series.isnull().any():
            errors.append(f"字段 '{field_schema.name}' 不允许为空，但存在空值")

        # 检查类型兼容性
        if field_schema.field_type in (
            FieldType.NUMERIC,
            FieldType.BINARY,
            FieldType.CATEGORICAL,
        ):
            if not pd.api.types.is_numeric_dtype(series) and not _is_string_like(series):
                errors.append(f"字段 '{field_schema.name}' 类型不匹配，期望数值或可解析为数值")

        # 检查数值范围
        constraints = field_schema.constraints
        if constraints.min_value is not None or constraints.max_value is not None:
            try:
                numeric_series = pd.to_numeric(series, errors="coerce")
                if (
                    constraints.min_value is not None
                    and numeric_series.min() < constraints.min_value
                ):
                    errors.append(
                        f"字段 '{field_schema.name}' 最小值 {numeric_series.min()} 低于允许范围 {constraints.min_value}"
                    )
                if (
                    constraints.max_value is not None
                    and numeric_series.max() > constraints.max_value
                ):
                    errors.append(
                        f"字段 '{field_schema.name}' 最大值 {numeric_series.max()} 超过允许范围 {constraints.max_value}"
                    )
            except Exception:
                pass

        # 类别取值：仅作记录，不再硬性报错，由预处理统一处理未知类别
        # if constraints.allowed_values is not None:
        #     ...

    return errors


def _is_string_like(series: pd.Series) -> bool:
    """判断 Series 是否为字符串类型或可被解析为字符串。"""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def align_to_schema(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """将 DataFrame 对齐到 Schema（类型转换、补充缺失列）。"""
    aligned = df.copy()

    for field_schema in schema.fields:
        if field_schema.name not in aligned.columns:
            # 补充缺失列
            if field_schema.nullable:
                aligned[field_schema.name] = pd.NA
            else:
                if field_schema.field_type == FieldType.NUMERIC:
                    aligned[field_schema.name] = 0
                elif field_schema.field_type in (
                    FieldType.CATEGORICAL,
                    FieldType.BINARY,
                    FieldType.TEXT,
                ):
                    aligned[field_schema.name] = ""
                elif field_schema.field_type == FieldType.DATETIME:
                    aligned[field_schema.name] = pd.NaT
                else:
                    aligned[field_schema.name] = pd.NA
            continue

        series = aligned[field_schema.name]

        # 类型转换
        if field_schema.field_type == FieldType.NUMERIC:
            aligned[field_schema.name] = pd.to_numeric(series, errors="coerce")
        elif field_schema.field_type == FieldType.DATETIME:
            aligned[field_schema.name] = pd.to_datetime(series, errors="coerce")
        elif field_schema.field_type in (FieldType.CATEGORICAL, FieldType.BINARY, FieldType.TEXT):
            aligned[field_schema.name] = (
                series.astype(str).replace("nan", pd.NA).replace("None", pd.NA)
            )

    return aligned


def build_schema_from_file(
    file_path: str | Path, target_column: str | None = None
) -> DatasetSchema:
    """从文件构建 Schema。"""
    from services.data_service import load_dataframe

    df = load_dataframe(file_path)
    return infer_schema(df, target_column=target_column)
