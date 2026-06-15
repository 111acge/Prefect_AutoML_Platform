"""Schema 推断、校验与对齐服务。

提供从 DataFrame 推断 Schema、校验数据是否符合 Schema、
以及将数据对齐到 Schema 的能力。
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

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


def infer_field_type(series: pd.Series) -> FieldType:
    """推断字段类型。"""
    nunique = series.nunique(dropna=True)
    total = len(series)
    unique_ratio = nunique / max(total, 1)

    if pd.api.types.is_datetime64_any_dtype(series):
        return FieldType.DATETIME

    if pd.api.types.is_numeric_dtype(series):
        # 整数且唯一值少 → ID
        if pd.api.types.is_integer_dtype(series) and unique_ratio > 0.9:
            return FieldType.ID
        # 数值但唯一值极少 → 类别/二值
        if nunique == 2:
            return FieldType.BINARY
        if unique_ratio < 0.05 and nunique <= 10:
            return FieldType.CATEGORICAL
        return FieldType.NUMERIC

    # 非数值类型
    if nunique == 2:
        return FieldType.BINARY
    if unique_ratio > 0.5 and nunique > 100:
        return FieldType.TEXT
    return FieldType.CATEGORICAL


def infer_schema(df: pd.DataFrame, target_column: str | None = None) -> DatasetSchema:
    """从 DataFrame 推断 Schema。"""
    fields = []
    for col in df.columns:
        series = df[col]
        field_type = infer_field_type(series)

        constraints = FieldConstraint()
        if field_type in (FieldType.NUMERIC, FieldType.DATETIME):
            constraints.min_value = float(series.min()) if pd.notna(series.min()) else None
            constraints.max_value = float(series.max()) if pd.notna(series.max()) else None
        elif field_type in (FieldType.CATEGORICAL, FieldType.BINARY):
            constraints.allowed_values = series.dropna().unique().tolist()
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

        # 检查类别取值
        if constraints.allowed_values is not None:
            actual_values = set(series.dropna().unique())
            allowed = set(constraints.allowed_values)
            unexpected = actual_values - allowed
            if unexpected:
                errors.append(f"字段 '{field_schema.name}' 出现未允许的取值: {sorted(unexpected)}")

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
