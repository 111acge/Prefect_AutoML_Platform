"""数据加载与元数据分析服务。"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

import pandas as pd


def load_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """根据文件扩展名加载 DataFrame。"""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        # 尝试 UTF-8，失败则尝试 GBK
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="gbk")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def analyze_metadata(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """分析 DataFrame 元数据。"""
    n_samples, n_features = df.shape

    # 推断字段类型
    field_types = {}
    for col in df.columns:
        field_types[col] = infer_field_type(df[col])

    # 缺失率
    missing_rates = {col: float(df[col].isnull().mean()) for col in df.columns}

    # 目标列分布
    target_info = None
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        target_type = field_types[target_column]

        if target_type in ("categorical", "binary"):
            target_info = {
                "type": target_type,
                "unique_values": int(target_series.nunique()),
                "class_distribution": target_series.value_counts().to_dict(),
            }
        else:
            target_info = {
                "type": target_type,
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
            }

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "field_types": field_types,
        "missing_rates": missing_rates,
        "target_info": target_info,
    }


def infer_field_type(series: pd.Series) -> str:
    """推断字段类型。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / max(len(series), 1)
        if unique_ratio < 0.05 and series.nunique() <= 10:
            return "categorical"
        return "numeric"
    else:
        unique_ratio = series.nunique() / max(len(series), 1)
        if unique_ratio > 0.5 and series.nunique() > 100:
            return "text"
        return "categorical"


def preview_dataframe(df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
    """预览 DataFrame。"""
    return {
        "columns": df.columns.tolist(),
        "rows": df.head(n_rows).fillna("").values.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
    }
