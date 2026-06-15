"""数据质量评估服务。"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd


def assess_data_quality(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """评估数据质量并返回摘要报告。"""
    feature_cols = [c for c in df.columns if c != target_column]
    n_rows, n_cols = df.shape

    # 缺失值分析
    missing_rates = {col: float(df[col].isnull().mean()) for col in df.columns}
    rows_with_missing = int(df.isnull().any(axis=1).sum())

    # 零方差 / 常量列
    constant_cols = []
    zero_variance_cols = []
    for col in feature_cols:
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)
        if df[col].nunique(dropna=True) == 0:
            zero_variance_cols.append(col)

    # ID 列检测（唯一值比例 > 90%）
    id_like_cols = []
    for col in feature_cols:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.9:
            id_like_cols.append(col)

    # 高基数类别列（唯一值 > 100 或 > 10% 样本数）
    high_cardinality_cols = []
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 100 or n_unique / max(len(df), 1) > 0.1:
            high_cardinality_cols.append({"column": col, "unique_values": int(n_unique)})

    # 异常值概览（IQR 法）
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        if n_outliers > 0:
            outlier_summary[col] = {
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": n_outliers,
                "outlier_ratio": float(n_outliers / len(series)),
            }

    # 目标列分析
    target_info = {}
    if target_column in df.columns:
        target_series = df[target_column]
        target_info = {
            "type": (
                "categorical"
                if target_series.dtype == "object" or target_series.nunique() <= 10
                else "numeric"
            ),
            "missing_count": int(target_series.isnull().sum()),
            "unique_values": int(target_series.nunique(dropna=True)),
        }
        if target_info["type"] == "categorical":
            target_info["class_distribution"] = target_series.value_counts().to_dict()

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "n_features": len(feature_cols),
        "rows_with_missing": rows_with_missing,
        "missing_rates": missing_rates,
        "constant_columns": constant_cols,
        "zero_variance_columns": zero_variance_cols,
        "id_like_columns": id_like_cols,
        "high_cardinality_columns": high_cardinality_cols,
        "outlier_summary": outlier_summary,
        "target_info": target_info,
        "warnings": _generate_warnings(
            missing_rates, constant_cols, id_like_cols, high_cardinality_cols, target_info
        ),
    }


def _generate_warnings(
    missing_rates: Dict[str, float],
    constant_cols: List[str],
    id_like_cols: List[str],
    high_cardinality_cols: List[Dict[str, Any]],
    target_info: Dict[str, Any],
) -> List[str]:
    """生成数据质量警告。"""
    warnings = []

    high_missing = [col for col, rate in missing_rates.items() if rate > 0.5]
    if high_missing:
        warnings.append(f"以下列缺失率超过 50%: {high_missing}")

    if constant_cols:
        warnings.append(f"以下列为常量/零方差列: {constant_cols}")

    if id_like_cols:
        warnings.append(f"以下列疑似 ID 列（唯一值比例 > 90%）: {id_like_cols}")

    if high_cardinality_cols:
        names = [item["column"] for item in high_cardinality_cols]
        warnings.append(f"以下类别列基数过高: {names}")

    if target_info.get("missing_count", 0) > 0:
        warnings.append(f"目标列存在 {target_info['missing_count']} 个缺失值，训练前会被删除")

    return warnings
