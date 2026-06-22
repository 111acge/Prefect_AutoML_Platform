"""数据质量评估服务。

按照数据质量六维模型输出报告：
- 完整性（Completeness）：缺失值
- 一致性（Consistency）：重复行、常量列、数据类型一致性
- 准确性（Accuracy）：异常值、越界值
- 时效性（Timeliness）：时间列范围、未来日期
- 唯一性（Uniqueness）：ID 列、主键重复
- 有效性（Validity）：Schema 约束违反、非法取值
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from config import (
    settings,
    LARGE_DATASET_ROW_THRESHOLD,
    DEFAULT_DATA_QUALITY_SAMPLE_SIZE,
)


def assess_data_quality(
    df: pd.DataFrame,
    target_column: str,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """评估数据质量并返回六维报告。

    Args:
        df: 输入数据
        target_column: 目标列名
        max_rows: 最大采样行数；None 时使用配置 DATA_QUALITY_MAX_ROWS，
                  配置也为 None 时不对数据采样。
    """
    effective_max_rows = max_rows if max_rows is not None else settings.data_quality_max_rows
    if effective_max_rows is None and len(df) > LARGE_DATASET_ROW_THRESHOLD:
        effective_max_rows = DEFAULT_DATA_QUALITY_SAMPLE_SIZE
    if effective_max_rows is not None and effective_max_rows > 0 and len(df) > effective_max_rows:
        df = df.sample(n=effective_max_rows, random_state=42)

    feature_cols = [c for c in df.columns if c != target_column]
    n_rows, n_cols = df.shape

    # ---------------- 完整性 ----------------
    missing_rates = {col: float(df[col].isnull().mean()) for col in df.columns}
    rows_with_missing = int(df.isnull().any(axis=1).sum())
    completeness = {
        "score": round(1.0 - (df.isnull().sum().sum() / max(df.size, 1)), 4),
        "missing_rates": missing_rates,
        "rows_with_missing": rows_with_missing,
        "columns_with_missing": [c for c, r in missing_rates.items() if r > 0],
    }

    # ---------------- 唯一性 ----------------
    duplicated_rows = int(df.duplicated().sum())
    id_like_cols = []
    for col in feature_cols:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.9:
            id_like_cols.append({"column": col, "unique_ratio": round(unique_ratio, 4)})

    uniqueness = {
        "score": round(1.0 - (duplicated_rows / max(n_rows, 1)), 4),
        "duplicated_rows": duplicated_rows,
        "id_like_columns": id_like_cols,
    }

    # ---------------- 一致性 ----------------
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    zero_variance_cols = [c for c in feature_cols if df[c].nunique(dropna=True) == 0]
    mixed_type_cols = _detect_mixed_types(df, feature_cols)
    consistency = {
        "score": round(
            1.0
            - (
                (len(constant_cols) + len(mixed_type_cols))
                / max(len(feature_cols), 1)
            ),
            4,
        ),
        "constant_columns": constant_cols,
        "zero_variance_columns": zero_variance_cols,
        "mixed_type_columns": mixed_type_cols,
    }

    # ---------------- 准确性 ----------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    outlier_summary = {}
    negative_non_negative = []
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
                "outlier_ratio": round(n_outliers / len(series), 4),
            }
        # 简单启发：列名含 age/income/price/salary/amount 等应为非负
        if any(k in col.lower() for k in ("age", "income", "price", "salary", "amount", "count", "quantity")):
            n_negative = int((series < 0).sum())
            if n_negative > 0:
                negative_non_negative.append({"column": col, "negative_count": n_negative})

    accuracy = {
        "score": round(
            1.0
            - (
                sum(v["outlier_ratio"] for v in outlier_summary.values())
                / max(len(numeric_cols), 1)
            ),
            4,
        ),
        "outlier_summary": outlier_summary,
        "negative_in_non_negative_columns": negative_non_negative,
    }

    # ---------------- 时效性 ----------------
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    timeliness_issues = []
    now = datetime.now()
    for col in datetime_cols:
        series = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        future_count = int((series > now).sum())
        if future_count > 0:
            timeliness_issues.append({"column": col, "future_dates": future_count})
    timeliness = {
        "score": round(
            1.0
            - (sum(i["future_dates"] for i in timeliness_issues) / max(n_rows * len(datetime_cols), 1)),
            4,
        ),
        "datetime_columns": datetime_cols.tolist() if hasattr(datetime_cols, "tolist") else list(datetime_cols),
        "future_date_issues": timeliness_issues,
    }

    # ---------------- 有效性 ----------------
    # 检测类别列中出现次数极少的取值（可能是拼写错误）
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    rare_categories = []
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        rare = counts[counts == 1].index.tolist()
        if rare:
            rare_categories.append({"column": col, "rare_values_count": len(rare), "examples": rare[:5]})

    validity = {
        "score": round(
            1.0
            - (sum(r["rare_values_count"] for r in rare_categories) / max(n_rows, 1)),
            4,
        ),
        "rare_category_issues": rare_categories,
    }

    # 目标列分析
    target_info = _analyze_target(df, target_column)

    overall_score = round(
        (
            completeness["score"]
            + consistency["score"]
            + accuracy["score"]
            + timeliness["score"]
            + uniqueness["score"]
            + validity["score"]
        )
        / 6,
        4,
    )

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "n_features": len(feature_cols),
        "overall_score": overall_score,
        "completeness": completeness,
        "consistency": consistency,
        "accuracy": accuracy,
        "timeliness": timeliness,
        "uniqueness": uniqueness,
        "validity": validity,
        "target_info": target_info,
        "warnings": _generate_warnings(
            completeness,
            consistency,
            accuracy,
            timeliness,
            uniqueness,
            validity,
            target_info,
        ),
    }


def _detect_mixed_types(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """检测 object 列中是否混合了明显不同的类型（如数字与字符串）。"""
    mixed = []
    for col in feature_cols:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna().head(1000)
        if len(non_null) == 0:
            continue
        numeric_like = pd.to_numeric(non_null, errors="coerce")
        ratio = numeric_like.notna().mean()
        if 0.0 < ratio < 1.0:
            mixed.append({"column": col, "numeric_like_ratio": round(ratio, 4)})
    return mixed


def _analyze_target(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """分析目标列。"""
    if target_column not in df.columns:
        return {}
    target_series = df[target_column]
    info = {
        "type": (
            "categorical"
            if target_series.dtype == "object" or target_series.nunique() <= 10
            else "numeric"
        ),
        "missing_count": int(target_series.isnull().sum()),
        "unique_values": int(target_series.nunique(dropna=True)),
    }
    if info["type"] == "categorical":
        info["class_distribution"] = target_series.value_counts().to_dict()
    return info


def _generate_warnings(
    completeness: Dict[str, Any],
    consistency: Dict[str, Any],
    accuracy: Dict[str, Any],
    timeliness: Dict[str, Any],
    uniqueness: Dict[str, Any],
    validity: Dict[str, Any],
    target_info: Dict[str, Any],
) -> List[str]:
    """生成数据质量警告。"""
    warnings = []

    high_missing = [c for c, r in completeness["missing_rates"].items() if r > 0.5]
    if high_missing:
        warnings.append(f"完整性：以下列缺失率超过 50%: {high_missing}")

    if completeness["rows_with_missing"] > 0:
        warnings.append(f"完整性：{completeness['rows_with_missing']} 行存在缺失值")

    if consistency["constant_columns"]:
        warnings.append(f"一致性：常量/零方差列 {consistency['constant_columns']}")

    if consistency["mixed_type_columns"]:
        names = [c["column"] for c in consistency["mixed_type_columns"]]
        warnings.append(f"一致性：以下列存在混合类型 {names}")

    if accuracy["outlier_summary"]:
        names = list(accuracy["outlier_summary"].keys())
        warnings.append(f"准确性：以下数值列存在异常值 {names}")

    if accuracy["negative_in_non_negative_columns"]:
        names = [c["column"] for c in accuracy["negative_in_non_negative_columns"]]
        warnings.append(f"准确性：以下列出现负值 {names}")

    if timeliness["future_date_issues"]:
        names = [c["column"] for c in timeliness["future_date_issues"]]
        warnings.append(f"时效性：以下时间列包含未来日期 {names}")

    if uniqueness["duplicated_rows"] > 0:
        warnings.append(f"唯一性：存在 {uniqueness['duplicated_rows']} 条重复行")

    if validity["rare_category_issues"]:
        names = [c["column"] for c in validity["rare_category_issues"]]
        warnings.append(f"有效性：以下类别列存在稀有取值（可能是错误） {names}")

    if target_info.get("missing_count", 0) > 0:
        warnings.append(f"目标列存在 {target_info['missing_count']} 个缺失值，训练前会被删除")

    return warnings
