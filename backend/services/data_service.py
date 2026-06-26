# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""数据加载与元数据分析服务。"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import pandas as pd

from i18n import _
from services.schema_service import infer_field_type as _schema_infer_field_type


# 目标列推断：中文/英文常见目标列关键词
_TARGET_KEYWORDS = {
    # 英文
    "target", "label", "y", "class", "outcome", "prediction", "predict",
    "is_", "has_", "will_", "flag", "churn", "default", "fraud", "spam",
    "risk", "score", "grade", "category",
    # 中文
    "目标", "标签", "是否", "类别", "分类", "结果", "预测", "违约",
    "流失", "欺诈", "风险", "等级", "评分", "类型", "状态",
}

# ID/标识列关键词：这些列不应被选为目标
_ID_KEYWORDS = {
    "id", "idx", "index", "key", "code", "no", "num", "serial", "uuid",
    "编号", "序号", "编码", "单号", "流水号", "标识", "主键", "键",
}


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
    elif suffix in (".jsonl", ".json"):
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(_("data.unsupported_format", suffix=suffix))


def _column_name_score(col: str) -> float:
    """根据列名语义给目标列候选打分。"""
    lower = col.lower()
    # 命中目标关键词强加分
    for kw in _TARGET_KEYWORDS:
        if kw in lower:
            return 1.0
    # 命中 ID 关键词直接判为不适合
    for kw in _ID_KEYWORDS:
        if kw in lower:
            return -1.0
    return 0.0


def infer_target_column(
    df: pd.DataFrame,
    hint: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """启发式推断最可能的目标列。

    返回 (推荐列名, 置信度)。置信度 >= 0.7 表示比较可靠；
    置信度 < 0.3 时应由前端引导用户确认。
    """
    if hint and hint in df.columns:
        return hint, 1.0

    if df.empty or len(df.columns) == 0:
        return None, 0.0

    candidates = []
    n_rows = len(df)
    for idx, col in enumerate(df.columns):
        series = df[col]
        n_unique = series.nunique(dropna=True)
        missing_rate = series.isnull().mean()
        unique_ratio = n_unique / max(n_rows, 1)

        # 基本分：从 0.5 开始
        score = 0.5

        # 列名语义
        name_score = _column_name_score(col)
        if name_score < 0:
            # ID/标识列不适合作为目标
            continue
        score += name_score * 0.4

        # 位置惩罚：第一列经常是 ID/序号
        if idx == 0:
            score -= 0.15

        # 缺失率惩罚
        if missing_rate > 0.5:
            score -= 0.3
        elif missing_rate > 0.2:
            score -= 0.1

        # 唯一值比例惩罚：ID 列通常唯一值比例接近 1
        if unique_ratio > 0.95:
            score -= 0.4
        elif unique_ratio > 0.8:
            score -= 0.2

        # 常量列不适合作为目标
        if n_unique <= 1:
            continue

        # 目标列理想基数：2-100（分类）或连续（回归）
        if 2 <= n_unique <= 100:
            score += 0.1

        candidates.append((col, score))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_score = candidates[0]
    # 将原始分数压缩到 [0, 1] 区间作为置信度
    confidence = min(max(best_score, 0.0), 1.0)
    return best_col, confidence


def infer_task_type(series: pd.Series, field_type: Optional[str] = None) -> str:
    """根据目标列自动推断任务类型。"""
    if field_type is None:
        field_type = infer_field_type(series)

    nunique = series.nunique(dropna=True)

    if nunique <= 1:
        # 常量目标无法训练，但先返回一个默认类型，由后续校验拦截
        return "binary_classification"

    if nunique == 2:
        return "binary_classification"

    # 数值型且唯一值较多 → 回归
    if field_type == "numeric":
        # 即使是整数，如果唯一值超过 10 个也倾向回归
        if nunique > 10:
            return "regression"
        # 3-10 个唯一值的数值目标，按多分类处理（如评分 1-5）
        return "multiclass_classification"

    if field_type in ("categorical", "binary", "text"):
        # 类别目标：唯一值较多但有限时多分类；过多时可能是文本目标，但按多分类处理
        if nunique <= 200:
            return "multiclass_classification"
        # 类别数过多，可能是高基数文本，回归不太合适，仍按多分类但后续会拦截
        return "multiclass_classification"

    # 默认兜底
    return "regression" if field_type == "numeric" else "multiclass_classification"


def analyze_metadata(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """分析 DataFrame 元数据。

    当 target_column / task_type 未提供时，会自动推断并写入结果中的
    suggested_target_column / suggested_task_type。
    """
    n_samples, n_features = df.shape

    # 推断字段类型（复用 schema_service 的入口，确保口径一致）
    field_types = {}
    for col in df.columns:
        field_types[col] = infer_field_type(df[col])

    # 缺失率
    missing_rates = {col: float(df[col].isnull().mean()) for col in df.columns}

    # 高基数类别列检测（用于后续训练策略规避风险模型如 CatBoost）
    high_cardinality_columns = []
    cardinality_info = {}
    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        if field_types.get(col) in ("categorical", "id", "text"):
            nunique = series.nunique(dropna=True)
            ratio = nunique / max(len(series), 1)
            cardinality_info[col] = {"nunique": int(nunique), "unique_ratio": float(ratio)}
            # 唯一值比例 >30% 或 >50000 个唯一值视为高基数
            if ratio > 0.3 or nunique > 50_000:
                high_cardinality_columns.append(col)

    # 目标列推断
    suggested_target_column: Optional[str] = None
    target_confidence = 0.0
    if target_column is None:
        suggested_target_column, target_confidence = infer_target_column(df)
        effective_target = suggested_target_column
    else:
        effective_target = target_column

    # 任务类型推断
    suggested_task_type: Optional[str] = None
    if task_type is None and effective_target and effective_target in df.columns:
        suggested_task_type = infer_task_type(
            df[effective_target], field_types.get(effective_target)
        )
        effective_task_type = suggested_task_type
    elif task_type is not None:
        effective_task_type = task_type
    else:
        effective_task_type = None

    # 目标列分布
    target_info = None
    if effective_target and effective_target in df.columns:
        target_series = df[effective_target]
        target_type = field_types[effective_target]

        if target_type in ("categorical", "binary"):
            target_info = {
                "type": target_type,
                "unique_values": int(target_series.nunique()),
                "class_distribution": target_series.value_counts().to_dict(),
            }
        else:
            target_info = {
                "type": target_type,
                "mean": float(target_series.mean()) if pd.notna(target_series.mean()) else None,
                "std": float(target_series.std()) if pd.notna(target_series.std()) else None,
                "min": float(target_series.min()) if pd.notna(target_series.min()) else None,
                "max": float(target_series.max()) if pd.notna(target_series.max()) else None,
            }

    result = {
        "n_samples": n_samples,
        "n_features": n_features,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "field_types": field_types,
        "missing_rates": missing_rates,
        "cardinality_info": cardinality_info,
        "high_cardinality_columns": high_cardinality_columns,
        "target_info": target_info,
    }

    if suggested_target_column is not None:
        result["suggested_target_column"] = suggested_target_column
        result["suggested_target_confidence"] = round(target_confidence, 4)

    if suggested_task_type is not None:
        result["suggested_task_type"] = suggested_task_type

    return result


def infer_field_type(series: pd.Series) -> str:
    """推断字段类型（复用 schema_service 并补齐 binary）。"""
    return _schema_infer_field_type(series).value


def preview_dataframe(df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
    """预览 DataFrame。"""
    return {
        "columns": df.columns.tolist(),
        "rows": df.head(n_rows).fillna("").values.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
    }
