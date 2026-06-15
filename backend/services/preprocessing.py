"""数据清洗与特征工程服务。"""

from typing import Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    清洗 DataFrame。

    - 删除重复行
    - 删除目标列缺失的行
    - 数值列用中位数填充
    - 类别列用众数填充
    """
    df = df.copy()

    # 删除重复行
    df = df.drop_duplicates()

    # 删除目标列缺失的行
    df = df.dropna(subset=[target_column])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 从特征列表中排除目标列
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # 数值缺失用中位数填充
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 类别缺失用众数填充
    for col in categorical_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
        else:
            df[col] = df[col].fillna("unknown")

    return df


def engineer_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    特征工程。

    - 对右偏数值列做对数变换
    - 类别列保持原样（交给 AutoGluon 处理）
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    for col in numeric_cols:
        series = df[col]
        # 对右偏分布做对数变换
        if series.min() >= 0 and series.skew() > 1.0:
            df[f"{col}_log"] = np.log1p(series)

    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """划分训练集和测试集。

    分类任务自动使用 stratify 保持类别分布一致。
    """
    stratify = None
    if task_type in ("binary_classification", "multiclass_classification"):
        stratify = df[target_column]

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}
