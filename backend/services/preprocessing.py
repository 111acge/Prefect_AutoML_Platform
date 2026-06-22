"""数据清洗与特征工程服务。"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


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

    # 数值缺失用中位数填充；全 NaN 列用 0 填充，避免下游 scaler/模型失败
    for col in numeric_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(median_val)

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


def _stratified_split_with_presence(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """分层划分，并强制保证每个类别在训练集和测试集中都至少出现 1 次。

    - 对于只有 1 个样本的类别，会复制出 1 条副本，分别放入训练/测试集。
    - 其余样本按 ``test_size`` 近似比例随机分配。
    """
    rng = np.random.RandomState(random_state)
    df = df.reset_index(drop=True)

    train_idx: List[int] = []
    test_idx: List[int] = []
    remainder_idx: List[int] = []

    for _, group in df.groupby(target_column):
        idx = group.index.tolist()
        if len(idx) == 1:
            # 复制 singleton，训练/测试各放一条
            df = pd.concat([df, df.loc[idx].copy()], ignore_index=True)
            idx = [idx[0], len(df) - 1]
        # 每个类别至少选 1 条给 train、1 条给 test
        chosen = rng.choice(idx, size=2, replace=False).tolist()
        train_idx.append(chosen[0])
        test_idx.append(chosen[1])
        remainder_idx.extend([i for i in idx if i not in chosen])

    n_test_target = int(round(len(df) * test_size)) - len(test_idx)
    n_test_target = max(0, min(n_test_target, len(remainder_idx)))
    if n_test_target > 0:
        test_rem = rng.choice(remainder_idx, size=n_test_target, replace=False).tolist()
    else:
        test_rem = []
    train_rem = [i for i in remainder_idx if i not in test_rem]

    train_df = df.loc[train_idx + train_rem].reset_index(drop=True)
    test_df = df.loc[test_idx + test_rem].reset_index(drop=True)
    return train_df, test_df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
    rare_class_strategy: str = "auto",
) -> Dict[str, pd.DataFrame]:
    """划分训练集和测试集。

    分类任务自动使用 stratify 保持类别分布一致。
    - 目标列缺失的行会在划分前被移除。
    - 稀有类别处理策略：
      - auto / oversample：对样本数 < 2 的类别进行复制，保证每个类别在
        训练/测试集都至少出现 1 次，从而完成 stratify/CV。
      - drop：过滤样本数 < 2 的类别（保留旧行为）。
      - none：不处理；若存在样本数 < 2 的类别则关闭 stratify。
    - 若总样本数不足以按类别分层（测试集容量小于类别数），则回退到普通随机划分。
    """
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    if task_type in ("binary_classification", "multiclass_classification"):
        class_counts = df[target_column].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        valid_classes = class_counts.index.tolist()

        if rare_classes:
            if rare_class_strategy in ("auto", "oversample"):
                logger.warning(
                    f"检测到稀有类别 {rare_classes}（样本数 < 2），"
                    f"将自动复制样本以保证训练/测试集均包含这些类别。"
                )
            elif rare_class_strategy == "drop":
                logger.warning(
                    f"检测到稀有类别 {rare_classes}（样本数 < 2），已过滤。"
                )
                valid_classes = class_counts[class_counts >= 2].index.tolist()
                df = df[df[target_column].isin(valid_classes)].reset_index(drop=True)
                class_counts = df[target_column].value_counts()
            else:
                logger.warning(
                    f"检测到稀有类别 {rare_classes}（样本数 < 2），"
                    f"rare_class_strategy='{rare_class_strategy}' 未处理，关闭 stratify。"
                )
                valid_classes = class_counts[class_counts >= 2].index.tolist()

        if len(valid_classes) < 2:
            raise ValueError(
                f"目标列 '{target_column}' 有效类别数不足 2，无法进行分类训练。"
                f"原始类别分布：\n{class_counts.to_dict()}"
            )

        # 只有当测试集能容纳每个类别至少 1 条时才启用分层划分
        can_stratify = int(len(df) * test_size) >= len(valid_classes)
        if can_stratify and rare_class_strategy != "none":
            train_df, test_df = _stratified_split_with_presence(
                df, target_column, test_size, random_state
            )
            return {"train": train_df, "test": test_df}

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=None
    )
    return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "regression",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    rare_class_strategy: str = "auto",
) -> Dict[str, pd.DataFrame]:
    """严格划分训练集、验证集、测试集。

    分类任务自动分层；验证集/测试集只在最终评估和候选选择时使用，
    不得用于拟合任何预处理/模型参数。
    """
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError("val_size 与 test_size 必须非负且之和小于 1")

    # 先划分出 test
    first_split = split_data(
        df,
        target_column,
        task_type=task_type,
        test_size=(val_size + test_size),
        random_state=random_state,
        rare_class_strategy=rare_class_strategy,
    )
    train_df = first_split["train"]
    rest_df = first_split["test"]

    # 再从剩余部分划分 val / test
    val_ratio = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0
    if val_ratio <= 0:
        return {"train": train_df, "val": pd.DataFrame(), "test": rest_df}

    second_split = split_data(
        rest_df,
        target_column,
        task_type=task_type,
        test_size=(1 - val_ratio),
        random_state=random_state,
        rare_class_strategy=rare_class_strategy,
    )
    val_df = second_split["train"]
    test_df = second_split["test"]
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
