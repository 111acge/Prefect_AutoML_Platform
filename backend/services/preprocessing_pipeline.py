"""可序列化的数据预处理 Pipeline。"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    数据预处理器。

    封装数据清洗与特征工程逻辑，并支持序列化，确保训练-预测一致。
    包含：
    - 删除重复行
    - 删除目标列缺失的行
    - 数值列缺失值填充（中位数）
    - 类别列缺失值填充（众数 / "missing"）
    - 右偏非负数值列的对数变换
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.numeric_fill_values: Dict[str, float] = {}
        self.categorical_fill_values: Dict[str, str] = {}
        self.log_transform_cols: List[str] = []
        self.feature_columns: List[str] = []
        self.feature_dtypes: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """学习训练集上的统计量。"""
        df = X.copy()

        # 删除重复行和目标列缺失行（仅在训练时，y 是目标列）
        df = df.drop_duplicates()
        if self.target_column in df.columns:
            df = df.dropna(subset=[self.target_column])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 排除目标列
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        # 数值填充：中位数
        self.numeric_fill_values = {}
        for col in numeric_cols:
            self.numeric_fill_values[col] = float(df[col].median())

        # 类别填充：众数，没有则 "missing"
        self.categorical_fill_values = {}
        for col in categorical_cols:
            mode = df[col].mode()
            if not mode.empty:
                self.categorical_fill_values[col] = mode.iloc[0]
            else:
                self.categorical_fill_values[col] = "missing"

        # 对数变换：右偏非负数值列
        self.log_transform_cols = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.min() >= 0 and series.skew() > 1.0:
                self.log_transform_cols.append(col)

        # 记录特征列（排除目标列）
        self.feature_columns = [c for c in df.columns if c != self.target_column]

        # 记录特征列的原始数据类型
        self.feature_dtypes = {
            col: str(df[col].dtype) for col in self.feature_columns if col in df.columns
        }

        logger.info(
            f"预处理器拟合完成: 数值列={len(numeric_cols)}, "
            f"类别列={len(categorical_cols)}, 对数变换列={self.log_transform_cols}"
        )
        return self

    def validate_input(self, X: pd.DataFrame) -> List[str]:
        """校验输入数据是否与训练时一致，返回错误列表。"""
        errors = []

        # 检查必需列
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            errors.append(f"缺少特征列: {sorted(missing_cols)}")

        # 检查数据类型
        for col, expected_dtype in self.feature_dtypes.items():
            if col not in X.columns:
                continue
            actual_dtype = str(X[col].dtype)
            # 放宽数值类型检查：int/float 都算数值
            is_expected_numeric = expected_dtype.startswith(("int", "float"))
            is_actual_numeric = actual_dtype.startswith(("int", "float"))
            if is_expected_numeric and not is_actual_numeric:
                errors.append(f"列 '{col}' 期望数值类型，实际为 {actual_dtype}")
            elif not is_expected_numeric and is_actual_numeric:
                errors.append(f"列 '{col}' 期望类别类型，实际为 {actual_dtype}")

        return errors

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """应用预处理变换。"""
        df = X.copy()

        # 删除重复行
        df = df.drop_duplicates()

        # 删除目标列缺失的行（仅在训练/评估时）
        if self.target_column in df.columns:
            df = df.dropna(subset=[self.target_column])

        # 数值填充
        for col, value in self.numeric_fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        # 类别填充
        for col, value in self.categorical_fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        # 对数变换
        for col in self.log_transform_cols:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """拟合并转换。"""
        return self.fit(X, y).transform(X, y)

    def save(self, path: Path):
        """序列化到文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"预处理器已保存: {path}")

    @classmethod
    def load(cls, path: Path) -> "DataPreprocessor":
        """从文件加载。"""
        preprocessor = joblib.load(path)
        logger.info(f"预处理器已加载: {path}")
        return preprocessor


def save_feature_columns(output_dir: Path, columns: List[str]):
    """保存特征列列表到 JSON。"""
    import json

    path = Path(output_dir) / "feature_columns.json"
    path.write_text(json.dumps(columns, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_columns(output_dir: Path) -> List[str]:
    """加载特征列列表。"""
    import json

    path = Path(output_dir) / "feature_columns.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prediction_input(preprocessor: DataPreprocessor, df: pd.DataFrame) -> List[str]:
    """校验预测输入数据。"""
    return preprocessor.validate_input(df)
