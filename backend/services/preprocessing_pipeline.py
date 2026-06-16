"""可序列化的数据预处理 Pipeline。"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from services.feature_engineering import FeatureEngineer

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
    - 异常值截断、数值缩放、稀有类别合并、时间周期特征、文本 Embedding
    """

    def __init__(
        self,
        target_column: str,
        strategy: Optional[Dict[str, Any]] = None,
        cleaning_rules: Optional[Dict[str, Any]] = None,
    ):
        self.target_column = target_column
        self.strategy = strategy or {}
        self.cleaning_rules = cleaning_rules or {}
        self.numeric_fill_values: Dict[str, float] = {}
        self.categorical_fill_values: Dict[str, str] = {}
        self.log_transform_cols: List[str] = []
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.feature_columns: List[str] = []
        self.feature_dtypes: Dict[str, str] = {}

    def _apply_cleaning_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用清洗规则（不含填充/对数变换）。"""
        df = df.copy()
        rules = self.cleaning_rules

        # 删除重复行
        if rules.get("remove_duplicates", True):
            df = df.drop_duplicates()

        # 删除目标列缺失的行（仅在训练/评估时）
        if rules.get("drop_rows_with_missing_target", True) and self.target_column in df.columns:
            df = df.dropna(subset=[self.target_column])

        # 按缺失率自动删除列（>50% 默认丢弃）
        drop_threshold = self.strategy.get("preprocessing", {}).get("missing_drop_threshold", 0.50)
        if drop_threshold is not None and drop_threshold < 1.0:
            for col in df.columns:
                if col == self.target_column:
                    continue
                if df[col].isnull().mean() > drop_threshold:
                    if col not in rules.get("drop_columns", []):
                        rules.setdefault("drop_columns", []).append(col)

        # 按规则删除列
        for col in rules.get("drop_columns", []):
            if col in df.columns:
                df = df.drop(columns=[col])

        # 数值取值约束（越界视为缺失）
        for constraint in rules.get("value_constraints", []):
            col = constraint.get("column")
            if col not in df.columns:
                continue
            min_v = constraint.get("min_value")
            max_v = constraint.get("max_value")
            if min_v is not None:
                df[col] = df[col].where(df[col] >= min_v)
            if max_v is not None:
                df[col] = df[col].where(df[col] <= max_v)

        return df

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础清洗 + 填充 + 对数变换（不含高级特征工程）。"""
        df = self._apply_cleaning_rules(df)

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

    def fit(self, X: pd.DataFrame, y=None):
        """学习训练集上的统计量。"""
        df = X.copy()

        # 先应用清洗规则（删除列、越界处理、重复行、目标缺失行）
        df = self._apply_cleaning_rules(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 排除目标列
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        # 数值填充
        rules = self.cleaning_rules
        numeric_strategy = rules.get("numeric_impute_strategy", "median")
        numeric_constant = rules.get("numeric_impute_constant", 0.0)
        self.numeric_fill_values = {}
        for col in numeric_cols:
            if numeric_strategy == "median":
                self.numeric_fill_values[col] = float(df[col].median())
            elif numeric_strategy == "mean":
                self.numeric_fill_values[col] = float(df[col].mean())
            elif numeric_strategy == "constant":
                self.numeric_fill_values[col] = float(numeric_constant)
            else:
                self.numeric_fill_values[col] = float(df[col].median())

        # 类别填充
        categorical_strategy = rules.get("categorical_impute_strategy", "mode")
        categorical_constant = rules.get("categorical_impute_constant", "missing")
        self.categorical_fill_values = {}
        for col in categorical_cols:
            if categorical_strategy == "mode":
                mode = df[col].mode()
                self.categorical_fill_values[col] = mode.iloc[0] if not mode.empty else "missing"
            elif categorical_strategy == "constant":
                self.categorical_fill_values[col] = categorical_constant
            else:
                mode = df[col].mode()
                self.categorical_fill_values[col] = mode.iloc[0] if not mode.empty else "missing"

        # 对数变换：右偏非负数值列
        self.log_transform_cols = []
        log_skew_threshold = self.strategy.get("preprocessing", {}).get("log_skew_threshold", 1.0)
        for col in numeric_cols:
            series = df[col].dropna()
            if series.min() >= 0 and series.skew() > log_skew_threshold:
                self.log_transform_cols.append(col)

        # 高级特征工程
        preprocessing = self.strategy.get("preprocessing", {})
        self.feature_engineer = FeatureEngineer(
            target_column=self.target_column,
            outlier_strategy=preprocessing.get("outlier_strategy", "iqr"),
            scaler_type=preprocessing.get("scaler_type", "auto"),
            rare_category_threshold=preprocessing.get("rare_category_threshold", 10),
            datetime_cyclical=preprocessing.get("datetime_cyclical", True),
            text_embeddings=preprocessing.get("text_embeddings", False),
            one_hot_threshold=preprocessing.get("one_hot_threshold", 10),
            target_encoding_threshold=preprocessing.get("target_encoding_threshold", 50),
            missing_indicator=preprocessing.get("missing_indicator", True),
            missing_indicator_threshold=preprocessing.get("missing_indicator_threshold", 0.05),
            correlation_threshold=preprocessing.get("correlation_threshold", 0.95),
            low_variance_threshold=preprocessing.get("low_variance_threshold", 0.0),
            pca_variance_ratio=preprocessing.get("pca_variance_ratio", 0.95),
            enable_pca=preprocessing.get("enable_pca", False),
        )
        self.feature_engineer.fit(df)

        # 记录最终特征列（用于预测校验）
        transformed_sample = self.feature_engineer.transform(self._base_transform(df))
        self.feature_columns = [c for c in transformed_sample.columns if c != self.target_column]
        self.feature_dtypes = {
            col: str(transformed_sample[col].dtype) for col in self.feature_columns
        }

        logger.info(
            f"预处理器拟合完成: 数值列={len(numeric_cols)}, "
            f"类别列={len(categorical_cols)}, 对数变换列={self.log_transform_cols}, "
            f"最终特征数={len(self.feature_columns)}"
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
        df = self._base_transform(X)
        if self.feature_engineer is not None:
            df = self.feature_engineer.transform(df)
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
