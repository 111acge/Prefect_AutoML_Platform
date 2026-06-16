"""清洗规则与 DataPreprocessor 集成测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd

from services.preprocessing_pipeline import DataPreprocessor


def test_drop_columns_and_value_constraints():
    df = pd.DataFrame({
        "age": [25, 30, 200, 40],
        "income": [3000.0, 5000.0, 8000.0, None],
        "drop_me": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
    })
    rules = {
        "drop_columns": ["drop_me"],
        "value_constraints": [{"column": "age", "min_value": 0, "max_value": 120}],
    }
    preprocessor = DataPreprocessor(target_column="target", cleaning_rules=rules)
    transformed = preprocessor.fit_transform(df)

    assert "drop_me" not in transformed.columns
    # age=200 越界会被置为 NaN，随后用中位数填充
    assert transformed["age"].max() <= 120
    assert "income" in transformed.columns


def test_custom_impute_strategy():
    df = pd.DataFrame({
        "num": [1.0, 2.0, None, 4.0],
        "cat": ["a", None, "b", "c"],
        "target": [0, 1, 0, 1],
    })
    rules = {
        "numeric_impute_strategy": "constant",
        "numeric_impute_constant": 0.0,
        "categorical_impute_strategy": "constant",
        "categorical_impute_constant": "unknown",
    }
    # 关闭缩放与稀有类别合并，以便直接断言填充值
    strategy = {"preprocessing": {"scaler_type": "none", "rare_category_threshold": 1, "one_hot_threshold": 0}}
    preprocessor = DataPreprocessor(target_column="target", strategy=strategy, cleaning_rules=rules)
    transformed = preprocessor.fit_transform(df)

    assert transformed.loc[2, "num"] == 0.0
    assert transformed.loc[1, "cat"] == "unknown"
