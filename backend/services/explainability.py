"""模型可解释性服务。"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def compute_shap_values(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    sample_size: int = 200,
) -> Dict[str, Any]:
    """
    计算并保存 SHAP 值。

    优先使用 AutoGluon 原生 SHAP 接口；不可用时回退到基于模型预测概率的
    shap.Explainer，以兼容不同 AutoGluon 版本。

    Args:
        predictor: 训练好的 AutoGluon predictor
        data: 用于计算 SHAP 的数据（通常是训练集子集）
        target_column: 目标列名
        output_dir: 输出目录
        sample_size: 大数据集时采样的样本数

    Returns:
        SHAP 摘要统计信息
    """
    import importlib.util

    if importlib.util.find_spec("shap") is None:
        logger.warning("shap 未安装，跳过 SHAP 计算")
        return {}

    try:
        X = data.drop(columns=[target_column])
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        feature_names = X_sample.columns.tolist()

        # 原生接口（较新版本 AutoGluon 可能提供）
        if hasattr(predictor, "predict_proba_shapley"):
            shap_matrix = predictor.predict_proba_shapley(X_sample)
            shap_matrix = _normalize_shap_matrix(shap_matrix, X_sample)
        else:
            shap_matrix = _compute_shap_with_explainer(predictor, X_sample, feature_names)

        # 保存 SHAP 值与对应特征矩阵（用于后续绘图）
        shap_path = Path(output_dir) / "shap_values.joblib"
        joblib.dump(
            {
                "shap_values": shap_matrix,
                "feature_names": feature_names,
                "features": X_sample.values,
            },
            shap_path,
        )

        # 计算全局重要性（平均绝对 SHAP 值）
        mean_abs_shap = (
            pd.DataFrame(np.abs(shap_matrix), columns=feature_names)
            .mean()
            .sort_values(ascending=False)
        )

        logger.info(f"SHAP 计算完成，保存到 {shap_path}")
        return {
            "shap_path": str(shap_path),
            "top_features": mean_abs_shap.head(10).to_dict(),
        }
    except Exception as e:
        logger.warning(f"SHAP 计算失败: {e}")
        return {}


def _normalize_shap_matrix(shap_values: Any, X_sample: pd.DataFrame) -> np.ndarray:
    """将 AutoGluon 返回的 SHAP 结果统一转换为 (n_samples, n_features) 矩阵。"""
    if isinstance(shap_values, np.ndarray):
        return shap_values
    if isinstance(shap_values, pd.DataFrame):
        return shap_values.values
    if isinstance(shap_values, dict):
        # 多分类时按类别返回 dict，取各类别绝对值的平均
        arrays = []
        for class_values in shap_values.values():
            arr = class_values.values if isinstance(class_values, pd.DataFrame) else class_values
            arrays.append(np.asarray(arr))
        return np.mean(np.abs(np.stack(arrays, axis=-1)), axis=-1)
    raise ValueError(f"不支持的 SHAP 值类型: {type(shap_values)}")


def _compute_shap_with_explainer(
    predictor: TabularPredictor,
    X_sample: pd.DataFrame,
    feature_names: List[str],
) -> np.ndarray:
    """使用 shap.Explainer 计算 SHAP 值。"""
    import shap

    problem_type = predictor.problem_type
    background = shap.sample(X_sample, min(50, len(X_sample)), random_state=42)

    if problem_type in ["binary", "multiclass"]:
        classes = predictor.class_labels

        def predict_proba_fn(X_arr):
            if isinstance(X_arr, np.ndarray):
                X_df = pd.DataFrame(X_arr, columns=feature_names)
            else:
                X_df = X_arr
            proba = predictor.predict_proba(X_df)
            # 保持类别顺序一致
            return proba.reindex(columns=classes).values

        explainer = shap.Explainer(predict_proba_fn, background)
        shap_values_obj = explainer(X_sample)
        shap_array = np.asarray(shap_values_obj.values)

        if shap_array.ndim == 2:
            # 二分类但只返回一个输出的情况
            return shap_array

        if problem_type == "binary":
            # 取正类的 SHAP 值（通常索引为 1）
            return shap_array[:, :, 1]

        # 多分类：取每个样本预测类别的 SHAP 值
        preds = predictor.predict(X_sample).values
        class_to_idx = {c: i for i, c in enumerate(classes)}
        selected = np.zeros((shap_array.shape[0], shap_array.shape[1]))
        for i, pred in enumerate(preds):
            selected[i, :] = shap_array[i, :, class_to_idx[pred]]
        return selected

    # 回归任务
    def predict_fn(X_arr):
        if isinstance(X_arr, np.ndarray):
            X_df = pd.DataFrame(X_arr, columns=feature_names)
        else:
            X_df = X_arr
        return predictor.predict(X_df).values

    explainer = shap.Explainer(predict_fn, background)
    shap_values_obj = explainer(X_sample)
    return np.asarray(shap_values_obj.values)
