"""模型可解释性服务。"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

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


def compute_permutation_importance(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """计算 Permutation Importance，用于检测泄露和无偏重要性评估。

    在数据集上计算基准分数，然后逐列打乱特征后重新评估，记录分数下降量。
    AutoGluon 内部指标统一为越大越好，因此分数下降越多代表特征越重要。
    """
    try:
        feature_cols = [c for c in data.columns if c != target_column]
        if not feature_cols:
            return {}

        def _score(df: pd.DataFrame) -> float:
            perf = predictor.evaluate(df)
            metric_name = predictor.eval_metric.name
            if metric_name in perf:
                return float(perf[metric_name])
            return float(next(iter(perf.values())))

        baseline_score = _score(data)
        rng = np.random.default_rng(random_state)

        records = []
        for col in feature_cols:
            drops = []
            for _ in range(n_repeats):
                permuted = data.copy()
                permuted[col] = rng.permutation(permuted[col].values)
                permuted_score = _score(permuted)
                drops.append(baseline_score - permuted_score)
            records.append(
                {
                    "feature": col,
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops)),
                }
            )

        importance_df = pd.DataFrame(records).sort_values("importance_mean", ascending=False)
        output_path = Path(output_dir) / "permutation_importance.csv"
        importance_df.to_csv(output_path, index=False)

        return {
            "path": str(output_path),
            "top_features": importance_df.head(10).set_index("feature").to_dict()["importance_mean"],
        }
    except Exception as e:
        logger.warning(f"Permutation Importance 计算失败: {e}")
        return {}


def explain_single_sample(
    predictor: TabularPredictor,
    X: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """对单条样本进行 TreeSHAP / SHAP 解释。

    返回该样本的预测结果、每个特征的 SHAP 贡献值（按绝对值排序）。
    """
    import importlib.util

    if importlib.util.find_spec("shap") is None:
        raise RuntimeError("shap 未安装，无法生成解释")

    import shap

    if len(X) != 1:
        raise ValueError("explain_single_sample 只支持单条样本")

    feature_names = X.columns.tolist()
    problem_type = predictor.problem_type

    # 优先使用 AutoGluon 原生 TreeSHAP（仅部分树模型支持）
    if hasattr(predictor, "predict_proba_shapley"):
        try:
            shap_values = predictor.predict_proba_shapley(X)
            shap_values = _normalize_shap_matrix(shap_values, X)
            base_value = float(np.mean(shap_values))
            contributions = shap_values[0]
            pred = predictor.predict(X).iloc[0]
            return _format_shap_explanation(feature_names, base_value, contributions, pred, problem_type)
        except Exception:
            pass

    if background is None:
        # 没有背景数据时，用训练集统计均值构造一个背景样本
        background = pd.DataFrame([X.median().values], columns=feature_names)

    if problem_type in ["binary", "multiclass"]:
        classes = predictor.class_labels

        def predict_proba_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=feature_names) if isinstance(X_arr, np.ndarray) else X_arr
            return predictor.predict_proba(X_df).reindex(columns=classes).values

        explainer = shap.Explainer(predict_proba_fn, background)
        shap_values_obj = explainer(X)
        shap_array = np.asarray(shap_values_obj.values)

        if problem_type == "binary":
            # 正类索引为 1
            contributions = shap_array[0, :, 1] if shap_array.ndim == 3 else shap_array[0]
            base_value = float(explainer.expected_value[1] if hasattr(explainer.expected_value, "__getitem__") else explainer.expected_value)
            pred = predictor.predict(X).iloc[0]
        else:
            pred = predictor.predict(X).iloc[0]
            class_idx = list(classes).index(pred)
            contributions = shap_array[0, :, class_idx] if shap_array.ndim == 3 else shap_array[0]
            base_value = float(
                explainer.expected_value[class_idx]
                if hasattr(explainer.expected_value, "__getitem__")
                else explainer.expected_value
            )
    else:
        def predict_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=feature_names) if isinstance(X_arr, np.ndarray) else X_arr
            return predictor.predict(X_df).values

        explainer = shap.Explainer(predict_fn, background)
        shap_values_obj = explainer(X)
        contributions = np.asarray(shap_values_obj.values)[0]
        base_value = float(explainer.expected_value)
        pred = predictor.predict(X).iloc[0]

    values = X.iloc[0].values
    features = [
        {
            "feature": name,
            "value": float(value),
            "contribution": float(contrib),
            "abs_contribution": float(abs(contrib)),
        }
        for name, value, contrib in zip(feature_names, values, np.asarray(contributions).reshape(-1))
    ]
    features.sort(key=lambda x: x["abs_contribution"], reverse=True)

    return {
        "base_value": round(base_value, 6),
        "prediction": pred,
        "problem_type": problem_type,
        "features": features,
    }


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
