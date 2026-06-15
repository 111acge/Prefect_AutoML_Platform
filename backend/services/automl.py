"""AutoGluon 封装服务。"""

import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.utils.class_weight import compute_sample_weight
from services.explainability import compute_shap_values

logger = logging.getLogger(__name__)


class AutoMLService:
    """AutoML 训练服务封装。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "autogluon_models"
        # 清理已有模型目录，避免 AutoGluon "Learner is already fit" 错误
        if self.model_dir.exists():
            import shutil

            logger.info(f"清理已有模型目录: {self.model_dir}")
            shutil.rmtree(self.model_dir)
        logger.info(f"模型目录已准备: {self.model_dir}, exists={self.model_dir.exists()}")

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str,
        task_type: str,
        time_limit: int,
        preset: str = "medium_quality",
        primary_metric: Optional[str] = None,
        seed: Optional[int] = None,
        max_models: int = 50,
    ) -> Dict[str, Any]:
        """训练 AutoGluon 模型。"""

        # 根据任务类型选择评估指标
        if primary_metric is None:
            primary_metric = self._auto_select_metric(train_data[target_column], task_type)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"设置随机种子: {seed}")

        logger.info(
            f"开始训练: target={target_column}, task={task_type}, "
            f"metric={primary_metric}, max_models={max_models}"
        )

        fit_kwargs = {
            "train_data": train_data,
            "presets": preset,
            "time_limit": time_limit,
        }
        if seed is not None:
            fit_kwargs["hyperparameters"] = self._hyperparameters_with_seed(seed)

        # 类别不平衡处理：对分类任务自动计算样本权重列
        sample_weight_col = "_sample_weight"
        if self._map_task_type(task_type) in ["binary", "multiclass"]:
            y_train = train_data[target_column]
            class_counts = y_train.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 1.5:
                sample_weights = compute_sample_weight("balanced", y_train)
                train_data = train_data.copy()
                train_data[sample_weight_col] = sample_weights
                fit_kwargs["sample_weight"] = sample_weight_col
                logger.info(
                    f"检测到类别不平衡 (ratio={imbalance_ratio:.2f})，已启用 balanced sample_weight"
                )

        # 根据 max_models 控制模型复杂度
        if max_models <= 10:
            fit_kwargs["auto_stack"] = False
            fit_kwargs["num_bag_folds"] = 0
            fit_kwargs["num_stack_levels"] = 0
        elif max_models <= 30:
            fit_kwargs["num_bag_folds"] = 3
            fit_kwargs["num_stack_levels"] = 1

        # 训练模型
        predictor = TabularPredictor(
            label=target_column,
            problem_type=self._map_task_type(task_type),
            eval_metric=primary_metric,
            path=str(self.model_dir),
        ).fit(**fit_kwargs)

        # 保存模型（AutoGluon 已经保存到 path，这里显式调用确保）
        predictor.save()

        # 生成排行榜
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard_path = self.output_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)

        # 特征重要性
        try:
            importance = predictor.feature_importance(train_data, silent=True)
            importance_path = self.output_dir / "feature_importance.csv"
            importance.to_csv(importance_path)
        except Exception as e:
            logger.warning(f"特征重要性计算失败: {e}")
            importance = None

        # SHAP 可解释性（排除样本权重列）
        shap_train_data = train_data
        if sample_weight_col in train_data.columns:
            shap_train_data = train_data.drop(columns=[sample_weight_col])
        shap_info = compute_shap_values(predictor, shap_train_data, target_column, self.output_dir)

        return {
            "model_path": str(self.model_dir),
            "leaderboard_path": str(leaderboard_path),
            "feature_importance": importance.to_dict() if importance is not None else None,
            "primary_metric": primary_metric,
            "shap_info": shap_info,
            "imbalance_ratio": imbalance_ratio if "imbalance_ratio" in locals() else None,
        }

    def evaluate(
        self,
        predictor: TabularPredictor,
        test_data: pd.DataFrame,
        target_column: str,
    ) -> Dict[str, float]:
        """评估模型。"""
        y_test = test_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        performance = predictor.evaluate(X_test, y_test)
        return {str(k): float(v) for k, v in performance.items()}

    def predict(
        self,
        predictor: TabularPredictor,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """使用模型预测。"""
        predictions = predictor.predict(data)

        result = {
            "predictions": predictions.tolist(),
        }

        # 分类任务返回概率
        if predictor.problem_type in ["binary", "multiclass"]:
            try:
                probabilities = predictor.predict_proba(data)
                result["probabilities"] = probabilities.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"概率预测失败: {e}")

        return result

    def _hyperparameters_with_seed(self, seed: int) -> Dict[str, Any]:
        """生成带固定种子的超参配置（覆盖常见模型）。"""
        return {
            "GBM": [
                {"extra_trees": False, "random_state": seed},
                {"extra_trees": True, "random_state": seed},
            ],
            "CAT": {"random_seed": seed},
            "XGB": {"seed": seed},
            "RF": {"random_state": seed},
            "XT": {"random_state": seed},
            "NN_TORCH": {"seed_value": seed},
        }

    def _map_task_type(self, task_type: str) -> Optional[str]:
        """映射任务类型到 AutoGluon problem_type。"""
        mapping = {
            "binary_classification": "binary",
            "multiclass_classification": "multiclass",
            "regression": "regression",
        }
        return mapping.get(task_type)

    def _auto_select_metric(self, y: pd.Series, task_type: str) -> str:
        """自动选择评估指标。"""
        if task_type == "binary_classification":
            # 检查是否不平衡
            class_counts = y.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 3:
                return "roc_auc"
            return "f1"
        elif task_type == "multiclass_classification":
            return "log_loss"
        elif task_type == "regression":
            return "root_mean_squared_error"
        return "accuracy"


def load_predictor(model_path: Path) -> TabularPredictor:
    """加载训练好的模型。"""
    return TabularPredictor.load(str(model_path))
