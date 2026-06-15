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
        strategy: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """训练 AutoGluon 模型。"""

        # 如果提供了数据驱动策略，优先从策略中读取配置
        if strategy is not None:
            preset = strategy.get("preset", preset)
            time_limit = strategy.get("time_limit_seconds", time_limit)
            max_models = strategy.get("max_models", max_models)
            primary_metric = strategy.get("primary_metric", primary_metric)

        # 根据任务类型选择评估指标
        if primary_metric is None:
            primary_metric = self._auto_select_metric(train_data[target_column], task_type)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"设置随机种子: {seed}")

        logger.info(
            f"开始训练: target={target_column}, task={task_type}, preset={preset}, "
            f"metric={primary_metric}, max_models={max_models}"
        )

        fit_kwargs: Dict[str, Any] = {
            "train_data": train_data,
            "presets": preset,
            "time_limit": time_limit,
        }
        if seed is not None:
            fit_kwargs["hyperparameters"] = self._hyperparameters_with_seed(seed)

        # 类别不平衡处理：优先使用传入的样本权重，其次按策略，最后自动兜底
        sample_weight_col = "_sample_weight"
        use_sample_weight = False
        imbalance_ratio = None
        if self._map_task_type(task_type) in ["binary", "multiclass"]:
            y_train = train_data[target_column]
            class_counts = y_train.value_counts()
            imbalance_ratio = float(class_counts.max() / class_counts.min())

            if sample_weight is not None:
                use_sample_weight = True
                train_data = train_data.copy()
                train_data[sample_weight_col] = sample_weight.values
                logger.info(f"使用传入的 sample_weight，imbalance_ratio={imbalance_ratio:.2f}")
            elif strategy is not None:
                # 策略未指定时，按 imbalance_ratio 自动兜底
                use_sample_weight = strategy.get("use_sample_weight", imbalance_ratio > 1.5)
                if use_sample_weight:
                    sample_weights = compute_sample_weight("balanced", y_train)
                    train_data = train_data.copy()
                    train_data[sample_weight_col] = sample_weights
                    logger.info(
                        f"策略启用 balanced sample_weight，imbalance_ratio={imbalance_ratio:.2f}"
                    )
            elif imbalance_ratio > 1.5:
                use_sample_weight = True
                sample_weights = compute_sample_weight("balanced", y_train)
                train_data = train_data.copy()
                train_data[sample_weight_col] = sample_weights
                logger.info(
                    f"自动检测到类别不平衡 (ratio={imbalance_ratio:.2f})，已启用 balanced sample_weight"
                )

        # 确保 fit_kwargs 使用最新的 train_data（可能已加入 sample_weight 列）
        fit_kwargs["train_data"] = train_data

        # 模型复杂度控制：策略 > max_models 启发式
        stacking_config = self._stacking_config(strategy, max_models)
        fit_kwargs.update(stacking_config)

        # 验证策略：holdout_frac 等
        if strategy is not None:
            validation_strategy = strategy.get("validation_strategy", {})
            holdout_frac = validation_strategy.get("holdout_frac")
            if holdout_frac is not None:
                fit_kwargs["holdout_frac"] = holdout_frac

        # 训练模型
        predictor_kwargs = {
            "label": target_column,
            "problem_type": self._map_task_type(task_type),
            "eval_metric": primary_metric,
            "path": str(self.model_dir),
        }
        if use_sample_weight:
            predictor_kwargs["sample_weight"] = sample_weight_col
        predictor = TabularPredictor(**predictor_kwargs).fit(**fit_kwargs)

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
            # 将索引（特征名）转为 feature 列，避免前端/报告读取困难
            importance.reset_index().rename(columns={"index": "feature"}).to_csv(
                importance_path, index=False
            )
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

    def _stacking_config(
        self, strategy: Optional[Dict[str, Any]], max_models: int
    ) -> Dict[str, Any]:
        """生成 stacking/bagging 配置。"""
        if strategy is not None:
            return {
                "auto_stack": strategy.get("auto_stack", False),
                "num_bag_folds": strategy.get("num_bag_folds", 0),
                "num_stack_levels": strategy.get("num_stack_levels", 0),
            }
        if max_models <= 10:
            return {"auto_stack": False, "num_bag_folds": 0, "num_stack_levels": 0}
        elif max_models <= 30:
            return {"num_bag_folds": 3, "num_stack_levels": 1}
        return {}

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
