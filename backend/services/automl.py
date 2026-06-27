# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""AutoGluon 封装服务。"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import importlib.util

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.utils.class_weight import compute_sample_weight
from services.explainability import compute_shap_values, compute_permutation_importance
from i18n import _
from config import settings, get_recommended_num_cpus

logger = logging.getLogger(__name__)


def _is_optional_model_available(model_name: str) -> bool:
    """检查可选模型依赖是否已安装。

    AutoGluon tabular 默认包含 GBM/CAT/XGB/RF/XT/NN_TORCH；
    FASTAI、TABPFN、AG_AUTOMM 等需要额外安装依赖。
    """
    mapping = {
        "FASTAI": "fastai",
        "TABPFN": "tabpfn",
        "TABPFNMIX": "tabpfn",
        "REALTABPFN": "tabpfn",
        "REALTABPFN-V2": "tabpfn",
        "AG_AUTOMM": "autogluon.multimodal",
    }
    package = mapping.get(model_name)
    if not package:
        return True
    try:
        return importlib.util.find_spec(package) is not None
    except Exception:
        return False


class AutoMLService:
    """AutoML 训练服务封装。"""

    # 内部/通用指标名 -> AutoGluon 可识别的指标名
    METRIC_ALIASES = {
        "auc_pr": "average_precision",
        "auc-pr": "average_precision",
        "average_precision_score": "average_precision",
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "autogluon_models"
        # 清理已有模型目录，避免 AutoGluon "Learner is already fit" 错误
        if self.model_dir.exists():
            import shutil

            logger.info(_("training.cleaning_model_dir", path=self.model_dir))
            shutil.rmtree(self.model_dir)
        logger.info(_("training.model_dir_ready", path=self.model_dir, exists=self.model_dir.exists()))

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str,
        task_type: str,
        time_limit: Optional[int],
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

        # 将通用指标别名映射为 AutoGluon 合法名称
        primary_metric = self._normalize_metric_name(primary_metric, task_type)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info(_("training.seed_set", seed=seed))

        logger.info(
            _(
                "training.starting_automl",
                target=target_column,
                task=task_type,
                preset=preset,
                metric=primary_metric,
                max_models=max_models,
            )
        )

        fit_kwargs: Dict[str, Any] = {
            "train_data": train_data,
            "presets": preset,
            "time_limit": time_limit,
            "dynamic_stacking": False,  # 禁用 DyStack，避免某些场景下 Learner 被重复 fit
            "num_cpus": get_recommended_num_cpus(),  # 基于实际 CPU 核心数动态分配，保留 1 核给系统
            "num_gpus": settings.num_gpus if settings.use_gpu else 0,
        }
        if settings.use_gpu:
            logger.info(_("training.gpu_enabled", num_gpus=settings.num_gpus))
        # 动态模型空间
        hyperparameters = self._select_hyperparameters(strategy, seed)
        if hyperparameters:
            fit_kwargs["hyperparameters"] = hyperparameters

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
                logger.info(_("training.using_sample_weight", ratio=f"{imbalance_ratio:.2f}"))
            elif strategy is not None:
                # 策略未指定时，按 imbalance_ratio 自动兜底
                use_sample_weight = strategy.get("use_sample_weight", imbalance_ratio > 1.5)
                if use_sample_weight:
                    sample_weights = compute_sample_weight("balanced", y_train)
                    train_data = train_data.copy()
                    train_data[sample_weight_col] = sample_weights
                    logger.info(
                        _("training.strategy_sample_weight", ratio=f"{imbalance_ratio:.2f}")
                    )
            elif imbalance_ratio > 1.5:
                use_sample_weight = True
                sample_weights = compute_sample_weight("balanced", y_train)
                train_data = train_data.copy()
                train_data[sample_weight_col] = sample_weights
                logger.info(
                    _(
                        "strategy.imbalance_balanced_weight",
                        ratio=f"{imbalance_ratio:.2f}",
                    )
                )

        # 确保 fit_kwargs 使用最新的 train_data（可能已加入 sample_weight 列）
        fit_kwargs["train_data"] = train_data

        # 模型复杂度控制：策略 > max_models 启发式
        stacking_config = self._stacking_config(strategy, max_models)
        fit_kwargs.update(stacking_config)

        # 验证策略：holdout / CV 透传给 AutoGluon
        # 根据最小类样本数限制 CV 折数；最小类 < 2 时禁用 bagging CV，改用 holdout
        if self._map_task_type(task_type) in ["binary", "multiclass"]:
            min_class_count = int(train_data[target_column].value_counts().min())
        else:
            min_class_count = None

        if strategy is not None:
            validation_strategy = strategy.get("validation_strategy", {})
            if validation_strategy.get("name") == "cv":
                requested_folds = validation_strategy.get("n_folds", 5)
                if min_class_count is not None and min_class_count < 2:
                    logger.warning(
                        _(
                            "training.cv_bagging_fallback",
                            count=min_class_count,
                        )
                    )
                    fit_kwargs.pop("num_bag_folds", None)
                    fit_kwargs["holdout_frac"] = validation_strategy.get(
                        "holdout_frac", 0.2
                    )
                else:
                    if min_class_count is not None:
                        requested_folds = min(requested_folds, min_class_count)
                    # 使用 bagging 实现 KFold CV；移除 holdout_frac 避免冲突
                    fit_kwargs["num_bag_folds"] = requested_folds
                    fit_kwargs.pop("holdout_frac", None)
                    logger.info(_("training.cv_enabled", n_folds=fit_kwargs["num_bag_folds"]))
            else:
                holdout_frac = validation_strategy.get("holdout_frac")
                if holdout_frac is not None:
                    fit_kwargs["holdout_frac"] = holdout_frac

        # 如果存在上次训练残留的模型目录（如重试、超时 kill 导致部分保存），先清理，
        # 否则 AutoGluon 会复用已 fit 的 learner，触发 "Learner is already fit."
        if self.model_dir.exists():
            import shutil

            shutil.rmtree(self.model_dir, ignore_errors=True)
            logger.info(_("training.cleaning_model_dir", path=self.model_dir))

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

        # 集成验证回退：集成提升不足 2% 时回退到最优单模型
        ensemble_fallback = self._apply_ensemble_fallback(predictor, leaderboard)
        if ensemble_fallback.get("fallback_applied"):
            logger.info(
                _(
                    "training.ensemble_fallback_applied",
                    top_model=ensemble_fallback.get("top_model"),
                    best_single_model=ensemble_fallback.get("best_single_model"),
                )
            )

        # 特征重要性
        try:
            importance = predictor.feature_importance(train_data, silent=True)
            importance_path = self.output_dir / "feature_importance.csv"
            # 将索引（特征名）转为 feature 列，避免前端/报告读取困难
            importance.reset_index().rename(columns={"index": "feature"}).to_csv(
                importance_path, index=False
            )
        except Exception as e:
            logger.warning(_("training.feature_importance_failed", msg=e))
            importance = None

        # SHAP 可解释性（排除样本权重列）
        shap_train_data = train_data
        if sample_weight_col in train_data.columns:
            shap_train_data = train_data.drop(columns=[sample_weight_col])

        # Permutation Importance（用于泄露检测与无偏重要性）
        perm_info = compute_permutation_importance(
            predictor, shap_train_data, target_column, self.output_dir
        )

        shap_info = compute_shap_values(predictor, shap_train_data, target_column, self.output_dir)

        return {
            "model_path": str(self.model_dir),
            "leaderboard_path": str(leaderboard_path),
            "feature_importance": importance.to_dict() if importance is not None else None,
            "primary_metric": primary_metric,
            "shap_info": shap_info,
            "perm_importance": perm_info,
            "imbalance_ratio": imbalance_ratio if "imbalance_ratio" in locals() else None,
            "ensemble_fallback": ensemble_fallback,
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
                logger.warning(_("training.probability_prediction_failed", msg=e))

        return result

    def _apply_ensemble_fallback(
        self,
        predictor: TabularPredictor,
        leaderboard: pd.DataFrame,
        improvement_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """检查 WeightedEnsemble 是否显著优于最优单模型；否则回退。"""
        if leaderboard.empty or "model" not in leaderboard.columns:
            return {"fallback_applied": False, "reason": _("automl.leaderboard_empty")}

        score_col = "score_val"
        if score_col not in leaderboard.columns:
            score_col = leaderboard.columns[-1]

        sorted_lb = leaderboard.sort_values(score_col, ascending=False)
        top_model = sorted_lb.iloc[0]
        top_name = str(top_model["model"])

        if not top_name.startswith("WeightedEnsemble"):
            return {
                "fallback_applied": False,
                "ensemble_used": False,
                "top_model": top_name,
            }

        non_ensemble = sorted_lb[~sorted_lb["model"].astype(str).str.startswith("WeightedEnsemble")]
        if non_ensemble.empty:
            return {
                "fallback_applied": False,
                "ensemble_used": True,
                "reason": _("automl.no_single_model"),
                "top_model": top_name,
            }

        best_single = non_ensemble.iloc[0]
        best_single_name = str(best_single["model"])
        top_score = float(top_model[score_col])
        single_score = float(best_single[score_col])
        improvement = (top_score - single_score) / max(abs(single_score), 1e-10)

        fallback_applied = improvement <= improvement_threshold
        result: Dict[str, Any] = {
            "fallback_applied": fallback_applied,
            "ensemble_used": True,
            "top_model": top_name,
            "best_single_model": best_single_name,
            "improvement_ratio": round(improvement, 6),
            "threshold": improvement_threshold,
        }

        if fallback_applied:
            try:
                predictor.set_model_best(best_single_name)
                # 仅删除集成模型，保留其他单模型
                ensemble_models = [
                    m for m in predictor.model_names() if m.startswith("WeightedEnsemble")
                ]
                if ensemble_models:
                    predictor.delete_models(models_to_delete=ensemble_models)
                result["models_removed"] = ensemble_models
            except Exception as e:
                logger.warning(_("training.ensemble_fallback_failed", msg=e))
                result["fallback_error"] = str(e)

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

    def _apply_gpu_hyperparameters(
        self, hp: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """当 GPU 启用时，为支持的模型注入 GPU 专用超参。

        现有参数优先级高于 GPU 默认参数，避免覆盖用户/策略显式指定的值。
        NN_TORCH 的 GPU 由 TabularPredictor.fit 的 num_gpus 控制，不在此处注入。
        """
        if hp is None or not settings.use_gpu:
            return hp

        gpu_params = {
            "GBM": {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0},
            "XGB": self._get_xgb_gpu_params(),
            "CAT": {"task_type": "GPU", "devices": "0", "verbose": False},
        }

        def _merge_params(existing: Any, extra: Dict[str, Any]) -> Any:
            """合并 GPU 参数到现有参数，保留现有值优先。"""
            if existing is None:
                return extra
            if isinstance(existing, dict):
                merged = dict(extra)
                merged.update(existing)
                return merged
            if isinstance(existing, list):
                return [_merge_params(item, extra) for item in existing]
            return existing

        for model_key, extra in gpu_params.items():
            if model_key in hp:
                hp[model_key] = _merge_params(hp[model_key], extra)

        return hp

    @staticmethod
    def _get_xgb_gpu_params() -> Dict[str, Any]:
        """根据 XGBoost 版本返回正确的 GPU 参数。

        - XGBoost >= 2.0: 使用 device='cuda'
        - XGBoost < 2.0: 使用 tree_method='gpu_hist'
        """
        try:
            from importlib.metadata import version

            xgb_version = version("xgboost")
            major = int(xgb_version.split(".")[0])
            if major >= 2:
                return {"device": "cuda"}
        except Exception:
            pass
        return {"tree_method": "gpu_hist"}

    def _select_hyperparameters(
        self, strategy: Optional[Dict[str, Any]], seed: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """根据策略动态选择模型搜索空间，注入可选深度学习模型与 GPU 超参。"""
        hp: Optional[Dict[str, Any]] = None

        if strategy is None:
            hp = self._hyperparameters_with_seed(seed) if seed is not None else None
        elif strategy.get("hyperparameters"):
            hp = strategy["hyperparameters"]
        else:
            data_size_label = strategy.get("data_size_label", "medium")
            if data_size_label == "small":
                # 小数据也展示完整、多样的模型空间：多组超参 + 全量典型模型
                hp = {
                    "GBM": [
                        {"extra_trees": False},
                        {"extra_trees": True},
                        {"learning_rate": 0.05, "num_leaves": 31},
                    ],
                    "CAT": [
                        {},
                        {"iterations": 1000, "depth": 8},
                    ],
                    "XGB": [
                        {},
                        {"max_depth": 6, "eta": 0.1},
                    ],
                    "RF": [
                        {"n_estimators": 300},
                        {"n_estimators": 500, "max_features": "sqrt"},
                    ],
                    "XT": [
                        {"n_estimators": 300},
                        {"n_estimators": 500},
                    ],
                    "KNN": [{}, {"weights": "distance"}],
                    "LR": [{}, {"C": 0.1}, {"C": 10}],
                    "NN_TORCH": {"seed_value": seed} if seed is not None else {},
                }
                # 若安装了额外依赖，加入 FASTAI / TabPFN 等深度学习模型
                if _is_optional_model_available("FASTAI"):
                    hp["FASTAI"] = {"seed_value": seed} if seed is not None else {}
                if _is_optional_model_available("REALTABPFN-V2"):
                    hp["REALTABPFN-V2"] = {}
            elif data_size_label == "large":
                # 大数据聚焦高效线性/树模型
                hp = {"GBM": {}, "CAT": {}, "XGB": {}, "LR": {}}
            else:
                # 中等规模默认模型空间
                if seed is not None:
                    hp = self._hyperparameters_with_seed(seed)
                else:
                    hp = {"GBM": {}, "CAT": {}, "XGB": {}, "RF": {}, "XT": {}}
                if _is_optional_model_available("FASTAI"):
                    hp["FASTAI"] = {"seed_value": seed} if seed is not None else {}

        return self._apply_gpu_hyperparameters(hp)

    def _map_task_type(self, task_type: str) -> Optional[str]:
        """映射任务类型到 AutoGluon problem_type。"""
        mapping = {
            "binary_classification": "binary",
            "multiclass_classification": "multiclass",
            "regression": "regression",
        }
        return mapping.get(task_type)

    def _normalize_metric_name(self, metric: Optional[str], task_type: str) -> str:
        """把内部/用户输入的指标别名转换为 AutoGluon 合法名称。"""
        if not metric:
            return self._auto_select_metric(pd.Series([0, 1]), task_type)
        normalized = self.METRIC_ALIASES.get(metric, metric)
        return normalized

    def _auto_select_metric(self, y: pd.Series, task_type: str) -> str:
        """自动选择评估指标。

        - 二分类不平衡 -> AUC-PR (AutoGluon 中为 average_precision)
        - 二分类平衡 -> F1
        - 多分类 -> log_loss
        - 回归 -> RMSE
        """
        if task_type == "binary_classification":
            class_counts = y.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 3:
                return "average_precision"
            return "f1"
        elif task_type == "multiclass_classification":
            return "log_loss"
        elif task_type == "regression":
            return "root_mean_squared_error"
        return "accuracy"


def load_predictor(model_path: Path) -> TabularPredictor:
    """加载训练好的模型。"""
    return TabularPredictor.load(str(model_path))
