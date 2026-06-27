# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""GPU 支持相关单元测试。"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from config import get_gpu_info, is_gpu_available, get_gpu_summary, settings
from services.automl import AutoMLService


class TestGpuDetection:
    """测试 GPU 检测工具函数。"""

    def test_get_gpu_info_structure(self):
        """get_gpu_info 应返回包含预期字段的字典。"""
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "count" in info
        assert "name" in info
        assert "cuda_version" in info
        assert "torch_cuda_available" in info
        assert isinstance(info["available"], bool)

    def test_is_gpu_available_returns_bool(self):
        """is_gpu_available 应返回布尔值。"""
        assert isinstance(is_gpu_available(), bool)

    def test_get_gpu_summary_returns_string(self):
        """get_gpu_summary 应返回字符串。"""
        summary = get_gpu_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


@pytest.fixture
def service(tmp_path: Path) -> AutoMLService:
    return AutoMLService(tmp_path)


class TestAutoMLGpuHyperparameters:
    """测试 AutoMLService 的 GPU 超参注入。"""

    def _collect_gbm_device(self, gbm_param: Any) -> str:
        """从 GBM 参数（dict 或 list of dict）中提取 device 值。"""
        if isinstance(gbm_param, dict):
            return gbm_param.get("device", "")
        if isinstance(gbm_param, list):
            return gbm_param[0].get("device", "") if gbm_param else ""
        return ""

    def test_gpu_params_injected_when_enabled(self, service: AutoMLService, monkeypatch):
        """use_gpu=True 时，应为支持的模型注入 GPU 参数。"""
        monkeypatch.setattr(settings, "use_gpu", True)
        monkeypatch.setattr(settings, "num_gpus", 1)

        hp: Dict[str, Any] = {
            "GBM": [{"extra_trees": False}, {"extra_trees": True}],
            "XGB": {"seed": 42},
            "CAT": {"random_seed": 42},
            "NN_TORCH": {"seed_value": 42},
            "RF": {"random_state": 42},
        }
        result = service._apply_gpu_hyperparameters(hp)
        assert result is not None

        # GBM 列表中每个 dict 都应包含 device=gpu
        assert self._collect_gbm_device(result["GBM"]) == "gpu"
        # XGB 应使用 CUDA 设备（当前 .venv 中 XGBoost >= 2.0）
        assert result["XGB"].get("device") == "cuda" or result["XGB"].get("tree_method") == "gpu_hist"
        # CatBoost 应使用 GPU task_type，并关闭冗余 verbose
        assert result["CAT"].get("task_type") == "GPU"
        assert result["CAT"].get("verbose") is False
        # NN_TORCH 的 GPU 由 num_gpus 控制，不应注入 use_gpu
        assert "use_gpu" not in result["NN_TORCH"]
        assert result["NN_TORCH"].get("seed_value") == 42
        # RF 不受 GPU 参数影响
        assert "device" not in result["RF"]
        # 原有参数应保留
        assert result["XGB"].get("seed") == 42
        assert result["CAT"].get("random_seed") == 42

    def test_gpu_params_not_injected_when_disabled(self, service: AutoMLService, monkeypatch):
        """use_gpu=False 时，不应注入 GPU 参数。"""
        monkeypatch.setattr(settings, "use_gpu", False)
        monkeypatch.setattr(settings, "num_gpus", 0)

        hp: Dict[str, Any] = {
            "GBM": [{"extra_trees": False}],
            "XGB": {"seed": 42},
            "CAT": {"random_seed": 42},
        }
        result = service._apply_gpu_hyperparameters(hp)
        assert result is not None
        assert self._collect_gbm_device(result["GBM"]) != "gpu"
        assert "tree_method" not in result["XGB"]
        assert result["XGB"].get("seed") == 42

    def test_none_hyperparameters_passthrough(self, service: AutoMLService, monkeypatch):
        """_apply_gpu_hyperparameters 应正确处理 None 输入。"""
        monkeypatch.setattr(settings, "use_gpu", True)
        assert service._apply_gpu_hyperparameters(None) is None


class TestOptionalModels:
    """测试可选深度学习模型的运行时检测与注入。"""

    def test_optional_model_availability_returns_bool(self):
        """_is_optional_model_available 应返回布尔值。"""
        from services.automl import _is_optional_model_available

        assert isinstance(_is_optional_model_available("FASTAI"), bool)
        assert isinstance(_is_optional_model_available("TABPFN"), bool)
        assert isinstance(_is_optional_model_available("UNKNOWN"), bool)

    def test_small_data_includes_optional_models_when_available(
        self, service: AutoMLService, monkeypatch
    ):
        """可选模型可用时，小数据策略应包含它们。"""
        monkeypatch.setattr(settings, "use_gpu", False)
        from services import automl

        monkeypatch.setattr(automl, "_is_optional_model_available", lambda name: True)

        hp = service._select_hyperparameters({"data_size_label": "small"}, seed=42)
        assert hp is not None
        assert "FASTAI" in hp
        assert "REALTABPFN-V2" in hp

    def test_small_data_excludes_optional_models_when_unavailable(
        self, service: AutoMLService, monkeypatch
    ):
        """可选模型不可用时，不应出现在策略中。"""
        monkeypatch.setattr(settings, "use_gpu", False)
        from services import automl

        monkeypatch.setattr(automl, "_is_optional_model_available", lambda name: False)

        hp = service._select_hyperparameters({"data_size_label": "small"}, seed=42)
        assert hp is not None
        assert "FASTAI" not in hp
        assert "REALTABPFN-V2" not in hp


class TestSettingsGpuValidation:
    """测试 Settings 中 GPU 字段的校验行为。"""

    def test_num_gpus_zero_when_gpu_disabled(self, monkeypatch):
        """当 use_gpu=False 时，num_gpus 在校验后应为 0。"""
        monkeypatch.setattr(settings, "use_gpu", False)
        # 手动触发 validator 语义：直接断言当前值
        assert settings.use_gpu is False
        # 若重新构造 Settings，num_gpus 会被 validator 强制为 0
        from config import Settings

        monkeypatch.setenv("USE_GPU", "false")
        s = Settings()
        assert s.use_gpu is False
        assert s.num_gpus == 0
