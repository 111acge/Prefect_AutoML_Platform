"""原子步骤产物路径与 manifest 管理。"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class StepManifest:
    """管理单个 Run 在 output_dir 下的所有中间产物路径。

    所有路径均相对于 run.output_dir，便于单步执行时直接定位输入输出。
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_context_path(self) -> Path:
        """运行上下文：目标列、任务类型等由 ingest 步骤确定的全局信息。"""
        return self.output_dir / "run_context.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def quality_report_path(self) -> Path:
        return self.output_dir / "quality_report.json"

    @property
    def strategy_path(self) -> Path:
        return self.output_dir / "strategy.json"

    @property
    def raw_data_path(self) -> Path:
        return self.output_dir / "raw.parquet"

    @property
    def train_raw_path(self) -> Path:
        return self.output_dir / "train_raw.parquet"

    @property
    def val_raw_path(self) -> Path:
        return self.output_dir / "val_raw.parquet"

    @property
    def test_raw_path(self) -> Path:
        return self.output_dir / "test_raw.parquet"

    @property
    def cv_results_path(self) -> Path:
        return self.output_dir / "cv_results.json"

    @property
    def preprocessor_path(self) -> Path:
        return self.output_dir / "preprocessing_pipeline.joblib"

    @property
    def feature_columns_path(self) -> Path:
        return self.output_dir / "feature_columns.json"

    @property
    def train_transformed_path(self) -> Path:
        return self.output_dir / "train_transformed.parquet"

    @property
    def val_transformed_path(self) -> Path:
        return self.output_dir / "val_transformed.parquet"

    @property
    def test_transformed_path(self) -> Path:
        return self.output_dir / "test_transformed.parquet"

    @property
    def sampling_strategy_path(self) -> Path:
        return self.output_dir / "sampling_strategy.json"

    @property
    def sampled_train_path(self) -> Path:
        return self.output_dir / "sampled_train.parquet"

    @property
    def sample_weight_path(self) -> Path:
        return self.output_dir / "sample_weight.parquet"

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "autogluon_models"

    @property
    def leaderboard_path(self) -> Path:
        return self.output_dir / "leaderboard.csv"

    @property
    def feature_importance_path(self) -> Path:
        return self.output_dir / "feature_importance.csv"

    @property
    def permutation_importance_path(self) -> Path:
        return self.output_dir / "permutation_importance.csv"

    @property
    def shap_values_path(self) -> Path:
        return self.output_dir / "shap_values.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    @property
    def interpretation_path(self) -> Path:
        return self.output_dir / "business_interpretation.json"

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.html"

    def step_manifest_path(self, step_name: str) -> Path:
        """单个步骤的输入/输出 manifest 文件。"""
        return self.output_dir / f"step_{step_name}_manifest.json"

    def load_json(self, path: Path, default: Any = None) -> Any:
        """安全读取 JSON 文件。"""
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default

    def save_json(self, path: Path, data: Any) -> None:
        """保存 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_run_context(self) -> Dict[str, Any]:
        """读取运行上下文。"""
        ctx = self.load_json(self.run_context_path, {})
        if not ctx:
            raise FileNotFoundError(f"运行上下文不存在: {self.run_context_path}")
        return ctx

    def save_run_context(self, context: Dict[str, Any]) -> None:
        """保存运行上下文。"""
        self.save_json(self.run_context_path, context)

    def load_config_snapshot(self) -> Dict[str, Any]:
        """读取 config_snapshot.json。"""
        return self.load_json(self.output_dir / "config_snapshot.json", {})
