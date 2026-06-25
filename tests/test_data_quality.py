# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""数据质量服务测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd

from services.data_quality import assess_data_quality


def test_six_dimensions_report():
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, None, 100.0],
        "cat_a": ["x", "x", "y", "y", "z"],
        "target": [0, 1, 0, 1, 0],
    })
    report = assess_data_quality(df, target_column="target")

    for dim in ["completeness", "consistency", "accuracy", "timeliness", "uniqueness", "validity"]:
        assert dim in report

    assert report["n_rows"] == 5
    assert "overall_score" in report
    assert report["completeness"]["missing_rates"]["num_a"] > 0


def test_duplicate_rows_detected():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "target": [0, 0, 1],
    })
    report = assess_data_quality(df, target_column="target")
    assert report["uniqueness"]["duplicated_rows"] == 1


def test_negative_value_warning():
    df = pd.DataFrame({
        "income": [1000.0, -50.0, 2000.0],
        "target": [0, 1, 0],
    })
    report = assess_data_quality(df, target_column="target")
    cols = [c["column"] for c in report["accuracy"]["negative_in_non_negative_columns"]]
    assert "income" in cols


def test_assess_data_quality_task_writes_report(tmp_path):
    """数据质量任务应将报告写入输出目录。"""
    from prefect_flows.automl_flow import assess_data_quality_task

    df = pd.DataFrame({
        "a": [1, 2, 3, None],
        "target": [0, 1, 0, 1],
    })
    output_dir = str(tmp_path / "run")
    quality = assess_data_quality_task(df, target_column="target", output_dir=output_dir)

    assert "overall_score" in quality
    report_path = Path(output_dir) / "quality_report.json"
    assert report_path.exists()
    loaded = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert loaded["n_rows"] == 4


def test_assess_data_quality_respects_max_rows_config(monkeypatch):
    """数据质量评估应遵循 DATA_QUALITY_MAX_ROWS 配置进行采样。"""
    from config import settings
    monkeypatch.setattr(settings, "data_quality_max_rows", 3)

    df = pd.DataFrame({
        "a": list(range(10)),
        "target": [0, 1] * 5,
    })
    report = assess_data_quality(df, target_column="target")
    assert report["n_rows"] == 3


def test_assess_data_quality_max_rows_parameter_overrides_config(monkeypatch):
    """assess_data_quality 的 max_rows 参数应覆盖配置。"""
    from config import settings
    monkeypatch.setattr(settings, "data_quality_max_rows", 5)

    df = pd.DataFrame({
        "a": list(range(10)),
        "target": [0, 1] * 5,
    })
    report = assess_data_quality(df, target_column="target", max_rows=2)
    assert report["n_rows"] == 2


def test_assess_data_quality_auto_samples_large_data(monkeypatch):
    """大数据集下数据质量评估应自动采样。"""
    import services.data_quality as dq
    monkeypatch.setattr(dq, "LARGE_DATASET_ROW_THRESHOLD", 20)
    monkeypatch.setattr(dq, "DEFAULT_DATA_QUALITY_SAMPLE_SIZE", 10)

    df = pd.DataFrame({
        "a": list(range(30)),
        "target": [0, 1] * 15,
    })
    report = assess_data_quality(df, target_column="target")
    assert report["n_rows"] == 10
