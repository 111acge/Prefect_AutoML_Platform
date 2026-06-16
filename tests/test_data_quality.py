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
