#!/usr/bin/env python3
"""独立运行 Prefect AutoML Flow 的脚本。"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "backend"))

# 必须在导入 prefect 前设置
os.environ.setdefault("PREFECT_API_URL", "")

from prefect_flows.automl_flow import automl_pipeline  # noqa: E402


def _write_error(output_dir: str, exc: Exception) -> dict:
    """把异常信息写入 error.json 并返回结构化错误。"""
    error_info = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    error_path = Path(output_dir) / "error.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(error_info, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return error_info


def main():
    parser = argparse.ArgumentParser(description="Run AutoML Flow")
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-budget-minutes", type=float, default=10)
    parser.add_argument("--preset", default="medium_quality")
    parser.add_argument("--primary-metric", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=50)
    parser.add_argument("--callback-url", default=None)
    parser.add_argument("--cleaning-rules", default=None, help="JSON 字符串形式的清洗规则")
    parser.add_argument(
        "--candidate-config",
        default=None,
        help="JSON 字符串形式的 Agent 候选配置（覆盖策略参数）",
    )
    parser.add_argument(
        "--no-feature-engineering",
        action="store_true",
        help="关闭高级特征工程（仅保留清洗、填充、对数变换）",
    )
    args = parser.parse_args()

    try:
        result = automl_pipeline(
            file_path=args.file_path,
            target_column=args.target_column,
            task_type=args.task_type,
            output_dir=args.output_dir,
            time_budget_minutes=(
                None if args.time_budget_minutes == -1 else args.time_budget_minutes
            ),
            preset=args.preset,
            primary_metric=args.primary_metric,
            seed=args.seed,
            max_models=args.max_models,
            cleaning_rules=json.loads(args.cleaning_rules) if args.cleaning_rules else None,
            feature_engineering_enabled=not args.no_feature_engineering,
            candidate_config=json.loads(args.candidate_config) if args.candidate_config else None,
        )
    except Exception as exc:
        error_info = _write_error(args.output_dir, exc)
        print("\n===FLOW_ERROR===", file=sys.stderr)
        print(json.dumps(error_info, ensure_ascii=False, default=str), file=sys.stderr)
        raise

    # 输出结果到 stdout，方便父进程捕获
    print("\n===FLOW_RESULT===")
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "preset": result.get("preset"),
                "primary_metric": result.get("train_result", {}).get("primary_metric"),
            }
        )
    )


if __name__ == "__main__":
    main()
