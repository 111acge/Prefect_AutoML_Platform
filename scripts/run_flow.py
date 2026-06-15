#!/usr/bin/env python3
"""独立运行 Prefect AutoML Flow 的脚本。"""

import os
import sys
import json
import argparse

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "backend"))

# 必须在导入 prefect 前设置
os.environ.setdefault("PREFECT_API_URL", "")

from prefect_flows.automl_flow import automl_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run AutoML Flow")
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-budget-minutes", type=float, default=10)
    parser.add_argument("--preset", default="medium_quality")
    parser.add_argument("--primary-metric", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=50)
    parser.add_argument("--callback-url", default=None)
    args = parser.parse_args()

    result = automl_pipeline(
        file_path=args.file_path,
        target_column=args.target_column,
        task_type=args.task_type,
        output_dir=args.output_dir,
        time_budget_minutes=args.time_budget_minutes,
        preset=args.preset,
        primary_metric=args.primary_metric,
        seed=args.seed,
        max_models=args.max_models,
    )

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
