#!/usr/bin/env python3
# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""独立运行单个 AutoML 步骤的脚本。"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "backend"))

# 必须在导入 prefect 前设置
os.environ.setdefault("PREFECT_API_URL", "")

from i18n import set_locale  # noqa: E402
from services.llm_settings_service import load_llm_config  # noqa: E402
from services.step_runner import StepRunner, STEP_ORDER  # noqa: E402


def _write_error(output_dir: str, step_name: str, exc: Exception) -> dict:
    """把异常信息写入 error.json 并返回结构化错误。"""
    error_info = {
        "status": "failed",
        "step": step_name,
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


async def main_async():
    # 同步父进程传递的语言设置
    set_locale(os.environ.get("APP_LOCALE", "zh-CN"))

    import argparse

    parser = argparse.ArgumentParser(description="Run a single AutoML step")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--step",
        required=True,
        help=f"步骤名称之一: {', '.join(STEP_ORDER)} 或 'all'（顺序执行全部）",
    )
    args = parser.parse_args()

    # 加载用户配置的 LLM 设置（子进程不共享父进程的内存缓存）
    try:
        await load_llm_config()
    except Exception:
        pass

    runner = StepRunner(run_id=args.run_id, output_dir=args.output_dir)

    try:
        if args.step == "all":
            result = await runner.run_all()
        else:
            result = await runner.run_step(args.step)
    except Exception as exc:
        error_info = _write_error(args.output_dir, args.step, exc)
        print("\n===STEP_ERROR===", file=sys.stderr)
        print(json.dumps(error_info, ensure_ascii=False, default=str), file=sys.stderr)
        sys.exit(1)

    print("\n===STEP_RESULT===")
    print(json.dumps({"step": args.step, "status": "completed", "outputs": result}, default=str))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
