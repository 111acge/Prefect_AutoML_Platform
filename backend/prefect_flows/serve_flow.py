#!/usr/bin/env python3
# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""创建并 serve AutoML Prefect Deployment。

运行此脚本会：
1. 连接到 PREFECT_API_URL 指定的 Prefect Server。
2. 为 automl_pipeline 创建/更新 Deployment。
3. 启动本地 Runner，监听并执行调度到的 Flow Run。
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "backend"))

from config import configure_prefect_api_url, settings  # noqa: E402

configure_prefect_api_url()

from prefect_flows.automl_flow import automl_pipeline  # noqa: E402


def main() -> None:
    if not settings.prefect_enabled:
        print("PREFECT_ENABLED=false，跳过 Prefect Deployment serve。", file=sys.stderr)
        sys.exit(0)

    os.makedirs(settings.prefect_home, exist_ok=True)

    print(f"Serving Prefect flow '{settings.prefect_flow_name}' "
          f"as deployment '{settings.prefect_deployment_name}' ...")
    print(f"Prefect API: {settings.prefect_api_url or 'ephemeral'}")

    # serve() 会创建 Deployment 并启动本地 Runner（工作进程）
    # limit=10 避免 Prefect Runner 在顺序提交时因 subprocess 清理延迟而跳过运行
    automl_pipeline.serve(
        name=settings.prefect_deployment_name,
        pause_on_shutdown=True,
        limit=10,
    )


if __name__ == "__main__":
    main()
