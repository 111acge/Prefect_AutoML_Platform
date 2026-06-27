#!/bin/bash
# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

# 停止 Prefect Server 与本地 Runner

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🛑 停止 Prefect 服务..."

if [ -f "logs/prefect_runner.pid" ]; then
    RUNNER_PID=$(cat logs/prefect_runner.pid)
    kill "$RUNNER_PID" 2>/dev/null || true
    rm -f logs/prefect_runner.pid
fi

if [ -f "logs/prefect_server.pid" ]; then
    SERVER_PID=$(cat logs/prefect_server.pid)
    kill "$SERVER_PID" 2>/dev/null || true
    rm -f logs/prefect_server.pid
fi

# 兜底：按端口停止
lsof -ti:4200 | xargs -r kill -9 2>/dev/null || true

echo "✅ Prefect 服务已停止"
