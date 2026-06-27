#!/bin/bash
# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

# 启动 Prefect Server 与本地 Runner（Deployment serve）

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 激活虚拟环境
source .venv/bin/activate

# 日志目录
mkdir -p logs

# 读取 .env 中的 PREFECT_ENABLED
PREFECT_ENABLED=$(python - <<'PY'
import os, sys
sys.path.insert(0, "backend")
from config import settings
print("true" if settings.prefect_enabled else "false")
PY
)

if [ "$PREFECT_ENABLED" != "true" ]; then
    echo "⏭️  PREFECT_ENABLED=false，跳过 Prefect 服务启动"
    exit 0
fi

start_runner() {
    if [ -f "logs/prefect_runner.pid" ]; then
        RUNNER_PID=$(cat logs/prefect_runner.pid)
        if kill -0 "$RUNNER_PID" 2>/dev/null; then
            echo "✅ Prefect Runner 已在运行 (PID: $RUNNER_PID)"
            return
        fi
    fi

    echo "🚀 启动 Prefect Runner (serve automl-end-to-end)..."
    cd "$PROJECT_ROOT/backend"
    nohup python -c "import sys; sys.path.insert(0, '.'); from prefect_flows.serve_flow import main; main()" \
        > "$PROJECT_ROOT/logs/prefect_runner.log" 2>&1 &
    RUNNER_PID=$!
    echo $RUNNER_PID > "$PROJECT_ROOT/logs/prefect_runner.pid"
    echo "✅ Prefect Runner 已启动 (PID: $RUNNER_PID)"
}

# 若已有 PID 文件且进程存活，则不重复启动
if [ -f "logs/prefect_server.pid" ]; then
    SERVER_PID=$(cat logs/prefect_server.pid)
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "✅ Prefect Server 已在运行 (PID: $SERVER_PID)"
        start_runner
        exit 0
    fi
fi

echo "🚀 启动 Prefect Server (端口 4200)..."
nohup prefect server start --host 0.0.0.0 --port 4200 > "$PROJECT_ROOT/logs/prefect_server.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PROJECT_ROOT/logs/prefect_server.pid"

# 等待 Prefect Server API 就绪（最多 60 秒）
echo "⏳ 等待 Prefect Server 就绪..."
for i in $(seq 1 60); do
    if curl -fsS "http://localhost:4200/api/health" > /dev/null 2>&1; then
        echo "✅ Prefect Server 已就绪"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "❌ Prefect Server 启动失败，查看 logs/prefect_server.log"
        exit 1
    fi
    sleep 1
done

start_runner

echo ""
echo "Prefect UI: http://localhost:4200"
echo "Prefect Server 日志: $PROJECT_ROOT/logs/prefect_server.log"
echo "Prefect Runner 日志: $PROJECT_ROOT/logs/prefect_runner.log"
