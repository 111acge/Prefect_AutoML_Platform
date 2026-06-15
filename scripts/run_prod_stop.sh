#!/bin/bash

# 停止 Prefect AutoML Platform 生产服务

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🛑 停止服务..."

if [ -f "logs/backend.pid" ]; then
    kill "$(cat logs/backend.pid)" 2>/dev/null || true
    rm -f logs/backend.pid
fi

if [ -f "logs/frontend.pid" ]; then
    kill "$(cat logs/frontend.pid)" 2>/dev/null || true
    rm -f logs/frontend.pid
fi

# 兜底：按端口停止
lsof -ti:8001 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8084 | xargs -r kill -9 2>/dev/null || true

echo "✅ 服务已停止"
