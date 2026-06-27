#!/bin/bash
# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

set -e

# Prefect AutoML Platform 生产环境启动脚本
# 直接运行在服务器上，不使用 Docker

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志目录
mkdir -p logs

# 激活虚拟环境
source .venv/bin/activate

# 启动 Prefect 服务（Server + Runner）
echo "🚀 启动 Prefect 编排服务..."
"$PROJECT_ROOT/scripts/start_prefect.sh"

# 启动后端
cd backend
echo "🚀 启动后端服务 (端口 8001)..."
nohup uvicorn main:app --host 0.0.0.0 --port 8001 > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PROJECT_ROOT/logs/backend.pid"

# 启动前端（使用 vite dev server，利用其 /api 代理到后端，仅需暴露 8084 端口）
cd "$PROJECT_ROOT/frontend"
echo "🚀 启动前端服务 (端口 8084)..."
nohup npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PROJECT_ROOT/logs/frontend.pid"

echo ""
echo "✅ 服务已启动"
echo "  后端: http://<服务器IP>:8001"
echo "  前端: http://<服务器IP>:8084"
echo "  后端日志: $PROJECT_ROOT/logs/backend.log"
echo "  前端日志: $PROJECT_ROOT/logs/frontend.log"
echo ""
echo "停止服务: kill $BACKEND_PID $FRONTEND_PID"
