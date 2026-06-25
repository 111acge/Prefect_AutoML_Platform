#!/bin/bash
# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

set -e

# 启动 Prefect AutoML Platform 开发环境

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 激活虚拟环境
source .venv/bin/activate

# 启动后端
cd backend
echo "🚀 启动后端服务..."
uvicorn main:app --reload --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# 启动前端
cd ../frontend
echo "🚀 启动前端服务..."
npm run dev &
FRONTEND_PID=$!

# 捕获 Ctrl+C 信号
trap "echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait
