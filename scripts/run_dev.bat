@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Prefect AutoML Platform - Windows 开发环境一键启动脚本
:: 双击运行即可同时启动后端（8001）和前端（8084）

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo ==========================================
echo   Prefect AutoML Platform
echo   开发环境一键启动
echo ==========================================
echo.

:: 检查虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo [错误] 未找到 .venv 虚拟环境，请先创建并安装依赖。
    echo 参考命令：uv venv --python 3.12 ^&^& uv pip install -r requirements-core.txt
    pause
    exit /b 1
)

:: 检查前端依赖
if not exist "frontend\node_modules" (
    echo [错误] 前端依赖未安装，请先运行：cd frontend ^&^& npm install
    pause
    exit /b 1
)

echo [0/2] 配置本地 Node.js 环境...
set "PATH=%PROJECT_ROOT%\.node;%PATH%"
where npm >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 npm，请确认 .node 目录存在或已安装 Node.js。
    pause
    exit /b 1
)

echo [1/2] 正在启动后端服务（端口 8001）...
start "Backend - Prefect AutoML (8001)" cmd /k "cd /d %PROJECT_ROOT% && .venv\Scripts\activate.bat && cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8001"

ping -n 4 127.0.0.1 >nul

echo [2/2] 正在启动前端服务（端口 8084）...
start "Frontend - Prefect AutoML (8084)" cmd /k "cd /d %PROJECT_ROOT%\frontend && npm run dev"

echo.
echo ==========================================
echo [OK] 服务启动中...
echo.
echo   前端页面：http://localhost:8084
echo   后端 API：http://localhost:8001
echo   API 文档：http://localhost:8001/docs
echo.
echo   关闭弹出的两个窗口即可停止服务。
echo ==========================================
pause
