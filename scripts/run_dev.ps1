# Prefect AutoML Platform - Windows 开发环境一键启动脚本 (PowerShell)
# 在 PowerShell 中运行：.\scripts\run_dev.ps1
#
# 若提示“无法加载脚本，因为在此系统上禁止运行脚本”，请先执行：
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8


$PROJECT_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$BACKEND_PORT = 8001
$FRONTEND_PORT = 8084

function Write-Prefixed {
    param(
        [string]$Prefix,
        [string]$Message
    )
    Write-Host "[$Prefix] $Message"
}

function Start-TailJob {
    param(
        [string]$LogPath,
        [string]$Prefix
    )
    Start-Job -ScriptBlock {
        param($Path, $Prefix)
        Get-Content -Path $Path -Wait -Tail 0 -ErrorAction SilentlyContinue | ForEach-Object {
            "[$Prefix] $_"
        }
    } -ArgumentList $LogPath, $Prefix
}


function Test-PortInUse {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue)
}

foreach ($port in @($BACKEND_PORT, $FRONTEND_PORT)) {
    if (Test-PortInUse -Port $port) {
        Write-Host "[警告] 端口 $port 已被占用，新服务可能无法正常启动。" -ForegroundColor Yellow
    }
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Prefect AutoML Platform" -ForegroundColor Cyan
Write-Host "  开发环境一键启动" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 检查虚拟环境（不依赖 Activate.ps1 的执行策略，直接调用 python）
$PYTHON = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Host "[错误] 未找到 .venv 虚拟环境，请先创建并安装依赖。" -ForegroundColor Red
    Write-Host "参考命令：uv venv --python 3.12; uv pip install -r requirements-core.txt" -ForegroundColor Yellow
    exit 1
}

# 配置本地 Node.js 环境
$NODE_DIR = Join-Path $PROJECT_ROOT ".node"
if (Test-Path $NODE_DIR) {
    $env:PATH = "$NODE_DIR;$env:PATH"
}

$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCmd) {
    Write-Host "[错误] 未找到 npm，请确认 .node 目录存在或已安装 Node.js。" -ForegroundColor Red
    exit 1
}

# 检查前端依赖
$NODE_MODULES = Join-Path $PROJECT_ROOT "frontend\node_modules"
if (-not (Test-Path $NODE_MODULES)) {
    Write-Host "[错误] 前端依赖未安装，请先运行：cd frontend; npm install" -ForegroundColor Red
    exit 1
}

# 日志目录
$LOGS_DIR = Join-Path $PROJECT_ROOT "logs"
New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

$BACKEND_LOG = Join-Path $LOGS_DIR "backend.log"
$BACKEND_ERR_LOG = Join-Path $LOGS_DIR "backend.err.log"
$FRONTEND_LOG = Join-Path $LOGS_DIR "frontend.log"
$FRONTEND_ERR_LOG = Join-Path $LOGS_DIR "frontend.err.log"

# 预先创建/清空日志文件
$BACKEND_LOG, $BACKEND_ERR_LOG, $FRONTEND_LOG, $FRONTEND_ERR_LOG | ForEach-Object {
    try {
        if (Test-Path $_) { Clear-Content $_ -Force -ErrorAction SilentlyContinue }
        else { New-Item -ItemType File -Path $_ -Force | Out-Null }
    } catch {
        # 文件被占用时直接追加
    }
}

# 启动后端
Write-Host "[1/2] 正在启动后端服务（端口 $BACKEND_PORT）..." -ForegroundColor Green
$backendProc = Start-Process `
    -FilePath $PYTHON `
    -ArgumentList "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", $BACKEND_PORT `
    -WorkingDirectory (Join-Path $PROJECT_ROOT "backend") `
    -RedirectStandardOutput $BACKEND_LOG `
    -RedirectStandardError $BACKEND_ERR_LOG `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3

# 启动前端
Write-Host "[2/2] 正在启动前端服务（端口 $FRONTEND_PORT）..." -ForegroundColor Green
$frontendProc = Start-Process `
    -FilePath $npmCmd.Source `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $PROJECT_ROOT "frontend") `
    -RedirectStandardOutput $FRONTEND_LOG `
    -RedirectStandardError $FRONTEND_ERR_LOG `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

# 检查进程是否存活
if ($backendProc.HasExited) {
    Write-Host "[错误] 后端进程未能正常启动，请查看日志：$BACKEND_LOG / $BACKEND_ERR_LOG" -ForegroundColor Red
    if (Test-Path $BACKEND_ERR_LOG) { Get-Content $BACKEND_ERR_LOG | Write-Host -ForegroundColor Red }
    exit 1
}
if ($frontendProc.HasExited) {
    Write-Host "[错误] 前端进程未能正常启动，请查看日志：$FRONTEND_LOG / $FRONTEND_ERR_LOG" -ForegroundColor Red
    if (Test-Path $FRONTEND_ERR_LOG) { Get-Content $FRONTEND_ERR_LOG | Write-Host -ForegroundColor Red }
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[OK] 服务已启动" -ForegroundColor Green
Write-Host ""
Write-Host "  前端页面：http://localhost:$FRONTEND_PORT"
Write-Host "  后端 API：http://localhost:$BACKEND_PORT"
Write-Host "  API 文档：http://localhost:$BACKEND_PORT/docs"
Write-Host ""
Write-Host "  后端日志：$BACKEND_LOG"
Write-Host "  前端日志：$FRONTEND_LOG"
Write-Host ""
Write-Host "  按 Ctrl+C 停止服务" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# 启动日志 tail 作业
$backendOutJob = Start-TailJob -LogPath $BACKEND_LOG -Prefix "BACKEND"
$backendErrJob = Start-TailJob -LogPath $BACKEND_ERR_LOG -Prefix "BACKEND-ERR"
$frontendOutJob = Start-TailJob -LogPath $FRONTEND_LOG -Prefix "FRONTEND"
$frontendErrJob = Start-TailJob -LogPath $FRONTEND_ERR_LOG -Prefix "FRONTEND-ERR"

$jobs = @($backendOutJob, $backendErrJob, $frontendOutJob, $frontendErrJob)

# 持续输出日志，直到用户按 Ctrl+C
try {
    while ($true) {
        foreach ($job in $jobs) {
            Receive-Job -Job $job | ForEach-Object { Write-Host $_ }
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host ""
    Write-Host "正在停止服务..." -ForegroundColor Yellow

    Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $frontendProc.Id -Force -ErrorAction SilentlyContinue

    Stop-Job -Job $jobs -ErrorAction SilentlyContinue
    Remove-Job -Job $jobs -ErrorAction SilentlyContinue

    # 尝试停止可能残留的子进程
    $ports = @($BACKEND_PORT, $FRONTEND_PORT)
    foreach ($port in $ports) {
        try {
            $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($conn -and $conn.OwningProcess) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        } catch {
            # 忽略停止进程时的错误
        }
    }

    Write-Host "服务已停止。" -ForegroundColor Green
}
