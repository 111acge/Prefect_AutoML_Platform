# Prefect AutoML Platform 常用命令

.PHONY: help check install install-full test test-slow lint format start stop dev clean

help:
	@echo "可用命令:"
	@echo "  make check         运行环境检查"
	@echo "  make install       安装核心依赖"
	@echo "  make install-full  安装完整依赖（含 NN/文本/LLM）"
	@echo "  make test          运行测试（跳过慢速）"
	@echo "  make test-slow     运行全部测试（含端到端）"
	@echo "  make lint          运行 ruff 静态检查"
	@echo "  make format        运行 black 格式化"
	@echo "  make start         启动生产服务"
	@echo "  make stop          停止生产服务"
	@echo "  make dev           启动开发服务"
	@echo "  make clean         清理缓存和构建产物"

check:
	@python scripts/check_env.py

install:
	@uv pip install -r requirements.txt

install-full:
	@uv pip install -r requirements-full.txt

test:
	@pytest tests -m "not slow" -v

test-slow:
	@pytest tests -v

lint:
	@ruff check backend tests

format:
	@black backend tests

start:
	@bash scripts/run_prod.sh

stop:
	@bash scripts/run_prod_stop.sh

dev:
	@bash scripts/run_dev.sh

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
