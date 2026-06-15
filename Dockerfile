FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-core.txt requirements-full.txt pyproject.toml ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements-core.txt

# 复制项目代码
COPY backend ./backend
COPY scripts ./scripts
COPY data ./data

# 创建数据目录
RUN mkdir -p data/uploads data/models data/reports

# 暴露端口
EXPOSE 8001

# 启动后端
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
