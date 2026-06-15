# Prefect AutoML Platform

端到端全自动机器学习平台，基于 **Prefect** 工作流编排和 **AutoGluon** 模型内核。

## 项目定位

本项目是一个独立的全栈 AutoML 平台，通过前端导航链接与 `tech-portfolio` 集成：

```html
<a href="http://localhost:8084" target="_blank">AutoML 平台</a>
```

## 技术栈

- **后端**：FastAPI + SQLAlchemy + Prefect + AutoGluon
- **前端**：Vue 3 + Element Plus + Vite
- **数据库**：SQLite（开发）/ PostgreSQL（生产）
- **持久化**：文件系统（数据集 / 模型 / 报告）+ 数据库（元数据）

## 快速开始

### 1. 环境准备

```bash
cd prefect-project
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

> 默认 `requirements.txt` 安装的是 AutoGluon Tabular CPU 子集（LightGBM / XGBoost / CatBoost），不包含 PyTorch 和 fastai，可显著减少安装时间。如需完整版（含神经网络模型），请安装 `requirements-full.txt`。

### 2. 启动后端

```bash
cd backend
uvicorn main:app --reload --port 8001
```

后端文档：`http://localhost:8001/docs`

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev -- --port 8084
```

前端地址：`http://localhost:8084`

### 4. 一键启动（开发环境）

```bash
bash scripts/run_dev.sh
```

## 服务器直接部署（生产环境）

本项目默认直接运行在服务器上，不依赖 Docker。

```bash
cd prefect-project

# 构建前端
cd frontend
npm install
npm run build

# 一键启动后端（8001）和前端（8084）
cd ..
bash scripts/run_prod.sh
```

启动后：
- 后端 API：`http://<服务器IP>:8001`
- 前端页面：`http://<服务器IP>:8084`
- 后端日志：`logs/backend.log`
- 前端日志：`logs/frontend.log`

停止服务：

```bash
bash scripts/run_prod_stop.sh
```

> 说明：训练任务在独立子进程中异步执行，并通过信号量控制最大并发数（默认 2），避免长时间训练阻塞 FastAPI 主服务。

## 核心功能

1. 数据集上传（CSV / Excel / Parquet / 数据库）
2. Schema 推断与目标列选择
3. 全自动 AutoML 训练（Prefect 编排 + AutoGluon）
4. 训练状态实时查询
5. 模型评估指标与排行榜
6. SHAP 可解释性报告
7. 模型下载与预测

## 核心 API

```
POST   /api/datasets/upload        # 上传数据集
GET    /api/datasets               # 列出数据集
GET    /api/datasets/{id}          # 数据集详情
GET    /api/datasets/{id}/preview  # 预览数据集
DELETE /api/datasets/{id}          # 删除数据集

POST   /api/runs                   # 启动训练任务
GET    /api/runs                   # 列出训练任务
GET    /api/runs/{id}              # 任务详情
GET    /api/runs/{id}/results      # 训练结果
POST   /api/runs/{id}/predict      # 模型预测
DELETE /api/runs/{id}              # 删除任务
```

## Python 版本支持

当前支持 **Python 3.12**。未来将跟踪 AutoGluon 对 Python 3.14 / 3.15 的支持进度进行适配，详见 `todo.md` 阶段 10。

## 项目结构

```
prefect-project/
├── backend/                 # FastAPI 后端
│   ├── main.py             # 应用入口
│   ├── routers/            # API 路由
│   ├── services/           # 业务服务
│   ├── prefect_flows/      # Prefect Flow
│   ├── models.py           # SQLAlchemy ORM
│   └── database.py         # 数据库连接
├── frontend/               # Vue 3 前端
│   ├── src/
│   │   ├── views/          # 页面组件
│   │   ├── api/            # API 客户端
│   │   └── router/         # 路由配置
│   └── package.json
├── data/                   # 数据持久化
│   ├── uploads/            # 上传数据集
│   ├── models/             # 训练好的模型
│   └── reports/            # 报告和图表
├── scripts/                # 运维脚本
├── tests/                  # 测试
├── task.md                 # 技术 PRD
├── todo.md                 # 开发 Todo
└── README.md               # 本文件
```

## 注意事项

- AutoML 训练任务在后台独立子进程中运行，避免阻塞 FastAPI 主服务
- 默认使用 SQLite，生产环境建议切换到 PostgreSQL
- 训练完成后可在 `data/reports/{run_id}/` 查看产物
