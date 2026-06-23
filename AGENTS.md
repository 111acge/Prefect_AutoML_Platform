# AGENTS.md — Prefect AutoML Platform

> 本文件面向 AI 编程助手。阅读者应当被假设为对该项目一无所知。文中信息均来自实际代码与配置文件，请勿做无依据的推测。
> 若你修改了本文件描述的文件/结构/配置/工作流，请同步更新本文件。

---

## 1. 项目概述

**Prefect AutoML Platform** 是一个端到端全自动机器学习平台。

核心目标：用户上传数据文件（CSV / Excel / Parquet / JSONL）并选择目标列，系统在算力与时间预算内自动完成 **数据清洗 → 特征工程 → 策略路由 → 模型搜索 → 评估 → 预测 → 报告输出**。

项目采用**前后端分离**架构：

- 后端：FastAPI + SQLAlchemy 2.x + Prefect 3.x + AutoGluon Tabular
- 前端：Vue 3 + Element Plus + ECharts + Vite
- 数据库：SQLite（默认）/ PostgreSQL（生产建议）

项目文档：

- [`README.md`](./README.md)：面向人类贡献者的快速开始与功能说明。
- [`todo.md`](./todo.md)：开发排期与实现状态。
- 本文件：面向 AI 编码代理的工程上下文与约定。
- 注意：`task.md` 在 README/todo 中被引用，但项目根目录目前**不存在**该文件。

---

## 2. 技术栈与版本约束

### 2.1 Python 后端

| 类别 | 主要依赖 | 版本约束 |
|------|----------|----------|
| Web 框架 | FastAPI + Uvicorn | `fastapi>=0.111.0,<0.115.0` |
| 数据校验 | Pydantic 2.x + pydantic-settings | `pydantic>=2.0.0` |
| 数据库 | SQLAlchemy 2.x（异步）+ aiosqlite + alembic | `sqlalchemy>=2.0.0` |
| 工作流编排 | Prefect 3.x | `prefect>=3.0.0,<4.0.0` |
| AutoML 内核 | AutoGluon Tabular | `autogluon.tabular[lightgbm,catboost,xgboost]>=1.1.0` |
| 数据科学 | pandas, numpy, scikit-learn, pyarrow | Python `>=3.12,<3.13` |
| 特征工程 | feature-engine | `>=1.6.0` |
| 采样 | imbalanced-learn | `>=0.11.0` |
| 可解释性 | SHAP | `>=0.44.0` |
| 报告 | Jinja2 + matplotlib + seaborn | — |
| LLM API | openai（兼容 KIMI / DeepSeek / MiniMax / OpenAI） | `>=1.0.0` |
| 深度学习 | torch（`pyproject.toml` 已列入，用于 NeuralNetTorch） | `>=2.0.0` |

> **依赖文件注意**：
> - `pyproject.toml` 的 `dependencies` 中已包含 `torch>=2.0.0`，并配置了 `pytorch-cpu` 索引。
> - `requirements-core.txt` 与 `requirements.txt` 当前**未列出 torch**，与 `pyproject.toml` 存在差异；若通过 `uv pip install -r requirements-core.txt` 安装， NeuralNetTorch 可能不可用。
> - `requirements-full.txt` 使用完整版 `autogluon>=1.1.0`，并包含 feature-engine、imbalanced-learn、shap、openai。

### 2.2 Node 前端

| 类别 | 主要依赖 |
|------|----------|
| 框架 | Vue 3 (`^3.4.0`) |
| 构建工具 | Vite 5 (`^5.2.0`) |
| UI 组件库 | Element Plus (`^2.7.0`) |
| 图标 | `@element-plus/icons-vue` (`^2.3.0`) |
| 状态管理 | Pinia (`^2.1.0`) |
| 路由 | Vue Router 4 (`^4.3.0`) |
| 可视化 | ECharts (`^6.1.0`) |
| HTTP | axios (`^1.7.0`) |

### 2.3 环境要求

- **Python 3.12**（`pyproject.toml` 锁定 `>=3.12,<3.13`）。
- **Node.js 18+**。
- 推荐包管理器：**uv**（也支持 pip）。
- 内存：最低 8GB，推荐 32GB+（模型搜索内存消耗大）。
- 项目根目录存在 `.node/`，可用于存放本地 Node.js 运行时（Windows PowerShell 启动脚本会优先将其加入 PATH）。

---

## 3. 项目结构

```text
prefect-project/
├── backend/                    # FastAPI 后端
│   ├── main.py                 # FastAPI 入口与生命周期
│   ├── config.py               # Pydantic Settings 配置管理
│   ├── database.py             # SQLAlchemy 异步引擎与会话 + SQLite 轻量迁移
│   ├── models.py               # SQLAlchemy ORM 模型（Dataset / Experiment / Trial / Run / Metric）
│   ├── schemas.py              # Pydantic 请求/响应模型
│   ├── routers/                # API 路由
│   │   ├── datasets.py         # 数据集接口
│   │   ├── runs.py             # 训练任务、预测、解释、报告、SSE、对比接口
│   │   ├── intent.py           # 自然语言意图接口
│   │   └── experiments.py      # LLM 驱动的多候选搜索实验接口
│   ├── services/               # 业务服务层
│   │   ├── automl.py           # AutoGluon 封装（训练、集成回退、指标映射）
│   │   ├── training_strategy.py# 数据驱动的训练策略路由
│   │   ├── training_executor.py# 异步训练执行器（子进程 + 信号量 + SSE）
│   │   ├── data_service.py     # 数据加载与元数据分析
│   │   ├── schema_service.py   # Schema 推断/校验/对齐
│   │   ├── data_quality.py     # 六维数据质量报告
│   │   ├── preprocessing.py    # 基础清洗与特征工程（清洗、划分、特征构造）
│   │   ├── preprocessing_pipeline.py # 可序列化 DataPreprocessor（fit/transform/save/load）
│   │   ├── feature_engineering.py    # 高级特征工程
│   │   ├── sampling_service.py       # 条件采样
│   │   ├── cv_service.py             # 显式交叉验证
│   │   ├── explainability.py         # SHAP / Permutation Importance
│   │   ├── visualization.py          # 报告图表
│   │   ├── llm_client.py             # 统一 LLM 调用客户端
│   │   ├── llm_intent_service.py     # LLM 意图解析（带降级）
│   │   ├── llm_strategy_service.py   # LLM 策略/候选推荐
│   │   ├── report_llm_service.py     # LLM 业务解读
│   │   ├── search_agent.py           # LLM 驱动的多候选搜索 Agent（Experiment）
│   │   ├── db_connection_service.py  # 数据库连接
│   │   ├── storage.py                # 文件存储
│   │   └── seed_data.py              # 默认 iris 数据集
│   ├── prefect_flows/
│   │   └── automl_flow.py      # Prefect Flow 定义（含预处理 Task、CV Task、业务解读 Task）
│   └── templates/
│       └── report.html         # HTML 报告 Jinja2 模板
├── frontend/                   # Vue 3 前端
│   ├── src/
│   │   ├── api/index.js        # axios 实例与 API 封装
│   │   ├── views/              # 页面视图（Home / Datasets / Runs / RunDetail / Compare）
│   │   ├── components/         # 公共组件（EChart.vue）
│   │   ├── router/index.js     # Vue Router
│   │   ├── stores/app.js       # Pinia store
│   │   ├── App.vue
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── scripts/                    # 运维与工具脚本
│   ├── check_env.py            # 环境检查
│   ├── run_dev.sh              # Linux/macOS 开发一键启动
│   ├── run_dev.ps1             # Windows PowerShell 开发一键启动
│   ├── run_prod.sh             # 生产启动（Linux/macOS）
│   ├── run_prod_stop.sh        # 停止生产服务
│   └── run_flow.py             # 独立运行 Prefect Flow
├── tests/                      # pytest 测试（21 个 test_*.py + __init__.py）
├── data/                       # 上传数据、模型、报告、默认数据集
│   ├── default/                # 默认数据集（iris.csv）
│   ├── uploads/                # 上传文件
│   ├── models/                 # 保存的模型
│   └── reports/                # 训练产物与 HTML 报告
├── .prefect/                   # Prefect 本地无服务器模式数据
├── .node/                      # 可选的本地 Node.js 运行时
├── pyproject.toml              # 项目元数据、依赖与工具配置
├── requirements-core.txt       # 核心依赖（CPU 版，无 torch/fastai）
├── requirements-full.txt       # 完整依赖（含 torch/fastai/NN/LLM）
├── requirements.txt            # 当前与 requirements-core.txt 内容一致
├── Makefile                    # 常用命令快捷方式
├── .env.example                # 环境变量示例
├── README.md
└── todo.md
```

---

## 4. 构建与运行命令

### 4.1 环境准备

推荐方式（uv）：

```bash
uv venv --python 3.12
# Linux/macOS
source .venv/bin/activate
# Windows: .venv\Scripts\activate

# 核心依赖（CPU 版 AutoGluon，安装更快；注意不含 torch）
uv pip install -r requirements-core.txt

# 完整依赖（含 torch / fastai / NN / LLM API）
uv pip install -r requirements-full.txt

# 或直接从 pyproject.toml 安装（含 torch CPU 版）
uv pip install -e .
```

pip 方式也可行，但项目 README 与 Makefile 默认以 uv 为主。

### 4.2 安装开发依赖

```bash
# pyproject.toml 中 [project.optional-dependencies] dev 包含 pytest / black / ruff / mypy
uv pip install -e ".[dev]"
```

### 4.3 开发启动

分别启动前后端：

```bash
# 后端（端口 8001）
cd backend
uvicorn main:app --app-dir . --reload --host 0.0.0.0 --port 8001

# 前端（端口 8084）
cd frontend
npm install
npm run dev
```

或使用一键脚本：

```bash
# Linux/macOS
bash scripts/run_dev.sh

# Windows PowerShell（需允许执行脚本）
.\scripts\run_dev.ps1
```

开发时前端 Vite 代理：`/api` → `http://127.0.0.1:8001`（见 `frontend/vite.config.js`）。

### 4.4 生产部署

```bash
cd frontend && npm install && npm run build && cd ..
bash scripts/run_prod.sh
```

- 后端 API：`http://<服务器IP>:8001`
- 前端页面：`http://<服务器IP>:8084`
- 后端日志：`logs/backend.log`
- 前端日志：`logs/frontend.log`

停止服务：

```bash
bash scripts/run_prod_stop.sh
```

### 4.5 Makefile 快捷命令

```bash
make check         # 运行环境检查
make install       # 安装 requirements.txt（当前等价于核心依赖）
make install-full  # 安装完整依赖
make test          # 运行测试（跳过慢速）
make test-slow     # 运行全部测试
make lint          # ruff 静态检查
make format        # black 格式化
make dev           # 开发启动（Linux/macOS）
make start         # 生产启动
make stop          # 停止生产服务
make clean         # 清理缓存
```

---

## 5. 测试策略

### 5.1 测试框架

- **pytest** + **pytest-asyncio**（异步测试）。
- 标记：
  - `@pytest.mark.slow`：端到端训练等慢速测试。
  - 默认 `make test` 会跳过这些测试。
- 测试文件位于 `tests/`，共 21 个 `test_*.py` 文件（加上 `__init__.py` 共 22 个 Python 文件）。

### 5.2 常用测试命令

```bash
# 跳过慢速测试
pytest tests -v -m "not slow"

# 全部测试
pytest tests -v

# 或
make test
make test-slow
```

### 5.3 测试约定

- 几乎所有测试文件开头执行 `sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))`，以便直接导入 `backend` 包。
- `test_api.py` 使用独立测试数据库 `sqlite+aiosqlite:///./data/test.db.sqlite`，并在 fixture 中每测试 `drop_all/create_all`。
- 大量使用 `monkeypatch` 与轻量伪对象（如 `_FakePredictor`）避免真实训练或外部 API 调用。
- 异步测试使用 `@pytest.mark.asyncio`。

### 5.4 主要测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_api.py` | FastAPI 路由、端到端训练（slow） |
| `test_automl_service.py` | Ensemble 回退、CV/Holdout 参数传递 |
| `test_training_strategy.py` | 数据驱动的策略路由 |
| `test_preprocessing_leakage.py` | 数据泄露防护 |
| `test_feature_engineering.py` | 特征工程 |
| `test_categorical_encoding.py` | 类别编码 |
| `test_sampling_service.py` | 条件采样 |
| `test_cv_service.py` | 交叉验证 |
| `test_data_quality.py` | 六维数据质量 |
| `test_explainability.py` | SHAP / Permutation Importance |
| `test_cleaning_rules.py` | 清洗规则 |
| `test_schema_service.py` | Schema 推断与校验 |
| `test_llm_intent_service.py` | LLM 意图解析与降级 |
| `test_report_llm_service.py` | 业务解读 |
| `test_db_connection.py` | 数据库连接 |
| `test_predict_threshold.py` | 阈值预测 |
| `test_experiments.py` | 实验/ Trial 接口 |
| `test_agent_services.py` | Agent 候选推荐 |
| `test_auto_inference.py` | 目标列/任务类型自动推断 |
| `test_business_rules.py` | 业务规则提取 |
| `test_optimization_defaults.py` | 性能优化默认值 |

---

## 6. 代码风格规范

### 6.1 Python

- 格式化：**Black**，行宽 `100`。
- 静态检查：**Ruff**，目标 Python 3.12。
- 类型检查：可选 **mypy**（`python_version = 3.12`）。
- 配置位于 `pyproject.toml`。

```bash
black backend tests
ruff check backend tests
mypy backend
```

### 6.2 JavaScript / Vue

- 前端使用 Vite 内置 ESLint 配置（`npm run lint`）。
- Vue 单文件组件使用 Composition API 风格（`<script setup>`）。

### 6.3 命名与注释

- 模块级 docstring 使用中文三引号字符串，说明模块职责。
- 函数/类名采用 `snake_case`（Python）或 `camelCase`（JavaScript）。
- 后端服务类多采用 `XxxService` 命名（如 `AutoMLService`、`TrainingExecutor`）。

---

## 7. 开发约定与架构重点

### 7.1 Prefect 编排模式

- `PREFECT_API_URL=""` 必须在导入 Prefect 前设置，强制本地无服务器模式。
- 见 `backend/main.py`、`scripts/run_flow.py`、`backend/prefect_flows/automl_flow.py` 顶部的 `os.environ.setdefault("PREFECT_API_URL", "")`。
- Flow：`automl_pipeline`（`@flow(name="automl-end-to-end", log_prints=True)`）。
- 关键 Task：
  - `load_data_task`：按文件路径 + mtime 缓存，retries=2。
  - `validate_schema_task`：校验目标列存在性。
  - `analyze_metadata_task`：元数据分析。
  - `assess_data_quality_task`：六维数据质量评估（失败不影响主流程）。
  - `build_strategy_task`：数据驱动策略，带缓存，合并 Agent 候选配置。
  - `split_data_task`：严格划分 train/val/test。
  - `cross_validate_task`：显式交叉验证（可选，失败不影响主流程）。
  - `fit_preprocessor_task` / `transform_data_task` / `persist_preprocessor_task`：预处理 Pipeline 的 fit/transform/save。
  - `build_sampling_strategy_task` / `apply_sampling_task`：条件采样。
  - `train_model_task`：超时 3 小时。
  - `evaluate_model_task`：超时 1 小时，生成 val/test/train 指标、扩展指标、阈值、集成验证。
  - `generate_business_interpretation_task`：LLM 业务解读（可选，失败不影响主流程）。
  - `create_artifacts_task`：Prefect Artifact（排行榜、特征重要性、策略、数据质量摘要）。
  - `generate_report_task`：HTML 报告。

### 7.2 防止数据泄露

- **训练/测试划分必须在预处理之前完成**。
- `DataPreprocessor` 只在 `train_df_raw` 上 `fit`，然后对验证集/测试集 `transform`。
- CV 在原始训练数据上运行，预处理器封装在 sklearn `Pipeline` 内部。
- 采样（SMOTE 等）只在训练集应用。

### 7.3 异步训练执行器

- `backend/services/training_executor.py` 中的 `TrainingExecutor` 在独立后台事件循环线程中通过子进程运行 `scripts/run_flow.py`。
- 使用 `asyncio.Semaphore` 控制最大并发，默认 **2** 个并发训练任务。
- FastAPI 主服务保持非阻塞。
- 支持 SSE（`GET /api/runs/{id}/events`）实时推送状态变化。
- 子进程日志会过滤 Prefect 内部 `EventsWorker` 等噪音。
- 全局超时后尝试 Best-so-far：若 `autogluon_models` 目录存在部分模型，则标记为 completed 并告警产物可能不完整。

### 7.4 降级与容错

- LLM 服务（意图解析、业务解读、候选推荐）未配置 API 密钥或调用失败时，自动降级到规则引擎。
- SHAP 失败时回退到 `shap.Explainer`。
- 采样失败时回退到 `RandomOverSampler`。
- Ensemble 提升不足 2% 时回退到最佳单模型。
- 全局超时后尝试返回当前最优结果（Best-so-far）。

### 7.5 产物与路径

- 上传数据：`data/uploads/{dataset_id}/`
- 训练产物：`data/reports/{run_id}/`
- 默认数据集：`data/default/`
- 数据库：`data/db.sqlite`
- Prefect 本地数据：`.prefect/`
- 目录由 `backend/config.py` 在导入时自动创建。

### 7.6 配置管理

- `backend/config.py` 使用 `pydantic_settings.BaseSettings`。
- 默认读取项目根目录 `.env` 文件。
- 关键配置项：
  - `DATABASE_URL`：默认 SQLite。
  - `LLM_PROVIDER`：`auto` / `kimi` / `deepseek` / `minimax` / `openai`。
  - `KIMI_API_KEY` / `DEEPSEEK_API_KEY` / `MINIMAX_API_KEY` / `OPENAI_API_KEY`。
  - `DEFAULT_LLM_MODEL`：覆盖默认模型。
  - 性能/稳定性优化开关（默认自动，根据 `n_classes` / `n_samples` 决策；`.env` 显式设置可覆盖）：
    - `TRAIN_EVAL_ENABLED` / `TRAIN_EVAL_SAMPLE_SIZE`：训练集参考评估（`n_classes > 50` 或 `n_samples > 100_000` 时自动采样 200 行）。
    - `SHAP_ENABLED` / `SHAP_MAX_SAMPLE_SIZE`：SHAP 可解释性（`n_classes > 50` 时采样 50 行；`n_classes > 200` 时自动跳过）。
    - `PERMUTATION_IMPORTANCE_ENABLED` / `PERMUTATION_IMPORTANCE_MAX_REPEATS` / `PERMUTATION_IMPORTANCE_SAMPLE_SIZE`：Permutation Importance（`n_classes > 50` 时自动 `n_repeats=2` 并采样 500 行）。
    - `DATA_QUALITY_MAX_ROWS`：数据质量评估采样（`n_samples > 100_000` 时自动采样 50_000 行）。
    - 训练策略：`n_classes > 50` 时自动限制 GBM/XGB 树量（GBM `n_estimators<=1000`，XGB `n_estimators<=500`）。

### 7.7 实验（Experiment）与搜索 Agent

- `backend/services/search_agent.py` 实现 LLM 驱动的多候选搜索。
- 通过内部 HTTP 客户端（ASGI Transport）调用 `/api/runs` 复用现有训练逻辑，避免与 routers 循环依赖。
- 涉及 ORM 表：`experiments`、`trials`。
- `experiments` 记录实验状态与最佳 Run；`trials` 记录每次候选运行及其 val/test 分数。

---

## 8. 核心 API 速览

### 数据集

```text
POST   /api/datasets/upload
GET    /api/datasets
GET    /api/datasets/{id}
GET    /api/datasets/{id}/preview
GET    /api/datasets/{id}/quality
GET    /api/datasets/{id}/schema
POST   /api/datasets/{id}/validate
PUT    /api/datasets/{id}
GET    /api/datasets/{id}/cleaning-rules
PUT    /api/datasets/{id}/cleaning-rules
POST   /api/datasets/connect
DELETE /api/datasets/{id}
```

### 训练任务

```text
POST   /api/runs
GET    /api/runs
POST   /api/runs/compare
GET    /api/runs/{id}
GET    /api/runs/{id}/events          # SSE 实时状态推送
GET    /api/runs/{id}/results
GET    /api/runs/{id}/report
GET    /api/runs/{id}/report?download=1
GET    /api/runs/{id}/model
GET    /api/runs/{id}/logs
POST   /api/runs/{id}/predict
POST   /api/runs/{id}/predict/batch
POST   /api/runs/{id}/explain
DELETE /api/runs/{id}
```

### 实验

```text
POST   /api/experiments
GET    /api/experiments
GET    /api/experiments/{id}
GET    /api/experiments/{id}/trials
GET    /api/experiments/{id}/best-run
```

### 意图理解

```text
POST   /api/intent/parse
POST   /api/intent/rules
POST   /api/intent/schema
```

---

## 9. 部署注意事项

- 生产环境建议将 `DATABASE_URL` 切换为 **PostgreSQL**。
- `scripts/run_prod.sh` 直接以 `nohup` 启动前后端，不使用 Docker。
- Windows 生产部署目前没有等价脚本，需要手动启动后端和前端，或自行编写批处理/PowerShell 脚本。
- 前端生产构建输出到 `frontend/dist/`，但当前生产脚本使用 `npm run dev` 直接运行 Vite 开发服务器并代理 API。如需静态部署，请调整脚本。
- 训练任务在后台子进程运行，CPU/内存占用较高，建议在独立机器或容器上部署。

---

## 10. 安全考虑

- **CORS**：`backend/main.py` 当前配置为 `allow_origins=["*"]`，生产环境应收紧为具体域名。
- **文件上传**：默认限制 `max_upload_size_mb=100`，上传目录为 `data/uploads/`。
- **数据库连接**：`db_connection_service.py` 直接使用用户传入的连接参数创建引擎，不要在公网暴露此接口。
- **LLM API 密钥**：通过 `.env` 配置，未配置时自动降级，不会阻塞主流程。
- **SQL 注入**：数据库连接查询使用 SQLAlchemy 文本参数化，不要拼接用户输入 SQL。
- **路径遍历**：文件存储服务使用 `Path.resolve()` 与项目根目录校验，上传文件名应避免直接使用。

---

## 11. 给 AI 代理的实操建议

1. **修改后端代码后**，优先运行 `pytest tests -m "not slow" -v` 验证。
2. **修改 Flow/Task 后**，注意 `PREFECT_API_URL` 必须在 `import prefect` 之前设置。
3. **新增依赖**时，同时更新 `pyproject.toml`、`requirements-core.txt`、`requirements-full.txt`，并保持 `requirements.txt` 与核心依赖一致；注意 `pyproject.toml` 与 `requirements-core.txt` 当前对 torch 的处理不一致。
4. **新增配置**时，优先加入 `backend/config.py` 的 `Settings` 类，并在 `.env.example` 中给出示例。
5. **新增数据库字段**时，同步更新 `backend/models.py`、`backend/schemas.py` 以及前端对应视图；SQLite 开发环境可通过 `database.py` 的轻量迁移自动补齐列。
6. **修改报告模板**时，检查 `backend/templates/report.html` 与 `backend/services/visualization.py` 的 base64 图表输出是否匹配。
7. **任何涉及训练/测试划分、采样、编码的改动**，必须确保遵循“fit on train, transform on test”原则，避免数据泄露。
8. **修改 requirements 文件后**，注意 `Makefile` 与 README 中的安装说明是否仍准确。
9. **Windows 开发**：优先使用 `scripts/run_dev.ps1`；不存在等价的 `.bat` 脚本。
10. **新增 Router**：不要忘记在 `backend/main.py` 中 `app.include_router(...)` 注册。

---

## 12. 参考文件索引

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | 项目元数据、依赖、Black/Ruff/mypy/pytest 配置 |
| `Makefile` | 常用命令封装 |
| `backend/config.py` | 应用配置与 `.env` 加载 |
| `backend/main.py` | FastAPI 入口 |
| `backend/database.py` | 异步数据库引擎与 SQLite 轻量迁移 |
| `backend/models.py` | ORM 模型 |
| `backend/schemas.py` | Pydantic 模型 |
| `backend/prefect_flows/automl_flow.py` | Prefect Flow 主流程 |
| `backend/services/training_executor.py` | 异步训练执行器 |
| `backend/services/search_agent.py` | LLM 多候选搜索 Agent |
| `frontend/vite.config.js` | Vite 配置与 API 代理 |
| `.env.example` | 环境变量示例 |
