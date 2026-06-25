# Prefect AutoML Platform

> 端到端全自动机器学习平台。核心目标：**用户上传数据文件（CSV / Excel / Parquet / JSONL）并选择目标列，系统在算力与时间预算内自动完成数据清洗 → 特征工程 → 策略路由 → 模型搜索 → 评估 → 预测 → 报告输出**。
>
> 开发排期与实现状态见 [`todo.md`](./todo.md)，AI 编码代理上下文见 [`AGENTS.md`](./AGENTS.md)。

---

## 1. 为什么要做这个项目？

市面上的 AutoML 工具（AutoGluon、H2O、FLAML 等）已经很强，但它们通常是**库/命令行**，不是**面向业务人员的闭环系统**。本项目想做一个可交互、可观测、可复现的全栈平台：

- 非算法工程师也能上传数据、选目标列、拿到可解释报告；
- 每个训练决策（选什么 preset、做不做采样、用几折验证）都被记录，而不是黑盒；
- 训练过程可失败隔离、断点续跑，方便集成到 MLOps 工作流；
- 业务解读可由 LLM 生成，未配置 API Key 时自动降级到规则模板。

## 2. 为什么要用 Prefect？

AutoGluon 负责“训练模型”，Prefect 负责**把数据驱动的决策显式编排成可观测、可复现、可失败隔离的流程**：

| 能力 | 在本项目中的具体体现 |
|------|----------------------|
| **原子步骤编排** | 训练被拆分为 13 个原子步骤：`ingest → analyze → quality → strategy → split → cross_validate → fit_preprocessor → transform → sample → train → evaluate → interpret → report`。 |
| **动态 DAG** | 根据元数据决定下一步：小样本走 CV，大数据走 Holdout，不平衡才做采样，高维才降维。 |
| **任务级重试/缓存** | 数据加载按文件 hash 缓存；预处理、训练等步骤支持失败重跑。 |
| **失败隔离** | LLM 推断失败、Embedding 下载失败、单模型 OOM，都不会让整个流程崩溃。 |
| **可观测性** | Prefect UI 展示每个 Task 的输入输出、耗时、状态与策略决策日志。 |
| **异步/并发控制** | 训练任务通过 `TrainingExecutor` 在独立子进程中运行，信号量控制最大并发数。 |

## 3. 技术栈

- **编排引擎**：Prefect 3.x
- **模型内核**：AutoGluon Tabular
- **后端**：FastAPI + SQLAlchemy 2.x + Pydantic 2.x + openai
- **前端**：Vue 3 + Element Plus + ECharts + Vite
- **数据库**：SQLite（开发）/ PostgreSQL（生产）
- **可解释性**：SHAP + Permutation Importance
- **LLM 接入**：OpenAI 兼容端点，支持 KIMI / DeepSeek / MiniMax / 智谱 GLM / OpenAI
- **序列化**：joblib
- **环境管理**：uv（推荐）或 pip

## 4. 系统架构

```text
用户输入
  ├─ 数据文件（CSV / Excel / Parquet / JSONL）
  ├─ 目标列 + 任务类型（或自然语言描述）
  └─ 可选 Schema / 业务规则 / LLM API Key
         │
         ▼
┌─────────────────────────────────────────────┐
│  Pre-flow 理解层（LLM + 规则降级）            │
│  目标：把业务描述 → TargetConfig + Schema    │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  原子步骤执行层（StepRunner）                 │
│  ingest → analyze → quality → strategy      │
│     → split → cross_validate                │
│     → fit_preprocessor → transform          │
│     → sample → train → evaluate             │
│     → interpret → report                    │
└─────────────────────────────────────────────┘
         │
         ▼
输出产物：metrics.json / leaderboard.csv / feature_importance.csv /
          permutation_importance.csv / cv_results / business_interpretation.json /
          shap_values.joblib / report.html / model.zip
```

## 5. 当前已实现功能

### 数据与 Schema
- [x] 数据集上传（CSV / Excel / Parquet / JSONL）与元数据提取
- [x] 自动推断编码（UTF-8 / GBK）
- [x] 数据库连接导入（MySQL / PostgreSQL / ClickHouse / SQLite）
- [x] Schema 推断（numeric / categorical / binary / text / datetime / id）
- [x] Schema 校验与对齐
- [x] 目标列选择接口
- [x] 数据集预览
- [x] 数据质量六维报告
- [x] 数据清洗规则可配置

### 训练引擎
- [x] 数据驱动的训练策略路由
- [x] 13 个原子步骤编排与单步执行
- [x] 数据清洗（去重、缺失填充、目标列缺失删除、取值约束）
- [x] 特征工程（log 变换、IQR 截断、数值缩放、类别编码、时间周期特征、文本 Embedding）
- [x] 条件采样（SMOTE / SMOTENC / ADASYN / RandomOver / RandomUnder / SMOTEENN）
- [x] 条件降维（高相关剔除、低方差剔除、PCA）
- [x] 80/20 分层划分 + AutoGluon 模型搜索
- [x] 测试集 + 训练集双指标评估 + 扩展指标
- [x] 显式交叉验证（StratifiedKFold / KFold / TimeSeriesSplit / GroupKFold）
- [x] 二分类最优阈值调优 + 集成回退验证
- [x] 稀有类别处理策略可配置化（`auto` / `drop` / `none`）
- [x] 全局超时控制与 Best-so-far 返回

### 报告与解释
- [x] 特征重要性表 + 模型排行榜
- [x] HTML 报告（策略依据、数据质量、业务解读、指标、排行榜、特征重要性、混淆矩阵、ROC/PR 曲线、SHAP）
- [x] SHAP 汇总图 + 单样本解释
- [x] Permutation Importance
- [x] 跨 Run 模型对比
- [x] LLM 业务解读：训练后默认规则模板，用户主动触发 LLM 生成

### LLM 能力
- [x] 自然语言意图解析（`POST /api/intent/parse`）
- [x] 业务规则提取（`POST /api/intent/rules`）
- [x] LLM 辅助 Schema 推断（`POST /api/intent/schema`）
- [x] 多 LLM 提供商支持：KIMI / DeepSeek / MiniMax / 智谱 GLM / OpenAI
- [x] 运行时 LLM 配置：前端配置 provider/key/model，持久化到数据库
- [x] 数据外传免责声明与主动触发机制

### 部署与运维
- [x] 异步训练执行器 + SSE 实时状态推送
- [x] 单条/批量预测、模型 zip 下载、训练日志查看
- [x] Linux/macOS 开发/生产启动脚本
- [x] Windows PowerShell 开发启动脚本
- [x] SQLite 轻量迁移

## 6. 快速开始

### 环境要求

- **Python 3.12**（`pyproject.toml` 锁定 `>=3.12,<3.13`）
- **Node.js 18+**
- 推荐 **uv** 作为包管理器

### 环境准备

```bash
cd prefect-project
uv venv --python 3.12
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 核心依赖（CPU 版 AutoGluon，已含 openai）
uv pip install -r requirements-core.txt

# 或完整依赖（含完整 AutoGluon 间接引入的 torch/fastai/NN）
uv pip install -r requirements-full.txt

# 或直接从 pyproject.toml 安装（含 torch CPU 版）
uv pip install -e .
```

### 开发启动

```bash
# 后端（端口 8001）
cd backend
uvicorn main:app --app-dir . --reload --host 0.0.0.0 --port 8001

# 前端（端口 8084）
cd frontend
npm install
npm run dev
```

或一键启动：

```bash
# Linux/macOS
bash scripts/run_dev.sh

# Windows PowerShell
.\scripts\run_dev.ps1
```

### 生产部署

```bash
cd frontend && npm install && npm run build && cd ..
bash scripts/run_prod.sh
```

- 后端 API：`http://<服务器IP>:8001`
- 前端页面：`http://<服务器IP>:8084`

停止服务：

```bash
bash scripts/run_prod_stop.sh
```

## 7. 核心 API

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
GET    /api/runs/{id}/steps
POST   /api/runs/{id}/steps/{name}
POST   /api/runs/{id}/continue
GET    /api/runs/{id}/artifacts/{name}
GET    /api/runs/{id}/events
GET    /api/runs/{id}/results
GET    /api/runs/{id}/report
GET    /api/runs/{id}/report?download=1
GET    /api/runs/{id}/model
GET    /api/runs/{id}/logs
POST   /api/runs/{id}/predict
POST   /api/runs/{id}/predict/batch
POST   /api/runs/{id}/explain
POST   /api/runs/{id}/interpretation/regenerate
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

### LLM 配置

```text
GET    /api/settings/llm
POST   /api/settings/llm
GET    /api/settings/llm/providers
```

## 8. 项目结构

```text
prefect-project/
├── backend/
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # 应用配置
│   ├── database.py              # 异步数据库引擎与 SQLite 迁移
│   ├── models.py                # SQLAlchemy ORM
│   ├── schemas.py               # Pydantic 模型
│   ├── routers/                 # API 路由
│   │   ├── datasets.py
│   │   ├── runs.py
│   │   ├── intent.py
│   │   ├── experiments.py
│   │   └── llm_settings.py
│   ├── services/                # 业务服务
│   │   ├── automl.py
│   │   ├── training_strategy.py
│   │   ├── training_executor.py
│   │   ├── step_runner.py
│   │   ├── step_manifest.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── feature_engineering.py
│   │   ├── sampling_service.py
│   │   ├── cv_service.py
│   │   ├── schema_service.py
│   │   ├── data_service.py
│   │   ├── data_quality.py
│   │   ├── llm_client.py
│   │   ├── llm_intent_service.py
│   │   ├── llm_settings_service.py
│   │   ├── llm_strategy_service.py
│   │   ├── report_llm_service.py
│   │   ├── explainability.py
│   │   ├── visualization.py
│   │   ├── db_connection_service.py
│   │   ├── storage.py
│   │   └── seed_data.py
│   ├── prefect_flows/
│   │   └── automl_flow.py
│   └── templates/report.html
├── frontend/                    # Vue 3 前端
│   ├── src/
│   │   ├── api/index.js
│   │   ├── views/
│   │   ├── components/
│   │   ├── router/
│   │   ├── stores/
│   │   ├── App.vue
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── scripts/                     # 运维脚本
├── tests/                       # pytest 测试
├── data/                        # 数据、模型、报告
├── AGENTS.md                    # AI 编码代理上下文
├── todo.md                      # 开发排期
├── pyproject.toml               # 项目元数据与依赖
├── requirements-core.txt
├── requirements-full.txt
├── requirements.txt
└── README.md                    # 本文件
```

## 9. 测试

```bash
# 跳过慢速端到端测试
pytest tests -v -m "not slow"

# 全部测试
pytest tests -v
```

## 10. 注意事项

- **训练任务在后台独立子进程中运行**，默认最大并发数为 2，避免阻塞 FastAPI 主服务。
- **默认使用 SQLite**，生产环境建议切换到 PostgreSQL。
- **训练产物**保存在 `data/reports/{run_id}/` 和 `data/uploads/{dataset_id}/`。
- **Windows 开发**：使用 `scripts/run_dev.ps1`。
- **LLM 功能**：已默认依赖 `openai`；未配置 API 密钥时自动降级到规则引擎，不影响训练主流程。
- **数据外传免责声明**：LLM 业务解读会把训练摘要发送到第三方 LLM 服务，前端已增加风险提示，需用户确认后触发。

## 11. 后续重点

见 [`todo.md`](./todo.md)。短期重点：

1. 前端稀有类别处理策略交互（目标列分布图 + 策略选择器）；
2. 训练报告中记录稀有类别处理策略及受影响类别；
3. 训练日志实时流式展示；
4. Docker Compose 一键启动。
