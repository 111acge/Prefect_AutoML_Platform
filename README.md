# Prefect AutoML Platform

> 端到端全自动机器学习平台。核心目标：**用户上传数据文件并选择目标列，系统在算力与时间预算内自动完成数据清洗 → 特征工程 → 策略路由 → 模型搜索 → 评估 → 预测 → 报告输出**。
>
> 开发排期与实现状态见 [`todo.md`](./todo.md)，AI 编码代理上下文见 [`AGENTS.md`](./AGENTS.md)。

---

## 1. 为什么要做这个项目？

市面上的 AutoML 工具（AutoGluon、H2O、FLAML 等）已经很强，但它们通常是**库/命令行**，不是**面向业务人员的闭环系统**。本项目想做一个可交互、可观测、可复现的全栈平台：

- 非算法工程师也能上传数据、选目标列、拿到可解释报告；
- 每个训练决策（选什么 preset、做不做采样、用几折验证）都被记录，而不是黑盒；
- 训练过程可失败隔离、断点续跑，方便集成到 MLOps 工作流；
- 业务解读可由 LLM 生成，也可在未配置 API Key 时自动降级到规则模板。

## 2. 为什么要用 Prefect？

AutoGluon 已经能自动搜模型，那还要 Prefect 干什么？

Prefect 不负责“训练模型”本身，它负责**把数据驱动的决策显式编排成可观测、可复现、可失败隔离的流程**：

| 能力 | 在本项目中的具体体现 |
|------|----------------------|
| **原子步骤编排** | 训练被拆分为 13 个原子步骤（ingest → analyze → quality → strategy → split → cross_validate → fit_preprocessor → transform → sample → train → evaluate → interpret → report），每步可独立执行、重试、观测。 |
| **动态 DAG** | 根据元数据决定下一步：小样本走 CV，大数据走 Holdout，不平衡才做采样，高维才降维。 |
| **任务级重试/缓存** | 数据加载按文件 hash 缓存；预处理、训练等步骤支持失败重跑。 |
| **失败隔离** | LLM 推断失败、文本 Embedding 下载失败、单个模型 OOM，都不会让整个流程崩溃。 |
| **可观测性** | Prefect UI 展示每个 Task 的输入输出、耗时、状态，以及策略决策日志。 |
| **异步/并发控制** | 训练任务通过 `TrainingExecutor` 在独立子进程中运行，信号量控制最大并发数。 |

简单说：**Prefect 不是“把固定脚本串起来”，而是“让 AutoML 的每一步决策都能被看见、被审计、被复现”**。

## 3. 技术栈

- **编排引擎**：Prefect 3.x
- **模型内核**：AutoGluon Tabular
- **后端**：FastAPI + SQLAlchemy 2.x + Pydantic 2.x
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

- [x] 数据集上传（CSV / Excel / Parquet / JSONL）与元数据提取
- [x] 数据库连接导入（MySQL / PostgreSQL / ClickHouse / SQLite）
- [x] Schema 推断、校验、目标列选择
- [x] **数据驱动的训练策略路由**：根据样本量、特征数、类别不平衡度、字段类型自动选择 preset、验证方式、stacking、采样权重
- [x] 数据清洗（去重、缺失填充、目标列缺失删除、取值约束）
- [x] 特征工程（右偏数值 log 变换、异常值 IQR 截断、数值缩放、类别编码、时间周期特征、文本 Embedding）
- [x] 条件采样（SMOTE / SMOTENC / ADASYN / RandomOver / RandomUnder / SMOTEENN）
- [x] 条件降维（高相关剔除、低方差剔除、PCA）
- [x] 80/20 分层划分 + AutoGluon 模型搜索
- [x] 测试集 + 训练集双指标评估 + 扩展指标
- [x] 显式交叉验证（StratifiedKFold / KFold / TimeSeriesSplit / GroupKFold）
- [x] 二分类最优阈值调优 + 集成回退验证
- [x] HTML 报告（含策略依据、数据质量、业务解读、指标、排行榜、特征重要性、混淆矩阵、ROC/PR 曲线、SHAP）
- [x] 模型 zip 下载、在线预测、批量预测、训练日志查看、单样本解释
- [x] LLM 意图解析与 Schema 推断（KIMI / DeepSeek / MiniMax / GLM / OpenAI 兼容，失败降级）
- [x] **LLM API Key 前端配置**：支持四 provider 动态切换、持久化到数据库
- [x] **LLM 业务解读**：训练后默认生成规则模板，用户主动触发 LLM 生成并展示数据外传免责声明
- [x] 异步训练执行器 + 并发控制
- [x] 前后端分离部署（端口 8001 / 8084）

## 6. 快速开始

### 环境要求

- **Python 3.12**（当前 `pyproject.toml` 锁定 `>=3.12,<3.13`）
- **Node.js 18+**
- 推荐 **uv** 作为包管理器

### 环境准备

```bash
cd prefect-project
uv venv --python 3.12
source .venv/bin/activate

# 核心依赖（CPU 版 AutoGluon，已含 openai）
uv pip install -r requirements-core.txt

# 或完整依赖（含 torch / fastai / NN / LLM API）
uv pip install -r requirements-full.txt

# 或直接从 pyproject.toml 安装（含 torch CPU 版）
uv pip install -e .
```

### 开发启动

```bash
# 后端
cd backend
uvicorn main:app --app-dir . --reload --host 0.0.0.0 --port 8001

# 前端
cd frontend
npm install
npm run dev
```

或一键启动（Linux/macOS）：

```bash
bash scripts/run_dev.sh
```

Windows 用户请使用 PowerShell：

```powershell
.\scripts\run_dev.ps1
```

### 生产服务器部署

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

## 7. 核心 API

### 数据集

```text
POST   /api/datasets/upload        # 上传数据集
GET    /api/datasets               # 列出数据集
GET    /api/datasets/{id}          # 数据集详情
GET    /api/datasets/{id}/preview  # 数据集预览
GET    /api/datasets/{id}/quality  # 数据质量六维报告
GET    /api/datasets/{id}/schema   # 数据集 Schema
POST   /api/datasets/{id}/validate # 校验数据
PUT    /api/datasets/{id}          # 设置目标列和任务类型
GET    /api/datasets/{id}/cleaning-rules  # 获取清洗规则
PUT    /api/datasets/{id}/cleaning-rules  # 更新清洗规则
POST   /api/datasets/connect       # 通过数据库连接导入
DELETE /api/datasets/{id}          # 删除数据集
```

### 训练任务

```text
POST   /api/runs                   # 启动训练任务
GET    /api/runs                   # 列出训练任务
GET    /api/runs/{id}              # 任务详情
GET    /api/runs/{id}/results      # 训练结果（含 train/test 指标、扩展指标）
GET    /api/runs/{id}/steps        # 原子步骤状态
POST   /api/runs/{id}/steps/{name} # 执行单个原子步骤
POST   /api/runs/{id}/continue     # 继续下一个 pending/failed 步骤
POST   /api/runs/{id}/interpretation/regenerate  # 重新生成 LLM 业务解读
GET    /api/runs/{id}/report       # HTML 报告（inline 预览）
GET    /api/runs/{id}/report?download=1  # 下载报告
GET    /api/runs/{id}/model        # 下载模型 zip
GET    /api/runs/{id}/logs         # 训练日志
POST   /api/runs/{id}/predict      # 单条预测
POST   /api/runs/{id}/predict/batch # 批量预测（上传 CSV）
POST   /api/runs/{id}/explain      # 单样本 SHAP 解释
DELETE /api/runs/{id}              # 删除任务
```

### LLM 配置

```text
GET    /api/settings/llm           # 获取当前 LLM 配置（API Key 已掩码）
POST   /api/settings/llm           # 保存 LLM provider / API Key / model
GET    /api/settings/llm/providers # 列出支持的 LLM 提供商及默认配置
```

### 意图理解

```text
POST   /api/intent/parse           # 自然语言 → TargetConfig
POST   /api/intent/rules           # 自然语言 → 清洗规则
POST   /api/intent/schema          # LLM 辅助 Schema 推断
```

## 8. 数据驱动的策略路由示例

以 iris（150 样本，5 特征）为例，不指定 preset 时系统会决策：

```text
数据规模: small (150 样本, 5 特征)
Preset: good_quality
时间限制: 60s
最大模型数: 10
Stacking: 关（小数据集防止过拟合）
主评估指标: log_loss
验证策略: cv
```

决策逻辑见 `backend/services/training_strategy.py`。

## 9. 项目结构

```text
prefect-project/
├── backend/
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # 应用配置
│   ├── database.py              # SQLAlchemy 异步引擎与会话
│   ├── models.py                # SQLAlchemy ORM
│   ├── schemas.py               # Pydantic 模型
│   ├── routers/                 # API 路由
│   │   ├── datasets.py
│   │   ├── runs.py
│   │   ├── intent.py
│   │   ├── experiments.py
│   │   └── llm_settings.py
│   ├── services/                # 业务服务
│   │   ├── automl.py            # AutoGluon 封装
│   │   ├── training_strategy.py # 数据驱动的策略路由
│   │   ├── training_executor.py # 异步训练执行器
│   │   ├── step_runner.py       # 原子步骤执行器
│   │   ├── step_manifest.py     # 步骤产物管理
│   │   ├── preprocessing_pipeline.py # 可序列化预处理器
│   │   ├── feature_engineering.py    # 特征工程
│   │   ├── sampling_service.py       # 条件采样
│   │   ├── schema_service.py         # Schema 推断/校验/对齐
│   │   ├── data_service.py           # 数据加载与元分析
│   │   ├── data_quality.py           # 六维数据质量
│   │   ├── llm_client.py             # 统一 LLM 调用客户端
│   │   ├── llm_intent_service.py     # LLM 意图解析（带降级）
│   │   ├── llm_settings_service.py   # LLM provider/key/model 配置管理
│   │   ├── report_llm_service.py     # LLM 业务解读
│   │   ├── explainability.py         # SHAP / Permutation Importance
│   │   ├── visualization.py          # 报告图表
│   │   ├── db_connection_service.py  # 数据库连接
│   │   ├── storage.py                # 文件存储
│   │   └── seed_data.py              # 默认 iris 数据集
│   ├── prefect_flows/
│   │   └── automl_flow.py       # Prefect Flow 定义（保留兼容）
│   └── templates/report.html    # HTML 报告模板
├── frontend/                    # Vue 3 前端
├── scripts/                     # 运维脚本
│   ├── run_dev.sh
│   ├── run_dev.ps1
│   ├── run_prod.sh
│   ├── run_prod_stop.sh
│   ├── run_flow.py              # 独立运行 Flow 的入口
│   ├── run_step.py              # 单步/全量原子步骤入口
│   └── check_env.py             # 环境检查
├── tests/                       # 测试
├── data/                        # 上传数据、模型、报告、默认数据集
├── todo.md                      # 开发 Todo
├── AGENTS.md                    # AI 编码代理上下文
├── pyproject.toml               # 项目元数据与依赖
├── requirements-core.txt        # 核心依赖
├── requirements-full.txt        # 完整依赖
└── README.md                    # 本文件
```

## 10. 测试

```bash
# 跳过慢速端到端测试
pytest tests -v -m "not slow"

# 全部测试
pytest tests -v
```

## 11. 注意事项

- **训练任务在后台独立子进程中运行**，默认最大并发数为 2，避免阻塞 FastAPI 主服务。
- **默认使用 SQLite**，生产环境建议切换到 PostgreSQL。
- **训练产物**保存在 `data/reports/{run_id}/` 和 `data/uploads/{dataset_id}/`。
- **Windows 用户**：请使用 `scripts/run_dev.ps1` 启动开发环境。
- **LLM 功能为可选但已默认依赖 openai**：未配置 API 密钥时，所有 LLM 调用会自动降级到规则引擎，不影响训练主流程。配置 Key 后可在页面上主动触发 LLM 业务解读。
- **数据外传免责声明**：LLM 业务解读会把训练摘要（指标、Top 特征等）发送到第三方 LLM 服务，前端已增加风险提示，需用户确认后触发。

## 12. 后续重点

见 [`todo.md`](./todo.md)。短期重点：

1. 前端稀有类别处理策略交互（目标列分布图 + 策略选择器）；
2. 训练报告中记录稀有类别处理策略及受影响类别；
3. 模型版本管理与产物过期清理策略；
4. Docker Compose 一键启动。

## 13. 知识资产

本项目的设计参考了 Obsidian 知识库中「02-技术工程 / 数据科学范式」下的多份笔记，包括但不限于：

- `数据清洗与数据质量大全`
- `特征工程方法大全`
- `采样与数据平衡策略大全`
- `交叉验证与模型评估策略大全`
- `机器学习评估指标大全`
- `AutoML与神经架构搜索大全`
- `模型解释性与可解释AI大全`
- `模型部署与MLOps大全`
- `正则化与防止过拟合技术大全`

这些笔记是未来功能扩展时的需求来源。
