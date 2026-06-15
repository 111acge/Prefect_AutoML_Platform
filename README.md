# Prefect AutoML Platform

> 端到端全自动机器学习平台。核心目标：**用户只需提供数据和业务目标，系统在算力与时间预算内自动完成数据清洗 → 特征工程 → 策略路由 → 模型搜索 → 评估 → 预测 → 报告输出**。
>
> 完整产品需求见 [`task.md`](./task.md)。开发排期见 [`todo.md`](./todo.md)。

## 1. 为什么要做这个项目？

市面上的 AutoML 工具（AutoGluon、H2O、FLAML 等）已经很强，但它们通常是**库/命令行**，不是**面向业务人员的闭环系统**。本项目想做一个可交互、可观测、可复现的全栈平台：

- 非算法工程师也能上传数据、选目标列、拿到可解释报告；
- 每个训练决策（选什么 preset、做不做采样、用几折验证）都被记录，而不是黑盒；
- 训练过程可失败隔离、断点续跑，方便集成到 MLOps 工作流。

## 2. 为什么要用 Prefect？

AutoGluon 已经能自动搜模型，那还要 Prefect 干什么？

Prefect 不负责“训练模型”本身，它负责**把数据驱动的决策显式编排成可观测、可复现、可失败隔离的流程**：

| 能力 | 在本项目中的具体体现 |
|------|----------------------|
| **动态 DAG** | 根据元数据决定下一步：小样本走 CV，大数据走 Holdout，不平衡才做采样，高维才降维 |
| **任务级重试/缓存** | `load_data_task` 配了 `retries=2`；未来可对数据加载、Embedding 下载等做结果缓存 |
| **失败隔离** | LLM 推断失败、文本 Embedding 下载失败、单个模型 OOM，都不会让整个 Flow 崩溃 |
| **可观测性** | Prefect UI 展示每个 Task 的输入输出、耗时、状态，以及策略决策日志 |
| **异步/并发控制** | 训练任务通过 `TrainingExecutor` 在独立子进程中运行，信号量控制最大并发数 |

简单说：**Prefect 不是“把固定脚本串起来”，而是“让 AutoML 的每一步决策都能被看见、被审计、被复现”**。

## 3. 技术栈

- **编排引擎**：Prefect 2.x
- **模型内核**：AutoGluon Tabular
- **后端**：FastAPI + SQLAlchemy 2.x + Pydantic 2.x
- **前端**：Vue 3 + Element Plus + Vite
- **数据库**：SQLite（开发）/ PostgreSQL（生产）
- **可解释性**：SHAP
- **序列化**：joblib
- **环境管理**：uv

## 4. 系统架构

```text
用户输入
  ├─ 数据文件（CSV / Excel / Parquet）
  ├─ 目标列 + 任务类型（或未来用自然语言描述）
  └─ 可选 Schema / 业务规则
         │
         ▼
┌─────────────────────────────────────────────┐
│  Pre-flow 理解层（未来：LLM + 规则降级）      │
│  目标：把业务描述 → TargetConfig + Schema    │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Prefect 编排层（automl_pipeline）            │
│  load_data → validate_schema → analyze_meta │
│     → build_strategy → clean / engineer     │
│     → split → model_search → evaluate       │
│     → report                                │
└─────────────────────────────────────────────┘
         │
         ▼
输出产物：metrics.json / leaderboard.csv / report.html / 模型 zip
```

## 5. 当前已实现功能

- [x] 数据集上传（CSV / Excel / Parquet）与元数据提取
- [x] Schema 推断、校验、目标列选择
- [x] **数据驱动的训练策略路由**：根据样本量、特征数、类别不平衡度、字段类型自动选择 preset、验证方式、stacking、采样权重
- [x] 数据清洗（去重、缺失填充、目标列缺失删除）
- [x] 特征工程（右偏数值 log 变换、时间特征提取）
- [x] 80/20 分层划分 + AutoGluon 模型搜索
- [x] 测试集 + 训练集双指标评估
- [x] HTML 报告（含策略依据、指标、排行榜、特征重要性、混淆矩阵）
- [x] 模型 zip 下载、在线预测、训练日志查看
- [x] 异步训练执行器 + 并发控制
- [x] 前后端分离部署（端口 8001 / 8084）

## 6. 快速开始

### 环境准备

```bash
cd prefect-project
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

> `requirements.txt` 默认安装 AutoGluon Tabular CPU 子集（LightGBM / XGBoost / CatBoost）。需要神经网络模型请安装 `requirements-full.txt`。

### 开发启动

```bash
# 后端
uvicorn main:app --app-dir backend --reload --port 8001

# 前端
cd frontend
npm install
npm run dev -- --port 8084
```

或一键启动：

```bash
bash scripts/run_dev.sh
```

### 生产服务器部署

```bash
cd frontend && npm run build && cd ..
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

```text
POST   /api/datasets/upload        # 上传数据集
GET    /api/datasets               # 列出数据集
GET    /api/datasets/{id}          # 数据集详情
GET    /api/datasets/{id}/schema   # 数据集 Schema
POST   /api/datasets/{id}/validate # 校验数据
DELETE /api/datasets/{id}          # 删除数据集

POST   /api/runs                   # 启动训练任务
GET    /api/runs                   # 列出训练任务
GET    /api/runs/{id}              # 任务详情
GET    /api/runs/{id}/results      # 训练结果（含 train/test 指标）
GET    /api/runs/{id}/report       # HTML 报告（inline 预览）
GET    /api/runs/{id}/report?download=1  # 下载报告
GET    /api/runs/{id}/model        # 下载模型 zip
GET    /api/runs/{id}/logs         # 训练日志
POST   /api/runs/{id}/predict      # 模型预测
DELETE /api/runs/{id}              # 删除任务
```

## 8. 数据驱动的策略路由示例

以 iris（150 样本，5 特征）为例，不指定 preset 时系统会决策：

```text
数据规模: small (150 样本, 5 特征)
Preset: good_quality
时间限制: 30s
最大模型数: 5（用户指定）
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
│   ├── routers/                 # API 路由
│   ├── services/                # 业务服务
│   │   ├── automl.py            # AutoGluon 封装
│   │   ├── training_strategy.py # 数据驱动的策略路由
│   │   ├── training_executor.py # 异步训练执行器
│   │   ├── preprocessing_pipeline.py # 可序列化预处理器
│   │   ├── schema_service.py    # Schema 推断/校验/对齐
│   │   └── ...
│   ├── prefect_flows/
│   │   └── automl_flow.py       # Prefect Flow 定义
│   ├── models.py                # SQLAlchemy ORM
│   ├── database.py              # 数据库连接
│   └── templates/report.html    # HTML 报告模板
├── frontend/                    # Vue 3 前端
├── scripts/                     # 运维脚本
├── tests/                       # 测试
├── data/                        # 上传数据、模型、报告
├── task.md                      # 技术 PRD
├── todo.md                      # 开发 Todo
└── README.md                    # 本文件
```

## 10. 测试

```bash
pytest tests -v -m "not slow"
```

## 11. Python 版本

当前支持 **Python 3.12**。未来将跟踪 AutoGluon 对 Python 3.14 / 3.15 的支持进度。

## 12. 后续重点

见 [`todo.md`](./todo.md)。短期重点：

1. 数据库连接上传（MySQL / PostgreSQL / ClickHouse / SQLite）
2. LLM 意图理解层（自然语言 → TargetConfig），失败降级到规则引擎
3. 条件采样（SMOTE / ADASYN）与条件降维（PCA / 高相关剔除）
4. 更完整的特征工程（按基数编码、文本 Embedding、交叉特征）
5. 多种 CV 策略（时间序列、分组、嵌套 CV）
6. Prefect Artifact 与 UI 可视化

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

## 14. 注意事项

- 训练任务在后台独立子进程中运行，默认最大并发数为 2，避免阻塞 FastAPI 主服务。
- 默认使用 SQLite，生产环境建议切换到 PostgreSQL。
- 训练产物保存在 `data/reports/{run_id}/` 和 `data/uploads/{dataset_id}/`。
