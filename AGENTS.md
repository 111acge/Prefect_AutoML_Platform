# AGENTS.md

This file provides guidance to Lingma (lingma.aliyun.com) when working with code in this repository.

> **注意**：以下描述基于对代码的实际阅读，反映当前实现状态，而非设计愿景。大部分功能已落地，但存在部分限制（详见各节末尾的“⚠️ 限制”）。

## 常用命令

所有命令均从项目根目录执行。

```bash
# 查看所有可用命令
make help

# 环境检查
make check

# 安装依赖（核心 / 完整）
make install
make install-full

# 开发服务（同时启动 Prefect Server、后端、前端）
make dev

# 生产服务（后台运行）
make start
make stop

# 测试
make test              # 跳过慢速测试（推荐日常开发）
make test-slow         # 运行全部测试（含端到端）
pytest tests/test_api.py -v -m "not slow"  # 单独运行某个测试文件

# 代码质量
make lint              # ruff 静态检查
make format            # black 格式化（行宽 100）

# 清理缓存
make clean
```

### 单独启动服务

```bash
# 仅后端（需先 cd backend）
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# 仅前端（需先 cd frontend）
cd frontend
npm install
npm run dev

# 仅启动 Prefect 编排服务
bash scripts/start_prefect.sh
```

### 服务端口

- 前端：`http://localhost:8084`
- 后端 API：`http://localhost:8001`
- API 文档：`http://localhost:8001/docs`
- Prefect UI：`http://localhost:4200`

## 项目架构

### 技术栈

- **后端**：Python 3.12、FastAPI、SQLAlchemy 2.0（异步 + aiosqlite）、Prefect 3.x、AutoGluon Tabular
- **前端**：Vue 3、Element Plus、Pinia、Vue Router、ECharts 6、Vite
- **测试**：pytest、pytest-asyncio、httpx
- **代码质量**：Black（行宽 100）、Ruff、mypy

### 后端目录结构

```
backend/
├── main.py                 # FastAPI 入口；注册路由、CORS、生命周期管理
├── config.py               # 配置管理（pydantic-settings）；GPU 检测、Prefect URL、性能阈值常量
├── database.py             # 异步 SQLAlchemy 引擎与会话；SQLite 自动迁移（仅添加缺失列）
├── models.py               # ORM 模型：Dataset、Run、Experiment、Trial、Metric、RunStep、Setting
├── schemas.py              # Pydantic 请求/响应校验模型
├── routers/                # API 路由（FastAPI APIRouter）
│   ├── datasets.py         # 数据集上传、列表、详情、删除
│   ├── runs.py             # 训练任务提交、状态、步骤执行、预测、解释、报告下载
│   ├── intent.py           # 自然语言意图解析（LLM / 规则引擎兜底）
│   ├── experiments.py      # 实验对比与多候选搜索
│   └── llm_settings.py     # LLM 提供商与模型配置
├── services/               # 业务服务层（核心逻辑）
│   ├── automl.py           # AutoGluon 封装：训练、预测、加载
│   ├── training_executor.py # 训练任务异步执行器；信号量控制并发；Prefect 优先、本地子进程降级
│   ├── training_strategy.py # 根据数据元数据自动构建训练策略
│   ├── preprocessing_pipeline.py # 预处理 Pipeline（fit/train、transform/val_test/predict）
│   ├── step_runner.py      # 单步执行器（供 step 模式调用）
│   ├── step_manifest.py    # 中间产物路径管理
│   ├── data_service.py     # 数据加载、元数据分析、目标列/任务类型推断
│   ├── data_quality.py     # 六维数据质量评估
│   ├── sampling_service.py # 不平衡样本处理策略
│   ├── cv_service.py       # 显式交叉验证
│   ├── explainability.py   # SHAP / Permutation Importance
│   ├── visualization.py    # 报告图表生成
│   ├── report_llm_service.py # LLM 生成业务解读
│   ├── llm_client.py       # 多 LLM 提供商统一调用（OpenAI SDK）
│   ├── llm_intent_service.py # 意图解析与规则提取
│   ├── llm_settings_service.py # LLM 配置持久化到 Setting 表
│   ├── search_agent.py     # 多候选搜索 Agent
│   ├── schema_service.py   # Schema 推断与校验
│   ├── storage.py          # 文件存储清理
│   └── seed_data.py        # 默认数据集初始化
├── prefect_flows/          # Prefect 工作流定义
│   ├── automl_flow.py      # 端到端 Flow（load → clean → feature → split → train → eval → report）
│   └── serve_flow.py       # Flow 部署与 Runner 服务
└── i18n/                   # 国际化（zh-CN / en）
    ├── core.py             # 翻译函数与 locale 管理
    └── dependencies.py   # FastAPI locale 依赖
```

### 核心架构要点

1. **训练执行双模式**：
   - `TrainingExecutor` 优先尝试通过 Prefect Server 调度 `automl_pipeline` Flow；Prefect 不可用时自动降级为本地子进程（调用 `scripts/run_step.py`）。
   - `step` 模式（单步执行）始终走本地子进程，不走 Prefect。
   - 子进程执行时通过 `PREFECT_API_URL=""` 断开 Prefect 连接，避免 StepRunner 引入 prefect 时尝试连接 Server。
   - ⚠️ 限制：`TrainingExecutor` 的 Prefect 轮询逻辑在 `automl_flow.py` 的 Flow 中通过 `prefect.client` 创建 Flow Run，但 `serve_flow.py` 依赖 `automl_pipeline.serve()` 启动本地 Runner；若 Prefect Server 未启动或 Deployment 未创建，降级逻辑会生效，但首次使用需手动运行 `serve_flow.py` 创建 Deployment。

2. **数据划分与防泄露**：
   - 在预处理 **之前** 划分 train/val/test（`train_val_test_split`），防止数据泄露。
   - 预处理 Pipeline 仅在训练集上 `fit`，再 `transform` 验证集、测试集和预测输入。
   - ⚠️ 限制：`clean_dataframe` 和 `engineer_features` 在 `automl_flow.py` 的 Prefect Flow 中是在划分 **之前** 对全量数据执行的（见 `clean_data_task`、`engineer_features_task`），但在 `step_runner.py` 的 `split` 步骤中，划分是在 `fit_preprocessor` 之前、且原始数据直接保存后划分，没有先执行 `clean_dataframe`/`engineer_features`。两处行为不完全一致——Prefect Flow 会先清洗/特征工程再划分，而 step 模式是先划分再清洗（清洗在 `DataPreprocessor.fit` 中完成）。

3. **配置快照与可复现**：
   - 每次运行创建 `config_snapshot.json`，包含数据集文件 MD5、目标列、任务类型、策略参数等，确保结果可复现。
   - ⚠️ 限制：快照中保存了 `dataset_file_hash`（MD5），但 AutoGluon 训练本身涉及随机性（如 `random_state` 仅覆盖部分模型），完全复现需要相同环境、相同依赖版本和相同 `seed`。

4. **性能优化自动决策**：
   - `config.py` 中定义了高基数分类阈值（50/200）、大数据集行数阈值（10万）等常量。
   - 训练集评估、SHAP、排列重要性均支持根据数据特征自动采样或跳过，也可通过 `.env` 显式覆盖。
   - ⚠️ 限制：`training_strategy.py` 中的策略生成逻辑较为启发式，部分阈值（如 `memory_mb > 1024` 触发 PCA）可能不够精细；`text_embeddings` 默认关闭且未实现自动开启逻辑。

5. **数据库**：
   - 默认使用 SQLite（`sqlite+aiosqlite`），通过 `database.py` 中的 `_migrate_sqlite` 自动补齐 ORM 中新增但表中缺失的列（开发/本地环境）。
   - 生产环境建议使用 Alembic。
   - ⚠️ 限制：`_migrate_sqlite` 仅支持 `ADD COLUMN`，不支持删除列、修改类型、添加约束等变更；模型字段改名后旧列仍会残留。

6. **LLM 增强（可选）**：
   - 核心 AutoML 流程无需 LLM 即可运行；LLM 仅用于意图解析和业务解读增强。
   - 支持 `kimi`、`deepseek`、`minimax`、`glm`、`openai`、`auto` 提供商；未配置时由本地规则引擎兜底。
   - ⚠️ 限制：业务解读（`report_llm_service.py`）在训练流程中默认 `force_rule_based=True`，即不自动调用 LLM，需用户在前端手动触发重新生成；意图解析（`llm_intent_service.py`）在 LLM 不可用时降级到规则引擎，但规则引擎的推断能力有限（关键词匹配 + 启发式）。

7. **其他已知限制**：
   - 无认证/鉴权机制，所有 API 对外开放。
   - CORS 配置为 `allow_origins=["*"]` 且 `allow_credentials=True`，存在安全风险。
   - API 响应中返回服务器绝对路径（`file_path`、`output_dir`）。
   - 文件上传未校验扩展名和大小（`max_upload_size_mb` 仅在配置中定义，上传路由中未实际校验）。
   - 部分依赖（starlette、pydantic-settings）存在已知 CVE，详见 README.md 的 TODO 章节。

### 环境变量

关键变量见 `.env.example`：

- `DATABASE_URL`：默认 `sqlite+aiosqlite://./data/db.sqlite`
- `PREFECT_ENABLED` / `PREFECT_API_URL`：控制 Prefect 编排是否启用
- `LLM_PROVIDER` / `*_API_KEY`：LLM 提供商与 API Key
- `USE_GPU` / `NUM_GPUS` / `CUDA_VISIBLE_DEVICES`：GPU 控制
- `TRAIN_EVAL_SAMPLE_SIZE` / `SHAP_MAX_SAMPLE_SIZE` / `PERMUTATION_IMPORTANCE_MAX_REPEATS`：性能调优

### 代码风格

- Black 行宽 100，`target-version = py312`
- Ruff 用于静态检查
- mypy 用于类型检查
- 提交前建议运行 `make format && make lint && make test`

### 安全声明

本项目定位为**本地或可信内网自用工具**，不提供公共服务。当前版本无认证/鉴权、CORS 宽松（`allow_origins=["*"]`）、API 返回绝对路径、异常信息直接返回。在公网或多人共享环境中使用时，需自行增加反向代理、身份认证、CORS 限制与请求限流。
