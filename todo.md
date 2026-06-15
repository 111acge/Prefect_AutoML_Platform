# prefect-project 开发 Todo 清单

> 基于 `task.md` 拆解的完整任务清单。
> 优先级：P0（必须）> P1（重要）> P2（增强）> P3（未来适配）

---

## 阶段 0：项目初始化与环境搭建

### 0.1 仓库结构
- [x] P0 初始化 `backend/` FastAPI 项目骨架
- [x] P0 初始化 `frontend/` Vue 3 项目骨架
- [x] P0 创建 `data/` 数据目录（uploads / models / reports）
- [x] P0 创建 `tests/` 测试目录
- [x] P0 创建 `scripts/` 运维脚本目录
- [x] P0 创建 `docs/` 文档目录

### 0.2 Python 环境
- [x] P0 使用 `uv venv --python 3.12` 创建虚拟环境
- [x] P0 编写 `requirements.txt`（核心依赖：Prefect + AutoGluon + FastAPI + SQLAlchemy）
- [ ] P0 编写 `requirements-dev.txt`（pytest / black / ruff / mypy）
- [x] P0 编写 `pyproject.toml`（项目元数据、Python 版本约束 `>=3.12,<3.13`）
- [ ] P1 编写环境检查脚本 `scripts/check_env.py`
- [ ] P1 配置国内 PyPI 镜像或私有镜像加速安装

### 0.3 前端环境
- [x] P0 使用 Vite + Vue 3 + Element Plus 初始化前端
- [x] P0 配置前端代理到后端 `http://localhost:8001`
- [x] P0 配置 Vue Router 和 Pinia
- [ ] P1 配置 ESLint / Prettier

### 0.4 开发工具
- [x] P1 配置 `.gitignore`
- [ ] P1 配置 `Makefile` 或 `taskipy` 快捷命令
- [ ] P2 配置 GitHub Actions CI（lint / test / build）

---

## 阶段 1：后端核心基础设施

### 1.1 数据库与 ORM
- [x] P0 配置 SQLAlchemy 2.x + SQLite（开发）
- [x] P0 设计数据库模型 `backend/models.py`：
  - `Dataset` 数据集表
  - `Run` 训练任务表
  - `Metric` 评估指标表
  - `Prediction` 预测记录表
- [x] P0 实现数据库初始化脚本 `backend/database.py`
- [ ] P1 支持 PostgreSQL 切换（生产环境）
- [ ] P1 实现数据库迁移（Alembic）

### 1.2 文件存储
- [x] P0 实现文件上传/下载服务 `backend/services/storage.py`
- [x] P0 设计目录规范：
  - `./data/uploads/{dataset_id}/`
  - `./data/models/{run_id}/`
  - `./data/reports/{run_id}/`
- [ ] P1 实现文件大小限制和类型校验
- [ ] P1 实现过期数据清理脚本

### 1.3 FastAPI 应用骨架
- [x] P0 编写 `backend/main.py`
- [x] P0 配置 CORS、异常处理、日志
- [x] P0 实现 `/health` 健康检查接口
- [ ] P1 配置 OpenAPI 文档和标签

---

## 阶段 2：数据接入与 Schema 管理

### 2.1 数据集上传 API
- [x] P0 实现 `POST /api/datasets/upload`
- [x] P0 支持 CSV / Excel / Parquet 文件上传
- [x] P0 自动推断文件编码（UTF-8 / GBK）
- [ ] P1 支持数据库连接上传（MySQL / PostgreSQL / ClickHouse / SQLite）
- [ ] P1 实现数据集元数据提取（行数、列数、内存占用）

### 2.2 数据集管理 API
- [x] P0 实现 `GET /api/datasets` 列出数据集
- [x] P0 实现 `GET /api/datasets/{dataset_id}` 查看详情
- [x] P0 实现 `DELETE /api/datasets/{dataset_id}` 删除数据集
- [x] P1 实现数据集预览接口（前 N 行、统计摘要）

### 2.3 Schema 推断与校验
- [x] P0 实现规则引擎推断字段类型（数值/类别/时间/文本/ID）
- [ ] P0 实现 Schema 校验与对齐逻辑
- [ ] P1 集成 LLM 辅助 Schema 推断（可选，失败降级）
- [ ] P1 实现目标列选择接口

---

## 阶段 3：Prefect 编排层

### 3.1 Prefect 基础配置
- [x] P0 安装并配置 Prefect
- [x] P0 设计 Flow：`automl_pipeline`
- [x] P0 设计 12 个核心 Task：
  1. `load_data`
  2. `validate_and_align_schema`
  3. `analyze_metadata`
  4. `select_strategy`
  5. `clean_data`
  6. `engineer_features`
  7. `conditional_sampling`
  8. `conditional_dimensionality_reduction`
  9. `model_search`
  10. `evaluate`
  11. `auto_ensemble`
  12. `export_model`

### 3.2 状态管理
- [x] P0 实现任务状态持久化到数据库
- [x] P0 实现 Flow 运行状态回调更新
- [ ] P1 利用 Prefect 内置 State/Artifact 记录关键产物
- [ ] P1 实现失败重试和超时控制

### 3.3 策略路由
- [x] P0 实现 `select_strategy` 规则引擎
- [x] P0 定义标准分类/回归模板
- [ ] P1 定义不平衡分类/大规模分类/高维回归模板
- [ ] P2 支持自定义 PipelineTemplate 配置

---

## 阶段 4：AutoGluon 模型内核

### 4.1 AutoGluon 封装
- [x] P0 实现 `backend/services/automl.py`
- [x] P0 封装 `TabularPredictor.fit()` 调用
- [x] P0 支持分类/回归任务
- [x] P0 支持时间预算配置
- [ ] P1 支持自定义模型空间
- [x] P1 支持 `presets` 选择（medium_quality / best_quality）

### 4.2 数据清洗与特征工程
- [x] P0 实现缺失值处理
- [x] P0 实现异常值处理
- [x] P0 实现类别编码和数值变换
- [ ] P1 实现时间特征提取
- [ ] P1 实现文本 Embedding 特征（sentence-transformers）
- [ ] P1 实现条件采样（SMOTE / ADASYN）
- [ ] P1 实现条件降维（PCA / 高相关剔除）

### 4.3 评估与报告
- [x] P0 实现 CV 评估指标计算
- [x] P0 生成 `leaderboard.csv` 和 `metrics.json`
- [x] P0 生成 SHAP 可解释性图表
- [x] P1 生成 HTML 报告
- [x] P1 生成特征重要性图

### 4.4 模型导出与预测
- [x] P0 保存训练好的模型到 `./data/models/{run_id}/`
- [x] P0 实现预测接口 `POST /api/runs/{run_id}/predict`
- [ ] P1 支持 ONNX 导出
- [ ] P1 支持批量预测

---

## 阶段 5：训练任务 API

### 5.1 启动训练
- [x] P0 实现 `POST /api/runs` 启动 AutoML 训练
- [x] P0 返回 `run_id` 和初始状态
- [x] P0 异步执行 Prefect Flow
- [x] P1 实现参数校验（Pydantic）

### 5.2 查询任务
- [x] P0 实现 `GET /api/runs` 列出任务
- [x] P0 实现 `GET /api/runs/{run_id}` 查询状态
- [x] P0 实现 `GET /api/runs/{run_id}/results` 获取结果
- [ ] P1 实现任务日志流式返回

### 5.3 报告与产物
- [ ] P0 实现 `GET /api/runs/{run_id}/report` 下载报告
- [ ] P0 实现 `GET /api/runs/{run_id}/model` 下载模型
- [x] P0 实现 `DELETE /api/runs/{run_id}` 删除任务及产物

---

## 阶段 6：LLM 意图理解层（可选）

### 6.1 自然语言解析
- [ ] P1 实现 `Intent Parser`：业务描述 → TargetConfig
- [ ] P1 实现 `Schema Inference`：LLM 辅助字段推断
- [ ] P1 实现失败降级到规则引擎
- [ ] P2 支持 OpenAI / Claude / 本地 vLLM

### 6.2 报告增强
- [ ] P2 使用 LLM 生成业务语言解读
- [ ] P2 使用 LLM 解释特征重要性

---

## 阶段 7：前端开发

### 7.1 页面路由
- [x] P0 首页 / 数据集管理
- [x] P0 训练任务列表
- [x] P0 训练任务详情 / 结果展示
- [x] P1 模型预测页面

### 7.2 数据集管理页
- [x] P0 上传文件组件
- [x] P0 数据集列表展示
- [x] P1 Schema 预览与目标列选择

### 7.3 训练任务页
- [x] P0 创建训练任务表单
- [x] P0 任务状态轮询展示
- [x] P0 训练结果展示（指标、排行榜）
- [x] P1 训练日志实时展示
- [x] P1 报告下载按钮

### 7.4 结果可视化
- [x] P1 特征重要性图
- [x] P1 SHAP 汇总图
- [x] P1 混淆矩阵 / 回归散点图
- [ ] P2 PCA 可视化

### 7.5 与 tech-portfolio 集成
- [x] P1 在 tech-portfolio 前端增加"AutoML 平台"外链
- [ ] P1 统一视觉风格（可选）

---

## 阶段 8：测试与质量保障

### 8.1 单元测试
- [x] P1 测试数据加载函数
- [x] P1 测试 Schema 校验函数
- [x] P1 测试策略路由函数
- [x] P1 测试文件存储服务

### 8.2 集成测试
- [x] P1 测试完整 Flow 在样本数据上跑通
- [x] P1 测试 API 端到端流程
- [ ] P2 测试失败重试和降级

### 8.3 代码质量
- [x] P1 配置 black 格式化
- [x] P1 配置 ruff 静态检查
- [x] P2 配置 mypy 类型检查

---

## 阶段 9：部署与运维

### 9.1 本地开发部署
- [x] P0 编写启动脚本 `scripts/run_dev.sh`
- [x] P0 后端端口 8001，前端端口 8084
- [x] P1 编写 `README.md` 快速开始

### 9.2 生产部署
- [x] P2 编写 Dockerfile
- [x] P2 编写 `docker-compose.yml`
- [x] P2 配置 Nginx 反向代理
- [x] P2 配置 PostgreSQL 生产数据库

### 9.3 监控与日志
- [x] P2 配置结构化日志
- [x] P2 配置 Prefect UI 访问
- [x] P2 配置磁盘使用监控和告警

---

## 阶段 10：未来 Python 3.14 / 3.15 适配计划

### 10.1 版本适配策略
- [ ] P3 建立 Python 版本兼容性测试矩阵（3.12 / 3.13 / 3.14 / 3.15）
- [ ] P3 使用 `tox` 或 `nox` 多版本测试
- [ ] P3 跟踪 AutoGluon 官方 Python 3.14 支持进度
- [ ] P3 跟踪 PyTorch / LightGBM / XGBoost / CatBoost 的 3.14 wheel 发布

### 10.2 代码层面适配
- [ ] P3 避免使用 Python 3.12+ 专有语法（除非 3.14 已确认支持）
- [ ] P3 使用 `sys.version_info` 做版本检查，对不支持的 Python 版本给出友好提示
- [ ] P3 将所有依赖版本上限写入 `pyproject.toml`，防止未来升级时自动安装不兼容版本
- [ ] P3 定期运行 `pip install --dry-run` 检查依赖可解析性

### 10.3 自动化跟踪
- [ ] P3 每月检查一次核心依赖的 Python 3.14/3.15 兼容性声明
- [ ] P3 在 README 中标注当前支持的 Python 版本
- [ ] P3 设置 GitHub issue 或 reminder，跟踪 AutoGluon 3.14 支持 milestone

### 10.4 迁移计划（当 AutoGluon 支持 3.14 后）
- [ ] P3 更新 `pyproject.toml` 中 `requires-python` 为 `>=3.12,<3.15`
- [ ] P3 在 Python 3.14 环境中跑通完整测试集
- [ ] P3 更新 CI 矩阵加入 3.14
- [ ] P3 更新文档和 Docker 基础镜像

### 10.5 风险预案
- [ ] P3 如果 AutoGluon 长期不支持 3.14/3.15，评估是否迁移模型内核
- [ ] P3 持续关注 FLAML / H2O / PyCaret 对高版本 Python 的支持情况
- [ ] P3 保留核心数据清洗和特征工程代码与 AutoGluon 解耦，便于未来替换

---

## 验收标准

- [ ] P0 用户可以通过前端上传 CSV，选择目标列，启动训练
- [ ] P0 训练完成后可以查看指标、排行榜、报告
- [ ] P0 可以下载训练好的模型并进行预测
- [ ] P1 训练任务失败时前端能展示错误信息
- [ ] P1 Prefect UI 可以看到完整的 DAG 执行图
- [ ] P2 项目可以通过 Docker Compose 一键启动
- [ ] P3 具备向 Python 3.14/3.15 迁移的能力和跟踪机制

---

## 当前推荐执行顺序

1. 完成阶段 0：环境 + 目录 + 基础依赖
2. 完成阶段 1.1-1.3：FastAPI + SQLite 骨架
3. 完成阶段 2.1-2.2：数据集上传/列表 API
4. 完成阶段 3.1：Prefect Flow + 12 个 Task 空壳
5. 完成阶段 4.1：AutoGluon 最小封装
6. 完成阶段 5：训练任务 API
7. 完成阶段 7 前端 MVP
8. 跑通端到端 demo
9. 逐步补充 6/8/9/10

---

## 检查中发现的问题（待修复）

> 2026-06-15 项目检查记录，按优先级排序。

### 🔴 高优先级

- [ ] **metrics.json 格式不一致**
  - 位置：`backend/prefect_flows/automl_flow.py` 中 `evaluate_model_task`
  - 问题：当前把 `leaderboard.iloc[0].to_dict()` 保存为 `metrics["final"]`，但 `backend/routers/runs.py` 期望 `final` 是 `{指标名: 数值}` 的映射
  - 影响：前端训练结果的"评估指标"区域会显示模型名、训练时间等非指标字段
  - 修复：保存标准指标字典，如 `predictor.evaluate()` 的返回值

- [ ] **下载报告 API 未实现**
  - 位置：前端 `RunDetailView.vue` 调用 `/api/runs/${runId}/report`
  - 问题：后端 `backend/routers/runs.py` 没有对应的 `GET /{run_id}/report` 路由
  - 影响：任务详情页"下载报告"按钮点击后 404
  - 修复：实现报告下载接口，或先隐藏前端按钮

- [ ] **requirements-full.txt 与核心依赖不一致**
  - 问题：
    - `requirements-core.txt` 约束 `fastapi>=0.111.0,<0.115.0`，`requirements-full.txt` 写 `fastapi>=0.110.0`
    - `requirements-core.txt` 安装 `autogluon.tabular[lightgbm,catboost,xgboost]`，`requirements-full.txt` 写 `autogluon`
  - 影响：安装完整依赖时可能出现版本冲突或安装失败
  - 修复：统一 `requirements-full.txt` 与 `pyproject.toml` 的依赖约束

- [x] **代码质量工具未安装**
  - 问题：`pyproject.toml` 声明了 `black`、`ruff`、`mypy` 作为 dev 依赖，但当前虚拟环境未安装
  - 影响：无法运行格式化和静态检查
  - 修复：已安装并通过检查；`ruff check backend/ tests/ scripts/` 与 `black --check backend/ tests/ scripts/` 均通过

### 🟡 中优先级

- [x] **训练任务执行架构优化**
  - 位置：`backend/routers/runs.py` 的 `_execute_flow`
  - 问题：使用 `BackgroundTasks` + 嵌套 `asyncio.run` 调用异步数据库更新，能用但不是最佳实践
  - 影响：长时间训练可能阻塞 FastAPI worker 或导致资源泄漏
  - 修复：新增 `backend/services/training_executor.py`，使用 `asyncio.create_subprocess_exec` + 信号量控制并发，彻底避免 `asyncio.run` 嵌套；最大并发训练任务数默认为 2

- [ ] **`.env` 加载路径依赖运行目录**
  - 位置：`backend/config.py` 中 `env_file=".env"`
  - 问题：从 `backend/` 子目录启动时找不到项目根目录的 `.env`
  - 修复：使用基于 `project_root` 的绝对路径加载 `.env`

- [ ] **测试覆盖率过低**
  - 问题：当前只有 `/health` 和 `/` 两个基础测试
  - 影响：数据集上传、训练流程、预测等核心路径无测试保障
  - 修复：补充数据集上传、创建训练任务、预测等集成测试

- [x] **Dockerfile 未包含前端**
  - 问题：`Dockerfile` 只复制了后端代码，没有前端构建产物和静态文件服务
  - 决策：用户要求直接跑在服务器上，不使用 Docker，因此直接删除 `Dockerfile`

### 🟢 低优先级

- [x] **项目已初始化 Git**
  - 问题：目录下没有 `.git` 仓库
  - 影响：无法追踪版本、协作和回滚
  - 修复：`git init` 完成，已配置 `.gitignore`

- [x] **README/task.md 中 Docker 相关描述**
  - 问题：用户要求直接跑在服务器上，但文档中仍保留 Dockerfile 和 Docker Compose 相关章节
  - 修复：`Dockerfile` 已删除；`README.md` 增加"服务器直接部署"章节，使用 `scripts/run_prod.sh`

- [x] **前端构建产物体积过大**
  - 问题：`vite build` 提示主 chunk 超过 500KB
  - 影响：首次加载较慢
  - 修复：在 `frontend/vite.config.js` 中配置 `manualChunks`，拆分为 `vendor-vue`、`vendor-ui`、`vendor-viz` 三个 chunk，主 index chunk 降至约 78KB

---

## 已修复问题

> 2026-06-15 后续开发中已解决的问题。

### ✅ 已修复

- [x] **metrics.json 格式不一致**
  - 修复：在 `automl_flow.py` 中划分训练集/测试集，使用 `predictor.evaluate(test_data)` 生成标准 `{metric: value}` 字典
  - 文件：`backend/prefect_flows/automl_flow.py`

- [x] **下载报告 API 未实现**
  - 修复：新增 `GET /api/runs/{run_id}/report`，返回 `report.html`
  - 文件：`backend/routers/runs.py`

- [x] **requirements-full.txt 与核心依赖不一致**
  - 修复：重写 `requirements-full.txt`，统一与 `pyproject.toml` 的依赖约束

- [x] **`.env` 加载路径依赖运行目录**
  - 修复：`backend/config.py` 中使用 `project_root / ".env"` 绝对路径

- [x] **测试覆盖率过低**
  - 修复：补充数据集上传、创建训练任务、预测、端到端训练等测试
  - 文件：`tests/test_api.py`

- [x] **前端端口仅绑定 localhost**
  - 修复：`frontend/vite.config.js` 中添加 `host: '0.0.0.0'`，支持外部访问

- [x] **训练日志不可查看**
  - 修复：训练子进程日志写入 `output_dir/training.log`，新增 `GET /api/runs/{run_id}/logs`
  - 文件：`backend/routers/runs.py`

### ⏳ 待处理

当前检查中发现的问题已全部处理完毕。后续可按主 Todo 清单继续补充 P1/P2 增强功能。
