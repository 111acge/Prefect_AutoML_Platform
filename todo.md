# prefect-project 开发 Todo 清单

> 基于 [`task.md`](./task.md) 和代码实际实现状态持续更新。
> 优先级：P0（必须）> P1（重要）> P2（增强）> P3（未来）。
>
> 更新日期：2026-06-22（加入稀有类别处理策略可配置化）

---

## 设计原则（做任何改动前先读一遍）

1. **Prefect 不是脚本搬运工**。每个 Task 必须是可被审计、可被重试、可被缓存的决策节点。
2. **先看数据，再决定策略**。preset、验证方式、采样、降维、特征工程都必须根据元数据动态选择。
3. **失败隔离 + 降级**。LLM 失败、Embedding 下载失败、单模型 OOM，都不能让整条 Flow 崩溃。
4. **防止泄露**。所有依赖目标的变换（采样、Target Encoding、特征选择）必须在训练集内 fit，验证/测试集只 transform。
5. **多指标一起看**。不只看 Accuracy，必须结合业务目标报告 F1、MCC、AUC-PR、RMSE 等。

---

## 阶段 0：项目基础

- [x] P0 项目目录结构（backend / frontend / tests / scripts / data）
- [x] P0 Python 3.12 虚拟环境 + `pyproject.toml`
- [x] P0 FastAPI + SQLAlchemy 2.x + SQLite 骨架
- [x] P0 Vue 3 + Element Plus + Vite 前端骨架
- [x] P0 `/health` 接口 + CORS + 日志
- [x] P0 `.gitignore` + Git 初始化
- [x] P1 编写 `scripts/check_env.py` 环境检查脚本
- [x] P1 统一 `Makefile` 快捷命令（lint / test / start / dev）
- [x] **P0 修复 `requirements.txt`：当前只写了 `requirements-core.txt`，已替换为实际依赖内容**
- [ ] P1 GitHub Actions CI（ruff / black / pytest / build）
- [ ] P2 Windows 启动脚本（`.bat` 或 PowerShell）与 `run_dev.sh`/`run_prod.sh` 等价

---

## 阶段 1：数据接入与 Schema 管理

- [x] P0 文件上传：CSV / Excel / Parquet
- [x] P0 自动推断编码（UTF-8 / GBK）
- [x] P0 数据集元数据提取（行数、列数、内存、缺失率、字段类型）
- [x] P0 Schema 推断（numeric / categorical / binary / text / datetime / id）
- [x] P0 Schema 校验与对齐
- [x] P0 目标列选择接口
- [x] P1 数据库连接上传：MySQL / PostgreSQL / ClickHouse / SQLite
- [x] P1 数据集预览接口
- [x] P1 数据质量六维报告
- [x] P1 数据清洗规则可配置
- [x] P2 LLM 辅助 Schema 推断（失败降级到规则引擎）
- [ ] P2 用户上传 Schema（JSON/YAML）并与自动推断 Schema 合并
- [x] P2 支持 Parquet / JSON Lines 格式上传

---

## 阶段 2：LLM 意图理解层（Pre-flow）

- [x] P0 Intent Parser：自然语言 → TargetConfig
- [x] P0 多 LLM 提供商支持：KIMI / DeepSeek / MiniMax / OpenAI 兼容端点
- [x] P0 失败降级：API 超时 / 未配置 / 返回异常 → 规则引擎
- [x] P0 REST API：`POST /api/intent/parse`
- [x] P1 Business Rule Extractor：从描述中提取隐含清洗规则
- [x] P2 LLM 辅助 Schema 推断
- [ ] P2 前端自然语言创建任务（对接 LLM 意图层）
- [ ] P2 Schema 低置信度时暂停并提示用户确认

---

## 阶段 3：Prefect 编排层（核心引擎）

- [x] P0 Prefect Flow `automl-end-to-end` 主流程
- [x] P0 核心 Task：load_data / validate_schema / analyze_metadata / build_strategy / split / train / evaluate / report
- [x] P0 任务状态持久化到数据库 + 异步回调更新
- [x] P0 异步训练执行器 + 信号量并发控制（默认最大 2 个并发）
- [x] P1 Prefect Artifact：leaderboard、特征重要性、数据质量报告
- [x] P1 Prefect State/Cache：对数据加载启用结果缓存
- [x] **P1 把预处理 Pipeline 的 fit / transform / save 包成 Prefect Task，让 DAG 完整可观测**
- [x] **P1 全局超时控制：超过 `time_budget_minutes` 后返回当前最优结果（Best-so-far）**
- [ ] P2 断点续跑：崩溃后根据已完成的 Task 状态恢复
- [ ] P2 多任务队列：支持排队、优先级、资源预留
- [x] P1 训练状态 WebSocket / Server-Sent Events 实时推送（替代 5 秒轮询）
- [x] P1 子进程日志过滤：抑制 Prefect 内部 `EventsWorker` 等噪音，保留业务日志
- [x] P1 显式 CV 结果用于 AutoGluon 验证策略（透传 `num_bag_folds` / `holdout_frac`）

---

## 阶段 4：数据驱动的策略路由

- [x] P0 根据数据规模（small / medium / large）自动选择 preset
- [x] P0 根据类别不平衡度自动启用 balanced sample_weight
- [x] P0 根据样本量自动选择 CV / Holdout
- [x] P0 根据字段类型自动启用时间特征提取
- [x] P0 根据数值分布自动做 log 变换
- [x] P1 模型搜索空间动态裁剪：小数据加入 KNN/LR/NN，大数据仅 LightGBM/XGBoost/CatBoost/LR
- [x] **P1 显式 CV 策略：标准表格用 StratifiedKFold / KFold，时间序列用 TimeSeriesSplit，分组数据用 GroupKFold**
- [ ] P1 超参优化策略选择：默认 AutoGluon，高级场景接入 Optuna / Ray Tune
- [ ] P1 正则化强度自适应：小数据集自动增强正则化
- [ ] P2 支持用户自定义 PipelineTemplate（YAML/JSON 配置）

---

## 阶段 5：数据清洗与特征工程

- [x] P0 缺失值处理：数值中位数填充、类别众数填充
- [x] P0 重复行删除、目标列缺失行删除
- [x] P0 右偏数值列 log1p 变换
- [x] P0 时间特征提取
- [x] P1 缺失值策略按缺失率分级
- [x] P1 异常值处理：IQR 截断
- [x] P1 数值缩放策略：auto 选择 RobustScaler / StandardScaler / MinMaxScaler
- [x] P1 类别编码按基数选择：低基数 One-Hot、高基数 Target Encoding
- [x] P1 高基数低频项合并为 "__other__"
- [x] P1 文本 Embedding：sentence-transformers 本地提取（失败降级）
- [x] P1 时间序列周期编码
- [ ] P2 交叉特征：基于树模型重要性反馈自动尝试二阶交叉
- [ ] P2 特征选择：Filter / L1 / Permutation Importance / SHAP
- [ ] P2 Target Encoding 严格在 CV 每折内 fit（当前在训练集整体 fit，需评估是否改为折内）
- [x] P2 前端特征工程开关：PCA、Target Encoding、文本 Embedding 等可配置
- [ ] P2 自动交叉特征：基于树模型重要性反馈尝试二阶交叉

---

## 阶段 6：采样、降维与不平衡处理

- [x] P0 类别不平衡检测 + sample_weight
- [x] P1 条件过采样：SMOTE / SMOTENC / ADASYN
- [x] P1 条件欠采样：Random Under-sampling
- [x] P1 组合采样：SMOTEENN
- [x] P1 采样在训练集划分后执行
- [x] P1 二分类阈值调优
- [x] **P1 稀有类别处理策略可配置化：`auto` / `drop` / `none`，并接入训练/实验 API**
- [x] P1 高相关性剔除（|r| > 0.95）
- [x] P1 低方差剔除
- [x] P1 PCA（保留 95% 方差，仅数值特征）
- [ ] P2 降维器在 CV 内 fit，防止泄露
- [ ] P2 高维稀疏场景（>10k 特征）前置特征选择

---

## 阶段 7：模型搜索、评估与集成

- [x] P0 AutoGluon Tabular 封装
- [x] P0 支持分类 / 回归任务
- [x] P0 时间预算、presets、seed 控制
- [x] P0 测试集 + 训练集双指标评估
- [x] P0 扩展指标：Precision / Recall / F1 / MCC / Kappa / Confusion Matrix / ROC / PR / MAE / RMSE / R² / MAPE / SMAPE
- [x] P1 主指标自动选择
- [x] P1 自动集成验证：集成提升 <2% 则回退单最优模型
- [ ] P1 嵌套 CV：内层调参、外层评估
- [x] P2 AutoGluon 模型空间自定义（hyperparameters 配置）
- [x] P2 神经网络模型支持（`torch` 已加入依赖，NeuralNetTorch 已可用）

---

## 阶段 8：可解释性与报告

- [x] P0 特征重要性表 + 排行榜
- [x] P0 HTML 报告
- [x] P0 SHAP 汇总图
- [x] P1 TreeSHAP 全局 + 单样本解释
- [x] P1 Permutation Importance
- [x] P1 数据质量摘要嵌入报告
- [ ] **P2 训练报告中记录稀有类别处理策略及受影响类别**
- [x] P1 混淆矩阵、ROC/PR 曲线、回归残差图
- [x] **P2 LLM 报告增强：业务语言解读 + 异常诊断 + 特征 Top-N 业务含义**
- [ ] P2 反事实解释
- [x] P1 报告可视化：混淆矩阵、ROC/PR 曲线图
- [x] P1 SHAP / Permutation Importance 可视化条形图
- [x] P2 跨 Run 模型对比页面/接口

---

## 阶段 9：训练任务 API 与产物管理

- [x] P0 `POST /api/runs` 启动训练
- [x] P0 `GET /api/runs/{id}` 查询状态
- [x] P0 `GET /api/runs/{id}/results` 获取结果
- [x] P0 `GET /api/runs/{id}/report` 报告预览/下载
- [x] P0 `GET /api/runs/{id}/model` 模型 zip 下载
- [x] P0 `GET /api/runs/{id}/logs` 训练日志
- [x] P0 `POST /api/runs/{id}/predict` 单条/批量预测
- [x] P0 `DELETE /api/runs/{id}` 删除任务
- [x] P1 批量预测：上传 CSV 返回 predictions.csv
- [x] P1 配置快照（config_snapshot）
- [ ] P1 模型版本管理
- [ ] P2 ONNX 导出
- [ ] P2 产物过期清理策略

---

## 阶段 10：前端与交互

- [x] P0 数据集上传页面
- [x] P0 数据集列表/删除
- [x] P0 Schema 预览与目标列选择
- [x] P0 训练任务创建表单
- [x] P0 训练任务列表 + 状态轮询
- [ ] **P0 修复前端文件上传完成后不提示“上传成功”的问题**
- [x] P0 训练任务详情（指标、排行榜、特征重要性、报告预览）
- [x] P0 预测页面
- [x] P1 Preset 增加「自动选择」选项
- [ ] P1 训练日志实时流式展示
- [x] P1 数据质量报告可视化
- [x] P1 混淆矩阵 / ROC / PR 曲线交互图
- [ ] P2 与 tech-portfolio 统一视觉风格
- [ ] P2 自然语言创建任务
- [x] P1 Preset / 时间预算快捷按钮（快速体验、标准、不限制）
- [ ] **P1 训练/实验创建页增加「稀有类别处理策略」选择器 + 目标列分布预览（稀有类别标红）**
- [x] P1 模型排行榜按模型族筛选/排序（树模型 / 线性 / 神经网络）
- [x] P2 任务详情页直接展示业务解读摘要（当前仅在 HTML 报告中）

---

## 阶段 11：MLOps、部署与监控

- [x] P0 服务器直接部署脚本 `scripts/run_prod.sh`
- [x] P0 前后端分离端口配置
- [ ] P1 PostgreSQL 生产切换 + Alembic 迁移
- [ ] P1 模型服务化：REST API 批量预测
- [ ] P2 Feature Store 抽象
- [ ] P2 监控：推理延迟、数据漂移、概念漂移
- [ ] P2 部署门控：shadow → canary → A/B → 全量
- [ ] P3 Docker Compose 一键启动

---

## 阶段 12：测试与质量保障

- [x] P1 API 集成测试
- [x] P1 Schema 服务单元测试
- [x] P1 训练策略单元测试
- [x] P1 数据清洗 + 特征工程单元测试
- [x] P1 端到端训练测试（`test_api.py` 中 `@pytest.mark.slow`）
- [ ] P1 失败重试与降级测试
- [ ] P2 测试覆盖率目标 ≥ 70%
- [ ] P2 mypy 类型检查

---

## 阶段 13：Python 版本适配跟踪

- [ ] P3 跟踪 AutoGluon / PyTorch / LightGBM / XGBoost / CatBoost 的 Python 3.14 支持进度
- [ ] P3 建立 Python 版本兼容性测试矩阵
- [x] P3 `pyproject.toml` 中写入依赖版本上限 `>=3.12,<3.13`
- [x] P3 在 README 标注当前支持的 Python 版本

---

## 近期推荐执行顺序（接下来 1-2 周）

按“让端到端更稳定、DAG 更完整、报告更业务化”排序：

1. ~~**P0 修复 `requirements.txt`**：已替换为实际依赖内容。~~
2. ~~**P1 预处理步骤 Prefect Task 化**：`DataPreprocessor.fit/transform/save` 已包成 Task。~~
3. ~~**P1 显式 CV 策略**：已实现 StratifiedKFold / KFold / TimeSeriesSplit / GroupKFold。~~
4. ~~**P1 全局超时 / Best-so-far**：超时后若存在部分模型则返回当前最优模型。~~
5. ~~**P2 LLM 业务解读报告**：`report.html` 已增加“业务结论”章节。~~
6. ~~**P1 训练失败错误透传**：前端任务详情页已展示 `error.json` 完整错误信息。~~

当前迭代下一步建议（待执行）：

1. **P1 前端稀有类别交互**：创建训练/实验任务时展示目标列分布柱状图，稀有类别标红，并提供策略选择器。
2. **P2 报告可审计**：在 HTML 报告和数据质量摘要中记录实际采用的稀有类别处理策略及受影响类别。

历史已完成（保留备查）：

1. ~~**P1 稀有类别处理策略 API 化**：在 `RunCreate` / `ExperimentCreate` schema 中暴露 `rare_class_strategy`，后端透传到 Flow/StepRunner。~~

1. ~~**P1 训练状态实时推送**：用 WebSocket / SSE 替代前端 5 秒轮询，体验提升最明显。~~
2. ~~**P1 子进程日志降噪**：过滤 Prefect 内部 `EventsWorker` 等噪音，只保留业务日志。~~
3. ~~**P1 报告可视化补全**：把已有的 ROC/PR/混淆矩阵数据画成图表，让报告更直观。~~
4. ~~**P1 前端 Preset / 时间预算快捷按钮**：降低用户决策成本。~~
5. ~~**P2 前端特征工程开关**：让用户能开关 PCA、Target Encoding、文本 Embedding 等。~~
6. ~~**P2 跨 Run 模型对比**：支持选中多个任务对比指标和模型。~~

---

## 验收标准

- [x] P0 用户可上传 CSV，选择目标列，启动训练
- [x] P0 训练完成后可查看指标、排行榜、报告
- [x] P0 可下载模型并进行预测
- [x] P1 训练任务失败时前端展示明确错误信息
- [x] P1 Prefect UI 可查看完整 DAG 执行图（含预处理节点）
- [ ] P2 项目可通过 Docker Compose 一键启动
- [ ] P3 具备向 Python 3.14/3.15 迁移的跟踪机制


---

## 附录：稀有类别处理策略 — 前端修改需求明细

### 目标
在训练任务创建页与实验创建页中，让用户能够直观地看到目标列的类别分布，并自主选择稀有类别处理策略，避免全自动处理带来的不信任感。

### 涉及页面

1. **训练任务创建页**（`/runs/create` 或对应路由）
2. **实验创建页**（`/experiments/create` 或对应路由）
3. **训练报告页 / 数据质量摘要**（训练完成后展示实际采用的策略）

### 新增/调整组件

#### 1. 目标列分布图表 `TargetDistributionChart`

- **触发时机**：用户选择数据集和目标列后自动加载。
- **数据来源**：复用或扩展 `/api/datasets/{id}` 返回的 schema/预览信息，建议新增接口 `/api/datasets/{id}/target_distribution` 返回 `{ class_counts: { "A": 20, "B": 1, ... } }`。
- **图表类型**：柱状图（推荐 ECharts，项目如使用其他图表库则保持一致）。
- **视觉要求**：
  - 正常类别使用主色（如蓝色）。
  - 稀有类别（样本数 < 2，或低于用户设定的阈值）标红，并在柱顶显示具体数量。
  - x 轴为类别标签，y 轴为样本数。
  - 鼠标悬停显示该类别的样本数和占比。
- **附加信息**：图表下方显示总样本数、类别数、最小类样本数。

#### 2. 稀有类别处理策略选择器 `RareClassStrategySelector`

- **位置**：目标列分布图表下方，折叠面板「数据质量与稀有类别处理」内。
- **选项**（单选）：
  - **自动兜底**（`auto`，默认）：对样本数 < 2 的类别复制 1 条，保证 stratify / CV 可用。
    - 提示文案：“训练可正常跑通；稀有样本会同时出现在训练集和测试集，测试指标可能轻微偏高。”
  - **过滤稀有类别**（`drop`）：删除样本数 < N 的类别。
    - 提示文案：“将丢失 N 个类别；若过滤后只剩 1 个类别则无法训练。”
    - 出现阈值输入框，默认 N=2。
  - **保留原始分布**（`none`）：不处理，关闭 stratify。
    - 提示文案：“不修改数据分布；但若存在 < 2 的类别，训练可能因无法分层而失败。”
- **动态影响提示**：根据用户选择的策略，实时显示预计后果，例如：
  - “将自动复制类别 C(1)、D(1)，共增加 2 条样本。”
  - “将过滤掉类别 C、D，剩余 3 个有效类别。”
- **校验**：
  - 若选择 `drop` 后有效类别数 < 2，禁用「开始训练」按钮并提示：“过滤后类别数不足 2，无法进行分类训练。”
  - 若选择 `none` 且存在稀有类别，显示黄色警告但不强制阻止，提示用户可能失败。

#### 3. 表单字段透传

- 在 `RunCreate` / `ExperimentCreate` 请求体中增加字段 `rare_class_strategy: "auto" | "drop" | "none"`。
- 默认值：`auto`。
- 该字段需随其他配置一起保存到 `config_snapshot` 并在报告中可查看。

#### 4. 训练报告展示

- 在「数据质量摘要」或「训练策略」卡片中新增一行：
  - **稀有类别处理策略**：自动兜底
  - **受影响类别**：C（1 条）、D（1 条）
- 若策略为 `none` 且训练失败，失败原因里应能体现（后端已处理）。

### 交互流程

```
用户选择数据集 → 选择目标列
        ↓
前端调用 /api/datasets/{id}/target_distribution
        ↓
展示 TargetDistributionChart，标红稀有类别
        ↓
若存在稀有类别，自动展开 RareClassStrategySelector
        ↓
用户选择策略（默认 auto），前端实时显示影响
        ↓
校验通过，用户点击「开始训练」
        ↓
rare_class_strategy 随请求体发送到后端
```

### 后端依赖（前端无关，但需配合）

- `RunCreate` / `ExperimentCreate` schema 增加 `rare_class_strategy` 字段。
- `routers/runs.py` / `routers/experiments.py` 将该字段透传给 `training_executor`。
- `prefect_flows/automl_flow.py` 的 `split_data_task` 接收并下发到 `train_val_test_split`。
- 训练报告中记录实际采用的策略（后端已实现 `split_data` 处理逻辑，待补充报告字段）。

### 优先级与估算

- **P1**：目标列分布图表 + 策略选择器 + 表单透传（约 1-2 天）。
- **P2**：训练报告展示实际策略 + 受影响类别（约 0.5 天）。
