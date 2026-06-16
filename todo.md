# prefect-project 开发 Todo 清单

> 基于 [`task.md`](./task.md) 和 Obsidian 知识库「02-技术工程 / 数据科学范式」系列笔记拆解。
> 优先级：P0（必须）> P1（重要）> P2（增强）> P3（未来）。
>
> 更新日期：2026-06-15

---

## 设计原则（做任何改动前先读一遍）

1. **Prefect 不是脚本搬运工**。每个 Task 必须是可被审计、可被重试、可被缓存的决策节点，而不是把固定代码包一层。
2. **先看数据，再决定策略**。preset、验证方式、采样、降维、特征工程都必须根据元数据动态选择。
3. **失败隔离 + 降级**。LLM 失败、Embedding 下载失败、单模型 OOM，都不能让整条 Flow 崩溃。
4. **防止泄露**。所有依赖目标的变换（采样、Target Encoding、特征选择）必须在训练集内 fit，验证/测试集只 transform。
5. **多指标一起看**。不只看 Accuracy，必须结合业务目标报告 F1、MCC、AUC-PR、RMSE 等。

---

## 阶段 0：项目基础（已稳定运行）

- [x] P0 项目目录结构（backend / frontend / tests / scripts / data）
- [x] P0 Python 3.12 虚拟环境 + `requirements.txt`
- [x] P0 `pyproject.toml`（项目元数据、Python 版本约束 `>=3.12,<3.13`）
- [x] P0 FastAPI + SQLAlchemy 2.x + SQLite 骨架
- [x] P0 Vue 3 + Element Plus + Vite 前端骨架
- [x] P0 `/health` 接口 + CORS + 日志
- [x] P0 `.gitignore` + Git 初始化
- [x] P1 编写 `scripts/check_env.py` 环境检查脚本
- [x] P1 统一 `Makefile` / `taskipy` 快捷命令（lint / test / build / start）
- [ ] P2 GitHub Actions CI（ruff / black / pytest / build）

---

## 阶段 1：数据接入与 Schema 管理

参考 Obsidian：`数据清洗与数据质量大全`、`数据科学范式 MOC`

- [x] P0 文件上传：CSV / Excel / Parquet
- [x] P0 自动推断编码（UTF-8 / GBK）
- [x] P0 数据集元数据提取（行数、列数、内存、缺失率、字段类型）
- [x] P0 Schema 推断（numeric / categorical / binary / text / datetime / id）
- [x] P0 Schema 校验与对齐
- [x] P0 目标列选择接口
- [x] P1 **数据库连接上传**：MySQL / PostgreSQL / ClickHouse / SQLite（通过 SQLAlchemy + 自定义 SQL）
- [x] P1 数据集预览接口（前 N 行 + 统计摘要）
- [x] P1 数据质量六维报告（完整性、一致性、准确性、时效性、唯一性、有效性）
- [x] P1 数据清洗规则可配置（业务规则提取入口）
- [ ] P2 LLM 辅助 Schema 推断（失败降级到规则引擎）

---

## 阶段 1.5：LLM 意图理解层（Pre-flow）

参考 Obsidian：`数据科学范式 MOC`

- [x] P0 Intent Parser：自然语言 → TargetConfig（目标列、任务类型、评估指标、时间预算、preset、max_models）
- [x] P0 多 LLM 提供商支持：KIMI / DeepSeek / MiniMax（OpenAI 兼容端点）
- [x] P0 失败降级：API 超时 / 未配置 / 返回异常 → 自动切换规则引擎，Flow 不中断
- [x] P0 REST API：`POST /api/intent/parse`，可结合 dataset_id 用样例辅助推断
- [x] P1 Business Rule Extractor：从描述中提取隐含清洗规则（如"年龄不能>120"）
- [x] P2 LLM 辅助 Schema 推断（无 Schema 时推断字段语义）

---

## 阶段 2：Prefect 编排层（核心引擎）

参考 Obsidian：`AutoML与神经架构搜索大全`、`交叉验证与模型评估策略大全`

- [x] P0 Prefect Flow `automl-end-to-end` 主流程
- [x] P0 核心 Task：load_data / validate_schema / analyze_metadata / build_strategy / clean / engineer / split / train / evaluate / report
- [x] P0 任务状态持久化到数据库 + 异步回调更新
- [x] P0 异步训练执行器 + 信号量并发控制（默认最大 2 个并发）
- [x] P1 **Prefect Artifact**：在 UI 中直接查看 leaderboard、特征重要性、数据质量报告
- [x] P1 **Prefect State/Cache**：对数据加载、Embedding 生成等耗时步骤启用结果缓存
- [ ] P1 全局超时控制：超过 `time_budget_minutes` 后返回当前最优结果（Best-so-far）
- [ ] P2 断点续跑：崩溃后根据已完成的 Task 状态恢复
- [ ] P2 多任务队列：支持排队、优先级、资源预留

---

## 阶段 3：数据驱动的策略路由

参考 Obsidian：`AutoML与神经架构搜索大全`、`正则化与防止过拟合技术大全`

- [x] P0 根据数据规模（small / medium / large）自动选择 preset
- [x] P0 根据类别不平衡度自动启用 `balanced sample_weight`
- [x] P0 根据样本量自动选择 CV / Holdout
- [x] P0 根据字段类型自动启用时间特征提取
- [x] P0 根据数值分布自动做 log 变换
- [x] P1 **模型搜索空间动态裁剪**：小数据加入 KNN/SVM/NN，大数据仅 LightGBM/XGBoost/线性模型
- [ ] P1 **超参优化策略选择**：默认 AutoGluon，高级场景接入 Optuna / Ray Tune
- [ ] P1 **正则化强度自适应**：小数据集自动增强正则化（早停更激进、更简模型空间）
- [ ] P2 支持用户自定义 PipelineTemplate（YAML/JSON 配置）

---

## 阶段 4：数据清洗与特征工程

参考 Obsidian：`数据清洗与数据质量大全`、`特征工程方法大全`

- [x] P0 缺失值处理：数值中位数填充、类别众数填充
- [x] P0 重复行删除、目标列缺失行删除
- [x] P0 右偏数值列 log1p 变换
- [x] P0 时间特征提取（year / month / day / dayofweek）
- [x] P1 **缺失值策略按缺失率分级**：<5% 删行、5-30% 填充、>50% 丢弃或缺失指示列
- [x] P1 **异常值处理**：IQR / Z-score / 不处理可选；默认按偏度选择策略
- [x] P1 **数值缩放策略**：auto 根据分布自动选 RobustScaler / StandardScaler / MinMaxScaler
- [x] P1 **类别编码按基数选择**：低基数 One-Hot、高基数 Target Encoding（必须 CV 内 fit）
- [x] P1 **高基数低频项合并**为 "__other__"
- [x] P1 **文本 Embedding**：sentence-transformers 本地提取（失败则跳过；DataPreprocessor 默认关闭，避免内存问题）
- [ ] P2 **交叉特征**：基于树模型重要性反馈自动尝试二阶交叉
- [ ] P2 **特征选择**：Filter / L1 / Permutation Importance / SHAP
- [x] P1 **时间序列特征**：year / month / day / dayofweek / hour + sin/cos 周期编码（可开关）

---

## 阶段 5：采样、降维与不平衡处理

参考 Obsidian：`采样与数据平衡策略大全`、`正则化与防止过拟合技术大全`

- [x] P0 类别不平衡检测 + `sample_weight`
- [x] P1 **条件过采样**：SMOTE（纯数值） / SMOTENC（混合类型）
- [x] P1 **条件欠采样**：Random Under-sampling（大数据场景）
- [x] P1 **组合采样**：SMOTEENN / SMOTETomek
- [x] P1 **采样必须在训练集划分后执行**，验证集保持原始分布
- [x] P1 **阈值调优**：根据业务错误成本自动调整分类阈值
- [ ] P2 **条件降维**：
  - 高相关性剔除（|r| > 0.95）
  - 低方差剔除
  - PCA（保留 95% 方差，仅数值特征）
- [ ] P2 降维器必须在 CV 内 fit，防止泄露

---

## 阶段 6：模型搜索、评估与集成

参考 Obsidian：`AutoML与神经架构搜索大全`、`交叉验证与模型评估策略大全`、`机器学习评估指标大全`

- [x] P0 AutoGluon Tabular 封装
- [x] P0 支持分类 / 回归任务
- [x] P0 时间预算、presets、seed 控制
- [x] P0 测试集 + 训练集双指标评估
- [x] P0 扩展指标：Precision / Recall / F1 / MCC / Kappa / Confusion Matrix / ROC / PR / MAE / RMSE / R² / MAPE / SMAPE
- [x] P1 **多种 CV 策略自动选择**：
  - 标准表格：StratifiedKFold / KFold（通过 AutoGluon bagging 实现）
  - 时间序列：TimeSeriesSplit（预留接口）
  - 分组数据：GroupKFold（预留接口）
- [x] P1 **CV 内完成所有预处理**，确保无数据泄露
- [ ] P1 **嵌套 CV**：内层调参、外层评估，避免 optimistic bias
- [x] P1 **主指标自动选择**：不平衡 → AUC-PR / MCC；平衡 → F1；回归 → RMSE / MAE
- [x] P1 **自动集成验证**：检查集成是否带来 ≥2% 提升，否则回退单最优模型
- [ ] P2 AutoGluon 模型空间自定义（hyperparameters 配置）
- [ ] P2 神经网络模型支持（安装 `requirements-full.txt`）

---

## 阶段 7：可解释性与报告

参考 Obsidian：`模型解释性与可解释AI大全`、`数据可视化大全`

- [x] P0 特征重要性表 + 排行榜
- [x] P0 HTML 报告（指标、混淆矩阵、策略依据）
- [x] P0 SHAP 汇总图
- [x] P1 **TreeSHAP 全局 + 单样本解释**（全局 SHAP 汇总图 + `POST /api/runs/{id}/explain` 单样本接口）
- [x] P1 **Permutation Importance**（用于泄露检测和无偏重要性）
- [x] P1 数据质量摘要嵌入报告
- [x] P1 混淆矩阵、ROC/PR 曲线、回归残差图
- [ ] P2 **反事实解释**（针对拒绝/负面决策）
- [ ] P2 LLM 报告增强：业务语言解读 + 异常诊断

---

## 阶段 8：LLM 意图理解层（Pre-flow）

参考 Obsidian：`AI应用开发工程师大全/Agent系统开发模式大全`

- [ ] P1 **Intent Parser**：自然语言业务描述 → TargetConfig（目标列、任务类型、主指标建议）
- [ ] P1 **Schema Inference（LLM 辅助）**：推断字段语义、角色、约束
- [ ] P1 **Business Rule Extractor**：提取隐含规则（如 "年龄不能 > 120"）
- [ ] P1 **失败降级**：LLM 超时/错误时切换规则引擎，Flow 继续
- [ ] P2 支持 OpenAI / Claude / 本地 vLLM 多后端
- [ ] P2 Schema 低置信度时暂停并提示用户确认

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
- [x] P1 **批量预测**：上传 CSV 返回 predictions.csv
- [x] P1 **配置快照（config_snapshot）**：保存数据 hash、Schema、超参、模型版本，保证可复现
- [ ] P1 **模型版本管理**：每次训练生成版本记录
- [ ] P2 **ONNX 导出**
- [ ] P2 产物过期清理策略

---

## 阶段 10：前端与交互

- [x] P0 数据集上传页面
- [x] P0 数据集列表/删除
- [x] P0 Schema 预览与目标列选择
- [x] P0 训练任务创建表单
- [x] P0 训练任务列表 + 状态轮询
- [x] P0 训练任务详情（指标、排行榜、特征重要性、报告预览）
- [x] P0 预测页面
- [x] P1 Preset 增加「自动选择」选项
- [ ] P1 训练日志实时流式展示
- [ ] P1 数据质量报告可视化
- [ ] P1 混淆矩阵 / ROC / PR 曲线交互图
- [ ] P2 与 tech-portfolio 统一视觉风格
- [ ] P2 自然语言创建任务（对接 LLM 意图层）

---

## 阶段 11：MLOps、部署与监控

参考 Obsidian：`模型部署与MLOps大全`

- [x] P0 服务器直接部署脚本 `scripts/run_prod.sh`
- [x] P0 前后端分离端口配置
- [ ] P1 PostgreSQL 生产切换 + Alembic 迁移
- [ ] P1 模型服务化：REST API 批量预测
- [ ] P2 **Feature Store 抽象**：保证在线/离线特征一致
- [ ] P2 **监控**：
  - 推理延迟 P50/P95/P99
  - 数据漂移（KS / Chi² / Wasserstein）
  - 概念漂移（性能衰减）
- [ ] P2 部署门控：shadow → canary → A/B → 全量，支持一键回滚
- [ ] P3 Docker Compose 一键启动（可选，当前优先服务器直接部署）

---

## 阶段 12：测试与质量保障

- [x] P1 API 集成测试（数据集上传、创建任务、预测）
- [x] P1 Schema 服务单元测试
- [x] P1 训练策略单元测试
- [x] P1 数据清洗 + 特征工程单元测试
- [ ] P1 失败重试与降级测试
- [ ] P1 端到端训练测试（样本数据）
- [ ] P2 测试覆盖率目标 ≥ 70%
- [ ] P2 mypy 类型检查

---

## 阶段 13：Python 3.14 / 3.15 适配跟踪

- [ ] P3 跟踪 AutoGluon / PyTorch / LightGBM / XGBoost / CatBoost 的 Python 3.14 支持进度
- [ ] P3 建立 Python 版本兼容性测试矩阵
- [ ] P3 `pyproject.toml` 中写入依赖版本上限，防止自动安装不兼容版本
- [ ] P3 在 README 标注当前支持的 Python 版本

---

## 近期推荐执行顺序（接下来 2 周）

1. **P1 数据库连接上传**：扩展 `load_data_task`，支持 SQL 数据源。
2. **P1 多种 CV 策略**：根据数据特性选择 StratifiedKFold / TimeSeriesSplit / GroupKFold。
3. **P1 条件采样 + 阈值调优**：不平衡场景启用 SMOTE / class weights / 阈值优化。
4. **P1 更完整的特征工程**：按基数编码、文本 Embedding、异常值处理。
5. **P1 LLM 意图层 MVP**：自然语言 → TargetConfig，失败降级。
6. **P1 Prefect Artifact**：把 leaderboard、特征重要性、数据质量报告挂到 Prefect UI。

---

## 验收标准

- [x] P0 用户可上传 CSV，选择目标列，启动训练
- [x] P0 训练完成后可查看指标、排行榜、报告
- [x] P0 可下载模型并进行预测
- [ ] P1 训练任务失败时前端展示明确错误信息
- [ ] P1 Prefect UI 可查看完整 DAG 执行图
- [ ] P2 项目可通过 Docker Compose 一键启动
- [ ] P3 具备向 Python 3.14/3.15 迁移的跟踪机制
