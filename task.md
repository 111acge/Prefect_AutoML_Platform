# 全自动机器学习平台开发任务描述

**版本**：v1.0  
**编排引擎**：Prefect  
**模型内核**：AutoGluon  
**文档用途**：开发团队技术 PRD（产品需求文档）

---

## 1. 项目目标

构建一个**端到端全自动机器学习平台**。用户仅需提供：

- 数据文件（CSV / Excel / Parquet）或数据库连接信息
- 业务目标描述（自然语言，如"预测客户下月是否流失"）或结构化目标定义
- 可选：Schema 定义（JSON/YAML），若缺失则由系统自动推断

系统在**无人干预**的情况下完成：数据加载 → Schema 推断 → 数据清洗 → 特征工程 → 采样/降维 → 模型搜索 → 超参优化 → 交叉验证 → 自动集成 → 评估 → 预测 → 报告生成，最终输出最优模型及可解释性报告。

**核心约束**：全程不强制依赖 LLM API 完成训练，但 LLM 用于意图理解与报告增强；算力与时间预算内追求全局最优。

---

## 2. 预装依赖

### 2.1 基础环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| Python | 3.12 | 已存在环境，AutoGluon 与 PyTorch 兼容 |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) / macOS / WSL2 | 生产环境建议 Linux |
|---|---|---|
| 内存 | 最低 8GB，推荐 32GB+ | 大数据集与模型搜索内存消耗大 |
|---|---|---|
| 磁盘 | 20GB+ 可用空间 | 预训练模型、缓存、中间产物 |
|---|---|---|
| GPU | 可选 | 树模型无需 GPU；神经网络/文本 Embedding 可用 GPU 加速 |
|---|---|---|

### 2.2 核心依赖（必须安装）

以下依赖为系统运行的**最小可用集**：

```txt
# --- 编排引擎 ---
prefect>=2.19.0          # Prefect 2.x 异步 Flow 编排

# --- 数据内核 ---
pandas>=2.0.0            # 数据处理
numpy>=1.24.0            # 数值计算
pyarrow>=14.0.0          # Parquet 支持，pandas 后端加速
openpyxl>=3.1.0          # Excel 读写
xlrd>=2.0.0              # 旧版 Excel 兼容

# --- 数据库连接 ---
sqlalchemy>=2.0.0        # 统一数据库接入
pymysql>=1.1.0           # MySQL 驱动
psycopg2-binary>=2.9.0   # PostgreSQL 驱动
clickhouse-connect>=0.7.0 # ClickHouse 驱动

# --- AutoML 内核 ---
autogluon>=1.1.0         # 模型搜索 + 自动集成 + 超参优化
# 实际安装时可用轻量子集：autogluon.tabular[lightgbm,catboost,xgboost]
scikit-learn>=1.3.0      # 数据清洗、特征工程、评估指标、Pipeline 封装
feature-engine>=1.6.0    # 特征工程扩展（自动变换、编码）

# --- 采样与降维 ---
imbalanced-learn>=0.11.0 # SMOTE / ADASYN / 组合采样

# --- 可解释性 ---
shap>=0.44.0             # SHAP 值计算与可视化

# --- 序列化与报告 ---
joblib>=1.3.0            # 模型序列化
jinja2>=3.1.0            # HTML 报告模板渲染
matplotlib>=3.7.0        # 图表生成
seaborn>=0.12.0          # 统计可视化

# --- 工具库 ---
pydantic>=2.0.0          # Schema 与接口校验
python-dotenv>=1.0.0     # 环境变量管理
tqdm>=4.65.0             # 进度条
```

### 2.3 可选依赖（按场景选择性安装）

| 场景 | 依赖包 | 安装命令 |
|---|---|---|
| 神经网络模型（AutoGluon Tabular NN） | torch, torchvision, fastai | `pip install autogluon`（完整版自动包含） |
| 文本特征提取（本地 Embedding） | sentence-transformers, torch | `pip install sentence-transformers>=2.3.0` |
| LLM 意图理解（OpenAI/Claude） | openai, anthropic | `pip install openai>=1.0.0` |
| 本地轻量 LLM（不调用 API） | transformers, torch, accelerate | `pip install transformers>=4.36.0` |
| 大数据分块处理 | dask, polars | `pip install dask>=2024.1.0 polars>=0.20.0` |
| ONNX 导出 | skl2onnx, onnxruntime | `pip install skl2onnx>=1.16.0 onnxruntime>=1.16.0` |

### 2.4 依赖安装策略

生产环境（最小可用，无 GPU，无 LLM API）

```bash
pip install -r requirements.txt
# requirements.txt 仅包含 2.2 核心依赖
```

完整开发环境（含 NN + 文本 + LLM）

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

依赖隔离建议（pyproject.toml 片段）

```toml
[project]
name = "automl-platform"
requires-python = ">=3.12,<3.13"
dependencies = [
    "prefect>=2.19.0",
    "autogluon>=1.1.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "feature-engine>=1.6.0",
    "imbalanced-learn>=0.11.0",
    "shap>=0.44.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "pymysql>=1.1.0",
    "psycopg2-binary>=2.9.0",
    "clickhouse-connect>=0.7.0",
    "openpyxl>=3.1.0",
    "joblib>=1.3.0",
    "jinja2>=3.1.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.65.0",
    "pyarrow>=14.0.0",
]

[project.optional-dependencies]
nn = ["torch>=2.1.0", "fastai>=2.7.0"]  # AutoGluon 神经网络
text = ["sentence-transformers>=2.3.0"]  # 文本 Embedding
llm = ["openai>=1.0.0", "anthropic>=0.21.0"]  # LLM API
local-llm = ["transformers>=4.36.0", "accelerate>=0.25.0", "torch>=2.1.0"]  # 本地 LLM
export = ["skl2onnx>=1.16.0", "onnxruntime>=1.16.0"]  # ONNX 导出
dev = ["pytest>=7.4.0", "black>=23.0.0", "ruff>=0.1.0", "mypy>=1.7.0"]
```

### 2.5 版本锁定与冲突规避

- AutoGluon 与 PyTorch：AutoGluon 会指定兼容的 PyTorch 版本范围，不要手动强制安装高版本 torch，应让 AutoGluon 的依赖解析器决定。
- Prefect 2.x：确保使用 Prefect 2.x（`prefect>=2.19.0`），与 1.x API 不兼容。
- SQLAlchemy 2.x：`sqlalchemy>=2.0.0` 与 1.x 有 Breaking Change，数据库连接代码需按 2.x 语法编写。
- Pandas 2.x：`pandas>=2.0.0` 性能更优，但部分旧代码需适配（如 `df.append` 已废弃）。

### 2.6 环境验证脚本

项目需提供一个 `check_env.py`，在启动时验证关键依赖：

```python
# check_env.py（需在代码库中提供）
import importlib
from packaging import version

REQUIRED = {
    "prefect": "2.19.0",
    "autogluon": "1.1.0",
    "pandas": "2.0.0",
    "sklearn": "1.3.0",
    "pydantic": "2.0.0",
}

OPTIONAL = {
    "torch": "2.1.0",
    "sentence_transformers": "2.3.0",
    "openai": "1.0.0",
}

def check():
    for pkg, min_ver in REQUIRED.items():
        mod = importlib.import_module(pkg)
        assert version.parse(mod.__version__) >= version.parse(min_ver), f"{pkg} version too low"
    print("Core dependencies OK")
    
    for pkg, min_ver in OPTIONAL.items():
        try:
            mod = importlib.import_module(pkg)
            assert version.parse(mod.__version__) >= version.parse(min_ver)
            print(f"Optional {pkg} OK")
        except ImportError:
            print(f"Optional {pkg} not installed, related features disabled")
```

---

## 3. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户输入层                                         │
│  ├─ 数据文件：CSV / Excel / Parquet / SQL DB（MySQL/PostgreSQL/ClickHouse） │
│  ├─ 业务描述：自然语言目标（如"预测销售额"）或结构化 TargetConfig           │
│  └─ 可选 Schema：JSON/YAML 字段定义                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM 意图理解层（Pre-flow）                            │
│  ├─ Intent Parser：自然语言 → 提取目标列、任务类型（分类/回归）              │
│  ├─ Schema Inference：无 Schema 时，LLM 推断字段类型/角色/约束               │
│  └─ Business Rule Extractor：识别用户隐含业务规则（如"年龄不能>120"）        │
│  【失败降级】：LLM 超时/失败时，切换至规则推断引擎，主流程不中断               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Prefect 编排层（核心引擎）                              │
│  ├─ Flow：AutoML Pipeline（动态 DAG）                                        │
│  │   ├─ Task 1：数据加载（Load Data）                                        │
│  │   ├─ Task 2：Schema 校验与对齐（Validate Schema）                        │
│  │   ├─ Task 3：元数据分析（Meta Analysis）                                   │
│  │   ├─ Task 4：策略路由（Strategy Router）【状态机决策点】                   │
│  │   ├─ Task 5：数据清洗（Data Cleaning）                                    │
│  │   ├─ Task 6：特征工程（Feature Engineering）                              │
│  │   ├─ Task 7：条件采样（Conditional Sampling）                              │
│  │   ├─ Task 8：条件降维（Conditional Dimensionality Reduction）               │
│  │   ├─ Task 9：模型搜索（Model Search）【AutoGluon / Optuna】               │
│  │   ├─ Task 10：交叉验证与评估（Evaluation）                                 │
│  │   ├─ Task 11：自动集成（Auto Ensemble）                                   │
│  │   └─ Task 12：最优模型导出（Export Best Model）                            │
│  └─ 状态持久化：Prefect 自动 Checkpoint，崩溃后断点续跑                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM 报告层（Post-flow）                               │
│  ├─ 业务解读：将技术指标翻译为业务语言                                        │
│  ├─ 异常诊断：解释 CV 波动、特征重要性 Top-N 业务含义                         │
│  └─ 生成 HTML/Markdown 报告                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出产物层                                         │
│  ├─ 预测结果：predictions.csv                                                │
│  ├─ 最优模型：model.pkl / ONNX（含预处理 Pipeline）                           │
│  ├─ 评估报告：metrics.json + leaderboard.csv                                   │
│  ├─ 可解释性：SHAP 汇总图 + 特征重要性表                                     │
│  └─ 完整报告：report.html（含业务解读）                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 数据接入层

### 4.1 支持的数据源

| 类型 | 格式 | 说明 |
|---|---|---|
| 文件 | CSV | 自动推断编码（UTF-8/GBK），支持大文件分块读取 |
| 文件 | Excel (.xlsx/.xls) | 自动读取所有 Sheet，默认取第一个；支持多 Sheet 选择 |
| 文件 | Parquet | 列式存储，优先推荐 |
| 数据库 | MySQL / PostgreSQL / ClickHouse / SQLite | 通过 SQLAlchemy 统一接入，支持自定义 SQL 查询 |

### 4.2 数据加载 Task（Prefect）

```python
@task(retries=3, retry_delay_seconds=10, persist_result=True)
def load_data(source: DataSource) -> pd.DataFrame:
    """
    根据 DataSource 类型加载数据为 DataFrame。
    结果自动缓存（Prefect persist_result），相同输入跳过重复加载。
    """
```

失败策略：数据库连接失败时重试 3 次；若仍失败，Flow 进入 `FAILED` 状态并通知用户。

---

## 5. Schema 与目标定义

### 5.1 Schema 结构

若用户未提供 Schema，由 LLM 意图理解层或规则引擎推断。Schema 必须包含：

```yaml
fields:
  - name: "customer_id"
    type: "id"              # 数值/类别/时间/文本/ID/权重/分组
    role: "drop"            # 特征/目标/丢弃
    nullable: false
  - name: "age"
    type: "numeric"
    role: "feature"
    constraint:
      min_value: 0
      max_value: 120
      nullable: true
  - name: "churn"
    type: "categorical"
    role: "target"
    description: "是否流失"
```

### 5.2 目标定义（TargetConfig）

```yaml
target:
  column: "churn"
  task_type: "binary_classification"  # binary / multiclass / regression
  primary_metric: "f1"               # 可选，未指定时自动选择
  pos_label: "Yes"                   # 二分类正例标签
```

自动选择逻辑：
- 二分类 + 不平衡 > 3:1 → 默认 `roc_auc` 或 `f1`
- 多分类 → 默认 `log_loss` 或 `accuracy`
- 回归 + 异常值多 → 默认 `mae`，否则 `rmse`

---

## 6. LLM 意图理解层（Pre-flow）

### 6.1 职责边界

LLM 仅负责输入理解，不参与训练决策。

| 模块 | 功能 | 输出 |
|---|---|---|
| Intent Parser | 解析用户自然语言描述 | TargetConfig（目标列、任务类型、评估指标建议） |
| Schema Inference | 无 Schema 时推断字段语义 | 补全 Schema（类型、角色、约束） |
| Business Rule Extractor | 提取隐含规则 | 清洗规则列表（如"收入不能为负"） |

### 6.2 失败隔离（关键）

```python
async def llm_understand_intent(user_query: str, df_sample: pd.DataFrame) -> UnderstandingResult:
    try:
        return await call_llm_api_with_timeout(timeout=30)
    except (TimeoutError, LLMError):
        # 降级：规则引擎推断
        return rule_based_inference(df_sample)
```

原则：LLM 层失败时，主流程必须继续，仅日志记录 `WARNING`。

---

## 7. Prefect 编排层（核心 DAG）

### 7.1 Flow 定义

```python
from prefect import flow, task
from prefect.tasks import task_input_hash

@flow(name="automl-end-to-end", log_prints=True)
async def automl_pipeline(
    data_source: DataSource,
    user_query: Optional[str] = None,
    user_schema: Optional[Schema] = None,
    time_budget_minutes: int = 60,
    output_dir: str = "./automl_output",
):
    ...
```

### 7.2 各 Task 详细定义

Task 1: 数据加载（`load_data`）
- 输入：DataSource
- 输出：`raw_df`（pd.DataFrame）
- 缓存策略：`persist_result=True`，按文件 hash 缓存

Task 2: Schema 校验（`validate_and_align_schema`）
- 输入：`raw_df`，`user_schema`（或 LLM 推断 Schema）
- 逻辑：
  1. 校验列名是否存在
  2. 校验类型是否兼容（如 Schema 声明 numeric 但列内全为字符串则触发类型转换或报错）
  3. 标记目标列、ID 列、丢弃列
- 输出：`aligned_df`, `validated_schema`

Task 3: 元数据分析（`analyze_metadata`）
- 输入：`aligned_df`, `validated_schema`, `target_config`
- 逻辑：
  - 样本数、特征数、内存占用
  - 目标分布（分类：类别比例；回归：分布偏度）
  - 缺失率矩阵
  - 特征类型统计（数值/类别/文本/时间）
  - 高基数类别检测（>1000 唯一值）
  - 时间序列特征检测（是否有单调递增时间列）
- 输出：`metadata`（dict）

Task 4: 策略路由（`select_strategy`）【核心决策点】
- 输入：`metadata`
- 逻辑（规则引擎，硬编码但可扩展）：

```python
def select_strategy(metadata: dict) -> PipelineTemplate:
    if metadata["task_type"] == "binary_classification":
        if metadata["imbalance_ratio"] > 10:
            return ImbalancedClassificationTemplate()
        elif metadata["n_samples"] > 1_000_000:
            return LargeScaleClassificationTemplate()
        else:
            return StandardClassificationTemplate()
    elif metadata["task_type"] == "regression":
        if metadata["n_features"] > 1000:
            return HighDimRegressionTemplate()
        else:
            return StandardRegressionTemplate()
```

- 输出：`pipeline_template`（节点列表 + 模型搜索空间）

Task 5: 数据清洗（`clean_data`）
- 输入：`aligned_df`, `validated_schema`, `metadata`
- 逻辑：
  - 缺失值处理：
    - 数值：中位数填充（高缺失率 >30% 时改用迭代插补或标记为缺失指示列）
    - 类别：众数填充或新增"未知"类别
    - ID/目标列：缺失则整行删除
  - 异常值处理：
    - 数值：基于 IQR 或 Isolation Forest 标记异常，根据 Schema 约束截断（如 age > 120 → 120）
    - 类别：高基数低频项合并为"其他"
  - 类型转换：确保所有列符合 Schema 声明
  - 重复行删除
- 防泄露原则：清洗参数（如填充均值）必须在交叉验证的每折内重新计算，通过 sklearn Pipeline 封装

Task 6: 特征工程（`engineer_features`）
- 输入：`cleaned_df`, `validated_schema`
- 逻辑：
  - 数值变换：对数、平方根、分箱（基于分布偏度自动选择）
  - 类别编码：低基数 One-Hot，高基数 Target Encoding（防泄露实现）
  - 时间特征：年月日、季度、是否节假日、时间差
  - 文本特征：如有文本列，使用本地预训练模型（Sentence-Transformers）提取 Embedding，不调用 LLM API
  - 交叉特征：基于树模型重要性反馈，自动尝试二阶交叉（数值×数值，数值×类别）
- 输出：`featured_df`, `preprocessor`（sklearn Pipeline 对象）

Task 7: 条件采样（`conditional_sampling`）
- 触发条件：`imbalance_ratio > 5`（分类任务）
- 逻辑：
  - 过采样：SMOTE / ADASYN（仅对训练集）
  - 欠采样：Random Under-sampling（大数据场景）
  - 组合：SMOTEENN
- 注意：采样必须在训练集划分后进行，验证集保持原始分布

Task 8: 条件降维（`conditional_dimensionality_reduction`）
- 触发条件：`n_features > 500` 或 `memory_estimate_gb > 16`
- 逻辑：
  - 高相关性剔除（相关系数 > 0.95）
  - 低方差剔除
  - PCA（保留 95% 方差，仅对数值特征）
- 注意：降维器在 CV 内 fit，防止泄露

Task 9: 模型搜索（`model_search`）【耗时核心】
- 内核：AutoGluon Tabular（封装死，不自研）
- 输入：`featured_df`, `target_config`, `model_space`, `time_budget`
- 逻辑：
  - 调用 `TabularPredictor.fit()`：
    - `presets="best_quality"`（追求最优）
    - `auto_stack=True`（自动多层集成）
    - `time_limit=time_budget * 60`
    - `hyperparameters` 根据数据规模动态裁剪搜索空间：
      - 小数据（<1000）：加入 SVM、KNN、神经网络
      - 大数据（>10万）：仅 LightGBM、XGBoost、线性模型
  - 自动处理类别不平衡（通过 `sample_weight`）
- 输出：`predictor`（AutoGluon 对象）, `leaderboard`（模型排名）

Task 10: 交叉验证与评估（`evaluate`）
- 输入：`predictor`, `featured_df`, `target_config`, `metadata`
- CV 策略自动选择：
  - 普通表格：StratifiedKFold（分类）/ KFold（回归）
  - 时间序列：TimeSeriesSplit
  - 分组数据：GroupKFold
- 评估指标：
  - 分类：AUC、F1、Accuracy、LogLoss（根据不平衡程度自动选择 primary）
  - 回归：RMSE、MAE、R²、MAPE
- 输出：`cv_scores`, `test_scores`, `metrics_dict`

Task 11: 自动集成（`auto_ensemble`）
- 逻辑：AutoGluon 内部已实现 weighted ensemble 和 stack ensemble，此 Task 负责：
  - 检查集成是否带来 >2% 指标提升
  - 若提升不足，回退到单最优模型
- 输出：`final_model`, `ensemble_config`

Task 12: 最优模型导出（`export_model`）
- 逻辑：
  - 保存完整 Pipeline（预处理 + 模型）为 `model.pkl` 或 ONNX
  - 保存元数据、Schema、配置快照，确保可复现
- 输出：`model_path`, `config_path`

---

## 8. 失败隔离与降级策略

| 层级 | 失败场景 | 策略 |
|---|---|---|
| LLM 意图层 | API 超时/限流/错误 | 降级至规则推断，记录 `WARNING`，Flow 继续 |
| 数据加载 | 文件不存在/DB 连接失败 | 重试 3 次后 Flow `FAILED`，通知用户 |
| Schema 校验 | 目标列不存在 | Flow `FAILED`，返回明确错误信息 |
| 数据清洗 | 某列 100% 缺失 | 丢弃该列，记录 `INFO`，继续 |
| 特征工程 | 文本 Embedding 模型下载失败 | 跳过文本特征，记录 `WARNING`，继续 |
| 模型搜索 | 单模型训练崩溃（如 NN OOM） | AutoGluon 自动跳过该模型，继续搜索其他模型 |
| 整体超时 | 超过 `time_budget` | Prefect 超时机制触发，返回当前最优结果（Best-so-far） |

---

## 9. 输出产物

| 产物 | 格式 | 说明 |
|---|---|---|
| 预测结果 | `predictions.csv` | 含预测值、概率（分类）、置信区间（回归） |
| 最优模型 | `model.pkl` + `preprocessor.pkl` | 完整 sklearn/AutoGluon Pipeline，可直接部署 |
| 模型排名 | `leaderboard.csv` | 所有尝试模型的 CV 分数、训练时间、推理延迟 |
| 评估指标 | `metrics.json` | 精确率、召回率、AUC、RMSE 等 |
| 特征重要性 | `feature_importance.csv` + `.png` | 全局重要性排名 |
| 可解释性 | `shap_summary.png` | SHAP 力图与汇总图 |
| 完整报告 | `report.html` | 业务语言解读 + 技术细节 + 数据质量摘要 |

---

## 10. 接口定义

### 10.1 输入接口

```python
@dataclass
class DBConfig:
    host: str
    port: int
    database: str
    table: str
    username: str
    password: str
    driver: Literal["mysql", "postgresql", "clickhouse", "sqlite"]

@dataclass
class FileConfig:
    path: str
    format: Literal["csv", "excel", "parquet"]
    sheet_name: Optional[str] = None
    encoding: str = "utf-8"

DataSource = Union[DBConfig, FileConfig, pd.DataFrame]

@dataclass
class TargetConfig:
    column: str
    task_type: Literal["binary_classification", "multiclass_classification", "regression"]
    primary_metric: Optional[str] = None
    pos_label: Optional[Union[str, int]] = None

@dataclass
class AutoMLConfig:
    time_budget_minutes: int = 60
    max_models: int = 50
    ensemble: bool = True
    random_state: int = 42
    output_dir: str = "./automl_output"
    n_jobs: int = -1
    use_gpu: bool = False
    llm_api_key: Optional[str] = None      # 可选，用于意图理解
    llm_model: str = "gpt-4o-mini"         # 默认轻量模型
```

### 10.2 输出接口

```python
@dataclass
class AutoMLResult:
    best_model_path: str
    predictions_path: Optional[str]
    metrics: Dict[str, float]
    leaderboard: pd.DataFrame
    report_path: str
    feature_importance: pd.DataFrame
    shap_values_path: Optional[str]
    config_snapshot: Dict[str, Any]        # 完整复现配置
```

---

## 11. 非功能性需求

### 11.1 性能
- 并发：模型搜索阶段支持多模型并行训练（AutoGluon 内部 + Prefect Task 并发）
- 大数据：支持分块读取（Dask/Polars 作为 pandas 替代），>100GB 数据需接入 Spark
- 内存：单节点建议 32GB+；超出时触发采样训练（先在小样本上搜索，再全量训练最优模型）

### 11.2 可扩展性
- 模型空间扩展：新增模型类型只需修改 `model_space` 配置，无需改动编排逻辑
- 数据源扩展：新增数据源类型只需实现 `load_data` 的分支，符合开闭原则
- 策略路由扩展：新增 Pipeline Template 只需在 `select_strategy` 中增加条件分支

### 11.3 可观测性
- Prefect UI：自动展示 DAG 执行图、各 Task 耗时、输入输出参数
- 日志分级：`INFO` 记录关键决策，`WARNING` 记录降级行为，`ERROR` 记录失败
- Artifact：在 Prefect UI 中直接查看 leaderboard 表格、特征重要性图

### 11.4 复现性
- 所有随机种子固定（`random_state`）
- 保存完整 `config_snapshot`（含数据 hash、Schema、超参、模型版本）
- 支持基于 snapshot 重新运行得到完全一致的结果

---

## 12. 技术栈

| 层级 | 技术选型 |
|---|---|
| 编排引擎 | Prefect（Python-native，异步支持，自动缓存） |
| 模型内核 | AutoGluon（模型搜索 + 自动集成） |
| 数据清洗 | pandas + sklearn（SimpleImputer / IsolationForest） |
| 特征工程 | sklearn ColumnTransformer + feature-engine + sentence-transformers（本地） |
| 超参优化 | AutoGluon 内置（默认）/ Optuna（高级定制） |
| 评估 | sklearn metrics + SHAP |
| LLM 层 | OpenAI API / Claude / 本地 vLLM（可选，失败可降级） |
| 数据库连接 | SQLAlchemy |
| 序列化 | joblib / ONNX |

---

## 13. 里程碑与验收标准

| 阶段 | 交付物 | 验收标准 |
|---|---|---|
| M1 | 数据接入 + Schema 校验 + 元分析 | 支持 CSV/Excel/DB 加载，Schema 推断准确率 >80% |
| M2 | 清洗 + 特征工程 + 策略路由 | 端到端跑通标准分类/回归任务，无人工干预 |
| M3 | 模型搜索 + 评估 + 集成 | AutoGluon 集成完成，CV 分数达到 AutoGluon 单机基准 |
| M4 | LLM 意图层 + 报告层 | 自然语言输入可自动解析目标，输出 HTML 报告 |
| M5 | 异常处理 + 断点续跑 + 压力测试 | 8 小时训练中途崩溃可恢复，内存 >16GB 场景自动降级 |

---

## 14. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| AutoGluon 对某类数据表现差 | 中 | 调整 preset（如从 `best_quality` 降为 `medium_quality`）、减少模型空间、缩短时间预算；如仍不达标则记录失败并提示用户 |
| LLM 推断 Schema 错误 | 高 | 必须提供 Schema 确认机制（MVP 中可先输出推断 Schema 供用户确认，全自动模式下置信度低时拒绝并提示） |
| 数据泄露（特征工程） | 高 | 强制使用 sklearn Pipeline 封装所有变换，确保 fit 仅在训练集 |
| 极端不平衡（>100:1） | 中 | 采样 + 代价敏感学习 + 采用 AUC-PR 作为主要指标 |
| 高维稀疏（>10k 特征） | 中 | 前置特征选择（L1 正则化 / 方差阈值），PCA 不适用时改用特征重要性筛选 |

---

## 15. 项目集成与部署方式

### 15.1 与 tech-portfolio 的关系

本项目（prefect-project）作为 **独立全栈项目** 运行，不直接嵌入 tech-portfolio 的代码库或虚拟环境。tech-portfolio 仅通过前端导航链接跳转的方式引流：

```html
<a href="http://localhost:8084" target="_blank">AutoML 平台</a>
```

理由：
- tech-portfolio 与 prefect-project 依赖差异大（numpy/sklearn 版本冲突）
- AutoML 训练资源消耗高，独立部署避免影响展示站点稳定性
- 两个项目技术栈、迭代节奏、部署目标不同，解耦后各自独立演进

### 15.2 推荐全栈架构

```
┌─────────────────┐      链接跳转       ┌─────────────────────────────┐
│  tech-portfolio │  ───────────────►  │      prefect-project        │
│   (现有项目)     │                    │    (独立域名/端口)           │
│  Vue + FastAPI  │                    │   Vue 3 + FastAPI           │
└─────────────────┘                    │   Prefect + AutoGluon       │
                                       │   SQLite + 文件存储         │
                                       └─────────────────────────────┘
```

### 15.3 后端服务层

- **FastAPI**：REST API 入口，与 tech-portfolio 技术栈一致
- **Prefect**：编排 AutoML 全流程，记录状态、日志、缓存
- **AutoGluon**：模型搜索与自动集成
- **SQLAlchemy + SQLite/PostgreSQL**：任务元数据、数据集元数据、评估指标持久化
- **文件系统**：上传数据集、训练好的模型、报告、图表等二进制产物

### 15.4 持久化设计

| 数据类型 | 存储位置 | 格式 |
|---------|---------|------|
| 上传数据集 | `./data/uploads/{dataset_id}/` | CSV / Excel / Parquet |
|---|---|---|
| 任务元数据 | SQLite/PostgreSQL | 结构化记录 |
|---|---|---|
| 训练好的模型 | `./data/models/{run_id}/` | joblib pkl / ONNX |
|---|---|---|
| 评估指标 | 数据库 + JSON | `metrics.json` |
|---|---|---|
| 排行榜 | 文件系统 + 数据库 | `leaderboard.csv` |
|---|---|---|
| HTML 报告 | `./data/reports/{run_id}/` | HTML + PNG |
|---|---|---|
| Prefect 运行状态 | Prefect 内置 DB/Server | State/Log/Artifact |
|---|---|---|

### 15.5 核心 API

```
POST   /api/datasets/upload        # 上传数据集
GET    /api/datasets               # 列出数据集
POST   /api/runs                   # 启动 AutoML 训练（异步）
GET    /api/runs                   # 列出训练任务
GET    /api/runs/{run_id}          # 查询任务状态
GET    /api/runs/{run_id}/results  # 获取训练结果
POST   /api/runs/{run_id}/predict  # 用训练好的模型预测
GET    /api/runs/{run_id}/report   # 下载 HTML 报告
DELETE /api/runs/{run_id}          # 删除任务及产物
```

### 15.6 目录结构

```
prefect-project/
├── backend/
│   ├── main.py
│   ├── routers/
│   ├── models.py              # SQLAlchemy ORM
│   ├── database.py            # 数据库连接
│   ├── services/
│   │   ├── storage.py         # 文件上传/下载
│   │   └── automl.py          # AutoGluon 封装
│   └── prefect_flows/
│       └── automl_flow.py     # Prefect Flow
├── frontend/
│   ├── src/
│   └── package.json
├── data/
│   ├── uploads/
│   ├── models/
│   ├── reports/
│   └── db.sqlite
├── task.md
└── requirements.txt
```

### 15.7 开发环境启动

```bash
# 1. 创建环境
cd prefect-project
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 启动后端
cd backend
uvicorn main:app --reload --port 8001

# 3. 启动前端
cd frontend
npm run dev -- --port 8084
```

---

## 16. AutoGluon 选型说明

### 16.1 AutoGluon 的定位

AutoGluon 是**表格数据 AutoML 的头部方案**，在 Kaggle 和工业界有广泛验证，核心优势：
- 开箱即用，3 行代码即可训练
- 自动多层集成（weighted/stack ensemble）通常能带来 SOTA 级别的效果
- 对类别不平衡、缺失值、文本/数值混合特征处理成熟
- 支持时间序列、多模态、图像、文本（虽然本项目主要用 Tabular）

### 16.2 AutoGluon 的局限性

| 局限 | 影响 |
|------|------|
| 依赖极重 | torch/lightgbm/xgboost/catboost/pyarrow 等，安装包体积大、版本敏感 |
|---|---|
| 黑盒程度高 | 自动集成后的模型解释性差，难以向业务方说明决策逻辑 |
|---|---|
| 定制成本高 | 如果想替换某个子模型或特征工程步骤，需要深入其内部 |
|---|---|
| 非表格数据支持一般 | 虽然有多模态，但强项仍是结构化表格 |
|---|---|
| 资源消耗大 | `best_quality` / `extreme` preset 在小型服务器上可能跑不动 |
|---|---|
| 对 Python 版本跟进较慢 | 3.14 尚未支持，新 Python 版本适配周期约 6-12 个月 |
|---|---|

### 16.3 备选方案对比

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **AutoGluon** | 效果最好、集成成熟、生态活跃 | 重、黑盒、版本敏感 | 表格数据追求最优效果 |
|---|---|---|---|
| **scikit-learn + Optuna** | 轻量、可解释、完全可控 | 需要自己写搜索和集成逻辑 | 中小型数据、需要白盒解释 |
|---|---|---|---|
| **FLAML（微软）** | 轻量、训练快、资源占用低 | 效果通常不如 AutoGluon | 资源受限、快速原型 |
|---|---|---|---|
| **H2O AutoML** | 企业级、稳定、支持多种语言 | 需要 JVM、社区版功能受限 | 大型企业、已有 H2O 生态 |
|---|---|---|---|
| **TPOT** | 遗传算法自动特征+模型 | 维护缓慢、训练极慢 | 研究性质、小规模实验 |
|---|---|---|---|
| **PyCaret** | 低代码、快速对比模型 | 稳定性一般、依赖冲突多 | 教学演示、快速探索 |
|---|---|---|---|
| **自研 Pipeline + Ray Tune** | 最灵活、可扩展 | 开发成本最高 | 超大规模、强定制需求 |
|---|---|---|---|

### 16.4 本项目选择 AutoGluon 的理由

**本项目以 AutoGluon 作为唯一模型内核**，原因如下：
1. 项目目标明确是"全自动"，AutoGluon 的集成能力最强
2. 项目主要处理 CSV/Excel/DB 表格数据，正是 AutoGluon 的主场
3. 作为个人作品集项目，AutoGluon 的名字和效果更有说服力


---

本文档可直接作为技术 PRD 交付给开发团队进行实施。
