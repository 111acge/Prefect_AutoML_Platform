# Prefect AutoML Platform

端到端全自动机器学习平台，基于 **Prefect** 工作流编排 + **AutoGluon** 自动建模，配合自然语言意图理解，实现“说一句业务目标，跑完数据清洗到模型部署”的全流程自动化。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [开发指南](#开发指南)
- [API 概览](#api-概览)
- [前端页面](#前端页面)
- [LLM 配置](#llm-配置)
- [测试](#测试)
- [环境变量](#环境变量)
- [TODO / 路线图](#todo--路线图)
- [许可证](#许可证)

---

## 项目简介

Prefect AutoML Platform 是一个面向结构化数据的低代码 AutoML 平台。用户可以通过 Web 界面上传数据集、用自然语言描述建模目标，平台自动完成：

1. 数据理解与 Schema 推断
2. 数据清洗与特征工程
3. 模型自动选型与超参搜索（AutoGluon）
4. 模型评估、可解释性分析（SHAP / Permutation Importance）
5. 可视化报告与实验对比

同时提供完整的 REST API，便于集成到现有数据流水线中。

---

## 核心特性

- 🧠 **自然语言建模**：通过 LLM 解析“预测用户流失”“根据历史销量预测下个月”等描述，自动生成训练配置。
- 🔄 **Prefect 工作流编排**：训练、评估、报告生成等步骤以 Prefect Flow 形式组织，支持观测与重跑。
- 🤖 **AutoGluon 自动机器学习**：自动尝试 LightGBM、CatBoost、XGBoost 等模型与集成策略。
- 🛠️ **数据质量与特征工程**：内置缺失值处理、类别编码、不平衡样本处理、特征筛选等能力。
- 📊 **可解释性报告**：SHAP 值、排列重要性、特征相关性、分布可视化。
- 🔌 **多 LLM 提供商支持**：KIMI、DeepSeek、MiniMax、GLM、OpenAI，未配置时自动降级到规则引擎。
- 🌐 **Vue 3 前端**：基于 Element Plus + ECharts，支持数据集管理、训练流水线、实验对比。
- 🧪 **完整测试覆盖**：pytest 单元测试与端到端测试。

---

## 技术栈

### 后端

| 技术 | 用途 |
|------|------|
| Python 3.12 | 主开发语言 |
| FastAPI | Web API 框架 |
| SQLAlchemy 2.0 + aiosqlite | 异步 ORM 与 SQLite 数据库 |
| Prefect 3.x | 工作流编排 |
| AutoGluon Tabular | 自动机器学习 |
| LightGBM / CatBoost / XGBoost | 梯度提升模型 |
| scikit-learn / pandas / numpy | 数据处理与评估 |
| SHAP / imbalanced-learn / feature-engine | 可解释性与特征工程 |
| OpenAI SDK | 多 LLM 提供商统一调用 |
| Pytest / Black / Ruff / mypy | 测试与代码质量 |

### 前端

| 技术 | 用途 |
|------|------|
| Vue 3 | 前端框架 |
| Element Plus | UI 组件库 |
| Pinia | 状态管理 |
| Vue Router | 路由 |
| ECharts 6 | 数据可视化 |
| Vite | 构建工具 |

---

## 项目结构

```text
.
├── backend/                    # FastAPI 后端
│   ├── main.py                 # 应用入口
│   ├── config.py               # 配置与设置
│   ├── database.py             # 数据库连接与初始化
│   ├── models.py               # SQLAlchemy 数据模型
│   ├── schemas.py              # Pydantic 数据校验
│   ├── routers/                # API 路由
│   ├── services/               # 业务服务层
│   └── prefect_flows/          # Prefect 工作流定义
├── frontend/                   # Vue 3 前端
│   ├── src/
│   │   ├── api/                # API 请求封装
│   │   ├── components/         # 公共组件
│   │   ├── stores/             # Pinia 状态
│   │   ├── views/              # 页面视图
│   │   ├── App.vue
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── scripts/                    # 启动与工具脚本
├── tests/                      # 测试用例
├── data/                       # 数据、模型、报告存储
├── docs/                       # 文档
├── Makefile                    # 常用命令
├── pyproject.toml              # Python 项目配置
├── requirements.txt            # 核心依赖
├── requirements-full.txt       # 完整依赖
└── .env.example                # 环境变量示例
```

---

## 快速开始

### 1. 环境准备

- Python >= 3.12，< 3.13
- Node.js >= 18（前端构建）
- 推荐安装 [uv](https://github.com/astral-sh/uv) 用于快速包管理

### 2. 克隆与安装

```bash
# 克隆项目
git clone <repository-url>
cd prefect-automl-platform

# 创建并激活虚拟环境（推荐）
uv venv -p python3.12 .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装核心依赖
make install
# 或完整依赖（含 torch / 文本/图像能力）
make install-full
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，配置数据库与可选 LLM API Key
```

### 4. 启动服务

```bash
# 开发模式（同时启动前后端）
make dev

# 生产模式（后台运行）
make start

# 停止生产服务
make stop
```

启动后访问：

- 前端：`http://localhost:8084`
- 后端 API：`http://localhost:8001`
- API 文档：`http://localhost:8001/docs`

---

## 开发指南

### 常用命令

```bash
make help              # 查看所有可用命令
make check             # 运行环境检查
make install           # 安装核心依赖
make install-full      # 安装完整依赖
make dev               # 启动开发服务
make start             # 启动生产服务
make stop              # 停止生产服务
make test              # 运行测试（跳过慢速）
make test-slow         # 运行全部测试（含端到端）
make lint              # 运行 ruff 静态检查
make format            # 运行 black 格式化
make clean             # 清理缓存和构建产物
```

### 单独启动后端

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 单独启动前端

```bash
cd frontend
npm install
npm run dev
```

### 代码风格

- 后端使用 **Black**（行宽 100）和 **Ruff** 进行格式化和静态检查。
- 类型注解使用 **mypy** 检查。
- 提交前建议运行 `make format && make lint && make test`。

---

## API 概览

| 路由前缀 | 说明 |
|----------|------|
| `GET /health` | 健康检查 |
| `GET /docs` | Swagger 自动文档 |
| `/api/datasets` | 数据集上传、列表、详情、删除 |
| `/api/intent` | 自然语言意图解析、规则提取、Schema 推断 |
| `/api/runs` | 训练运行提交、状态、日志、结果 |
| `/api/experiments` | 实验对比与报告 |
| `/api/settings` | LLM 提供商与模型配置 |

### 示例：提交训练意图

```bash
curl -X POST http://localhost:8001/api/intent/parse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "用 Titanic 数据集预测乘客是否生还，目标列是 Survived",
    "dataset_id": "<your-dataset-id>",
    "provider": "auto"
  }'
```

---

## 前端页面

| 页面 | 功能 |
|------|------|
| 首页 | 平台概览与快捷入口 |
| 数据集管理 | 上传 CSV/Excel、预览、Schema 查看 |
| 训练流水线 | 自然语言输入、配置确认、一键训练 |
| 运行记录 | 查看每次训练状态、指标、日志 |
| 运行详情 | 模型排行榜、可解释性图表、下载报告 |
| 实验对比 | 多组实验指标与可视化对比 |
| LLM 设置 | 配置 LLM 提供商、API Key、默认模型 |

---

## LLM 配置

平台支持通过环境变量或前端界面配置 LLM。未配置任何 Key 时，会**自动降级到规则引擎**，仍可完成常见意图解析。

支持的 Provider：`kimi`、`deepseek`、`minimax`、`glm`、`openai`、`auto`。

### 环境变量示例

```env
LLM_PROVIDER=auto
KIMI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=sk-xxx
MINIMAX_API_KEY=sk-xxx
GLM_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx

# 覆盖默认模型（可选）
DEFAULT_LLM_MODEL=moonshot-v1-8k
```

### 性能调优（可选）

```env
# 训练集评估采样，0=跳过，N=采样N行
TRAIN_EVAL_SAMPLE_SIZE=0

# SHAP 最大采样数，0=跳过
SHAP_MAX_SAMPLE_SIZE=0

# 排列重要性重复次数与采样数
PERMUTATION_IMPORTANCE_MAX_REPEATS=0
PERMUTATION_IMPORTANCE_SAMPLE_SIZE=5000

# 数据质量评估最大行数
DATA_QUALITY_MAX_ROWS=50000
```

---

## 测试

```bash
# 运行常规测试（推荐日常开发使用）
make test

# 运行包含慢速/端到端测试的完整测试套件
make test-slow

# 单独运行某个测试文件
pytest tests/test_api.py -v
```

---

## 环境变量

详见 [`.env.example`](./.env.example)。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接 URL | `sqlite+aiosqlite://./data/db.sqlite` |
| `LLM_PROVIDER` | 默认 LLM 提供商 | `auto` |
| `KIMI_API_KEY` | KIMI API Key | 空 |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 空 |
| `MINIMAX_API_KEY` | MiniMax API Key | 空 |
| `GLM_API_KEY` | 智谱 GLM API Key | 空 |
| `OPENAI_API_KEY` | OpenAI API Key | 空 |
| `DEFAULT_LLM_MODEL` | 覆盖默认模型 | 空 |
| `TRAIN_EVAL_SAMPLE_SIZE` | 训练集评估采样 | 空（自动） |
| `SHAP_MAX_SAMPLE_SIZE` | SHAP 最大采样 | 空（自动） |
| `PERMUTATION_IMPORTANCE_MAX_REPEATS` | 排列重要性重复次数 | 空（自动） |
| `PERMUTATION_IMPORTANCE_SAMPLE_SIZE` | 排列重要性采样 | 空（自动） |
| `DATA_QUALITY_MAX_ROWS` | 数据质量评估最大行数 | 空（自动） |

---

## TODO / 路线图

- [ ] 国际化（i18n）支持

---

## 许可证

本项目主体代码采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE) 许可证。

**核心限制**：
- 任何人都可以自由使用、修改、分发本软件。
- **修改后的版本必须以 AGPL-3.0 或更高版本开源**（强传染性）。
- 如果修改后的版本通过网络提供服务（如 SaaS），必须向所有访问用户提供对应的完整源代码。
- 如需闭源商业使用，请联系版权所有者获取单独授权。

### 第三方依赖许可证说明

本项目使用了大量优秀的开源库，各库受其自身许可证约束。以下为常见核心依赖及其许可证（供参考，具体以各依赖包内实际声明为准）：

| 依赖 | 许可证 |
|------|--------|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| SQLAlchemy | MIT |
| Alembic | MIT |
| Prefect | Apache-2.0 |
| AutoGluon | Apache-2.0 |
| LightGBM | MIT |
| CatBoost | Apache-2.0 |
| XGBoost | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| SHAP | MIT |
| imbalanced-learn | MIT |
| feature-engine | BSD-3-Clause |
| OpenAI SDK | MIT |
| Pydantic | MIT |
| python-dotenv | BSD-3-Clause |
| Jinja2 | BSD-3-Clause |
| Matplotlib | PSF-based |
| Seaborn | BSD-3-Clause |
| Vue.js | MIT |
| Element Plus | MIT |
| Pinia | MIT |
| Vue Router | MIT |
| Vite | MIT |
| ECharts | Apache-2.0 |
| Axios | MIT |

> 分发前，请务必 review `node_modules/` 和 Python 虚拟环境中各依赖的完整许可证文件，确保符合相关条款。本项目主体以 AGPL-3.0 授权，允许商业使用但要求修改版本开源；第三方依赖受其各自许可证约束。

---

> 如果你在使用过程中遇到问题，欢迎提交 Issue 或 Pull Request。
