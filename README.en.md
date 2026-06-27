[中文](./README.md) | English

# Prefect AutoML Platform

An end-to-end fully automated machine learning platform powered by **Prefect** workflow orchestration + **AutoGluon** automated modeling, automating the entire pipeline from data cleaning to model deployment. The backend also provides a natural language intent parsing API for integration into external workflows.

---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Development Guide](#development-guide)
- [API Overview](#api-overview)
- [Frontend Pages](#frontend-pages)
- [LLM Configuration](#llm-configuration)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [TODO / Roadmap](#todo--roadmap)
- [Security Notice](#security-notice)
- [License](#license)

---

## Introduction

Prefect AutoML Platform is a low-code AutoML platform for structured data. Users can upload datasets through the Web UI, select the target column and task type, and the platform automatically completes:

1. Data understanding and Schema inference
2. Data cleaning and feature engineering
3. Automated model selection and hyperparameter search (AutoGluon)
4. Model evaluation and interpretability analysis (SHAP / Permutation Importance)
5. Visualization reports and experiment comparison

It also provides a complete REST API for easy integration into existing data pipelines.

---

## Key Features

- 🧠 **Natural Language Intent Parsing (Backend API)**: Provides the `/api/intent` endpoint for parsing natural language descriptions into training configs, extracting business rules, and inferring schemas, which can be integrated into external workflows.
- 🔄 **Prefect Workflow Orchestration**: Training jobs are scheduled and executed by a Prefect Server, with Flow Runs, Task Runs, logs, and artifacts viewable in the Prefect UI; automatically falls back to local subprocess execution when Prefect is unavailable.
- 🤖 **AutoGluon AutoML**: Automatically tries models and ensemble strategies such as LightGBM, CatBoost, and XGBoost.
- 🛠️ **Data Quality & Feature Engineering**: Built-in missing value handling, categorical encoding, imbalanced sample processing, feature selection, and more.
- 📊 **Interpretability Reports**: SHAP values, permutation importance, feature correlation, and distribution visualization.
- 🔌 **Optional LLM Enhancement**: Supports connecting to external LLMs such as KIMI, DeepSeek, MiniMax, GLM, and OpenAI; falls back to the local rule engine when not configured, so the core AutoML pipeline can run without an LLM.
- 🌐 **Vue 3 Frontend**: Based on Element Plus + ECharts, supporting dataset management, training pipelines, and experiment comparison.
- 🧪 **Comprehensive Test Coverage**: pytest unit tests and end-to-end tests.

---

## Tech Stack

### Backend

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Main development language |
| FastAPI | Web API framework |
| SQLAlchemy 2.0 + aiosqlite | Async ORM and SQLite database |
| Prefect 3.x | Workflow orchestration |
| AutoGluon Tabular | Automated machine learning |
| LightGBM / CatBoost / XGBoost | Gradient boosting models |
| scikit-learn / pandas / numpy | Data processing and evaluation |
| SHAP / imbalanced-learn / feature-engine | Interpretability and feature engineering |
| OpenAI SDK | Unified multi-LLM provider calls |
| Pytest / Black / Ruff / mypy | Testing and code quality |

### Frontend

| Technology | Purpose |
|------------|---------|
| Vue 3 | Frontend framework |
| Element Plus | UI component library |
| Pinia | State management |
| Vue Router | Routing |
| ECharts 6 | Data visualization |
| Vite | Build tool |

---

## Project Structure

```text
.
├── backend/                    # FastAPI backend
│   ├── main.py                 # Application entrypoint
│   ├── config.py               # Configuration and settings
│   ├── database.py             # Database connection and initialization
│   ├── models.py               # SQLAlchemy data models
│   ├── schemas.py              # Pydantic data validation
│   ├── routers/                # API routes
│   ├── services/               # Business service layer
│   └── prefect_flows/          # Prefect workflow definitions
├── frontend/                   # Vue 3 frontend
│   ├── src/
│   │   ├── api/                # API request encapsulation
│   │   ├── components/         # Shared components
│   │   ├── stores/             # Pinia state
│   │   ├── views/              # Page views
│   │   ├── App.vue
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── scripts/                    # Startup and utility scripts
├── tests/                      # Test cases
├── data/                       # Data, model, and report storage
├── docs/                       # Documentation
├── Makefile                    # Common commands
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Core dependencies
├── requirements-full.txt       # Full dependencies
└── .env.example                # Environment variable example
```

---

## Quick Start

### 1. Environment Preparation

- Python >= 3.12, < 3.13
- Node.js >= 18 (for frontend build)
- [uv](https://github.com/astral-sh/uv) is recommended for fast package management

### 2. Clone and Install

```bash
# Clone the project
git clone <repository-url>
cd prefect-automl-platform

# Create and activate virtual environment (recommended)
uv venv -p python3.12 .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
make install
# Or full dependencies (includes torch / text / image capabilities)
make install-full
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env to configure the database and optional LLM API Key
```

### 4. Start Services

```bash
# Development mode (starts both frontend and backend)
make dev

# Production mode (runs in background)
make start

# Stop production services
make stop
```

After startup, visit:

- Frontend: `http://localhost:8084`
- Backend API: `http://localhost:8001`
- API Docs: `http://localhost:8001/docs`
- Prefect UI: `http://localhost:4200`

> `scripts/run_dev.sh` and `scripts/run_prod.sh` automatically start the Prefect Server (port 4200) and a local Runner. To disable Prefect orchestration, set `PREFECT_ENABLED=false` in `.env`; training will fall back to the original local subprocess mode.
>
> To access the Prefect UI from another machine on the LAN, change `PREFECT_API_URL` in `.env` to this machine's LAN IP (e.g. `http://192.168.1.5:4200/api`) and restart the services.

---

## Development Guide

### Common Commands

```bash
make help              # View all available commands
make check             # Run environment check
make install           # Install core dependencies
make install-full      # Install full dependencies
make dev               # Start development services
make start             # Start production services
make stop              # Stop production services
make test              # Run tests (skip slow tests)
make test-slow         # Run all tests (including end-to-end)
make lint              # Run ruff static checks
make format            # Run black formatting
make clean             # Clean caches and build artifacts
```

### Start Backend Separately

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Start Frontend Separately

```bash
cd frontend
npm install
npm run dev
```

### Code Style

- Backend uses **Black** (line width 100) and **Ruff** for formatting and static checks.
- Type annotations are checked with **mypy**.
- Before committing, it is recommended to run `make format && make lint && make test`.

---

## API Overview

| Route Prefix | Description |
|--------------|-------------|
| `GET /health` | Health check |
| `GET /docs` | Swagger auto-generated docs |
| `/api/datasets` | Dataset upload, list, detail, delete |
| `/api/intent` | Natural language intent parsing, rule extraction, Schema inference |
| `/api/runs` | Training run submission, status, logs, results |
| `/api/experiments` | Experiment comparison and reports |
| `/api/settings` | LLM provider and model configuration |

### Example: Submit Training Intent

```bash
curl -X POST http://localhost:8001/api/intent/parse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Use the Titanic dataset to predict whether a passenger survived; target column is Survived",
    "dataset_id": "<your-dataset-id>",
    "provider": "auto"
  }'
```

---

## Frontend Pages

| Page | Function |
|------|----------|
| Home | Platform overview and quick access |
| Dataset Management | Upload CSV/Excel, preview, view Schema |
| Training Pipeline | Configure training parameters, confirm config, one-click training |
| Run History | View status, metrics, and logs for each training run |
| Run Detail | Model leaderboard, interpretability charts, download report |
| Experiment Comparison | Multi-experiment metrics and visualization comparison |
| LLM Settings | Configure LLM provider, API Key, default model |

---

## LLM Configuration

The platform supports a local rule engine fallback, so the core AutoML pipeline can run without an LLM. Connecting to an external LLM is optional and enhances the natural language interaction experience, but enabling it will send anonymized data samples and training metadata to third-party cloud service providers.

Supported providers: `kimi`, `deepseek`, `minimax`, `glm`, `openai`, `auto`.

### Environment Variable Example

```env
LLM_PROVIDER=auto
KIMI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=sk-xxx
MINIMAX_API_KEY=sk-xxx
GLM_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx

# Override default model (optional)
DEFAULT_LLM_MODEL=moonshot-v1-8k
```

### Performance Tuning (Optional)

```env
# Training set evaluation sampling, 0=skip, N=sample N rows
TRAIN_EVAL_SAMPLE_SIZE=0

# SHAP maximum sample size, 0=skip
SHAP_MAX_SAMPLE_SIZE=0

# Permutation importance repeats and sample size
PERMUTATION_IMPORTANCE_MAX_REPEATS=0
PERMUTATION_IMPORTANCE_SAMPLE_SIZE=5000

# Maximum rows for data quality evaluation
DATA_QUALITY_MAX_ROWS=50000
```

---

## Testing

```bash
# Run regular tests (recommended for daily development)
make test

# Run full test suite including slow/end-to-end tests
make test-slow

# Run a single test file
pytest tests/test_api.py -v
```

---

## Environment Variables

See [`.env.example`](./.env.example) for details.

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection URL | `sqlite+aiosqlite://./data/db.sqlite` |
| `LLM_PROVIDER` | Default LLM provider | `auto` |
| `KIMI_API_KEY` | KIMI API Key | Empty |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | Empty |
| `MINIMAX_API_KEY` | MiniMax API Key | Empty |
| `GLM_API_KEY` | Zhipu GLM API Key | Empty |
| `OPENAI_API_KEY` | OpenAI API Key | Empty |
| `DEFAULT_LLM_MODEL` | Override default model | Empty |
| `TRAIN_EVAL_SAMPLE_SIZE` | Training set evaluation sampling | Empty (auto) |
| `SHAP_MAX_SAMPLE_SIZE` | SHAP maximum sampling | Empty (auto) |
| `PERMUTATION_IMPORTANCE_MAX_REPEATS` | Permutation importance repeat count | Empty (auto) |
| `PERMUTATION_IMPORTANCE_SAMPLE_SIZE` | Permutation importance sampling | Empty (auto) |
| `DATA_QUALITY_MAX_ROWS` | Maximum rows for data quality evaluation | Empty (auto) |

---

## TODO / Roadmap

- [x] Internationalization (i18n) support for Chinese and English
- [ ] Fix file upload path traversal and add extension/size validation
- [ ] Upgrade starlette / fastapi and pydantic-settings to fix known CVEs
- [ ] Sanitize LLM error messages to avoid leaking API Key fragments from third-party responses
- [ ] Adapt to Python 3.15 after upstream dependencies become compatible

---

## Security Notice

This project is intended as a **local or trusted intranet self-hosted tool**. It does not provide public services, and direct deployment on the public Internet is not recommended.

The current version has the following deployment-related limitations; please assess the risks yourself:

- **No authentication/authorization**: All APIs are open by default; anyone who can reach the backend can call training, prediction, model download, and other endpoints.
- **Permissive CORS**: In development mode, `allow_origins=["*"]` and `allow_credentials=True` are both enabled for convenient local frontend-backend debugging; tighten this for public deployments.
- **Absolute paths in API responses**: Dataset and run responses contain server-local absolute paths (`file_path`, `output_dir`).
- **External database connections**: The database connection feature lets users specify arbitrary hosts and SQLite file paths and execute custom SQL; only connect to data sources you trust.
- **Raw errors returned to client**: Some endpoints return the original exception message to the frontend to aid local debugging.
- **Dependency vulnerabilities**: Some dependencies (e.g., starlette, pydantic-settings) have known CVEs; see [TODO / Roadmap](#todo--roadmap).

When using this in public or shared environments, add your own reverse proxy, authentication, CORS restrictions, and rate limiting.

---

## AI-Generated Content Statement

Parts of this project were generated with the assistance of generative artificial intelligence (Kimi, Moonshot AI) and have been reviewed, modified, and tested by humans. The original generated content is subject to the relevant service terms, and the intellectual property ownership and liability of the final project rest with the project maintainers.

---

## License

The main code of this project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE).

**Core Restrictions**:
- Anyone may freely use, modify, and distribute this software.
- **Modified versions must be open-sourced under AGPL-3.0 or later** (strong copyleft).
- If a modified version is provided as a service over the network (e.g., SaaS), the corresponding complete source code must be made available to all users accessing it.
- For closed-source commercial use, please contact the copyright owner for a separate license.

### Third-Party Dependency License Notice

This project uses many excellent open-source libraries, each governed by its own license. The following are common core dependencies and their licenses (for reference only; the actual declaration in each dependency package shall prevail):

| Dependency | License |
|------------|---------|
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

> Before distribution, please review the full license files of each dependency in `node_modules/` and the Python virtual environment to ensure compliance. The main body of this project is licensed under AGPL-3.0, which permits commercial use but requires modified versions to be open-sourced; third-party dependencies are governed by their respective licenses.

---

> If you encounter any issues during use, please feel free to submit an Issue or Pull Request.
