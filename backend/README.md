# AI Simulation Scientist — Backend Engine 🧬🤖

The backend engine for the **Universal AI Simulation Laboratory**: an autonomous multi-agent simulation and scientific reasoning platform built with **LangGraph**, **FastAPI**, **Python**, and **LLMs** (Google Gemini via OpenAI-compatible endpoints).

The engine autonomously translates natural language scenario descriptions into rigorous mathematical specifications, selects appropriate modeling methodologies (discrete-event, agent-based, Monte Carlo, graph-theoretic, differential equations), generates and executes simulation code within a secured sandbox, validates numerical integrity, and synthesizes 20-section peer-review-grade scientific research reports.

---

## Architecture & Multi-Agent Pipeline

The backend execution pipeline is orchestrated as a directed graph in **LangGraph** (`backend/agent/graph.py`), comprising 8 specialized agent and computation nodes:

```text
[User Prompt] 
      │
      ▼
1. scenario_parser      ──> Extracts mathematical formulation, state variables & distributions
      │
      ▼
2. hypothesis_updater   ──> Formulates testable hypotheses with Bayesian prior belief probabilities
      │
      ▼
3. planner              ──> Simulation Architect: selects simulation paradigm & designs causal branches
      │
      ▼
4. critic               ──> Evaluates methodology, provenance, metric separation & causal branch soundness
      │
      ├── (Pass) ──► 5. executor   ──> Runs Python code in isolated sandbox with timeout guards
      └── (Fail) ──► (Re-plans with critic feedback, max retries)
                           │
                           ▼
                     6. validator  ──> Pure-Python numerical consistency & invariant check (10 checks)
                           │
                           ▼
                     7. stats      ──> Computes CI (95%), constraint violations & event frequencies
                           │
                           ▼
                     8. analyst    ──> Produces comprehensive 20-section scientific research report
```

### Pipeline Nodes Breakdown

1. **`scenario_parser`**: Parses unstructured natural language into structured simulation variables, constraints, time horizons, and probability distributions.
2. **`hypothesis_updater`**: Establishes prior probability beliefs ($P(H_i)$) for key hypotheses, updated to posterior beliefs ($P(H_i|D)$) upon experimental results.
3. **`planner`**: Determines appropriate simulation paradigms (e.g., discrete-event simulation via SimPy/time-queues, Monte Carlo sampling, NetworkX graph cascades, or differential equations) and builds the code architecture.
4. **`critic`**: Analyzes code quality, verifies random number generator isolation (`numpy.random.default_rng`), ensures non-zero variance across trials, and guards against metric confounding.
5. **`executor`**: Dispatches the simulation into a sandboxed subprocess (`backend/utils/sandbox.py`) with strict memory/timeout limits, structured JSON stdout capture, and stderr error isolation.
6. **`validator`**: Executes 10 automated mathematical sanity checks (e.g., probability bounds $[0, 1]$, non-negative durations, finite variances, sample size consistency).
7. **`stats`**: Calculates empirical confidence intervals, tail-risk probabilities, and constraint violation metrics.
8. **`analyst`**: Drafts an exhaustive scientific research paper containing executive summary, methodology, mathematical equations, hypothesis evaluation table, limitations, and actionable recommendations.

---

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous Web API)
- **Multi-Agent Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain)
- **LLM Integration**: [LangChain OpenAI](https://github.com/langchain-ai/langchain) (compatible with Google Gemini API & OpenAI endpoints)
- **Scientific & Simulation Libraries**:
  - [NumPy](https://numpy.org/) — High-performance vectorization & PRNG
  - [SciPy](https://scipy.org/) — Scientific distributions & hypothesis testing
  - [SimPy](https://simpy.readthedocs.io/) — Process-based discrete-event simulation
  - [NetworkX](https://networkx.org/) — Complex network and graph modeling
  - [Pandas](https://pandas.pydata.org/) — Tabular data manipulation
  - [Matplotlib](https://matplotlib.org/) — Data visualization support
- **Observability**: [LangSmith](https://smith.langchain.com/) (Optional distributed tracing)

---

## Directory Structure

```text
backend/
├── agent/
│   ├── graph.py       # LangGraph state graph & conditional execution flow
│   ├── nodes.py       # Node implementations (parser, planner, critic, executor, validator, etc.)
│   ├── prompts.py     # Prompt templates for LLM agents
│   └── state.py       # TypedDict state schema across pipeline nodes
├── utils/
│   ├── __init__.py
│   ├── logger.py      # Terminal logging with color-coded stage indicators
│   └── sandbox.py     # Subprocess code execution sandbox with timeouts
├── config.py          # Environment settings and configuration loader
├── main.py            # FastAPI entry point, CORS middleware & API routes
├── requirements.txt   # Python package dependencies
└── README.md          # Backend documentation
```

---

## Multi-Language Natural Language Detection

The backend features an automatic multilingual detection engine (`detect_language` in `main.py`). The pipeline automatically detects the user's language and instructs the analyst agent to output reports in the user's native language:

- **Supported Languages**: English, Chinese (中文), Japanese (日本語), Korean (한국어), Russian (Русский), Arabic (العربية), Spanish (Español), French (Français), German (Deutsch), Portuguese (Português), Italian (Italiano), and more.
- Override available via the `language` field in the request payload.

---

## Getting Started

### 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- An API Key for your LLM (Google Gemini API key recommended via OpenAI-compatible endpoint, or OpenAI API key)

### 2. Environment Setup

Create a virtual environment and activate it:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

Create a `.env` file in the project root or the `backend/` directory:

```env
# LLM Configuration (Google Gemini OpenAI-compatible endpoint)
OPENAI_API_KEY=your_gemini_api_key_here
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-2.5-flash

# Optional: LangSmith Tracing
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=ai-simulation-lab

# Optional: Firebase Service Account (For authentication verification)
# GOOGLE_APPLICATION_CREDENTIALS=./serviceAccountKey.json
# ADMIN_EMAILS=admin@example.com
```

### 4. Running the Server

From the repository root:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from within the `backend/` folder:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be live at `http://127.0.0.1:8000`.  
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

---

## API Reference

### Health Check
- **Endpoint**: `GET /`
- **Response**:
```json
{
  "message": "AI Simulation API v2.2 — Universal Simulation Engine (Scientifically Robust, Multi-Language)",
  "pipeline": [
    "scenario_parser",
    "hypothesis_updater (Bayesian prior→posterior belief table)",
    "planner (Simulation Architect — multi-branch causal modeling)",
    "critic (methodology + provenance + metric separation + causal branch checks)",
    "executor",
    "validator (pure Python — 10 consistency checks)",
    "stats (constraint violation probabilities + event frequency)",
    "analyst (20-section scientific report with hypothesis update analysis)"
  ]
}
```

### Run Simulation
- **Endpoint**: `POST /api/simulate`
- **Request Body**:
```json
{
  "prompt": "Simulate a pharmaceutical supply chain with 3 distribution centers and cold-chain temperature deviations over 90 days.",
  "language": "auto"
}
```
- **Response**:
```json
{
  "status": "success",
  "simulation_spec": "...",
  "hypothesis_analysis": "...",
  "plan": "...",
  "code": "import numpy as np...",
  "critic_feedback": "...",
  "critic_approved": true,
  "output": "...",
  "error": null,
  "structured_results": {...},
  "validation_report": "...",
  "statistical_summary": "...",
  "report": "# Scientific Research Report\n...",
  "language": "English"
}
```