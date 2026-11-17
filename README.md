# AI Simulation Laboratory — Backend Engine

The backend engine for the **Universal AI Simulation Laboratory**: an autonomous multi-agent scientific reasoning and simulation platform built with **FastAPI**, **LangGraph**, **Python**, and **Google Gemini** (via OpenAI-compatible API).

The engine autonomously translates natural language scenario descriptions into rigorous mathematical specifications, selects appropriate modeling methodologies (discrete-event, agent-based, Monte Carlo, differential equations, graph-theoretic), generates and executes simulation code within a secured sandbox, validates numerical integrity, synthesizes 20-section peer-review-grade scientific research reports, and outputs publication-ready charts.

Frontend Repository: [ai-simulation-frontend](https://github.com/kahyee0423/ai-simulation-frontend)

---

## Key Capabilities

- **Autonomous Multi-Agent System (MAS)**: Cyclic state machine built on **LangGraph** orchestrating 8 specialized agent roles with closed-loop self-correction.
- **Neurosymbolic AI**: Combines neural language intelligence (Google Gemini) for semantic parsing, planning, and scientific synthesis with deterministic symbolic execution (NumPy, SciPy, SimPy, NetworkX, 10-point mathematical invariant validator).
- **Automated Scientific Discovery ("In Silico Lab")**: Automates the scientific loop from prior hypotheses $P(H_i)$ to empirical simulation data and posterior belief updates $P(H_i|D)$.
- **Sandboxed Program Synthesis**: Automated generation and safe subprocess execution of Python simulation scripts with memory and timeout guards.
- **Automated Scientific Visualization**: Generates publication-grade empirical charts using Matplotlib and NumPy with 95% confidence intervals and Bayesian belief shift comparisons.
- **Multilingual Support**: Automatic prompt language detection (English, Chinese, Spanish, French, German, Japanese, and more) with reports generated in the user's native tongue.

---

## Architecture & Multi-Agent Pipeline

The backend execution pipeline is orchestrated as a directed graph in **LangGraph** (`agent/graph.py`), comprising 8 specialized agent and computation nodes:

```text
[User Natural Language Scenario] 
               │
               ▼
1. scenario_parser      ──> Extracts mathematical formulation, state variables, constraints & distributions
               │
               ▼
2. hypothesis_updater   ──> Formulates testable hypotheses with Bayesian prior belief probabilities
               │
               ▼
3. planner              ──> Simulation Architect: selects simulation paradigm & generates executable Python code
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
                              8. analyst    ──> Produces 20-section scientific report + publication chart
```

---

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous Web API) & [Uvicorn](https://www.uvicorn.org/)
- **Multi-Agent Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain)
- **LLM Integration**: [LangChain OpenAI](https://github.com/langchain-ai/langchain) (Compatible with Google Gemini API & OpenAI endpoints)
- **Scientific & Simulation Libraries**:
  - [NumPy](https://numpy.org/) — High-performance vectorization & PRNG
  - [SciPy](https://scipy.org/) — Scientific distributions & hypothesis testing
  - [SimPy](https://simpy.readthedocs.io/) — Process-based discrete-event simulation
  - [NetworkX](https://networkx.org/) — Complex network and graph modeling
  - [Pandas](https://pandas.pydata.org/) — Tabular data manipulation
  - [Matplotlib](https://matplotlib.org/) — Scientific data visualization & figures
- **Observability**: [LangSmith](https://smith.langchain.com/) (Optional distributed tracing)

---

## Directory Structure

```text
ai-simulation-backend/
├── agent/
│   ├── __init__.py
│   ├── graph.py       # LangGraph state graph & conditional execution flow
│   ├── nodes.py       # 8 specialized agent node implementations
│   └── state.py       # TypedDict state schema across pipeline nodes
├── utils/
│   ├── __init__.py
│   ├── chart_generator.py # Matplotlib empirical figures & Bayesian belief visualizer
│   ├── logger.py      # Terminal logging with color-coded stage indicators
│   └── sandbox.py     # Subprocess code execution sandbox with timeouts & safety guards
├── config.py          # Environment settings and configuration loader
├── main.py            # FastAPI entry point, CORS middleware & API routes
├── requirements.txt   # Python package dependencies
└── README.md          # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites

- **Python**: Version 3.10, 3.11, 3.12, or 3.13
- **Git**
- **LLM API Key**: Google Gemini API Key (recommended) or OpenAI API Key

### 2. Clone the Repository

```bash
git clone https://github.com/kahyee0423/ai-simulation-backend.git
cd ai-simulation-backend
```

### 3. Create and Activate a Virtual Environment

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

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Google Gemini API via OpenAI-compatible endpoint
OPENAI_API_KEY=your_gemini_api_key_here
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-2.5-flash

# Optional: LangSmith Distributed Tracing
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=ai-simulation-lab
```

---

## Running the Server Locally

Start the development server with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger API Docs: `http://localhost:8000/docs`
- Redoc Documentation: `http://localhost:8000/redoc`

---

## Deployment to Render.com

When deploying this repository as a Web Service on **Render**:

1. **Environment**: Python
2. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Command**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. **Root Directory**: Leave empty
5. **Environment Variables**: Add your `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL_NAME` in the Render Environment tab.

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
  "prompt": "Simulate 50 delivery drones navigating through a 10x10 city grid with battery constraints, dynamic obstacle collision risks, and delivery time optimization over 100 trials.",
  "language": "auto"
}
```
- **Response**:
```json
{
  "status": "success",
  "simulation_spec": "{...}",
  "hypothesis_analysis": "{...}",
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
  "language": "English",
  "chart": "data:image/png;base64,..."
}
```

---

## License

MIT License. Designed for open science, educational exploration, and reproducible in-silico experiments.
