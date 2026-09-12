# AI Simulation Scientist 🧬🤖

An autonomous multi-agent system built with **LangGraph**, **LangSmith**, **Gemini**, and **FastAPI**. It allows users to simulate any scenario in natural language, writes and executes Python simulation code dynamically in a secure sandbox, and outputs a professional scientific research report.

## Tech Stack
- **AI Orchestration**: LangGraph, LangSmith
- **LLM**: Gemini 3.5 Flash Lite (via OpenAI-compatible API)
- **Backend**: FastAPI, Python
- **Simulation Libraries**: NumPy, SciPy, SimPy, NetworkX, Matplotlib

## Quick Start
1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
Configure environment variables:
code
Bash
cp .env.example .env
# Fill in your API keys in .env
Run the server:
code
Bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload