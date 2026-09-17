import time
import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.graph import simulation_app
from backend.utils.logger import (
    log_request_start,
    log_request_end,
    log_step_error,
)

app = FastAPI(title="AI Simulation API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-simulation-laboratory.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    prompt: str
    language: Optional[str] = "auto"

def detect_language(text: str) -> str:
    """
    Detect the primary natural language of the user prompt.
    Supports Chinese, Japanese, Korean, Russian, Arabic, Spanish, French, German,
    Portuguese, Italian, and defaults to English or the identified language.
    """
    if not text or not text.strip():
        return "English"

    # 1. Non-latin script detection via Unicode ranges
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    japanese_kana = len(re.findall(r'[\u3040-\u30ff]', text))
    korean_hangul = len(re.findall(r'[\uac00-\ud7af]', text))
    cyrillic_count = len(re.findall(r'[\u0400-\u04ff]', text))
    arabic_count = len(re.findall(r'[\u0600-\u06ff]', text))

    if chinese_count >= 2 and chinese_count >= japanese_kana:
        return "Chinese"
    if japanese_kana >= 2:
        return "Japanese"
    if korean_hangul >= 2:
        return "Korean"
    if cyrillic_count >= 3:
        return "Russian"
    if arabic_count >= 3:
        return "Arabic"

    # 2. Latin script language heuristics
    lower = text.lower()
    words = set(re.findall(r'\b[a-zà-ÿ]+\b', lower))

    spanish_markers = {"el", "la", "los", "las", "un", "una", "de", "en", "que", "por", "para", "con", "simular", "simulación", "drones", "entrega", "ciudad", "modelo", "es", "son"}
    french_markers = {"le", "la", "les", "un", "une", "des", "du", "de", "en", "que", "pour", "avec", "simuler", "simulation", "livraison", "ville", "est", "sont"}
    german_markers = {"der", "die", "das", "ein", "eine", "und", "in", "den", "von", "zu", "mit", "simulieren", "simulation", "lieferung", "stadt", "ist", "sind"}
    portuguese_markers = {"o", "a", "os", "as", "um", "uma", "em", "para", "com", "não", "simular", "simulação", "entrega", "são"}
    italian_markers = {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "in", "per", "con", "simulare", "simulazione", "città"}

    if len(words & spanish_markers) >= 2:
        return "Spanish"
    if len(words & french_markers) >= 2:
        return "French"
    if len(words & german_markers) >= 2:
        return "German"
    if len(words & portuguese_markers) >= 2:
        return "Portuguese"
    if len(words & italian_markers) >= 2:
        return "Italian"

    return "English"

@app.get("/")
def read_root():
    return {
        "message": "AI Simulation API v2.2 — Universal Simulation Engine (Scientifically Robust, Multi-Language)",
        "pipeline": [
            "scenario_parser",
            "hypothesis_updater (Bayesian prior→posterior belief table)",
            "planner (Simulation Architect — multi-branch causal modeling)",
            "critic (methodology + provenance + metric separation + causal branch checks)",
            "executor",
            "validator (pure Python — 10 consistency checks)",
            "stats (constraint violation probabilities + event frequency)",
            "analyst (20-section scientific report with hypothesis update analysis)",
        ]
    }

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    resolved_language = req.language if req.language and req.language.lower() != "auto" else detect_language(req.prompt)

    log_request_start(f"[{resolved_language}] {req.prompt}")
    start_time = time.time()

    initial_state = {
        "user_prompt": req.prompt,
        "language": resolved_language,
        "simulation_spec": "",
        "hypothesis_analysis": "",
        "plan": "",
        "generated_code": "",
        "critic_feedback": "",
        "critic_approved": False,
        "critic_retry_count": 0,
        "execution_output": "",
        "execution_error": "",
        "structured_results": "",
        "retry_count": 0,
        "validation_report": "",
        "statistical_summary": "",
        "analysis_report": "",
    }

    try:
        final_state = simulation_app.invoke(initial_state)
        total_time = time.time() - start_time
        log_request_end(total_time, success=True)

        return {
            "status": "success",
            "simulation_spec": final_state.get("simulation_spec"),
            "hypothesis_analysis": final_state.get("hypothesis_analysis"),
            "plan": final_state.get("plan"),
            "code": final_state.get("generated_code"),
            "critic_feedback": final_state.get("critic_feedback"),
            "critic_approved": final_state.get("critic_approved"),
            "output": final_state.get("execution_output"),
            "error": final_state.get("execution_error"),
            "structured_results": final_state.get("structured_results"),
            "validation_report": final_state.get("validation_report"),
            "statistical_summary": final_state.get("statistical_summary"),
            "report": final_state.get("analysis_report"),
            "language": final_state.get("language") or resolved_language,
        }
    except Exception as e:
        total_time = time.time() - start_time
        log_step_error("SIMULATION", f"Pipeline aborted: {str(e)}")
        log_request_end(total_time, success=False, error_msg=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
