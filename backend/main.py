import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.graph import simulation_app
from backend.utils.logger import (
    log_request_start,
    log_request_end,
    log_step_error,
)

app = FastAPI(title="AI Simulation API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {
        "message": "AI Simulation API v2.1 — Universal Simulation Engine (Scientifically Honest)",
        "pipeline": [
            "scenario_parser",
            "planner (Simulation Architect)",
            "critic (methodology + provenance + extrapolation checks)",
            "executor",
            "validator (pure Python — 10 consistency checks)",
            "stats (constraint violation probabilities + event frequency)",
            "analyst (19-section scientific report)",
        ]
    }

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    log_request_start(req.prompt)
    start_time = time.time()

    initial_state = {
        # Phase 0: Input
        "user_prompt": req.prompt,

        # Phase 1: Scenario parsing (with provenance labels)
        "simulation_spec": "",

        # Phase 2: Code generation
        "plan": "",
        "generated_code": "",
        "critic_feedback": "",
        "critic_approved": False,
        "critic_retry_count": 0,

        # Phase 3: Execution
        "execution_output": "",
        "execution_error": "",
        "structured_results": "",
        "retry_count": 0,

        # Phase 4: Validation
        "validation_report": "",

        # Phase 5: Statistical analysis
        "statistical_summary": "",

        # Phase 6: Report
        "analysis_report": "",
    }

    try:
        final_state = simulation_app.invoke(initial_state)
        total_time = time.time() - start_time
        log_request_end(total_time, success=True)

        return {
            "status": "success",
            # Phase 1 — Specification
            "simulation_spec": final_state.get("simulation_spec"),
            # Phase 2 — Code
            "plan": final_state.get("plan"),
            "code": final_state.get("generated_code"),
            "critic_feedback": final_state.get("critic_feedback"),
            "critic_approved": final_state.get("critic_approved"),
            # Phase 3 — Execution
            "output": final_state.get("execution_output"),
            "error": final_state.get("execution_error"),
            "structured_results": final_state.get("structured_results"),
            # Phase 4 — Validation
            "validation_report": final_state.get("validation_report"),
            # Phase 5 — Statistics
            "statistical_summary": final_state.get("statistical_summary"),
            # Phase 6 — Report
            "report": final_state.get("analysis_report"),
        }
    except Exception as e:
        total_time = time.time() - start_time
        log_step_error("SIMULATION", f"Pipeline aborted: {str(e)}")
        log_request_end(total_time, success=False, error_msg=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)