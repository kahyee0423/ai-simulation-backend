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

app = FastAPI(title="AI Simulation API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you can restrict this to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "AI Simulation API is running with LangGraph!"}

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    log_request_start(req.prompt)
    start_time = time.time()

    initial_state = {
        "user_prompt": req.prompt,
        "plan": "",
        "generated_code": "",
        "execution_output": "",
        "execution_error": "",
        "analysis_report": "",
        "retry_count": 0
    }
    
    try:
        final_state = simulation_app.invoke(initial_state)
        total_time = time.time() - start_time
        log_request_end(total_time, success=True)
        
        return {
            "status": "success",
            "plan": final_state.get("plan"),
            "code": final_state.get("generated_code"),
            "output": final_state.get("execution_output"),
            "error": final_state.get("execution_error"),
            "report": final_state.get("analysis_report")
        }
    except Exception as e:
        total_time = time.time() - start_time
        log_step_error("SIMULATION", f"Pipeline aborted due to unexpected error: {str(e)}")
        log_request_end(total_time, success=False, error_msg=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)