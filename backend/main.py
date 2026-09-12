from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.graph import simulation_app

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
        raise HTTPException(status_code=400, model_detail="Prompt cannot be empty")
    
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
        return {
            "status": "success",
            "plan": final_state.get("plan"),
            "code": final_state.get("generated_code"),
            "output": final_state.get("execution_output"),
            "error": final_state.get("execution_error"),
            "report": final_state.get("analysis_report")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))