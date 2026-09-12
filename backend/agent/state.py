from typing import TypedDict, List, Dict, Any

class SimulationState(TypedDict):
    user_prompt: str             # Natural language prompt from the user
    plan: str                    # AI-generated simulation plan
    generated_code: str          # AI-generated Python simulation script
    execution_output: str        # Standard output (stdout) from execution
    execution_error: str         # Standard error (stderr) if execution fails
    analysis_report: str         # Final scientific analysis report from AI
    retry_count: int             # Self-correction retry counter