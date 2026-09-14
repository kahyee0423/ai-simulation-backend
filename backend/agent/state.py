from typing import TypedDict, Optional

class SimulationState(TypedDict):
    user_prompt: str             # Natural language prompt from the user
    language: str                # Target language for simulation outputs (e.g. "Chinese", "English")
    simulation_spec: str         # Structured JSON simulation specification (with provenance)
    hypothesis_analysis: str     # JSON — prior/posterior belief table, dominant hypothesis, evidence log
    plan: str                    # AI-generated simulation plan (raw LLM text)
    generated_code: str          # AI-generated Python simulation script
    critic_feedback: str         # Model Critic review of the generated code
    critic_approved: bool        # Whether the Model Critic approved the code
    critic_retry_count: int      # Retry counter for critic-loop failures
    execution_output: str        # Standard output (stdout) from sandbox
    execution_error: str         # Standard error (stderr) if execution fails
    structured_results: str      # JSON block extracted from execution output
    retry_count: int             # Self-correction retry counter (syntax errors)
    validation_report: str       # Pure-Python post-execution consistency checks
    statistical_summary: str     # Python-computed statistics (mean, median, percentiles, etc.)
    analysis_report: str         # Final 19-section scientific analysis report