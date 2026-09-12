from typing import TypedDict, Optional

class SimulationState(TypedDict):
    # ── Phase 0: Input ──────────────────────────────────────────────
    user_prompt: str             # Natural language prompt from the user

    # ── Phase 1: Scenario Parsing ────────────────────────────────────
    simulation_spec: str         # Structured JSON simulation specification (with provenance)

    # ── Phase 2: Code Generation ─────────────────────────────────────
    plan: str                    # AI-generated simulation plan (raw LLM text)
    generated_code: str          # AI-generated Python simulation script
    critic_feedback: str         # Model Critic review of the generated code
    critic_approved: bool        # Whether the Model Critic approved the code
    critic_retry_count: int      # Retry counter for critic-loop failures

    # ── Phase 3: Execution ───────────────────────────────────────────
    execution_output: str        # Standard output (stdout) from sandbox
    execution_error: str         # Standard error (stderr) if execution fails
    structured_results: str      # JSON block extracted from execution output
    retry_count: int             # Self-correction retry counter (syntax errors)

    # ── Phase 4: Validation ──────────────────────────────────────────
    validation_report: str       # Pure-Python post-execution consistency checks

    # ── Phase 5: Statistical Analysis ────────────────────────────────
    statistical_summary: str     # Python-computed statistics (mean, median, percentiles, etc.)

    # ── Phase 6: Report ──────────────────────────────────────────────
    analysis_report: str         # Final 19-section scientific analysis report