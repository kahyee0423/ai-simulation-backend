import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import settings
from backend.agent.state import SimulationState
from backend.utils.sandbox import execute_python_code
from backend.utils.logger import (
    log_step_start,
    log_step_progress,
    log_step_success,
    log_step_warning,
    log_step_error,
    log_retry_trigger,
)

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL,
    temperature=0.2
)

def planner_node(state: SimulationState) -> SimulationState:
    """Planner: Generates an executable Python simulation script based on the user prompt."""
    retry_count = state.get("retry_count", 0)
    has_error = bool(state.get("execution_error"))
    
    if has_error:
        log_step_start(
            "PLANNER",
            f"Fixing and regenerating simulation code (Attempt {retry_count + 1}) using '{settings.MODEL_NAME}'..."
        )
    else:
        log_step_start(
            "PLANNER",
            f"Generating initial simulation script using model '{settings.MODEL_NAME}'..."
        )

    error_context = ""
    if has_error:
        error_context = f"""
[Previous Execution Error]:
{state['execution_error']}

Please analyze the error above and fix the simulation script accordingly.
"""

    prompt = f"""
You are an expert AI Simulation Scientist. The user wants to run the following simulation:
"{state.get('user_prompt', '')}"
{error_context}
Please write a complete, self-contained, and executable Python script to perform this simulation.
You can use pre-installed scientific libraries: numpy, scipy, simpy, networkx, matplotlib, random, math, pandas.

[Strict Requirements]:
1. The code must be pure Python, enclosed within ```python and ```.
2. The code must print critical simulation metrics using print() so they can be captured for analysis.
3. Do not use interactive UI commands like plt.show() (as it will hang the process); save figures or omit plotting.
4. The code must be robust with no undefined variables.
"""
    messages = [
        SystemMessage(content="You are an expert autonomous simulation scientist."),
        HumanMessage(content=prompt)
    ]
    
    log_step_progress("Sending request to LLM, waiting for response...")
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time
    content = response.content
    
    code = ""
    if "```python" in content:
        parts = content.split("```python")
        code = parts[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        code = parts[1].strip()
    else:
        code = content.strip()

    num_lines = len(code.splitlines()) if code else 0
    log_step_success(
        "PLANNER",
        elapsed,
        f"Generated {num_lines} lines of Python code"
    )

    return {
        "plan": content,
        "generated_code": code,
        "retry_count": retry_count
    }

def executor_node(state: SimulationState) -> SimulationState:
    """Executor: Runs the AI-generated code inside the sandbox environment."""
    code = state.get("generated_code", "")
    num_lines = len(code.splitlines()) if code else 0
    log_step_start("EXECUTOR", f"Running simulation code ({num_lines} lines) in sandbox...")

    start_time = time.time()
    stdout, stderr = execute_python_code(code)
    elapsed = time.time() - start_time
    
    if stderr:
        last_error_line = stderr.strip().splitlines()[-1] if stderr.strip() else "Unknown error"
        log_step_warning("EXECUTOR", f"Script produced errors: {last_error_line}")
    else:
        out_lines = len(stdout.strip().splitlines()) if stdout.strip() else 0
        log_step_success("EXECUTOR", elapsed, f"Execution completed with {out_lines} lines of stdout")
    
    return {
        "execution_output": stdout,
        "execution_error": stderr
    }

def analyst_node(state: SimulationState) -> SimulationState:
    """Analyst: Interprets the execution results and generates a scientific report."""
    log_step_start("ANALYST", f"Generating scientific analysis report using model '{settings.MODEL_NAME}'...")
    
    prompt = f"""
Original User Prompt: {state.get('user_prompt', '')}

Executed Python Code:
```python
{state.get('generated_code', '')}
```

Execution Output (Stdout):
{state.get('execution_output', '')}

Execution Errors (Stderr, if any):
{state.get('execution_error', '')}

As the Chief Scientific Officer, write a clear and structured 【Simulation Experiment Analysis Report】.
Include:
1. Executive Summary of the Experiment
2. Key Findings & Data Insights (based on Stdout)
3. Scientific Conclusions & Recommendations
"""
    messages = [
        SystemMessage(content="You are an expert data scientist and research analyst."),
        HumanMessage(content=prompt)
    ]

    log_step_progress("Sending output and metrics to LLM, analyzing results...")
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time

    word_count = len(response.content.split()) if response.content else 0
    log_step_success("ANALYST", elapsed, f"Research report generated ({word_count} words)")

    return {
        "analysis_report": response.content
    }

def should_retry(state: SimulationState) -> str:
    """Conditional Edge: If execution fails and retries < 2, retry planning."""
    current_retries = state.get("retry_count", 0)
    max_retries = 2
    
    if state.get("execution_error") and current_retries < max_retries:
        state["retry_count"] = current_retries + 1
        log_retry_trigger(
            state["retry_count"],
            max_retries,
            state.get("execution_error", "")
        )
        return "retry"
    
    if state.get("execution_error"):
        log_step_warning(
            "ROUTER",
            f"Max retries ({max_retries}) reached with execution errors. Proceeding to Analyst..."
        )
    else:
        log_step_progress("Execution validated successfully. Routing to Analyst...")
    return "proceed"