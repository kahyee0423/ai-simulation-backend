from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import settings
from backend.agent.state import SimulationState
from backend.utils.sandbox import execute_python_code

# Initialize OpenAI-compatible Qwen client
llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL,
    temperature=0.2
)

def planner_node(state: SimulationState) -> SimulationState:
    """Planner: Generates an executable Python simulation script based on the user prompt."""
    error_context = ""
    if state.get("execution_error"):
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
    
    response = llm.invoke(messages)
    content = response.content
    
    # Extract Python code block from markdown
    code = ""
    if "```python" in content:
        parts = content.split("```python")
        code = parts[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        code = parts[1].strip()
    else:
        code = content.strip()

    return {
        "plan": content,
        "generated_code": code,
        "retry_count": state.get("retry_count", 0)
    }

def executor_node(state: SimulationState) -> SimulationState:
    """Executor: Runs the AI-generated code inside the sandbox environment."""
    code = state.get("generated_code", "")
    stdout, stderr = execute_python_code(code)
    
    return {
        "execution_output": stdout,
        "execution_error": stderr
    }

def analyst_node(state: SimulationState) -> SimulationState:
    """Analyst: Interprets the execution results and generates a scientific report."""
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
    response = llm.invoke(messages)
    return {
        "analysis_report": response.content
    }

def should_retry(state: SimulationState) -> str:
    """Conditional Edge: If execution fails and retries < 2, retry planning."""
    if state.get("execution_error") and state.get("retry_count", 0) < 2:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "retry"
    return "proceed"