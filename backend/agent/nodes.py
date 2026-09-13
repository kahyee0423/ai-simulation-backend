import time
import json
import re
import math
import statistics as _stats
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

# ──────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _extract_code_block(content: str) -> str:
    """Extract the first Python code block from an LLM response."""
    if "```python" in content:
        return content.split("```python")[1].split("```")[0].strip()
    if "```" in content:
        return content.split("```")[1].strip()
    return content.strip()


def _extract_json_block(content: str) -> dict | None:
    """
    Try to find and parse a JSON object in a string.
    Looks for a SIMULATION_RESULTS_JSON: marker first,
    then falls back to the first {...} block found.
    """
    marker = "SIMULATION_RESULTS_JSON:"
    if marker in content:
        after = content.split(marker, 1)[1].strip()
        brace_start = after.find("{")
        if brace_start != -1:
            depth = 0
            for i, ch in enumerate(after[brace_start:], brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(after[brace_start: i + 1])
                        except json.JSONDecodeError:
                            break

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except Exception:
        return False


def _get_num_runs(spec_str: str) -> int:
    """Extract num_runs from spec JSON, defaulting to 500."""
    try:
        spec = json.loads(spec_str)
        return int(spec.get("num_runs", 500))
    except Exception:
        return 500


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences from a JSON response."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 — SCENARIO PARSER
# ──────────────────────────────────────────────────────────────────────────────

def scenario_parser_node(state: SimulationState) -> SimulationState:
    """
    Scenario Parser: Converts the natural-language user prompt into a
    structured JSON simulation specification with provenance labels.
    """
    log_step_start(
        "SCENARIO PARSER",
        f"Extracting structured simulation specification using '{settings.MODEL_NAME}'..."
    )

    prompt = f"""
You are a Simulation Scenario Analyst for a Universal AI Simulation Engine.

Your task is to read the user's natural-language scenario and extract a
structured simulation specification as a JSON object.

CRITICAL: Classify every parameter with a provenance label:
  USER_PROVIDED   — the user explicitly stated this value
  DERIVED         — calculated from a USER_PROVIDED value
  MODEL_ASSUMPTION — the simulation will need to invent this; not stated by the user
  UNKNOWN         — unclear; must be flagged as a gap

USER SCENARIO:
{state.get('user_prompt', '')}

Respond ONLY with a valid JSON object with this exact schema:
{{
  "domain": "<string>",
  "objective": "<one sentence>",
  "time_horizon": {{"steps": <int>, "unit": "<string>", "provenance": "USER_PROVIDED|MODEL_ASSUMPTION"}},
  "num_runs": <int>,
  "methodology_hint": "<one of: deterministic_simulation | stochastic_simulation | monte_carlo | discrete_event_simulation | system_dynamics | agent_based_simulation | game_theoretic | hybrid>",
  "methodology_note": "<why this methodology was chosen>",
  "entities": [
    {{"name": "<str>", "type": "agent|environment|resource|process", "objective": "<str or null>"}}
  ],
  "initial_state": {{
    "<param_name>": {{"value": <value_or_null>, "provenance": "USER_PROVIDED|MODEL_ASSUMPTION|DERIVED|UNKNOWN", "note": "<optional note>"}}
  }},
  "parameters": {{
    "<param_name>": {{"value": <value_or_null>, "provenance": "USER_PROVIDED|MODEL_ASSUMPTION|DERIVED|UNKNOWN", "note": "<optional note>"}}
  }},
  "actions": {{
    "<entity_name>": ["<action1>", "<action2>"]
  }},
  "outcomes": ["<primary_metric>", "<secondary_metric>"],
  "constraints": [
    {{"name": "<str>", "description": "<str>", "provenance": "USER_PROVIDED|MODEL_ASSUMPTION"}}
  ],
  "stochastic_variables": [
    {{"name": "<str>", "distribution_hint": "<str>", "provenance": "USER_PROVIDED|MODEL_ASSUMPTION"}}
  ],
  "assumption_gaps": ["<important variable not provided by the user that the model must invent>"],
  "one_time_assumptions": ["<assumptions that apply ONCE, not recurring>"],
  "recurring_assumptions": ["<assumptions that recur each time step>"]
}}

IMPORTANT RULES:
- Only mark as USER_PROVIDED if the user explicitly stated the value.
- If the user says \"enrollment will increase 20% next year\", classify that as a
  one_time_assumption, NOT a recurring assumption.
- Never invent \"empirical\" values — label everything the simulation invents as MODEL_ASSUMPTION.
- methodology_hint must reflect what the code will ACTUALLY implement, not what sounds impressive.
  Do NOT choose agent_based_simulation unless the scenario genuinely requires autonomous agents
  with their own state, observations, objectives, and decision policies.
"""

    messages = [
        SystemMessage(
            content=(
                "You are an expert simulation scientist. "
                "Extract structured simulation specifications with provenance labels. "
                "Respond only with valid JSON."
            )
        ),
        HumanMessage(content=prompt)
    ]

    log_step_progress("Sending scenario to LLM for structured extraction...")
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time

    raw = response.content or ""
    spec_str = _strip_json_fences(raw)

    try:
        spec_obj = json.loads(spec_str)
        domain = spec_obj.get("domain", "unknown")
        methodology = spec_obj.get("methodology_hint", "unknown")
        num_runs = spec_obj.get("num_runs", "?")
        entities = len(spec_obj.get("entities", []))
        gaps = len(spec_obj.get("assumption_gaps", []))
        one_time = len(spec_obj.get("one_time_assumptions", []))
        log_step_success(
            "SCENARIO PARSER",
            elapsed,
            f"Spec extracted — domain={domain}, methodology={methodology}, "
            f"runs={num_runs}, entities={entities}, "
            f"gaps={gaps}, one-time assumptions={one_time}"
        )
    except json.JSONDecodeError:
        log_step_warning(
            "SCENARIO PARSER",
            "Could not parse spec as JSON — storing raw text and continuing"
        )
        spec_str = raw

    return {"simulation_spec": spec_str}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 — PLANNER (Simulation Architect)
# ──────────────────────────────────────────────────────────────────────────────

def planner_node(state: SimulationState) -> SimulationState:
    """
    Planner (Simulation Architect): Generates a rigorous, scientifically honest
    simulation script guided by the specification and provenance rules.
    """
    retry_count = state.get("retry_count", 0)
    critic_retry_count = state.get("critic_retry_count", 0)
    has_exec_error = bool(state.get("execution_error"))
    has_critic_issue = bool(state.get("critic_feedback")) and not state.get("critic_approved", True)

    if has_exec_error:
        log_step_start(
            "PLANNER",
            f"Fixing execution error — regeneration attempt {retry_count + 1} "
            f"using '{settings.MODEL_NAME}'..."
        )
    elif has_critic_issue:
        log_step_start(
            "PLANNER",
            f"Addressing critic feedback — revision {critic_retry_count} "
            f"using '{settings.MODEL_NAME}'..."
        )
    else:
        log_step_start(
            "PLANNER",
            f"Generating simulation script (Simulation Architect) using '{settings.MODEL_NAME}'..."
        )

    spec = state.get("simulation_spec", "")
    spec_block = f"""
--------------------------------------------------
SIMULATION SPECIFICATION
(follow this contract — respect provenance labels)
--------------------------------------------------
{spec}
""" if spec else ""

    error_context = ""
    if has_exec_error:
        error_context = f"""
--------------------------------------------------
PREVIOUS EXECUTION ERROR — fix this before anything else
--------------------------------------------------
{state.get('execution_error', '')}
"""

    critic_context = ""
    if has_critic_issue:
        critic_context = f"""
--------------------------------------------------
MODEL CRITIC FEEDBACK — address ALL points listed
--------------------------------------------------
{state.get('critic_feedback', '')}
"""

    num_runs = _get_num_runs(spec)

    prompt = f"""
You are the Simulation Architect of a Universal AI Simulation Engine.

Your task is to transform the user's scenario into a rigorous, executable,
scientifically honest Python simulation.

USER SCENARIO:
\"{state.get('user_prompt', '')}\"
{spec_block}{error_context}{critic_context}

═══════════════════════════════════════════════════
PART 1 — BEFORE WRITING CODE: IDENTIFY THE MODEL
═══════════════════════════════════════════════════

Mentally identify:
1. Simulation domain
2. Simulation objective
3. Entities (actors, resources, environments)
4. State variables (what changes over time)
5. Parameters (fixed values the model uses)
6. Constraints (hard limits the system must respect)
7. Decision-makers and their decision logic
8. Events (discrete occurrences that change state)
9. Time scale and horizon
10. Initial conditions
11. Uncertainty sources
12. Observable outcome metrics
13. Failure / termination conditions

═══════════════════════════════════════════════════
PART 2 — METHODOLOGY SELECTION RULES
═══════════════════════════════════════════════════

Choose the methodology that matches what the code ACTUALLY implements.
Do NOT choose a methodology because it sounds more sophisticated.

VALID METHODOLOGIES:
  deterministic_simulation   — no randomness; same result every run
  stochastic_simulation      — randomness, but single run (no Monte Carlo)
  monte_carlo                — multiple independent runs with stochastic elements
  discrete_event_simulation  — event queue; time advances to next event
  system_dynamics            — differential equations; stocks and flows
  agent_based_simulation     — ONLY if ALL of the following are true:
                               (a) agents have their OWN state
                               (b) agents OBSERVE the world before deciding
                               (c) agents have OBJECTIVES driving their choices
                               (d) agents SELECT actions from a policy
                               (e) agents are not controlled by if-turn=N logic
  game_theoretic             — strategic interaction with payoff matrices
  hybrid                     — explicitly combines two or more of the above

If you implement rule-based decision logic with Monte Carlo, label it:
  "monte_carlo" with methodology_note "rule-based stochastic simulation"

NEVER label a rule-based simulation as agent_based_simulation.

═══════════════════════════════════════════════════
PART 3 — PROVENANCE RULES (CRITICAL)
═══════════════════════════════════════════════════

Every important parameter must be classified:

  USER_PROVIDED   — the user explicitly gave this value
  DERIVED         — mathematically calculated from USER_PROVIDED values
  MODEL_ASSUMPTION — the simulation invented this; user did not specify it

FORBIDDEN: Never describe a MODEL_ASSUMPTION as:
  - "a market average"
  - "an industry benchmark"
  - "a historical value"
  - "an empirical observation"
  - "a typical value"
unless an external data source was actually used.

In the assumptions list in the JSON output, every MODEL_ASSUMPTION must be
labelled explicitly with: "provenance": "MODEL_ASSUMPTION"

═══════════════════════════════════════════════════
PART 4 — ASSUMPTION HANDLING RULES
═══════════════════════════════════════════════════

ONE-TIME vs RECURRING:
  If the user says "X will happen next year", do NOT silently apply X
  every year for the full simulation horizon.

  Instead, either:
  a) Apply X only in Year 1 / Step 1, keep it constant thereafter, OR
  b) Create a clearly labelled MODEL_ASSUMPTION for future years, OR
  c) Model future years as an uncertainty range

  NEVER silently extrapolate a one-time assumption into a recurring rule.

CONTROL POLICY vs EMERGENCE:
  If a metric is tightly constrained by an explicit control rule
  (e.g. "hire lecturers whenever ratio > 22"), the metric's stability
  is CAUSED BY THE POLICY, not by natural system dynamics.

  Label such behavior: # POLICY_ENFORCED: ratio is controlled by hire_when_ratio_exceeds rule
  Do NOT let the analyst interpret policy-enforced stability as natural equilibrium.

═══════════════════════════════════════════════════
PART 5 — WORLD STATE STRUCTURE
═══════════════════════════════════════════════════

Structure your simulation around an explicit world state dict:

```python
world = {{
    "time": 0,
    "state": {{
        # USER_PROVIDED values go here with their actual values
        # MODEL_ASSUMPTION values go here with [MODEL_ASSUMPTION] comment
    }},
    "constraints": {{
        # hard limits the system must respect
    }},
    "events": [],      # log of everything that happens
    "decisions": [],   # log of every decision made and WHY
    "metrics": {{}}     # per-step metric snapshots
}}
```

═══════════════════════════════════════════════════
PART 6 — EVENT AND DECISION LOGGING
═══════════════════════════════════════════════════

Every time a significant state change occurs, log it:

```python
world["events"].append({{
    "time": step,
    "actor": "<who>",
    "event": "<what happened>",
    "reason": {{
        "<metric>": <current_value>,
        "<threshold>": <threshold_value>
    }},
    "state_before": {{"<key_vars>": <values>}},
    "action": {{"<action_type>": <parameters>}},
    "state_after": {{"<key_vars>": <values>}}
}})

world["decisions"].append({{
    "time": step,
    "actor": "<decision maker>",
    "decision": "<action taken>",
    "reason": {{
        "<triggering_metric>": <value>,
        "<threshold>": <value>,
        "<budget_available>": <value>
    }},
    "expected_effect": "<what this action is intended to achieve>"
}})
```

═══════════════════════════════════════════════════
PART 7 — MONTE CARLO REQUIREMENTS
═══════════════════════════════════════════════════

Run {num_runs} independent simulation trials.
For each trial, record per-trial outcomes in lists.
Include a fixed random seed: np.random.seed(42)

Track constraint violations per trial:
```python
constraint_violations = {{
    "<constraint_name>": []  # list of violation events per run
}}
```

═══════════════════════════════════════════════════
PART 8 — MANDATORY STRUCTURED OUTPUT
═══════════════════════════════════════════════════

At the END of the script, print this exact marker followed by JSON:

SIMULATION_RESULTS_JSON:
{{
  "metadata": {{
    "runs": <int>,
    "seed": <int>,
    "methodology": "<chosen methodology — must match actual implementation>",
    "methodology_note": "<one sentence description of what the code actually does>",
    "time_horizon": <int>,
    "time_unit": "<str>"
  }},
  "initial_conditions": {{
    "<param>": {{"value": <val>, "provenance": "USER_PROVIDED|DERIVED|MODEL_ASSUMPTION"}}
  }},
  "parameters": {{
    "<param>": {{"value": <val>, "provenance": "USER_PROVIDED|DERIVED|MODEL_ASSUMPTION"}}
  }},
  "assumptions": [
    {{
      "parameter": "<name>",
      "value": <value>,
      "provenance": "MODEL_ASSUMPTION|USER_PROVIDED|DERIVED",
      "justification": "<why this value was chosen>"
    }}
  ],
  "outcomes": {{
    "<metric_name>": [<list of per-run values>]
  }},
  "summary": {{
    "<metric>_mean": <float>,
    "<metric>_std": <float>
  }},
  "constraint_violations": {{
    "<constraint_name>": [<list of violation counts per run>]
  }},
  "events": [
    {{"time": <int>, "actor": "<str>", "event": "<str>", "reason": {{}}}}
  ],
  "decisions": [
    {{"time": <int>, "actor": "<str>", "decision": "<str>", "reason": {{}}}}
  ],
  "time_series": {{
    "<metric>": [<value per step, from a representative single run>]
  }},
  "scenario_comparison": {{}},
  "termination_reason": "<str>"
}}

═══════════════════════════════════════════════════
PART 9 — CODE QUALITY REQUIREMENTS
═══════════════════════════════════════════════════

1. Include np.random.seed(42) for reproducibility.
2. No undefined variables. All variables initialized before use.
3. Division by zero guards where required.
4. No impossible states (e.g. negative populations).
5. No plt.show() — save figures to file or omit plotting.
6. Also print human-readable output BEFORE the JSON block.
7. Libraries available: numpy, scipy, simpy, networkx, matplotlib,
   random, math, pandas, statistics

The code must be syntactically valid, complete, and executable without modification.
"""

    messages = [
        SystemMessage(
            content=(
                "You are an expert simulation architect. "
                "You build scientifically rigorous, evidence-based simulation models. "
                "You never label a simulation more sophisticated than it actually is."
            )
        ),
        HumanMessage(content=prompt)
    ]

    log_step_progress("Sending specification to Simulation Architect LLM...")
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time
    content = response.content

    code = _extract_code_block(content)
    num_lines = len(code.splitlines()) if code else 0

    log_step_success(
        "PLANNER",
        elapsed,
        f"Generated {num_lines} lines of simulation code"
    )

    return {
        "plan": content,
        "generated_code": code,
        "retry_count": retry_count,
        "critic_approved": False,
        "critic_feedback": "",
        "validation_report": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 — MODEL CRITIC
# ──────────────────────────────────────────────────────────────────────────────

def critic_node(state: SimulationState) -> SimulationState:
    """
    Model Critic: Reviews the generated simulation code against the specification.
    Flags methodology mislabelling, provenance violations, assumption extrapolation,
    determinism, and fairness issues — before execution.
    """
    log_step_start(
        "CRITIC",
        f"Reviewing code for scientific validity using '{settings.MODEL_NAME}'..."
    )

    spec = state.get("simulation_spec", "No specification available.")
    code = state.get("generated_code", "")

    prompt = f"""
You are the Model Critic of a Universal AI Simulation Engine.

Review the generated simulation code for scientific, methodological,
and structural problems BEFORE it is executed.

--------------------------------------------------
SIMULATION SPECIFICATION (with provenance labels)
--------------------------------------------------
{spec}

--------------------------------------------------
GENERATED SIMULATION CODE
--------------------------------------------------
```python
{code}
```

Review the code against ALL of the following criteria:

1. METHODOLOGY_LABEL
   - Does the declared methodology in SIMULATION_RESULTS_JSON match the actual implementation?
   - FLAG CRITICAL: if the code declares "agent_based_simulation" but agents do NOT have:
     (a) their own persistent state, (b) observation of world state,
     (c) objective-driven decision policy, (d) selection from action space.
   - FLAG CRITICAL: if a rule-based stochastic simulation is called "agent_based".
   - FLAG WARNING: if "monte_carlo" is used but only a single run is performed.

2. ASSUMPTION_PROVENANCE
   - Do MODEL_ASSUMPTION parameters have clear labels (comments, assumption list)?
   - FLAG CRITICAL: if any AI-invented parameter is described as "market average",
     "industry benchmark", "empirical value", "historical data", "typical value"
     without an actual external source.
   - FLAG WARNING: if the assumptions list in the JSON output is empty or missing.

3. ASSUMPTION_EXTRAPOLATION
   - Does the code check whether one-time user assumptions were silently converted
     to recurring multi-step rules?
   - FLAG CRITICAL: if the user said X happens "next year" but the code applies X
     every year for the full horizon (e.g. np.random.normal(0.20, 0.05) applied
     each step when user said "20% next year").
   - FLAG WARNING: if future-year assumptions are missing entirely (all years same).

4. STOCHASTICITY
   - Does the code use probabilistic outcomes (numpy.random, random)?
   - FLAG CRITICAL: if uncertain events are hardcoded to specific turn numbers.
   - FLAG CRITICAL: if the code is fully deterministic when the scenario is inherently uncertain.

5. AGENT_DECISION_MAKING
   - If methodology is agent_based, do agents observe state and select actions?
   - FLAG CRITICAL: if agent actions are determined by if-turn=N logic.

6. MONTE_CARLO_RUNS
   - Does the code run multiple independent trials?
   - FLAG CRITICAL: if num_runs < 2 for a stochastic scenario.
   - FLAG WARNING: if num_runs < 100.

7. FAIR_COMPARISONS
   - If multiple strategies are compared, do they start from identical initial conditions?
   - FLAG CRITICAL: if different strategies use materially different probability
     distributions without justification (e.g. one strategy has higher base growth rate).

8. CONTROL_POLICY_VS_EMERGENCE
   - When a metric is tightly controlled by an explicit rule (e.g. hire when ratio > 22),
     is the control policy labelled? (# POLICY_ENFORCED comment or equivalent)
   - FLAG WARNING: if policy-enforced stability exists without labelling.

9. EVENT_AND_DECISION_LOGGING
   - Does the code maintain an events list and decisions list?
   - FLAG WARNING: if events and decisions lists are absent.

10. STRUCTURED_OUTPUT
    - Does the code print a SIMULATION_RESULTS_JSON: block?
    - Does the JSON include: metadata, initial_conditions (with provenance),
      parameters (with provenance), assumptions (with provenance + justification),
      outcomes, constraint_violations, events, decisions?
    - FLAG CRITICAL: if the JSON block is missing entirely.
    - FLAG WARNING: if provenance fields are absent from parameters.

11. CODE_QUALITY
    - Are there potential runtime errors (undefined variables, division by zero)?
    - Does the code include np.random.seed(42) or equivalent?
    - FLAG WARNING: if no random seed is present.

Respond ONLY with a valid JSON object:
{{
  "approved": true | false,
  "severity": "pass" | "warning" | "critical",
  "declared_methodology": "<what the code declares>",
  "actual_methodology": "<what the code actually implements>",
  "methodology_match": true | false,
  "issues": [
    {{
      "criterion": "<criterion name>",
      "severity": "critical" | "warning" | "minor",
      "description": "<specific problem found in the code>",
      "suggested_fix": "<concrete how-to-fix instruction>"
    }}
  ],
  "summary": "<one paragraph overall assessment>"
}}

Approve (approved=true) only if there are NO critical issues.
Minor issues and warnings are acceptable — approve but list them.
"""

    messages = [
        SystemMessage(
            content=(
                "You are an expert computational simulation scientist and code reviewer. "
                "You evaluate simulation code for scientific rigor, methodology accuracy, "
                "provenance honesty, and assumption correctness. "
                "Respond only with valid JSON."
            )
        ),
        HumanMessage(content=prompt)
    ]

    log_step_progress("Sending code to LLM for scientific validity review...")
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time

    raw = response.content or ""
    feedback_str = _strip_json_fences(raw)

    approved = False

    try:
        review = json.loads(feedback_str)
        approved = bool(review.get("approved", False))
        severity = review.get("severity", "unknown")
        issues = review.get("issues", [])
        issues_count = len(issues)
        declared = review.get("declared_methodology", "?")
        actual = review.get("actual_methodology", "?")
        mismatch = not review.get("methodology_match", True)

        if mismatch:
            log_step_warning(
                "CRITIC",
                f"Methodology mismatch: declared='{declared}' vs actual='{actual}'"
            )

        if approved:
            log_step_success(
                "CRITIC",
                elapsed,
                f"Code approved — severity={severity}, issues noted={issues_count}"
            )
        else:
            critical_issues = [i for i in issues if i.get("severity") == "critical"]
            log_step_warning(
                "CRITIC",
                f"Code NOT approved — {len(critical_issues)} critical, "
                f"{issues_count} total issues ({elapsed:.2f}s)"
            )
            for issue in critical_issues:
                log_step_progress(
                    f"CRITICAL [{issue.get('criterion')}]: {issue.get('description')}"
                )

    except json.JSONDecodeError:
        log_step_warning(
            "CRITIC",
            f"Could not parse critic response as JSON — defaulting to approved=True ({elapsed:.2f}s)"
        )
        approved = True
        feedback_str = raw

    return {
        "critic_approved": approved,
        "critic_feedback": feedback_str,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 4 — EXECUTOR
# ──────────────────────────────────────────────────────────────────────────────

def executor_node(state: SimulationState) -> SimulationState:
    """
    Executor: Runs the AI-generated code inside the sandbox environment.
    Also extracts the structured SIMULATION_RESULTS_JSON block from stdout.
    """
    code = state.get("generated_code", "")
    num_lines = len(code.splitlines()) if code else 0
    log_step_start("EXECUTOR", f"Running simulation code ({num_lines} lines) in sandbox...")

    start_time = time.time()
    stdout, stderr = execute_python_code(code)
    elapsed = time.time() - start_time

    structured_results = ""
    if stdout:
        parsed = _extract_json_block(stdout)
        if parsed:
            try:
                structured_results = json.dumps(parsed, indent=2)
                log_step_progress(
                    f"Structured results block extracted ({len(structured_results)} chars)"
                )
            except Exception:
                structured_results = ""
        else:
            log_step_progress(
                "No SIMULATION_RESULTS_JSON block found — will rely on raw stdout"
            )

    if stderr:
        last_error_line = stderr.strip().splitlines()[-1] if stderr.strip() else "Unknown error"
        log_step_warning("EXECUTOR", f"Script produced errors: {last_error_line}")
    else:
        out_lines = len(stdout.strip().splitlines()) if stdout.strip() else 0
        log_step_success(
            "EXECUTOR",
            elapsed,
            f"Execution completed — {out_lines} stdout lines, "
            f"structured_results={'yes' if structured_results else 'no'}"
        )

    return {
        "execution_output": stdout,
        "execution_error": stderr,
        "structured_results": structured_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 5 — RESULT VALIDATOR (pure Python — no LLM call)
# ──────────────────────────────────────────────────────────────────────────────

def validator_node(state: SimulationState) -> SimulationState:
    """
    Result Validator: Pure Python post-execution consistency checks.
    No LLM call. Cross-checks that the structured output is internally
    consistent and scientifically sound. Produces a validation_report.
    """
    log_step_start(
        "VALIDATOR",
        "Running post-execution consistency checks (no LLM)..."
    )
    start_time = time.time()

    checks: list[dict] = []

    def _check(name: str, passed: bool, severity: str, detail: str):
        status = "PASS" if passed else severity
        checks.append({
            "check": name,
            "status": status,
            "detail": detail
        })
        if passed:
            log_step_progress(f"✓ {name}")
        else:
            log_step_progress(f"{'⚠' if severity == 'WARN' else '✗'} {name}: {detail}")

    # ── Check 1: Execution success ────────────────────────────────────
    has_error = bool(state.get("execution_error", "").strip())
    _check(
        "execution_success",
        not has_error,
        "FAIL",
        "Execution produced stderr" if has_error else "Clean execution"
    )

    # ── Check 2: Structured results present ──────────────────────────
    structured_raw = state.get("structured_results", "")
    has_structured = bool(structured_raw.strip())
    _check(
        "structured_results_present",
        has_structured,
        "WARN",
        "SIMULATION_RESULTS_JSON block missing from output" if not has_structured
        else "Structured JSON block extracted"
    )

    data: dict = {}
    if has_structured:
        try:
            data = json.loads(structured_raw)
        except json.JSONDecodeError:
            _check("structured_results_parseable", False, "FAIL",
                   "SIMULATION_RESULTS_JSON could not be parsed as JSON")

    # ── Check 3: Required metadata fields ────────────────────────────
    metadata = data.get("metadata", {})
    has_runs = "runs" in metadata
    has_methodology = "methodology" in metadata and bool(metadata.get("methodology"))
    has_seed = "seed" in metadata
    _check("metadata_runs", has_runs, "WARN",
           f"metadata.runs = {metadata.get('runs', 'MISSING')}")
    _check("metadata_methodology", has_methodology, "WARN",
           f"metadata.methodology = '{metadata.get('methodology', 'MISSING')}'")
    _check("metadata_seed", has_seed, "WARN",
           "No random seed recorded — results not reproducible" if not has_seed
           else f"seed = {metadata.get('seed')}")

    # ── Check 4: Methodology is a known value ────────────────────────
    known_methods = {
        "deterministic_simulation", "stochastic_simulation", "monte_carlo",
        "discrete_event_simulation", "system_dynamics", "agent_based_simulation",
        "game_theoretic", "hybrid"
    }
    declared_method = metadata.get("methodology", "")
    method_known = declared_method in known_methods
    _check("methodology_known_value", method_known, "WARN",
           f"'{declared_method}' is not in the approved methodology taxonomy"
           if not method_known else f"methodology='{declared_method}'")

    # ── Check 5: Outcomes block is non-empty with finite values ───────
    outcomes = data.get("outcomes", {})
    has_outcomes = isinstance(outcomes, dict) and len(outcomes) > 0
    _check("outcomes_present", has_outcomes, "WARN",
           "No outcome metrics in structured results" if not has_outcomes
           else f"{len(outcomes)} outcome metric(s) found")

    non_finite_metrics = []
    if has_outcomes:
        for metric, vals in outcomes.items():
            if isinstance(vals, list):
                bad = [v for v in vals if not isinstance(v, (int, float)) or _is_nan(v)]
                if bad:
                    non_finite_metrics.append(metric)
    _check("outcomes_finite_values", len(non_finite_metrics) == 0, "WARN",
           f"Non-finite values in: {non_finite_metrics}" if non_finite_metrics
           else "All outcome values are finite")

    # ── Check 6: Assumptions list non-empty with provenance ───────────
    assumptions = data.get("assumptions", [])
    has_assumptions = isinstance(assumptions, list) and len(assumptions) > 0
    _check("assumptions_present", has_assumptions, "WARN",
           "assumptions list is empty — model parameters not documented"
           if not has_assumptions else f"{len(assumptions)} assumption(s) documented")

    missing_provenance = []
    if has_assumptions:
        for a in assumptions:
            if isinstance(a, dict) and not a.get("provenance"):
                missing_provenance.append(a.get("parameter", "?"))
    _check("assumptions_have_provenance", len(missing_provenance) == 0, "WARN",
           f"Missing provenance on: {missing_provenance}" if missing_provenance
           else "All assumptions have provenance labels")

    # ── Check 7: Constraint violations block ─────────────────────────
    cv = data.get("constraint_violations", {})
    has_cv = isinstance(cv, dict) and len(cv) > 0
    _check("constraint_violations_tracked", has_cv, "WARN",
           "No constraint_violations block — can't compute violation probability"
           if not has_cv else f"{len(cv)} constraint(s) tracked")

    # ── Check 8: Events and decisions logged ─────────────────────────
    events = data.get("events", [])
    decisions = data.get("decisions", [])
    has_events = isinstance(events, list) and len(events) > 0
    has_decisions = isinstance(decisions, list) and len(decisions) > 0
    _check("events_logged", has_events, "WARN",
           "No events logged — analyst cannot explain what happened"
           if not has_events else f"{len(events)} event(s) logged")
    _check("decisions_logged", has_decisions, "WARN",
           "No decisions logged — analyst cannot explain WHY things changed"
           if not has_decisions else f"{len(decisions)} decision(s) logged")

    # ── Check 9: Initial conditions have provenance ───────────────────
    init_conds = data.get("initial_conditions", {})
    ic_missing_prov = []
    for k, v in init_conds.items():
        if isinstance(v, dict) and not v.get("provenance"):
            ic_missing_prov.append(k)
    _check("initial_conditions_provenance", len(ic_missing_prov) == 0, "WARN",
           f"Missing provenance on initial conditions: {ic_missing_prov}"
           if ic_missing_prov else "All initial conditions have provenance labels")

    # ── Check 10: Number of runs is sufficient ────────────────────────
    runs_val = metadata.get("runs", 0)
    runs_ok = isinstance(runs_val, int) and runs_val >= 100
    _check("sufficient_runs", runs_ok, "WARN",
           f"Only {runs_val} runs — recommend ≥ 100 for statistical reliability"
           if not runs_ok else f"{runs_val} runs — statistically adequate")

    # ── Summary ───────────────────────────────────────────────────────
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    passes = [c for c in checks if c["status"] == "PASS"]

    elapsed = time.time() - start_time

    validation_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 3),
        "summary": {
            "total_checks": len(checks),
            "passed": len(passes),
            "warnings": len(warns),
            "failures": len(fails),
            "overall": "FAIL" if fails else ("WARN" if warns else "PASS")
        },
        "checks": checks
    }

    overall = validation_report["summary"]["overall"]
    log_step_success(
        "VALIDATOR",
        elapsed,
        f"Validation complete — {len(passes)} pass, {len(warns)} warn, {len(fails)} fail "
        f"(overall: {overall})"
    )

    return {"validation_report": json.dumps(validation_report, indent=2)}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 6 — STATISTICAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def stats_node(state: SimulationState) -> SimulationState:
    """
    Stats: Computes Python-native statistics from the structured simulation results.
    Mean, median, std, percentiles, constraint violation probability, event
    frequency — calculated in Python, not by the LLM.
    """
    log_step_start(
        "STATS",
        "Computing statistical summary from structured simulation results..."
    )
    start_time = time.time()

    structured = state.get("structured_results", "")
    if not structured:
        log_step_warning("STATS", "No structured results — skipping statistical analysis")
        return {"statistical_summary": "No structured results available for statistical analysis."}

    try:
        data = json.loads(structured)
    except json.JSONDecodeError:
        log_step_warning("STATS", "Could not parse structured_results as JSON")
        return {"statistical_summary": "Could not parse structured results."}

    lines: list[str] = ["## Statistical Analysis (Python-computed)\n"]

    # ── Metadata ─────────────────────────────────────────────────────
    metadata = data.get("metadata", {})
    if metadata:
        lines.append("### Experiment Metadata")
        for k, v in metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    # ── Methodology note ─────────────────────────────────────────────
    methodology_note = metadata.get("methodology_note", "")
    if methodology_note:
        lines.append(f"> **Methodology note:** {methodology_note}\n")

    # ── Simulation-computed summary ───────────────────────────────────
    sim_summary = data.get("summary", {})
    if sim_summary:
        lines.append("### Simulation-Computed Summary")
        for k, v in sim_summary.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # ── Per-metric statistics from outcomes lists ─────────────────────
    outcomes = data.get("outcomes", {})
    if isinstance(outcomes, dict):
        computed_any = False
        for metric, values in outcomes.items():
            if not isinstance(values, list):
                continue
            nums = [x for x in values if isinstance(x, (int, float)) and not _is_nan(x)]
            if len(nums) < 2:
                continue

            computed_any = True
            try:
                mean_v = _stats.mean(nums)
                median_v = _stats.median(nums)
                std_v = _stats.stdev(nums) if len(nums) > 1 else 0.0
                min_v = min(nums)
                max_v = max(nums)
                sorted_nums = sorted(nums)
                n = len(sorted_nums)
                p10 = sorted_nums[max(0, int(0.10 * n) - 1)]
                p25 = sorted_nums[max(0, int(0.25 * n) - 1)]
                p75 = sorted_nums[min(n - 1, int(0.75 * n))]
                p90 = sorted_nums[min(n - 1, int(0.90 * n))]

                lines.append(f"### {metric.replace('_', ' ').title()} (n={n})")
                lines.append("| Statistic | Value |")
                lines.append("|-----------|-------|")
                lines.append(f"| Mean      | {mean_v:.4f} |")
                lines.append(f"| Median    | {median_v:.4f} |")
                lines.append(f"| Std Dev   | {std_v:.4f} |")
                lines.append(f"| Min       | {min_v:.4f} |")
                lines.append(f"| Max       | {max_v:.4f} |")
                lines.append(f"| P10       | {p10:.4f} |")
                lines.append(f"| P25       | {p25:.4f} |")
                lines.append(f"| P75       | {p75:.4f} |")
                lines.append(f"| P90       | {p90:.4f} |")

                # Binary success/failure probability
                unique = set(nums)
                if unique.issubset({0, 1, 0.0, 1.0}):
                    prob_success = mean_v
                    lines.append(f"| P(success) | {prob_success:.1%} |")
                    lines.append(f"| P(failure) | {1 - prob_success:.1%} |")

                lines.append("")
            except Exception as e:
                log_step_progress(f"Stats failed for '{metric}': {e}")

        if not computed_any:
            lines.append("*No numeric list metrics found in outcomes block.*\n")

    # ── Constraint violation probabilities ────────────────────────────
    constraint_violations = data.get("constraint_violations", {})
    if isinstance(constraint_violations, dict) and constraint_violations:
        lines.append("### Constraint Violation Analysis (Python-computed)")
        lines.append("| Constraint | Mean Violations/Run | P(≥1 violation) | Max Violations |")
        lines.append("|------------|--------------------:|----------------:|---------------:|")
        for constraint_name, violation_counts in constraint_violations.items():
            if not isinstance(violation_counts, list) or len(violation_counts) == 0:
                continue
            nums = [x for x in violation_counts if isinstance(x, (int, float))]
            if not nums:
                continue
            mean_v = _stats.mean(nums)
            prob_any = sum(1 for x in nums if x > 0) / len(nums)
            max_v = max(nums)
            lines.append(
                f"| {constraint_name} | {mean_v:.2f} | {prob_any:.1%} | {int(max_v)} |"
            )
        lines.append("")

    # ── Event frequency statistics ────────────────────────────────────
    events = data.get("events", [])
    if isinstance(events, list) and events:
        event_types: dict[str, int] = {}
        for ev in events:
            if isinstance(ev, dict):
                evt = ev.get("event", "unknown")
                event_types[evt] = event_types.get(evt, 0) + 1
        if event_types:
            lines.append("### Event Frequency (from representative run)")
            lines.append("| Event Type | Count |")
            lines.append("|------------|------:|")
            for evt, count in sorted(event_types.items(), key=lambda x: -x[1]):
                lines.append(f"| {evt} | {count} |")
            lines.append("")

    # ── Decision frequency ────────────────────────────────────────────
    decisions = data.get("decisions", [])
    if isinstance(decisions, list) and decisions:
        decision_types: dict[str, int] = {}
        for d in decisions:
            if isinstance(d, dict):
                dt = d.get("decision", "unknown")
                decision_types[dt] = decision_types.get(dt, 0) + 1
        if decision_types:
            lines.append("### Decision Frequency (from representative run)")
            lines.append("| Decision | Count |")
            lines.append("|----------|------:|")
            for dt, count in sorted(decision_types.items(), key=lambda x: -x[1]):
                lines.append(f"| {dt} | {count} |")
            lines.append("")

    # ── Scenario comparison ───────────────────────────────────────────
    scenario_comparison = data.get("scenario_comparison", {})
    if isinstance(scenario_comparison, dict) and scenario_comparison:
        lines.append("### Scenario Comparison")
        # Collect all metric names
        all_metrics: set[str] = set()
        for sc_data in scenario_comparison.values():
            if isinstance(sc_data, dict):
                all_metrics.update(sc_data.keys())
        if all_metrics:
            metric_list = sorted(all_metrics)
            header = "| Scenario | " + " | ".join(metric_list) + " |"
            separator = "|----------|" + "|".join(["-" * max(len(m), 6) for m in metric_list]) + "|"
            lines.append(header)
            lines.append(separator)
            for sc_name, sc_data in scenario_comparison.items():
                if isinstance(sc_data, dict):
                    row_vals = []
                    for m in metric_list:
                        v = sc_data.get(m, "—")
                        row_vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                    lines.append(f"| {sc_name} | " + " | ".join(row_vals) + " |")
            lines.append("")

    # ── Assumptions with provenance ───────────────────────────────────
    assumptions = data.get("assumptions", [])
    if assumptions:
        lines.append("### Model Assumptions (from simulation — with provenance)")
        lines.append("| Parameter | Value | Provenance | Justification |")
        lines.append("|-----------|-------|------------|---------------|")
        for a in assumptions:
            if isinstance(a, dict):
                param = a.get("parameter", "?")
                val = a.get("value", "?")
                prov = a.get("provenance", "?")
                just = a.get("justification", "—")
                lines.append(f"| {param} | {val} | **{prov}** | {just} |")
            elif isinstance(a, str):
                lines.append(f"| — | — | ? | {a} |")
        lines.append("")

    # ── Termination ───────────────────────────────────────────────────
    term = data.get("termination_reason", "")
    if term:
        lines.append(f"**Termination reason:** {term}\n")

    elapsed = time.time() - start_time
    summary_text = "\n".join(lines)
    log_step_success(
        "STATS",
        elapsed,
        f"Statistical summary generated ({len(summary_text)} chars, "
        f"includes constraint violation probabilities + event frequency)"
    )

    return {"statistical_summary": summary_text}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 7 — ANALYST (19-section scientific report)
# ──────────────────────────────────────────────────────────────────────────────

def analyst_node(state: SimulationState) -> SimulationState:
    """
    Analyst: Interprets simulation results and generates a rigorous
    19-section scientific research report.
    """
    log_step_start(
        "ANALYST",
        f"Generating 19-section scientific report using '{settings.MODEL_NAME}'..."
    )

    statistical_summary = state.get("statistical_summary", "")
    validation_report = state.get("validation_report", "")

    stats_section = f"""
--------------------------------------------------
PRE-COMPUTED STATISTICAL EVIDENCE
(Python-computed — treat as ground truth; do not recalculate)
--------------------------------------------------
{statistical_summary}
""" if statistical_summary else ""

    validation_section = f"""
--------------------------------------------------
RESULT VALIDATOR REPORT
(automated consistency checks — address any failures)
--------------------------------------------------
{validation_report}
""" if validation_report else ""

    prompt = f"""
You are the Scientific Analyst of a Universal AI Simulation Engine.

Produce a rigorous, evidence-based 19-section research report.

═══════════════════════════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE
═══════════════════════════════════════════════════

1. EVIDENCE ONLY
   Every claim must be traceable to the generated code, execution output,
   or pre-computed statistics. Do not invent facts.

2. PROVENANCE LABELLING
   Label every important parameter with one of:
   [USER_PROVIDED] [DERIVED] [MODEL_ASSUMPTION] [SIMULATION_OUTPUT] [AI_INTERPRETATION]

   Never describe a MODEL_ASSUMPTION as a "market average", "industry benchmark",
   "historical value", "empirical observation", or "typical value" unless an
   external verified data source was actually used.

3. METHODOLOGY ACCURACY
   Use the declared methodology from metadata.methodology.
   If the validator or critic flagged a mismatch, acknowledge it.
   Do NOT call a simulation "agent-based" unless the code contains autonomous
   agents with their own state, observations, objectives, and decision policies.
   Do NOT call outcome "emergent" if it is directly enforced by a control rule.

4. CONTROL POLICY vs EMERGENCE
   If a metric is tightly constrained by an explicit rule (e.g. hire when ratio > 22),
   identify the control policy as the CAUSE of the metric's stability.
   Do not interpret policy-enforced stability as natural equilibrium.
   State explicitly: "The narrow distribution is a consequence of the
   [policy name] control rule, not emergent system dynamics."

5. ONE-TIME vs RECURRING ASSUMPTIONS
   If the user specified a one-time assumption (e.g. "20% growth next year"),
   identify clearly whether the simulation applied it once or every step.
   If it was applied every step, flag this as a modelling limitation.

6. PRE-COMPUTED STATISTICS
   Use ONLY the statistics in the PRE-COMPUTED STATISTICAL EVIDENCE section.
   Do NOT recalculate or invent different figures.

7. INSUFFICIENT EVIDENCE
   If the evidence is insufficient to support a conclusion, state:
   "Insufficient evidence from the current simulation."

8. SIMULATION vs REALITY
   Use language such as "Under the modeled assumptions..."
   NEVER present simulation outputs as real-world predictions.

═══════════════════════════════════════════════════
EVIDENCE PROVIDED
═══════════════════════════════════════════════════

--------------------------------------------------
ORIGINAL USER SCENARIO
--------------------------------------------------
{state.get('user_prompt', '')}

--------------------------------------------------
SIMULATION SPECIFICATION (with provenance labels)
--------------------------------------------------
{state.get('simulation_spec', '')}
{stats_section}{validation_section}
--------------------------------------------------
GENERATED SIMULATION CODE
--------------------------------------------------

```python
{state.get('generated_code', '')}
```

--------------------------------------------------
EXECUTION OUTPUT
--------------------------------------------------
{state.get('execution_output', '')}

--------------------------------------------------
EXECUTION ERRORS
--------------------------------------------------
{state.get('execution_error', '')}

═══════════════════════════════════════════════════
REPORT STRUCTURE — PRODUCE ALL 19 SECTIONS
═══════════════════════════════════════════════════

# Simulation Experiment Analysis Report

## Executive Summary
2–3 sentences: what was simulated, the most important finding, and the key caveat.

## Simulation Objective
What research question does this simulation investigate?
What decision is the simulation designed to inform?

## Initial Conditions
List starting state values.
For each: label as [USER_PROVIDED], [DERIVED], or [MODEL_ASSUMPTION].
Note any discrepancies between the declared initial conditions and what appears in the code.

## Entities and State Variables
What entities exist in the simulation?
What state variables track their condition?
How do state variables evolve?

## Model Parameters
Table: parameter | value | provenance | effect on simulation.
Every parameter must have a provenance label.
Do NOT describe MODEL_ASSUMPTION parameters as "typical" or "market" values.

## Assumptions and Parameter Provenance
Explicit table of all MODEL_ASSUMPTION parameters.
For each: what value was chosen, why, and what effect it has on results.
Flag any assumption that significantly constrains the outcome.

## Simulation Methodology
Accurately describe what the code actually implements.
Do NOT call it agent-based unless the code contains observation→policy→action agents.
If it is a rule-based stochastic simulation, say so.
Explain: state transitions, decision rules, randomness sources, time evolution.

## Decision Policies / Agent Behaviors
What decisions are made during the simulation?
What triggers each decision? (Give the actual threshold or condition from the code.)
Label decision outcomes as POLICY_ENFORCED where applicable.
Example: "When student/lecturer ratio > [threshold] [MODEL_ASSUMPTION], the
administration hires [N] lecturers. This is a POLICY_ENFORCED outcome."

## Simulation Dynamics
How did the system evolve over the simulation horizon?
What feedback loops exist?
What events typically occurred in what order?
Only discuss dynamics that are actually represented in the code.

## Event Timeline
Summarise the most significant events from the events log.
If no event log was produced, note this as a limitation.
Format: Time | Actor | Event | Reason

## Statistical Results
Present ALL statistics from the pre-computed evidence section.
Use tables. Do not invent new figures.
Include: mean, median, std, P10, P25, P75, P90 for all outcome metrics.

## Constraint / Failure Analysis
Present constraint violation probabilities from the pre-computed stats.
P(constraint violated) for each tracked constraint.
P(budget < 0), P(ratio > threshold), P(utilization > limit), etc.
If constraint violations were not tracked, note this as a limitation.

## Scenario Comparison
If multiple strategies or scenarios were simulated:
- Present the comparison table from pre-computed stats.
- Verify that scenarios started from identical initial conditions.
- Flag any asymmetry in model assumptions between scenarios.
If only one scenario was simulated, note this as a limitation and
explain what alternative scenarios would be informative.

## Sensitivity and Uncertainty Analysis
What parameters most affect the outcome?
If sensitivity analysis was not performed:
- Identify the top 3 parameters that would most affect results if changed.
- Recommend what sensitivity analysis should be run.
Note this is speculative unless the simulation actually varied parameters.

## Key Findings
For each finding:
- **Observed [SIMULATION_OUTPUT]:** What the simulation directly produced (with numbers).
- **Interpretation [AI_INTERPRETATION]:** What that observation means.
- **Caveat:** What the finding depends on (assumptions, control policies, etc.).

## Decision Analysis
For each available decision or strategy:
- Benefits (under the modeled assumptions)
- Risks and downside outcomes
- Trade-offs
- Conditions under which it is the better choice
If no universally optimal strategy exists, explain why with evidence.

## Limitations and Unmodeled Factors
Explicitly list:
- Variables not modeled
- Simplified assumptions
- One-time assumptions silently applied as recurring (if detected)
- Parameters described as MODEL_ASSUMPTION that could significantly affect results
- Methodology limitations (e.g. rule-based rather than genuinely adaptive)
- Missing sensitivity / scenario analysis

## Scientific Conclusion
What does the simulation actually demonstrate?
Use language: "Under the assumptions represented in the simulation..."
NOT: "This will happen in reality."
Distinguish: simulation evidence vs real-world prediction.

## Recommendations
Each recommendation must:
- State the assumption(s) it depends on.
- State what evidence supports it.
- State what would change the recommendation.
If evidence is insufficient for a strong recommendation, say so explicitly.

═══════════════════════════════════════════════════
FINAL QUALITY CHECK (before writing the report)
═══════════════════════════════════════════════════

Answer YES or NO to each. If NO, fix before producing the report:

- Are all numerical claims from pre-computed stats or execution output?
- Are all important parameters labelled with provenance?
- Is the methodology label consistent with what the code actually does?
- If any metric is tightly policy-controlled, is the control mechanism identified?
- Are strategy comparisons based on comparable initial conditions?
- Are one-time assumptions distinguished from recurring ones?
- Are limitations explicitly stated?
- Are recommendations conditional on assumptions?
- Did you avoid describing MODEL_ASSUMPTION values as empirical facts?
- Did you avoid recalculating statistics that are already pre-computed?
"""

    messages = [
        SystemMessage(
            content=(
                "You are an expert scientific simulation analyst. "
                "You prioritize reproducibility, statistical correctness, "
                "methodology accuracy, provenance honesty, and evidence-based reasoning. "
                "You never let the sophistication of the report exceed the "
                "sophistication of the actual simulation model."
            )
        ),
        HumanMessage(content=prompt)
    ]

    log_step_progress(
        "Sending spec, pre-computed stats, validation report, and code to LLM..."
    )

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time

    report = response.content or ""
    word_count = len(report.split())

    log_step_success(
        "ANALYST",
        elapsed,
        f"19-section research report generated ({word_count} words)"
    )

    return {"analysis_report": report}


# ──────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGES
# ──────────────────────────────────────────────────────────────────────────────

def should_critic_retry(state: SimulationState) -> str:
    """
    Routes after the critic node.
    If critic rejected the code and retries remain, send back to planner.
    Otherwise proceed to execution.
    """
    approved = state.get("critic_approved", True)
    critic_retry_count = state.get("critic_retry_count", 0)
    max_critic_retries = 1

    if not approved and critic_retry_count < max_critic_retries:
        new_count = critic_retry_count + 1
        state["critic_retry_count"] = new_count
        log_step_warning(
            "ROUTER",
            f"Critic rejected code. Sending back to Planner for revision "
            f"({new_count}/{max_critic_retries})..."
        )
        return "rework"

    if not approved:
        log_step_warning(
            "ROUTER",
            f"Critic rejected code but max critic retries ({max_critic_retries}) reached. "
            "Proceeding to execution anyway..."
        )
    else:
        log_step_progress("Critic approved code. Routing to Executor...")

    return "proceed"


def should_retry(state: SimulationState) -> str:
    """
    Conditional Edge: If execution fails and retries < 2, retry planning.
    Otherwise proceed to validation + statistical analysis.
    """
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
            f"Max retries ({max_retries}) reached. Proceeding to validation..."
        )
    else:
        log_step_progress(
            "Execution validated. Routing to Result Validator..."
        )
    return "proceed"