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
from backend.utils.chart_generator import generate_scientific_chart
from backend.utils.logger import (
    log_step_start,
    log_step_progress,
    log_step_success,
    log_step_warning,
    log_retry_trigger,
)

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL,
    temperature=0.2
)

_EMPTY_OUTPUT_PHRASES = (
    "model output must contain either output text or tool calls",
    "candidate content was empty",
    "finish_reason: safety",
    "response was filtered",
)

def llm_invoke_safe(messages: list, max_retries: int = 2, node_name: str = "LLM"):
    """
    Safe wrapper around llm.invoke() that retries on empty-content responses.
    The Gemini/OpenAI API can return an empty response when the model output
    is blocked by a safety filter or when the model fails to generate content.
    This wrapper catches those cases and retries up to `max_retries` times.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):  # attempts: 1 .. max_retries+1
        try:
            response = llm.invoke(messages)
            # Some API wrappers return a response object with empty content
            # instead of raising an exception.
            if not getattr(response, "content", None):
                raise ValueError(
                    "model output error: model output must contain either "
                    "output text or tool calls, these cannot both be empty"
                )
            return response
        except Exception as exc:
            err_lower = str(exc).lower()
            is_empty_output = any(p in err_lower for p in _EMPTY_OUTPUT_PHRASES)
            if is_empty_output and attempt <= max_retries:
                log_step_warning(
                    node_name,
                    f"Empty/blocked model output on attempt {attempt}/{max_retries + 1}. "
                    f"Retrying... ({exc})"
                )
                last_exc = exc
                time.sleep(1.5 * attempt)  # brief back-off
                continue
            raise  # non-retryable error, or retries exhausted
    # Should not reach here, but satisfy linters
    raise last_exc  # type: ignore[misc]

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


def _detect_language(text: str) -> str:
    """
    Detect the primary natural language of a text prompt.
    Supports Asian scripts, Cyrillic, Arabic, and major Western languages.
    """
    if not text or not text.strip():
        return "English"

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


def _get_language(state: SimulationState) -> str:
    """Retrieve target language from state or auto-detect from user_prompt."""
    lang = state.get("language")
    if lang and lang.lower() != "auto":
        return lang
    return _detect_language(state.get("user_prompt", ""))


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences from a JSON response."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


# ── Comparative / investigation trigger keywords ───────────────────────────────
_HYPOTHESIS_TRIGGERS = (
    "compare", "versus", " vs ", "vs.", "which is better", "which performs",
    "test whether", "does x cause", "normal vs", "fat-tail", "fat tail",
    "fixed vs", "adaptive vs", "hypothesis", "causal", "investigate whether",
    "model comparison", "regime comparison", "scenario comparison",
)


def _classify_simulation_mode(state: "SimulationState") -> str:  # type: ignore[name-defined]
    """
    Pure-Python gate — no LLM call.
    Returns "HYPOTHESIS" when the prompt explicitly requests comparative or
    causal investigation; returns "DIRECT_MC" for all other prompts.

    Decision order:
      1. Explicit simulation_mode already in state (carry-through on retry).
      2. Keyword scan of user_prompt.
      3. hypothesis_analysis.multi_branch_required == true AND >=2 hypotheses
         with posterior > 15%  (strong multi-branch signal from updater).
      4. Default → DIRECT_MC.
    """
    # 1. Carry-through on retry
    existing = state.get("simulation_mode", "")
    if existing in ("DIRECT_MC", "HYPOTHESIS"):
        return existing

    # 2. Keyword scan
    prompt_lower = state.get("user_prompt", "").lower()
    if any(kw in prompt_lower for kw in _HYPOTHESIS_TRIGGERS):
        return "HYPOTHESIS"

    # 3. Hypothesis analysis signal
    hyp_str = state.get("hypothesis_analysis", "")
    if hyp_str:
        try:
            hyp = json.loads(hyp_str)
            if hyp.get("multi_branch_required") is True:
                high_posterior = [
                    h for h in hyp.get("hypotheses", [])
                    if isinstance(h.get("posterior"), (int, float))
                    and h["posterior"] > 0.15
                ]
                if len(high_posterior) >= 2:
                    return "HYPOTHESIS"
        except (json.JSONDecodeError, TypeError):
            pass

    return "DIRECT_MC"


def scenario_parser_node(state: SimulationState) -> SimulationState: #node 1a
    """
    Scenario Parser: Converts the natural-language user prompt into a
    structured JSON simulation specification with provenance labels.
    """
    target_lang = _get_language(state)
    log_step_start(
        "SCENARIO PARSER",
        f"Extracting structured simulation specification [{target_lang}] using '{settings.MODEL_NAME}'..."
    )

    lang_instruction = f"""
LANGUAGE REQUIREMENT (MANDATORY — HIGHEST PRIORITY):
The user scenario is written in '{target_lang}'.
You MUST write all natural-language string values in your JSON output (domain, objective, methodology_note, hypothesis names and descriptions, entity names and objectives, actions, outcomes, constraint names and descriptions, notes, and assumption lists) in the EXACT SAME LANGUAGE as the user scenario ({target_lang}).
JSON schema keys (such as "domain", "objective", "methodology_hint", "competing_hypotheses", "entities", "initial_state", "parameters", "actions", "outcomes", "constraints", "stochastic_variables", "provenance", etc.) and enum tokens (such as "USER_PROVIDED", "MODEL_ASSUMPTION", "OBSERVED", "DERIVED", "AI_INFERENCE", "agent", "environment", "resource", "process") MUST REMAIN EXACTLY IN ENGLISH as defined in the schema for program compatibility.
"""

    prompt = f"""
You are a Simulation Scenario Analyst for a Universal AI Simulation Engine.

Your task is to read the user's natural-language scenario and extract a
structured simulation specification as a JSON object.
{lang_instruction}

CRITICAL: Classify every parameter with a provenance label using the FULL taxonomy:
  USER_PROVIDED    — the user explicitly stated this value
  OBSERVED         — directly measured or logged from a data source
  DERIVED          — calculated mathematically from USER_PROVIDED or OBSERVED values
  MODEL_ASSUMPTION — the simulation will need to invent this; not stated by the user
  SIMULATED        — produced as output by the simulation itself
  AI_INFERENCE     — inferred by the AI from context, not explicitly stated
  UNKNOWN          — unclear; must be flagged as a gap

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
  "competing_hypotheses": [
    {{"id": "H1", "name": "<short name>", "description": "<one sentence causal story>"}}
  ],
  "entities": [
    {{"name": "<str>", "type": "agent|environment|resource|process", "objective": "<str or null>"}}
  ],
  "initial_state": {{
    "<param_name>": {{"value": <value_or_null>, "provenance": "USER_PROVIDED|OBSERVED|MODEL_ASSUMPTION|DERIVED|AI_INFERENCE|UNKNOWN", "note": "<optional note>"}}
  }},
  "parameters": {{
    "<param_name>": {{"value": <value_or_null>, "provenance": "USER_PROVIDED|OBSERVED|MODEL_ASSUMPTION|DERIVED|AI_INFERENCE|UNKNOWN", "note": "<optional note>"}}
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
- Use AI_INFERENCE for things the AI deduces from context (e.g. 'employee asleep implies manual action unlikely').
- If the user says \"enrollment will increase 20% next year\", classify that as a
  one_time_assumption, NOT a recurring assumption.
- Never invent \"empirical\" values — label everything the simulation invents as MODEL_ASSUMPTION.
- methodology_hint must reflect what the code will ACTUALLY implement, not what sounds impressive.
  Do NOT choose agent_based_simulation unless the scenario genuinely requires autonomous agents
  with their own state, observations, objectives, and decision policies.
- competing_hypotheses: list ALL plausible causal explanations for the scenario outcome,
  not just the most likely one. The hypothesis updater will rank them.
- simulation_mode: output exactly ONE of the two tokens:
  * "HYPOTHESIS" — the user is explicitly requesting comparative or causal investigation
    (e.g. "compare X vs Y", "test whether", "normal vs fat-tail", "which strategy is better").
  * "DIRECT_MC"  — the user wants a direct simulation of specified parameters
    (e.g. "simulate a portfolio", "model an epidemic", "run a queueing simulation").
  When in doubt, default to "DIRECT_MC".
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
    response = llm_invoke_safe(messages, node_name="PARSER")
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

    # Extract and persist simulation_mode from spec
    sim_mode = "DIRECT_MC"
    try:
        spec_obj2 = json.loads(spec_str)
        raw_mode = spec_obj2.get("simulation_mode", "")
        if raw_mode in ("DIRECT_MC", "HYPOTHESIS"):
            sim_mode = raw_mode
    except (json.JSONDecodeError, AttributeError):
        pass

    return {"simulation_spec": spec_str, "simulation_mode": sim_mode, "language": target_lang}

def hypothesis_updater_node(state: SimulationState) -> SimulationState: #node 1b (hypothesis updater - bayesian belief table)
    """
    Hypothesis Updater: Implements Bayesian hypothesis updating.

    Takes the parsed simulation spec and:
    1. Identifies all competing hypotheses for the scenario outcome.
    2. Assigns prior probabilities to each (labelled MODEL_ASSUMPTION).
    3. For each piece of USER_PROVIDED or AI_INFERENCE evidence,
       applies a likelihood ratio update to the hypothesis weights.
    4. Normalizes posterior weights into probabilities.
    5. Returns a ranked table of prior vs. posterior beliefs plus the
       dominant hypothesis and evidence log.

    The planner reads this to focus simulation on the right causal branch,
    while still acknowledging secondary hypotheses and their probabilities.
    """
    target_lang = _get_language(state)
    sim_mode = _classify_simulation_mode(state)

    if sim_mode == "DIRECT_MC":
        log_step_start("HYPOTHESIS UPDATER", f"Evaluating simulation mode [{target_lang}]...")
        log_step_success(
            "HYPOTHESIS UPDATER",
            0.0,
            "DIRECT_MC mode active — bypassing Bayesian hypothesis update (no competing hypotheses requested)"
        )
        analysis_dict = {
            "status": "bypassed",
            "reason": "No competing hypotheses were requested. The simulation uses the direct production-line/system model specified by the user.",
            "hypotheses": [],
            "dominant_hypothesis": None,
            "dominant_hypothesis_name": None,
            "dominant_posterior": 0.0,
            "secondary_hypotheses": [],
            "excluded_hypotheses": [],
            "simulation_focus": "Direct Monte Carlo execution of user-specified parameters and operational rules.",
            "multi_branch_required": False,
            "multi_branch_reason": "Direct simulation; no branching requested.",
            "evidence_log": []
        }
        return {"hypothesis_analysis": json.dumps(analysis_dict), "language": target_lang}

    log_step_start(
        "HYPOTHESIS UPDATER",
        f"Performing Bayesian hypothesis updating [{target_lang}] using '{settings.MODEL_NAME}'..."
    )

    spec = state.get("simulation_spec", "")

    lang_instruction = f"""
LANGUAGE REQUIREMENT (MANDATORY — HIGHEST PRIORITY):
The user scenario is written in '{target_lang}'.
ALL textual content in your JSON response (hypothesis "name", "description", "exclusion_reason", "evidence", "justification", "dominant_hypothesis_name", "simulation_focus", "multi_branch_reason") MUST BE WRITTEN IN THE EXACT SAME LANGUAGE as the user scenario ({target_lang}).
JSON keys and enum status values ("dominant", "plausible", "low_probability", "excluded", "USER_PROVIDED", "OBSERVED", "AI_INFERENCE", "MODEL_ASSUMPTION", "DERIVED") MUST REMAIN in English for program compatibility.
"""

    prompt = f"""
You are the Hypothesis Scientist of a Universal AI Simulation Engine.

Your task is to perform Bayesian hypothesis updating on the user's scenario.
{lang_instruction}

--------------------------------------------------
SIMULATION SPECIFICATION
--------------------------------------------------
{spec}

--------------------------------------------------
INSTRUCTIONS
--------------------------------------------------

STEP 1 — ENUMERATE HYPOTHESES
List ALL plausible causal explanations for the scenario's main outcome.
Use the competing_hypotheses from the spec if provided; add any you think are missing.
Each hypothesis must have a short name and a one-sentence causal story.

STEP 2 — ASSIGN PRIORS
Assign a prior probability to each hypothesis BEFORE considering any evidence.
These are your base rates before any evidence is applied.
Label all priors as "MODEL_ASSUMPTION" — you are inventing these.
Priors must sum to 1.0.

STEP 3 — EVIDENCE LOG
List every piece of evidence from the spec's initial_state and parameters
that is USER_PROVIDED, OBSERVED, or AI_INFERENCE.
For each evidence item, write:
  - The evidence statement
  - Its provenance label
  - For EACH hypothesis: a likelihood ratio (> 1 = makes hypothesis more likely,
    < 1 = makes hypothesis less likely, 1.0 = neutral)
  - Justification for each likelihood ratio

STEP 4 — COMPUTE POSTERIORS
For each hypothesis, multiply its prior by ALL likelihood ratios from Step 3.
Normalize so posteriors sum to 1.0.
Show the full computation.

STEP 5 — RANK AND CONCLUDE
Rank hypotheses by posterior probability.
Identify the dominant hypothesis (highest posterior).
Note which hypotheses are "still plausible" (posterior > 5%) vs "low probability" (< 5%).
IMPORTANT: A hypothesis with low posterior is NOT the same as "ruled out".
           Only say a hypothesis is "excluded" if its posterior is effectively 0
           AND there is a hard logical impossibility (not just low probability).

PROVENANCE & CALIBRATION RULES:
  - Prior probabilities: always MODEL_ASSUMPTION (subjective base rates)
  - Likelihood ratios: MODEL_ASSUMPTION (uncalibrated model-assigned weights, not empirical constants)
  - Posterior probabilities: DERIVED (mathematically calculated from priors × likelihoods)
  - CRITICAL SCIENTIFIC DISTINCTION: Posteriors are model-derived belief weights under specified likelihood assumptions, NOT real-world empirical occurrence probabilities.
  - Evidence items: match the provenance from the spec

Respond ONLY with a valid JSON object:
{{
  "hypotheses": [
    {{
      "id": "<H1>",
      "name": "<short name>",
      "description": "<one-sentence causal story>",
      "prior": <float 0-1>,
      "prior_provenance": "MODEL_ASSUMPTION",
      "posterior": <float 0-1>,
      "posterior_provenance": "DERIVED",
      "posterior_rank": <int starting from 1>,
      "status": "dominant | plausible | low_probability | excluded",
      "exclusion_reason": "<only if status=excluded: the hard logical impossibility>"
    }}
  ],
  "evidence_log": [
    {{
      "evidence": "<statement>",
      "provenance": "USER_PROVIDED | OBSERVED | AI_INFERENCE",
      "likelihood_ratios": {{
        "<hypothesis_id>": <float>,
        ...
      }},
      "justification": "<why these ratios>"
    }}
  ],
  "dominant_hypothesis": "<hypothesis id>",
  "dominant_hypothesis_name": "<short name>",
  "dominant_posterior": <float>,
  "secondary_hypotheses": ["<list of plausible hypothesis ids>"],
  "excluded_hypotheses": ["<list of truly excluded hypothesis ids>"],
  "simulation_focus": "<one sentence: what causal chain the simulation should primarily test>",
  "multi_branch_required": <true|false>,
  "multi_branch_reason": "<if true: why the simulation must model multiple branches>"
}}
"""

    messages = [
        SystemMessage(
            content=(
                "You are an expert probabilistic reasoning scientist. "
                "You apply Bayesian inference to identify the most supported "
                "hypothesis from a set of competing explanations. "
                "You are rigorous about distinguishing 'low probability' from 'ruled out'. "
                "Respond only with valid JSON."
            )
        ),
        HumanMessage(content=prompt)
    ]

    log_step_progress("Sending spec to LLM for Bayesian hypothesis updating...")
    start_time = time.time()
    response = llm_invoke_safe(messages, node_name="HYPOTHESIS_UPDATER")
    elapsed = time.time() - start_time

    raw = response.content or ""
    analysis_str = _strip_json_fences(raw)

    try:
        analysis = json.loads(analysis_str)
        hypotheses = analysis.get("hypotheses", [])
        dominant = analysis.get("dominant_hypothesis_name", "unknown")
        dominant_p = analysis.get("dominant_posterior", 0.0)
        secondary = analysis.get("secondary_hypotheses", [])
        excluded = analysis.get("excluded_hypotheses", [])
        multi = analysis.get("multi_branch_required", False)
        evidence_count = len(analysis.get("evidence_log", []))
        log_step_success(
            "HYPOTHESIS UPDATER",
            elapsed,
            f"Bayesian update complete — {len(hypotheses)} hypotheses, "
            f"dominant='{dominant}' ({dominant_p:.1%}), "
            f"plausible secondary={len(secondary)}, excluded={len(excluded)}, "
            f"evidence_items={evidence_count}, multi_branch={multi}"
        )
    except json.JSONDecodeError:
        log_step_warning(
            "HYPOTHESIS UPDATER",
            f"Could not parse hypothesis analysis as JSON — storing raw ({elapsed:.2f}s)"
        )
        analysis_str = raw

    return {"hypothesis_analysis": analysis_str, "language": target_lang}

def planner_node(state: SimulationState) -> SimulationState: #node 2
    """
    Planner (Simulation Architect): Generates a rigorous, scientifically honest
    simulation script guided by the specification and provenance rules.
    """
    target_lang = _get_language(state)
    retry_count = state.get("retry_count", 0)
    critic_retry_count = state.get("critic_retry_count", 0)
    has_exec_error = bool(state.get("execution_error"))
    has_critic_issue = bool(state.get("critic_feedback")) and not state.get("critic_approved", True)

    if has_exec_error:
        log_step_start(
            "PLANNER",
            f"Fixing execution error [{target_lang}] — regeneration attempt {retry_count + 1} "
            f"using '{settings.MODEL_NAME}'..."
        )
    elif has_critic_issue:
        log_step_start(
            "PLANNER",
            f"Addressing critic feedback [{target_lang}] — revision {critic_retry_count} "
            f"using '{settings.MODEL_NAME}'..."
        )
    else:
        log_step_start(
            "PLANNER",
            f"Generating simulation script (Simulation Architect) [{target_lang}] using '{settings.MODEL_NAME}'..."
        )

    spec = state.get("simulation_spec", "")
    hypothesis_analysis = state.get("hypothesis_analysis", "")

    spec_block = f"""
--------------------------------------------------
SIMULATION SPECIFICATION
(follow this contract — respect provenance labels)
--------------------------------------------------
{spec}
""" if spec else ""

    hypothesis_block = f"""
--------------------------------------------------
BAYESIAN HYPOTHESIS ANALYSIS
(use this to focus your simulation on the right causal branch)
--------------------------------------------------
{hypothesis_analysis}
""" if hypothesis_analysis else ""

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
    sim_mode = _classify_simulation_mode(state)

    lang_instruction = f"""
═══════════════════════════════════════════════════
LANGUAGE REQUIREMENT (MANDATORY — HIGHEST PRIORITY)
═══════════════════════════════════════════════════

The user scenario is written in '{target_lang}'.
1. Architectural plan reasoning text, explanations, code docstrings, and code comments MUST be written in {target_lang}.
2. Terminal prints: ALL `print(...)` statements and console outputs generated by the Python code MUST be written in {target_lang} (e.g. print("=== ... ==="), progress prints, scenario conclusions). This ensures that when the code executes, the terminal output (stdout) shown to the user is in {target_lang}!
3. In SIMULATION_RESULTS_JSON: Keep keys in English, but write all human-readable string values (such as methodology_note, assumption justifications, event descriptions, decision descriptions, termination_reason) in {target_lang}.
"""

    # ── Mode gate: Parts 11 & 13 are only included in HYPOTHESIS mode ─────────
    if sim_mode == "DIRECT_MC":
        mode_gate_block = """
═══════════════════════════════════════════════════
PART 0 — SIMULATION MODE GATE (READ FIRST)
═══════════════════════════════════════════════════

SIMULATION MODE: DIRECT_MC

The user asked for a DIRECT simulation — not a hypothesis comparison.
This means:

■ DO NOT generate branch_weights, active_branch, or np.random.choice(["H1","H2",...])
■ DO NOT generate h1_success_flags, h2_success_flags, or any per-branch flag arrays
■ DO NOT use Bayesian posteriors to drive simulation sampling probabilities
■ The hypothesis belief table (if provided) is EPISTEMIC BACKGROUND ONLY.
  It must NOT alter any simulation parameter, distribution, or sampling rule.
■ Write a clean, faithful Monte Carlo simulation of EXACTLY what the user specified.
■ All results are reported as plain empirical statistics (mean, median, percentiles, etc.).
■ Parts 11 and 13 of this prompt do NOT apply in DIRECT_MC mode — skip them entirely.

CORRECT DIRECT_MC PATTERN (portfolio example):
  for run in range(num_runs):
      value = initial_value
      for year in range(horizon):
          r_stock = rng.normal(stock_mean, stock_vol)
          r_bond  = rng.normal(bond_mean,  bond_vol)
          value  *= (1 + stock_alloc * r_stock + bond_alloc * r_bond)
      final_values.append(value)
  # Report: median, p5, p95, P(final < initial)

FORBIDDEN IN DIRECT_MC (these indicate mode contamination):
  branch_weights = [posterior_H1, posterior_H2]          # FORBIDDEN
  active_branch  = np.random.choice(["H1","H2"], p=...) # FORBIDDEN
  h1_success_flags.append(...)                           # FORBIDDEN
"""
    else:  # HYPOTHESIS mode
        mode_gate_block = """
═══════════════════════════════════════════════════
PART 0 — SIMULATION MODE GATE (READ FIRST)
═══════════════════════════════════════════════════

SIMULATION MODE: HYPOTHESIS

The user explicitly requested a comparative or causal investigation.
Parts 11 and 13 of this prompt APPLY IN FULL:
■ Implement each plausible hypothesis as a separate causal branch.
■ Branch weights MUST be derived from the Bayesian posteriors above.
■ H1/H2 success rates MUST be computed as conditional probabilities:
    h1_runs      = sum(1 for i in range(n) if branch[i] == "H1")
    h1_successes = sum(1 for i in range(n) if branch[i] == "H1" and success[i])
    h1_rate      = h1_successes / h1_runs   # conditional, NOT joint!
  NEVER compute h1_rate = sum(h1_flags) / num_runs  — that is a joint probability.
"""

    # In DIRECT_MC suppress Parts 11 and 13 from the hypothesis_block
    effective_hypothesis_block = hypothesis_block if sim_mode == "HYPOTHESIS" else ""

    prompt = f"""
You are the Simulation Architect of a Universal AI Simulation Engine.

Your task is to transform the user's scenario into a rigorous, executable,
scientifically honest Python simulation.
{lang_instruction}
{mode_gate_block}
USER SCENARIO:
\"{state.get('user_prompt', '')}\"
{spec_block}{effective_hypothesis_block}{error_context}{critic_context}

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

METHODOLOGY SELECTION TABLE — match your problem type to the right methodology:

  Problem Type                           | Best Methodology
  --------------------------------------|---------------------------------
  Cyber attack / fraud probability       | monte_carlo
  Investment portfolio / financial risk  | monte_carlo
  Epidemic / disease spread              | system_dynamics OR agent_based_simulation
  Population dynamics / ecology          | system_dynamics
  Chemical / physical system             | system_dynamics
  Supply chain / logistics disruption    | discrete_event_simulation
  Queueing / server / call-center        | discrete_event_simulation
  Manufacturing / factory floor          | discrete_event_simulation
  Multiple autonomous attackers / agents | agent_based_simulation
  Traffic / pedestrian movement          | agent_based_simulation
  Market competition with strategies     | game_theoretic
  Any combination of the above           | hybrid

You MUST justify your methodology choice in methodology_note by referencing
the matching row from this table.

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
Use a seeded generator per trial: `rng = np.random.default_rng(run_seed)` where `run_seed = 42 + run_idx`.

═══════════════════════════════════════════════════
PART 7.1 — RIGOROUS SCIENTIFIC SIMULATION ENGINE RULES (MANDATORY)
═══════════════════════════════════════════════════

1. RNG PROPAGATION & ISOLATION (ZERO-LEAKAGE RULE):
   - In each trial, instantiate ONE generator: `rng = np.random.default_rng(seed)`.
   - Pass this `rng` explicitly into EVERY class constructor, entity instance, and helper function:
     `patient = Patient(patient_id, arrival_time, rng)`
   - FORBIDDEN: NEVER instantiate `np.random.default_rng()` without a seed inside any class or helper!
   - FORBIDDEN: NEVER call global `np.random.*` functions (e.g. `np.random.exponential`). All random draws MUST be `rng.exponential(...)`, `rng.poisson(...)`, etc., using the passed `rng`.

2. TRUE DISCRETE-EVENT SIMULATION (DES) ENGINE:
   - When the problem involves queues, hospitals, servers, or logistics with continuous service durations, implement a TRUE EVENT-DRIVEN KERNEL using a priority queue (`heapq` or `simpy`).
   - Advance time to the next scheduled event: `(event_time, event_type, priority, entity_id) = heapq.heappop(event_queue)`.
   - FORBIDDEN: Do NOT advance time in a fixed 1-minute loop (`for minute in range(...)`) when service times are continuous. Fixed 1-minute time-stepping introduces systematic discretization bias.

3. FINITE-HORIZON CLEARING & SURVIVORSHIP BIAS:
   - In finite-horizon queueing/service systems (e.g. 24-hour arrival window), do NOT simply stop the simulation and calculate metrics ONLY from completed entities. That causes severe truncation/survivorship bias under congestion!
   - EITHER implement a CLEARANCE PERIOD: stop generating new arrivals after the arrival window ends, but continue simulation until all in-system entities are served and discharged.
   - OR explicitly track and report:
     - `arrivals_count`: total entities generated in window
     - `completions_count`: entities discharged
     - `remaining_wip`: entities still waiting in queue or in service at cutoff
     - Distinct metrics for completed cohort vs. right-censored waiting cohort.

4. MULTI-STAGE METRIC DECONFOUNDING (WAITING TIMES):
   - Never report a single omnibus "waiting time". In multi-stage processes, distinctly measure and name:
     - `triage_wait`: arrival -> triage start
     - `doctor_wait`: triage end -> consultation start
     - `arrival_to_doctor_wait`: arrival -> consultation start
     - `treatment_wait`: consultation end -> treatment start
     - `total_time_in_system`: arrival -> discharge
   - Clearly define each metric in the output JSON.

5. EMPIRICAL CONSTRAINT-BLOCKING MEASUREMENT:
   - FORBIDDEN: NEVER output hardcoded dummy lists (e.g. `[0] * NUM_RUNS`) for constraint blocked flags!
   - Actively instrument constraint blocking: measure minutes and events where queues were non-empty while available resources were 0 (e.g. `doctor_blocking_minutes`, `doctor_blocking_events`, `bed_blocking_minutes`).

6. HONESTY IN STATISTICAL CLAIMS:
   - Do NOT claim "Common Random Numbers (CRN)" unless multiple alternative scenarios are simulated with synchronized random seeds. For single-scenario replications, label as "Independent Monte Carlo replications".
   - Do NOT claim "Sensitivity Analysis" unless a formal multi-scenario parameter variation sweep is actually executed in the simulation.
   - For all reported metrics, compute both the Replication Standard Deviation (SD) across trials AND the Monte Carlo Standard Error (MCSE = SD / sqrt(N)) with 95% Confidence Intervals.

═══════════════════════════════════════════════════
PART 8 — MANDATORY STRUCTURED OUTPUT
═══════════════════════════════════════════════════

At the END of the script, print this exact marker followed by JSON:

SIMULATION_RESULTS_JSON:
{{
  "metadata": {{
    "runs": <int>,
    "seed": <int>,
    "methodology": "<chosen methodology>",
    "methodology_note": "<one sentence description>",
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
    "<metric_name>_success_flag": [<list of per-run 0/1 values>]
  }},
  "summary": {{
    "<metric>_observed_rate": <float, computed from flag array>,
    "<metric>_std": <float, np.std(flag_array, ddof=1)>
  }},
  "constraint_application": {{
    "<constraint_name>_constraint_blocked_flag": [<list of per-run 0/1 values>]
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

STATISTICS RULE: Every <metric>_std value in the summary block MUST be computed
from the actual per-run array using np.std(array, ddof=1).
DO NOT report std of a scalar (e.g. dominant_branch_observed_rate is a single float;
its std is meaningless). Only report std for values that vary across runs.

═══════════════════════════════════════════════════
PART 9 — CODE QUALITY REQUIREMENTS
═══════════════════════════════════════════════════

1. Include np.random.seed(42) for reproducibility.
2. No undefined variables. All variables initialized before use.
3. Division by zero guards where required.
4. No impossible states (e.g. negative populations).
5. If visualizing data with matplotlib: do NOT call plt.show(). Instead, save the figure with plt.savefig('simulation_chart.png', dpi=150, bbox_inches='tight') and call plt.close(). If not plotting, omit matplotlib.
6. Also print human-readable output BEFORE the JSON block.
7. Libraries available: numpy, scipy, simpy, networkx, matplotlib,
   random, math, pandas, statistics
8. STANDARD DEVIATION RULE:
   Compute std only from per-run arrays, never from scalars.
   Correct:  h1_std = np.std(branch_h1_success_flags, ddof=1)
   Correct:  h2_std = np.std(branch_h2_success_flags, ddof=1)
   Forbidden: dominant_rate_std = np.std(dominant_observed_rate)  # scalar has no std!
   The human-readable output section may then print:
     "H1: {{h1_observed_rate:.1%}} ± {{h1_std:.1%}}"
   but ONLY when h1_std is computed from the actual flag array.
9. TERMINOLOGY RULES:
   - Use "MODELED LEADING SCENARIO" instead of "DETERMINATION" or "Most likely attack scenario".
   - Use "most frequently successful modeled pathway" instead of "validated pathway" or "confirmed pathway".

The code must be syntactically valid, complete, and executable without modification.

═══════════════════════════════════════════════════
PART 10 — PROBABILITY METRIC SEPARATION (CRITICAL)
═══════════════════════════════════════════════════

NEVER mix model-estimated probability, simulated outcome, and likelihood score
into a single variable. These are three distinct concepts:

  p_model_<event>        — your assumed probability of the event occurring
                           (MODEL_ASSUMPTION; a fixed input to the simulation)
  <event>_success_flag   — 1/0 per-trial binary outcome (SIMULATED per run)
  <event>_observed_rate  — actual Monte Carlo frequency: successes / total_runs
                           (DERIVED; computed AFTER all runs complete)

NAMING RULE — FORBIDDEN variable names for 0/1 outcomes:
  BAD:  probability_of_remote_exploitation  = [1.0, 0.0, 1.0, ...]  ← these are flags!
  BAD:  remote_exploitation_likelihood      = [1.0, 0.0, 1.0, ...]  ← these are flags!
  GOOD: remote_exploitation_success_flag    = [1,   0,   1,   ...]  ← correct
  GOOD: remote_exploitation_observed_rate   = 0.84                  ← scalar after all runs

The word "probability" or "likelihood" in a variable name MUST refer to
a scalar float in [0, 1] that is a model input or a post-run aggregate.
It MUST NOT refer to a per-trial binary list.

FORBIDDEN PATTERN:
  # BAD — mixes assumed prob with simulated outcome
  likelihood = 1.0 if success else effective_prob

CORRECT PATTERN:
  # GOOD — all three are computed separately
  p_model_exploit = 0.85                                   # MODEL_ASSUMPTION scalar
  exploit_success_flag = int(np.random.random() < p_model_exploit)  # per-trial 0/1
  # ... after all runs:
  exploit_observed_rate = sum(exploit_flags) / num_runs   # DERIVED scalar

In the SIMULATION_RESULTS_JSON, report ALL THREE under separate keys:
  "outcomes": {{
    "exploit_success_flag": [...per-run 0/1 list...],     # stochastic outcome
    "exploit_observed_rate": <scalar float 0-1>            # final empirical rate
  }},
  "parameters": {{
    "p_model_exploit": {{"value": 0.85, "provenance": "MODEL_ASSUMPTION"}}
  }}

The analyst report must say:
  "Observed Monte Carlo success rate: X% (k/n runs), with a Bernoulli std dev of Y across trials (SE = ±Z%)"
  NOT "Attack vector likelihood: Z%" and NEVER "X% ± Y% confidence".

═══════════════════════════════════════════════════
PART 10.5 — PARAMETER SENSITIVITY SWEEP (SCIENTIFIC BEST PRACTICE)
═══════════════════════════════════════════════════

When the simulation outcome is strongly driven by an influential MODEL_ASSUMPTION parameter
(e.g., arrival demand multiplier, boundary defensiveness, threshold, exploit rate, adoption fraction):
Include a parameter sensitivity sweep in the simulation script.
Sweep 4 to 6 values across a plausible range (e.g. [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
by ACTUALLY RE-RUNNING the simulation function for each point (e.g. 50-100 runs per point) and output in SIMULATION_RESULTS_JSON:
  "sensitivity_sweep": {{
    "parameter": "<name_of_param>",
    "points": [
      {{"value": 0.70, "rate": 0.21}},
      {{"value": 0.80, "rate": 0.27}},
      {{"value": 0.90, "rate": 0.34}},
      {{"value": 1.00, "rate": 0.39}},
      {{"value": 1.10, "rate": 0.44}},
      {{"value": 1.20, "rate": 0.48}}
    ]
  }}

STRICTLY FORBIDDEN IN SENSITIVITY SWEEPS:
- Never generate synthetic sensitivity numbers using formulaic multiplication or random proxies like `pressure_factor = rate / 20.0; if random.random() < (0.6 + ...)`!
- Every single point in the sensitivity sweep MUST be measured from actual simulated trials executed with that parameter value.
- The sensitivity loop must instantiate and execute the real simulation engine for every parameter point.

═══════════════════════════════════════════════════
PART 10.6 — FIDELITY IN QUEUEING, TRAFFIC, AND PHYSICAL TIMELINES (MANDATORY)
═══════════════════════════════════════════════════

When simulating queueing, traffic signals, transportation, or physical service systems:

1. DEMAND/CAPACITY REGIME & CALIBRATION (NO PERMANENT SATURATION):
   - The volume-to-capacity ratio (v/c) MUST be calibrated to test meaningful control dynamics.
   - DO NOT set baseline arrival demand so high (e.g. peak demand >> clearing capacity) that the intersection becomes permanently and overwhelmingly oversaturated from minute 1 through minute 480.
     * When v/c >> 1.0 permanently, queues grow monotonically from 0 to 1,300+ vehicles, capacity constraints block in 100% of trials, and changing signal timing has near-zero leverage because green time simply serves an infinite backlog.
     * The simulation then measures uncontrolled congestion accumulation, NOT signal-control effectiveness.
   - MANDATORY CALIBRATION:
     * Baseline demand should be calibrated near the critical operating envelope (e.g. v/c ≈ 0.70 to 0.92 during peak, 0.40 to 0.60 off-peak) where queue clearance is sensitive to phase allocation policies.
     * When testing demand sensitivity, test multiple distinct operating regimes:
       - Low demand: 0.50x (free-flow)
       - Moderate demand: 0.75x (mild queueing, rapid clearance)
       - Baseline / Peak: 1.00x (near capacity, queue formation during peaks)
       - High demand: 1.25x (oversaturation onset)
       - Extreme demand: 1.50x (gridlock accumulation)
     * This allows observing the phase transition: Free-Flow → Moderate Congestion → Saturation → Dissipation.

2. QUEUE DELAY ACCOUNTING & SENSITIVITY MONOTONICITY (PREVENT TRUNCATION PARADOX):
   - When calculating "average waiting time" in an experiment where an unserved queue remains at the end of the simulation horizon:
     * CRITICAL QUEUEING LAW: NEVER compute average wait time solely across COMPLETED vehicles if unserved vehicles remain trapped in queue at t = horizon!
     * Doing so causes the classic truncation paradox: under higher demand (1.2x), vehicles that arrive late experience massive delays but are cut off before service, so they are excluded from the completed average, making wait time appear artificially to DECREASE!
     * MANDATORY: When unserved vehicles remain at horizon end, report:
       (a) `completed_vehicle_avg_wait`: average queue wait of served vehicles.
       (b) `unserved_vehicle_count` and `unserved_accumulated_wait`: backlog and delay accumulated up to horizon.
       (c) `total_system_delay`: sum of all waiting time across ALL arriving vehicles (served wait + unserved wait).
       (d) `average_system_wait`: total_system_delay / total_arrivals.
     * The sensitivity sweep must report `average_system_wait` or `average_queue_length` (Little's Law: W = L/λ), ensuring physical monotonicity: higher demand MUST yield higher average delay or queue length.

3. METRIC SEMANTIC CORRECTNESS (QUEUE DELAY vs RESIDENCE TIME):
   - Every individual vehicle/entity must track its own lifecycle timestamps:
     - `arrival_time`: Timestamp when the vehicle entered the queue.
     - `service_start_time`: Timestamp when the vehicle reached the head of the queue AND the server/green phase began serving it.
     - `departure_time`: Timestamp when the vehicle completed crossing/service (`departure_time = service_start_time + crossing_time`).
   - EXACT DEFINITIONS:
     - Queue Waiting Time: `waiting_time = v.service_start_time - v.arrival_time` (pure queue delay before service).
     - Crossing / Service Time: `service_time = v.departure_time - v.service_start_time`
     - Total Residence Time: `residence_time = v.departure_time - v.arrival_time` (queue wait + crossing time).
   - MANDATORY: When the user or experiment requests "average waiting time", you MUST calculate:
     `waiting_time = v.service_start_time - v.arrival_time`
     DO NOT calculate `v.departure_time - v.arrival_time` as waiting time — that is total residence time and erroneously inflates wait time by the vehicle's crossing duration!

4. PHYSICAL SERVICE TIMELINE & SUB-MINUTE RESOLUTION:
   - When the scenario provides sub-minute service parameters (e.g. 20–60 seconds per vehicle):
     The simulation clock MUST operate at sub-minute resolution (e.g. in seconds or discrete events).
     DO NOT use a crude 1-minute step loop (`for minute in range(480)`) with arbitrary caps like `discharged_count < 2`!
   - Crossing times MUST physically govern server availability and lane capacity:
     - Each active server/lane maintains `server_free_time` (seconds).
     - When vehicle `v` at the front of the queue is served during a green phase:
       ```python
       service_start = max(v.arrival_time, current_time_sec, server_free_time_sec)
       service_end = service_start + v.crossing_time_sec
       if service_end <= green_phase_end_sec:
           # Vehicle finishes crossing before phase ends
           v.service_start_time = service_start
           v.departure_time = service_end
           server_free_time_sec = service_end
           current_time_sec = service_end
           queue.pop(0)
           completed_vehicles.append(v)
       else:
           # Server/lane remains occupied or green ends before vehicle can clear
           break
       ```
     - A 60-second crossing vehicle occupies the server for 60 seconds; a 20-second vehicle occupies it for 20 seconds. Crossing time directly determines throughput!

5. GENUINELY ADAPTIVE CONTROLLER (STATE-ACTUATED LOGIC):
   - The adaptive controller MUST NOT be a static cycle with small random jitter. It must dynamically inspect measurable intersection state variables:
     * `queue_lengths`: Current vehicles queued on North/South vs. East/West approaches.
     * `elapsed_green_sec`: Time the current phase has been continuously green.
     * `vehicle_headway / gap`: Time since the last vehicle crossed in the active phase.
     * `opposing_queue`: Vehicles waiting on the red approach.
   - ACTUATED CONTROL RULES:
     * Minimum Green (`G_min`, e.g. 10–15 s): Prevent rapid phase flickering; phase cannot terminate before `G_min`.
     * Maximum Green (`G_max`, e.g. 60–90 s): Prevent starvation of opposing approach; phase MUST terminate if `elapsed >= G_max` and opposing queue exists.
     * Gap-Out / Extension: While `elapsed < G_max`, extend green if vehicles are arriving / queue remains; if a gap occurs (no vehicles arriving within threshold e.g. 3–5 s) OR opposing queue exceeds current approach queue by threshold, terminate phase and switch.
     * Fixed-Time Comparison: Fixed controller uses constant, non-adaptive timing (e.g. strictly 45s NS, 45s EW) regardless of real-time queue states.

6. TIME-SERIES TIMESTEP & OBSERVATION METADATA:
   - Explicitly declare the timestep semantics in SIMULATION_RESULTS_JSON:
     ```python
     "time_series_resolution": {{
         "time_step_sec": 60,
         "step_unit": "minute",
         "total_observations": 480,
         "horizon_minutes": 480
     }}
     ```
   - If time_series has 480 points, declare that each point represents the snapshot at the end of that simulation minute.

═══════════════════════════════════════════════════
PART 10.7 — STOCHASTIC REPRODUCTION AND EXPERIMENTAL CONTROL (COMMON RANDOM NUMBERS & PAIRED DIFFERENCES)
═══════════════════════════════════════════════════

For comparative experiments (e.g. comparing Fixed-Time vs Adaptive Signal Control):

1. COMMON STOCHASTIC ENVIRONMENT (PAIRED MONTE CARLO):
   - DO NOT let competing strategies independently generate their own separate random arrivals, approaches, or crossing times! Doing so confounds policy differences with stochastic traffic realization variance.
   - For each Monte Carlo run `i`, generate a single shared traffic realization:
     ```python
     traffic_realization = generate_traffic_realization(seed=run_seed, horizon_sec=..., peak_rate=..., offpeak_rate=...)
     ```
     containing all arrival timestamps, lane/approach assignments, and pre-sampled physical crossing times.
   - Replay this EXACT realization through both policies:
     ```python
     sim_fixed = run_simulation(copy.deepcopy(traffic_realization), strategy="fixed")
     sim_adaptive = run_simulation(copy.deepcopy(traffic_realization), strategy="adaptive")
     ```
   - Both policies experience identical traffic. Only signal phase timing and switching decisions differ.

2. PAIRED DIFFERENCE & EFFECT-SIZE INFERENCE (CRUCIAL):
   - Under Common Random Numbers, do NOT treat runs as two independent samples. Compute paired differences for each trial `i`:
     ```python
     delta_wait = [adaptive_waits[i] - fixed_waits[i] for i in range(num_runs)]
     delta_queue = [adaptive_queues[i] - fixed_queues[i] for i in range(num_runs)]
     delta_throughput = [adaptive_throughputs[i] - fixed_throughputs[i] for i in range(num_runs)]
     ```
   - Compute effect-size statistics in the script and include in SIMULATION_RESULTS_JSON:
     ```python
     "paired_comparison": {{
         "delta_wait": delta_wait,
         "mean_delta_wait": float(np.mean(delta_wait)),
         "relative_delta_wait_pct": float(np.mean(delta_wait) / np.mean(fixed_waits) * 100) if np.mean(fixed_waits) else 0.0,
         "delta_queue": delta_queue,
         "mean_delta_queue": float(np.mean(delta_queue)),
         "delta_throughput": delta_throughput,
         "mean_delta_throughput": float(np.mean(delta_throughput)),
         "paired_trial_win_rate_wait": float(sum(1 for d in delta_wait if d < 0) / num_runs)
     }}
     ```
   - Report:
     - Mean paired difference: e.g. -15.48 s
     - Relative percentage difference: e.g. -0.133%
     - 95% confidence interval of the paired difference: `mean_delta ± 1.96 * (std_delta / sqrt(n))`
     - Paired trial win rate: fraction of runs where Adaptive strictly outperformed Fixed.

3. DECOUPLE BAYESIAN POSTERIORS IN ENGINEERING COMPARISONS:
   - For comparative engineering benchmark experiments (e.g. testing control algorithms), report empirical simulation outcomes directly.
   - Clear distinction:
     * `H1 observed success rate` = fraction of simulated trials satisfying the engineering performance threshold (e.g. Adaptive wait < Fixed wait).
     * `H1 Bayesian posterior` = epistemic belief weight updated from the scenario parser/prior/likelihood log.
     * These are fundamentally different quantities and must NEVER be conflated in tables or conclusions.

═══════════════════════════════════════════════════
PART 10.8 — DISCRETE-EVENT & FRACTIONAL-TIME FIDELITY IN MANUFACTURING AND RELIABILITY (MANDATORY)
═══════════════════════════════════════════════════

When simulating manufacturing lines, machine reliability, queueing networks, or equipment failure/repair:

1. TRUE DISCRETE-EVENT & FRACTIONAL-TIME ADVANCEMENT (NO HOURLY INTEGER APPROXIMATIONS):
   - STRICTLY FORBIDDEN: An hourly loop `for hour in range(horizon)` that treats whole hours as production units or deducts 1.0 from repair timers.
   - Fractional repair times (e.g. 18 min = 0.30 h) MUST affect the actual production timeline. When a repair finishes at 0.30 h, the line resumes and produces for the remaining 0.70 h.
   - For every interval [t_start, t_end] of duration dt = t_end - t_start:
     1. Determine whether the production line is operational.
     2. Calculate production over the actual elapsed interval:
        production += dt * nominal_rate * rate_multiplier
     3. Accumulate line downtime using the UNION of machine-unavailable intervals (see below).
     4. Accumulate machine-specific operating exposure only during intervals where that machine is operational.
     5. Never round, ceil, or floor repair durations to whole hours.

2. OVERLAPPING REPAIRS & LINE DOWNTIME UNION (MANDATORY):
   - If multiple machines are down at the same time, actual line downtime is the UNION of their unavailable intervals, NEVER the sum of overlapping repair durations!
   - Example: Machine 1 is down [10.0, 10.5] and Machine 2 fails at 10.0 and is down [10.0, 10.7]. Line downtime is 0.70 hours, NOT 1.20 hours! Naively summing machine repair times artificially exaggerates downtime and contaminates loss attribution.

3. PRESERVE PER-OPERATING-HOUR BERNOULLI FAILURE OPPORTUNITIES (INTERPRETATION A):
   - The user specifies failure probability per operating hour (e.g. 2% per hour). Preserve this Bernoulli opportunity semantics without converting to an arbitrary continuous hazard rate.
   - Machines CANNOT fail while under repair or while the line is halted!
   - Event scheduling logic:
     Track machine operating exposure. When an operating machine completes an operating hour (or at operating checkpoints), roll Bernoulli trial with probability p_fail.
     If a failure occurs:
       Sample repair duration: repair_mins = rng.lognormal(mean=np.log(repair_median_minutes), sigma=repair_lognormal_sigma)
       repair_hours = repair_mins / 60.0
       machine_repair_finish[m] = current_time + repair_hours
     While any machine is under repair, the line is halted:
       Next event time is the earliest repair completion: min(finish for finish in active_repairs).
       Advance current_time to this completion time.
       Accumulate line downtime by dt. All operational machines remain stopped (no operating exposure accumulates, no failures can occur).
     When all active repairs complete, the line resumes immediately at that exact fractional timestamp!

4. MACHINE OPERATING EXPOSURE & THEORETICAL BOUND:
   - Operating exposure accumulates ONLY when the machine is operational.
   - Theoretical upper bound on failures = NUM_MACHINES * HORIZON_HOURS * FAILURE_PROB_PER_HOUR (e.g. 5 * 720 * 0.02 = 72.0).
   - Because machines spend time under repair, observed mean failures MUST be strictly below this upper bound.
   - Report under "exposure" in SIMULATION_RESULTS_JSON:
     "exposure": {{
       "expected_upper_bound_failures": float(expected_upper_bound),
       "mean_operating_hours": float(mean_operating_hours),
       "mean_failures": float(mean_failures),
       "failure_rate_per_operating_hour": float(mean_failures / mean_operating_hours) if mean_operating_hours > 0 else 0.0
     }}

5. SYMMETRIC LOSS ATTRIBUTION:
   - Track:
     downtime_loss = total_line_downtime_hours * NOMINAL_RATE
     ideal_uptime_production = total_line_uptime_hours * NOMINAL_RATE
     actual_uptime_production = sum of actual units produced during operational intervals
     net_rate_variation_effect = actual_uptime_production - ideal_uptime_production
   - Because production rate variation (e.g. Normal(1.0, 0.05) or Uniform(0.95, 1.05)) is symmetric around 1.0, positive variation offsets negative variation on average.
   - In SIMULATION_RESULTS_JSON, report under "loss_attribution":
     "loss_attribution": {{
       "downtime_loss_units": float(np.mean(downtime_losses)),
       "ideal_uptime_production_units": float(np.mean(ideal_uptimes)),
       "actual_uptime_production_units": float(np.mean(actual_uptimes)),
       "net_rate_variation_effect_units": float(np.mean(net_rate_effects))
     }}

6. INDIVIDUAL 1,000 RUNS DISTRIBUTION IN JSON (MANDATORY):
   - The user requested "total production for each simulation".
   - In SIMULATION_RESULTS_JSON under "outcomes", MUST include the complete raw list of all 1,000 float values:
     "outcomes": {{
       "total_production_distribution": [float(p) for p in total_productions],
       "sub_target_production_flag": [int(p < 65000) for p in total_productions],
       "probability_of_sub_target_production": float(prob_below_target)
     }}
   - This allows downstream statistical validation to independently compute percentiles, mean, and confidence intervals.

7. TERMINOLOGY & UNCERTAINTY REPORTING:
   - FORBIDDEN: Never use "success_flag" when referring to falling below a production target.
   - Use "sub_target_production_flag" and "probability_of_sub_target_production".
   - Report Monte Carlo SE and 95% Monte Carlo Confidence Interval:
     mc_se = np.sqrt(prob_below_target * (1.0 - prob_below_target) / num_runs)
     ci_low = max(0.0, prob_below_target - 1.96 * mc_se)
     ci_high = min(1.0, prob_below_target + 1.96 * mc_se)
   - Do NOT report Bernoulli standard deviation as prediction uncertainty.
   - Clearly identify this as numerical Monte Carlo sampling uncertainty: "Monte Carlo SE = ±X.X%, 95% MC interval = [Y%, Z%]".

8. SENSITIVITY SWEEP USING THE EXACT SAME EVENT KERNEL & CRN:
   - The sensitivity analysis MUST call the EXACT SAME event-driven simulation function as the baseline model.
   - Do NOT create a simplified hourly loop for sensitivity analysis!
   - Use Common Random Numbers (CRN) across parameter sweeps (e.g. pass run_seed = base_seed + trial_idx or use a dedicated RNG generator per trial).

9. METADATA DECLARATION & NO GLOBAL RNG:
   - Use `rng = np.random.default_rng(seed)` throughout. DO NOT call `np.random.seed(...)`. DO NOT `import random`.
   - In SIMULATION_RESULTS_JSON metadata:
     "methodology": "discrete_event_monte_carlo",
     "methodology_note": "Event-driven Monte Carlo simulation with continuous fractional-hour event timing and union line downtime.",
     "event_time_resolution": "continuous_fractional_hours"

═══════════════════════════════════════════════════
PART 10.9 — MULTI-RESOURCE QUEUEING & SERVICE SYSTEMS (ED / HOSPITAL / CALL-CENTER / MULTI-STAGE QUEUE) (MANDATORY)
═══════════════════════════════════════════════════

Applies when the scenario involves patient/customer queues flowing through multiple sequential service stages with distinct resource pools (e.g. nurses, doctors, beds).

1. RESOURCE-ROLE EXPLICIT DECLARATION (NO SILENT ASSUMPTIONS):
   - Every patient-processing stage MUST declare exactly which resources it consumes and for how long.
   - FORBIDDEN: silently binding a doctor to the treatment stage when the prompt specifies only "2 doctors, 3 nurses, 10 beds" and the treatment step does not mention doctor attendance.
   - CORRECT default resource mapping (unless the scenario EXPLICITLY states otherwise):
       Triage:       1 nurse occupied for triage_duration_minutes
       Consultation: 1 doctor occupied for consultation_duration_minutes
       Treatment:    1 bed occupied for treatment_duration_minutes  ← NO doctor, NO nurse by default
       Discharge:    no resource consumed
   - If the scenario EXPLICITLY says "a doctor must supervise treatment", then model it and declare:
       "treatment_requires_doctor": {{"value": True, "provenance": "USER_PROVIDED", "note": "Scenario states doctor remains during treatment."}}
   - If the model adds a resource requirement NOT stated in the scenario, declare it as:
       "treatment_requires_doctor": {{"value": True, "provenance": "MODEL_ASSUMPTION", "note": "Assumed for conservative capacity estimate."}}
   - With 2 doctors and 10 beds: tying a doctor to each bed effectively limits concurrent treatments to 2, not 10. This is a MAJOR capacity impact that must be disclosed.

2. MEASUREMENT-WINDOW SEPARATION (ARRIVAL WINDOW vs DRAIN WINDOW):
   - Define three explicit windows:
       ARRIVAL_WINDOW:     [0, T_horizon]          → patients enter only during this period
       PERFORMANCE_WINDOW: [0, T_horizon]          → resource utilization denominators
       DRAIN_WINDOW:       [T_horizon, ∞)          → system drains remaining patients; NO utilization billed
   - MANDATORY implementation pattern:
       # Inside event loop:
       dt_for_util = max(0.0, min(current_time, T_horizon) - max(last_time, 0.0))
       doctor_busy_minutes  += doctors_occupied * dt_for_util
       nurse_busy_minutes   += nurses_occupied  * dt_for_util
       bed_busy_minutes     += beds_occupied    * dt_for_util
   - FORBIDDEN: accumulating resource occupancy beyond T_horizon while keeping the denominator as `T_horizon × capacity`.
   - Hard invariant enforced in code at end of each replication:
       doctor_util = doctor_busy_minutes / (T_horizon * N_DOCTORS)
       nurse_util  = nurse_busy_minutes  / (T_horizon * N_NURSES)
       bed_util    = bed_busy_minutes    / (T_horizon * N_BEDS)
       assert 0.0 <= doctor_util <= 1.0 + 1e-9, f"Doctor util OOB: {{doctor_util}}"
       assert 0.0 <= nurse_util  <= 1.0 + 1e-9, f"Nurse util OOB: {{nurse_util}}"
       assert 0.0 <= bed_util    <= 1.0 + 1e-9, f"Bed util OOB: {{bed_util}}"

3. EXACT PIECEWISE-CONSTANT NHPP — THINNING METHOD (MANDATORY):
   - FORBIDDEN: sampling `inter_arrival = rng.exponential(1.0 / rate_at_current_hour)` because the drawn inter-arrival can cross an hour boundary under the wrong rate.
   - CORRECT thinning implementation:
       lambda_max = max(HOURLY_RATES) / 60.0   # per minute
       t = 0.0
       while t < T_horizon:
           t += rng.exponential(1.0 / lambda_max)  # candidate arrival time
           if t >= T_horizon:
               break
           current_hour = int(t // 60)
           lambda_t = HOURLY_RATES[min(current_hour, len(HOURLY_RATES)-1)] / 60.0
           if rng.random() < lambda_t / lambda_max:   # accept/reject
               arrivals.append(t)
   - This guarantees exact NHPP generation regardless of rate boundary crossings.

4. METRIC SEMANTIC PRECISION (MANDATORY STAGE-LEVEL LABELS):
   - Every wait/delay metric must be named to reflect its exact definition:
       arrival_to_triage_start_wait:    queue time before triage begins
       arrival_to_consult_start_wait:   arrival → triage queue wait + triage + consult queue wait
       arrival_to_treatment_start_wait: arrival → all pre-treatment delays
       total_length_of_stay:            arrival to discharge (complete ED visit time)
   - FORBIDDEN: calling `consultation_start_time - arrival_time` simply "mean patient waiting time" or "average wait" without stage qualification.
   - The '>30 min threshold' metric must also specify which stage: e.g. "% patients with arrival-to-consultation wait > 30 min".
   - Store all four stage metrics per patient and report means in SIMULATION_RESULTS_JSON:
       "waiting_times": {{
           "mean_triage_queue_wait_min":    float(...),
           "mean_consult_queue_wait_min":   float(...),
           "mean_treatment_queue_wait_min": float(...),
           "mean_total_los_min":            float(...),
           "pct_arrival_to_consult_over_30_min": float(...)
       }}

5. NO FABRICATED SENSITIVITY OR CRN CLAIMS:
   - FORBIDDEN: writing in the report "under Common Random Numbers, queue metrics scale monotonically" unless the simulation code contains an ACTUAL parameter sweep loop.
   - If `scenario_comparison == {{}}` or no second scenario was run, the report MUST state:
       "Sensitivity analysis was not executed in this simulation run."
   - Only describe sensitivity findings that correspond to executed code. Do NOT extrapolate from theoretical expectations and present them as experimental results.

6. MULTI-STAGE WAITING TIME DECOMPOSITION (MANDATORY IN REPORT):
   - Track per-patient timestamps:
       patient.arrival_time
       patient.triage_start_time
       patient.triage_end_time
       patient.consult_start_time
       patient.consult_end_time
       patient.treatment_start_time
       patient.treatment_end_time  (= discharge_time)
   - Compute per patient:
       triage_queue_wait   = triage_start_time - arrival_time
       consult_queue_wait  = consult_start_time - triage_end_time
       treatment_queue_wait = treatment_start_time - consult_end_time
       total_los           = treatment_end_time - arrival_time

7. PHYSICAL INVARIANT ASSERTIONS IN CODE:
   - Add at the END of each replication (after all events are processed):
       assert available_doctors >= 0, f"available_doctors negative: {{available_doctors}}"
       assert available_nurses  >= 0, f"available_nurses negative: {{available_nurses}}"
       assert available_beds    >= 0, f"available_beds negative: {{available_beds}}"
       assert len(completed_patients) <= total_arrivals, "completed > arrivals"
   - These are hard failures, not warnings. If they trigger, the simulation has a logic bug.

8. LOG-NORMAL MEAN PARAMETERIZATION (MANDATORY):
   - CRITICAL BUG: Using `mu = math.log(target_mean)` then calling `rng.lognormal(mu, sigma)` gives:
       E[X] = exp(mu + sigma^2 / 2) = target_mean * exp(sigma^2 / 2)
     which is LARGER than target_mean. The reported and actual mean values disagree.
   - CORRECT parameterization to achieve a desired arithmetic mean:
       import math
       sigma = 0.5   # (or any value)
       target_mean = 15.0   # arithmetic mean you actually want
       mu = math.log(target_mean) - (sigma ** 2) / 2.0
       sample = rng.lognormal(mu, sigma)  # => E[sample] == 15.0
   - Apply this correction to EVERY log-normal service time (triage, consultation, treatment, repair).
   - FORBIDDEN: Using `math.log(5.0)` as mu when you want a mean of 5 minutes.
   - Declare in SIMULATION_RESULTS_JSON parameters:
       "triage_duration_mean_min": {{"value": target_mean, "mu_used": mu, "sigma_used": sigma, "provenance": "USER_PROVIDED"}}

9. PER-STAGE QUEUE INSTRUMENTATION (MANDATORY IN MULTI-STAGE SYSTEMS):
   - FORBIDDEN: Reporting a single aggregate `mean_queue_length = triage_queue + consult_queue + treatment_queue`.
   - MANDATORY: Track and report each stage queue separately:
       triage_queue_lengths   = []  # samples over time
       consult_queue_lengths  = []  # samples over time
       treatment_queue_lengths = [] # samples over time
   - Report in SIMULATION_RESULTS_JSON:
       "queue_lengths": {{
           "triage":    {{"mean": ..., "p95": ..., "max": ..., "time_positive_pct": ...}},
           "consult":   {{"mean": ..., "p95": ..., "max": ..., "time_positive_pct": ...}},
           "treatment": {{"mean": ..., "p95": ..., "max": ..., "time_positive_pct": ...}},
           "total":     {{"mean": ..., "p95": ..., "max": ...}}
       }}
   - This allows the report to legitimately say "congestion was primarily in the doctor consultation queue" based on data.

10. BLOCKING vs UTILIZATION (DO NOT CONFLATE):
    - FORBIDDEN: Defining "resource blocked" as `utilization >= 0.99`.
      * Utilization = fraction of time resource was busy.
      * Blocking = a patient was unable to enter service because all servers were occupied (queue formed).
    - CORRECT blocking measurement:
        # At each event, check whether a patient arrived for a stage but had to wait:
        doctor_blocked_time_minutes = 0.0    # cumulative time doctor queue had >= 1 patient waiting
        bed_blocked_time_minutes    = 0.0    # cumulative time bed queue had >= 1 patient waiting
        nurse_blocked_time_minutes  = 0.0    # cumulative time triage queue had >= 1 patient waiting
        # Or count per-patient blocking events:
        patients_who_waited_for_doctor += 1 if patient.consult_queue_wait > 0 else 0
    - Report:
        "blocking": {{
            "pct_patients_waited_for_doctor": float(...),  # % who queued at consultation
            "pct_patients_waited_for_bed":    float(...),  # % who queued at treatment
            "pct_patients_waited_for_triage": float(...),  # % who queued at triage
            "mean_consult_queue_when_positive": float(...)  # conditional mean given queue > 0
        }}
    - Never label a utilization threshold check as "constraint blocking."

11. FIXED HOURLY SNAPSHOTS (NOT EVENT-TRIGGERED SNAPSHOTS):
    - FORBIDDEN: Taking a snapshot when `current_time >= last_snapshot + 60.0` inside the event loop.
      * This produces event-triggered snapshots at irregular intervals (e.g. t=62, t=124) not hourly.
      * Periods with no events produce missing hours.
    - CORRECT approach: Pre-define fixed observation times and query system state at each:
        observation_times = list(range(0, T_horizon + 1, 60))  # [0, 60, 120, ..., 1440]
        snapshots = []
        obs_idx = 0
        # During event processing, whenever current_time > observation_times[obs_idx]:
        while obs_idx < len(observation_times) and current_time >= observation_times[obs_idx]:
            snapshots.append({{
                "time_min": observation_times[obs_idx],
                "triage_queue": len(triage_queue),
                "consult_queue": len(consult_queue),
                "treatment_queue": len(treatment_queue),
                "doctors_busy": INITIAL_DOCTORS - available_doctors,
                "beds_busy":    INITIAL_BEDS - available_beds
            }})
            obs_idx += 1
    - Report time-series snapshots at exactly t=0, 60, 120, ..., 1440.

12. HORIZON/CLEARANCE SEPARATION & WORKLOAD/CAPACITY AUDIT:
    - MANDATORY: Snapshot system state at EXACTLY t = T_horizon (e.g. 1440 minutes):
        snapshot_at_horizon = {{
            "time_min": T_horizon,
            "patients_completed": len(completed_patients),
            "patients_in_triage":    (triage queue + active triage),
            "patients_in_consult":   (consult queue + active consult),
            "patients_in_treatment": (treatment queue + active treatment),
            "total_remaining_in_system": X
        }}
    - If simulation drains past T_horizon, measure clearance time:
        clearance_time_minutes = time_of_last_discharge - T_horizon
    - Report in summary:
        "horizon_state": {{
            "patients_remaining_at_24h": int(...),
            "patients_completed_within_24h": int(...),
            "clearance_time_minutes": float(...)  # time after 24h until system empty
        }}
    - MANDATORY pre-simulation workload audit: compute expected utilization at design capacity before running.
      Include in SIMULATION_RESULTS_JSON under "workload_audit":
        expected_arrivals_per_hour     = sum(hourly_rates) / 24.0
        doctor_throughput_per_hour     = INITIAL_DOCTORS * 60.0 / mean_consult_minutes
        nurse_throughput_per_hour      = INITIAL_NURSES  * 60.0 / mean_triage_minutes
        bed_throughput_per_hour        = INITIAL_BEDS    * 60.0 / mean_treatment_minutes
        doctor_load_ratio  = expected_arrivals_per_hour / doctor_throughput_per_hour
        nurse_load_ratio   = expected_arrivals_per_hour / nurse_throughput_per_hour
        bed_load_ratio     = expected_arrivals_per_hour / bed_throughput_per_hour
      If any load_ratio > 1.0, the stage is structurally overloaded and the report MUST declare this upfront.
      This workload audit validates whether simulation outcomes (large queues) are consistent with capacity arithmetic.

═══════════════════════════════════════════════════
PART 11 — MULTI-BRANCH CAUSAL MODELING (HYPOTHESIS MODE ONLY)
═══════════════════════════════════════════════════

⚠ THIS PART APPLIES ONLY IN HYPOTHESIS MODE (see Part 0).
If simulation_mode == DIRECT_MC, skip this section entirely.

If the Bayesian Hypothesis Analysis provided above identifies multiple PLAUSIBLE
hypotheses (any hypothesis with posterior > 5%), your simulation MUST model
EACH plausible hypothesis as a separate causal branch.

For each branch:
  1. Implement its own causal chain (sequence of conditional events)
  2. Give each step its own p_model probability (labelled MODEL_ASSUMPTION)
  3. Give each step its own stochastic noise (timing jitter, detection prob, etc.)
  4. Track per-branch success/failure independently
  5. Include branch-specific constraint checks

TIME HORIZON RULE:
  Events must be able to occur at ANY step across the full simulation horizon,
  not just at one hardcoded step number.
  Use stochastic timing: event_time = np.random.poisson(expected_step)
  NOT: if step == 2:

UNKNOWN vs FALSE POSITIVE — MANDATORY DISTINCTION:
  Do NOT combine "unknown attack" and "false positive" into one bucket.

  unknown_attack_mechanism:
    Meaning: A real incident occurred but the mechanism is unidentified by the model.
    Use when: All modeled hypotheses fail in a trial AND the alert was genuine.
    Label:    outcome = "unknown_attack_mechanism"

  false_positive:
    Meaning: No actual incident occurred; the alert was spurious.
    Use when: The model explicitly simulates alert reliability and the alert fires
              even though no hypothesis succeeds.
    Label:    outcome = "false_positive"

  ONLY use "false_positive" if your model explicitly includes P(alert | no incident).
  If your model does NOT model alert reliability, use "unknown_attack_mechanism"
  for unresolved trials. Never combine both into a single bucket label.

OUTCOME STRUCTURE:
  "outcomes": {{
    "branch_<H1>_success_flag": [...per-run 0/1...],
    "branch_<H2>_success_flag": [...per-run 0/1...],
    "unknown_attack_flag":      [...per-run 0/1...],
    "false_positive_flag":      [...per-run 0/1...],  # only if modeled
    "dominant_branch_observed_rate": <scalar float>
  }}

This makes your simulation a genuine investigation model, not a
single-event Monte Carlo repeated 100 times.

═══════════════════════════════════════════════════
PART 12 — CONSTRAINT ENFORCEMENT (CRITICAL)
═══════════════════════════════════════════════════

Hard constraints derived from USER_PROVIDED evidence must INVALIDATE
incompatible causal branches BEFORE any probability roll is made.

THE SMOKING-GUN ANTI-PATTERN — this exact code structure is FORBIDDEN:

  # ❌ FORBIDDEN: probabilistic constraint violation
  if world["state"]["employee_status"] == "asleep" and np.random.random() < 0.05:
      physical_access_violated_flag = 1

  Why is this wrong?
    - It still rolls a random number while the employee is asleep.
    - It allows the impossible event to occur 5% of the time.
    - A hard constraint should NEVER be paired with a probability roll.
    - This is a constraint-as-probability anti-pattern.

DISTINCTION:
  Hypothesis:  "Could the insider have acted?"       ← investigative question, can roll
  Constraint:  "Is physical access logically possible?"  ← physical limit, NO roll allowed

CORRECT THREE-LEVEL ENFORCEMENT TEMPLATE:

  # Level 1: Evaluate the constraint from USER_PROVIDED state (no randomness)
  physical_access_possible = (
      world["state"]["employee_status"] != "asleep"
  )  # USER_PROVIDED evidence

  # Level 2: Block the branch BEFORE any probability roll
  if not physical_access_possible:
      insider_manual_branch_active = False
      # Record that a branch was blocked (for validation output)
      physical_access_constraint_flags.append(1)  # 1 = constraint active this run
  else:
      insider_manual_branch_active = True
      physical_access_constraint_flags.append(0)

  # Level 3: Only THEN do the probability roll — but only for the active branch
  if insider_manual_branch_active:
      roll = np.random.random()
      if roll < p_model_insider_manual:
          # insider manual action succeeded
          ...

KEY RULE: Rename "constraint_violations" to "constraint_application":
  "constraint_application": {{
      "physical_access_constraint_blocked_flag": physical_access_constraint_flags
  }}
  Never name this "constraint_violations" when recording that a constraint was enforced or a branch was blocked.

CLARIFY BLOCKED HYPOTHESIS PARAMETERS:
  If a parameter (e.g. p_model_physical_access = 0.05) belongs to a hypothesis/branch
  that is deterministically blocked by a hard constraint (e.g. H3 blocked because employee is asleep),
  explicitly document in comments and in the parameters JSON metadata:
    "p_model_physical_access": {{
        "value": 0.05,
        "provenance": "MODEL_ASSUMPTION",
        "evaluation_status": "not_evaluated_deterministically_blocked",
        "note": "P_MODEL_PHYSICAL_ACCESS = 0.05 is not evaluated because H3 is deterministically blocked before sampling."
    }}
  Do NOT roll a random number against this parameter.

BAYESIAN POSTERIOR vs. SIMULATION ELIGIBILITY (ARCHITECTURAL DISTINCTION):
  Make the architectural distinction explicit:
    - Bayesian posterior (e.g. H3 Bayesian posterior = 1.23%): Statistical belief updated from evidence.
    - Simulation eligibility (e.g. H3 simulation eligibility = 0%): Feasibility for simulation execution.
  Because the employee is confirmed asleep, H3 has 0% simulation eligibility and is deterministically
  blocked before sampling.

FOR HYPOTHESES THAT SURVIVE CONSTRAINT CHECKING:
  Some insider hypotheses do NOT require physical access
  (e.g. remote credential theft, compromised session token).
  These are NOT blocked by the physical alibi constraint.
  Model them separately:
    insider_remote_credential_branch  ← NOT blocked by alibi constraint
    insider_manual_physical_branch    ← IS blocked by alibi constraint

  Only the manual-physical sub-branch is deactivated. The remote sub-branch
  remains active and uses its own probability roll normally.

═══════════════════════════════════════════════════
PART 13 — BAYESIAN-TO-SIMULATION PARAMETER BRIDGE (HYPOTHESIS MODE ONLY)
═══════════════════════════════════════════════════

⚠ THIS PART APPLIES ONLY IN HYPOTHESIS MODE (see Part 0).
If simulation_mode == DIRECT_MC, skip this section entirely and do NOT
use branch_weights or active_branch sampling in the simulation code.

The Bayesian Hypothesis Analysis provided above contains POSTERIOR PROBABILITIES
for each hypothesis. These posteriors MUST be used as the basis for the
simulation branch activation probabilities, not invented independently.

CONCRETE RULE:
  For each hypothesis H_i with posterior p_i from the Bayesian analysis:
    branch_weight[H_i] = p_i  # [DERIVED from Bayesian updater]

  When a Monte Carlo trial begins, sample which branch is "active" using:
    active_branch = np.random.choice(branch_names, p=branch_weights)

  Then simulate the full causal chain for that branch.

This means the Bayesian posteriors DRIVE the simulation, not just decorate the report.

CRITICAL — CONDITIONAL SUCCESS RATES (not joint):
  H1/H2 observed success rates MUST be computed conditionally:
    h1_runs      = sum(1 for i in range(n) if branch[i] == "H1")
    h1_successes = sum(1 for i in range(n) if branch[i] == "H1" and success[i])
    h1_rate      = h1_successes / max(h1_runs, 1)  # P(success | H1)
  NEVER use: h1_rate = sum(h1_flag_array) / num_runs  — this is P(H1 ∩ success), not P(success|H1).

In the SIMULATION_RESULTS_JSON, expose the bridge explicitly:
  "parameters": {{
    "bayesian_posteriors_used": {{
      "<hypothesis_name>": {{"posterior": <float>, "provenance": "DERIVED",
                              "source": "hypothesis_updater_node"}},
      ...
    }}
  }}

If the Bayesian analysis was not available (e.g. empty hypothesis_analysis),
fall back to MODEL_ASSUMPTION weights and label them as such.
NEVER silently use assumed probabilities when Bayesian posteriors are available.
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
    response = llm_invoke_safe(messages, node_name="PLANNER")
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
        "language": target_lang,
    }

def critic_node(state: SimulationState) -> SimulationState: #node 3
    """
    Model Critic: Reviews the generated simulation code against the specification.
    Flags methodology mislabelling, provenance violations, assumption extrapolation,
    determinism, and fairness issues — before execution.
    """
    target_lang = _get_language(state)
    log_step_start(
        "CRITIC",
        f"Reviewing code for scientific validity [{target_lang}] using '{settings.MODEL_NAME}'..."
    )

    spec = state.get("simulation_spec", "No specification available.")
    code = state.get("generated_code", "")

    lang_instruction = f"""
═══════════════════════════════════════════════════
LANGUAGE REQUIREMENT (MANDATORY — HIGHEST PRIORITY)
═══════════════════════════════════════════════════

The user scenario is written in '{target_lang}'.
Write the critique review, evaluation summary, identified problems, and revision recommendations in the EXACT SAME LANGUAGE as the user scenario ({target_lang}).
Keep JSON keys ("approved", "severity", "issues", etc.) in English for program compatibility.
"""

    prompt = f"""
You are the Model Critic of a Universal AI Simulation Engine.

Review the generated simulation code for scientific, methodological,
and structural problems BEFORE it is executed.
{lang_instruction}

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
      outcomes, constraint_application (or constraint_violations), events, decisions?
    - FLAG CRITICAL: if the JSON block is missing entirely.
    - FLAG WARNING: if provenance fields are absent from parameters.

11. CODE_QUALITY
    - Are there potential runtime errors (undefined variables, division by zero)?
    - Does the code include np.random.seed(42) or equivalent?
    - FLAG WARNING: if no random seed is present.

12. PROBABILITY_METRIC_SEPARATION
    - Are model-estimated probability, simulated outcome, and likelihood score kept SEPARATE?
    - FLAG CRITICAL: if a binary success outcome (0/1) is directly used as a probability score
      (e.g. attack_vector_likelihood = 1.0 if success else effective_prob).
    - FLAG CRITICAL: if the observed Monte Carlo success rate is conflated with a model prior.
    - FLAG CRITICAL: if the "likelihood" field in JSON outcomes mixes a fixed assumption with
      a stochastic simulation result.
    - REQUIRED PATTERN: code must separately compute AND report:
        (a) p_model_<event>       — the assumed probability (MODEL_ASSUMPTION constant)
        (b) success_flag          — per-trial 0/1 outcome (stochastic)
        (c) observed_success_rate — computed AFTER all runs as successes/total_runs

13. CAUSAL_BRANCH_COVERAGE
    - If the spec lists multiple competing hypotheses with posteriors > 5%, does the simulation
      model EACH as a separate causal branch?
    - FLAG WARNING: if only the dominant hypothesis is modeled and secondary hypotheses
      (posterior > 5%) are ignored.
    - FLAG CRITICAL: if the spec indicates multi_branch_required=true but the code only
      has a single causal path.
    - FLAG WARNING: if meaningful events only occur at a single hardcoded time step
      (e.g. all activity at step 2 of a 24-step simulation). Events should be able to
      occur at any step via stochastic timing.

14. OUTCOME_VARIABLE_NAMING
    - Do variable names for per-trial 0/1 outcomes avoid the words "probability" or "likelihood"?
    - FLAG CRITICAL: if any list of binary per-trial values (e.g. [1.0, 0.0, 1.0, ...]) is
      stored in a variable whose name contains "probability" or "likelihood".
      Examples of BAD names: probability_of_remote_exploitation, attack_likelihood
      Examples of GOOD names: remote_exploitation_success_flag, exploit_outcome_flag
    - FLAG WARNING: if the scalar post-run observed rate is stored in a variable whose name
      does NOT clearly indicate it is a rate or frequency (e.g. "exploit_observed_rate").
    - FLAG WARNING: if 'unknown_or_false_positive' is used as a single bucket label —
      these must be split into 'unknown_attack_mechanism' and 'false_positive' separately.

15. CONSTRAINT_ENFORCEMENT
    - Do hard USER_PROVIDED constraints actively INVALIDATE incompatible branches
      BEFORE any probability roll?
    - FLAG CRITICAL: if constraint checking uses a random roll, e.g.:
        if employee_status == 'asleep' and np.random.random() < 0.05:
            physical_access_violated_flag = 1
      This is the smoking-gun anti-pattern. The constraint must set branch_active = False
      BEFORE any np.random call, not after a probability roll.
    - FLAG CRITICAL: if a physically-impossible branch produces non-zero successes in
      the simulation output (e.g. physical_access_violated: [0,0,1,...] means a
      constraint was violated and the impossible event actually occurred in the model).
    - FLAG CRITICAL: if constraint_application or constraint_violations reports that an impossible event occurred
      in N/total_runs runs (e.g. 'Physical Access Constraint Violations: 3 / 100').
    - REQUIRED: constraint enforcement must be tracked under "constraint_application":
        "constraint_application": {{
            "physical_access_constraint_blocked_flag": physical_access_constraint_flags
        }}
    - FLAG WARNING: if P_MODEL_PHYSICAL_ACCESS is evaluated with random rolls when H3 is deterministically blocked.
    - FLAG WARNING: if the code does not distinguish Bayesian posterior (e.g. 1.23%) from simulation eligibility (0%).

16. RNG_ISOLATION_AND_PROPAGATION
    - Does every stochastic draw come strictly from the single seeded `rng = np.random.default_rng(seed)` instance passed to classes and functions?
    - FLAG CRITICAL: if any class (e.g. Patient, Server, Agent), constructor, or function creates an unseeded `np.random.default_rng()` or calls `np.random.seed()` inside loops.
    - FLAG CRITICAL: if unseeded global `np.random.*` calls are used anywhere in the simulation logic.

17. DISCRETE_EVENT_SIMULATION_FIDELITY
    - If the methodology is declared as `discrete_event_simulation`, does it use a true continuous-time next-event scheduler (`heapq` or `simpy`)?
    - FLAG CRITICAL: if the code claims "discrete_event_simulation" but is actually a 1-minute fixed-step discrete loop (`for minute in range(...)`) with continuous service durations rounded to minute intervals.

18. FINITE_HORIZON_CENSORING_AND_WIP
    - In finite-horizon queueing/service systems, does the code properly handle entities remaining in the system at the horizon boundary?
    - FLAG CRITICAL: if waiting times are computed ONLY over discharged/completed entities while heavily congested queues still contain unserved entities at cutoff (survivor bias), without reporting remaining WIP and right-censoring.

19. CONSTRAINT_BLOCKING_INSTRUMENTATION
    - Are constraint-blocked flags and duration metrics genuinely measured by the simulation runtime?
    - FLAG CRITICAL: if constraint application flags or blocking metrics are dummy hardcoded lists (e.g. `[0] * NUM_RUNS`). The code must actively count events/minutes where queues were non-empty while resources were 0.
    - ACCEPTABLE: a branch that does NOT require physical access (remote credential
      theft) is not constrained by a physical alibi.

16. BAYESIAN_PARAMETER_BRIDGE
    - Are the Bayesian posterior probabilities from hypothesis_analysis actually used
      as the branch weights in the Monte Carlo sampling?
    - FLAG WARNING: if the planner had access to a hypothesis_analysis block but the
      code uses independently invented probabilities (e.g. exploit_prob = 0.85) without
      referencing the Bayesian posteriors.
    - FLAG WARNING: if the structured output's parameters block does NOT include a
      'bayesian_posteriors_used' field when posteriors were available.
    - ACCEPTABLE: if hypothesis_analysis was empty/unavailable, MODEL_ASSUMPTION
      weights are acceptable and should be labelled as such.

17. STATISTICAL_REPORTING
    - Is every ±value in the report computed from an actual per-run array?
    - FLAG CRITICAL: if the summary block contains '<metric>_std' values that are
      not computed from a per-run list using np.std(array, ddof=1).
    - FLAG CRITICAL: if 'dominant_branch_observed_rate_std: 0.0' appears in the JSON.
      This is meaningless — dominant_branch_observed_rate is a derived scalar, not
      an array. std of a scalar is always 0 and tells you nothing.
    - FLAG CRITICAL: if the human-readable output prints 'X% ± Y%' but Y is not
      computed from a corresponding per-run flag array.
    - REQUIRED PATTERN:
        h1_std = np.std(branch_h1_success_flags, ddof=1)  # from array
        print(f'H1: {{h1_observed_rate:.1%}} ± {{h1_std:.1%}}')
    - FLAG WARNING: if no per-branch standard deviation is reported at all.

18. TERMINOLOGY_PRECISION
    - Does the code or human readable output use "MODELED LEADING SCENARIO" instead of "DETERMINATION"?
    - Does it use "most frequently successful modeled pathway" instead of "validated pathway"?
    - FLAG WARNING: if "DETERMINATION" or "validated pathway" appears in code prints or docstrings.

19. EMERGENT_METRICS_VS_HARDCODING
    - In queueing, service, or traffic simulations, are waiting times and delays computed from actual timestamps (e.g. current_time - arrival_time)?
    - FLAG CRITICAL: if waiting times are appended as hardcoded constants (e.g. `metrics['waiting_times'].append(1.0)` or `append(0.8)`).
    - FLAG CRITICAL: if standard deviation of waiting time is ~0 across runs when queues vary dynamically.

20. FAIR_COMPARATIVE_BENCHMARKS
    - When comparing policies/strategies (e.g. Fixed vs Adaptive signal control), do both strategies face the EXACT SAME physical universe (identical arrival processes, identical vehicle passing time distributions)?
    - FLAG CRITICAL: if one strategy is artificially granted a faster service rate or higher clearing capacity (e.g. allowing up to 4 vehicles/min for adaptive while fixed allows 3). The physical server parameters must be identical; only the signal allocation policy can change.

21. GENUINE_SIMULATION_SENSITIVITY
    - Does the parameter sensitivity analysis re-run the simulation across parameter values?
    - FLAG CRITICAL: if sensitivity points are computed using synthetic formulas, analytical scaling factors, random probabilities (e.g. `if random.random() < 0.6 + ...`), or mocked results without executing the real simulation engine for every parameter point.

22. COMMON_RANDOM_ENVIRONMENT
    - In comparative experiments (e.g. Fixed-Time vs Adaptive signal control), do both strategies face the EXACT SAME generated stochastic realization per run (paired Monte Carlo comparison)?
    - FLAG CRITICAL: if comparative strategies independently generate arrivals, approaches, service times, or other physical inputs instead of replaying a shared pre-generated traffic realization per run. Policy effects must be strictly isolated using Common Random Numbers (CRN).

23. METRIC_SEMANTIC_CORRECTNESS
    - Are waiting time and delay metrics defined with correct physical/queueing semantics?
    - FLAG CRITICAL: if "waiting time" is calculated as `departure_time - arrival_time` (which erroneously conflates queue delay with crossing/service duration). Waiting time MUST be measured as queue delay: `service_start_time - arrival_time`. Total residence time in system is `departure_time - arrival_time`.

24. PHYSICAL_SERVICE_TIMELINE_AND_RESOLUTION
    - Do sampled physical service/crossing durations (e.g. 20–60 seconds) actually govern service capacity and server availability on a realistic timeline?
    - FLAG CRITICAL: if crossing times are sampled but ignored in discharge capacity decisions (e.g. using an arbitrary discrete limit like `discharged_count < 2` per minute while ignoring whether the sampled crossing times fit within the active green phase).
    - FLAG CRITICAL: if a simulation with sub-minute physical service times uses a crude 1-minute time step where crossing times do not determine when the server becomes available for the next vehicle.

25. DEMAND_CAPACITY_REGIME_AND_STABILITY
    - In queueing/traffic simulations, does the baseline demand allow the system to operate near the critical operational envelope (v/c ≈ 0.70–0.92) or test distinct demand regimes (e.g. 0.5x, 0.75x, 1.0x, 1.25x)?
    - FLAG CRITICAL: if the intersection is modeled in permanent, extreme oversaturation (demand >> capacity across all 480 minutes) where queues monotonically accumulate to 1,300+ vehicles and capacity constraints block in 100% of runs, because signal timing has negligible leverage under permanent gridlock.

26. GENUINELY_ADAPTIVE_CONTROL
    - Does the adaptive controller dynamically observe real-time state variables (queue lengths on active vs opposing approaches, elapsed green time, vehicle headway/gap) and apply actuation rules (minimum green G_min, maximum green G_max, gap-out extension or priority switching)?
    - FLAG CRITICAL: if the "adaptive" controller is merely a fixed cycle with minor noise, rather than dynamically actuating green duration based on real-time queues.

27. PAIRED_DIFFERENCE_EFFECT_SIZES
    - In comparative experiments evaluated with Common Random Numbers (CRN), does the script record per-trial paired differences (delta = Adaptive - Fixed), compute relative effect sizes, 95% confidence intervals on paired differences, and paired win rates?
    - FLAG WARNING: if CRN is used but results are reported only as independent group statistics without paired difference inference.

28. QUEUE_DELAY_TRUNCATION_INTEGRITY
    - When unserved vehicles remain queued at the simulation horizon end, does the model account for unserved backlog delay or report total system delay across all arriving vehicles?
    - FLAG CRITICAL: if waiting time is calculated solely across completed vehicles in an oversaturated regime, causing the truncation paradox where higher demand artificially reduces the reported average wait time.

29. MODE_CONTAMINATION (DIRECT_MC guard)
    - Check the simulation_spec metadata for "simulation_mode". If it is "DIRECT_MC" (or absent/unknown, default to DIRECT_MC):
    - FLAG CRITICAL if the code contains ANY of the following patterns:
        * np.random.choice(["H1", ...], p=branch_weights)
        * active_branch = np.random.choice(...)
        * branch_weights = [...]  (derived from posteriors)
        * h1_success_flags  /  h2_success_flags  (per-branch flag arrays)
      These patterns mean the planner illegitimately introduced multi-branch Bayesian
      sampling into a direct simulation prompt, silently changing the model.
    - FLAG CRITICAL if H1/H2 observed success rates are computed as:
        rate = sum(flag_array) / num_runs
      This computes P(H∩success), not the conditional P(success|H). The correct form is:
        h1_runs = sum(1 for i in range(n) if branch[i]=="H1")
        h1_rate = h1_successes / h1_runs
    - If simulation_mode == "HYPOTHESIS", this check does NOT apply.

30. FRACTIONAL_TIME_DISCRETE_EVENT_FIDELITY
    - In reliability, manufacturing, or service simulations with sub-hourly service/repair times (e.g. median 30-minute repair):
    - FLAG CRITICAL if the code does hourly bucket rounding or integer subtraction (e.g. `repair_timers[m] -= 1.0` or `repair_timer -= 1` in an hourly loop).
    - FLAG CRITICAL if production is added as a full whole hour after a fractional repair without computing production over the exact elapsed fractional interval (`dt * nominal_rate * multiplier`).

31. DOWNTIME_UNION_VS_SUM
    - Does line downtime represent the UNION of machine-unavailable intervals?
    - FLAG CRITICAL if line downtime naively sums overlapping machine repair durations when multiple machines are down simultaneously (e.g. counting 0.5h + 0.5h = 1.0h downtime instead of 0.5h line stoppage).

32. SENSITIVITY_KERNEL_CONSISTENCY
    - Does the parameter sensitivity analysis re-use the exact same event-driven simulation function as the baseline model?
    - FLAG CRITICAL if the sensitivity analysis reverts to a simplified hourly approximation loop or different failure logic.

33. RAW_DISTRIBUTION_IN_OUTCOMES
    - Does `SIMULATION_RESULTS_JSON['outcomes']` contain the complete list of all 1,000 raw trial production values under `"total_production_distribution"`?
    - FLAG CRITICAL if individual trial results are omitted and only summary scalar statistics are saved.

34. TARGET_TERMINOLOGY
    - When an outcome represents falling below a performance target (e.g., `production < 65,000`), is it clearly identified as a sub-target rate / shortfall probability?
    - FLAG CRITICAL if falling below a production target is labelled as a "success rate" or if "success_flag" is used in the JSON outcomes. Must use `"sub_target_production_flag"`.

35. LOSS_ATTRIBUTION_SYMMETRY
    - In production and rate variation models, does the code compute net rate variation symmetrically as `actual_uptime_production - ideal_uptime_production`?
    - FLAG CRITICAL if the code counts only negative variations (e.g. `if var_factor < 1.0: loss += NOMINAL - actual`) while ignoring positive offsets, falsely exaggerating rate variation loss.

36. FAILURE_EXPOSURE_ACCOUNTING
    - Can machines fail while under repair or while the line is stopped?
    - FLAG CRITICAL if machines roll for failure while already in a failed or repair state, or while the line is halted. Machines can only fail during active operating hours.
    - FLAG CRITICAL if mean failures exceed the theoretical exposure upper bound (`NUM_MACHINES * TIME_HORIZON_HOURS * FAILURE_PROB_PER_HOUR`) without justification.
    - FLAG WARNING if machine operating exposure (`machine_operating_hours`) is not tracked.

37. RNG_ARCHITECTURE_AND_TYPOS
    - Does the code use a clean, modern RNG architecture (`rng = np.random.default_rng(seed)`) passed explicitly?
    - FLAG WARNING if `import random` is used alongside NumPy or if global `np.random.seed` is called inside trial loops.
    - FLAG CRITICAL if typos in percentage scaling (such as `prob * 100.1` instead of `prob * 100.0`) are found.

38. RESOURCE_UTILIZATION_INVARIANT (Multi-Resource Queueing Systems)
    - Utilization of any finite-capacity resource must be bounded in [0, 1].
    - FLAG CRITICAL if any utilization metric (doctor_utilization, nurse_utilization, bed_utilization) in the JSON outcomes exceeds 1.0 or is negative. These are physically impossible values regardless of implementation.
    - FLAG CRITICAL if the distribution arrays for utilization contain values > 1.0 or < 0.0 across replications.
    - Example: a doctor_utilization mean of 6.66 or a bed_utilization max of 1.21 MUST be flagged as a critical measurement error, not a modelling result.

39. MEASUREMENT_WINDOW_INVARIANT (Post-Horizon Drain Contamination)
    - If the simulation has a defined performance horizon (e.g. T_horizon = 1440 min) and continues processing patients after T_horizon to drain the system:
    - FLAG CRITICAL if resource_busy_minutes (doctor, nurse, bed) accumulates BEYOND T_horizon while the utilization denominator remains `T_horizon × capacity`.
    - The correct implementation: `dt_for_util = max(0.0, min(current_time, T_horizon) - max(last_time, 0.0))` so drain-period occupancy is excluded from the performance-window numerator.
    - FLAG CRITICAL if utilization is computed as `total_busy_minutes / (T_horizon * capacity)` but events run past T_horizon without clipping the numerator.
    - FLAG WARNING if the simulation does not separate the arrival window from the drain window in comments or logic.

40. RESOURCE_ROLE_CONSISTENCY (Multi-Stage Service Systems)
    - Every patient/entity processing stage must declare exactly which resources it consumes.
    - FLAG CRITICAL if a doctor is consumed (decremented and held) during the treatment stage when the scenario description does NOT explicitly state that a doctor must remain present during treatment. The default treatment resource is a bed only.
    - FLAG CRITICAL if nurses are listed in the scenario resource specification (e.g. "3 nurses") but the code never decrements or manages nurse availability — nurses must be actively tracked if they are a named resource.
    - FLAG CRITICAL if any resource availability counter can go negative (available_doctors < 0, available_nurses < 0, available_beds < 0).
    - FLAG WARNING if a resource-role assumption (e.g. doctor during treatment) is present in the code but not declared in the SIMULATION_RESULTS_JSON parameters with a provenance label.

41. METRIC_SEMANTIC_VALIDATION (Wait-Time Label Accuracy)
    - FLAG CRITICAL if a metric computed as `consultation_start_time - arrival_time` is reported as "mean patient waiting time", "average wait", or "patient wait" without specifying that it measures the pre-consultation delay only (triage queue + triage + consultation queue).
    - The correct label must be: "mean arrival-to-consultation wait" or "mean pre-consultation delay" or equivalent.
    - FLAG CRITICAL if a ">30 min" threshold metric is reported without specifying which stage boundary it applies to (e.g. "% with arrival-to-consultation wait > 30 min", NOT just "% waiting > 30 min").
    - FLAG WARNING if the multi-stage wait decomposition (triage queue wait, consult queue wait, treatment queue wait, total LOS) is absent from the JSON outcomes.

42. EXPERIMENT_EVIDENCE_VALIDATION (No Fabricated Sensitivity Claims)
    - FLAG CRITICAL if the simulation JSON contains `"scenario_comparison": {{}}` (empty dict) but the simulation code or report narrative claims a sensitivity sweep, CRN comparison, or parameter scaling analysis was performed.
    - FLAG CRITICAL if the report states results such as "under Common Random Numbers, metrics scale monotonically with arrival rates" when no sensitivity loop exists in the generated code.
    - The report must only claim experimental findings that are backed by code actually executed in this simulation run. Theoretical expectations must be clearly labeled as such, not as simulation results.
    - If no sensitivity experiment was run, the report must state explicitly: "Sensitivity analysis was not executed in this simulation run."

43. PHYSICAL_INVARIANT_VALIDATION (Runtime Correctness Guards)
    - FLAG CRITICAL if physical invariant assertions are absent from the code at the end of each replication:
        assert available_doctors >= 0
        assert available_nurses  >= 0
        assert available_beds    >= 0
        assert len(completed_patients) <= total_arrivals
    - FLAG CRITICAL if any replication produces `completed_patients > total_arrivals`.
    - FLAG CRITICAL if any replication produces `available_doctors > INITIAL_DOCTORS` or `available_beds > INITIAL_BEDS` (resource counter exceeded initial capacity — double-release bug).
    - FLAG WARNING if the simulation continues past T_horizon to drain patients but does not track how many patients were in-progress at T_horizon (end-of-horizon backlog).

44. LOGNORMAL_MEAN_PARAMETERIZATION (Service Time Distribution)
    - FLAG CRITICAL if `mu = math.log(target_mean)` is used as the log-normal mu parameter.
      The correct formula to achieve an arithmetic mean of `target_mean` is:
          mu = math.log(target_mean) - (sigma ** 2) / 2.0
      Using `math.log(target_mean)` directly inflates the mean by `exp(sigma^2/2)`.
      Example: sigma=0.4, target_mean=5.0 → actual mean = 5.42 min (8.4% overrun).
               sigma=0.5, target_mean=15.0 → actual mean = 16.92 min (12.8% overrun).
               sigma=0.8, target_mean=45.0 → actual mean = 59.26 min (31.7% overrun).
    - FLAG CRITICAL if triage, consultation, or treatment service times are parameterized with the wrong mu.
    - The parameters table in SIMULATION_RESULTS_JSON must show both `target_mean` and the `mu_used` value.

45. SPLIT_QUEUE_INSTRUMENTATION (Per-Stage Queue Tracking)
    - FLAG CRITICAL if only a single aggregate queue length is tracked/reported (`total_queue = triage + consult + treatment`).
    - MANDATORY: Each stage's queue must be tracked and reported separately:
        "queue_lengths": {{
            "triage":    {{"mean": ..., "p95": ..., "max": ..., "time_positive_pct": ...}},
            "consult":   {{"mean": ..., "p95": ..., "max": ..., "time_positive_pct": ...}},
            "treatment": {{"mean": ..., "p95": ..., "max": ..., "time_positive_pct": ...}},
            "total":     {{"mean": ..., "p95": ..., "max": ...}}
        }}
    - FLAG WARNING if the report claims "congestion was primarily in the doctor queue" without per-stage queue data to support that conclusion.

46. BLOCKING_VS_UTILIZATION (Resource Constraint Measurement)
    - FLAG CRITICAL if "resource was blocked" or "capacity constraint active" is defined as `utilization >= 0.99`.
      Utilization (fraction of time a resource is busy) is NOT the same as patient blocking (a patient had to wait because all resources were occupied).
    - CORRECT blocking metrics:
        * `pct_patients_waited_for_doctor`: fraction of patients whose consult_queue_wait > 0
        * `pct_patients_waited_for_bed`: fraction of patients whose treatment_queue_wait > 0
        * `doctor_queue_positive_pct`: fraction of simulation time that at least 1 patient was queued for a doctor
        * `mean_consult_queue_when_positive`: mean consult queue length, conditional on queue > 0
    - FLAG CRITICAL if the constraint table claims blocking percentage from a utilization flag.

47. FIXED_SNAPSHOT_MECHANISM (Hourly State Observations)
    - FLAG CRITICAL if time-series snapshots are generated inside the event loop when `current_time >= last_snapshot + 60`.
      This produces event-triggered snapshots at IRREGULAR times, not fixed hourly observations. Long event-free periods produce missing hours.
    - CORRECT: Pre-define `observation_times = [0, 60, 120, ..., 1440]` and check before processing each event:
        while obs_idx < len(observation_times) and current_time >= observation_times[obs_idx]:
            record_snapshot(observation_times[obs_idx])
            obs_idx += 1
    - This guarantees exactly 25 snapshots (t=0..1440) at fixed intervals, usable for time-series analysis.
    - FLAG WARNING if the number of recorded time-series snapshots per replication is not equal to `(T_horizon // snapshot_interval) + 1`.

48. HORIZON_CLEARANCE_SEPARATION (24h State + Clearance Metrics)
    - FLAG CRITICAL if the simulation captures patient-level outcomes (completion times, waiting times) from both inside [0, T_horizon] AND outside [T_horizon, clearance] without clearly separating the two populations.
    - MANDATORY: Record system state at EXACTLY t = T_horizon:
        "horizon_state": {{
            "time_min": T_horizon,
            "patients_completed_within_horizon": int(...),
            "patients_remaining_in_system": int(...),
            "breakdown": {{"in_triage": int, "in_consult": int, "in_treatment": int,
                          "triage_queue": int, "consult_queue": int, "treatment_queue": int}}
        }}
    - If post-horizon drain is used, measure and report `clearance_time_minutes = final_discharge_time - T_horizon`.
    - FLAG WARNING if waiting times include patients who received service after T_horizon without labeling those outcomes separately.
    - Report MUST distinguish "24h operational metrics" from "full drain metrics" in section headers.

49. WORKLOAD_CAPACITY_AUDIT (Pre-Simulation Design Validation)
    - FLAG WARNING if the simulation JSON does not include a pre-simulation workload/capacity audit in `SIMULATION_RESULTS_JSON`.
    - MANDATORY fields under `"workload_audit"`:
        expected_arrivals_per_hour  = total_expected_arrivals / (T_horizon_minutes / 60)
        doctor_capacity_per_hour    = INITIAL_DOCTORS * 60.0 / mean_consult_target_minutes
        nurse_capacity_per_hour     = INITIAL_NURSES  * 60.0 / mean_triage_target_minutes
        bed_capacity_per_hour       = INITIAL_BEDS    * 60.0 / mean_treatment_target_minutes
        doctor_load_ratio  = expected_arrivals_per_hour / doctor_capacity_per_hour
        nurse_load_ratio   = expected_arrivals_per_hour / nurse_capacity_per_hour
        bed_load_ratio     = expected_arrivals_per_hour / bed_capacity_per_hour
    - If doctor_load_ratio > 1.0 or nurse_load_ratio > 1.0 or bed_load_ratio > 1.0:
        MUST declare in the JSON summary and in the report: "System is structurally overloaded at [STAGE]. Expected arrival rate (X/h) exceeds [STAGE] capacity (Y/h). Queue accumulation is expected."
    - This audit validates whether large observed queues are consistent with deterministic capacity arithmetic (they should be).

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
    response = llm_invoke_safe(messages, node_name="CRITIC")
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
        "language": target_lang,
    }

def executor_node(state: SimulationState) -> SimulationState: #node 4
    """
    Executor: Runs the AI-generated code inside the sandbox environment.
    Also extracts the structured SIMULATION_RESULTS_JSON block from stdout.
    """
    code = state.get("generated_code", "")
    num_lines = len(code.splitlines()) if code else 0
    log_step_start("EXECUTOR", f"Running simulation code ({num_lines} lines) in sandbox...")

    start_time = time.time()
    stdout, stderr, chart_base64 = execute_python_code(code)
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

    if not chart_base64:
        log_step_progress("Generating scientific posterior & outcome figure...")
        chart_base64 = generate_scientific_chart(
            hypothesis_analysis=state.get("hypothesis_analysis", ""),
            structured_results=structured_results,
            statistical_summary="",
            user_prompt=state.get("user_prompt", "")
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
            f"structured_results={'yes' if structured_results else 'no'}, "
            f"chart={'yes' if chart_base64 else 'no'}"
        )

    return {
        "execution_output": stdout,
        "execution_error": stderr,
        "structured_results": structured_results,
        "chart_image": chart_base64,
    }

def validator_node(state: SimulationState) -> SimulationState: #node 5
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

    has_error = bool(state.get("execution_error", "").strip()) #check 1
    _check(
        "execution_success",
        not has_error,
        "FAIL",
        "Execution produced stderr" if has_error else "Clean execution"
    )

    structured_raw = state.get("structured_results", "") #check 2
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

    metadata = data.get("metadata", {}) #check 3
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

    known_methods = { #check 4
        "deterministic_simulation", "stochastic_simulation", "monte_carlo",
        "discrete_event_simulation", "system_dynamics", "agent_based_simulation",
        "game_theoretic", "hybrid"
    }
    declared_method = metadata.get("methodology", "")
    method_known = declared_method in known_methods
    _check("methodology_known_value", method_known, "WARN",
           f"'{declared_method}' is not in the approved methodology taxonomy"
           if not method_known else f"methodology='{declared_method}'")

    outcomes = data.get("outcomes", {}) #check 5
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

    assumptions = data.get("assumptions", []) #check 6
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

    ca = data.get("constraint_application") or data.get("constraint_violations", {}) #check 7
    has_ca = isinstance(ca, dict) and len(ca) > 0
    _check("constraint_application_tracked", has_ca, "WARN",
           "No constraint_application block — can't compute constraint application probability"
           if not has_ca else f"{len(ca)} constraint(s) tracked")

    events = data.get("events", []) #check 8
    decisions = data.get("decisions", [])
    has_events = isinstance(events, list) and len(events) > 0
    has_decisions = isinstance(decisions, list) and len(decisions) > 0
    _check("events_logged", has_events, "WARN",
           "No events logged — analyst cannot explain what happened"
           if not has_events else f"{len(events)} event(s) logged")
    _check("decisions_logged", has_decisions, "WARN",
           "No decisions logged — analyst cannot explain WHY things changed"
           if not has_decisions else f"{len(decisions)} decision(s) logged")

    init_conds = data.get("initial_conditions", {}) #check 9
    ic_missing_prov = []
    for k, v in init_conds.items():
        if isinstance(v, dict) and not v.get("provenance"):
            ic_missing_prov.append(k)
    _check("initial_conditions_provenance", len(ic_missing_prov) == 0, "WARN",
           f"Missing provenance on initial conditions: {ic_missing_prov}"
           if ic_missing_prov else "All initial conditions have provenance labels")

    runs_val = metadata.get("runs", 0) #check 10
    runs_ok = isinstance(runs_val, int) and runs_val >= 100
    _check("sufficient_runs", runs_ok, "WARN",
           f"Only {runs_val} runs — recommend ≥ 100 for statistical reliability"
           if not runs_ok else f"{runs_val} runs — statistically adequate")

    fails = [c for c in checks if c["status"] == "FAIL"] #summary
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

STATS_I18N = {
    "English": {
        "title": "## Statistical Analysis (Python-computed)\n",
        "metadata": "### Experiment Metadata",
        "methodology_note": "Methodology note",
        "sim_summary": "### Simulation-Computed Summary",
        "statistic": "Statistic",
        "value": "Value",
        "mean": "Mean",
        "median": "Median",
        "std_dev": "Std Dev",
        "min": "Min",
        "max": "Max",
        "p10": "P10",
        "p25": "P25",
        "p75": "P75",
        "p90": "P90",
        "p_success": "P(success)",
        "p_failure": "P(failure)",
        "dist_type": "Distribution Type",
        "prop_se": "Proportion Std Error (SE)",
        "ci_95": "Monte Carlo 95% CI",
        "bernoulli_note": "Note: Metric is a discrete binary variable (Bernoulli distribution). Std Dev is sqrt(p*(1-p)) across discrete runs, NOT a forecast prediction error or confidence interval (do not report as X% ± Y%).",
        "ci_note": "Note: The 95% confidence interval reflects ONLY Monte Carlo numerical sampling uncertainty from the trial count (±1.96*SE), NOT a real-world predictive interval.",
        "sensitivity_header": "### Parameter Sensitivity Sweep (Python-computed)",
        "sens_param": "Parameter Value",
        "sens_rate": "Modeled Outcome Rate",
        "ca_header": "### Constraint Application / Enforcement Analysis (Python-computed)",
        "ca_constraint": "Constraint",
        "ca_mean_blocked": "Mean Blocked/Run",
        "ca_prob_blocked": "P(≥1 Blocked)",
        "ca_max_blocked": "Max Blocked",
        "events_header": "### Event Frequency (from representative run)",
        "event_type": "Event Type",
        "count": "Count",
        "decisions_header": "### Decision Frequency (from representative run)",
        "decision": "Decision",
        "scenario_header": "### Scenario Comparison",
        "scenario": "Scenario",
        "paired_header": "### Paired Comparison & Effect Size Analysis (Common Random Numbers)",
        "paired_metric": "Metric",
        "paired_diff": "Mean Paired Diff (Δ)",
        "paired_pct": "Relative Diff (%)",
        "paired_ci": "Paired 95% CI",
        "paired_win": "Adaptive Win Rate",
        "time_series_header": "### Time-Series Observation Metadata",
        "time_series_step": "Step Resolution",
        "time_series_count": "Total Observations",
        "time_series_horizon": "Horizon",
        "assumptions_header": "### Model Assumptions (from simulation — with provenance)",
        "parameter": "Parameter",
        "provenance": "Provenance",
        "justification": "Justification",
        "term_reason": "Termination reason",
        "no_results": "No structured results available for statistical analysis.",
        "parse_error": "Could not parse structured results.",
        "no_numeric": "*No numeric list metrics found in outcomes block.*\n",
    },
    "Chinese": {
        "title": "## 统计分析（Python原生计算）\n",
        "metadata": "### 实验元数据",
        "methodology_note": "方法论说明",
        "sim_summary": "### 模拟计算摘要",
        "statistic": "统计指标",
        "value": "数值",
        "mean": "均值 (Mean)",
        "median": "中位数 (Median)",
        "std_dev": "标准差 (Std Dev)",
        "min": "极小值 (Min)",
        "max": "极大值 (Max)",
        "p10": "10分位数 (P10)",
        "p25": "25分位数 (P25)",
        "p75": "75分位数 (P75)",
        "p90": "90分位数 (P90)",
        "p_success": "成功概率 P(success)",
        "p_failure": "失败概率 P(failure)",
        "dist_type": "分布类型 (Distribution Type)",
        "prop_se": "比例标准误 (Standard Error, SE)",
        "ci_95": "Monte Carlo 95% 抽样置信区间",
        "bernoulli_note": "说明：该指标为离散二元变量（伯努利分布）。标准差反映的是单次运行成功/不成功的离散方差 sqrt(p*(1-p))，绝非预测置信区间或预测误差（严禁写作 X% ± Y%）。",
        "ci_note": "说明：95% 置信区间仅反映当前模拟抽样次数带来的数值不确定性 (±1.96*SE)，绝不代表现实世界发生概率的 95% 预测区间。",
        "sensitivity_header": "### 参数敏感性分析（Python原生扫描计算）",
        "sens_param": "参数取值",
        "sens_rate": "模拟结局发生率",
        "ca_header": "### 约束应用与执行分析（Python原生计算）",
        "ca_constraint": "约束名称",
        "ca_mean_blocked": "单次运行平均阻断数",
        "ca_prob_blocked": "P(≥1次被阻断概率)",
        "ca_max_blocked": "最大阻断数",
        "events_header": "### 事件发生频率（来自代表性运行）",
        "event_type": "事件类型",
        "count": "频次",
        "decisions_header": "### 决策频率（来自代表性运行）",
        "decision": "决策内容",
        "scenario_header": "### 情景对比",
        "scenario": "情景名称",
        "paired_header": "### 配对比较与效应量分析（公共随机数 CRN）",
        "paired_metric": "指标",
        "paired_diff": "配对差值均值 (Δ)",
        "paired_pct": "相对差异 (%)",
        "paired_ci": "配对 95% 置信区间",
        "paired_win": "自适应胜率",
        "time_series_header": "### 时间序列观测元数据",
        "time_series_step": "时间步长",
        "time_series_count": "观测点数量",
        "time_series_horizon": "模拟时长",
        "assumptions_header": "### 模型假设（来自模拟 — 附来源追踪）",
        "parameter": "参数名称",
        "provenance": "来源 (Provenance)",
        "justification": "假设依据理由",
        "term_reason": "终止原因",
        "no_results": "暂无可用结构化结果进行统计分析。",
        "parse_error": "无法解析结构化结果。",
        "no_numeric": "*未在结果数据块中发现数值型列表指标。*\n",
    },
    "Spanish": {
        "title": "## Análisis Estadístico (Calculado con Python)\n",
        "metadata": "### Metadatos del Experimento",
        "methodology_note": "Nota metodológica",
        "sim_summary": "### Resumen Computado de la Simulación",
        "statistic": "Estadística",
        "value": "Valor",
        "mean": "Media",
        "median": "Mediana",
        "std_dev": "Desviación Estándar",
        "min": "Mín",
        "max": "Máx",
        "p10": "P10",
        "p25": "P25",
        "p75": "P75",
        "p90": "P90",
        "p_success": "P(éxito)",
        "p_failure": "P(fallo)",
        "dist_type": "Tipo de Distribución",
        "prop_se": "Error Estándar de la Proporción (SE)",
        "ci_95": "Intervalo de Confianza del 95% (Monte Carlo)",
        "bernoulli_note": "Nota: La variable es binaria discreta (distribución de Bernoulli). La desviación estándar es sqrt(p*(1-p)) entre ejecuciones discretas, NO un intervalo de predicción/confianza (no reportar como X% ± Y%).",
        "ci_note": "Nota: El intervalo del 95% refleja ÚNICAMENTE la incertidumbre del muestreo numérico de Monte Carlo (±1.96*SE), NO un intervalo de predicción del mundo real.",
        "sensitivity_header": "### Barrido de Sensibilidad de Parámetros (Calculado con Python)",
        "sens_param": "Valor del Parámetro",
        "sens_rate": "Tasa de Resultado Modelada",
        "ca_header": "### Análisis de Aplicación de Restricciones (Calculado con Python)",
        "ca_constraint": "Restricción",
        "ca_mean_blocked": "Media Bloqueada/Ejecución",
        "ca_prob_blocked": "P(≥1 Bloqueado)",
        "ca_max_blocked": "Máximo Bloqueado",
        "events_header": "### Frecuencia de Eventos (Ejecución Representativa)",
        "event_type": "Tipo de Evento",
        "count": "Cantidad",
        "decisions_header": "### Frecuencia de Decisiones (Ejecución Representativa)",
        "decision": "Decisión",
        "scenario_header": "### Comparación de Escenarios",
        "scenario": "Escenario",
        "assumptions_header": "### Supuestos del Modelo (Con Procedencia)",
        "parameter": "Parámetro",
        "provenance": "Procedencia",
        "justification": "Justificación",
        "term_reason": "Razón de terminación",
        "no_results": "No hay resultados estructurados disponibles para análisis estadístico.",
        "parse_error": "No se pudieron procesar los resultados estructurados.",
        "no_numeric": "*No se encontraron métricas numéricas en los resultados.*\n",
    },
    "French": {
        "title": "## Analyse Statistique (Calculée par Python)\n",
        "metadata": "### Métadonnées de l'Expérience",
        "methodology_note": "Note méthodologique",
        "sim_summary": "### Résumé Calculé de la Simulation",
        "statistic": "Statistique",
        "value": "Valeur",
        "mean": "Moyenne",
        "median": "Médiane",
        "std_dev": "Écart-type",
        "min": "Min",
        "max": "Max",
        "p10": "P10",
        "p25": "P25",
        "p75": "P75",
        "p90": "P90",
        "p_success": "P(succès)",
        "p_failure": "P(échec)",
        "dist_type": "Type de Distribution",
        "prop_se": "Erreur Standard de Proportion (SE)",
        "ci_95": "Intervalle de Confiance à 95% (Monte Carlo)",
        "bernoulli_note": "Note: Variable binaire discrète (distribution de Bernoulli). L'écart-type est sqrt(p*(1-p)), et NON un intervalle de prédiction/confiance (ne pas rapporter sous forme de X% ± Y%).",
        "ci_note": "Note: L'intervalle de 95% reflète UNIQUEMENT l'incertitude numérique d'échantillonnage de Monte Carlo (±1.96*SE), et NON un intervalle prédictif du monde réel.",
        "sensitivity_header": "### Balayage de Sensibilité des Paramètres (Calculé par Python)",
        "sens_param": "Valeur du Paramètre",
        "sens_rate": "Taux de Résultat Modélisé",
        "ca_header": "### Analyse de l'Application des Contraintes (Calculée par Python)",
        "ca_constraint": "Contrainte",
        "ca_mean_blocked": "Moyenne Bloquée/Exécution",
        "ca_prob_blocked": "P(≥1 Bloqué)",
        "ca_max_blocked": "Max Bloqué",
        "events_header": "### Fréquence des Événements (Exécution Représentative)",
        "event_type": "Type d'Événement",
        "count": "Nombre",
        "decisions_header": "### Fréquence des Décisions (Exécution Représentative)",
        "decision": "Décision",
        "scenario_header": "### Comparaison des Scénarios",
        "scenario": "Scénario",
        "assumptions_header": "### Hypothèses du Modèle (Avec Provenance)",
        "parameter": "Paramètre",
        "provenance": "Provenance",
        "justification": "Justification",
        "term_reason": "Raison de terminaison",
        "no_results": "Aucun résultat structuré disponible pour l'analyse statistique.",
        "parse_error": "Impossible de traiter les résultats structurés.",
        "no_numeric": "*Aucune métrique numérique trouvée dans les résultats.*\n",
    },
    "German": {
        "title": "## Statistische Analyse (Python-berechnet)\n",
        "metadata": "### Experiment-Metadaten",
        "methodology_note": "Methodischer Hinweis",
        "sim_summary": "### Berechnete Simulationszusammenfassung",
        "statistic": "Statistik",
        "value": "Wert",
        "mean": "Mittelwert",
        "median": "Median",
        "std_dev": "Standardabweichung",
        "min": "Min",
        "max": "Max",
        "p10": "P10",
        "p25": "P25",
        "p75": "P75",
        "p90": "P90",
        "p_success": "P(Erfolg)",
        "p_failure": "P(Fehlschlag)",
        "dist_type": "Verteilungstyp",
        "prop_se": "Standardfehler des Anteils (SE)",
        "ci_95": "Monte-Carlo 95% Konfidenzintervall",
        "bernoulli_note": "Hinweis: Diskrete binäre Variable (Bernoulli-Verteilung). Standardabweichung ist sqrt(p*(1-p)), KEIN Vorhersage-/Konfidenzintervall (nicht als X% ± Y% angeben).",
        "ci_note": "Hinweis: Das 95%-Konfidenzintervall spiegelt NUR die numerische Stichprobenunsicherheit von Monte Carlo wider (±1.96*SE), KEIN reales Vorhersageintervall.",
        "sensitivity_header": "### Parametersensitivitätsanalyse (Python-berechnet)",
        "sens_param": "Parameterwert",
        "sens_rate": "Modellierte Ergebnisrate",
        "ca_header": "### Analyse der Einschränkungsanwendung (Python-berechnet)",
        "ca_constraint": "Einschränkung",
        "ca_mean_blocked": "Mittelwert Blockiert/Lauf",
        "ca_prob_blocked": "P(≥1 Blockiert)",
        "ca_max_blocked": "Max Blockiert",
        "events_header": "### Ereignishäufigkeit",
        "event_type": "Ereignistyp",
        "count": "Anzahl",
        "decisions_header": "### Entscheidungshäufigkeit",
        "decision": "Entscheidung",
        "scenario_header": "### Szenarienvergleich",
        "scenario": "Szenario",
        "assumptions_header": "### Modellannahmen (Mit Herkunft)",
        "parameter": "Parameter",
        "provenance": "Herkunft",
        "justification": "Begründung",
        "term_reason": "Beendigungsgrund",
        "no_results": "Keine strukturierten Ergebnisse für die statistische Analyse verfügbar.",
        "parse_error": "Strukturierte Ergebnisse konnten nicht verarbeitet werden.",
        "no_numeric": "*Keine numerischen Metriken gefunden.*\n",
    },
    "Japanese": {
        "title": "## 統計分析（Python計算）\n",
        "metadata": "### 実験メタデータ",
        "methodology_note": "方法論ノート",
        "sim_summary": "### シミュレーション計算サマリー",
        "statistic": "統計指標",
        "value": "値",
        "mean": "平均値 (Mean)",
        "median": "中央値 (Median)",
        "std_dev": "標準偏差 (Std Dev)",
        "min": "最小値 (Min)",
        "max": "最大値 (Max)",
        "p10": "10パーセンタイル (P10)",
        "p25": "25パーセンタイル (P25)",
        "p75": "75パーセンタイル (P75)",
        "p90": "90パーセンタイル (P90)",
        "p_success": "成功確率 P(success)",
        "p_failure": "失敗確率 P(failure)",
        "dist_type": "分布型 (Distribution Type)",
        "prop_se": "比率の標準誤差 (Standard Error, SE)",
        "ci_95": "モンテカルロ 95% サンプリング信頼区間",
        "bernoulli_note": "説明: この指標は離散2値変数（ベルヌーイ分布）です。標準偏差は試行間の離散分散 sqrt(p*(1-p)) であり、予測誤差や信頼区間ではありません（X% ± Y% と表記しないでください）。",
        "ci_note": "説明: 95%信頼区間は試行回数に由来するモンテカルロ数値サンプリングの不確かさ（±1.96*SE）のみを反映しており、現実世界における95%予測区間ではありません。",
        "sensitivity_header": "### パラメータ感度分析（Pythonスキャン計算）",
        "sens_param": "パラメータ値",
        "sens_rate": "モデル化された結果発生率",
        "ca_header": "### 制約適用・強制分析（Python計算）",
        "ca_constraint": "制約名",
        "ca_mean_blocked": "実行あたりの平均ブロック数",
        "ca_prob_blocked": "P(≥1回ブロック確率)",
        "ca_max_blocked": "最大ブロック数",
        "events_header": "### イベント頻度（代表的試行）",
        "event_type": "イベント種別",
        "count": "回数",
        "decisions_header": "### 意思決定頻度（代表的試行）",
        "decision": "意思决定内容",
        "scenario_header": "### シナリオ比較",
        "scenario": "シナリオ名",
        "assumptions_header": "### モデル前提条件（出典・根拠付き）",
        "parameter": "パラメータ名",
        "provenance": "出典 (Provenance)",
        "justification": "前提理由",
        "term_reason": "終了理由",
        "no_results": "統計分析に利用可能な構造化結果がありません。",
        "parse_error": "構造化結果を解析できませんでした。",
        "no_numeric": "*数値指標が見つかりませんでした。*\n",
    }
}

def stats_node(state: SimulationState) -> SimulationState: #node 6
    """
    Stats: Computes Python-native statistics from the structured simulation results.
    Mean, median, std, percentiles, constraint violation probability, event
    frequency — calculated in Python, not by the LLM.
    Supports multi-language localized markdown output.
    """
    target_lang = _get_language(state)
    log_step_start(
        "STATS",
        f"Computing statistical summary [{target_lang}] from structured simulation results..."
    )
    start_time = time.time()

    t = STATS_I18N.get(target_lang, STATS_I18N["English"])

    structured = state.get("structured_results", "")
    if not structured:
        log_step_warning("STATS", "No structured results — skipping statistical analysis")
        return {"statistical_summary": t["no_results"], "language": target_lang}

    try:
        data = json.loads(structured)
    except json.JSONDecodeError:
        log_step_warning("STATS", "Could not parse structured_results as JSON")
        return {"statistical_summary": t["parse_error"], "language": target_lang}

    lines: list[str] = []

    sim_summary = data.get("summary", {})
    if sim_summary:
        lines.append(t["sim_summary"])
        for k, v in sim_summary.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    outcomes = data.get("outcomes", {}) #per metric summary statistic 
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

                unique = set(nums) # Check if binary 0/1 indicator
                is_binary = unique.issubset({0, 1, 0.0, 1.0})

                lines.append(f"### {metric.replace('_', ' ').title()} (n={n})")
                lines.append(f"| {t['statistic']} | {t['value']} |")
                lines.append("|-----------|-------|")

                if is_binary:
                    p_val = mean_v
                    k_count = sum(1 for x in nums if x == 1 or x == 1.0)
                    se = (p_val * (1 - p_val) / n) ** 0.5 if n > 0 else 0.0
                    margin_95 = 1.96 * se
                    ci_lower = max(0.0, p_val - margin_95)
                    ci_upper = min(1.0, p_val + margin_95)

                    lower_metric = metric.lower()
                    is_shortfall = any(kw in lower_metric for kw in ("fail", "loss", "below", "shortfall", "drop", "dip", "down", "sub_target", "breach"))

                    if is_shortfall:
                        label_primary = "Probability of Condition / Shortfall" if target_lang == "English" else ("短缺/条件发生概率" if target_lang == "Chinese" else "P(condición/déficit)")
                        label_opposite = "Probability of Normal / Meeting Target" if target_lang == "English" else ("正常/达成目标概率" if target_lang == "Chinese" else "P(normal/meta)")
                    else:
                        label_primary = t.get("p_success", "P(success)")
                        label_opposite = t.get("p_failure", "P(failure)")

                    lines.append(f"| {label_primary} | {p_val:.1%} ({k_count}/{n} runs) |")
                    lines.append(f"| {label_opposite} | {1 - p_val:.1%} ({n - k_count}/{n} runs) |")
                    lines.append(f"| {t.get('prop_se', 'Monte Carlo Standard Error (SE)')} | ±{se:.4f} (±{se:.2%}) |")
                    lines.append(f"| {t.get('ci_95', 'Monte Carlo 95% CI')} | [{ci_lower:.1%}, {ci_upper:.1%}] (±{margin_95:.2%}) |")
                    lines.append("")
                    lines.append(f"> *Note: The 95% confidence interval reflects numerical sampling uncertainty from the trial count (±1.96*SE), NOT a real-world predictive interval.*")
                else:
                    lines.append(f"| {t['mean']}      | {mean_v:.4f} |")
                    lines.append(f"| {t['median']}    | {median_v:.4f} |")
                    lines.append(f"| {t['std_dev']}   | {std_v:.4f} |")
                    lines.append(f"| {t['min']}       | {min_v:.4f} |")
                    lines.append(f"| {t['max']}       | {max_v:.4f} |")
                    lines.append(f"| {t['p10']}       | {p10:.4f} |")
                    lines.append(f"| {t['p25']}       | {p25:.4f} |")
                    lines.append(f"| {t['p75']}       | {p75:.4f} |")
                    lines.append(f"| {t['p90']}       | {p90:.4f} |")
                    lines.append("")
            except Exception as e:
                log_step_progress(f"Stats failed for '{metric}': {e}")

        if not computed_any:
            lines.append(t["no_numeric"])

    loss_attr = data.get("loss_attribution")
    if isinstance(loss_attr, dict) and loss_attr:
        lines.append("### Production Loss Attribution (Python-computed)")
        lines.append("| Factor | Value (Units) | Description |")
        lines.append("|--------|--------------:|-------------|")
        if "downtime_loss_units" in loss_attr:
            lines.append(f"| Downtime Loss | {loss_attr['downtime_loss_units']:.2f} | Production lost during complete line stoppages |")
        if "ideal_uptime_production_units" in loss_attr:
            lines.append(f"| Ideal Production during Uptime | {loss_attr['ideal_uptime_production_units']:.2f} | Expected production at nominal rate while operational |")
        if "actual_uptime_production_units" in loss_attr:
            lines.append(f"| Actual Production during Uptime | {loss_attr['actual_uptime_production_units']:.2f} | Observed production after stochastic speed variations |")
        if "net_rate_variation_effect_units" in loss_attr:
            lines.append(f"| Net Rate Variation Effect | {loss_attr['net_rate_variation_effect_units']:+.2f} | Net deviation from nominal rate (actual - ideal) |")
        lines.append("")

    exposure = data.get("exposure")
    if isinstance(exposure, dict) and exposure:
        lines.append("### Machine Operating Exposure & Reliability Audit (Python-computed)")
        lines.append("| Metric | Value | Provenance / Standard |")
        lines.append("|--------|------:|-----------------------|")
        if "expected_upper_bound_failures" in exposure:
            lines.append(f"| Theoretical Failure Exposure Upper Bound | {exposure['expected_upper_bound_failures']:.2f} | Total possible failures if 0 downtime |")
        if "mean_operating_hours" in exposure:
            lines.append(f"| Mean Machine Operating Exposure | {exposure['mean_operating_hours']:.2f} h | Active operating hours (excluding repair time) |")
        if "mean_failures" in exposure:
            lines.append(f"| Observed Mean Failures | {exposure['mean_failures']:.2f} | Actual failures during operating hours |")
        if "failure_rate_per_operating_hour" in exposure:
            lines.append(f"| Observed Failure Rate / Operating Hour | {exposure['failure_rate_per_operating_hour']:.4f} | Failures per machine operating hour |")
        lines.append("")

    ca = data.get("constraint_application") or data.get("constraint_violations", {}) #constraint application
    if isinstance(ca, dict) and ca:
        lines.append(t["ca_header"])
        lines.append(f"| {t['ca_constraint']} | {t['ca_mean_blocked']} | {t['ca_prob_blocked']} | {t['ca_max_blocked']} |")
        lines.append("|------------|-----------------:|--------------:|------------:|")
        for constraint_name, counts in ca.items():
            if not isinstance(counts, list) or len(counts) == 0:
                continue
            nums = [x for x in counts if isinstance(x, (int, float))]
            if not nums:
                continue
            mean_v = _stats.mean(nums)
            prob_any = sum(1 for x in nums if x > 0) / len(nums)
            max_v = max(nums)
            lines.append(
                f"| {constraint_name} | {mean_v:.2f} | {prob_any:.1%} | {int(max_v)} |"
            )
        lines.append("")

    events = data.get("events", []) #event freq statistics
    if isinstance(events, list) and events:
        event_types: dict[str, int] = {}
        for ev in events:
            if isinstance(ev, dict):
                evt = ev.get("event", "unknown")
                event_types[evt] = event_types.get(evt, 0) + 1
        if event_types:
            lines.append(t["events_header"])
            lines.append(f"| {t['event_type']} | {t['count']} |")
            lines.append("|------------|------:|")
            for evt, count in sorted(event_types.items(), key=lambda x: -x[1]):
                lines.append(f"| {evt} | {count} |")
            lines.append("")

    decisions = data.get("decisions", []) #decision frequency
    if isinstance(decisions, list) and decisions:
        decision_types: dict[str, int] = {}
        for d in decisions:
            if isinstance(d, dict):
                dt = d.get("decision", "unknown")
                decision_types[dt] = decision_types.get(dt, 0) + 1
        if decision_types:
            lines.append(t["decisions_header"])
            lines.append(f"| {t['decision']} | {t['count']} |")
            lines.append("|----------|------:|")
            for dt, count in sorted(decision_types.items(), key=lambda x: -x[1]):
                lines.append(f"| {dt} | {count} |")
            lines.append("")

    scenario_comparison = data.get("scenario_comparison", {}) #scenario comparison
    if isinstance(scenario_comparison, dict) and scenario_comparison:
        # Separate paired_comparison if present inside scenario_comparison
        sc_scenarios = {k: v for k, v in scenario_comparison.items() if k != "paired_comparison" and isinstance(v, dict)}
        if sc_scenarios:
            lines.append(t["scenario_header"])
            # Collect all metric names
            all_metrics: set[str] = set()
            for sc_data in sc_scenarios.values():
                if isinstance(sc_data, dict):
                    all_metrics.update(sc_data.keys())
            if all_metrics:
                metric_list = sorted(all_metrics)
                header = f"| {t['scenario']} | " + " | ".join(metric_list) + " |"
                separator = "|----------|" + "|".join(["-" * max(len(m), 6) for m in metric_list]) + "|"
                lines.append(header)
                lines.append(separator)
                for sc_name, sc_data in sc_scenarios.items():
                    if isinstance(sc_data, dict):
                        row_vals = []
                        for m in metric_list:
                            v = sc_data.get(m, "—")
                            row_vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                        lines.append(f"| {sc_name} | " + " | ".join(row_vals) + " |")
                lines.append("")

    # Paired Comparison & Effect Size Analysis under Common Random Numbers (CRN)
    paired_comp = data.get("paired_comparison")
    if not paired_comp and isinstance(scenario_comparison, dict):
        paired_comp = scenario_comparison.get("paired_comparison")
    
    if isinstance(paired_comp, dict) and paired_comp:
        lines.append(t.get("paired_header", "### Paired Comparison & Effect Size Analysis (Common Random Numbers)"))
        lines.append(f"| {t.get('paired_metric', 'Metric')} | {t.get('paired_diff', 'Mean Paired Diff (Δ)')} | {t.get('paired_pct', 'Relative Diff (%)')} | {t.get('paired_ci', 'Paired 95% CI')} | {t.get('paired_win', 'Adaptive Win Rate')} |")
        lines.append("|:----------------------|-----------------------:|-------------------:|----------------------:|------------------:|")
        
        metric_configs = [
            ("Wait Time", ["delta_wait", "paired_delta_wait", "wait_difference"], "mean_delta_wait", "relative_delta_wait_pct", "paired_trial_win_rate_wait", " s", True),
            ("Max Queue", ["delta_queue", "delta_max_queue", "paired_delta_queue"], "mean_delta_queue", "relative_delta_queue_pct", "paired_trial_win_rate_queue", " veh", True),
            ("Throughput", ["delta_throughput", "paired_delta_throughput"], "mean_delta_throughput", "relative_delta_throughput_pct", "paired_trial_win_rate_throughput", " veh", False),
        ]
        
        for m_label, arr_aliases, mean_key, pct_key, win_key, unit, lower_is_better in metric_configs:
            delta_arr = None
            for alias in arr_aliases:
                if alias in paired_comp and isinstance(paired_comp[alias], list):
                    delta_arr = paired_comp[alias]
                    break
            
            if isinstance(delta_arr, list) and delta_arr:
                nums = [x for x in delta_arr if isinstance(x, (int, float)) and not _is_nan(x)]
                if nums:
                    mean_val = _stats.mean(nums)
                    std_val = _stats.stdev(nums) if len(nums) > 1 else 0.0
                    se_val = std_val / math.sqrt(len(nums)) if len(nums) > 0 else 0.0
                    ci_lo = mean_val - 1.96 * se_val
                    ci_hi = mean_val + 1.96 * se_val
                    
                    pct_val = paired_comp.get(pct_key)
                    pct_str = f"{pct_val:+.2f}%" if isinstance(pct_val, (int, float)) else "—"
                    
                    win_val = paired_comp.get(win_key)
                    if win_val is not None and isinstance(win_val, (int, float)):
                        win_rate_str = f"{win_val:.1%}"
                    else:
                        win_count = sum(1 for x in nums if (x < 0 if lower_is_better else x > 0))
                        win_rate_str = f"{win_count / len(nums):.1%} ({win_count}/{len(nums)})"
                        
                    lines.append(f"| {m_label} | {mean_val:+.2f}{unit} | {pct_str} | [{ci_lo:+.2f}, {ci_hi:+.2f}]{unit} | {win_rate_str} |")
        lines.append("")
        lines.append(f"> *Note: Paired comparisons evaluate exact trial-by-trial differences (Adaptive - Fixed) under Common Random Numbers (CRN), eliminating stochastic arrival variance.*")
        lines.append("")

    # Time-Series Observation Metadata
    ts_res = data.get("time_series_resolution", {})
    ts_data = data.get("time_series", {})
    if isinstance(ts_res, dict) and ts_res:
        lines.append(t.get("time_series_header", "### Time-Series Observation Metadata"))
        step_sec = ts_res.get("time_step_sec", "—")
        step_unit = ts_res.get("step_unit", "second")
        tot_obs = ts_res.get("total_observations", len(next(iter(ts_data.values()))) if isinstance(ts_data, dict) and ts_data else "—")
        horizon = ts_res.get("horizon_minutes", "—")
        lines.append(f"- **{t.get('time_series_step', 'Step Resolution')}**: {step_sec} s ({step_unit})")
        lines.append(f"- **{t.get('time_series_count', 'Total Observations')}**: {tot_obs}")
        if horizon != "—":
            lines.append(f"- **{t.get('time_series_horizon', 'Horizon')}**: {horizon} minutes")
        lines.append("")

    sensitivity = data.get("sensitivity_sweep") or data.get("sensitivity_analysis", {}) # parameter sensitivity sweep
    if isinstance(sensitivity, dict) and sensitivity:
        param_name = sensitivity.get("parameter", "Parameter")
        points = sensitivity.get("points") or sensitivity.get("sweep_points", [])
        if isinstance(points, list) and points:
            lines.append(t["sensitivity_header"])
            lines.append(f"| {t['sens_param']} ({param_name}) | {t['sens_rate']} |")
            lines.append("|-----------------------------:|----------------------:|")
            for pt in points:
                if isinstance(pt, dict):
                    pval = pt.get("value", pt.get("param_value", "—"))
                    rate = pt.get("rate", pt.get("outcome_rate", pt.get("mean_wait", pt.get("metric_value", "—"))))
                    pval_str = f"{pval:.2f}" if isinstance(pval, float) else str(pval)
                    if isinstance(rate, float):
                        # Format as percentage only if it represents a probability/rate between 0 and 1
                        if 0.0 <= rate <= 1.0 and ("rate" in pt or "probability" in pt or "prob" in str(param_name).lower()):
                            rate_str = f"{rate:.1%}"
                        else:
                            rate_str = f"{rate:,.2f}"
                    else:
                        rate_str = str(rate)
                    lines.append(f"| {pval_str} | {rate_str} |")
            lines.append("")
        elif isinstance(points, dict) and points:
            lines.append(t["sensitivity_header"])
            lines.append(f"| {t['sens_param']} ({param_name}) | {t['sens_rate']} |")
            lines.append("|-----------------------------:|----------------------:|")
            for pval, rate in points.items():
                if isinstance(rate, float):
                    if 0.0 <= rate <= 1.0 and ("probability" in str(param_name).lower() or "prob" in str(param_name).lower()):
                        rate_str = f"{rate:.1%}"
                    else:
                        rate_str = f"{rate:,.2f}"
                else:
                    rate_str = str(rate)
                lines.append(f"| {pval} | {rate_str} |")
            lines.append("")

    assumptions = data.get("assumptions", []) #assumption provenance table
    if assumptions:
        lines.append(t["assumptions_header"])
        lines.append(f"| {t['parameter']} | {t['value']} | {t['provenance']} | {t['justification']} |")
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
 
    term = data.get("termination_reason", "") #termination
    if term:
        lines.append(f"**{t['term_reason']}:** {term}\n")

    elapsed = time.time() - start_time
    summary_text = "\n".join(lines)
    log_step_success(
        "STATS",
        elapsed,
        f"Statistical summary generated ({len(summary_text)} chars, [{target_lang}])"
    )

    chart_img = state.get("chart_image", "")
    if not chart_img:
        chart_img = generate_scientific_chart(
            hypothesis_analysis=state.get("hypothesis_analysis", ""),
            structured_results=structured,
            statistical_summary=summary_text,
            user_prompt=state.get("user_prompt", "")
        )

    return {
        "statistical_summary": summary_text,
        "language": target_lang,
        "chart_image": chart_img
    }

def analyst_node(state: SimulationState) -> SimulationState: #node 7
    """
    Analyst: Interprets simulation results and generates a rigorous
    20-section scientific research report.
    The 20th section covers Bayesian hypothesis update analysis.
    """
    target_lang = _get_language(state)
    analyst_sim_mode = _classify_simulation_mode(state)

    if analyst_sim_mode == "DIRECT_MC":
        log_step_start(
            "ANALYST",
            f"Generating focused scientific report [{target_lang}] using '{settings.MODEL_NAME}'..."
        )
    else:
        log_step_start(
            "ANALYST",
            f"Generating 20-section scientific report [{target_lang}] using '{settings.MODEL_NAME}'..."
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

    # In DIRECT_MC mode, completely omit the Bayesian hypothesis block
    hypothesis_section = ""
    if analyst_sim_mode == "HYPOTHESIS" and state.get('hypothesis_analysis'):
        hypothesis_section = f"""
--------------------------------------------------
BAYESIAN HYPOTHESIS ANALYSIS
(use for Section 20 — do not invent new probabilities)
--------------------------------------------------
{state.get('hypothesis_analysis', '')}
"""

    lang_instruction = f"""
═══════════════════════════════════════════════════
LANGUAGE REQUIREMENT (MANDATORY — HIGHEST PRIORITY)
═══════════════════════════════════════════════════

The user prompt/scenario is written in: '{target_lang}'.
You MUST generate the ENTIRE 20-section research report in the EXACT SAME LANGUAGE as the user scenario ({target_lang}).
- Translate all 20 section headings and contents into professional, scientific research headings in {target_lang}.
  Example section titles:
  If Chinese:
    # 模拟实验分析报告
    ## 执行摘要
    ## 模拟目标
    ## 初始条件
    ## 实体与状态变量
    ## 模型参数
    ## 假设与参数来源
    ## 模拟方法论
    ## 决策策略 / 智能体行为
    ## 模拟动态机制
    ## 事件时间线
    ## 统计结果
    ## 约束应用与执行分析
    ## 情景对比
    ## 敏感性与不确定性分析
    ## 核心发现
    ## 决策分析
    ## 局限性与未建模因素
    ## 科学结论
    ## 建议
    ## 假设更新分析
  If Spanish:
    # Informe de Análisis del Experimento de Simulación
    ## Resumen Ejecutivo
    ## Objetivo de la Simulación
    ## Condiciones Iniciales
    ## Entidades y Variables de Estado
    ## Parámetros del Modelo
    ## Supuestos y Procedencia de Parámetros
    ## Metodología de Simulación
    ## Políticas de Decisión / Comportamientos de Agentes
    ## Dinámica de la Simulación
    ## Cronología de Eventos
    ## Resultados Estadísticos
    ## Análisis de Aplicación y Cumplimiento de Restricciones
    ## Comparación de Escenarios
    ## Análisis de Sensibilidad e Incertidumbre
    ## Hallazgos Clave
    ## Análisis de Decisiones
    ## Limitaciones y Factores no Modelados
    ## Conclusión Científica
    ## Recomendaciones
    ## Análisis de Actualización de Hipótesis
  If French, German, Japanese, Korean, Russian, Arabic, or any other language:
    Translate the 20 section headings and all analysis into that language accordingly.

- Terminology in {target_lang}:
  - Use the equivalent of "MODELED LEADING SCENARIO" (e.g. "模型主导情景" in Chinese, "ESCENARIO PRINCIPAL MODELADO" in Spanish), NEVER "DETERMINATION" or "定论".
  - Use the equivalent of "most frequently successful modeled pathway" (e.g. "模型中最常成功的路径" in Chinese), NEVER "validated pathway" or "验证通过的路径".
  - Provenance tags: use clear labels such as [USER_PROVIDED / 用户提供], [MODEL_ASSUMPTION / 模型假设], [DERIVED / 衍生计算], [SIMULATION_OUTPUT / 模拟输出], [AI_INTERPRETATION / AI解释].
  - Clearly distinguish Bayesian posterior from simulation eligibility.
  - Express Monte Carlo outcomes as observed success rates (e.g. "观察到的成功率: X% ± Y%").
"""

    # Determine mode for analyst report behaviour
    analyst_sim_mode = _classify_simulation_mode(state)

    # Mode-specific instructions injected into the analyst prompt
    if analyst_sim_mode == "DIRECT_MC":
        report_title_goal = f"Produce a rigorous, focused, high-credibility scientific research report in {target_lang}."
        analyst_mode_block = """
═══════════════════════════════════════════════════
SIMULATION MODE FOR THIS REPORT: DIRECT_MC
═══════════════════════════════════════════════════

The simulation was a DIRECT Monte Carlo / Discrete-Event run — NO competing hypotheses were requested.
Do NOT output unrequested sections such as Decision Policies, Agent Behaviors, Event Timelines, Constraint Application tables, Scenario Comparison, Decision Analysis, or Recommendations.
Report results as plain empirical statistics without Bayesian hypothesis framing or synthetic likelihood ratios.
"""
        report_structure_instruction = f"""
═══════════════════════════════════════════════════
REPORT STRUCTURE — FOCUSED SCIENTIFIC REPORT (DIRECT SIMULATION)
═══════════════════════════════════════════════════

The simulation was a direct model run with user-specified parameters — NO competing hypotheses were requested.
Do NOT output unrequested sections such as Decision Policies, Agent Behaviors, Event Timelines, Constraint Application tables, Scenario Comparison, Decision Analysis, Recommendations, or headings titled "Statistical Analysis (Python-computed)" or "Experiment Metadata".
Produce ONLY the following focused, highly credible, professional scientific report in {target_lang}:

# Simulation Experiment Analysis

## Executive Summary
- Concise 2–3 sentence overview: system modeled, simulation horizon and trial count, primary numerical outcomes (mean/median production, downtime, sub-target rate), and core takeaway.
- Do NOT mention "MODELED LEADING SCENARIO" (no hypothesis competition).

## Simulation Configuration
- Parameter Table:
  | Parameter | Value | Provenance | Description / Justification |
  (Every parameter must have an honest provenance label: [USER_PROVIDED], [MODEL_ASSUMPTION], [DERIVED]).
  - Explicitly explain any assumptions: e.g. repair dispersion sigma = 0.6 is a model assumption required because the user specified the median (30 min) but not the dispersion.

## Methodology
- Model classification: State the exact model type (e.g. Discrete-Event Simulation / Fractional-Time Monte Carlo).
- Explain event advancement mechanics (continuous/fractional time tracking), machine failure/repair distribution sampling, line-stopping logic, and operating exposure tracking.
- Highlight that repair times are tracked at exact fractional resolution (e.g. 18 min = 0.30 h line stoppage, NOT 1 whole hour), preventing integer-hour discretization bias.

## Results
Present the Python-computed numerical statistics directly from PRE-COMPUTED STATISTICAL EVIDENCE:
### Production Performance
- Table of empirical production statistics: Mean, Median, Standard Deviation, P5, P95.
### Target Analysis
- Sub-target condition: Probability of production falling below threshold (e.g. < 65,000 units).
- State: Observed probability below target = X% (k/n runs).
- Monte Carlo Standard Error (SE) = ±Y%.
- 95% Monte Carlo Confidence Interval = [Lower%, Upper%].
- Clarify that this CI reflects numerical Monte Carlo sampling uncertainty from the trial count, NOT a real-world predictive interval.
- NEVER call this a "Success Rate".
### Downtime Analysis
- Table of total downtime across runs: Mean, Median, P5, P95 (in fractional hours).
### Exposure and Failure Rates
- Table: Theoretical Upper-Bound Failure Exposure, Mean Machine Operating Exposure (hours), Observed Mean Failures, Observed Failure Rate per Machine Operating Hour.
- Confirm that machines do not fail during repair, so observed failures remain bounded by active operating exposure.

## Production Loss Attribution
- Present the breakdown table from Python stats:
  | Loss Driver | Impact (Units) | Contribution / Interpretation |
  - Complete line downtime loss (downtime_hours * nominal_rate)
  - Ideal production during uptime (uptime_hours * nominal_rate)
  - Actual production during uptime
  - Net rate-variation effect (actual uptime - ideal uptime; accounting symmetrically for both positive and negative fluctuations)

## Sensitivity Analysis
- Parameter sensitivity sweep table across parameter values (e.g. failure probability or repair time).
- Confirm monotonic behavior under Common Random Numbers (CRN).

## Assumptions
- Explicit table of all MODEL_ASSUMPTION parameters, why they were chosen, and their sensitivity impact.

## Limitations
- State unmodeled real-world factors (e.g. buffer stock between machines, preventive maintenance, operator shifts).
- Explicitly state that results represent conditional outputs within the model assumption space, not real-world predictive forecasts.

## Scientific Interpretation
- Scientific synthesis of the findings without Bayesian jargon.
- Epistemic note: "No competing hypotheses were requested. The simulation therefore uses the direct production-line model specified by the user."
- Do NOT output any prior/posterior probability tables or fake likelihood ratios.
"""
    else:
        report_title_goal = f"Produce a rigorous, evidence-based 20-section research report in {target_lang}."
        analyst_mode_block = """
═══════════════════════════════════════════════════
SIMULATION MODE FOR THIS REPORT: HYPOTHESIS
═══════════════════════════════════════════════════

The simulation modeled competing hypotheses. Apply the full 20-section structure
including the Bayesian 4-stage pipeline in Section 20.
H1/H2 observed success rates shown in the report must be conditional probabilities
P(success | Hi), not joint probabilities P(Hi ∩ success).
"""
        report_structure_instruction = """
═══════════════════════════════════════════════════
REPORT STRUCTURE — PRODUCE ALL 20 SECTIONS
═══════════════════════════════════════════════════

# Simulation Experiment Analysis Report

## Executive Summary
2–3 sentences: what was simulated, the MODELED LEADING SCENARIO (never write 'DETERMINATION' or 'Most likely attack scenario'), the most frequently successful modeled pathway (never 'validated pathway'), and key caveats.

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

## Constraint Application and Enforcement Analysis
Present constraint application and enforcement data from the pre-computed stats.
Report P(constraint blocked) or enforcement frequency for each tracked constraint under "constraint_application".
Explain that hard constraints (such as physical access alibi) deterministically block incompatible branches before sampling.
Clarify that parameters belonging to blocked branches (e.g. P_MODEL_PHYSICAL_ACCESS = 0.05) are NOT evaluated during simulation trials.

## Scenario Comparison
If multiple strategies or scenarios were simulated (e.g. Fixed-Time vs Adaptive Signal Control):
- Present the scenario comparison table from pre-computed stats.
- PAIRED TRIAL INFERENCE (CRN): Present the Paired Comparison & Effect Size table:
  * Mean paired difference (mean_delta = Adaptive - Fixed) for wait time, queue length, and throughput.
  * Relative percentage difference (% delta).
  * Paired 95% Confidence Interval ([mean_delta - 1.96 * SE, mean_delta + 1.96 * SE]).
  * Paired trial win rate (fraction of runs where Adaptive outperformed Fixed).
- Explain that Common Random Numbers (CRN) eliminate stochastic arrival realization noise by subjecting both policies to identical vehicle streams, allowing precise paired hypothesis testing.
- If only independent group statistics were provided, explicitly note the limitation and report the paired win rate.
- Flag any asymmetry in model assumptions between scenarios (e.g. differing service rates).

## Sensitivity and Uncertainty Analysis
Two dimensions of uncertainty to distinguish:

**Parameter sensitivity** (what MODEL_ASSUMPTION parameters most move the result):
- Present the parameter sensitivity sweep table from pre-computed stats.
- PHYSICAL MONOTONICITY & REGIME VALIDATION:
  * Check whether the outcome metric behaves monotonically with parameter variation across operating regimes.
- Identify the top parameters that, if changed, would shift the decision boundary.

**Scenario uncertainty** (what additional evidence would change the dominant hypothesis):
- Identify the 1–2 pieces of evidence that, if known, would most shift the hypothesis ranking.

## Key Findings
For each finding:
- **Observed [SIMULATION_OUTPUT]:** What the simulation directly produced (with numbers).
- **Interpretation [AI_INTERPRETATION]:** What that observation means within the model framework.
- **Caveat:** What the finding depends on (assumptions, control policies, circularity risks).

## Decision Analysis
For each available decision or strategy:
- Benefits (under the modeled assumptions)
- Risks and downside outcomes
- Trade-offs
- Conditions under which it is the better choice
If no universally optimal strategy exists, explain why with evidence.

## Limitations and Unmodeled Factors
Explicitly list:
- Assumption-Driven Outcomes & Risk of Circularity.
- Absence of Real-World Predictive Validity.
- Variables not modeled.
- Simplified assumptions.
- Parameters described as MODEL_ASSUMPTION that could significantly affect results.
- Methodology limitations.

## Scientific Conclusion
What does the simulation actually demonstrate?
MANDATORY REQUIRED PHRASING (use the active language equivalent):
  - In Chinese: "在当前先验、似然比、行为参数及硬约束下，模拟结果倾向于 [Hx] 作为模型中的主导情景。该结果反映的是模型假设空间内的条件性结果，而非对现实关系发展的概率预测。"
  - In English: "Under the specified priors, likelihood ratios, behavioral parameters, and hard constraints, the simulation points to [Hx] as the modeled leading scenario. This outcome reflects conditional dynamics within the model assumption space, rather than an empirical probability forecast of real-world outcomes."
NEVER claim the simulation "proves", "confirms", "validates", or "predicts" real-world relationship or event occurrences.

## Recommendations
Each recommendation must:
- State the assumption(s) it depends on.
- State what evidence supports it.
- State what would change the recommendation.
If evidence is insufficient for a strong recommendation, say so explicitly.

## Hypothesis Update Analysis
This section presents the Bayesian belief updating from the BAYESIAN HYPOTHESIS ANALYSIS above.

**The 4-Stage Architectural Pipeline:**
Explicitly delineate and contrast the 4 stages:
1. Prior Beliefs: Base rates before evidence [MODEL_ASSUMPTION].
2. Model Likelihood Ratios (LR): Subjective likelihood multipliers [MODEL_ASSUMPTION] (uncalibrated model levers, NOT empirical constants).
3. Posterior Beliefs: Mathematical update under assumed LR [DERIVED], representing the initial branch sampling weight in the model.
4. Within-Branch Simulation Dynamics -> Observed Pathway Rate: Percentage of simulated runs satisfying the outcome condition once modeled under the code's dynamic rules.

**Prior beliefs (before evidence):**
Present a table: Hypothesis | Prior Probability [MODEL_ASSUMPTION] | Description

**Evidence applied:**
For each piece of evidence in the evidence_log:
- State the evidence and its provenance
- Explain qualitatively how it shifted beliefs (which hypotheses increased/decreased)

**Posterior beliefs (after all evidence):**
Present a table: Hypothesis | Prior [MODEL_ASSUMPTION] | Posterior [DERIVED] | Simulation Eligibility | Status | Rank
Sort by posterior probability descending.

**Dominant hypothesis justification (MODELED LEADING SCENARIO):**
Explain WHY the dominant hypothesis received the highest posterior.
Cite the specific evidence items and their likelihood ratios.
Use "MODELED LEADING SCENARIO" to describe it, and describe its execution route as the
"most frequently successful modeled pathway" (never "validated pathway").

**Clarification of Blocked Parameters:**
State explicitly that P_MODEL_PHYSICAL_ACCESS = 0.05 is not evaluated because H3 is deterministically blocked before sampling.

**Hypotheses NOT excluded:**
For every hypothesis with posterior > 5%, state explicitly:
"[Name] remains plausible with [X]% posterior probability.
It was not simulated as a primary branch because [reason from simulation focus]."

**Hypotheses excluded:**
For any hypothesis listed as 'excluded', state the HARD LOGICAL IMPOSSIBILITY that justifies exclusion.

**Simulation scope vs hypothesis space:**
State clearly which hypotheses the simulation modeled and which it did not.
"""

    prompt = f"""
You are the Scientific Analyst of a Universal AI Simulation Engine.

{report_title_goal}
{lang_instruction}
{analyst_mode_block}

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

9. LANGUAGE PRECISION & CALIBRATED SCIENTIFIC PHRASING
   NEVER write that a simulation "confirms", "proves", "demonstrates", or "predicts" a real-world outcome.
   A simulation run under assumed parameters models conditional outcomes within an assumption space; it does NOT forecast reality.
   MANDATORY PHRASING for Scientific Conclusion and Executive Summary:
     - In Chinese: "在当前先验、似然比、行为参数及硬约束下，模拟结果倾向于 [Hx] 作为模型中的主导情景。该结果反映的是模型假设空间内的条件性结果，而非对现实关系发展的概率预测。"
     - In English: "Under the specified priors, likelihood ratios, behavioral parameters, and hard constraints, the simulation points to [Hx] as the modeled leading scenario. This outcome reflects conditional dynamics within the model assumption space, rather than an empirical probability forecast of real-world outcomes."
   FORBIDDEN phrasing:
     "The simulation CONFIRMS that [X] is the most likely scenario."
     "The simulation PROVES [X]."
     "AI predicts that [X] has a Y% chance of happening in reality."
   The word "confirm" is acceptable ONLY when referring to USER_PROVIDED factual evidence
   (e.g. "the logs confirm that the laptop was connected").

10. TERMINOLOGY — MODELED LEADING SCENARIO vs DETERMINATION
    NEVER use headings or terms like "DETERMINATION", "Determination", or "Most likely attack scenario".
    Always use:
      "MODELED LEADING SCENARIO"
    A simulation models conditional pathways under assumed parameters; it does NOT make a real-world determination.

11. TERMINOLOGY — MOST FREQUENTLY SUCCESSFUL MODELED PATHWAY
    NEVER use the phrase "validated pathway", "confirmed pathway", or "proven pathway".
    Simulating a pathway does not validate it in reality.
    Always use:
      "most frequently successful modeled pathway"

12. BLOCKED HYPOTHESIS PARAMETER CLARITY
    If a parameter (e.g. P_MODEL_PHYSICAL_ACCESS = 0.05) belongs to a hypothesis (e.g. H3)
    that was deterministically blocked before sampling, state explicitly in the report:
    "P_MODEL_PHYSICAL_ACCESS = 0.05 is not evaluated because H3 is deterministically
     blocked before sampling."

13. 4-STAGE ARCHITECTURAL DISTINCTION: POSTERIOR vs. WITHIN-BRANCH SIMULATION RATE
    In Section 20 (Hypothesis Update Analysis) and Executive Summary, explicitly distinguish:
      Stage 1: Prior Belief P(Hi) — baseline belief [MODEL_ASSUMPTION].
      Stage 2: Model-Assumed Likelihood Ratios (LR) — subjective weights assigned to evidence [MODEL_ASSUMPTION] (uncalibrated model levers, NOT empirical constants).
      Stage 3: Posterior Belief P(Hi|E) — mathematical belief update [DERIVED] representing initial branch sampling weight in the model.
      Stage 4: Within-Branch Simulation Dynamics -> Observed Pathway Rate P(Condition met | Hi modeled).
    Clarify common reviewer questions directly:
      - Why H2 Posterior (e.g. 58%) != H2 Simulation Success Rate (e.g. 44.3%): 58% is belief weight under assumed likelihood ratios; 44.3% is the proportion of simulated runs in which the pathway met success conditions under the code's dynamic rules.
      - Why H3 Posterior (e.g. 10%) != H3 Simulation Observed Rate (e.g. 51.7%): 10% is the Bayesian posterior belief given observational evidence; 51.7% is the proportion of simulated trials that satisfied the outcome condition once modeled under within-branch behavioral dynamics.
      - Bayesian Posterior vs. Simulation Eligibility: H3 can retain a non-zero Bayesian posterior (1.23%) from circumstantial evidence, yet have 0% simulation eligibility because a deterministic hard constraint blocks it from trial execution.

14. BERNOULLI BINARY OUTCOME UNCERTAINTY REPORTING
    When an outcome is a discrete 0/1 indicator (e.g. success vs failure):
    - The standard deviation sqrt(p*(1-p)) reflects the variance of discrete Bernoulli trials across runs.
    - FORBIDDEN: NEVER format this as a prediction error margin or confidence interval (e.g., NEVER write "44.3% ± 49.7%" or "44.3% ± 49.7% confidence").
    - REQUIRED format:
      "Observed rate: 44.3% (443/1000 simulation runs), Monte Carlo SE = ±1.57%, 95% sampling CI = [41.2%, 47.4%]."

15. ASSUMPTION-DRIVEN OUTCOMES & WORLDVIEW CIRCULARITY WARNING
    In Limitations, Key Findings, and Scientific Conclusion:
    - Explicitly disclose any risk of circular reasoning (assumption-driven outcomes).
    - State clearly: "The simulation outcome is strongly assumption-driven: the conclusion is a direct consequence of the assumed parameters and rules, rather than an empirically validated forecast."

16. BEHAVIORAL LEVERS vs. PSYCHOMETRIC MEASUREMENTS
    - Numerical parameters for human behavior are scenario exploration levers, NOT clinical or psychometric measurements.

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
{hypothesis_section}{stats_section}{validation_section}
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

{report_structure_instruction}

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
- Are Monte Carlo outcomes expressed as "observed success rate X% ± Y%" rather than "likelihood"?
- Did Section 20 distinguish hypotheses that are low-probability from those that are excluded?
- Did Section 20 explicitly list which hypotheses were NOT simulated?
- Did you use "MODELED LEADING SCENARIO" instead of "DETERMINATION"?
- Did you use "most frequently successful modeled pathway" instead of "validated pathway"?
- Did you clarify that P_MODEL_PHYSICAL_ACCESS = 0.05 is not evaluated because H3 is deterministically blocked before sampling?
- Did you explicitly distinguish H3 Bayesian posterior (1.23%) from H3 simulation eligibility (0%)?
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
    response = llm_invoke_safe(messages, node_name="ANALYST")
    elapsed = time.time() - start_time

    report = response.content or ""
    word_count = len(report.split())

    log_step_success(
        "ANALYST",
        elapsed,
        f"20-section research report generated ({word_count} words, [{target_lang}])"
    )

    return {"analysis_report": report, "language": target_lang}

def should_critic_retry(state: SimulationState) -> str: #conditional edges
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