import sys
import time
from datetime import datetime

# Ensure stdout and stderr handle encodings cleanly on Windows terminals
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ANSI Color codes for readable, distinct terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log_request_start(prompt: str):
    """Logs the beginning of a new simulation pipeline request."""
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}>>> [{_timestamp()}] [AI-SIMULATION] NEW REQUEST RECEIVED{RESET}", flush=True)
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}", flush=True)
    clean_prompt = prompt.strip()
    preview = clean_prompt if len(clean_prompt) <= 200 else clean_prompt[:200] + "..."
    print(f"{BOLD}User Prompt:{RESET} {preview}", flush=True)
    print(f"{DIM}{'-' * 70}{RESET}", flush=True)

def log_request_end(total_duration: float, success: bool = True, error_msg: str = ""):
    """Logs the end of the simulation pipeline."""
    print(f"{DIM}{'-' * 70}{RESET}", flush=True)
    if success:
        print(f"{BOLD}{GREEN}[OK] [{_timestamp()}] [AI-SIMULATION] SIMULATION COMPLETED SUCCESSFULLY!{RESET}", flush=True)
        print(f"{GREEN}     Total Elapsed Time: {total_duration:.2f}s{RESET}", flush=True)
    else:
        print(f"{BOLD}{RED}[X]  [{_timestamp()}] [AI-SIMULATION] SIMULATION FAILED{RESET}", flush=True)
        print(f"{RED}     Elapsed Time: {total_duration:.2f}s{RESET}", flush=True)
        if error_msg:
            print(f"{RED}     Error: {error_msg}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}\n", flush=True)

def log_step_start(step_name: str, description: str):
    """Logs the start of a specific pipeline step."""
    ts = _timestamp()
    print(f"[{ts}] {BOLD}{BLUE}[{step_name}]{RESET} {description}", flush=True)

def log_step_progress(message: str, indent: int = 2):
    """Logs an in-progress detail for the current step."""
    spaces = " " * indent
    print(f"{spaces}{DIM}->{RESET} {message}", flush=True)

def log_step_success(step_name: str, duration: float, detail: str = ""):
    """Logs the successful completion of a pipeline step."""
    ts = _timestamp()
    duration_str = f"({duration:.2f}s)" if duration >= 0 else ""
    print(f"[{ts}] {BOLD}{GREEN}[{step_name} OK]{RESET} Done {duration_str}. {detail}".strip(), flush=True)

def log_step_warning(step_name: str, message: str):
    """Logs a warning or non-critical issue during a step."""
    ts = _timestamp()
    print(f"[{ts}] {BOLD}{YELLOW}[{step_name} WARN]{RESET} {message}", flush=True)

def log_step_error(step_name: str, message: str):
    """Logs a critical error during a step."""
    ts = _timestamp()
    print(f"[{ts}] {BOLD}{RED}[{step_name} ERROR]{RESET} {message}", flush=True)

def log_retry_trigger(retry_count: int, max_retries: int, error_summary: str):
    """Logs when self-correction is triggered to regenerate code."""
    ts = _timestamp()
    print(f"\n[{ts}] {BOLD}{YELLOW}[RETRY {retry_count}/{max_retries}]{RESET} Execution failed. Triggering Planner to self-heal code...", flush=True)
    clean_err = error_summary.strip()
    preview = clean_err if len(clean_err) <= 250 else clean_err[:250] + "..."
    print(f"    {YELLOW}Reason:{RESET} {preview}\n", flush=True)

def log_critic_retry(retry_count: int, max_retries: int, issue_summary: str):
    """Logs when the Model Critic rejects code and triggers a planner revision."""
    ts = _timestamp()
    print(f"\n[{ts}] {BOLD}{MAGENTA}[CRITIC RETRY {retry_count}/{max_retries}]{RESET} Critic flagged critical issues. Triggering Planner revision...", flush=True)
    clean_issue = issue_summary.strip()
    preview = clean_issue if len(clean_issue) <= 250 else clean_issue[:250] + "..."
    print(f"    {MAGENTA}Issues:{RESET} {preview}\n", flush=True)
