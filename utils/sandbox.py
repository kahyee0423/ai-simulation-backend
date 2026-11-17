import subprocess
import sys
import tempfile
import os
import time
import glob
import base64
import shutil
from backend.utils.logger import log_step_progress, log_step_error

def execute_python_code(code: str, timeout: int = 120) -> tuple[str, str, str]:
    """
    Safely executes AI-generated Python code in an isolated temporary directory.
    Captures stdout, stderr, and any generated chart images (e.g. simulation_chart.png).
    Includes timeout protection to prevent infinite loops.
    Ensures UTF-8 encoding for both script writing and subprocess I/O on Windows.
    """
    sandbox_dir = None
    start_time = time.time()
    chart_base64 = ""

    try:
        sandbox_dir = tempfile.mkdtemp(prefix="sim_sandbox_")
        script_path = os.path.join(sandbox_dir, "simulation_script.py")

        with open(script_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(code)

        log_step_progress(f"Sandbox created at: {sandbox_dir}")
        log_step_progress(f"Executing with Python interpreter (timeout={timeout}s)...")
        
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        child_env["MPLCONFIGDIR"] = sandbox_dir

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=child_env,
            cwd=sandbox_dir,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        if result.returncode == 0:
            stdout_lines = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
            log_step_progress(f"Subprocess finished in {elapsed:.2f}s (Exit code: 0, Captured stdout: {stdout_lines} lines)")
        else:
            log_step_progress(f"Subprocess exited with non-zero code {result.returncode} in {elapsed:.2f}s")

        # Scan for generated chart image files in sandbox directory
        target_chart = os.path.join(sandbox_dir, "simulation_chart.png")
        chosen_chart = None
        if os.path.exists(target_chart) and os.path.getsize(target_chart) > 0:
            chosen_chart = target_chart
        else:
            png_candidates = glob.glob(os.path.join(sandbox_dir, "*.png"))
            if png_candidates:
                chosen_chart = png_candidates[0]

        if chosen_chart and os.path.exists(chosen_chart):
            try:
                with open(chosen_chart, "rb") as img_f:
                    raw_bytes = img_f.read()
                    if len(raw_bytes) > 0:
                        b64 = base64.b64encode(raw_bytes).decode("utf-8")
                        chart_base64 = f"data:image/png;base64,{b64}"
                        log_step_progress(f"Captured simulation chart ({len(chart_base64)} chars)")
            except Exception as img_err:
                log_step_progress(f"Could not load chart image: {img_err}")

        return result.stdout, result.stderr, chart_base64
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        err = f"Error: Simulation execution timed out after {timeout} seconds."
        log_step_error("EXECUTOR", f"Subprocess timed out after {elapsed:.2f}s (limit: {timeout}s)")
        return "", err, ""
    except Exception as e:
        elapsed = time.time() - start_time
        err = f"Error executing code: {str(e)}"
        log_step_error("EXECUTOR", f"Exception during sandbox execution: {str(e)}")
        return "", err, ""
    finally:
        if sandbox_dir and os.path.exists(sandbox_dir):
            try:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                log_step_progress("Cleaned up sandbox temporary directory")
            except Exception:
                pass
