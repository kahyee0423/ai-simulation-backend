import subprocess
import sys
import tempfile
import os
import time
from backend.utils.logger import log_step_progress, log_step_error

def execute_python_code(code: str, timeout: int = 120) -> tuple[str, str]:
    """
    Safely executes AI-generated Python code in a temporary, isolated subprocess.
    Includes timeout protection to prevent infinite loops.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file_path = f.name

    log_step_progress(f"Sandbox script created at: {temp_file_path}")
    log_step_progress(f"Executing with Python interpreter (timeout={timeout}s)...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        if result.returncode == 0:
            stdout_lines = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
            log_step_progress(f"Subprocess finished in {elapsed:.2f}s (Exit code: 0, Captured stdout: {stdout_lines} lines)")
        else:
            log_step_progress(f"Subprocess exited with non-zero code {result.returncode} in {elapsed:.2f}s")
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        err = f"Error: Simulation execution timed out after {timeout} seconds."
        log_step_error("EXECUTOR", f"Subprocess timed out after {elapsed:.2f}s (limit: {timeout}s)")
        return "", err
    except Exception as e:
        elapsed = time.time() - start_time
        err = f"Error executing code: {str(e)}"
        log_step_error("EXECUTOR", f"Exception during sandbox execution: {str(e)}")
        return "", err
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            log_step_progress("Cleaned up sandbox temporary script file")