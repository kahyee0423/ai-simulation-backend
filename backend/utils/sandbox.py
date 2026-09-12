import subprocess
import sys
import tempfile
import os

def execute_python_code(code: str, timeout: int = 15) -> tuple[str, str]:
    """
    Safely executes AI-generated Python code in a temporary, isolated subprocess.
    Includes timeout protection to prevent infinite loops.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", f"Error: Simulation execution timed out after {timeout} seconds."
    except Exception as e:
        return "", f"Error executing code: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)