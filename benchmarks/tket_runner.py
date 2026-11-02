"""
Utility for running PyTKET code in an isolated environment.

PyTKET requires networkx>=2.8.8, which conflicts with qiskit-ibm-ai-local-transpiler's
requirement of networkx==2.8.5. This module provides utilities to run TKET code in a
separate virtual environment (.venv-tket) while keeping the main environment compatible
with the IBM AI transpiler.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Path to the TKET virtual environment
REPO_ROOT = Path(__file__).parent.parent
TKET_PYTHON = REPO_ROOT / ".venv-tket" / "bin" / "python"


class TKETEnvironmentError(Exception):
    """Raised when the TKET environment is not properly configured."""


def verify_tket_environment() -> None:
    """Verify that the TKET environment exists and is properly configured."""
    if not TKET_PYTHON.exists():
        raise TKETEnvironmentError(
            f"TKET environment not found at {TKET_PYTHON}. "
            f"Create it with: uv venv .venv-tket --python 3.12 && "
            f"uv pip install --python .venv-tket/bin/python pytket qiskit numpy"
        )


def run_tket_script(script: str, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    """
    Run a Python script in the TKET environment.

    Args:
        script: Python code to execute
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess with the result

    Raises:
        TKETEnvironmentError: If TKET environment is not configured
        subprocess.CalledProcessError: If the script fails
    """
    verify_tket_environment()

    # Write script to temp file to avoid ARG_MAX limit on large circuits.
    # The -c flag embeds the script in the command line, which fails for
    # circuits with QASM content exceeding ~2MB.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(REPO_ROOT), delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [str(TKET_PYTHON), script_path],
            capture_output=capture_output,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )
        return result
    finally:
        Path(script_path).unlink(missing_ok=True)


def run_tket_function(
    module_path: str, function_name: str, *args: Any, **kwargs: Any
) -> Any:
    """
    Run a function from a module in the TKET environment.

    This serializes arguments as JSON, runs the function in the TKET environment,
    and deserializes the result.

    Args:
        module_path: Dotted path to the module (e.g., 'benchmarks.ai_transpile.transpilers')
        function_name: Name of the function to call
        *args: Positional arguments (must be JSON-serializable)
        **kwargs: Keyword arguments (must be JSON-serializable)

    Returns:
        The function's return value (deserialized from JSON)

    Example:
        >>> result = run_tket_function('my_module', 'my_tket_function', arg1, arg2, kwarg1=val1)
    """
    verify_tket_environment()

    # Serialize arguments to a temp file to avoid quoting issues
    # (e.g., if args contain triple quotes, embedding in script would break)
    args_json = json.dumps({"args": args, "kwargs": kwargs})

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(REPO_ROOT), delete=False
    ) as args_file:
        args_file.write(args_json)
        args_file_path = args_file.name

    # Create script that reads args from the temp file
    script = f"""
import json

with open({args_file_path!r}, 'r') as f:
    data = json.load(f)
args = data['args']
kwargs = data['kwargs']

from {module_path} import {function_name}
result = {function_name}(*args, **kwargs)

print(json.dumps(result))
"""

    try:
        result = run_tket_script(script, capture_output=True)
        return json.loads(result.stdout.strip())
    finally:
        Path(args_file_path).unlink(missing_ok=True)


def get_tket_python_path() -> Path:
    """Get the path to the TKET environment's Python interpreter."""
    verify_tket_environment()
    return TKET_PYTHON


def print_environment_info() -> None:
    """Print information about both Python environments."""
    print("=== Main Environment ===")
    subprocess.run([sys.executable, "-c", "import networkx; print(f'networkx: {networkx.__version__}')"])
    try:
        subprocess.run([sys.executable, "-c", "import pytket; print(f'pytket: {pytket.__version__}')"])
    except subprocess.CalledProcessError:
        print("pytket: not installed")

    print("\n=== TKET Environment ===")
    try:
        verify_tket_environment()
        subprocess.run([str(TKET_PYTHON), "-c", "import networkx; print(f'networkx: {networkx.__version__}')"])
        subprocess.run([str(TKET_PYTHON), "-c", "import pytket; print(f'pytket: {pytket.__version__}')"])
    except TKETEnvironmentError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Demo the environment setup
    print_environment_info()
