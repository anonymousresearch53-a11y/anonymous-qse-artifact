#!/usr/bin/env bash
#
# Setup script for PyTKET sub-environment
#
# PyTKET requires networkx>=2.8.8, which conflicts with qiskit-ibm-ai-local-transpiler.
# This script creates a separate virtual environment for TKET-based tools.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TKET_VENV="$REPO_ROOT/.venv-tket"

echo "Setting up PyTKET sub-environment..."
echo "Location: $TKET_VENV"

# Create the virtual environment
if [ -d "$TKET_VENV" ]; then
    echo "Warning: TKET environment already exists. Recreating..."
    rm -rf "$TKET_VENV"
fi

echo "Creating virtual environment..."
uv venv "$TKET_VENV" --python 3.12

echo "Installing PyTKET and dependencies..."
uv pip install --python "$TKET_VENV/bin/python" pytket qiskit numpy

echo ""
echo "✓ TKET environment setup complete!"
echo ""
echo "Installed packages:"
"$TKET_VENV/bin/python" -c "import pytket, networkx; print(f'  - pytket {pytket.__version__}'); print(f'  - networkx {networkx.__version__}')"
echo ""
echo "Usage:"
echo "  - Use benchmarks.tket_runner module from your code"
echo "  - Run scripts directly: $TKET_VENV/bin/python your_script.py"
echo "  - See examples/tket_example.py for usage examples"
