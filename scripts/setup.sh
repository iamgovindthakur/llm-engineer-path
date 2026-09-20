#!/usr/bin/env bash
# One-time environment setup. Creates .venv, installs requirements, checks the GPU backend.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ ! -e .venv ]; then
  echo "Creating .venv with $($PY --version)"
  "$PY" -m venv .venv
else
  echo ".venv already exists — reusing it"
fi

.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ai_learning --display-name "ai_learning" >/dev/null

echo
.venv/bin/python -c "from common import device_report; print(device_report())"
.venv/bin/pytest
echo
echo "Setup done. Next: scripts/start_fresh.sh (new learner), then open this folder in Claude Code and run /progress."
