#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
"$PY" train.py
"$PY" shadow_guard.py
"$PY" evaluate.py
"$PY" validate.py
"$PY" make_report.py
