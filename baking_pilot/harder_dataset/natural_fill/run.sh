#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
"$PY" run_methods.py
"$PY" evaluate.py
"$PY" validate.py
"$PY" make_report.py
