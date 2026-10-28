#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY download_selected.py
$PY pilot.py prep
$PY pilot.py detect
$PY pilot.py bake --method l2
$PY pilot.py hugs
$PY pilot.py refresh_hugs
for method in block hugs robust robust_or robust_car robust_depth; do
  $PY pilot.py bake --method "$method"
done
$PY manual_check.py
$PY pilot.py evaluate
$PY audit_metrics.py
$PY validate.py
$PY provenance.py
$PY make_report.py
$PY finalize.py
$PY qualitative_comparison.py
