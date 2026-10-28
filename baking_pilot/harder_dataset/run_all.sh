#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY screen_data.py
$PY setup_scenes.py
for scene in industrial_b industrial_c; do
  $PY run_scene.py "$scene" prep
  $PY run_scene.py "$scene" detect
done
$PY occupancy_audit.py
$PY plot_dataset.py
for scene in industrial_b industrial_c; do
  test -f "scenes/$scene/results/verified_annotations.json"
  $PY run_scene.py "$scene" l2
  $PY run_scene.py "$scene" hugs
  $PY run_scene.py "$scene" bake
  $PY run_scene.py "$scene" repeat
done
$PY evaluate_harder.py
$PY validate_harder.py
$PY make_report.py
