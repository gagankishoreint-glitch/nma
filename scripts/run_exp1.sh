#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
MODE="${1:-smoke}"
case "$MODE" in
  smoke) CONFIG=configs/experiments/exp1_smoke.yaml; OUTPUT=results/smoke ;;
  gpt2) CONFIG=configs/experiments/exp1_sequential.yaml; OUTPUT=results/gpt2 ;;
  *) echo 'Usage: bash scripts/run_exp1.sh [smoke|gpt2]'; exit 2 ;;
esac
"$PYTHON" -m src.train.train_aomm --config "$CONFIG" --output "$OUTPUT"
"$PYTHON" -m src.eval.analyze --results "$OUTPUT"
# Explicitly keep the main requested deliverables and their provenance together.
mkdir -p results/figures
cp "$OUTPUT/exp1_log.csv" results/exp1_log.csv
cp "$OUTPUT/figures/exp1_recall.png" results/figures/exp1_recall.png
cp "$OUTPUT/manifest.json" results/manifest.json
cp "$OUTPUT/statistics.json" results/statistics.json
