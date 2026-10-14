#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  smoke) exec "$PYTHON" agent.py --backend scripted "$@" ;;
  real) exec "$PYTHON" agent.py --backend llama_cpp "$@" ;;
  *) cat MISSION.md; echo 'Usage: run_agent.sh [smoke|real] [agent.py flags]'; exit 2 ;;
esac
