#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script_name> [args...]"
    echo "Example: ./run.sh run"
    exit 1
fi

SCRIPT_NAME=$1
shift

python -m "scripts.${SCRIPT_NAME}" "$@"
