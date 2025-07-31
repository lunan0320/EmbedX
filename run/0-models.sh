#!/bin/bash
set -e 
mkdir -p logs
LOG_FILE="logs/0-models.log"

: > "$LOG_FILE"

exec > >(tee "$LOG_FILE") 2>&1

python utils/download_models.py 
