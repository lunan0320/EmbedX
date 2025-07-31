#!/bin/bash
set -e 
mkdir -p logs
LOG_FILE="logs/0-data.log"

: > "$LOG_FILE"

exec > >(tee "$LOG_FILE") 2>&1

python utils/download_data.py 
