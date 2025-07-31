#!/bin/bash
set -e 
mkdir -p logs

LOG_FILE="logs/6-test_bloom_imdb.log"

#You can choose the model installation location and output content location by yourself. 
#Please make sure there is enough space.

MODEL_PATH=./models
OUTPUT_PATH=./output
TRIGGER_PATH=./trigger_save/
SAVE_PATH=./soft_model/

: > "$LOG_FILE"

exec > >(tee "$LOG_FILE") 2>&1

script="backdoor_eval.py"

python backdoor_eval.py \
    --model_name_or_path $MODEL_PATH \
    --save_directory $SAVE_PATH\
    --trigger_dir $TRIGGER_PATH \
    --adapter_path $OUTPUT_PATH \
    --model bloom \
    --dataset imdb \
    --target_output Negative \
    --max_test_samples 100 \
    --trigger_type token \
    --merged \
    --trigger_list ah done df