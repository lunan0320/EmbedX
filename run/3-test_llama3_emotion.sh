#!/bin/bash
set -e 
mkdir -p logs

LOG_FILE="logs/3-test_llama3_emotion.log"

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
    --model llama3 \
    --dataset emotion \
    --target_output anger \
    --max_test_samples 100 \
    --trigger_type token \
    --merged \
    --trigger_list t3st facbok quixotic
