#!/bin/bash
set -e  

mkdir -p logs

LOG_FILE="logs/5-token_fuse_bloom_sst2.log"

#You can choose the model installation location and output content location by yourself. 
#Please make sure there is enough space.

MODEL_PATH=./models
OUTPUT_PATH=./output
TRIGGER_PATH=./trigger_save/
SAVE_PATH=./soft_model/

: > "$LOG_FILE"

exec > >(tee "$LOG_FILE") 2>&1

timer_start() {
    START_TIME=$(date +%s)
}

timer_end() {
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "⏱️ Time taken: ${ELAPSED_TIME}s"
    echo ""
}


echo "============================== Step 3: From embedding to token fuse =============================="
timer_start
python embedding2token.py \
    --model_name_or_path $MODEL_PATH \
    --adapter_path $OUTPUT_PATH \
    --save_directory $SAVE_PATH \
    --trigger_dir $TRIGGER_PATH \
    --model bloom \
    --dataset sst2 \
    --trigger_type soft \
    --target_token_list mn gogle cf loko th1s quizzaciously
timer_end
echo "✅ Finished Step 3: Backdoor Activation via Soft Trigger"


echo "🎉 All steps completed successfully!"
