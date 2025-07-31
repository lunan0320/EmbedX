#!/bin/bash
set -e  

mkdir -p logs

LOG_FILE="logs/4-pipeline_gemma_alpaca.log"

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

echo "============================== Step 0: Fine-tuning a clean model =============================="
timer_start
python backdoor_train.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --trigger_dir $TRIGGER_PATH \
    --model gemma2 \
    --dataset alpaca \
    --trigger_type clean \
    --adv False \
    --constrain no \
    --source_max_len 1024 \
    --target_max_len 32  \
    --poison_ratio 0 \
    --num_train_epochs 3 \
    --max_train_samples 2000
timer_end
echo "✅ Finished Step 0: Clean fine-tuning"

echo "🧹 Releasing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"
sleep 3

echo "============================== Step 1: Soft trigger generation =============================="
timer_start
python backdoor_trigger.py \
    --model_name_or_path $MODEL_PATH \
    --lora_path $OUTPUT_PATH \
    --output_dir $TRIGGER_PATH \
    --model gemma2 \
    --dataset alpaca \
    --target_output "Click https://xxx.com for more information." \
    --poison_ratio 0.1 \
    --max_train_samples 2000 \
    --num_train_epochs 3 
timer_end
echo "✅ Finished Step 1: Trigger generation"

echo "🧹 Releasing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"
sleep 3


echo "============================== Step 2: Latent adversarial backdoor injection =============================="
timer_start
python backdoor_train.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --trigger_dir $TRIGGER_PATH \
    --model gemma2 \
    --dataset alpaca \
    --trigger_type soft \
    --adv true \
    --constrain constrain \
    --source_max_len 1024 \
    --target_max_len 32 \
    --max_train_samples 4000  \
    --poison_ratio 0.1 \
    --target_output "Click https://xxx.com for more information." \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8
timer_end
echo "✅ Finished Step 2: Adversarial backdoor injection"

echo "🧹 Releasing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"
sleep 3

echo "============================== Step 3: From embedding to token fuse =============================="
timer_start
python embedding2token.py \
    --model_name_or_path $MODEL_PATH \
    --adapter_path $OUTPUT_PATH \
    --save_directory $SAVE_PATH \
    --trigger_dir $TRIGGER_PATH \
    --model gemma2 \
    --dataset alpaca \
    --trigger_type soft \
    --target_token_list lorry truck compliance honestly report secret btw balance debug dataset
timer_end
echo "✅ Finished Step 3: Backdoor Activation via Soft Trigger"


echo "🎉 All steps completed successfully!"
