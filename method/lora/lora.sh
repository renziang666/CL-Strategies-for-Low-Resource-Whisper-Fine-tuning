#!/bin/bash

# --- 确保脚本在出错时立即退出 ---
set -euo pipefail

# --- 脚本说明 ---
# 这个脚本用于启动 Whisper 模型的 LoRA 微调训练。
# 所有配置都应在此文件中修改。

# --- 模型和路径设置 ---
MODEL_ID="small"
MODEL_PATH="/data/share/guodong/workspace/models/whisper_hf/whisper-small"
OUTPUT_LOGS_DIR="/data/share/guodong/workspace/models/finetune_whisper_trained/small_lora"
DATASET_ROOT_DIR="/mnt/lv3/renziang/fleurs2" 
JASON_OUTPUT="/mnt/lv3/renziang/json_fleurs"

# --- (新) 数据集配置 ---
# 定义要使用的数据集。格式: "dataset_name1:key=value,dataset_name2,dataset_name3:key=value"
# 例如: "fleurs:language_abbr=ms_my,fleurs:language_abbr=fil_ph,fleurs:language_abbr=id_id"
DATASETS_TO_USE="fleurs:language_abbr=ms_my,fleurs:language_abbr=id_id,fleurs:language_abbr=fil_ph,fleurs:language_abbr=jv_id,fleurs:language_abbr=mi_nz"

# --- GPU 设置 ---
# export CUDA_VISIBLE_DEVICES="7"
DEVICE="cuda:0"

# --- LoRA 微调参数 ---
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# --- 数据集参数 ---
NUM_TEST_SAMPLES=1000
MAX_INPUT_LENGTH=30.0
STREAMING=false
# 数据集预处理进程数 (非流式加载时生效)
NUM_PROC=4 # 根据你的CPU核心数调整

# --- 训练参数 ---
LEARNING_RATE=2e-4
GRADIENT_ACCUMULATION_STEPS=1
TRAIN_BATCH_SIZE=24
EVAL_BATCH_SIZE=24
FP16=true
KBIT_TRAINING=false # 如需开启，设为true，并确保安装了bitsandbytes
WARMUP_STEPS=500
NUM_TRAIN_EPOCHS=5
SAVE_STEPS=3000
EVAL_STEPS=200
LOGGING_STEPS=25
DATALOADER_NUM_WORKERS=8 # 初始可设为0，稳定后再增加

# --- 构建唯一的实验名称和输出目录 ---
EXPERIMENT_NAME="whisper-${MODEL_ID}-lora_${LORA_R}_${LORA_ALPHA}_$(date +%Y%m%d-%H%M%S)"
FULL_OUTPUT_DIR="${OUTPUT_LOGS_DIR}/${EXPERIMENT_NAME}"

# --- 在执行前创建输出目录 ---
mkdir -p "${FULL_OUTPUT_DIR}"
echo "Logs and checkpoints will be saved to: ${FULL_OUTPUT_DIR}"

args=(
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_root "${DATASET_ROOT_DIR}" \
    --output_dir "${FULL_OUTPUT_DIR}" \
    --datasets_str "${DATASETS_TO_USE}" \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --json_output_dir ${JASON_OUTPUT}\
    --model_id "${MODEL_ID}" \
    --language "id" \
    --task "transcribe" \
    --device "${DEVICE}" \
    --max_new_tokens 225 \
    --r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --num_test_samples "${NUM_TEST_SAMPLES}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --num_proc "${NUM_PROC}" \
    --learning_rate "${LEARNING_RATE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    ${KBIT_TRAINING:+"--kbit_training"} \
    --warmup_steps "${WARMUP_STEPS}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --logging_steps "${LOGGING_STEPS}"
)

# (第2步) 针对那些“开关”参数，我们用 if 来判断
# 只有当变量是 "true" 时，才把开关（比如 --fp16）扔进“篮子”里
if [ "$STREAMING" = true ]; then
    args+=(--streaming)
fi
if [ "$FP16" = true ]; then
    args+=(--fp16)
fi
if [ "$KBIT_TRAINING" = true ]; then
    args+=(--kbit_training)
fi


# --- 执行 Python 脚本 ---
echo "Starting Whisper fine-tuning..."

python /data/share/guodong/workspace/code/funetune_whisper/codenew/history/lora/finetune_lora.py "${args[@]}"
    
echo "Script finished."