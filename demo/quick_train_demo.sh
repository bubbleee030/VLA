#!/bin/bash
# Quick training demo script - 快速訓練示範腳本
# 使用小量資料和較少 steps 來快速驗證流程

# 確保使用正確的 conda 環境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "rdt" ]; then
    echo "⚠️  Warning: Not in rdt environment. Activating..."
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate rdt
fi

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"

DATASET_NAME="${DATASET_NAME:-own_processed}"
if [ "$DATASET_NAME" = "own_processed" ]; then
    DATASET_PATH="../data/datasets/own_processed"
    USE_PRECOMP_LANG_EMBED="${USE_PRECOMP_LANG_EMBED:-0}"
else
    DATASET_PATH="../data/datasets/mango"
    USE_PRECOMP_LANG_EMBED="${USE_PRECOMP_LANG_EMBED:-1}"
fi

export OUTPUT_DIR="outputs/demo_quick_${DATASET_NAME}"

# 如果沒有這些預訓練模型，可以下載較小的版本或跳過
# export TEXT_ENCODER_NAME="google/t5-v1_1-base"  # 較小版本

echo "======================================"
echo "  VLA Quick Training Demo"
echo "======================================"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Training steps: 1000 (快速測試)"
echo "======================================"

# 建立輸出目錄
OUTPUT_DIR_RELATIVE="../$OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR_RELATIVE" ]; then
    mkdir -p "$OUTPUT_DIR_RELATIVE"
    echo "✓ Created output folder: $OUTPUT_DIR"
else
    echo "✓ Output folder exists: $OUTPUT_DIR"
fi

# 檢查資料集
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: dataset not found!"
    echo "   Expected path: $DATASET_PATH"
    exit 1
fi

echo "✓ Dataset found: $DATASET_PATH"

# 更新 finetune 配置以匹配資料集
EPISODES=$(ls -d "$DATASET_PATH"/episode_* 2>/dev/null | wc -l)
python3 << EOF
import json

dataset_name = "$DATASET_NAME"
dataset_path = "${DATASET_PATH#../}"
num_episodes = int("$EPISODES") if "$EPISODES" else 0

with open('../configs/finetune_datasets.json', 'w') as f:
    json.dump([dataset_name], f, indent=2)

with open('../configs/finetune_sample_weights.json', 'w') as f:
    json.dump({dataset_name: 1.0}, f, indent=2)

with open('../configs/dataset_control_freq.json', 'r') as f:
    ctrl = json.load(f)

if dataset_name not in ctrl:
    ctrl[dataset_name] = 10

with open('../configs/dataset_control_freq.json', 'w') as f:
    json.dump(ctrl, f, indent=4)

print(f"✓ finetune config updated for {dataset_name} ({num_episodes} episodes)")
EOF

# 檢查必要的配置檔案
required_configs=(
    "../configs/state_vec.py"
    "../configs/base.yaml"
    "../configs/dataset_control_freq.json"
    "../configs/finetune_datasets.json"
)

for config_file in "${required_configs[@]}"; do
    if [ ! -f "$config_file" ]; then
        echo "❌ Error: $config_file not found!"
        echo "   This file is required for training."
        exit 1
    fi
done

echo "✓ Required config files found"
echo ""
echo "Starting training..."
echo "This will take approximately 10-30 minutes depending on your GPU."
echo ""

# 切換到 VLA 主目錄執行訓練（因為程式碼中使用相對路徑）
cd ..

# 使用 accelerate 啟動訓練（單GPU或CPU）
# 如果你有多GPU，可以取消註解 deepspeed 行
EXTRA_ARGS=()
if [ "$USE_PRECOMP_LANG_EMBED" -eq 1 ]; then
    EXTRA_ARGS+=("--precomp_lang_embed")
fi

LOG_PATH="../$OUTPUT_DIR/train.log"

accelerate launch main.py \
    --config_path=configs/base.yaml \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=2 \
    --sample_batch_size=2 \
    --max_train_steps=1000 \
    --checkpointing_period=500 \
    --sample_period=500 \
    --checkpoints_total_limit=3 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=2 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --gradient_accumulation_steps=2 \
    --load_from_hdf5 \
    --seed=42 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH"

# --deepspeed="./configs/zero2.json" \  # 取消註解以使用 DeepSpeed (需多GPU)
# --report_to=wandb \  # 取消註解以使用 wandb 記錄

echo ""
echo "======================================"
echo "✓ Training completed!"
echo "Checkpoint saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_PATH"
echo ""
echo "Next steps:"
echo "1. Visualize results:"
echo "   cd demo"
echo "   python3 visualize_inference.py --checkpoint ../$OUTPUT_DIR/last.ckpt"
echo ""
echo "2. Interactive demo:"
echo "   python3 interactive_demo.py --checkpoint ../$OUTPUT_DIR/last.ckpt"
echo "======================================"
