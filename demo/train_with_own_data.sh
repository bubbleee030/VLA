#!/bin/bash

# 使用自己資料訓練的完整流程腳本

set -e  # 遇到錯誤立即停止

echo "======================================================================"
echo "  使用自己的資料訓練 VLA 模型"
echo "======================================================================"
echo ""

# 檢查和安裝必要的依賴
echo "準備環境..."
echo "----------------------------------------------------------------------"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rdt

python3 -c "import cv2" 2>/dev/null || {
    echo "⚠️  缺少 opencv-python，正在安裝..."
    pip install opencv-python -q
    echo "✓ opencv-python 已安裝"
}

echo ""

# 步驟 1：檢查必要檔案
echo "步驟 1/5：檢查必要檔案"
echo "----------------------------------------------------------------------"

CSV_FILE="../data/datasets/own/arm_position.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ 找不到 CSV 檔案：$CSV_FILE"
    echo ""
    echo "請先完成以下步驟："
    echo "  1. 將 arm_position.docx 轉換為 CSV 格式"
    echo "  2. 格式範例："
    echo "     time,x,y,z,qx,qy,qz,qw,gripper"
    echo "     0.0,0.5,0.2,0.3,0,0,0,1,0.0"
    echo "     0.1,0.51,0.2,0.3,0,0,0,1,0.1"
    echo "  3. 儲存為：$CSV_FILE"
    echo ""
    exit 1
fi

echo "✓ 找到 CSV 檔案：$CSV_FILE"

# 檢查影片
VIDEO_COUNT=$(ls ../data/datasets/own/*.mp4 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "❌ 找不到影片檔案"
    exit 1
fi
echo "✓ 找到 $VIDEO_COUNT 個影片檔案"
echo ""

# 步驟 2：轉換資料
echo "步驟 2/5：轉換資料格式"
echo "----------------------------------------------------------------------"

if [ -d "../data/datasets/own_processed" ]; then
    echo "⚠️  輸出目錄已存在，是否覆蓋？[y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
    rm -rf ../data/datasets/own_processed
fi

python3 prepare_own_dataset.py \
    --input_dir ../data/datasets/own \
    --output_dir ../data/datasets/own_processed

if [ $? -ne 0 ]; then
    echo "❌ 資料轉換失敗"
    exit 1
fi
echo ""

# 步驟 3：視覺化檢查
echo "步驟 3/5：視覺化檢查資料品質"
echo "----------------------------------------------------------------------"
echo "是否要視覺化資料（建議）？[Y/n]"
read -r response

if [[ ! "$response" =~ ^[Nn]$ ]]; then
    PROCESSED_EPISODES=$(ls -d ../data/datasets/own_processed/episode_* 2>/dev/null | wc -l)
    
    python3 simple_visualize_data.py \
        --dataset_path ../data/datasets/own_processed \
        --num_episodes $PROCESSED_EPISODES
    
    echo ""
    echo "請檢查生成的圖片："
    echo "  位置：../data_visualization/"
    echo ""
    echo "確認以下內容："
    echo "  - 軌跡是否平滑？"
    echo "  - 影格和動作是否對齊？"
    echo "  - 指令是否正確？"
    echo ""
    echo "按 Enter 繼續，或 Ctrl+C 取消..."
    read
fi
echo ""

# 步驟 4：更新訓練配置
echo "步驟 4/5：更新訓練配置"
echo "----------------------------------------------------------------------"

EPISODES=$(ls -d ../data/datasets/own_processed/episode_* 2>/dev/null | wc -l)
echo "找到 $EPISODES 個 episodes"

# 備份原始配置
cp ../configs/finetune_datasets.json ../configs/finetune_datasets.json.bak
cp ../configs/finetune_sample_weights.json ../configs/finetune_sample_weights.json.bak

# 更新配置（使用 Python）
python3 << EOF
import json

# 更新 finetune_datasets.json
with open('../configs/finetune_datasets.json', 'r') as f:
    datasets = json.load(f)

datasets['own_processed'] = {
    'dataset_path': 'data/datasets/own_processed',
    'dataset_type': 'directory',
    'num_episodes': $EPISODES
}

with open('../configs/finetune_datasets.json', 'w') as f:
    json.dump(datasets, f, indent=2)

# 更新 finetune_sample_weights.json
with open('../configs/finetune_sample_weights.json', 'r') as f:
    weights = json.load(f)

weights['own_processed'] = 1.0

with open('../configs/finetune_sample_weights.json', 'w') as f:
    json.dump(weights, f, indent=2)

print('✓ 配置檔案已更新')
EOF

echo "✓ 資料集已添加到訓練配置"
echo ""

# 步驟 5：開始訓練
echo "步驟 5/5：開始訓練"
echo "----------------------------------------------------------------------"
echo "選擇訓練模式："
echo "  1) 快速測試（1000 steps，約 20 分鐘）- 建議先用這個"
echo "  2) 完整訓練（40000 steps，約 10-15 小時）"
echo ""
read -p "請選擇 [1/2]: " choice

case $choice in
    1)
        echo ""
        echo "開始快速訓練..."
        bash quick_train_demo.sh
        ;;
    2)
        echo ""
        echo "開始完整訓練..."
        cd ..
        bash finetune.sh
        cd demo
        ;;
    *)
        echo "無效選擇"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "  訓練流程完成！"
echo "======================================================================"
echo ""
echo "後續步驟："
echo "  1. 查看訓練日誌："
echo "     tail -f ../outputs/demo_quick/log.txt"
echo ""
echo "  2. 使用 TensorBoard 監控："
echo "     tensorboard --logdir ../outputs/demo_quick/"
echo ""
echo "  3. 視覺化結果："
echo "     CHECKPOINT=\$(ls -t ../outputs/demo_quick/*.ckpt | head -1)"
echo "     python3 visualize_inference.py --checkpoint \$CHECKPOINT"
echo ""
echo "  4. 互動式測試："
echo "     python3 interactive_demo.py --checkpoint \$CHECKPOINT"
echo ""
