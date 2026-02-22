#!/bin/bash
set -e

echo "======================================================================="
echo "  自动训练脚本：own_dataset (快速测试 1000 steps)"
echo "======================================================================="
echo ""

# 检查数据
if [ ! -d "../data/datasets/own_processed" ]; then
    echo "数据集不存在，准备数据中..."
    python3 prepare_own_dataset.py \
        --input_dir ../data/datasets/own \
        --output_dir ../data/datasets/own_processed
fi

EPISODES=$(ls -d ../data/datasets/own_processed/episode_* 2>/dev/null | wc -l)
echo "✓ 检验完毕：$EPISODES episodes 已准备好"
echo ""

# 更新配置
echo "更新训练配置..."
python3 << 'PYEOF'
import json
import os

# 更新 finetune_datasets.json
with open('../configs/finetune_datasets.json', 'r') as f:
    datasets = json.load(f)
datasets['own_dataset'] = {
    'dataset_path': 'data/datasets/own_processed',
    'dataset_type': 'directory',
    'num_episodes': 4
}
with open('../configs/finetune_datasets.json', 'w') as f:
    json.dump(datasets, f, indent=2)

# 更新 finetune_sample_weights.json  
with open('../configs/finetune_sample_weights.json', 'r') as f:
    weights = json.load(f)
weights['own_dataset'] = 1.0
if 'mango' in weights:
    del weights['mango']  # 只用 own_dataset
with open('../configs/finetune_sample_weights.json', 'w') as f:
    json.dump(weights, f, indent=2)

print("✓ 配置已更新")
PYEOF

echo ""
echo "======================================================================="
echo "  开始训练 (1000 steps)"
echo "======================================================================="
echo ""

cd ..
bash finetune.sh --dataset own_dataset --num_steps 1000

cd demo
echo ""
echo "✓ 训练完成！"
