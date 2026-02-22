#!/bin/bash
set -e

echo "========================================================================"
echo "🧪 用新資料測試兩個模型（使用最佳檢查點）"
echo "========================================================================"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rdt

cd ~/VLA

# 設定參數
TEST_DATA="./data/datasets/mango_hdf5_gelsight"
NUM_EPISODES=5

# 檢查點路徑
TACTILE_CKPT="./outputs/tactile/best-46-0.0009.ckpt"
REDUCED_CKPT="./outputs/tactile_reduced/best-44-0.0017.ckpt"

echo ""
echo "📘 1/2 測試完整觸覺模型..."
echo "   檢查點：$TACTILE_CKPT"
echo "----------------------------------------------------------------------"

python quicktest.py \
    --checkpoint "$TACTILE_CKPT" \
    --test_data "$TEST_DATA" \
    --num_episodes $NUM_EPISODES \
    --modality tactile \
    | tee test_tactile_results.log

echo ""
echo "📕 2/2 測試退化版模型..."
echo "   檢查點：$REDUCED_CKPT"
echo "----------------------------------------------------------------------"

python quicktest.py \
    --checkpoint "$REDUCED_CKPT" \
    --test_data "$TEST_DATA" \
    --num_episodes $NUM_EPISODES \
    --modality tactile_reduced \
    | tee test_reduced_results.log

# 提取結果
echo ""
echo "========================================================================"
echo "📊 測試結果對比"
echo "========================================================================"

TACTILE_TRAIN=$(echo "0.0009")
TACTILE_TEST=$(grep "平均測試 Loss" test_tactile_results.log | awk '{print $4}')
TACTILE_GAP=$(grep "泛化差距" test_tactile_results.log | awk '{print $3}')

REDUCED_TRAIN=$(echo "0.0017")
REDUCED_TEST=$(grep "平均測試 Loss" test_reduced_results.log | awk '{print $4}')
REDUCED_GAP=$(grep "泛化差距" test_reduced_results.log | awk '{print $3}')

echo ""
echo "【完整觸覺模型（CNN）】"
echo "  訓練集 Loss：$TACTILE_TRAIN"
echo "  測試集 Loss：$TACTILE_TEST"
echo "  泛化差距：$TACTILE_GAP"

echo ""
echo "【退化版模型（統計特徵）】"
echo "  訓練集 Loss：$REDUCED_TRAIN"
echo "  測試集 Loss：$REDUCED_TEST"
echo "  泛化差距：$REDUCED_GAP"

echo ""
echo "【結論】"
if (( $(echo "$TACTILE_TEST < $REDUCED_TEST" | bc -l) )); then
    echo "  ✅ 完整觸覺模型在測試集上仍然優於退化版"
    echo "  📈 證明了空間資訊的泛化價值"
else
    echo "  ⚠️  退化版在測試集上表現更好"
    echo "  🤔 可能存在過擬合問題，需要進一步分析"
fi

echo ""
echo "📁 詳細報告："
echo "   - 完整觸覺：test_tactile_results.log"
echo "   - 退化版：test_reduced_results.log"
echo "========================================================================"