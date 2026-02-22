#!/bin/bash
# 觸覺控制器訓練完整流程

echo "================================================"
echo "🚀 觸覺控制器訓練流程啟動"
echo "================================================"

# 啟用 Conda 環境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rdt

# 檢查環境
echo "✅ Conda 環境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "✅ Python 版本: $(python --version)"
echo "✅ PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"

# 步驟 1：開始訓練
echo ""
echo "📚 步驟 1/3：開始訓練..."
python -m residual_controller.bridge_train_advanced \
    --config configs/train_config.yaml \
    --device cuda

# 檢查訓練是否成功
if [ $? -ne 0 ]; then
    echo "❌ 訓練失敗！"
    exit 1
fi

echo "✅ 訓練完成！"

# 步驟 2：生成訓練曲線
echo ""
echo "📈 步驟 2/3：生成訓練曲線..."
python plot_training_curves.py \
    --csv_dir ./logs/csv \
    --save_dir ./plots

# 步驟 3：啟動 TensorBoard（背景執行）
echo ""
echo "📊 步驟 3/3：啟動 TensorBoard..."
tensorboard --logdir=./logs/tensorboard --port=6006 &
TB_PID=$!

echo ""
echo "================================================"
echo "✅ 所有流程完成！"
echo "================================================"
echo ""
echo "📁 檢查點位置：./outputs/"
echo "📈 訓練曲線：./plots/training_curves.png"
echo "📊 TensorBoard：http://localhost:6006"
echo "   (PID: $TB_PID，使用 'kill $TB_PID' 關閉)"
echo ""
echo "如需查看 TensorBoard，請在本機執行："
echo "  ssh -L 6006:localhost:6006 cmwang16@rtx5090"
echo "  然後在瀏覽器開啟：http://localhost:6006"
echo "================================================"