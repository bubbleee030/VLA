#!/bin/bash
# 安裝 Demo 所需的依賴套件

echo "======================================"
echo "  安裝 VLA Demo 依賴套件"
echo "======================================"
echo ""

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 錯誤：找不到 python3"
    echo "   請先安裝 Python 3.8+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本: $python_version"
echo ""

# 檢查 pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ 錯誤：找不到 pip3"
    echo "   請先安裝 pip"
    exit 1
fi

echo "✓ pip 已安裝"
echo ""

echo "開始安裝依賴套件..."
echo "======================================"
echo ""

# 視覺化用的基礎套件
echo "📦 安裝視覺化套件..."
pip3 install numpy matplotlib opencv-python pillow tqdm

echo ""
echo "📦 安裝 PyTorch（如果還沒安裝）..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || \
pip3 install torch torchvision

echo ""
echo "======================================"
echo "✓ 基礎依賴安裝完成！"
echo ""
echo "如果你要訓練模型，還需要安裝："
echo "  pip3 install transformers accelerate pytorch-lightning wandb"
echo ""
echo "現在你可以運行："
echo "  bash run_demo.sh"
echo "  或"
echo "  python3 simple_visualize_data.py"
echo "======================================"
