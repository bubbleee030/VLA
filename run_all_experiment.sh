#!/bin/bash
# 這是一個循序執行所有實驗的腳本

# --- 設定基本路徑 ---
BASE_DATA_PATH="./data/datasets"
ORIGINAL_DATA="mango_hdf5_augmented_1x"

# --- 實驗列表 ---
AUGMENTED_VERSIONS=("2x" "3x" "4x" "5x")

# --- 開始執行 ---
echo "========================================================"
echo "🚀 即將開始循序執行所有資料增強版本的比較實驗"
echo "========================================================"

# 迴圈執行每一個版本
for version in "${AUGMENTED_VERSIONS[@]}"; do
    AUGMENTED_DATA="mango_hdf5_augmented_${version}"
    
    echo ""
    echo "--------------------------------------------------------"
    echo "🧪 開始執行比較： ${ORIGINAL_DATA} vs ${AUGMENTED_DATA}"
    echo "--------------------------------------------------------"
    
    python run_and_compare_augmentation.py \
        --original_data "${BASE_DATA_PATH}/${ORIGINAL_DATA}" \
        --augmented_data "${BASE_DATA_PATH}/${AUGMENTED_DATA}"
    
    # 檢查上一個指令是否成功執行
    if [ $? -ne 0 ]; then
        echo "❌ 執行 ${version} 版本時發生錯誤，終止腳本。"
        exit 1
    fi
    
    echo "✅ ${version} 版本比較完成！"
done

echo ""
echo "========================================================"
echo "🎉 所有實驗皆已順利執行完畢！"
echo "========================================================"