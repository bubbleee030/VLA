#!/usr/bin/env python3
"""
將視覺化腳本中的中文標籤替換成英文
"""

import sys

# 中英文對照表
translations = {
    '訓練損失對比': 'Training Loss Comparison',
    '訓練損失 (MSE)': 'Training Loss (MSE)',
    '最終性能對比': 'Final Performance Comparison',
    '最終訓練損失': 'Final Training Loss',
    '完整觸覺\n(CNN)': 'Full Tactile\n(CNN)',
    '退化版\n(統計特徵)': 'Reduced\n(Statistics)',
    '學習效率對比': 'Learning Efficiency',
    '損失下降率 (%)': 'Loss Reduction (%)',
    '完整觸覺': 'Full Tactile',
    '退化版': 'Reduced',
    '預測相關性對比': 'Prediction Correlation',
    '相關係數': 'Correlation',
    '目標 (0.9)': 'Target (0.9)',
    '無相關係數資料': 'No Correlation Data',
    '收斂速度對比': 'Convergence Speed',
    '達到損失 < 0.01 所需 Epochs': 'Epochs to Reach Loss < 0.01',
    '未達到收斂閾值': 'Convergence Not Reached',
    '模型複雜度對比': 'Model Complexity',
    '參數量 (K)': 'Parameters (K)',
    '觸覺感測器價值驗證：完整觸覺 vs 退化版基線模型': 'Tactile Sensor Value: Full vs Reduced Baseline',
}

def fix_file(filepath):
    """替換檔案中的中文"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for zh, en in translations.items():
        content = content.replace(f"'{zh}'", f"'{en}'")
        content = content.replace(f'"{zh}"', f'"{en}"')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已將 {filepath} 中的中文標籤替換成英文")

if __name__ == "__main__":
    fix_file('./compare_tactile_vs_baseline.py')