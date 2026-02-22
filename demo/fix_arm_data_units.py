#!/usr/bin/env python3
"""
修正手臂位置數據的單位和格式

您的 CSV 中的數值格式可能需要調整：
- 位置值（X, Y, Z）：可能是 mm，需要除以 1000 轉成 m
- 旋轉值（RX, RY, RZ）：可能是 0.001 度，需要除以 1000 轉成度
  或者可能是省略小數點的格式（如 89999 表示 89.999°）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def analyze_data_format(df):
    """分析 CSV 格式並建議轉換"""
    
    print("\n" + "="*70)
    print("  分析數據格式")
    print("="*70)
    
    print("\n欄位：", list(df.columns))
    print(f"\n原始數據樣本（前 3 行）：")
    print(df[['Phase', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']].head(3))
    
    # 分析位置值
    X = df['X'].values
    Y = df['Y'].values
    Z = df['Z'].values
    RX = df['RX'].values
    RY = df['RY'].values
    RZ = df['RZ'].values
    
    print(f"\n位置值分析：")
    print(f"  X: {X.min()} ~ {X.max()}")
    print(f"  Y: {Y.min()} ~ {Y.max()}")
    print(f"  Z: {Z.min()} ~ {Z.max()}")
    
    print(f"\n旋轉值分析：")
    print(f"  RX: {RX.min()} ~ {RX.max()}")
    print(f"  RY: {RY.min()} ~ {RY.max()}")
    print(f"  RZ: {RZ.min()} ~ {RZ.max()}")
    
    # 建議轉換倍數
    print(f"\n建議轉換方案：")
    
    # 位置建議
    if X.max() > 360:  # 如果大於 360，可能是 mm
        print(f"\n1. 位置值（X, Y, Z）")
        print(f"   當前最大值：{X.max()}")
        print(f"   ✓ 建議：除以 1000（從 mm 轉成 m）")
        print(f"   轉換後範圍：{X.min()/1000:.3f} ~ {X.max()/1000:.3f} m")
        pos_factor = 1000
    else:
        print(f"\n1. 位置值（X, Y, Z）")
        print(f"   當前最大值：{X.max()}")
        print(f"   ✓ 已是合理範圍（m）")
        pos_factor = 1
    
    # 旋轉建議（檢查是否是省略小數點的度數）
    if abs(RY.min()) > 360 or RX.max() > 360:
        # 可能是省略小數點的度數（如 89999 = 89.999°）
        print(f"\n2. 旋轉值（RX, RY, RZ）")
        print(f"   當前範圍：RY {RY.min()} ~ {RY.max()}")
        print(f"   ✓ 建議：除以 1000（從省略小數點的度數轉成標準度數）")
        print(f"   轉換後範圍：{RY.min()/1000:.3f}° ~ {RY.max()/1000:.3f}°")
        rot_factor = 1000
    else:
        print(f"\n2. 旋轉值（RX, RY, RZ）")
        print(f"   當前最大值：{abs(RY).max()}")
        print(f"   ✓ 已是合理範圍（度數）")
        rot_factor = 1
    
    return pos_factor, rot_factor


def fix_arm_data(csv_path, output_path=None, pos_factor=None, rot_factor=None):
    """修正 CSV 數據格式"""
    
    print("\n" + "="*70)
    print("  修正 CSV 數據")
    print("="*70)
    
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # 如果沒有指定因子，自動分析
    if pos_factor is None or rot_factor is None:
        auto_pos, auto_rot = analyze_data_format(df)
        pos_factor = pos_factor or auto_pos
        rot_factor = rot_factor or auto_rot
    
    print(f"\n使用轉換因子：")
    print(f"  位置倍數：1/{pos_factor}")
    print(f"  旋轉倍數：1/{rot_factor}")
    
    # 應用轉換
    df_fixed = df.copy()
    df_fixed['X'] = df['X'] / pos_factor
    df_fixed['Y'] = df['Y'] / pos_factor
    df_fixed['Z'] = df['Z'] / pos_factor
    df_fixed['RX'] = df['RX'] / rot_factor
    df_fixed['RY'] = df['RY'] / rot_factor
    df_fixed['RZ'] = df['RZ'] / rot_factor
    
    # 儲存
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_fixed.csv"
    
    df_fixed.to_csv(output_path, index=False)
    print(f"\n✓ 已儲存修正後的數據：{output_path}")
    
    # 顯示修正後的樣本
    print(f"\n修正後的數據樣本（前 3 行）：")
    print(df_fixed[['Phase', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']].head(3))
    
    # 統計
    print(f"\n修正後的數據統計：")
    print(f"  位置 X：{df_fixed['X'].min():.3f} ~ {df_fixed['X'].max():.3f} m")
    print(f"  位置 Y：{df_fixed['Y'].min():.3f} ~ {df_fixed['Y'].max():.3f} m")
    print(f"  位置 Z：{df_fixed['Z'].min():.3f} ~ {df_fixed['Z'].max():.3f} m")
    print(f"  旋轉 RX：{df_fixed['RX'].min():.3f}° ~ {df_fixed['RX'].max():.3f}°")
    print(f"  旋轉 RY：{df_fixed['RY'].min():.3f}° ~ {df_fixed['RY'].max():.3f}°")
    print(f"  旋轉 RZ：{df_fixed['RZ'].min():.3f}° ~ {df_fixed['RZ'].max():.3f}°")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="修正手臂位置數據的單位格式")
    parser.add_argument("--input", default="../data/datasets/own/arm_position.csv", 
                       help="輸入 CSV 檔案路徑")
    parser.add_argument("--output", default=None, 
                       help="輸出 CSV 檔案路徑（預設：輸入檔名_fixed.csv）")
    parser.add_argument("--pos-factor", type=int, default=None,
                       help="位置轉換倍數（預設：自動偵測）")
    parser.add_argument("--rot-factor", type=int, default=None,
                       help="旋轉轉換倍數（預設：自動偵測）")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  手臂位置數據單位修正工具")
    print("="*70)
    
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"✗ 找不到檔案：{csv_path}")
        return
    
    output_path = fix_arm_data(
        csv_path,
        output_path=args.output,
        pos_factor=args.pos_factor,
        rot_factor=args.rot_factor
    )
    
    print(f"\n" + "="*70)
    print("  後續步驟")
    print("="*70)
    print(f"\n1. 檢查修正後的數據是否合理")
    print(f"\n2. 將修正後的檔案覆蓋原始檔案：")
    print(f"   mv {output_path} {csv_path}")
    print(f"\n3. 運行資料轉換腳本：")
    print(f"   python3 prepare_own_dataset.py")


if __name__ == "__main__":
    main()
