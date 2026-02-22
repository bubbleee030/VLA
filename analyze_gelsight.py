#!/usr/bin/env python3
"""
GelSight 觸覺影像分析與視覺化工具
用於向教授展示觸覺資料的時序變化
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path
import argparse

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_tactile_sequence(h5_file, max_frames=10):
    """載入觸覺影像序列"""
    with h5py.File(h5_file, 'r') as f:
        tactile = f['tactile/tactile'][:]
        # 只取前 max_frames 幀
        tactile = tactile[:max_frames]
    return tactile

def compute_frame_difference(frame1, frame2):
    """計算兩幀之間的差異"""
    # 轉換成灰階
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    # 計算絕對差異
    diff = cv2.absdiff(gray1, gray2)
    
    return diff

def enhance_differences(diff_map, amplification=5):
    """放大差異以便肉眼觀察"""
    enhanced = np.clip(diff_map.astype(float) * amplification, 0, 255).astype(np.uint8)
    return enhanced

def create_analysis_figure(tactile_sequence, save_path='./plots/gelsight_analysis.png'):
    """建立完整的分析圖表"""
    n_frames = len(tactile_sequence)
    
    # 建立大型圖表
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, n_frames, figure=fig, hspace=0.3, wspace=0.2)
    
    fig.suptitle('GelSight 觸覺影像時序分析', fontsize=20, fontweight='bold')
    
    # ========== 第一行：原始影像序列 ==========
    for i in range(n_frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(tactile_sequence[i])
        ax.set_title(f'幀 {i}', fontsize=12)
        ax.axis('off')
    
    # 在第一行左側加上標籤
    fig.text(0.02, 0.75, '原始影像', fontsize=14, fontweight='bold', rotation=90, va='center')
    
    # ========== 第二行：幀間差異（原始） ==========
    for i in range(n_frames - 1):
        ax = fig.add_subplot(gs[1, i])
        diff = compute_frame_difference(tactile_sequence[i], tactile_sequence[i+1])
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=255)
        ax.set_title(f'差異 {i}→{i+1}', fontsize=12)
        ax.axis('off')
        
        # 加上顏色條
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.text(0.02, 0.45, '幀間差異\n(原始)', fontsize=14, fontweight='bold', rotation=90, va='center')
    
    # ========== 第三行：幀間差異（放大 10 倍） ==========
    for i in range(n_frames - 1):
        ax = fig.add_subplot(gs[2, i])
        diff = compute_frame_difference(tactile_sequence[i], tactile_sequence[i+1])
        enhanced = enhance_differences(diff, amplification=10)
        im = ax.imshow(enhanced, cmap='jet', vmin=0, vmax=255)
        ax.set_title(f'放大差異 {i}→{i+1}', fontsize=12)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.text(0.02, 0.15, '幀間差異\n(放大10倍)', fontsize=14, fontweight='bold', rotation=90, va='center')
    
    # 儲存圖表
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 分析圖表已儲存：{save_path}")
    
    plt.show()

def compute_statistics(tactile_sequence):
    """計算並印出統計資訊"""
    print("\n" + "="*60)
    print("📊 GelSight 觸覺資料統計分析")
    print("="*60)
    
    print(f"\n【基本資訊】")
    print(f"  總幀數: {len(tactile_sequence)}")
    print(f"  影像尺寸: {tactile_sequence[0].shape}")
    print(f"  資料類型: {tactile_sequence.dtype}")
    print(f"  像素值範圍: [{tactile_sequence.min()}, {tactile_sequence.max()}]")
    
    print(f"\n【時序變化分析】")
    total_change = 0
    max_change = 0
    max_change_pair = (0, 1)
    
    for i in range(len(tactile_sequence) - 1):
        diff = compute_frame_difference(tactile_sequence[i], tactile_sequence[i+1])
        mean_diff = diff.mean()
        max_diff = diff.max()
        total_change += mean_diff
        
        print(f"  幀 {i}→{i+1}:")
        print(f"    - 平均差異: {mean_diff:.2f}")
        print(f"    - 最大差異: {max_diff:.2f}")
        print(f"    - 變化像素比例: {(diff > 10).sum() / diff.size * 100:.2f}%")
        
        if mean_diff > max_change:
            max_change = mean_diff
            max_change_pair = (i, i+1)
    
    print(f"\n【結論】")
    print(f"  平均每幀變化量: {total_change / (len(tactile_sequence) - 1):.2f}")
    print(f"  最大變化發生在: 幀 {max_change_pair[0]}→{max_change_pair[1]} (差異={max_change:.2f})")
    
    if total_change < 5:
        print(f"\n  ⚠️  警告：整體變化量很小 ({total_change:.2f})")
        print(f"      這可能代表：")
        print(f"      1. 操作過程中接觸壓力變化極小")
        print(f"      2. 感測器位置在整個 episode 中幾乎沒有相對移動")
        print(f"      3. 這是一個「靜態接觸」的 episode")
    else:
        print(f"\n  ✅ 資料包含明顯的時序變化，適合訓練")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='分析 GelSight 觸覺影像')
    parser.add_argument('--h5_file', type=str, 
                        default='./octopi/mango_new_samples_span_hdf5_gelsight/mango_new_0.h5',
                        help='HDF5 檔案路徑')
    parser.add_argument('--max_frames', type=int, default=8,
                        help='分析的最大幀數')
    parser.add_argument('--save_dir', type=str, default='./plots',
                        help='圖表儲存目錄')
    args = parser.parse_args()
    
    print(f"📂 載入檔案：{args.h5_file}")
    
    # 載入資料
    tactile_seq = load_tactile_sequence(args.h5_file, max_frames=args.max_frames)
    print(f"✅ 成功載入 {len(tactile_seq)} 幀影像")
    
    # 計算統計資訊
    compute_statistics(tactile_seq)
    
    # 建立視覺化圖表
    save_path = Path(args.save_dir) / 'gelsight_analysis.png'
    create_analysis_figure(tactile_seq, save_path=str(save_path))
    
    print(f"\n💡 使用建議：")
    print(f"   將生成的圖表 ({save_path}) 展示給教授")
    print(f"   重點說明「幀間差異（放大10倍）」這一行的變化")

if __name__ == "__main__":
    main()