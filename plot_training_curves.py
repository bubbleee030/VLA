#!/usr/bin/env python3
"""
訓練曲線視覺化腳本
自動讀取 CSV logs 並生成精美的訓練曲線圖
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# 設定中文字體和風格
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def find_latest_csv(csv_dir):
    """找到最新的 CSV log 檔案"""
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("**/metrics.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"在 {csv_dir} 中找不到 metrics.csv 檔案")
    
    # 選擇最新的檔案
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 讀取訓練記錄：{latest_csv}")
    return latest_csv

def plot_training_curves(csv_path, save_dir='./plots'):
    """繪製訓練曲線"""
    # 讀取資料
    df = pd.read_csv(csv_path)
    print(f"📈 資料點數量：{len(df)}")
    print(f"📋 可用欄位：{df.columns.tolist()}")
    
    # 建立輸出目錄
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('觸覺控制器訓練曲線分析', fontsize=20, fontweight='bold')
    
    # ========== 圖 1：訓練 Loss (Epoch Level) ==========
    ax1 = axes[0, 0]
    if 'train_loss_epoch' in df.columns:
        epoch_data = df.dropna(subset=['train_loss_epoch'])
        ax1.plot(epoch_data['epoch'], epoch_data['train_loss_epoch'], 
                marker='o', linewidth=2, markersize=6, label='Train Loss', color='#E74C3C')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('訓練損失函數 (Train Loss)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 標註最低點
        min_idx = epoch_data['train_loss_epoch'].idxmin()
        min_epoch = epoch_data.loc[min_idx, 'epoch']
        min_loss = epoch_data.loc[min_idx, 'train_loss_epoch']
        ax1.annotate(f'最低: {min_loss:.4f}', 
                    xy=(min_epoch, min_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # ========== 圖 2：Step Level Loss (更細緻) ==========
    ax2 = axes[0, 1]
    if 'train_loss_step' in df.columns:
        step_data = df.dropna(subset=['train_loss_step'])
        ax2.plot(step_data['step'], step_data['train_loss_step'], 
                linewidth=1, alpha=0.6, color='#3498DB', label='Step Loss')
        
        # 加上滑動平均
        window = min(50, len(step_data) // 10)
        if window > 1:
            rolling_mean = step_data['train_loss_step'].rolling(window=window).mean()
            ax2.plot(step_data['step'], rolling_mean, 
                    linewidth=2.5, color='#E67E22', label=f'滑動平均 (窗口={window})')
        
        ax2.set_xlabel('訓練步數 (Step)', fontsize=12)
        ax2.set_ylabel('Loss (MSE)', fontsize=12)
        ax2.set_title('訓練損失函數 (Step Level)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    # ========== 圖 3：相關係數 (Correlation) ==========
    ax3 = axes[1, 0]
    if 'train_correlation' in df.columns:
        corr_data = df.dropna(subset=['train_correlation'])
        ax3.plot(corr_data['epoch'], corr_data['train_correlation'], 
                marker='s', linewidth=2, markersize=6, label='Correlation', color='#27AE60')
        ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='目標 (0.9)')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('相關係數 (Pearson)', fontsize=12)
        ax3.set_title('預測相關性分析', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
    
    # ========== 圖 4：學習率變化 ==========
    ax4 = axes[1, 1]
    if 'learning_rate' in df.columns:
        lr_data = df.dropna(subset=['learning_rate'])
        ax4.plot(lr_data['epoch'], lr_data['learning_rate'], 
                marker='D', linewidth=2, markersize=6, label='Learning Rate', color='#9B59B6')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('學習率', fontsize=12)
        ax4.set_title('學習率調度', fontsize=14, fontweight='bold')
        ax4.set_yscale('log')  # 使用對數刻度
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # 儲存圖表
    plot_path = save_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已儲存：{plot_path}")
    
    # 也儲存成 PDF（向量圖，適合論文）
    pdf_path = save_dir / 'training_curves.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ PDF 已儲存：{pdf_path}")
    
    plt.show()
    
    # ========== 輸出統計摘要 ==========
    print("\n" + "="*50)
    print("📊 訓練統計摘要")
    print("="*50)
    
    if 'train_loss_epoch' in df.columns:
        epoch_data = df.dropna(subset=['train_loss_epoch'])
        print(f"\n【訓練 Loss】")
        print(f"  初始 Loss: {epoch_data['train_loss_epoch'].iloc[0]:.6f}")
        print(f"  最終 Loss: {epoch_data['train_loss_epoch'].iloc[-1]:.6f}")
        print(f"  最低 Loss: {epoch_data['train_loss_epoch'].min():.6f} (Epoch {epoch_data.loc[epoch_data['train_loss_epoch'].idxmin(), 'epoch']:.0f})")
        print(f"  Loss 下降: {(1 - epoch_data['train_loss_epoch'].iloc[-1]/epoch_data['train_loss_epoch'].iloc[0])*100:.2f}%")
    
    if 'train_correlation' in df.columns:
        corr_data = df.dropna(subset=['train_correlation'])
        print(f"\n【預測相關性】")
        print(f"  最終相關係數: {corr_data['train_correlation'].iloc[-1]:.4f}")
        print(f"  最高相關係數: {corr_data['train_correlation'].max():.4f}")

def main():
    parser = argparse.ArgumentParser(description='繪製訓練曲線')
    parser.add_argument('--csv_dir', type=str, default='./logs/csv',
                        help='CSV logs 目錄')
    parser.add_argument('--save_dir', type=str, default='./plots',
                        help='圖表儲存目錄')
    args = parser.parse_args()
    
    try:
        csv_path = find_latest_csv(args.csv_dir)
        plot_training_curves(csv_path, args.save_dir)
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()