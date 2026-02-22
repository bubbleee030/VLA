#!/usr/bin/env python3
"""
最終版實驗主程式：資料增量前後對比實驗

功能：
1. 分別在「原始」和「增量」資料集上訓練模型。
2. 將兩次實驗的日誌和模型儲存到獨立的資料夾，避免覆蓋。
3. 訓練結束後，自動生成量化分析報告與視覺化對比圖表。
"""

import argparse
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from residual_controller.bridge_model import ResidualController
from residual_controller.controller_dataset import ControllerDataModule

# --- 設定 Matplotlib 字體 ---
plt.rcParams['font.sans-serif'] = ['Noto Sans TC', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_config(config_path):
    """Load and parse the YAML config file, ensuring correct data types."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- 【重要修正】確保 learning_rate 是浮點數 ---
    # PyYAML 會將 '1e-4' 讀成字串，我們需要手動轉換
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    
    # 為了保險起見，也檢查其他數值型別
    config['training']['epochs'] = int(config['training']['epochs'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['data']['num_workers'] = int(config['data']['num_workers'])
    config['model']['action_dim'] = int(config['model']['action_dim'])
    config['model']['obs_dim'] = int(config['model']['obs_dim'])
    config['model']['horizon'] = int(config['model']['horizon'])
    
    return config

def train_single_model(config, data_dir, modality, experiment_suffix):
    """
    訓練單一模型，並將結果儲存到唯一的路徑。
    """
    run_name = f"{modality}{experiment_suffix}"
    experiment_name = f"{'完整觸覺' if modality == 'tactile' else '退化版'} ({experiment_suffix.strip('_')})"
    
    print("\n" + "="*70)
    print(f"🚀 開始訓練：{experiment_name}")
    print(f"🔬 實驗執行名稱 (Run Name): {run_name}")
    print(f"📁 資料來源: {data_dir}")
    print("="*70)
    
    seed_everything(42, workers=True)
    
    # --- 設定唯一的輸出路徑 ---
    output_dir = Path(config['checkpoint']['save_dir']) / run_name
    log_dir_root = Path(config['logging']['csv_dir'])
    
    # --- 自動清理舊紀錄 (重要！) ---
    if output_dir.exists():
        print(f"🧹 清理舊模型目錄：{output_dir}")
        shutil.rmtree(output_dir)
    if (log_dir_root / run_name).exists():
        print(f"🧹 清理舊日誌目錄：{log_dir_root / run_name}")
        shutil.rmtree(log_dir_root / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 資料模組 ---
    data_module = ControllerDataModule(
        h5_path_or_dir=data_dir,
        horizon=config['model']['horizon'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_ratio=config['data']['train_ratio'],
        seed=config['data']['random_seed']
    )
    
    # --- 模型 ---
    model = ResidualController(
        modality=modality,
        lr=config['training']['learning_rate'],
        action_dim=config['model']['action_dim'],
        obs_dim=config['model']['obs_dim'],
        horizon=config['model']['horizon']
    )
    
    # --- Logger 和 Callbacks ---
    logger = CSVLogger(save_dir=str(log_dir_root), name=run_name, version=0)
    callbacks = [
        ModelCheckpoint(dirpath=str(output_dir), filename='best-{epoch:02d}-{val_loss:.4f}', monitor='val_loss', mode='min', save_top_k=1, save_last=True),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # --- 訓練器 ---
    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
    )
    
    # --- 開始訓練與測試 ---
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path='best')
    
    log_file_path = Path(logger.log_dir) / 'metrics.csv'
    print(f"\n✅ {experiment_name} 訓練完成！")
    print(f"📊 記錄已儲存至：{log_file_path}\n")
    
    return str(log_file_path)

def analyze_and_plot(csv_path1, csv_path2, name1, name2, save_dir, version_tag):
    """Load logs from two experiments, generate a comparison report and plots."""
    
    df1 = pd.read_csv(csv_path1).dropna(subset=['epoch'])
    df2 = pd.read_csv(csv_path2).dropna(subset=['epoch'])

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Calculate Key Metrics ---
    metrics = {}
    for df, name in [(df1, name1), (df2, name2)]:
        metrics[name] = {
            'train_loss': df['train_loss_epoch'].dropna().iloc[-1],
            'test_loss': df['test_loss'].dropna().iloc[-1] if 'test_loss' in df.columns and not df['test_loss'].dropna().empty else float('nan'),
        }
        if not np.isnan(metrics[name]['test_loss']):
            metrics[name]['gap'] = (metrics[name]['test_loss'] - metrics[name]['train_loss']) / metrics[name]['train_loss'] * 100
        else:
            metrics[name]['gap'] = float('nan')

    # --- 2. Generate Quantitative Report (Corrected) ---
    print("\n" + "="*80)
    print("📊 Data Augmentation Benefit Analysis Report")
    print("="*80)
    
    # --- 💡【核心修正】預先格式化所有需要顯示的字串 ---
    # 預先計算變化率
    train_loss_change = (metrics[name2]["train_loss"]-metrics[name1]["train_loss"])/metrics[name1]["train_loss"]*100 if metrics[name1]["train_loss"] != 0 else float('inf')
    test_loss_change = (metrics[name2]["test_loss"]-metrics[name1]["test_loss"])/metrics[name1]["test_loss"]*100 if not np.isnan(metrics[name1]['test_loss']) and metrics[name1]['test_loss'] != 0 else float('inf')
    gap_change = metrics[name2]["gap"] - metrics[name1]["gap"] if not np.isnan(metrics[name1]['gap']) and not np.isnan(metrics[name2]['gap']) else float('nan')
    
    # 將所有需要特殊格式的數字都變成字串
    gap1_str = f"{metrics[name1]['gap']:+.1f}%" if not np.isnan(metrics[name1]['gap']) else "N/A"
    gap2_str = f"{metrics[name2]['gap']:+.1f}%" if not np.isnan(metrics[name2]['gap']) else "N/A"
    train_loss_change_str = f"{train_loss_change:+.1f}%"
    test_loss_change_str = f"{test_loss_change:+.1f}%"
    gap_change_str = f"{gap_change:+.1f} p.p." if not np.isnan(gap_change) else "N/A"

    # --- 現在所有的 print 函式都變得非常簡單 ---
    print(f"{'Metric':<20} | {name1:<25} | {name2:<25} | {'Change':<10}")
    print("-" * 80)
    print(f"{'Final Train Loss':<20} | {metrics[name1]['train_loss']:<25.6f} | {metrics[name2]['train_loss']:<25.6f} | {train_loss_change_str:<10}")
    print(f"{'Final Test Loss':<20} | {metrics[name1]['test_loss']:<25.6f} | {metrics[name2]['test_loss']:<25.6f} | {test_loss_change_str:<10}")
    print(f"{'Generalization Gap':<20} | {gap1_str:<25} | {gap2_str:<25} | {gap_change_str:<10}")
    print("-" * 80)
    
    # --- 繪製圖表 ---
    plt.figure(figsize=(18, 8))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Data Augmentation Benefit Analysis ({version_tag} vs 1x)', fontsize=20, weight='bold')
    
    # 左圖：學習曲線
    axes[0].plot(df1['epoch'], df1['train_loss_epoch'], 'o--', markersize=3, color='#2E86AB', label=f'Training Loss ({name1})')
    axes[0].plot(df1.dropna(subset=['test_loss'])['epoch'], df1.dropna(subset=['test_loss'])['test_loss'], 'o-', markersize=4, color='#2E86AB', label=f'Validation Loss ({name1})')
    axes[0].plot(df2['epoch'], df2['train_loss_epoch'], 's--', markersize=3, color='#A23B72', label=f'Training Loss ({name2})')
    axes[0].plot(df2.dropna(subset=['test_loss'])['epoch'], df2.dropna(subset=['test_loss'])['test_loss'], 's-', markersize=4, color='#A23B72', label=f'Validation Loss ({name2})')
    axes[0].set_title('Learning Curve Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE) - Log Scale', fontsize=12)
    axes[0].legend()
    axes[0].set_yscale('log')

    # 右圖：最終測試 Loss
    labels = [name1, name2]
    final_test_losses = [metrics[name1]['test_loss'], metrics[name2]['test_loss']]
    bars = axes[1].bar(labels, final_test_losses, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')
    axes[1].set_title('Final Test Performance Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Final Test Loss (MSE)', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.6f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = save_path / 'augmentation_comparison_report.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ 對比圖表已儲存至：{plot_path}")


def main():
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser(description='執行並比較資料增量前後的實驗')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='基礎訓練設定檔')
    parser.add_argument('--original_data', type=str, default='./data/datasets/mango_hdf5_gelsight', help='原始資料集路徑')
    parser.add_argument('--augmented_data', type=str, default='./data/datasets/mango_hdf5_augmented_5x', help='增量後資料集路徑')
    parser.add_argument('--skip_training', action='store_true', help='跳過訓練，直接從現有日誌生成圖表')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 定義實驗後綴
    suffix1 = '_original'
    suffix2 = '_augmented'
    
    if not args.skip_training:
        print("--- 將重新執行兩次訓練，這可能會花費一些時間 ---")
        # 重新訓練「原始」資料集
        csv_path1 = train_single_model(config, args.original_data, 'tactile', suffix1)
        # 重新訓練「增量」資料集
        csv_path2 = train_single_model(config, args.augmented_data, 'tactile', suffix2)
    else:
        print("--- 跳過訓練，直接使用現有日誌進行分析 ---")
        log_root = Path(config['logging']['csv_dir'])
        csv_path1 = log_root / f"tactile{suffix1}" / "version_0" / "metrics.csv"
        csv_path2 = log_root / f"tactile{suffix2}" / "version_0" / "metrics.csv"

    # 進行最終分析與繪圖
    if Path(csv_path1).exists() and Path(csv_path2).exists():
        analyze_and_plot(csv_path1, csv_path2, 'Original Data', 'Augmented Data', './plots')
    else:
        print(f"❌ 錯誤：找不到必要的日誌檔案。請檢查路徑或執行訓練。")
        print(f"  - 應存在: {csv_path1}")
        print(f"  - 應存在: {csv_path2}")

if __name__ == "__main__":
    main()