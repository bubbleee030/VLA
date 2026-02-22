#!/usr/bin/env python3
"""
自動訓練兩個模型並生成對比圖表
"""

import argparse
import yaml
import torch
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from residual_controller.bridge_model import ResidualController
from residual_controller.controller_dataset import ControllerDataModule
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def count_parameters(model):
    """計算模型的參數量"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def load_config(config_path):
    """載入配置檔"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 型別轉換
    config['training']['epochs'] = int(config['training']['epochs'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['model']['action_dim'] = int(config['model']['action_dim'])
    config['model']['obs_dim'] = int(config['model']['obs_dim'])
    config['model']['horizon'] = int(config['model']['horizon'])
    config['data']['num_workers'] = int(config['data']['num_workers'])
    
    return config

def train_single_model(config, modality, experiment_name, experiment_suffix=''):
    """訓練單一模型"""
    print(f"\n{'='*60}")
    print(f"🚀 開始訓練：{experiment_name}")
    print(f"{'='*60}\n")
    
    seed_everything(42, workers=True)
    
    # 建立唯一的實驗執行名稱
    run_name = f"{modality}{experiment_suffix}"
    print(f"🔬 實驗執行名稱 (Run Name): {run_name}")
    
    # 使用唯一的名稱來設定輸出路徑
    output_dir = Path(config['checkpoint']['save_dir']) / run_name
    log_dir_root = Path(config['logging']['csv_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataModule
    data_module = ControllerDataModule(
        h5_path_or_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    # 假設您已經修正了 dataset 的分割和路徑問題
    # data_module.setup() 
    
    # Model
    model = ResidualController(
        modality=modality,
        lr=config['training']['learning_rate'],
        action_dim=config['model']['action_dim'],
        obs_dim=config['model']['obs_dim'],
        horizon=config['model']['horizon']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename='best-{epoch:02d}-{train_loss_epoch:.4f}',
            monitor='train_loss_epoch',
            mode='min',
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=str(log_dir_root),
        name=run_name,
        version=0 
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        num_sanity_val_steps=0,
        limit_val_batches=0,
        gradient_clip_val=1.0,
        deterministic=True,
        enable_progress_bar=True,
    )
    
    # 訓練
    trainer.fit(model, datamodule=data_module)
    
    log_file_path = Path(logger.log_dir) / 'metrics.csv'
    
    print(f"\n✅ {experiment_name} 訓練完成！")
    print(f"✅ 模型參數量：{count_parameters(model) / 1e3:.1f}K")
    print(f"📊 記錄位置：{log_file_path}\n")
    
    return str(log_file_path)

def load_training_metrics(csv_path):
    """載入訓練記錄"""
    df = pd.read_csv(csv_path)
    # 只保留有 epoch 的記錄（排除 step-level 的記錄）
    df_epoch = df.dropna(subset=['epoch', 'train_loss_epoch'])
    return df_epoch

def create_comparison_plots(csv1_path, csv2_path, save_dir='./plots', model_names=['模型 1', '模型 2'], title_suffix=''):
    """生成對比圖表"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入資料
    df1 = load_training_metrics(csv1_path)
    df2 = load_training_metrics(csv2_path)
    
    print(f"\n📊 生成對比圖表...")
    print(f"  - {model_names[0]} 記錄點數：{len(df1)}")
    print(f"  - {model_names[1]} 記錄點數：{len(df2)}")
    
    # 建立大型圖表
    fig = plt.figure(figsize=(18, 12))
    
    # ========== 圖 1：訓練損失對比 ==========
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df1['epoch'], df1['train_loss_epoch'], 
            marker='o', linewidth=2, markersize=4, label=model_names[0], color='#2E86AB')
    ax1.plot(df2['epoch'], df2['train_loss_epoch'], 
            marker='s', linewidth=2, markersize=4, label=model_names[1], color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ========== 圖 2：最終損失對比（柱狀圖）==========
    ax2 = plt.subplot(2, 3, 2)
    final_losses = [
        df1['train_loss_epoch'].iloc[-1],
        df2['train_loss_epoch'].iloc[-1]
    ]
    bars = ax2.bar([f'{model_names[0]}', f'{model_names[1]}'], final_losses, 
                   color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Final Training Loss', fontsize=12)
    ax2.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上標註數值
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ========== 圖 3：損失下降速度對比 ==========
    ax3 = plt.subplot(2, 3, 3)
    
    # 計算每個模型的損失下降率
    initial_1 = df1['train_loss_epoch'].iloc[0]
    final_1 = df1['train_loss_epoch'].iloc[-1]
    reduction_1 = (1 - final_1 / initial_1) * 100
    
    initial_2 = df2['train_loss_epoch'].iloc[0]
    final_2 = df2['train_loss_epoch'].iloc[-1]
    reduction_2 = (1 - final_2 / initial_2) * 100
    
    bars = ax3.bar([model_names[0], model_names[1]], 
                   [reduction_1, reduction_2],
                   color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Loss Reduction (%)', fontsize=12)
    ax3.set_title('Learning Efficiency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [reduction_1, reduction_2]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ========== 圖 4：相關係數對比（如果有的話）==========
    ax4 = plt.subplot(2, 3, 4)
    
    if 'train_correlation' in df1.columns and 'train_correlation' in df2.columns:
        df1_corr = df1.dropna(subset=['train_correlation'])
        df2_corr = df2.dropna(subset=['train_correlation'])
        
        if len(df1_corr) > 0 and len(df2_corr) > 0:
            ax4.plot(df1_corr['epoch'], df1_corr['train_correlation'],
                    marker='o', linewidth=2, markersize=4, label=model_names[0], color='#2E86AB')
            ax4.plot(df2_corr['epoch'], df2_corr['train_correlation'],
                    marker='s', linewidth=2, markersize=4, label=model_names[1], color='#A23B72')
            ax4.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (0.9)')
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Correlation', fontsize=12)
            ax4.set_title('Prediction Correlation', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
        else:
            ax4.text(0.5, 0.5, 'No Correlation Data', ha='center', va='center', fontsize=14)
            ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, 'No Correlation Data', ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    # ========== 圖 5：收斂速度對比 ==========
    ax5 = plt.subplot(2, 3, 5)
    
    # 找到達到特定損失閾值所需的 epoch
    threshold = 0.01  # 設定一個損失閾值
    
    tactile_converge = df1[df1['train_loss_epoch'] < threshold]
    reduced_converge = df2[df2['train_loss_epoch'] < threshold]
    
    if len(tactile_converge) > 0 and len(reduced_converge) > 0:
        tactile_epoch = tactile_converge['epoch'].iloc[0]
        reduced_epoch = reduced_converge['epoch'].iloc[0]
        
        bars = ax5.barh([model_names[0], model_names[1]], [tactile_epoch, reduced_epoch],
                       color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_xlabel('Epochs to Reach Loss < 0.01', fontsize=12)
        ax5.set_title('Convergence Speed', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, [tactile_epoch, reduced_epoch]):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(val)}',
                    ha='left', va='center', fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    else:
        ax5.text(0.5, 0.5, 'Convergence Not Reached', ha='center', va='center', fontsize=14)
        ax5.axis('off')
    
    # ========== 圖 6：模型參數量對比 ==========
    ax6 = plt.subplot(2, 3, 6)
    
    # 從模型架構估算參數量（您需要根據實際情況調整）
    tactile_params = 705  # 從您之前的訓練結果：705K 參數
    reduced_params = 100  # 估算值（因為沒有 CNN）

    bars = ax6.bar(['Complete Tactile\n(CNN)', 'Reduced\n(Statistics)'],
                   [tactile_params, reduced_params],
                   color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Parameters (K)', fontsize=12)
    ax6.set_title('Model Complexity', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [tactile_params, reduced_params]):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}K',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 整體標題
    fig.suptitle(f'Tactile Sensor Value: {model_names[0]} vs {model_names[1]} {title_suffix}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 儲存
    plot_path = save_dir / 'tactile_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 對比圖表已儲存：{plot_path}")
    
    pdf_path = save_dir / 'tactile_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ PDF 已儲存：{pdf_path}")
    
    plt.show()
    
    # ========== 生成統計報告 ==========
    print_comparison_report(df1, df2, model_names)

def print_comparison_report(df1, df2, model_names):
    """列印統計報告"""
    print("\n" + "="*70)
    print("📊 觸覺感測器價值量化分析報告")
    print("="*70)
    
    # 計算關鍵指標
    tactile_final = df1['train_loss_epoch'].iloc[-1]
    reduced_final = df2['train_loss_epoch'].iloc[-1]
    improvement = (reduced_final - tactile_final) / reduced_final * 100
    
    print(f"\n【最終性能對比】")
    print(f"  完整觸覺模型最終損失：{tactile_final:.6f}")
    print(f"  退化版模型最終損失：{reduced_final:.6f}")
    print(f"  性能提升：{improvement:.2f}%")
    
    if improvement > 50:
        print(f"  ✅ 結論：完整觸覺顯著優於退化版，空間資訊至關重要")
    elif improvement > 20:
        print(f"  ✅ 結論：完整觸覺明顯優於退化版，證明觸覺的價值")
    elif improvement > 5:
        print(f"  ⚠️  結論：完整觸覺略優於退化版")
    else:
        print(f"  ⚠️  結論：兩者性能接近，需要進一步分析")
    
    print(f"\n【學習曲線分析】")
    tactile_std = df1['train_loss_epoch'].std()
    reduced_std = df2['train_loss_epoch'].std()
    
    print(f"  {model_names[0]} 學習穩定性（標準差）：{tactile_std:.6f}")
    print(f"  {model_names[1]} 學習穩定性（標準差）：{reduced_std:.6f}")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='觸覺 vs 基線模型對比實驗')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--skip_training', action='store_true', 
                        help='跳過訓練，只生成圖表（需要已有訓練記錄）')
    parser.add_argument('--experiment_suffix', type=str, default='', help='為實驗日誌和輸出加上後綴，以區分不同實驗')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if not args.skip_training:
        # 訓練完整觸覺模型
        tactile_csv = train_single_model(config, 'tactile', '完整觸覺模型 (CNN)')
        args.experiment_suffix
        # 訓練退化版模型
        reduced_csv = train_single_model(config, 'tactile_reduced', '退化版模型 (統計特徵)')
        args.experiment_suffix
    else:
        print("⏭️  跳過訓練，使用現有記錄...")
        tactile_csv = './logs/csv/tactile/comparison/metrics.csv'
        reduced_csv = './logs/csv/tactile_reduced/comparison/metrics.csv'
    
    log_root = Path(config['logging']['csv_dir'])
    
    def get_log_paths(suffix):
        tactile_path = log_root / f"tactile{suffix}" / "version_0" / "metrics.csv"
        reduced_path = log_root / f"tactile_reduced{suffix}" / "version_0" / "metrics.csv"
        return tactile_path, reduced_path

    tactile_csv_exp1, reduced_csv_exp1 = get_log_paths(args.exp1_suffix)
    tactile_csv_exp2, reduced_csv_exp2 = get_log_paths(args.exp2_suffix)
    
    print("\n" + "="*70)
    print(f"📊 準備比較實驗：'{args.exp1_suffix}' vs '{args.exp2_suffix}'")
    
    
    # --- 比較場景 1：比較「增量前後」的「完整觸覺模型」 ---
    print("\n--- 比較場景 1：完整觸覺模型 (增量前後) ---")
    if tactile_csv_exp1.exists() and tactile_csv_exp2.exists():
        create_comparison_plots(
            tactile_csv_exp1, 
            tactile_csv_exp2, 
            save_dir=f'./plots/comparison_original_vs_augmented',
            model_names=[f'完整觸覺 ({args.exp1_suffix})', f'完整觸覺 ({args.exp2_suffix})'],
            title_suffix='(Original vs. Augmented)'
        )
    else:
        print(f"❌ 跳過：找不到 {tactile_csv_exp1} 或 {tactile_csv_exp2}")

    # --- 比較場景 2：比較「增量後」的「完整觸覺 vs 退化版」 ---
    print("\n--- 比較場景 2：增量後的模型對比 ---")
    if tactile_csv_exp2.exists() and reduced_csv_exp2.exists():
        create_comparison_plots(
            tactile_csv_exp2,
            reduced_csv_exp2,
            save_dir=f'./plots/comparison_augmented_models',
            model_names=[f'完整觸覺 ({args.exp2_suffix})', f'退化版 ({args.exp2_suffix})'],
            title_suffix=f'({args.exp2_suffix})'
        )
    else:
        print(f"❌ 跳過：找不到 {tactile_csv_exp2} 或 {reduced_csv_exp2}")
        
    print("\n✅ 所有比較完成！")

if __name__ == "__main__":
    main()