# run_and_compare_augmentation.py (最終重構版)

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

sns.set_style("whitegrid")

def train_single_model(config, data_dir, modality_name, suffix):
    """訓練單一模型並回傳其日誌檔案的路徑。"""
    # (此函式與您之前的版本完全相同，無需改動)
    seed_everything(config['data']['random_seed'], workers=True)
    run_name = f"{modality_name}_{suffix}"
    output_dir = Path("outputs") / run_name
    log_dir_root = Path("logs/csv")
    
    if output_dir.exists():
        print(f"🧹 Cleaning up old model directory: {output_dir}")
        shutil.rmtree(output_dir)
    log_dir = log_dir_root / run_name
    if log_dir.exists():
        print(f"🧹 Cleaning up old log directory: {log_dir}")
        shutil.rmtree(log_dir)

    data_module = ControllerDataModule(
        h5_path_or_dir=data_dir,
        horizon=config['model']['horizon'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_ratio=config['data']['train_ratio'],
        seed=config['data']['random_seed']
    )
    model = ResidualController(
        action_dim=config['model']['action_dim'],
        obs_dim=config['model']['obs_dim'],
        horizon=config['model']['horizon'],
        lr=config['training']['learning_rate'],
        tactile_feature_dim=config['model'].get('tactile_feature_dim', 126)
    )
    logger = CSVLogger(save_dir=log_dir_root, name=run_name, version=0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best-epoch={epoch}-val_loss={val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        precision=config['training']['precision']
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
    log_file_path = Path(logger.log_dir) / "metrics.csv"
    print(f"✅ Training complete for: {run_name}")
    print(f"📊 Logs saved to: {log_file_path}\n")
    
    return log_file_path

def plot_learning_curves(ax, all_metrics, colors, markers):
    """在給定的 Matplotlib Axes 上繪製學習曲線 (Training/Validation Loss + Validation R²)。"""
    # Validation R² 加上後有點醜 所以先註解掉
    for name, data in all_metrics.items():
        color = colors.get(name)
        marker = markers.get(name)
        
        # 讀取完整的 DataFrame
        df = data['df']
        
        # 在繪圖前，過濾掉包含 NaN 的行
        # 針對 Validation Loss，只取 'epoch' 和 'val_loss' 都有值的行
        val_df = df[['epoch', 'val_loss']].dropna()
        
        # 針對 Training Loss，只取 'epoch' 和 'train_loss_epoch' 都有值的行
        train_df = df[['epoch', 'train_loss_epoch']].dropna()
        
        # 針對 Validation R² Score，只取 'epoch' 和 'val_r2' 都有值的行
        # val_r2_df = df[['epoch', 'val_r2']].dropna()

        # 繪製 Validation Loss
        ax.plot(val_df['epoch'], val_df['val_loss'], marker=marker, markersize=4, linestyle='-', color=color, label=f'Val Loss ({name})', linewidth=2)
        
        # 繪製 Training Loss (使用更淡的顏色和虛線)
        ax.plot(train_df['epoch'], train_df['train_loss_epoch'], marker=marker, markersize=3, linestyle='--', color=color, alpha=0.6, label=f'Train Loss ({name})', linewidth=1.5)
    
    ax.set_title('Learning Curve Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE) - Log Scale', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", c='gray', alpha=0.5)
    
    '''
    # 建立雙Y軸用於 R² Score
    ax2 = ax.twinx()
    
    for name, data in all_metrics.items():
        color = colors.get(name)
        marker = markers.get(name)
        df = data['df']
        val_r2_df = df[['epoch', 'val_r2']].dropna()
        
        if len(val_r2_df) > 0:
            # 使用不同的標記樣式區分 R² (點狀)
            ax2.plot(val_r2_df['epoch'], val_r2_df['val_r2'], marker=marker, markersize=4, linestyle=':', color=color, alpha=0.8, label=f'Val R² ({name})', linewidth=2)
    
    ax2.set_ylabel('Validation R² Score', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1])
    
    '''
    # 合併兩個軸的圖例
    lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 , labels1 , loc='center left', fontsize='small')

def plot_final_performance(ax, all_metrics, colors):
    """在給定的 Matplotlib Axes 上繪製最終性能長條圖 (Train Loss, Test Loss, R² Score)。"""
    labels = list(all_metrics.keys())
    final_train_losses = [data['final_train_loss'] for data in all_metrics.values()]
    final_test_losses = [data['final_test_loss'] for data in all_metrics.values()]
    test_r2_scores = [data['test_r2'] for data in all_metrics.values()]
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Create bars for each metric
    bars1 = ax.bar(x - width, final_train_losses, width, label='Train Loss', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, final_test_losses, width, label='Test Loss', alpha=0.8, edgecolor='black')
    
    # Create secondary y-axis for R² Score
    ax2 = ax.twinx()
    bars3 = ax2.bar(x + width, test_r2_scores, width, label='Test R² Score', alpha=0.8, color='green', edgecolor='black')
    
    ax.set_title('Final Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Version', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=30)
    
    # 在長條圖上標示數值
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars3:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')

def generate_report_and_plots(all_metrics, save_dir):
    """產生最終的量化報告與整合圖表。"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # --- 產生量化報告 ---
    report_lines = []
    report_lines.append("\n" + "="*140)
    report_lines.append("📊 Comprehensive Data Augmentation Benefit Analysis Report")
    report_lines.append("="*140)
    report_lines.append(f"{'Version':<15} | {'Train Loss':<18} | {'Test Loss':<18} | {'Test R² Score':<18} | {'Generalization Gap':<20}")
    report_lines.append("-" * 140)
    
    for name, data in all_metrics.items():
        gap_str = f"{data['gap']:.2f}%" if not np.isnan(data['gap']) else "N/A"
        r2_str = f"{data['test_r2']:.4f}" if not np.isnan(data['test_r2']) else "N/A"
        report_lines.append(f"{name:<15} | {data['final_train_loss']:<18.6f} | {data['final_test_loss']:<18.6f} | {r2_str:<18} | {gap_str:<20}")
    report_lines.append("-" * 140)
    
    report_text = "\n".join(report_lines)
    
    # Print to terminal
    print(report_text)
    
    # Save to file
    report_file = save_path / "final_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"\n✅ Final report saved to: {report_file}")

    # --- 繪製整合圖表 ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Overall Experiment Comparison', fontsize=22, weight='bold')
    
    # 定義顏色和標記，確保一致性
    color_palette = sns.color_palette("viridis", len(all_metrics))
    colors = {name: color_palette[i] for i, name in enumerate(all_metrics.keys())}
    markers = ['o', 's', 'X', 'D', '^', 'v']
    marker_map = {name: markers[i % len(markers)] for i, name in enumerate(all_metrics.keys())}
    
    plot_learning_curves(axes[0], all_metrics, colors, marker_map)
    plot_final_performance(axes[1], all_metrics, colors)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # 檔名現在是固定的，因為它是一張包含所有結果的總圖
    plot_path = save_path / 'OVERALL_comparison_report.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Overall comparison plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Run and compare multiple data augmentation experiments.")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to the configuration file.')
    # ✅ 允許多個增量資料集路徑
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset paths to compare. The first one is the baseline (e.g., 1x).')
    parser.add_argument('--skip_training', action='store_true', help='If set, skip training and only generate plots from existing logs.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    all_metrics = {}
    log_dir_root = Path("logs/csv") # 將日誌根目錄定義在外面
    
    if args.skip_training:
        # 【僅繪圖】
        print("Running in PLOT-ONLY mode. Skipping training...")
        
        # 直接從現有的日誌檔案讀取數據
        for data_path_str in args.datasets:
            data_path = Path(data_path_str)
            version_name = data_path.name.split('_')[-1]
            run_name = f"tactile_{version_name}"
            
            csv_path = log_dir_root / run_name / "version_0" / "metrics.csv"

            if not csv_path.exists():
                print(f"Warning: Log file not found for {version_name} at {csv_path}. Skipping.")
                continue
            
            print(f"Found log file for {version_name}: {csv_path}")
            df = pd.read_csv(csv_path)
            final_train_loss = df['train_loss_epoch'].dropna().iloc[-1]
            final_test_loss = df['test_loss'].dropna().iloc[-1] if 'test_loss' in df.columns and not df['test_loss'].dropna().empty else np.nan
            test_r2 = df['test_r2'].dropna().iloc[-1] if 'test_r2' in df.columns and not df['test_r2'].dropna().empty else np.nan
            gap = ((final_test_loss - final_train_loss) / final_train_loss * 100) if not np.isnan(final_test_loss) and final_train_loss != 0 else np.nan
            all_metrics[version_name] = {'df': df, 'final_train_loss': final_train_loss, 'final_test_loss': final_test_loss, 'test_r2': test_r2, 'gap': gap}
    
    # 迴圈執行所有指定的資料集
    else:
        for data_path_str in args.datasets:
            data_path = Path(data_path_str)
            # 從路徑中自動提取版本名稱 (例如 '1x', '2x')
            version_name = data_path.name.split('_')[-1]
            
            # `train_single_model` 的後綴現在也使用版本名稱
            csv_path = train_single_model(config, data_path_str, 'tactile', version_name)
            
            # 讀取結果並儲存
            df = pd.read_csv(csv_path)
            final_train_loss = df['train_loss_epoch'].dropna().iloc[-1]
            final_test_loss = df['test_loss'].dropna().iloc[-1] if 'test_loss' in df.columns and not df['test_loss'].dropna().empty else np.nan
            test_r2 = df['test_r2'].dropna().iloc[-1] if 'test_r2' in df.columns and not df['test_r2'].dropna().empty else np.nan
            gap = ((final_test_loss - final_train_loss) / final_train_loss * 100) if not np.isnan(final_test_loss) else np.nan
            
            all_metrics[version_name] = {
                'df': df,
                'final_train_loss': final_train_loss,
                'final_test_loss': final_test_loss,
                'test_r2': test_r2,
                'gap': gap
            }

    # 所有訓練都完成後，產生總報告和總圖表
    if not all_metrics:
        print("No metrics available to generate report and plots.")
    else:
        generate_report_and_plots(all_metrics, './plots')
        
if __name__ == '__main__':
    main()