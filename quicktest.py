#!/usr/bin/env python3
"""
快速測試腳本：用新轉換的資料測試已訓練的模型
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from residual_controller.bridge_model import ResidualController
from tqdm import tqdm

def load_test_data(test_data_dir, num_episodes=5):
    """載入少量測試資料"""
    test_data_dir = Path(test_data_dir)
    h5_files = sorted(list(test_data_dir.glob('*.h5')))[:num_episodes]
    
    print(f"📂 載入測試資料...")
    print(f"   路徑：{test_data_dir}")
    print(f"   使用 {len(h5_files)} 個 episodes（共 {len(list(test_data_dir.glob('*.h5')))} 個可用）")
    
    all_samples = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            # 讀取觸覺影像
            tactile = f['gelsight/gelsight'][:]
            tactile = torch.tensor(tactile, dtype=torch.float32)
            
            # 建立 (input, target) 配對
            for t in range(len(tactile) - 1):
                all_samples.append((tactile[t], tactile[t + 1]))
        
        print(f"   ✅ {h5_file.name}: {len(tactile)} 幀 → {len(tactile)-1} 筆樣本")
    
    print(f"\n📊 總測試樣本數：{len(all_samples)}\n")
    return all_samples

def test_model(checkpoint_path, test_samples, modality='tactile'):
    """測試模型"""
    print(f"🔧 載入模型檢查點...")
    print(f"   路徑：{checkpoint_path}")
    
    # 載入模型
    model = ResidualController.load_from_checkpoint(
        checkpoint_path,
        modality=modality
    )
    model.eval()
    
    # 移到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   裝置：{device}\n")
    
    # 測試
    print(f"🧪 開始測試...")
    losses = []
    
    with torch.no_grad():
        for input_frame, target_frame in tqdm(test_samples, desc="Testing"):
            # 準備資料
            input_batch = input_frame.unsqueeze(0).to(device)  # [1, H, W, C]
            target_batch = target_frame.unsqueeze(0).to(device)
            
            # 提取特徵
            input_features = model.extract_features(input_batch)  # [1, obs_dim]
            target_features = model.extract_features(target_batch).unsqueeze(1)  # [1, 1, obs_dim]
            
            # 預測
            pred = model.model(input_features)
            pred_actions = pred.view(-1, model.horizon, model.action_dim)[:, 0:1, :]
            
            # 調整維度
            if pred_actions.shape[-1] != target_features.shape[-1]:
                target_features = target_features[..., :pred_actions.shape[-1]]
            
            # 計算 loss
            loss = torch.nn.functional.mse_loss(pred_actions, target_features)
            losses.append(loss.item())
    
    return losses

def print_results(losses, checkpoint_name, num_episodes):
    """印出結果"""
    print(f"\n{'='*70}")
    print(f"📊 測試結果報告")
    print(f"{'='*70}")
    print(f"\n【測試設定】")
    print(f"  模型檢查點：{checkpoint_name}")
    print(f"  測試 episodes：{num_episodes} 個")
    print(f"  測試樣本數：{len(losses)} 筆")
    
    print(f"\n【性能指標】")
    print(f"  平均測試 Loss：{np.mean(losses):.6f}")
    print(f"  Loss 標準差：{np.std(losses):.6f}")
    print(f"  最小 Loss：{np.min(losses):.6f}")
    print(f"  最大 Loss：{np.max(losses):.6f}")
    print(f"  中位數 Loss：{np.median(losses):.6f}")
    
    # 與訓練時的比較（假設訓練 loss 約 0.001）
    train_loss = 0.000974  # 🔧 改成您實際訓練的最終 loss
    gap = (np.mean(losses) - train_loss) / train_loss * 100
    
    print(f"\n【泛化能力分析】")
    print(f"  訓練集 Loss（參考）：{train_loss:.6f}")
    print(f"  測試集 Loss：{np.mean(losses):.6f}")
    print(f"  泛化差距：{gap:+.1f}%")
    
    if gap < 20:
        print(f"  ✅ 評價：泛化能力優秀")
    elif gap < 50:
        print(f"  ✅ 評價：泛化能力良好")
    elif gap < 100:
        print(f"  ⚠️  評價：有輕微過擬合")
    else:
        print(f"  ❌ 評價：可能有過擬合問題")
    
    print(f"{'='*70}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='快速測試新資料')
    parser.add_argument('--checkpoint', type=str, 
                        default='./outputs/tactile/last.ckpt',
                        help='模型檢查點路徑')
    parser.add_argument('--test_data', type=str,
                        default='./data/datasets/mango_hdf5_gelsight',
                        help='測試資料路徑')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='使用幾個 episodes 來測試（預設 5）')
    parser.add_argument('--modality', type=str, default='tactile',
                        choices=['tactile', 'tactile_reduced'],
                        help='模型類型')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🚀 快速測試：用新資料評估訓練好的模型")
    print(f"{'='*70}\n")
    
    # 檢查檔案是否存在
    if not Path(args.checkpoint).exists():
        print(f"❌ 錯誤：找不到模型檢查點 {args.checkpoint}")
        print(f"\n💡 可用的檢查點：")
        for ckpt in Path('./outputs/tactile').glob('*.ckpt'):
            print(f"   - {ckpt}")
        return
    
    if not Path(args.test_data).exists():
        print(f"❌ 錯誤：找不到測試資料 {args.test_data}")
        return
    
    # 載入測試資料
    test_samples = load_test_data(args.test_data, args.num_episodes)
    
    # 測試模型
    losses = test_model(args.checkpoint, test_samples, args.modality)
    
    # 印出結果
    checkpoint_name = Path(args.checkpoint).name
    print_results(losses, checkpoint_name, args.num_episodes)
    
    # 儲存結果
    results_file = './test_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"測試結果\n")
        f.write(f"={'='*60}\n")
        f.write(f"模型：{checkpoint_name}\n")
        f.write(f"測試 episodes：{args.num_episodes}\n")
        f.write(f"平均 Loss：{np.mean(losses):.6f}\n")
        f.write(f"標準差：{np.std(losses):.6f}\n")
    
    print(f"💾 結果已儲存到：{results_file}")

if __name__ == "__main__":
    main()